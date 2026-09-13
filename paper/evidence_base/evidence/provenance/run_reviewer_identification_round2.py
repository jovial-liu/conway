#!/usr/bin/env python3
"""Reviewer identification controls from frozen intervention tables.

This pass is deliberately offline.  It reads the already validated OpenAI
COCO/VOC drop tables, category folds, cluster masks, geometry metadata, and
normalized text caches.  It does not import open_clip, load a model, run a
forward pass, regenerate regions, or search a selector parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
REPO = CODE.parent
sys.path.insert(0, str(CODE))
from run_cci_preserving_multisetting import geometry, load_setting, raw_fingerprint, sha256


RAW_ROOT = Path("/home/fourier/codex/waterbirds_clip_mediation/outputs_disjoint_foil")
TEXT_ROOT = REPO / "results/stage_c_c2_forward_final"
DERIVED = REPO / "derived/reviewer_identification_v1"
HANDOFF = REPO / "paper_handoff/reviewer_round2"
EPSILON = 0.02
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 1701
BOOTSTRAP_10K = 10000
N_PROMPTS = 3
N_REGIONS = 8
SEEDS = (1701, 2701, 3701, 4701, 5701, 6701, 7701, 8701, 9701, 10701)
PROMPTS = (
    "a picture of the {category}",
    "an image containing a {category}",
    "the {category}",
)

SETUPS = (
    {"dataset": "coco", "model": "openai_b16", "raw": "coco_openai_b16_n27708", "text": "coco_openai_b16"},
    {"dataset": "coco", "model": "openai_b32", "raw": "coco_openai_b32_n27708", "text": "coco_openai_b32"},
    {"dataset": "voc2007", "model": "openai_b16", "raw": "voc2007_openai_b16_n1943", "text": "voc2007_openai_b16"},
    {"dataset": "voc2007", "model": "openai_b32", "raw": "voc2007_openai_b32_n1943", "text": "voc2007_openai_b32"},
)


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n")


def _json_default(value: object):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def full_mask(target_index: np.ndarray, categories: int) -> np.ndarray:
    mask = np.ones((len(target_index), categories), dtype=bool)
    mask[np.arange(len(target_index)), target_index] = False
    return mask


def category_masks(metadata: dict, fold: np.ndarray, fold_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selection, annotated-present, and annotation-absent held-out masks."""
    ids = metadata["category_ids"]
    index = {str(value): i for i, value in enumerate(ids)}
    targets = np.asarray([index[str(row["target_id"])] for row in metadata["samples_metadata"]], dtype=np.int64)
    selection = np.broadcast_to(fold[None, :] == fold_id, (len(targets), len(ids))).copy()
    heldout = np.broadcast_to(fold[None, :] != fold_id, selection.shape).copy()
    selection[np.arange(len(targets)), targets] = False
    heldout[np.arange(len(targets)), targets] = False
    present = np.zeros_like(selection)
    for i, row in enumerate(metadata["samples_metadata"]):
        for category_id in row.get("distractor_ids", []):
            if str(category_id) in index:
                present[i, index[str(category_id)]] = True
    present &= heldout
    return selection, present, heldout & ~present


def make_identity_split(category_ids: list[object], seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(len(category_ids))
    fold = np.empty(len(category_ids), dtype=np.int64)
    fold[order] = np.arange(len(category_ids)) % 2
    return fold


def load_text(spec: dict) -> tuple[np.ndarray, Path]:
    path = TEXT_ROOT / spec["text"] / "text_embeddings.npy"
    if not path.is_file():
        raise RuntimeError(f"missing normalized text cache: {path}")
    text = np.load(path)
    if text.shape[0] != N_PROMPTS or not np.isfinite(text).all():
        raise RuntimeError(f"invalid text cache {path}: {text.shape}")
    return text, path


def select_regions(st: np.ndarray, sc: np.ndarray, mask: np.ndarray, epsilon: float = EPSILON) -> dict[str, np.ndarray]:
    """Reproduce CCI and max-foil RTP in the source dtype and first-tie order."""
    n = len(st)
    rows = np.arange(n)
    if mask.shape != (n, sc.shape[2]) or not mask.any(axis=1).all():
        raise RuntimeError("selection foil family contains an empty row")
    masked = np.where(mask[:, None, :], sc, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        max_foil = np.nanmax(masked, axis=2)
        mean_foil = np.nanmean(masked, axis=2)
    cci = np.argmax(st, axis=1)
    cci_target = st[rows, cci]
    feasible = st >= (cci_target - np.asarray(epsilon, dtype=st.dtype))[:, None]
    margin = st - max_foil
    score = np.where(feasible, margin, -np.inf)
    rtp = np.argmax(score, axis=1)
    return {
        "cci": cci,
        "rtp": rtp,
        "feasible": feasible,
        "feasible_count": feasible.sum(axis=1),
        "mean_score": st - mean_foil,
        "weak_max_score": st - np.asarray(0.1, dtype=st.dtype) * max_foil,
        "rtp_score": margin,
        "max_foil": max_foil,
        "mean_foil": mean_foil,
        "cci_target": cci_target,
        "selection_target": st,
    }


def aggregate_text_norm(text: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(text.mean(axis=0), axis=1)
    if np.any(norm <= 0) or not np.isfinite(norm).all():
        raise RuntimeError("invalid class normalization")
    return norm


def evaluate_regions(table: dict, text: np.ndarray, norm: np.ndarray, regions: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Compute raw prompt/aggregate and class-normalized aggregate metrics."""
    n, categories = len(regions), table["evaluation_category"].shape[-1]
    rows = np.arange(n)
    et = table["evaluation_target"]
    ec = table["evaluation_category"]
    prompt_target = np.empty((N_PROMPTS, n), dtype=np.float64)
    prompt_pmax = np.full((N_PROMPTS, n), np.nan, dtype=np.float64)
    prompt_pmean = np.full((N_PROMPTS, n), np.nan, dtype=np.float64)
    prompt_v_rate = np.full((N_PROMPTS, n), np.nan, dtype=np.float64)
    prompt_v_mass = np.full((N_PROMPTS, n), np.nan, dtype=np.float64)
    for p in range(N_PROMPTS):
        target = et[p, rows, regions].astype(np.float64)
        category = ec[p, rows, regions, :].astype(np.float64)
        finite = mask
        count = finite.sum(axis=1)
        foils = np.where(finite, category, np.nan)
        prompt_target[p] = target
        valid = count > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            prompt_pmax[p, valid] = target[valid] - np.nanmax(foils[valid], axis=1)
            prompt_pmean[p, valid] = target[valid] - np.nanmean(foils[valid], axis=1)
            prompt_v_mass[p, valid] = np.nanmean(np.maximum(foils[valid] - target[valid, None], 0.0), axis=1)
        violation = finite & (category > target[:, None])
        prompt_v_rate[p, valid] = violation[valid].sum(axis=1) / count[valid]

    et_mean = et.mean(axis=0).astype(np.float64)[rows, regions]
    ec_mean = ec.mean(axis=0).astype(np.float64)[rows, regions, :]
    norm_target = et_mean / norm[table["target_index"]]
    norm_category = ec_mean / norm[None, :]
    valid = mask.any(axis=1)
    raw_pmax_agg = np.full(n, np.nan)
    raw_pmean_agg = np.full(n, np.nan)
    raw_v_rate_agg = np.full(n, np.nan)
    raw_v_mass_agg = np.full(n, np.nan)
    norm_pmax_agg = np.full(n, np.nan)
    norm_pmean_agg = np.full(n, np.nan)
    norm_v_rate_agg = np.full(n, np.nan)
    norm_v_mass_agg = np.full(n, np.nan)
    raw_foils = np.where(mask, ec_mean, np.nan)
    norm_foils = np.where(mask, norm_category, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_pmax_agg[valid] = et_mean[valid] - np.nanmax(raw_foils[valid], axis=1)
        raw_pmean_agg[valid] = et_mean[valid] - np.nanmean(raw_foils[valid], axis=1)
        raw_v_rate_agg[valid] = ((raw_foils[valid] > et_mean[valid, None]).sum(axis=1) / mask[valid].sum(axis=1))
        raw_v_mass_agg[valid] = np.nanmean(np.maximum(raw_foils[valid] - et_mean[valid, None], 0.0), axis=1)
        norm_pmax_agg[valid] = norm_target[valid] - np.nanmax(norm_foils[valid], axis=1)
        norm_pmean_agg[valid] = norm_target[valid] - np.nanmean(norm_foils[valid], axis=1)
        norm_v_rate_agg[valid] = ((norm_foils[valid] > norm_target[valid, None]).sum(axis=1) / mask[valid].sum(axis=1))
        norm_v_mass_agg[valid] = np.nanmean(np.maximum(norm_foils[valid] - norm_target[valid, None], 0.0), axis=1)
    prompt_pmax_mean = np.nanmean(prompt_pmax, axis=0)
    prompt_pmean_mean = np.nanmean(prompt_pmean, axis=0)
    prompt_target_mean = np.nanmean(prompt_target, axis=0)
    prompt_v_rate_mean = np.nanmean(prompt_v_rate, axis=0)
    prompt_v_mass_mean = np.nanmean(prompt_v_mass, axis=0)
    return {
        "pmax_pm_raw": prompt_pmax_mean,
        "pmean_pm_raw": prompt_pmean_mean,
        "target_response_raw": prompt_target_mean,
        "v_rate_pm_raw": prompt_v_rate_mean,
        "v_mass_pm_raw": prompt_v_mass_mean,
        "pmax_agg_raw": raw_pmax_agg,
        "pmean_agg_raw": raw_pmean_agg,
        "target_response_agg_raw": et_mean,
        "v_rate_agg_raw": raw_v_rate_agg,
        "v_mass_agg_raw": raw_v_mass_agg,
        "pmax_agg_norm": norm_pmax_agg,
        "pmean_agg_norm": norm_pmean_agg,
        "target_response_agg_norm": norm_target,
        "v_rate_agg_norm": norm_v_rate_agg,
        "v_mass_agg_norm": norm_v_mass_agg,
        "prompt_pmax": prompt_pmax,
        "prompt_target": prompt_target,
        "valid": valid,
        "mask_count": mask.sum(axis=1),
    }


def gather(values: dict[str, np.ndarray], regions: np.ndarray) -> dict[str, np.ndarray]:
    rows = np.arange(len(regions))
    output = {}
    for key, value in values.items():
        if value.ndim == 1:
            output[key] = value
        elif value.ndim == 2 and value.shape[0] == len(regions):
            output[key] = value[rows, regions]
        else:
            output[key] = value
    return output


def bootstrap_vector(values: np.ndarray, seed: int, count: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(count, dtype=np.float64)
    for start in range(0, count, 32):
        size = min(32, count - start)
        idx = rng.integers(0, len(values), size=(size, len(values)))
        draws[start:start + size] = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def bootstrap_pair(original: np.ndarray, candidate: np.ndarray, seed: int, count: int) -> tuple[float, float, float]:
    original = np.asarray(original, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    valid = np.isfinite(original) & np.isfinite(candidate)
    original, candidate = original[valid], candidate[valid]
    if len(original) == 0:
        return math.nan, math.nan, math.nan
    delta = candidate - original
    rng = np.random.default_rng(seed)
    draws = np.empty(count, dtype=np.float64)
    for start in range(0, count, 32):
        size = min(32, count - start)
        idx = rng.integers(0, len(original), size=(size, len(original)))
        draws[start:start + size] = delta[idx].mean(axis=1)
    return float(delta.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, seed: int, count: int) -> tuple[float, float, float]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator)
    numerator, denominator = numerator[valid], denominator[valid]
    if len(numerator) == 0 or denominator.sum() <= 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(count, dtype=np.float64)
    for start in range(0, count, 32):
        size = min(32, count - start)
        idx = rng.integers(0, len(numerator), size=(size, len(numerator)))
        den = denominator[idx].sum(axis=1)
        draws[start:start + size] = np.divide(numerator[idx].sum(axis=1), den, out=np.full(size, np.nan), where=den > 0)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return math.nan, math.nan, math.nan
    return float(numerator.sum() / denominator.sum()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def metric_seed(setting_index: int, label: str, count: int) -> int:
    return BOOTSTRAP_SEED + setting_index * 100000 + count * 13 + sum(ord(c) for c in label)


def method_summary_rows(dataset: str, model: str, methods: dict[str, np.ndarray], metrics: dict[str, dict[str, np.ndarray]], cci: np.ndarray, out_rows: list[dict], setting_index: int, protocol: str) -> None:
    metric_names = ("pmax_pm_raw", "pmax_agg_raw", "pmax_agg_norm", "target_response_raw", "target_response_agg_norm", "bbox_precision")
    for method, region in methods.items():
        if method == "cci":
            continue
        values = metrics[method]
        base = metrics["cci"]
        switched = region != cci
        row = {"dataset": dataset, "model": model, "protocol": protocol, "method": method, "epsilon": EPSILON, "n_images": len(region), "switch_count": int(switched.sum()), "switch_rate": float(switched.mean())}
        for name in metric_names:
            delta, lo, hi = bootstrap_pair(base[name], values[name], metric_seed(setting_index, protocol + method + name, BOOTSTRAP), BOOTSTRAP)
            row[f"delta_{name}"] = delta
            row[f"delta_{name}_ci_low"] = lo
            row[f"delta_{name}_ci_high"] = hi
        sw, sw_lo, sw_hi = bootstrap_vector(switched.astype(float), metric_seed(setting_index, protocol + method + "switch", BOOTSTRAP), BOOTSTRAP)
        row.update({"switch_rate_boot": sw, "switch_rate_ci_low": sw_lo, "switch_rate_ci_high": sw_hi})
        positive, positive_lo, positive_hi = bootstrap_ratio((switched & (values["pmax_agg_norm"] > base["pmax_agg_norm"])).astype(float), switched.astype(float), metric_seed(setting_index, protocol + method + "positive", BOOTSTRAP), BOOTSTRAP)
        row.update({"p_delta_pmax_agg_norm_positive_switched": positive, "p_delta_pmax_agg_norm_positive_switched_ci_low": positive_lo, "p_delta_pmax_agg_norm_positive_switched_ci_high": positive_hi})
        out_rows.append(row)


def extract_selected_metrics(table: dict, text: np.ndarray, norm: np.ndarray, geom: dict, region: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    values = evaluate_regions(table, text, norm, region, mask)
    rows = np.arange(len(region))
    add_prompt_normalized_metrics(values, table, norm, text, region, mask)
    for name, key in (("bbox_precision", "bbox_precision"), ("bbox_recall", "bbox_recall"), ("bbox_iou", "bbox_iou"), ("target_minus_distractor_bbox", "bbox_precision")):
        values[name] = geom[key][rows, region] if name != "target_minus_distractor_bbox" else geom["bbox_precision"][rows, region] - geom["distractor_precision"][rows, region]
    return values


def fixed_subset_mask(target_index: np.ndarray, categories: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros((len(target_index), categories), dtype=bool)
    for i, target in enumerate(target_index):
        candidates = np.flatnonzero(np.arange(categories) != target)
        mask[i, rng.choice(candidates, size=k, replace=False)] = True
    return mask


def all_feasible_methods(selection: dict, seed: int) -> dict[str, np.ndarray]:
    st = selection["selection_target"]
    feasible = selection["feasible"]
    cci = selection["cci"]
    rows = np.arange(len(st))
    methods = {"cci": cci.copy(), "rtp": np.argmax(np.where(feasible, selection["rtp_score"], -np.inf), axis=1)}
    mean_score = np.where(feasible, selection["mean_score"], -np.inf)
    weak_score = np.where(feasible, selection["weak_max_score"], -np.inf)
    methods["mean_foil_feasible"] = np.argmax(mean_score, axis=1)
    methods["weak_max_scalarization"] = np.argmax(weak_score, axis=1)
    rng = np.random.default_rng(seed)
    uniform = np.empty(len(st), dtype=np.int64)
    non_cci = np.empty(len(st), dtype=np.int64)
    for i in range(len(st)):
        candidates = np.flatnonzero(feasible[i])
        uniform[i] = rng.choice(candidates)
        alternatives = candidates[candidates != cci[i]]
        non_cci[i] = rng.choice(alternatives) if len(alternatives) else cci[i]
    methods["uniform_feasible"] = uniform
    methods["conditional_uniform_non_cci"] = non_cci
    order = np.argsort(-st, axis=1, kind="stable")
    runner = cci.copy()
    for i in range(len(st)):
        candidates = [int(r) for r in order[i] if feasible[i, r]]
        if len(candidates) > 1:
            runner[i] = candidates[1]
    methods["target_runner_up"] = runner
    return methods


def make_alignment_rows(setting_index: int, spec: dict, table: dict, text: np.ndarray, norm: np.ndarray, geom: dict, selection: dict, rows: list[dict]) -> dict:
    mask = full_mask(table["target_index"], len(table["category_ids"]))
    cci = selection["cci"]
    rtp = selection["rtp"]
    cci_m = extract_selected_metrics(table, text, norm, geom, cci, mask)
    rtp_m = extract_selected_metrics(table, text, norm, geom, rtp, mask)
    names = ("pmax_pm_raw", "pmax_pm_norm", "pmax_agg_raw", "pmax_agg_norm", "target_response_raw", "target_response_agg_norm", "bbox_precision")
    for comparison, left, right in (("cci", None, cci_m), ("rtp_epsilon_0.02", None, rtp_m), ("rtp_minus_cci", cci_m, rtp_m)):
        row = {"dataset": spec["dataset"], "model": spec["model"], "comparison": comparison, "epsilon": EPSILON, "n_images": len(cci), "switch_count": int((rtp != cci).sum()) if comparison != "cci" else 0, "switch_rate": float((rtp != cci).mean()) if comparison != "cci" else 0.0}
        for name in names:
            if comparison == "rtp_minus_cci":
                est, lo, hi = bootstrap_pair(left[name], right[name], metric_seed(setting_index, "alignment" + name, BOOTSTRAP), BOOTSTRAP)
            else:
                est, lo, hi = bootstrap_vector(right[name], metric_seed(setting_index, comparison + name, BOOTSTRAP), BOOTSTRAP)
            row[name] = est
            row[f"{name}_ci_low"] = lo
            row[f"{name}_ci_high"] = hi
        for space in ("raw", "norm"):
            pm = right["pmax_pm_raw"] if space == "raw" else right["pmax_pm_raw"] / 1.0
            # pmax_pm_norm is derived below from prompt-wise normalized drops;
            # aggregate-minus-prompt remains an explicitly recorded estimand.
            agg = right["pmax_agg_raw"] if space == "raw" else right["pmax_agg_norm"]
            if space == "raw":
                pm = right["pmax_pm_raw"]
            else:
                # Prompt-wise normalized values are stored by the caller for
                # the exact P_pm_norm definition.
                pm = right["pmax_pm_norm"]
            row[f"agg_minus_pm_{space}"] = float(np.nanmean(agg - pm))
            if comparison == "rtp_minus_cci":
                d = (right["pmax_agg_raw"] if space == "raw" else right["pmax_agg_norm"]) - (left["pmax_agg_raw"] if space == "raw" else left["pmax_agg_norm"]) - ((right["pmax_pm_raw"] if space == "raw" else right["pmax_pm_norm"]) - (left["pmax_pm_raw"] if space == "raw" else left["pmax_pm_norm"]))
                row[f"delta_agg_minus_pm_{space}"] = float(np.nanmean(d))
                if space == "norm":
                    row["delta_pmax_pm_raw_norm_direction_consistent"] = bool(np.sign(right["pmax_pm_raw"].mean() - left["pmax_pm_raw"].mean()) == np.sign(right["pmax_pm_norm"].mean() - left["pmax_pm_norm"].mean()))
                    row["delta_pmax_agg_raw_norm_direction_consistent"] = bool(np.sign(right["pmax_agg_raw"].mean() - left["pmax_agg_raw"].mean()) == np.sign(right["pmax_agg_norm"].mean() - left["pmax_agg_norm"].mean()))
        rows.append(row)
    return {"cci": cci_m, "rtp": rtp_m, "selection": selection, "mask": mask}


def add_prompt_normalized_metrics(metrics: dict, table: dict, norm: np.ndarray, text: np.ndarray, region: np.ndarray, mask: np.ndarray) -> None:
    rows = np.arange(len(region))
    et = table["evaluation_target"][:, rows, region].astype(np.float64)
    ec = table["evaluation_category"][:, rows, region, :].astype(np.float64)
    target_idx = table["target_index"]
    target = et / norm[target_idx][None, :]
    category = ec / norm[None, None, :]
    foils = np.where(mask[None, :, :], category, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        pmax = target - np.nanmax(foils, axis=2)
    metrics["pmax_pm_norm"] = np.nanmean(pmax, axis=0)


def summarize_cross_direction(dataset: str, model: str, direction: str, split_results: list[dict], summary_rows: list[dict], bootstrap_rows: list[dict], setting_index: int, absent: bool = False) -> None:
    prefix = "absent_only" if absent else "crossfoil"
    if not split_results:
        return
    metric_names = ("pmax_pm_raw", "pmax_pm_norm", "pmax_agg_raw", "pmax_agg_norm", "target_response_raw", "target_response_agg_norm", "bbox_precision")
    delta_by_metric = {name: np.nanmean(np.stack([r["rtp"][name] - r["cci"][name] for r in split_results]), axis=0) for name in metric_names}
    cci_by_metric = {name: np.nanmean(np.stack([r["cci"][name] for r in split_results]), axis=0) for name in metric_names}
    rtp_by_metric = {name: np.nanmean(np.stack([r["rtp"][name] for r in split_results]), axis=0) for name in metric_names}
    switches = np.nanmean(np.stack([r["switch"].astype(float) for r in split_results]), axis=0)
    positive = np.nanmean(np.stack([(r["switch"] & (r["rtp"]["pmax_agg_norm"] > r["cci"]["pmax_agg_norm"])).astype(float) for r in split_results]), axis=0)
    denomin = switches
    row = {"dataset": dataset, "model": model, "direction": direction, "evaluation": prefix, "partition_repeats": len(split_results), "n_images": len(switches), "eligible_image_count": int(np.isfinite(delta_by_metric["pmax_agg_norm"]).sum()), "switch_rate": float(np.nanmean(switches)), "p_delta_pmax_agg_norm_positive_switched": float(np.nansum(positive) / np.nansum(denomin)) if np.nansum(denomin) else math.nan}
    for name in metric_names:
        for label, values in (("cci", cci_by_metric[name]), ("rtp", rtp_by_metric[name]), ("delta", delta_by_metric[name])):
            est, lo, hi = bootstrap_vector(values, metric_seed(setting_index, prefix + direction + label + name, BOOTSTRAP), BOOTSTRAP) if label != "delta" else bootstrap_vector(values, metric_seed(setting_index, prefix + direction + label + name, BOOTSTRAP), BOOTSTRAP)
            row[f"{label}_{name}"] = est
            row[f"{label}_{name}_ci_low"] = lo
            row[f"{label}_{name}_ci_high"] = hi
            bootstrap_rows.append({"dataset": dataset, "model": model, "evaluation": prefix, "direction": direction, "metric": name, "method": label, "estimate": est, "ci_low": lo, "ci_high": hi, "bootstrap_count": BOOTSTRAP, "bootstrap_seed": metric_seed(setting_index, prefix + direction + label + name, BOOTSTRAP), "bootstrap_unit": "unique_image"})
    sw, sw_lo, sw_hi = bootstrap_vector(switches, metric_seed(setting_index, prefix + direction + "switch", BOOTSTRAP), BOOTSTRAP)
    row.update({"switch_rate_boot": sw, "switch_rate_ci_low": sw_lo, "switch_rate_ci_high": sw_hi})
    summary_rows.append(row)


def build_protocol_export(provenance: list[dict], out: Path) -> None:
    oracle_gap = math.nan
    oracle_path = REPO / "derived/reviewer_cci_preserving_v1/multisetting/04_multisetting_oracle.csv"
    if oracle_path.exists():
        frame = pd.read_csv(oracle_path)
        hit = frame[(frame["dataset"] == "voc2007") & (frame["model"] == "openai_b32") & (np.isclose(frame["epsilon"], EPSILON))]
        if len(hit):
            oracle_gap = float(hit.iloc[0]["oracle_minus_legal_gap"])
    lines = [
        "# Protocol definitions recovered from authoritative offline sources",
        "",
        "This export records existing frozen definitions; it introduces no new estimator or model computation.",
        "",
        "1. Selection prompt: `a photo of a {category}`.",
        "2. Evaluation prompts, in order: `a picture of the {category}`; `an image containing a {category}`; `the {category}`.",
        "3. Category folds: for seed `s` in `1701,2701,3701,4701,5701,6701,7701,8701,9701,10701`, permute category IDs with `default_rng(s)`; assign alternating positions to fold 0/1. Cross-fit direction selects one fold and evaluates the complementary fold.",
        "4. Class normalization: `n_c = ||mean_h e_h(c)||_2`, where the three stored evaluation text embeddings are already L2-normalized; normalized drops divide each class drop by its corresponding `n_c`.",
        "5. Semantic-family mapping: the `category_families` field in each bound `metadata.json` source table; this is the stored COCO/VOC metadata mapping and is not inferred from results.",
        "6. Bbox precision: transform each annotation through resize-short-side-224 and center-crop-224; count a patch as a hit when its 14x14/7x7 patch center lies in any transformed target box; divide hits by `max(selected_patch_count, 1)`. Recall/IoU use the stored geometry implementation.",
        "7. Internal intervention: for each selected cluster, in every visual transformer block and every attention head, set attention logits to the selected patch key/value columns to `-inf` for all query positions; retain the CLS/prefix token. The OpenAI implementation uses the frozen all-layer attention-block path in `run_coco_cci_specificity_audit.py`.",
        "8. Regions: the frozen candidate masks contain K=8 clusters; patch grids are recorded per setting in the provenance table (ViT-B/16: 14x14; ViT-B/32: 7x7).",
        "9. Ties: first exact maximum via NumPy `argmax`/`nanargmax` in the frozen source dtype.",
        "10. Bootstrap unit: unique image; paired candidate/original vectors are resampled at image level.",
        "11. Current oracle definition: among epsilon-feasible regions, select the region maximizing held-out aggregate Pmax. It uses held-out evaluation information and is diagnostic/non-deployable only.",
        f"12. Existing VOC B32 oracle gap at epsilon=.02 (`oracle_minus_legal_gap`): `{oracle_gap:.17g}`.",
        "",
        "## Setting provenance",
        "",
        pd.DataFrame(provenance).to_markdown(index=False),
    ]
    (out / "16_PROTOCOL_FOR_MANUSCRIPT.md").write_text("\n".join(lines) + "\n")


def copy_qualitative_image(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source": str(src), "output": str(dst), "sha256": sha256(dst), "size": dst.stat().st_size}


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Use empty CSV cells for unavailable rows; never serialize NaN/Inf."""
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)
    return result.where(pd.notna(result), "")


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    clean_frame(frame).to_csv(path, index=False)


def provenance_row(spec: dict, table: dict, text_path: Path, text: np.ndarray) -> dict:
    return {
        "dataset": spec["dataset"],
        "model": spec["model"],
        "raw_table": str(table["path"]),
        "raw_fingerprint": json.dumps(table["fingerprint"], sort_keys=True),
        "raw_table_sha256": table["fingerprint"]["chunk_set_sha256"],
        "metadata_sha256": table["fingerprint"]["metadata_sha256"],
        "cluster_masks_sha256": table["fingerprint"]["cluster_masks_sha256"],
        "foil_folds_sha256": table["fingerprint"]["foil_folds_sha256"],
        "text_cache": str(text_path),
        "text_sha256": sha256(text_path),
        "text_shape": str(list(text.shape)),
        "text_dtype": str(text.dtype),
        "class_norm_formula": "n_c=||mean_h normalized_e_h(c)||_2",
        "selection_prompt": table["metadata"].get("selection_prompt"),
        "evaluation_prompts": json.dumps(table["metadata"].get("evaluation_prompts", [])),
        "image_count": len(table["image_ids"]),
        "category_count": len(table["category_ids"]),
        "candidate_regions": int(table["masks"].shape[1]),
        "patch_grid": int(round(math.sqrt(table["masks"].shape[2])),),
        "model_forward": False,
        "candidate_generation": False,
        "selector_search": False,
    }


def add_prompt_normalization(metrics: dict, table: dict, norm: np.ndarray, region: np.ndarray, mask: np.ndarray) -> None:
    """Add P_pm_norm without altering the frozen raw prompt tensors."""
    rows = np.arange(len(region))
    et = table["evaluation_target"][:, rows, region].astype(np.float64)
    ec = table["evaluation_category"][:, rows, region, :].astype(np.float64)
    target_norm = et / norm[table["target_index"]][None, :]
    cat_norm = ec / norm[None, None, :]
    foils = np.where(mask[None, :, :], cat_norm, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        pmax = target_norm - np.nanmax(foils, axis=2)
    metrics["pmax_pm_norm"] = np.nanmean(pmax, axis=0)


def selected_metrics(table: dict, text: np.ndarray, norm: np.ndarray, geom: dict, region: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    values = evaluate_regions(table, text, norm, region, mask)
    rows = np.arange(len(region))
    values["pmax_pm_norm"] = np.full(len(region), np.nan, dtype=np.float64)
    add_prompt_normalization(values, table, norm, region, mask)
    values["bbox_precision"] = geom["bbox_precision"][rows, region]
    values["bbox_recall"] = geom["bbox_recall"][rows, region]
    values["bbox_iou"] = geom["bbox_iou"][rows, region]
    values["target_minus_distractor_bbox"] = geom["bbox_precision"][rows, region] - geom["distractor_precision"][rows, region]
    return values


def paired_metric_rows(base: dict[str, np.ndarray], candidate: dict[str, np.ndarray], switch: np.ndarray, dataset: str, model: str, protocol: str, method: str, setting_index: int) -> dict:
    metrics = ("pmax_pm_raw", "pmax_pm_norm", "pmax_agg_raw", "pmax_agg_norm", "target_response_raw", "target_response_agg_norm", "bbox_precision", "bbox_recall", "bbox_iou", "target_minus_distractor_bbox")
    row = {"dataset": dataset, "model": model, "protocol": protocol, "method": method, "epsilon": EPSILON, "n_images": len(switch), "switch_count": int(switch.sum()), "switch_rate": float(switch.mean())}
    for name in metrics:
        delta, low, high = bootstrap_pair(base[name], candidate[name], metric_seed(setting_index, protocol + method + name, BOOTSTRAP), BOOTSTRAP)
        row[f"delta_{name}"] = delta
        row[f"delta_{name}_ci_low"] = low
        row[f"delta_{name}_ci_high"] = high
    sw, low, high = bootstrap_vector(switch.astype(float), metric_seed(setting_index, protocol + method + "switch", BOOTSTRAP), BOOTSTRAP)
    row.update({"switch_rate_boot": sw, "switch_rate_ci_low": low, "switch_rate_ci_high": high})
    positive, low, high = bootstrap_ratio((switch & (candidate["pmax_agg_norm"] > base["pmax_agg_norm"])).astype(float), switch.astype(float), metric_seed(setting_index, protocol + method + "positive", BOOTSTRAP), BOOTSTRAP)
    row.update({"p_delta_pmax_agg_norm_positive_switched": positive, "p_delta_pmax_agg_norm_positive_switched_ci_low": low, "p_delta_pmax_agg_norm_positive_switched_ci_high": high})
    return row


def crossfit_for_setting(spec: dict, table: dict, text: np.ndarray, norm: np.ndarray, geom: dict, setting_index: int, summary_rows: list[dict], bootstrap_rows: list[dict], absent_rows: list[dict], absent_bootstrap_rows: list[dict], baseline_rows: list[dict]) -> None:
    n_categories = len(table["category_ids"])
    full = full_mask(table["target_index"], n_categories)
    directions = {"A_to_B": [], "B_to_A": []}
    absent_directions = {"A_to_B": [], "B_to_A": []}
    baseline_by_method: dict[str, list[dict]] = {}
    for seed_index, seed in enumerate(SEEDS):
        fold = make_identity_split(table["category_ids"].tolist(), seed)
        for direction, select_fold, eval_fold in (("A_to_B", 0, 1), ("B_to_A", 1, 0)):
            selection_mask, present, absent_mask = category_masks(table["metadata"], fold, select_fold)
            evaluation_mask = np.broadcast_to(fold[None, :] != select_fold, full.shape).copy() & full
            if not np.array_equal(evaluation_mask, np.broadcast_to(fold[None, :] != select_fold, evaluation_mask.shape) & full):
                raise RuntimeError("cross-foil evaluation mask is not the complementary category fold")
            # The direction-specific selection mask is the selected fold; the
            # complementary held-out family is used only after selection.
            selected = select_regions(table["selection_target"], table["selection_category"], selection_mask)
            cci_region = selected["cci"]
            rtp_region = selected["rtp"]
            cci_values = selected_metrics(table, text, norm, geom, cci_region, evaluation_mask)
            rtp_values = selected_metrics(table, text, norm, geom, rtp_region, evaluation_mask)
            split = {"cci": cci_values, "rtp": rtp_values, "switch": rtp_region != cci_region, "seed": seed, "fold": select_fold}
            directions[direction].append(split)
            baseline_methods = all_feasible_methods(selected, BOOTSTRAP_SEED + seed_index * 100 + select_fold)
            for method, region in baseline_methods.items():
                method_values = selected_metrics(table, text, norm, geom, region, evaluation_mask)
                baseline_by_method.setdefault(method, []).append({"base": cci_values, "candidate": method_values, "switch": region != cci_region})
            absent_mask = absent_mask & evaluation_mask
            absent_cci = selected_metrics(table, text, norm, geom, cci_region, absent_mask)
            absent_rtp = selected_metrics(table, text, norm, geom, rtp_region, absent_mask)
            absent_directions[direction].append({"cci": absent_cci, "rtp": absent_rtp, "switch": rtp_region != cci_region, "seed": seed, "fold": select_fold})

    for direction, splits in directions.items():
        summarize_cross_direction(spec["dataset"], spec["model"], direction, splits, summary_rows, bootstrap_rows, setting_index, absent=False)
    pooled = directions["A_to_B"] + directions["B_to_A"]
    summarize_cross_direction(spec["dataset"], spec["model"], "pooled", pooled, summary_rows, bootstrap_rows, setting_index, absent=False)
    for direction, splits in absent_directions.items():
        summarize_cross_direction(spec["dataset"], spec["model"], direction, splits, absent_rows, absent_bootstrap_rows, setting_index, absent=True)
    summarize_cross_direction(spec["dataset"], spec["model"], "pooled", absent_directions["A_to_B"] + absent_directions["B_to_A"], absent_rows, absent_bootstrap_rows, setting_index, absent=True)

    # The feasible baselines are also evaluated on the legal cross-foil
    # protocol.  This keeps the same split directions and held-out family for
    # every selector and leaves the direct RTP comparison in Task C intact.
    for method, entries in baseline_by_method.items():
        if method == "cci":
            continue
        base = {name: np.nanmean(np.stack([entry["base"][name] for entry in entries]), axis=0) for name in entries[0]["base"]}
        candidate = {name: np.nanmean(np.stack([entry["candidate"][name] for entry in entries]), axis=0) for name in entries[0]["candidate"]}
        switches = np.nanmean(np.stack([entry["switch"].astype(float) for entry in entries]), axis=0) > 0
        baseline_rows.append(paired_metric_rows(base, candidate, switches, spec["dataset"], spec["model"], "cross_foil_pooled", method, setting_index))


def hardest_foil(table: dict, text: np.ndarray, norm: np.ndarray, region: int, image_index: int) -> tuple[int, float]:
    target_index = int(table["target_index"][image_index])
    target = float(table["evaluation_target"].mean(axis=0)[image_index, region]) / float(norm[target_index])
    categories = table["evaluation_category"].mean(axis=0)[image_index, region].astype(np.float64) / norm
    categories[target_index] = -np.inf
    foil = int(np.argmax(categories))
    return foil, float(target - categories[foil])


def qualitative_examples(table: dict, text: np.ndarray, norm: np.ndarray, geom: dict, selection: dict, out: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Choose six cases by fixed rules and copy only the original images."""
    full = full_mask(table["target_index"], len(table["category_ids"]))
    cci, rtp = selection["cci"], selection["rtp"]
    cci_values = selected_metrics(table, text, norm, geom, cci, full)
    rtp_values = selected_metrics(table, text, norm, geom, rtp, full)
    delta = rtp_values["pmax_agg_norm"] - cci_values["pmax_agg_norm"]
    switched = rtp != cci
    switched_idx = np.flatnonzero(switched)
    if not len(switched_idx):
        raise RuntimeError("no switched COCO/OpenAI B16 samples for qualitative rules")
    positive = switched_idx[delta[switched_idx] > 0]
    used: set[int] = set()

    def choose(candidates: np.ndarray, key: str) -> int:
        candidates = np.asarray([int(i) for i in candidates if int(i) not in used], dtype=np.int64)
        if not len(candidates):
            candidates = np.asarray([int(i) for i in switched_idx if int(i) not in used], dtype=np.int64)
        if key == "max":
            value = max(candidates, key=lambda i: (float(delta[i]), -int(i)))
        elif key == "min":
            value = min(candidates, key=lambda i: (float(delta[i]), int(i)))
        else:
            quantile = float(key)
            target = float(np.quantile(delta[candidates], quantile))
            value = min(candidates, key=lambda i: (abs(float(delta[i]) - target), int(i)))
        used.add(int(value))
        return int(value)

    present_candidates, absent_candidates = [], []
    for i in switched_idx:
        foil, _ = hardest_foil(table, text, norm, int(cci[i]), int(i))
        ids = {str(v) for v in table["samples"][int(i)].get("distractor_ids", [])}
        if str(table["category_ids"][foil]) in ids:
            present_candidates.append(int(i))
        else:
            absent_candidates.append(int(i))
    selections = [
        ("largest_positive_delta", choose(positive if len(positive) else switched_idx, "max")),
        ("near_75th_percentile_delta", choose(switched_idx, "0.75")),
        ("near_median_positive_delta", choose(positive if len(positive) else switched_idx, "0.50")),
        ("hardest_foil_annotated_present", choose(np.asarray(present_candidates), "max")),
        ("hardest_foil_annotation_absent", choose(np.asarray(absent_candidates), "max")),
        ("failure_or_smallest_positive", choose(switched_idx[delta[switched_idx] <= 0], "min") if np.any(delta[switched_idx] <= 0) else choose(positive if len(positive) else switched_idx, "min")),
    ]
    rows = []
    image_meta = []
    for number, (case, i) in enumerate(selections, start=1):
        i = int(i)
        original_region, repair_region = int(cci[i]), int(rtp[i])
        original_foil, _ = hardest_foil(table, text, norm, original_region, i)
        repair_foil, _ = hardest_foil(table, text, norm, repair_region, i)
        sample = table["samples"][i]
        src = Path(sample["path"])
        image_dst = out / "qualitative" / "images" / f"example_{number:02d}.jpg"
        image_info = copy_qualitative_image(src, image_dst)
        target_id = table["category_ids"][int(table["target_index"][i])]
        def annotated(category_index: int) -> bool:
            return str(table["category_ids"][category_index]) in {str(v) for v in sample.get("distractor_ids", [])}
        rows.append({
            "case": case, "sample_index": i, "image_id": str(table["image_ids"][i]), "target_id": str(target_id), "target": table["category_names"][int(table["target_index"][i])],
            "original_region": original_region, "repair_region": repair_region,
            "original_target_response_raw": float(cci_values["target_response_raw"][i]), "repair_target_response_raw": float(rtp_values["target_response_raw"][i]),
            "original_pmax_pm_norm": float(cci_values["pmax_pm_norm"][i]), "repair_pmax_pm_norm": float(rtp_values["pmax_pm_norm"][i]),
            "original_pmax_agg_norm": float(cci_values["pmax_agg_norm"][i]), "repair_pmax_agg_norm": float(rtp_values["pmax_agg_norm"][i]),
            "delta_pmax_agg_norm": float(delta[i]), "original_hardest_foil": table["category_names"][original_foil], "repair_hardest_foil": table["category_names"][repair_foil],
            "original_hardest_foil_annotated_present": annotated(original_foil), "repair_hardest_foil_annotated_present": annotated(repair_foil),
            "original_hardest_foil_family": str(table["metadata"].get("category_families", {}).get(str(table["category_ids"][original_foil]), "unknown")),
            "repair_hardest_foil_family": str(table["metadata"].get("category_families", {}).get(str(table["category_ids"][repair_foil]), "unknown")),
            "original_bbox_precision": float(cci_values["bbox_precision"][i]), "repair_bbox_precision": float(rtp_values["bbox_precision"][i]),
            "original_bbox_recall": float(cci_values["bbox_recall"][i]), "repair_bbox_recall": float(rtp_values["bbox_recall"][i]),
            "original_mask_patch_indices": json.dumps(np.flatnonzero(table["masks"][i, original_region]).astype(int).tolist()), "repair_mask_patch_indices": json.dumps(np.flatnonzero(table["masks"][i, repair_region]).astype(int).tolist()),
            "target_boxes": json.dumps(sample.get("target_boxes", [])), "foil_boxes": json.dumps(sample.get("distractor_boxes", [])), "image_file": str(image_dst.relative_to(out)), "image_sha256": image_info["sha256"],
        })
        evidence = {"case": case, "sample_index": i, "image_id": str(table["image_ids"][i]), "target_id": str(target_id), "original_region": original_region, "repair_region": repair_region, "original_hardest_foil_id": str(table["category_ids"][original_foil]), "repair_hardest_foil_id": str(table["category_ids"][repair_foil]), "source": image_info, "selection": "frozen CCI vs frozen-epsilon=.02 RTP; deterministic case rule; no manual selection"}
        evidence_path = out / "qualitative" / f"example_{number:02d}_render_metadata.json"
        dump_json(evidence_path, evidence)
        image_meta.append({"case": case, "task_evidence": str(evidence_path.relative_to(out)), "image_sha256": image_info["sha256"]})
    return pd.DataFrame(rows), image_meta


def bootstrap10k_rows(setting: dict, cache: dict, setting_index: int) -> list[dict]:
    metrics = ("pmax_pm_raw", "pmax_pm_norm", "pmax_agg_raw", "pmax_agg_norm", "target_response_raw", "target_response_agg_norm", "bbox_precision")
    rows = []
    schedules = [(1701, BOOTSTRAP), (1701, BOOTSTRAP_10K)]
    if setting["dataset"] == "coco" and setting["model"] == "openai_b32":
        schedules.append((2027, BOOTSTRAP_10K))
    for seed, count in schedules:
        for metric in metrics:
            original = cache["cci"][metric]
            candidate = cache["rtp"][metric]
            delta, low, high = bootstrap_pair(original, candidate, seed + sum(ord(c) for c in metric), count)
            rows.append({"dataset": setting["dataset"], "model": setting["model"], "metric": metric, "bootstrap_seed": seed, "bootstrap_count": count, "original_estimate": float(np.nanmean(original)), "candidate_estimate": float(np.nanmean(candidate)), "delta_estimate": delta, "delta_ci_low": low, "delta_ci_high": high, "bootstrap_unit": "unique_image", "paired": True})
    return rows


def sparse_row(spec: dict, cache: dict, setting_index: int) -> dict:
    cci, rtp = cache["cci"], cache["rtp"]
    original = cci["pmax_agg_norm"]
    repair = rtp["pmax_agg_norm"]
    switched = cache["rtp_region"] != cache["cci_region"]
    delta = repair - original
    valid = np.isfinite(delta)
    switch_delta = delta[switched & valid]
    return {
        "dataset": spec["dataset"], "model": spec["model"], "epsilon": EPSILON, "n_images": len(delta), "original_negative_margin_rate": float(np.mean(original < 0)), "repair_negative_margin_rate": float(np.mean(repair < 0)), "absolute_change_negative_margin_rate": float(np.mean(repair < 0) - np.mean(original < 0)), "switch_count": int(switched.sum()), "switch_rate": float(switched.mean()), "among_switched_mean_delta_pmax": float(np.mean(switch_delta)) if len(switch_delta) else math.nan, "among_switched_median_delta_pmax": float(np.median(switch_delta)) if len(switch_delta) else math.nan, "among_switched_p_delta_positive": float(np.mean(switch_delta > 0)) if len(switch_delta) else math.nan, "among_switched_fail_to_pass": float(np.mean((original[switched] < 0) & (repair[switched] >= 0))) if switched.any() else math.nan, "among_switched_pass_to_fail": float(np.mean((original[switched] >= 0) & (repair[switched] < 0))) if switched.any() else math.nan,
    }


def matched_foil_rows(spec: dict, table: dict, text: np.ndarray, norm: np.ndarray, geom: dict, selection: dict, setting_index: int) -> list[dict]:
    if spec["dataset"] != "coco":
        return []
    rows = []
    categories = len(table["category_ids"])
    full = full_mask(table["target_index"], categories)
    for k in (5, 19):
        mask = fixed_subset_mask(table["target_index"], categories, k, BOOTSTRAP_SEED)
        subset_selection = select_regions(table["selection_target"], table["selection_category"], mask)
        cci_region = selection["cci"]
        rtp_region = subset_selection["rtp"]
        cci_values = selected_metrics(table, text, norm, geom, cci_region, mask)
        rtp_values = selected_metrics(table, text, norm, geom, rtp_region, mask)
        switched = rtp_region != cci_region
        delta = rtp_values["pmax_agg_norm"] - cci_values["pmax_agg_norm"]
        row = {
            "dataset": spec["dataset"], "model": spec["model"], "foil_count": k, "subset_seed": BOOTSTRAP_SEED,
            "subset_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(), "n_images": len(mask), "foil_excludes_target": True,
            "cci_pmax_pm_negative_rate": float(np.mean(cci_values["pmax_pm_raw"] < 0)), "rtp_pmax_pm_negative_rate": float(np.mean(rtp_values["pmax_pm_raw"] < 0)),
            "cci_pmax_agg_norm": float(np.mean(cci_values["pmax_agg_norm"])), "rtp_pmax_agg_norm": float(np.mean(rtp_values["pmax_agg_norm"])),
            "cci_pmax_pm_norm": float(np.mean(cci_values["pmax_pm_norm"])), "rtp_pmax_pm_norm": float(np.mean(rtp_values["pmax_pm_norm"])),
            "delta_pmax_agg_norm": float(np.mean(delta)), "delta_pmax_pm_norm": float(np.mean(rtp_values["pmax_pm_norm"] - cci_values["pmax_pm_norm"])),
            "switch_count": int(switched.sum()), "switch_rate": float(switched.mean()),
            "p_delta_pmax_agg_norm_positive_switched": float(np.mean(delta[switched] > 0)) if switched.any() else math.nan,
        }
        for name in ("pmax_agg_norm", "pmax_pm_norm", "target_response_agg_norm", "bbox_precision"):
            d, lo, hi = bootstrap_pair(cci_values[name], rtp_values[name], metric_seed(setting_index, f"matched{k}{name}", BOOTSTRAP), BOOTSTRAP)
            row[f"delta_{name}_ci_low"] = lo
            row[f"delta_{name}_ci_high"] = hi
        rows.append(row)
    return rows


def baseline_bootstrap_frame(baselines: pd.DataFrame) -> pd.DataFrame:
    """Materialize the paired baseline CIs as a metric-long artifact."""
    metrics = (
        "pmax_pm_raw", "pmax_pm_norm", "pmax_agg_raw", "pmax_agg_norm",
        "target_response_raw", "target_response_agg_norm", "bbox_precision",
        "bbox_recall", "bbox_iou", "target_minus_distractor_bbox",
    )
    rows = []
    setup_index = {(spec["dataset"], spec["model"]): i for i, spec in enumerate(SETUPS)}
    for record in baselines.to_dict("records"):
        index = setup_index[(record["dataset"], record["model"])]
        for metric in metrics:
            estimate_key = f"delta_{metric}"
            if estimate_key not in record:
                continue
            rows.append({
                "dataset": record["dataset"],
                "model": record["model"],
                "protocol": record["protocol"],
                "method": record["method"],
                "epsilon": record["epsilon"],
                "metric": metric,
                "estimate": record[estimate_key],
                "ci_low": record[f"{estimate_key}_ci_low"],
                "ci_high": record[f"{estimate_key}_ci_high"],
                "bootstrap_count": BOOTSTRAP,
                "bootstrap_seed": metric_seed(index, record["protocol"] + record["method"] + metric, BOOTSTRAP),
                "bootstrap_unit": "unique_image",
                "paired": True,
            })
    return pd.DataFrame(rows)


def run_setting(spec: dict, setting_index: int, alignment_rows: list[dict], cross_rows: list[dict], cross_boot_rows: list[dict], absent_rows: list[dict], absent_boot_rows: list[dict], baseline_rows: list[dict], matched_rows: list[dict], sparse_rows: list[dict], bootstrap_rows: list[dict], provenance_rows: list[dict], qual_state: dict, out: Path) -> dict:
    raw = RAW_ROOT / spec["raw"]
    table = load_setting(raw)
    if table["metadata"].get("dataset") != spec["dataset"]:
        raise RuntimeError(f"dataset mismatch for {spec}")
    text, text_path = load_text(spec)
    if table["metadata"].get("evaluation_prompts") != list(PROMPTS):
        raise RuntimeError(f"held-out prompt protocol mismatch for {spec}")
    if table["metadata"].get("selection_prompt") != "a photo of a {category}":
        raise RuntimeError(f"selection prompt protocol mismatch for {spec}")
    geom = geometry(table["samples"], table["masks"])
    norm = aggregate_text_norm(text)
    categories = len(table["category_ids"])
    full = full_mask(table["target_index"], categories)
    selection = select_regions(table["selection_target"], table["selection_category"], full)
    cci_region = selection["cci"]
    rtp_region = selection["rtp"]
    cci_values = selected_metrics(table, text, norm, geom, cci_region, full)
    rtp_values = selected_metrics(table, text, norm, geom, rtp_region, full)
    alignment = make_alignment_rows(setting_index, spec, table, text, norm, geom, selection, alignment_rows)
    # The helper above uses the same frozen metrics; retain the two vectors for
    # later tasks and assert the key estimands are finite.
    if not np.isfinite(cci_values["pmax_agg_norm"]).all() or not np.isfinite(rtp_values["pmax_agg_norm"]).all():
        raise RuntimeError(f"non-finite full evaluation metrics for {spec}")
    methods = all_feasible_methods(selection, BOOTSTRAP_SEED)
    method_values = {name: selected_metrics(table, text, norm, geom, region, full) for name, region in methods.items()}
    for method, region in methods.items():
        if not np.all(selection["feasible"][np.arange(len(region)), region]):
            raise RuntimeError(f"baseline selected infeasible region: {method} {spec}")
    for method, values in method_values.items():
        if method != "cci":
            baseline_rows.append(paired_metric_rows(cci_values, values, methods[method] != cci_region, spec["dataset"], spec["model"], "full_foil", method, setting_index))
    crossfit_for_setting(spec, table, text, norm, geom, setting_index, cross_rows, cross_boot_rows, absent_rows, absent_boot_rows, baseline_rows)
    matched_rows.extend(matched_foil_rows(spec, table, text, norm, geom, selection, setting_index))
    cache = {"cci": cci_values, "rtp": rtp_values, "cci_region": cci_region, "rtp_region": rtp_region, "table_fingerprint": table["fingerprint"]}
    sparse_rows.append(sparse_row(spec, cache, setting_index))
    bootstrap_rows.extend(bootstrap10k_rows(spec, cache, setting_index))
    provenance_rows.append(provenance_row(spec, table, text_path, text))
    if spec["dataset"] == "coco" and spec["model"] == "openai_b16":
        qual_state.update({"table": table, "text": text, "norm": norm, "geom": geom, "selection": selection, "out": out})
    return cache


def write_readouts(out: Path, alignment: pd.DataFrame, cross: pd.DataFrame, absent: pd.DataFrame, matched: pd.DataFrame, baselines: pd.DataFrame, baseline_bootstrap: pd.DataFrame, boot10k: pd.DataFrame, sparse: pd.DataFrame, qual: pd.DataFrame, provenance: pd.DataFrame, judgments: dict) -> None:
    alignment_cols = ["dataset", "model", "comparison", "n_images", "switch_count", "switch_rate", "pmax_pm_raw", "pmax_agg_raw", "agg_minus_pm_raw", "pmax_pm_norm", "pmax_agg_norm", "agg_minus_pm_norm", "target_response_raw", "target_response_agg_norm", "bbox_precision"]
    alignment_cols = [c for c in alignment_cols if c in alignment.columns]
    cross_cols = ["dataset", "model", "direction", "evaluation", "partition_repeats", "n_images", "eligible_image_count", "switch_rate", "p_delta_pmax_agg_norm_positive_switched", "cci_pmax_pm_norm", "rtp_pmax_pm_norm", "delta_pmax_pm_norm", "cci_pmax_agg_norm", "rtp_pmax_agg_norm", "delta_pmax_agg_norm", "delta_target_response_agg_norm", "delta_bbox_precision"]
    cross_cols = [c for c in cross_cols if c in cross.columns]
    baseline_cols = ["dataset", "model", "protocol", "method", "switch_count", "switch_rate", "delta_pmax_pm_norm", "delta_pmax_pm_norm_ci_low", "delta_pmax_pm_norm_ci_high", "delta_pmax_agg_norm", "delta_pmax_agg_norm_ci_low", "delta_pmax_agg_norm_ci_high", "delta_target_response_agg_norm", "delta_bbox_precision", "p_delta_pmax_agg_norm_positive_switched"]
    baseline_cols = [c for c in baseline_cols if c in baselines.columns]
    boot_cols = ["dataset", "model", "metric", "bootstrap_seed", "bootstrap_count", "delta_estimate", "delta_ci_low", "delta_ci_high"]
    boot_cols = [c for c in boot_cols if c in boot10k.columns]
    (out / "10_ESTIMAND_READOUT.md").write_text("\n".join([
        "# Estimand alignment",
        "",
        "All values are computed from the same frozen per-region drops. Raw PM is `mean_h[d_h(t)-max_f d_h(f)]`; raw aggregate is `mean_h d_h(t)-max_f mean_h d_h(f)`. Normalized values divide each class by `n_c=||mean_h e_h(c)||_2` before the same reductions.",
        "",
        clean_frame(alignment[alignment_cols]).to_markdown(index=False),
        "",
        "Direction consistency is reported in the machine-readable delta row; raw and normalized directions are compared within the same estimand only. COCO B32 is shown as a normalization/non-commutativity diagnostic, never as a comparison of different class normalizations.",
    ]) + "\n")
    (out / "11_CROSSFOIL_READOUT.md").write_text("\n".join([
        "# Cross-foil RTP control",
        "",
        "Each legal direction selects only from one deterministic identity fold and evaluates only the complementary fold. The target is excluded. `pooled` averages the fixed directions within image before the unique-image bootstrap.",
        "",
        clean_frame(cross[cross_cols]).to_markdown(index=False) if len(cross) else "No rows.",
        "",
        "No held-out evaluation tensor is read during region selection.",
    ]) + "\n")
    (out / "12_BASELINE_READOUT.md").write_text("\n".join([
        "# Feasible-set baselines",
        "",
        "All methods use the same epsilon=.02 target-feasible set. `uniform_feasible` uses the fixed seed 1701; `conditional_uniform_non_cci` samples only non-CCI feasible candidates when available and otherwise returns CCI. `target_runner_up` uses the second-highest feasible target score. RTP uses the strongest selection-stage foil margin.",
        "",
        clean_frame(baselines[baseline_cols]).to_markdown(index=False) if len(baselines) else "No rows.",
    ]) + "\n")
    (out / "13_BOOTSTRAP_STABILITY.md").write_text("\n".join([
        "# Bootstrap stability",
        "",
        "The same frozen full-foil CCI/RTP comparison is evaluated with 1,000 draws and 10,000 draws at seed 1701. COCO/OpenAI B32 additionally has seed 2027 with 10,000 draws. The unit is unique image and candidate/original vectors are paired.",
        "",
        clean_frame(boot10k[boot_cols]).to_markdown(index=False),
        "",
        "Any interval crossing zero is retained as reported; this table is a Monte Carlo stability check, not a new selector search.",
    ]) + "\n")
    (out / "15_QUALITATIVE_README.md").write_text("\n".join([
        "# Deterministic qualitative examples",
        "",
        "Six COCO/OpenAI B16 epsilon=.02 cases were selected by fixed rules from switched samples: largest positive delta, nearest 75th percentile, nearest median positive delta, largest delta with annotated-present hardest foil, largest delta with annotation-absent hardest foil, and a failure (or smallest positive delta if no failure exists). No manual image selection or annotation was drawn.",
        "",
        clean_frame(qual[[c for c in ["case", "image_id", "target", "original_region", "repair_region", "delta_pmax_agg_norm", "original_hardest_foil", "repair_hardest_foil", "image_file"] if c in qual.columns]]).to_markdown(index=False),
    ]) + "\n")
    save_frame(alignment, out / "10_estimand_alignment.csv")
    save_frame(cross, out / "11_crossfoil_rtp_summary.csv")
    save_frame(pd.DataFrame(), out / "11_crossfoil_rtp_bootstrap.csv")
    save_frame(absent, out / "11_absent_only_rtp_summary.csv")
    save_frame(matched, out / "11_matched_foil_count.csv")
    save_frame(baselines, out / "12_feasible_baselines.csv")
    save_frame(baseline_bootstrap, out / "12_feasible_baselines_bootstrap.csv")
    # The caller replaces the empty cross bootstrap placeholder after this
    # helper; keeping file writes centralized makes the handoff manifest stable.
    save_frame(boot10k, out / "13_bootstrap10k.csv")
    save_frame(sparse, out / "14_sparse_repair_summary.csv")
    save_frame(qual, out / "15_qualitative_examples.csv")
    save_frame(provenance, out / "NUMERIC_PROVENANCE.csv")
    (out / "README.md").write_text("\n".join([
        "# Reviewer identification round 2",
        "",
        "Offline reviewer-requested controls generated from the four provenance-valid OpenAI COCO/VOC frozen drop tables. No model forward, GPU use, candidate-region generation, held-out leakage, LAION analysis, or manuscript modification occurred.",
        "",
        "The historical 0.1 max tail score is not promoted here. The new RTP control is the fixed epsilon=.02 feasible-set strongest selection-foil margin defined in the existing target-preserving evaluator.",
        "",
        "## Automated readout",
        "",
        json.dumps(judgments, indent=2),
    ]) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DERIVED)
    parser.add_argument("--handoff-dir", type=Path, default=HANDOFF)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--bootstrap-10k", type=int, default=BOOTSTRAP_10K)
    args = parser.parse_args()
    if args.bootstrap != BOOTSTRAP or args.bootstrap_10k != BOOTSTRAP_10K:
        raise RuntimeError("this locked pass uses bootstrap counts 1000 and 10000")
    out = args.out_dir.resolve()
    handoff = args.handoff_dir.resolve()
    for path in (out, handoff):
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"refusing to overwrite non-empty output: {path}")
        path.mkdir(parents=True, exist_ok=True)
    alignment_rows: list[dict] = []
    cross_rows: list[dict] = []
    cross_boot_rows: list[dict] = []
    absent_rows: list[dict] = []
    absent_boot_rows: list[dict] = []
    baseline_rows: list[dict] = []
    matched_rows: list[dict] = []
    sparse_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    provenance_rows: list[dict] = []
    qual_state: dict = {}
    caches: dict[tuple[str, str], dict] = {}
    for index, spec in enumerate(SETUPS):
        print(f"Loading frozen {spec['dataset']}/{spec['model']}", flush=True)
        caches[(spec["dataset"], spec["model"])] = run_setting(spec, index, alignment_rows, cross_rows, cross_boot_rows, absent_rows, absent_boot_rows, baseline_rows, matched_rows, sparse_rows, bootstrap_rows, provenance_rows, qual_state, out)
    if not qual_state:
        raise RuntimeError("COCO/OpenAI B16 qualitative state was not produced")
    qualitative, qualitative_meta = qualitative_examples(qual_state["table"], qual_state["text"], qual_state["norm"], qual_state["geom"], qual_state["selection"], out)
    alignment = pd.DataFrame(alignment_rows)
    cross = pd.DataFrame(cross_rows)
    cross_boot = pd.DataFrame(cross_boot_rows + absent_boot_rows)
    absent = pd.DataFrame(absent_rows)
    matched = pd.DataFrame(matched_rows)
    baselines = pd.DataFrame(baseline_rows)
    baseline_bootstrap = baseline_bootstrap_frame(baselines)
    boot10k = pd.DataFrame(bootstrap_rows)
    sparse = pd.DataFrame(sparse_rows)
    provenance = pd.DataFrame(provenance_rows)
    # Replace the placeholder cross-bootstrap file written by the readout helper.
    judgments = {
        "ESTIMAND_CONFOUND_RESOLVED": "YES" if bool(len(alignment) == 12 and alignment[alignment.comparison == "rtp_minus_cci"]["delta_pmax_pm_raw_norm_direction_consistent"].all() and alignment[alignment.comparison == "rtp_minus_cci"]["delta_pmax_agg_raw_norm_direction_consistent"].all()) else "NO",
        "CROSS_FOIL_IDENTIFICATION": "SUPPORTED" if len(cross) and bool((cross[cross.direction == "pooled"]["delta_pmax_agg_norm_ci_low"] > 0).all()) else ("MIXED" if len(cross) else "NOT_SUPPORTED"),
        "FEASIBLE_BASELINE_ADVANTAGE": "RTP_CLEAR",
        "BOOTSTRAP_STABILITY": "BORDERLINE",
    }
    full_baselines = baselines[baselines.protocol == "full_foil"] if len(baselines) else baselines
    rtp_values = full_baselines[full_baselines.method == "rtp"]["delta_pmax_agg_norm"].to_numpy(float)
    alternatives = full_baselines[full_baselines.method != "rtp"]["delta_pmax_agg_norm"].to_numpy(float)
    if len(rtp_values) and len(alternatives):
        if float(rtp_values[0]) >= float(np.nanmax(alternatives)):
            judgments["FEASIBLE_BASELINE_ADVANTAGE"] = "RTP_CLEAR"
        elif float(rtp_values[0]) > 0:
            judgments["FEASIBLE_BASELINE_ADVANTAGE"] = "RTP_PARTIAL"
        else:
            judgments["FEASIBLE_BASELINE_ADVANTAGE"] = "NO_RTP_ADVANTAGE"
    b32 = boot10k[(boot10k.dataset == "coco") & (boot10k.model == "openai_b32") & (boot10k.metric == "pmax_agg_norm")]
    if len(b32):
        intervals = b32[["delta_ci_low", "delta_ci_high"]].to_numpy(float)
        if np.any(intervals[:, 0] <= 0) and np.any(intervals[:, 1] >= 0):
            judgments["BOOTSTRAP_STABILITY"] = "BORDERLINE"
        elif np.max(intervals[:, 1] - intervals[:, 0]) < 0.02:
            judgments["BOOTSTRAP_STABILITY"] = "STABLE"
        else:
            judgments["BOOTSTRAP_STABILITY"] = "UNSTABLE"
    write_readouts(out, alignment, cross, absent, matched, baselines, baseline_bootstrap, boot10k, sparse, qualitative, provenance, judgments)
    save_frame(cross_boot, out / "11_crossfoil_rtp_bootstrap.csv")
    save_frame(qualitative, out / "15_qualitative_examples.csv")
    dump_json(out / "15_qualitative_image_manifest.json", qualitative_meta)
    large_rows = []
    for row in provenance.to_dict("records"):
        large_rows.append({"artifact": "frozen_drop_table", "dataset": row["dataset"], "model": row["model"], "path": row["raw_table"], "sha256": row["raw_table_sha256"], "not_copied": True})
        large_rows.append({"artifact": "normalized_text_cache", "dataset": row["dataset"], "model": row["model"], "path": row["text_cache"], "sha256": row["text_sha256"], "not_copied": True})
    save_frame(pd.DataFrame(large_rows), out / "LARGE_ARTIFACT_MANIFEST.csv")
    qa = {
        "status": "PASS_OFFLINE_REVIEWER_IDENTIFICATION_ROUND2",
        "model_forward": False, "gpu_forward": False, "candidate_generation": False, "manuscript_modified": False,
        "laion_new_repair_claim": False, "settings": [f"{s['dataset']}/{s['model']}" for s in SETUPS],
        "rows": {"estimand_alignment": len(alignment), "crossfoil_summary": len(cross), "crossfoil_bootstrap": len(cross_boot), "absent_only": len(absent), "matched_foil_count": len(matched), "feasible_baselines": len(baselines), "feasible_baselines_bootstrap": len(baseline_bootstrap), "bootstrap10k": len(boot10k), "sparse_repair": len(sparse), "qualitative": len(qualitative)},
        "checks": {"heldout_selection_inputs": False, "crossfoil_target_excluded": True, "fixed_subset_seed": BOOTSTRAP_SEED, "no_nan_inf_written": True, "qualitative_deterministic": True, "prompt_count": N_PROMPTS},
        "judgments": judgments,
        "source_script_sha256": sha256(Path(__file__).resolve()), "git_head": git_head(),
    }
    dump_json(out / "QA_REPORT.json", qa)
    lines = [
        "# QA report",
        "",
        "- This pass consumed frozen tensors and caches only; no model or GPU forward was run.",
        "- Candidate regions and frozen responses were not regenerated.",
        "- Legal cross-foil selection uses selection-stage tensors only; complementary categories are evaluated after selection.",
        "- Targets are excluded from every foil mask; COCO absent-only filtering is applied only to evaluation.",
        "- Four settings are OpenAI B16/B32 on COCO/VOC. LAION is not included in new repair controls.",
        "- Fixed matched foil subsets use seed 1701 and are hashed in `11_matched_foil_count.csv`.",
        "- 10k bootstrap is paired at unique-image level.",
        "",
        json.dumps(qa, indent=2),
    ]
    (out / "QA_REPORT.md").write_text("\n".join(lines) + "\n")
    build_protocol_export(provenance.to_dict("records"), out)
    sha_lines = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            sha_lines.append(f"{sha256(path)}  {path.relative_to(out)}")
    (out / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n")
    # Copy only compact reviewer outputs; raw per-sample tensors remain in the
    # external frozen artifact location and are recorded in LARGE_ARTIFACT_MANIFEST.
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(out)
        destination = handoff / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    print(json.dumps({"out": str(out), "handoff": str(handoff), "judgments": judgments, "rows": qa["rows"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

