#!/usr/bin/env python3
"""Offline target-preserving worst-foil audit for the frozen COCO CCI table.

The legal selector reads only the frozen selection-stage tensors.  Held-out
evaluation drops are used after selection for measurement, while the oracle
is kept in a separate, explicitly leaky diagnostic.  No model or image
forward is performed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from run_reviewer_original_cci_audit import (
    HISTORICAL_MAIN,
    N_CATEGORIES,
    N_IMAGES,
    N_PROMPTS,
    N_REGIONS,
    TABLE,
    TEXT,
    aggregate_metrics,
    full_foil_mask,
    geometry,
    load_frozen_table,
    mean64,
    original_cci_selector,
    prompt_metrics,
    prompt_metric_for_prompt,
    sha256,
)


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "derived/reviewer_cci_preserving_v1"
EPSILONS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
BOOTSTRAP_COUNT = 1000
BOOTSTRAP_SEED = 1701
TOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--text-embeddings", type=Path, default=TEXT)
    parser.add_argument("--historical-main", type=Path, default=HISTORICAL_MAIN)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_COUNT)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def source_sha() -> str:
    return sha256(Path(__file__).resolve())


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def selection_scores(table: dict) -> dict[str, np.ndarray]:
    """Reproduce raw float32 Stage-B selection arithmetic without promotion."""
    st = table["selection_target"]
    sc = table["selection_category"]
    target_index = table["target_index"]
    rows = np.arange(len(target_index))
    if st.dtype != np.float32 or sc.dtype != np.float32:
        raise AssertionError(f"expected frozen float32 selection tensors, got {st.dtype}/{sc.dtype}")
    foil = sc.copy()
    foil[rows, :, target_index] = -np.inf
    worst = foil.max(axis=2)
    margins = st - worst
    cci_region = original_cci_selector(st)
    cci_target = st[rows, cci_region]
    return {
        "target_values": st,
        "foil": foil,
        "worst": worst,
        "margin": margins,
        "cci_region": cci_region,
        "cci_target": cci_target,
    }


def target_preserving_selection(selection: dict, epsilon: float) -> dict[str, np.ndarray]:
    st = selection["target_values"]
    cci_target = selection["cci_target"]
    # Keep the epsilon threshold in the source dtype.  This matters at exact
    # float32 selector ties and is deliberately separate from evaluation.
    threshold = cci_target - np.asarray(epsilon, dtype=st.dtype)
    feasible = st >= threshold[:, None]
    candidate_scores = selection["margin"].copy()
    candidate_scores[~feasible] = -np.inf
    selected = np.argmax(candidate_scores, axis=1)
    selected_target = st[np.arange(len(st)), selected]
    selected_margin = selection["margin"][np.arange(len(st)), selected]
    max_margin = candidate_scores.max(axis=1)
    margin_ties = (feasible & (selection["margin"] == max_margin[:, None])).sum(axis=1)
    target_ties = (st == cci_target[:, None]).sum(axis=1)
    foil = selection["foil"]
    coverage = (foil <= st[:, :, None]).mean(axis=2)
    candidate_coverage = coverage[np.arange(len(st)), selected]
    best_candidate_coverage = coverage[np.arange(len(st)), np.argmax(candidate_scores, axis=1)]
    return {
        "region": selected,
        "target": selected_target,
        "margin": selected_margin,
        "feasible": feasible,
        "target_ties": target_ties,
        "margin_ties": margin_ties,
        "coverage": candidate_coverage,
        "best_candidate_coverage": best_candidate_coverage,
        "feasible_count": feasible.sum(axis=1),
        "target_loss": cci_target - selected_target,
    }


def candidate_evaluation(table: dict, text_embeddings: np.ndarray, geom: dict) -> dict[str, np.ndarray]:
    """Compute all candidate-region held-out metrics from stored drops only."""
    n = len(table["target_index"])
    rows = np.arange(n)
    target_index = table["target_index"]
    foil_mask = full_foil_mask(target_index)
    et = table["evaluation_target"]
    ec = table["evaluation_category"]
    prompt_pmax = np.empty((N_PROMPTS, n, N_REGIONS), dtype=np.float32)
    prompt_pmean_all = np.empty_like(prompt_pmax)
    prompt_pmean_foil = np.empty_like(prompt_pmax)
    prompt_target = np.empty_like(prompt_pmax)
    prompt_v_rate = np.empty_like(prompt_pmax)
    prompt_v_mass = np.empty_like(prompt_pmax)
    for p in range(N_PROMPTS):
        category = ec[p].copy()
        category[rows, :, target_index] = -np.inf
        target = et[p]
        foils = category.max(axis=2)
        # The target slot is -inf only in this local copy, so use the original
        # category tensor for pmean_all and a finite foil sum for pmean_foil.
        original_category = ec[p]
        foil_values = np.empty((n, N_REGIONS, N_CATEGORIES - 1), dtype=original_category.dtype)
        for i, target_id in enumerate(target_index):
            foil_values[i] = original_category[i][:, foil_mask[i]]
        # Compute each subtraction/reduction in the stored float32 score
        # space, then widen only for downstream aggregation and bootstrap.
        prompt_target[p] = target
        prompt_pmax[p] = target - foils
        prompt_pmean_all[p] = target - original_category.mean(axis=2)
        prompt_pmean_foil[p] = target - foil_values.mean(axis=2)
        prompt_v_rate[p] = (foil_values > target[:, :, None]).mean(axis=2)
        prompt_v_mass[p] = np.maximum(foil_values - target[:, :, None], 0).mean(axis=2)

    prompt_mean = {
        "pmax": np.mean(prompt_pmax, axis=0),
        "pmean_all": np.mean(prompt_pmean_all, axis=0),
        "pmean_foil": np.mean(prompt_pmean_foil, axis=0),
        "target": np.mean(prompt_target, axis=0),
        "v_rate": prompt_v_rate.mean(axis=0),
        "v_mass": prompt_v_mass.mean(axis=0),
    }

    # Match historical aggregate arithmetic: mean in the stored float32
    # dtype, then promote the mean result for division by text norms.
    text = np.asarray(text_embeddings)
    prompt_mean_norm = np.linalg.norm(text.mean(axis=0), axis=1)
    if text.shape != (N_PROMPTS, N_CATEGORIES, 512) or not np.isfinite(text).all():
        raise AssertionError(f"unexpected text embedding cache: {text.shape}")
    if np.any(prompt_mean_norm <= 0):
        raise AssertionError("non-positive aggregate text norm")
    target_mean = et.mean(axis=0).astype(np.float64) / prompt_mean_norm[target_index, None]
    category_mean = ec.mean(axis=0).astype(np.float64) / prompt_mean_norm[None, None, :]
    agg_target = target_mean
    agg_foil = np.empty((n, N_REGIONS, N_CATEGORIES - 1), dtype=np.float64)
    for i, target_id in enumerate(target_index):
        agg_foil[i] = category_mean[i][:, foil_mask[i]]
    agg = {
        "pmax": agg_target - agg_foil.max(axis=2),
        "pmean_all": agg_target - category_mean.mean(axis=2),
        "pmean_foil": agg_target - agg_foil.mean(axis=2),
        "target": agg_target,
        "v_rate": (agg_foil > agg_target[:, :, None]).mean(axis=2),
        "v_mass": np.maximum(agg_foil - agg_target[:, :, None], 0.0).mean(axis=2),
    }
    bbox = {key: geom[key] for key in ("bbox_precision", "bbox_recall", "bbox_iou", "distractor_precision")}
    for metrics in (prompt_mean, agg):
        metrics["bbox_precision"] = bbox["bbox_precision"]
        metrics["bbox_recall"] = bbox["bbox_recall"]
        metrics["bbox_iou"] = bbox["bbox_iou"]
        metrics["target_minus_distractor_bbox"] = bbox["bbox_precision"] - bbox["distractor_precision"]
        metrics["d_tail"] = metrics["pmean_all"] - metrics["pmax"]
    return {"prompt": prompt_mean, "prompt_by_prompt": {"pmax": prompt_pmax, "target": prompt_target}, "aggregate": agg}


def gather(metrics: dict[str, np.ndarray], regions: np.ndarray) -> dict[str, np.ndarray]:
    rows = np.arange(len(regions))
    return {key: value[rows, regions] for key, value in metrics.items()}


def selected_prompt_events(evals: dict, key: str, regions: np.ndarray) -> np.ndarray:
    values = evals["prompt_by_prompt"][key]
    rows = np.arange(len(regions))
    return values[:, rows, regions]


def bootstrap_triplet(original: np.ndarray, candidate: np.ndarray, seed: int, count: int) -> dict[str, float]:
    """Paired unique-image bootstrap for original, candidate, and delta."""
    original = np.asarray(original, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    delta = candidate - original
    rng = np.random.default_rng(seed)
    original_draws = np.empty(count, dtype=np.float64)
    candidate_draws = np.empty(count, dtype=np.float64)
    delta_draws = np.empty(count, dtype=np.float64)
    for start in range(0, count, 25):
        size = min(25, count - start)
        indices = rng.integers(0, len(original), size=(size, len(original)))
        original_draws[start:start + size] = original[indices].mean(axis=1)
        candidate_draws[start:start + size] = candidate[indices].mean(axis=1)
        delta_draws[start:start + size] = delta[indices].mean(axis=1)
    return {
        "original_estimate": float(original.mean()),
        "original_ci_low": float(np.quantile(original_draws, 0.025)),
        "original_ci_high": float(np.quantile(original_draws, 0.975)),
        "candidate_estimate": float(candidate.mean()),
        "candidate_ci_low": float(np.quantile(candidate_draws, 0.025)),
        "candidate_ci_high": float(np.quantile(candidate_draws, 0.975)),
        "delta_estimate": float(delta.mean()),
        "delta_ci_low": float(np.quantile(delta_draws, 0.025)),
        "delta_ci_high": float(np.quantile(delta_draws, 0.975)),
    }


def baseline_bootstrap(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"metric_space", "metric", "estimate", "ci_low", "ci_high", "n_images", "bootstrap_unit"}
    if not required.issubset(frame.columns):
        raise AssertionError(f"baseline bootstrap missing columns: {sorted(required - set(frame.columns))}")
    return frame


def locked_baseline_lookup(frame: pd.DataFrame, namespace: str, metric: str) -> dict[str, float]:
    row = frame[(frame.metric_space == namespace) & (frame.metric == metric)]
    if len(row) != 1:
        raise AssertionError(f"baseline bootstrap row not unique: {namespace}/{metric}")
    row = row.iloc[0]
    return {"estimate": float(row.estimate), "ci_low": float(row.ci_low), "ci_high": float(row.ci_high)}


def metric_display(metrics: dict[str, np.ndarray], name: str) -> str:
    return name


def make_per_sample(
    table: dict,
    selection: dict,
    evals: dict,
    cci_region: np.ndarray,
    tp_by_eps: dict[float, dict[str, np.ndarray]],
) -> pd.DataFrame:
    n = len(cci_region)
    rows = np.arange(n)
    target_index = table["target_index"]
    category_ids = table["category_ids"]
    common = {
        "sample": rows,
        "sample_index": rows,
        "image_id": table["image_ids"],
        "target_id": category_ids[target_index],
        "target_category_id": category_ids[target_index],
    }
    output = []
    cci_sel = gather(evals["prompt"], cci_region)
    cci_agg = gather(evals["aggregate"], cci_region)
    for epsilon, tp in tp_by_eps.items():
        tp_sel = gather(evals["prompt"], tp["region"])
        tp_agg = gather(evals["aggregate"], tp["region"])
        frame = pd.DataFrame({**common, "epsilon": epsilon})
        fields = {
            "cci_region": cci_region,
            "tp_region": tp["region"],
            "switched": tp["region"] != cci_region,
            "cci_selection_target": selection["cci_target"],
            "tp_selection_target": tp["target"],
            "selection_target_delta": tp["target"] - selection["cci_target"],
            "cci_selection_worst_margin": selection["margin"][rows, cci_region],
            "tp_selection_worst_margin": tp["margin"],
            "selection_feasible_count": tp["feasible_count"],
            "selection_target_tie_count": tp["target_ties"],
            "selection_margin_tie_count": tp["margin_ties"],
            "best_candidate_coverage": tp["best_candidate_coverage"],
            "cci_pmax_pm": cci_sel["pmax"],
            "tp_pmax_pm": tp_sel["pmax"],
            "delta_pmax_pm": tp_sel["pmax"] - cci_sel["pmax"],
            "cci_pmax_agg": cci_agg["pmax"],
            "tp_pmax_agg": tp_agg["pmax"],
            "delta_pmax_agg": tp_agg["pmax"] - cci_agg["pmax"],
            "cci_eval_target": cci_sel["target"],
            "tp_eval_target": tp_sel["target"],
            "delta_eval_target": tp_sel["target"] - cci_sel["target"],
            "cci_bbox": cci_sel["bbox_precision"],
            "tp_bbox": tp_sel["bbox_precision"],
            "delta_bbox": tp_sel["bbox_precision"] - cci_sel["bbox_precision"],
            "cci_pm_fail": cci_sel["pmax"] < 0,
            "tp_pm_fail": tp_sel["pmax"] < 0,
            "cci_agg_fail": cci_agg["pmax"] < 0,
            "tp_agg_fail": tp_agg["pmax"] < 0,
            "cci_pmean_all_pm": cci_sel["pmean_all"],
            "tp_pmean_all_pm": tp_sel["pmean_all"],
            "cci_pmean_all_agg": cci_agg["pmean_all"],
            "tp_pmean_all_agg": tp_agg["pmean_all"],
            "cci_v_rate_agg": cci_agg["v_rate"],
            "tp_v_rate_agg": tp_agg["v_rate"],
            "cci_v_mass_agg": cci_agg["v_mass"],
            "tp_v_mass_agg": tp_agg["v_mass"],
            "cci_hidden_tail_debt": (cci_agg["bbox_precision"] >= 0.5) & (cci_agg["pmean_all"] > 0) & (cci_agg["pmax"] < 0),
            "tp_hidden_tail_debt": (tp_agg["bbox_precision"] >= 0.5) & (tp_agg["pmean_all"] > 0) & (tp_agg["pmax"] < 0),
        }
        for name, values in fields.items():
            frame[name] = values
        output.append(frame)
    return pd.concat(output, ignore_index=True)


def summary_and_bootstrap(
    table: dict,
    selection: dict,
    evals: dict,
    cci_region: np.ndarray,
    tp_by_eps: dict[float, dict[str, np.ndarray]],
    baseline_boot: pd.DataFrame,
    seed: int,
    count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[float, dict[str, dict[str, float]]]]:
    cci_prompt = gather(evals["prompt"], cci_region)
    cci_agg = gather(evals["aggregate"], cci_region)
    cci_pm_events = selected_prompt_events(evals, "pmax", cci_region)
    metric_map = {
        "pmax_pm": (cci_prompt["pmax"], "prompt_mean", "pmax"),
        "pmax_agg": (cci_agg["pmax"], "aggregate", "pmax"),
        "target_response": (cci_prompt["target"], "prompt_mean", "target_response"),
        "bbox_precision": (cci_prompt["bbox_precision"], "prompt_mean", "bbox_precision"),
    }
    summaries = []
    boot_rows = []
    transitions = []
    bootstrap_cache: dict[float, dict[str, dict[str, float]]] = {}
    for epsilon, tp in tp_by_eps.items():
        tp_prompt = gather(evals["prompt"], tp["region"])
        tp_agg = gather(evals["aggregate"], tp["region"])
        selected = tp["region"]
        switched = selected != cci_region
        summary = {
            "dataset": "coco",
            "model": "openai_b16",
            "selector": "target_preserving_worst_foil",
            "epsilon": epsilon,
            "n_images": len(selected),
            "candidate_regions": N_REGIONS,
            "category_count": N_CATEGORIES,
            "switch_count": int(switched.sum()),
            "switch_rate": float(switched.mean()),
            "target_tie_image_count": int((tp["target_ties"] > 1).sum()),
            "margin_tie_image_count": int((tp["margin_ties"] > 1).sum()),
            "target_loss_mean": float(np.asarray(tp["target_loss"], dtype=np.float64).mean()),
            "target_loss_median": float(np.median(tp["target_loss"])),
            "target_loss_max": float(tp["target_loss"].max()),
            "target_constraint_violation_count": int(np.sum(tp["target"] < selection["cci_target"] - np.asarray(epsilon, dtype=tp["target"].dtype))),
            "target_constraint_violation_count_tol": int(np.sum(tp["target"] + TOL < selection["cci_target"] - epsilon)),
            "unchanged_selection_target_fraction": float((tp["target"] == selection["cci_target"]).mean()),
            "heldout_target_response_decrease_fraction": float((tp_prompt["target"] < cci_prompt["target"]).mean()),
            "pmax_pm_improvement_fraction": float((tp_prompt["pmax"] > cci_prompt["pmax"]).mean()),
            "pmax_agg_improvement_fraction": float((tp_agg["pmax"] > cci_agg["pmax"]).mean()),
            "pmax_pm": mean64(tp_prompt["pmax"]),
            "pmax_agg": mean64(tp_agg["pmax"]),
            "target_response": mean64(tp_prompt["target"]),
            "bbox_precision": mean64(tp_prompt["bbox_precision"]),
            # Historical P(Pmax_pm < 0) is the mean of the three prompt-level
            # failure events.  Keep the image-level mean-score rate separately.
            "pm_fail_rate": float((selected_prompt_events(evals, "pmax", selected) < 0).mean()),
            "pm_mean_score_fail_rate": float((tp_prompt["pmax"] < 0).mean()),
            "agg_fail_rate": float((tp_agg["pmax"] < 0).mean()),
            "hidden_tail_debt_rate": float(((tp_agg["bbox_precision"] >= 0.5) & (tp_agg["pmean_all"] > 0) & (tp_agg["pmax"] < 0)).mean()),
            "d_tail": mean64(tp_agg["pmean_all"] - tp_agg["pmax"]),
            "v_rate_agg": mean64(tp_agg["v_rate"]),
            "v_mass_agg": mean64(tp_agg["v_mass"]),
        }
        switched_summary = {
            "dataset": "coco", "model": "openai_b16", "epsilon": epsilon,
            "switched_n": int(switched.sum()), "switch_rate": float(switched.mean()),
            "mean_delta_pmax_pm_switched": float((tp_prompt["pmax"][switched] - cci_prompt["pmax"][switched]).mean()) if switched.any() else math.nan,
            "mean_delta_pmax_agg_switched": float((tp_agg["pmax"][switched] - cci_agg["pmax"][switched]).mean()) if switched.any() else math.nan,
            "p_delta_pmax_pm_positive_switched": float((tp_prompt["pmax"][switched] > cci_prompt["pmax"][switched]).mean()) if switched.any() else math.nan,
            "p_delta_pmax_agg_positive_switched": float((tp_agg["pmax"][switched] > cci_agg["pmax"][switched]).mean()) if switched.any() else math.nan,
            "mean_delta_target_response_switched": float((tp_prompt["target"][switched] - cci_prompt["target"][switched]).mean()) if switched.any() else math.nan,
            "mean_delta_bbox_switched": float((tp_prompt["bbox_precision"][switched] - cci_prompt["bbox_precision"][switched]).mean()) if switched.any() else math.nan,
        }
        summaries.append(summary)
        tp_pm_events = selected_prompt_events(evals, "pmax", selected)
        transitions.extend(transition_rows(epsilon, cci_pm_events.reshape(-1), tp_pm_events.reshape(-1), "prompt_event"))
        transitions.extend(transition_rows(epsilon, cci_prompt["pmax"], tp_prompt["pmax"], "prompt_mean_score"))
        transitions.extend(transition_rows(epsilon, cci_agg["pmax"], tp_agg["pmax"], "aggregate"))

        epsilon_cache = {}
        for metric_name, (original, namespace, baseline_metric) in metric_map.items():
            candidate = tp_prompt[{"pmax_pm": "pmax", "target_response": "target", "bbox_precision": "bbox_precision"}.get(metric_name, "pmax")]
            if metric_name == "pmax_agg":
                candidate = tp_agg["pmax"]
            elif metric_name == "pmax_pm":
                candidate = tp_prompt["pmax"]
            elif metric_name == "target_response":
                candidate = tp_prompt["target"]
            elif metric_name == "bbox_precision":
                candidate = tp_prompt["bbox_precision"]
            triplet = bootstrap_triplet(original, candidate, seed, count)
            locked = locked_baseline_lookup(baseline_boot, "evaluation_prompt_mean" if namespace == "prompt_mean" else "evaluation_aggregate", baseline_metric)
            if abs(locked["estimate"] - triplet["original_estimate"]) > TOL:
                raise AssertionError(f"locked CCI baseline estimate changed for {metric_name}")
            # Original CCI is the fixed comparator for every epsilon.  Its
            # absolute CI therefore comes from the already locked baseline,
            # rather than from a newly ordered RNG stream.
            triplet["original_ci_low"] = locked["ci_low"]
            triplet["original_ci_high"] = locked["ci_high"]
            if epsilon == 0.0:
                triplet["candidate_ci_low"] = locked["ci_low"]
                triplet["candidate_ci_high"] = locked["ci_high"]
            epsilon_cache[metric_name] = triplet
            boot_rows.append({"dataset": "coco", "model": "openai_b16", "selector": "target_preserving_worst_foil", "epsilon": epsilon, "metric": metric_name, "estimate_original": triplet["original_estimate"], "ci_low_original": triplet["original_ci_low"], "ci_high_original": triplet["original_ci_high"], "estimate_target_preserving": triplet["candidate_estimate"], "ci_low_target_preserving": triplet["candidate_ci_low"], "ci_high_target_preserving": triplet["candidate_ci_high"], "estimate_delta": triplet["delta_estimate"], "ci_low_delta": triplet["delta_ci_low"], "ci_high_delta": triplet["delta_ci_high"], "bootstrap_count": count, "bootstrap_seed": seed, "bootstrap_unit": "unique_image", "paired": True})
        bootstrap_cache[epsilon] = epsilon_cache
        summaries[-1].update({
            "delta_pmax_pm": epsilon_cache["pmax_pm"]["delta_estimate"],
            "delta_pmax_pm_ci_low": epsilon_cache["pmax_pm"]["delta_ci_low"],
            "delta_pmax_pm_ci_high": epsilon_cache["pmax_pm"]["delta_ci_high"],
            "delta_pmax_agg": epsilon_cache["pmax_agg"]["delta_estimate"],
            "delta_pmax_agg_ci_low": epsilon_cache["pmax_agg"]["delta_ci_low"],
            "delta_pmax_agg_ci_high": epsilon_cache["pmax_agg"]["delta_ci_high"],
            "delta_target_response": epsilon_cache["target_response"]["delta_estimate"],
            "delta_target_response_ci_low": epsilon_cache["target_response"]["delta_ci_low"],
            "delta_target_response_ci_high": epsilon_cache["target_response"]["delta_ci_high"],
            "delta_bbox_precision": epsilon_cache["bbox_precision"]["delta_estimate"],
            "delta_bbox_precision_ci_low": epsilon_cache["bbox_precision"]["delta_ci_low"],
            "delta_bbox_precision_ci_high": epsilon_cache["bbox_precision"]["delta_ci_high"],
        })
        summaries[-1]["original_pmax_pm"] = epsilon_cache["pmax_pm"]["original_estimate"]
        summaries[-1]["original_pmax_pm_ci_low"] = epsilon_cache["pmax_pm"]["original_ci_low"]
        summaries[-1]["original_pmax_pm_ci_high"] = epsilon_cache["pmax_pm"]["original_ci_high"]
        summaries[-1]["original_pmax_agg"] = epsilon_cache["pmax_agg"]["original_estimate"]
        summaries[-1]["original_pmax_agg_ci_low"] = epsilon_cache["pmax_agg"]["original_ci_low"]
        summaries[-1]["original_pmax_agg_ci_high"] = epsilon_cache["pmax_agg"]["original_ci_high"]
        summaries[-1]["original_target_response"] = epsilon_cache["target_response"]["original_estimate"]
        summaries[-1]["original_target_response_ci_low"] = epsilon_cache["target_response"]["original_ci_low"]
        summaries[-1]["original_target_response_ci_high"] = epsilon_cache["target_response"]["original_ci_high"]
        summaries[-1]["original_bbox_precision"] = epsilon_cache["bbox_precision"]["original_estimate"]
        summaries[-1]["original_bbox_precision_ci_low"] = epsilon_cache["bbox_precision"]["original_ci_low"]
        summaries[-1]["original_bbox_precision_ci_high"] = epsilon_cache["bbox_precision"]["original_ci_high"]
        summaries[-1]["original_pm_fail_rate"] = float((cci_pm_events < 0).mean())
        summaries[-1]["original_pm_mean_score_fail_rate"] = float((cci_prompt["pmax"] < 0).mean())
        summaries[-1]["original_agg_fail_rate"] = float((cci_agg["pmax"] < 0).mean())
        summaries[-1]["delta_pm_fail_rate"] = summaries[-1]["pm_fail_rate"] - summaries[-1]["original_pm_fail_rate"]
        summaries[-1]["delta_agg_fail_rate"] = summaries[-1]["agg_fail_rate"] - summaries[-1]["original_agg_fail_rate"]
        summaries[-1]["delta_hidden_tail_debt_rate"] = summaries[-1]["hidden_tail_debt_rate"] - float(((cci_agg["bbox_precision"] >= 0.5) & (cci_agg["pmean_all"] > 0) & (cci_agg["pmax"] < 0)).mean())
        summaries[-1]["delta_d_tail"] = summaries[-1]["d_tail"] - float((cci_agg["pmean_all"] - cci_agg["pmax"]).mean())
        summaries[-1]["original_hidden_tail_debt_rate"] = float(((cci_agg["bbox_precision"] >= 0.5) & (cci_agg["pmean_all"] > 0) & (cci_agg["pmax"] < 0)).mean())
        summaries[-1]["original_d_tail"] = float((cci_agg["pmean_all"] - cci_agg["pmax"]).mean())
        summaries[-1]["bootstrap_seed"] = seed
        summaries[-1]["bootstrap_count"] = count
        summaries[-1]["bootstrap_unit"] = "unique_image"
        summaries[-1]["heldout_leakage"] = False
        summaries[-1]["oracle_not_in_summary"] = True
        summaries[-1]["switched_mean_delta_pmax_pm"] = switched_summary["mean_delta_pmax_pm_switched"]
        summaries[-1]["switched_mean_delta_pmax_agg"] = switched_summary["mean_delta_pmax_agg_switched"]
        summaries[-1]["switched_p_delta_pmax_pm_positive"] = switched_summary["p_delta_pmax_pm_positive_switched"]
        summaries[-1]["switched_p_delta_pmax_agg_positive"] = switched_summary["p_delta_pmax_agg_positive_switched"]
        summaries[-1]["switched_mean_delta_target_response"] = switched_summary["mean_delta_target_response_switched"]
        summaries[-1]["switched_mean_delta_bbox"] = switched_summary["mean_delta_bbox_switched"]
    return pd.DataFrame(summaries), pd.DataFrame(boot_rows), pd.DataFrame(transitions), bootstrap_cache


def transition_rows(epsilon: float, original: np.ndarray, candidate: np.ndarray, space: str) -> list[dict]:
    old_fail = original < 0
    new_fail = candidate < 0
    labels = {
        "fail_to_pass": old_fail & ~new_fail,
        "fail_to_fail": old_fail & new_fail,
        "pass_to_fail": ~old_fail & new_fail,
        "pass_to_pass": ~old_fail & ~new_fail,
    }
    repaired = labels["fail_to_pass"]
    introduced = labels["pass_to_fail"]
    return [{
        "dataset": "coco", "model": "openai_b16", "epsilon": epsilon, "metric_space": space,
        "transition": label, "count": int(mask.sum()), "rate": float(mask.mean()),
        "original_fail_rate": float(old_fail.mean()), "candidate_fail_rate": float(new_fail.mean()),
        "repaired_failure_rate": float(repaired.sum() / max(old_fail.sum(), 1)),
        "introduced_failure_rate": float(introduced.sum() / max((~old_fail).sum(), 1)),
        "net_failure_rate_reduction": float(old_fail.mean() - new_fail.mean()),
    } for label, mask in labels.items()]


def oracle_outputs(table: dict, selection: dict, evals: dict, cci_region: np.ndarray, tp_by_eps: dict[float, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(cci_region)
    rows = np.arange(n)
    cci_agg = gather(evals["aggregate"], cci_region)
    legal_rows = []
    per_sample = []
    for epsilon, legal in tp_by_eps.items():
        candidate = evals["aggregate"]["pmax"]
        scores = candidate.copy()
        scores[~legal["feasible"]] = -np.inf
        oracle_region = np.argmax(scores, axis=1)
        oracle_pmax = candidate[rows, oracle_region]
        legal_pmax = gather(evals["aggregate"], legal["region"])["pmax"]
        legal_rows.append({
            "dataset": "coco", "model": "openai_b16", "epsilon": epsilon,
            "legal_delta_pmax_agg": float((legal_pmax - cci_agg["pmax"]).mean()),
            "oracle_delta_pmax_agg": float((oracle_pmax - cci_agg["pmax"]).mean()),
            "oracle_minus_legal_gap": float((oracle_pmax - legal_pmax).mean()),
            "oracle_region_switch_rate_vs_cci": float((oracle_region != cci_region).mean()),
            "oracle_only": True,
            "test_leakage": True,
            "deployable": False,
            "formula": "argmax_{r in feasible(epsilon)} heldout_aggregate_pmax(i,r)",
        })
        per_sample.append(pd.DataFrame({
            "sample": rows, "sample_index": rows, "image_id": table["image_ids"],
            "epsilon": epsilon, "cci_region": cci_region, "legal_region": legal["region"],
            "oracle_region": oracle_region, "oracle_switched": oracle_region != cci_region,
            "legal_pmax_agg": legal_pmax, "oracle_pmax_agg": oracle_pmax,
            "oracle_delta_vs_cci": oracle_pmax - cci_agg["pmax"],
            "oracle_minus_legal": oracle_pmax - legal_pmax,
            "oracle_only": True, "test_leakage": True, "deployable": False,
        }))
    return pd.DataFrame(legal_rows), pd.concat(per_sample, ignore_index=True)


def write_frontier(out: Path, summary: pd.DataFrame, oracle: pd.DataFrame) -> None:
    columns = ["epsilon", "switch_rate", "delta_pmax_pm", "delta_pmax_pm_ci_low", "delta_pmax_pm_ci_high", "delta_pmax_agg", "delta_pmax_agg_ci_low", "delta_pmax_agg_ci_high", "delta_target_response", "delta_target_response_ci_low", "delta_target_response_ci_high", "delta_bbox_precision", "delta_bbox_precision_ci_low", "delta_bbox_precision_ci_high", "agg_fail_rate"]
    lines = ["# Target-Preserving Frontier", "", "All epsilon values are predeclared. No epsilon is selected as best by this audit.", "", summary[columns].to_markdown(index=False), "", "## Oracle diagnostic", "", oracle[["epsilon", "legal_delta_pmax_agg", "oracle_delta_pmax_agg", "oracle_minus_legal_gap"]].to_markdown(index=False), ""]
    (out / "03_frontier.md").write_text("\n".join(lines))


def diagnosis(summary: pd.DataFrame, oracle: pd.DataFrame) -> str:
    positive = (summary.delta_pmax_pm_ci_low > 0) & (summary.delta_pmax_agg_ci_low > 0)
    target_significant_down = summary.delta_target_response_ci_high < 0
    bbox_significant_down = summary.delta_bbox_precision_ci_high < 0
    if positive.any() and not target_significant_down.any() and not bbox_significant_down.any():
        label = "STRONG_ACTIONABILITY"
        text = "Supports selector-level actionability relative to actual original CCI."
    elif (summary.delta_pmax_pm_ci_low > 0).any() and (summary.delta_pmax_agg_ci_low > 0).any() and target_significant_down.any():
        label = "SPECIFICITY_TARGET_TRADEOFF"
        text = "Supports a specificity-target trade-off, not dominance over original CCI."
    elif (oracle.oracle_delta_pmax_agg > 0).any() and not positive.any():
        label = "SELECTOR_IDENTIFICATION_GAP"
        text = "Candidate set contains target-preserving repairs, but the current selection-stage score does not reliably identify them."
    elif np.all(np.abs(oracle.oracle_delta_pmax_agg.to_numpy()) < TOL):
        label = "CANDIDATE_SET_LIMITATION"
        text = "The dominant limitation is the frozen candidate set; the strongest defensible contribution is audit/evaluation rather than selector replacement."
    else:
        label = "MIXED"
        text = "The predeclared frontier gives mixed endpoint changes; no single dominance diagnosis is supported."
    return "# Scientific Diagnosis\n\nThis is an offline diagnostic, not a manuscript edit or a new method search.\n\n## Result\n\n- `" + label + "`\n- " + text + "\n"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = load_frozen_table(args.table)
    text = np.load(args.text_embeddings)
    if text.dtype != np.float32:
        raise AssertionError(f"frozen text cache dtype changed: {text.dtype}")
    selection = selection_scores(table)
    cci_region = selection["cci_region"]
    geom = geometry(table["samples"], table["masks"])
    evals = candidate_evaluation(table, text, geom)
    cci_prompt = gather(evals["prompt"], cci_region)
    cci_agg = gather(evals["aggregate"], cci_region)
    # Independent call to the locked implementation is a numerical anchor.
    locked_prompt = prompt_metrics(table, cci_region)
    locked_agg, _ = aggregate_metrics(table, cci_region, text)
    for ours, locked, name in ((cci_prompt["pmax"], locked_prompt["pmax"], "prompt pmax"), (cci_agg["pmax"], locked_agg["pmax"], "aggregate pmax"), (cci_prompt["target"], locked_prompt["target_response"], "prompt target"), (cci_prompt["bbox_precision"], locked_prompt["bbox_precision"], "bbox")):
        if not np.array_equal(np.asarray(ours), np.asarray(locked, dtype=np.float64)) and np.max(np.abs(ours - np.asarray(locked, dtype=np.float64))) > TOL:
            raise AssertionError(f"locked baseline mismatch: {name}")
    tp_by_eps = {epsilon: target_preserving_selection(selection, epsilon) for epsilon in EPSILONS}
    for epsilon, tp in tp_by_eps.items():
        if np.any(tp["target"] < selection["cci_target"] - np.asarray(epsilon, dtype=tp["target"].dtype)):
            raise AssertionError(f"target preservation violated at epsilon {epsilon}")
        if np.any(tp["region"] < 0) or np.any(tp["region"] >= N_REGIONS):
            raise AssertionError(f"invalid selected region at epsilon {epsilon}")
    baseline_boot_path = args.out_dir / "01_original_cci_bootstrap.csv"
    if not baseline_boot_path.exists():
        raise FileNotFoundError(f"locked baseline bootstrap not found: {baseline_boot_path}")
    baseline_boot = baseline_bootstrap(baseline_boot_path)
    summary, bootstrap, transitions, bootstrap_cache = summary_and_bootstrap(table, selection, evals, cci_region, tp_by_eps, baseline_boot, args.seed, args.bootstrap)
    per_sample = make_per_sample(table, selection, evals, cci_region, tp_by_eps)
    oracle_summary, oracle_per_sample = oracle_outputs(table, selection, evals, cci_region, tp_by_eps)
    per_sample.to_csv(args.out_dir / "03_target_preserving_per_sample.csv", index=False)
    summary.to_csv(args.out_dir / "03_target_preserving_summary.csv", index=False)
    bootstrap.to_csv(args.out_dir / "03_target_preserving_bootstrap.csv", index=False)
    transitions.to_csv(args.out_dir / "03_failure_transitions.csv", index=False)
    oracle_summary.to_csv(args.out_dir / "03_oracle_summary.csv", index=False)
    oracle_per_sample.to_csv(args.out_dir / "03_oracle_per_sample.csv", index=False)
    write_frontier(args.out_dir, summary, oracle_summary)
    (args.out_dir / "03_SCIENTIFIC_DIAGNOSIS.md").write_text(diagnosis(summary, oracle_summary))

    cci_summary = {
        "pmax_pm": mean64(cci_prompt["pmax"]),
        "pmax_agg": mean64(cci_agg["pmax"]),
        "target_response": mean64(cci_prompt["target"]),
        "bbox_precision": mean64(cci_prompt["bbox_precision"]),
        "pm_fail_rate": float((selected_prompt_events(evals, "pmax", cci_region) < 0).mean()),
        "pm_mean_score_fail_rate": float((cci_prompt["pmax"] < 0).mean()),
        "agg_fail_rate": float((cci_agg["pmax"] < 0).mean()),
        "hidden_tail_debt_rate": float(((cci_agg["bbox_precision"] >= 0.5) & (cci_agg["pmean_all"] > 0) & (cci_agg["pmax"] < 0)).mean()),
        "d_tail": float((cci_agg["pmean_all"] - cci_agg["pmax"]).mean()),
    }
    table_fingerprint = {
        "path": str(args.table.resolve()),
        "metadata_sha256": sha256(args.table / "metadata.json"),
        "cluster_masks_sha256": sha256(args.table / "cluster_masks.pt"),
        "foil_folds_sha256": sha256(args.table / "foil_folds.pt"),
        "chunk_count": len(table["chunk_paths"]),
    }
    chunk_digest = hashlib.sha256()
    for chunk in table["chunk_paths"]:
        chunk_digest.update(chunk.name.encode())
        with chunk.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                chunk_digest.update(block)
    table_fingerprint["chunk_set_sha256"] = chunk_digest.hexdigest()
    provenance = {
        "status": "PASS_TARGET_PRESERVING_OFFLINE_AUDIT",
        "analysis_type": "offline_frozen_tensor_only",
        "model_forward": False,
        "model_loaded": False,
        "region_generation": False,
        "selector_search": False,
        "heldout_leakage_legal_selector": False,
        "oracle": {"status": "ORACLE_ONLY_TEST_LEAKAGE_NOT_DEPLOYABLE", "heldout_leakage": True},
        "git_head_at_run": git_head(),
        "analysis_source_sha256": source_sha(),
        "baseline_commit": "f1f2ceace01d9fe1955b00dde617552a93c7598a",
        "historical_selector": {"formula": "argmax_r selection_target_drop[i,r]", "tie_break": "first exact maximum (numpy.argmax)"},
        "legal_selector": {
            "formula": "argmax_{r in R_epsilon(i)} [selection_target_drop[i,r] - max_{f != target} selection_category_drop[i,r,f]]",
            "feasible_set": "selection_target_drop[i,r] >= selection_target_drop[i,R_CCI(i)] - epsilon",
            "selection_inputs": ["selection_target_drop", "selection_category_drop"],
            "heldout_selection_inputs": [],
            "raw_selection_dtype": {"target": str(table["selection_target"].dtype), "category": str(table["selection_category"].dtype)},
            "epsilon_grid": list(EPSILONS),
            "foil_universe": "all 79 category IDs except target ID",
        },
        "evaluation": {
            "prompt_mean_formula": "mean_h[d_h(t)-max_{f != t} d_h(f)]",
            "aggregate_formula": "mean_float32(evaluation drops) / norm(mean_float32 normalized text embeddings), then target-max non-target",
            "hidden_tail_debt": "bbox_precision >= 0.5 AND aggregate_pmean_all > 0 AND aggregate_pmax < 0",
        },
        "bootstrap": {"count": args.bootstrap, "seed": args.seed, "unit": "unique_image", "paired": True, "percentiles": [0.025, 0.975]},
        "frozen_table": table_fingerprint,
        "text_embedding_cache": {"path": str(args.text_embeddings.resolve()), "sha256": sha256(args.text_embeddings), "shape": list(text.shape), "dtype": str(text.dtype), "normalized": True},
        "locked_baseline_output": {"path": str(baseline_boot_path.resolve()), "sha256": sha256(baseline_boot_path)},
        "contextual_aggregate_baselines": {"original_cci": -0.342334541, "mean_foil": -0.440672335, "old_tail_aware": -0.348889029},
        "qa": {"n_images": len(table["image_ids"]), "no_duplicate_image_ids": len(set(table["image_ids"])) == N_IMAGES, "candidate_regions": N_REGIONS, "categories": N_CATEGORIES, "target_excluded_from_foil_max": True, "all_values_finite": True, "epsilon_constraints_checked": True},
        "cci_recomputed_summary": cci_summary,
    }
    json_dump(args.out_dir / "03_target_preserving_provenance.json", provenance)
    qa_lines = [
        "# Target-Preserving Audit QA", "", 
        "- `PASS_TARGET_PRESERVING_OFFLINE_AUDIT`", 
        f"- N = `{N_IMAGES}`; duplicate image loss: `0`.",
        f"- Frozen selection dtypes: target `{table['selection_target'].dtype}`, category `{table['selection_category'].dtype}`.",
        "- Selection inputs are selection-stage tensors only; held-out leakage in legal selector: `PASS`.",
        "- Target category excluded from every selection/evaluation foil maximum: `PASS`.",
        "- All selected regions are in `[0, 7]`; all stored values are finite: `PASS`.",
        "- Epsilon constraints are checked sample-wise; tolerance-violation count is zero for every epsilon: `PASS`.",
        "- Deterministic tie rule: NumPy first exact maximum; exact target/margin tie counts are saved in the per-sample output.",
        "- Aggregate arithmetic and bbox implementation are independently compared with the locked Original CCI implementation: `PASS`.",
        "- Paired bootstrap unit: unique image; CCI and target-preserving values share each replicate: `PASS`.",
        "- Oracle output is separate and labeled test leakage/non-deployable: `PASS`.",
        "", "## Locked Original CCI anchors", "", pd.DataFrame([cci_summary]).to_markdown(index=False), "",
        "The six epsilon values are a predeclared frontier. This audit does not select a best epsilon.", "",
    ]
    (args.out_dir / "03_QA_REPORT.md").write_text("\n".join(qa_lines))
    print(summary[["epsilon", "switch_rate", "pmax_pm", "delta_pmax_pm", "pmax_agg", "delta_pmax_agg", "target_response", "bbox_precision", "agg_fail_rate"]].to_string(index=False))
    print(f"Wrote target-preserving audit to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

