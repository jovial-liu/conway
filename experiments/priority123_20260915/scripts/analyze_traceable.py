#!/usr/bin/env python3
"""Analyze the independent, same-generation traceable captures.

The traceable batch files are the authority for candidate responses, masks and
candidate choices.  This script deliberately does not read the old frozen
arrays to fill new values; the old local-rerun CSVs are used only for a
separate comparison report when supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


SETTINGS = (
    ("coco_openai_b16", "coco", "openai_b16"),
    ("coco_openai_b32", "coco", "openai_b32"),
    ("voc2007_openai_b16", "voc2007", "openai_b16"),
    ("voc2007_openai_b32", "voc2007", "openai_b32"),
)
METHODS = ("cci", "wf", "mean", "max_0_1")
METRICS = ("margin_norm", "target_drop_raw", "target_drop_norm", "bbox_precision")
EPSILONS = (0.01, 0.02, 0.05, 0.10, 0.20)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trace-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--old-runs-root", type=Path)
    p.add_argument("--old-results-root", type=Path)
    p.add_argument("--bootstrap-count", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=1701)
    p.add_argument("--mc-seed-count", type=int, default=100)
    return p.parse_args()


def canonical_id(value) -> str:
    s = str(value)
    return str(int(s)) if s.isdigit() else s


def json_array(value) -> str:
    return json.dumps(np.asarray(value).tolist(), ensure_ascii=False, separators=(",", ":"))


def exact_q(n: int, k: int, m: int) -> float:
    if k == 0 or m == 0:
        return 0.0
    if k > n - m:
        return 1.0
    return 1.0 - math.comb(n - m, k) / math.comb(n, k)


def bootstrap_mean(values: np.ndarray, seed: int, count: int) -> tuple[float, float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    draws = np.empty(count, dtype=float)
    for start in range(0, count, 64):
        size = min(64, count - start)
        idx = rng.integers(0, len(values), size=(size, len(values)))
        draws[start:start + size] = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), int(len(values))


def bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, seed: int, count: int) -> tuple[float, float, float, int]:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator)
    numerator, denominator = numerator[valid], denominator[valid]
    if not len(numerator):
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    draws = np.empty(count, dtype=float)
    for start in range(0, count, 64):
        size = min(64, count - start)
        idx = rng.integers(0, len(numerator), size=(size, len(numerator)))
        den = denominator[idx].sum(axis=1)
        draws[start:start + size] = np.divide(
            numerator[idx].sum(axis=1), den,
            out=np.full(size, np.nan), where=den > 0,
        )
    draws = draws[np.isfinite(draws)]
    estimate = float(numerator.sum() / denominator.sum()) if denominator.sum() else np.nan
    return estimate, float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), int(len(numerator))


def bootstrap_rows(values: np.ndarray, setting: str, label: str, seed_base: int, count: int, unit: str) -> dict:
    seed = seed_base + sum(map(ord, setting + label))
    est, lo, hi, n = bootstrap_mean(values, seed, count)
    return {
        "status": "local_rerun_traceable",
        "dataset_model": setting,
        "quantity": label,
        "estimate": est,
        "ci_low": lo,
        "ci_high": hi,
        "bootstrap_count": count,
        "bootstrap_seed": seed,
        "bootstrap_interval": "percentile_2.5_97.5",
        "bootstrap_unit": unit,
        "n_valid": n,
    }


def load_batches(setting_dir: Path) -> pd.DataFrame:
    batch_paths = sorted((setting_dir / "traceable_batches").glob("*.npz"))
    if not batch_paths:
        raise FileNotFoundError(f"no traceable batches in {setting_dir}")
    rows: list[dict] = []
    for path in batch_paths:
        x = np.load(path, allow_pickle=False)
        required = {
            "sample_index", "image_id", "response_raw", "response_norm", "bbox",
            "selection_target", "selection_all", "selection_mean_foil", "selection_max_foil",
            "feasible_mask", "cci_region", "wf_region", "mean_region", "max_0_1_region",
            "target_id", "target_name", "annotated_category_ids", "category_ids",
        }
        missing = required - set(x.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        raw = np.asarray(x["response_raw"], dtype=float)
        norm = np.asarray(x["response_norm"], dtype=float)
        sel_target = np.asarray(x["selection_target"], dtype=float)
        sel_all = np.asarray(x["selection_all"], dtype=float)
        bbox = np.asarray(x["bbox"], dtype=float)
        feasible = np.asarray(x["feasible_mask"], dtype=bool)
        cci = np.asarray(x["cci_region"], dtype=int)
        strategy_regions = {
            "cci": cci,
            "wf": np.asarray(x["wf_region"], dtype=int),
            "mean": np.asarray(x["mean_region"], dtype=int),
            "max_0_1": np.asarray(x["max_0_1_region"], dtype=int),
        }
        category_ids = [canonical_id(v) for v in x["category_ids"].tolist()]
        cat_index = {v: i for i, v in enumerate(category_ids)}
        target_ids = [canonical_id(v) for v in x["target_id"].tolist()]
        target_pos = np.asarray([cat_index[v] for v in target_ids], dtype=int)
        b = len(target_ids)
        if raw.shape[0] != b or raw.shape[1] != 8 or norm.shape != raw.shape:
            raise ValueError(f"{path}: response shape is not (batch,8,class): {raw.shape}/{norm.shape}")
        if sel_target.shape != (b, 8) or sel_all.shape != (b, 8, len(category_ids)):
            raise ValueError(f"{path}: selection shape mismatch")
        if bbox.shape != (b, 8) or feasible.shape != (b, 8):
            raise ValueError(f"{path}: candidate geometry shape mismatch")
        target_resp_norm = norm[np.arange(b)[:, None], np.arange(8)[None, :], target_pos[:, None]]
        target_resp_raw = raw[np.arange(b)[:, None], np.arange(8)[None, :], target_pos[:, None]]
        foil_mask = np.ones((b, len(category_ids)), dtype=bool)
        foil_mask[np.arange(b), target_pos] = False
        foil_values = np.where(foil_mask[:, None, :], norm, -np.inf)
        margin = target_resp_norm - foil_values.max(axis=2)
        pmean = target_resp_norm - np.where(foil_mask[:, None, :], norm, 0.0).sum(axis=2) / (len(category_ids) - 1)
        selection_target_at_cci = sel_target[np.arange(b), cci]
        selection_foil = {
            "max": np.asarray(x["selection_max_foil"], dtype=float),
            "mean": np.asarray(x["selection_mean_foil"], dtype=float),
        }
        for i in range(b):
            annotated = {canonical_id(v) for v in json.loads(str(x["annotated_category_ids"][i]))}
            target = target_ids[i]
            non_target = [c for c in category_ids if c != target]
            absent = [c for c in non_target if c not in annotated]
            absent_pos = np.asarray([cat_index[c] for c in absent], dtype=int)
            cci_target = target_resp_norm[i, cci[i]]
            full_vals = norm[i, cci[i], [cat_index[c] for c in non_target]]
            m = int(np.sum(full_vals > cci_target))
            q = exact_q(len(non_target), len(absent), m)
            absent_failure = False
            absent_margin = np.nan
            absent_worst = np.nan
            if len(absent_pos):
                absent_worst = float(norm[i, cci[i], absent_pos].max())
                absent_margin = float(cci_target - absent_worst)
                absent_failure = absent_margin < 0
            row = {
                "status": "local_rerun_traceable",
                "sample_index": int(x["sample_index"][i]),
                "image_id": canonical_id(x["image_id"][i]),
                "target_id": target,
                "target_name": str(x["target_name"][i]),
                "annotated_category_ids": json.dumps(sorted(annotated), ensure_ascii=False),
                "category_ids": json.dumps(category_ids, ensure_ascii=False),
                "category_count": len(category_ids),
                "cci_region": int(cci[i]),
                "wf_region": int(strategy_regions["wf"][i]),
                "mean_region": int(strategy_regions["mean"][i]),
                "max_0_1_region": int(strategy_regions["max_0_1"][i]),
                "feasible_set_size": int(feasible[i].sum()),
                "feasible_mask": json_array(feasible[i].astype(int)),
                "selection_target_at_cci": float(selection_target_at_cci[i]),
                "full_non_target_foil_count": len(non_target),
                "annotation_absent_foil_count": len(absent),
                "full_outranking_foil_count": m,
                "exact_matched_random_failure_probability": q,
                "full_failure": int(m > 0),
                "annotation_absent_failure": int(absent_failure),
                "annotation_absent_margin_norm": absent_margin,
                "annotation_absent_worst_response_norm": absent_worst,
            }
            for method, region in strategy_regions.items():
                r = int(region[i])
                row[f"{method}_bbox_precision"] = float(bbox[i, r])
                row[f"{method}_target_drop_raw"] = float(target_resp_raw[i, r])
                row[f"{method}_target_drop_norm"] = float(target_resp_norm[i, r])
                row[f"{method}_margin_norm"] = float(margin[i, r])
                row[f"{method}_pmean_norm"] = float(pmean[i, r])
            row["A"] = int(row["cci_bbox_precision"] >= .5 and row["cci_pmean_norm"] > 0)
            row["Aplus"] = int(row["A"] and row["cci_target_drop_norm"] > 0)
            row["B"] = int(row["A"] and row["cci_margin_norm"] < 0)
            passing = margin[i] >= 0
            feasible_passing = passing & feasible[i]
            row["C0"] = int(row["B"] and not passing.any())
            row["C1"] = int(row["B"] and passing.any() and not feasible_passing.any())
            row["C2"] = int(row["B"] and feasible_passing.any() and not passing[int(strategy_regions["wf"][i])])
            row["repair"] = int(row["B"] and passing[int(strategy_regions["wf"][i])])
            wf_r = int(strategy_regions["wf"][i])
            row["joint_J"] = int(row["repair"] and bbox[i, wf_r] >= .5 and pmean[i, wf_r] > 0 and target_resp_norm[i, wf_r] > 0)
            # Keep all candidate-level values in the local traceable CSV.  The
            # batch NPZ remains the byte-level source for masks and responses.
            row["candidate_selection_target"] = json_array(sel_target[i])
            row["candidate_selection_mean_foil"] = json_array(selection_foil["mean"][i])
            row["candidate_selection_max_foil"] = json_array(selection_foil["max"][i])
            row["candidate_eval_margin_norm"] = json_array(margin[i])
            row["candidate_eval_pmean_norm"] = json_array(pmean[i])
            row["candidate_eval_target_drop_raw"] = json_array(target_resp_raw[i])
            row["candidate_eval_target_drop_norm"] = json_array(target_resp_norm[i])
            row["candidate_bbox_precision"] = json_array(bbox[i])
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values("sample_index").reset_index(drop=True)
    expected = np.arange(len(frame), dtype=int)
    if not np.array_equal(frame.sample_index.to_numpy(dtype=int), expected):
        raise ValueError(f"{setting_dir}: sample indices are not complete consecutive 0..n-1")
    if frame.image_id.duplicated().any():
        raise ValueError(f"{setting_dir}: duplicate image IDs")
    return frame


def strategy_summary(frame: pd.DataFrame, setting: str, count: int, seed: int) -> list[dict]:
    rows = []
    subsets = {"all": np.ones(len(frame), dtype=bool), "A": frame.A.to_numpy(bool), "Aplus": frame.Aplus.to_numpy(bool), "B": frame.B.to_numpy(bool), "repair_r": frame.repair.to_numpy(bool), "joint_j": frame.joint_J.to_numpy(bool)}
    for method in METHODS:
        for subset, mask in subsets.items():
            if not mask.any():
                continue
            for metric in METRICS:
                values = frame.loc[mask, f"{method}_{metric}"].to_numpy(float)
                rows.append({
                    "status": "local_rerun_traceable", "dataset_model": setting,
                    "method": method, "subset": subset, "metric": metric,
                    "n": int(mask.sum()), "mean": float(np.mean(values)),
                    "ci_low": np.nan, "ci_high": np.nan,
                    "bootstrap_count": 0, "bootstrap_seed": "",
                    "bootstrap_interval": "not_requested_for_descriptive_strategy_summary",
                    "bootstrap_unit": "",
                })
    return rows


def paired_analysis(frame: pd.DataFrame, setting: str, count: int, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    wide = frame[["sample_index", "image_id"]].copy()
    summary = []
    for left, right in (("wf", "mean"), ("wf", "max_0_1")):
        comparison = f"{left}_minus_{right}"
        for metric in METRICS:
            field = f"{comparison}_{metric}"
            values = frame[f"{left}_{metric}"].to_numpy(float) - frame[f"{right}_{metric}"].to_numpy(float)
            wide[field] = values
            s = seed + sum(map(ord, setting + comparison + metric))
            est, lo, hi, n = bootstrap_mean(values, s, count)
            summary.append({
                "status": "local_rerun_traceable", "dataset_model": setting,
                "comparison": comparison, "metric": metric, "estimate": est,
                "ci_low": lo, "ci_high": hi, "bootstrap_count": count,
                "bootstrap_seed": s, "bootstrap_interval": "percentile_2.5_97.5",
                "bootstrap_unit": "same image; direct paired difference",
                "difference_definition": f"{left}_{metric} - {right}_{metric}", "n_valid": n,
                "raw_or_normalized": "raw" if metric == "target_drop_raw" else "normalized" if metric in {"margin_norm", "target_drop_norm"} else "raw geometry",
            })
    return wide, summary


def screening_analysis(frame: pd.DataFrame, setting: str, count: int, seed: int) -> list[dict]:
    rows = []
    for method in METHODS:
        screen = (frame[f"{method}_bbox_precision"] >= .5) & (frame[f"{method}_pmean_norm"] > 0)
        failure = frame[f"{method}_margin_norm"] < 0
        joint = screen & failure
        base = seed + sum(map(ord, setting + method + "screen"))
        sp = bootstrap_mean(screen.to_numpy(float), base + 1, count)
        fr = bootstrap_mean(failure.to_numpy(float), base + 2, count)
        jr = bootstrap_mean(joint.to_numpy(float), base + 3, count)
        cond = bootstrap_ratio(joint.to_numpy(float), screen.to_numpy(float), base + 4, count)
        rows.append({
            "status": "local_rerun_traceable", "dataset_model": setting, "method": method,
            "n_images": len(frame), "screen_definition": "bbox_precision >= 0.5 AND normalized foil-mean margin > 0",
            "failure_definition": "normalized aggregate worst-foil margin < 0",
            "screen_pass_count": int(screen.sum()), "failure_count": int(failure.sum()), "joint_failure_count": int(joint.sum()),
            "screen_pass_rate": sp[0], "screen_pass_ci_low": sp[1], "screen_pass_ci_high": sp[2],
            "failure_rate": fr[0], "failure_ci_low": fr[1], "failure_ci_high": fr[2],
            "joint_failure_rate": jr[0], "joint_failure_ci_low": jr[1], "joint_failure_ci_high": jr[2],
            "failure_given_screen_pass": cond[0], "conditional_ci_low": cond[1], "conditional_ci_high": cond[2],
            "bootstrap_count": count, "bootstrap_base_seed": base,
            "screen_pass_bootstrap_seed": base + 1, "failure_bootstrap_seed": base + 2,
            "joint_failure_bootstrap_seed": base + 3, "conditional_bootstrap_seed": base + 4,
            "bootstrap_interval": "percentile_2.5_97.5",
            "bootstrap_unit": "image; conditional numerator and denominator jointly resampled",
            "screen_valid_images": sp[3], "failure_valid_images": fr[3],
            "joint_valid_images": jr[3], "conditional_ratio_valid_images": cond[3],
        })
    return rows


def restricted_foil_analysis(frame: pd.DataFrame, setting: str, count: int, seed: int, mc_seed_count: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    rows, controls, plus, mc_rows = [], [], [], []
    for subset_name, mask in (("all", np.ones(len(frame), dtype=bool)), ("A", frame.A.to_numpy(bool)), ("Aplus", frame.Aplus.to_numpy(bool))):
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        full = frame.full_failure.to_numpy(float)[idx]
        absent = frame.annotation_absent_failure.to_numpy(float)[idx]
        q = frame.exact_matched_random_failure_probability.to_numpy(float)[idx]
        base = seed + sum(map(ord, setting + subset_name + "restricted"))
        f = bootstrap_mean(full, base, count)
        a = bootstrap_mean(absent, base + 1, count)
        d = bootstrap_mean(absent - full, base + 2, count)
        r = bootstrap_mean(q, base + 3, count)
        dr = bootstrap_mean(absent - q, base + 4, count)
        rows.append({
            "status": "local_rerun_traceable", "dataset_model": setting, "subset": subset_name,
            "n": len(idx), "full_failure_count": int(full.sum()), "annotation_absent_failure_count": int(absent.sum()),
            "Pr_F_full": f[0], "Pr_F_full_CI_lo": f[1], "Pr_F_full_CI_hi": f[2],
            "Pr_F_absent": a[0], "Pr_F_absent_CI_lo": a[1], "Pr_F_absent_CI_hi": a[2],
            "absent_minus_full": d[0], "absent_minus_full_CI_lo": d[1], "absent_minus_full_CI_hi": d[2],
            "bootstrap_count": count, "bootstrap_interval": "percentile_2.5/97.5", "bootstrap_unit": "image",
            "bootstrap_seed_full": base, "bootstrap_seed_absent": base + 1, "bootstrap_seed_paired_absent_minus_full": base + 2,
        })
        controls.append({
            "status": "local_rerun_traceable", "dataset_model": setting, "subset": subset_name, "n": len(idx),
            "full_non_target_foil_n_mean": float(frame.full_non_target_foil_count.to_numpy(float)[idx].mean()),
            "annotation_absent_foil_n_mean": float(frame.annotation_absent_foil_count.to_numpy(float)[idx].mean()),
            "full_outranking_n_mean": float(frame.full_outranking_foil_count.to_numpy(float)[idx].mean()),
            "exact_matched_random_failure_rate": r[0], "exact_matched_random_CI_lo": r[1], "exact_matched_random_CI_hi": r[2],
            "absent_minus_matched_random": dr[0], "absent_minus_matched_random_CI_lo": dr[1], "absent_minus_matched_random_CI_hi": dr[2],
            "bootstrap_count": count, "bootstrap_interval": "percentile_2.5/97.5", "bootstrap_unit": "image",
            "bootstrap_seed_random": base + 3, "bootstrap_seed_paired_absent_minus_random": base + 4,
        })
        if subset_name == "Aplus":
            plus.append({"status": "local_rerun_traceable", "dataset_model": setting, "n_Aplus": len(idx), "Aplus_full_failure_count": int(full.sum()), "Aplus_absent_failure_count": int(absent.sum()), "Aplus_Pr_F_full": f[0], "Aplus_Pr_F_full_CI_lo": f[1], "Aplus_Pr_F_full_CI_hi": f[2], "Aplus_Pr_F_absent": a[0], "Aplus_Pr_F_absent_CI_lo": a[1], "Aplus_Pr_F_absent_CI_hi": a[2], "Aplus_exact_matched_random_rate": r[0], "Aplus_exact_matched_random_CI_lo": r[1], "Aplus_exact_matched_random_CI_hi": r[2]})
    return rows, controls, plus, mc_rows


def switch_analysis(frame: pd.DataFrame, setting: str, count: int, seed: int) -> tuple[pd.DataFrame, dict]:
    switched = frame.wf_region.to_numpy(int) != frame.cci_region.to_numpy(int)
    subset = frame.loc[switched]
    rows = []
    for metric in METRICS:
        values = subset[f"wf_{metric}"].to_numpy(float) - subset[f"cci_{metric}"].to_numpy(float)
        s = seed + sum(map(ord, setting + "switch" + metric))
        est, lo, hi, n = bootstrap_mean(values, s, count)
        rows.append({
            "status": "local_rerun_traceable", "dataset_model": setting,
            "subset": "cci_to_wf_switched", "metric": metric, "n_images": len(subset),
            "estimate": est, "ci_low": lo, "ci_high": hi,
            "p10": float(np.quantile(values, .10)) if len(values) else np.nan,
            "p25": float(np.quantile(values, .25)) if len(values) else np.nan,
            "p50": float(np.quantile(values, .50)) if len(values) else np.nan,
            "p75": float(np.quantile(values, .75)) if len(values) else np.nan,
            "p90": float(np.quantile(values, .90)) if len(values) else np.nan,
            "bootstrap_count": count, "bootstrap_seed": s,
            "bootstrap_interval": "percentile_2.5_97.5", "bootstrap_unit": "switched image",
            "difference_definition": f"wf_{metric} - cci_{metric}", "n_valid": n,
        })
    margin_delta = frame.wf_margin_norm.to_numpy(float) - frame.cci_margin_norm.to_numpy(float)
    sign_repair = (frame.cci_margin_norm.to_numpy(float) < 0) & (frame.wf_margin_norm.to_numpy(float) >= 0)
    summary = {
        "status": "local_rerun_traceable", "dataset_model": setting, "n_images": len(frame),
        "switch_count": int(switched.sum()), "switch_rate": float(switched.mean()) if len(frame) else np.nan,
        "delta_margin_improvement_count": int((switched & (margin_delta > 0)).sum()),
        "delta_margin_improvement_fraction_among_switched": float((margin_delta[switched] > 0).mean()) if switched.any() else np.nan,
        "sign_repair_count": int((switched & sign_repair).sum()),
        "sign_repair_fraction_among_switched": float(sign_repair[switched].mean()) if switched.any() else np.nan,
        "joint_J_count": int((switched & frame.joint_J.to_numpy(bool)).sum()),
        "margin_improvement_definition": "WF normalized aggregate worst-foil margin > CCI normalized aggregate worst-foil margin",
        "sign_repair_definition": "CCI margin < 0 and WF margin >= 0",
        "joint_J_definition": "B and WF sign pass, WF bbox >= .5, WF normalized foil-mean margin > 0, WF target drop norm > 0",
    }
    return pd.DataFrame(rows), summary


def budget_analysis(frame: pd.DataFrame, setting: str) -> pd.DataFrame:
    n = len(frame)
    sel = np.asarray([json.loads(v) for v in frame.candidate_selection_target], dtype=float)
    eval_margin = np.asarray([json.loads(v) for v in frame.candidate_eval_margin_norm], dtype=float)
    eval_raw = np.asarray([json.loads(v) for v in frame.candidate_eval_target_drop_raw], dtype=float)
    bbox = np.asarray([json.loads(v) for v in frame.candidate_bbox_precision], dtype=float)
    cci = frame.cci_region.to_numpy(int)
    cci_sel = sel[np.arange(n), cci]
    cci_raw = eval_raw[np.arange(n), cci]
    cci_bbox = bbox[np.arange(n), cci]
    B = frame.B.to_numpy(bool)
    rows = []
    for eps in EPSILONS:
        feasible = sel >= (cci_sel[:, None] - eps)
        # The per-image export contains the candidate-level selection max-foil
        # vector, so WF can be recomputed without guessing from the display
        # name or from an aggregate feasible-set size.
        max_foil = np.asarray([json.loads(v) for v in frame.candidate_selection_max_foil], dtype=float)
        wf = np.where(feasible, sel - max_foil, -np.inf).argmax(axis=1)
        passing = eval_margin >= 0
        rows.append({
            "status": "local_rerun_traceable", "dataset_model": setting, "epsilon": eps,
            "multi_feasible_count": int((feasible.sum(axis=1) > 1).sum()),
            "multi_feasible_pct_all_images": float((feasible.sum(axis=1) > 1).mean()),
            "unrestricted_oracle_count_B": int((passing[B].any(axis=1)).sum()) if B.any() else 0,
            "unrestricted_oracle_pct_B": float(passing[B].any(axis=1).mean()) if B.any() else np.nan,
            "epsilon_oracle_count_B": int((passing[B] & feasible[B]).any(axis=1).sum()) if B.any() else 0,
            "epsilon_oracle_pct_B": float((passing[B] & feasible[B]).any(axis=1).mean()) if B.any() else np.nan,
            "epsilon_wf_repair_count_B": int(passing[np.arange(n), wf][B].sum()),
            "epsilon_wf_repair_pct_B": float(passing[np.arange(n), wf][B].mean()) if B.any() else np.nan,
            "all_image_raw_target_change": float((eval_raw[np.arange(n), wf] - cci_raw).mean()),
            "all_image_bbox_change_pp": float(100 * (bbox[np.arange(n), wf] - cci_bbox).mean()),
            "definition": "candidate feasibility uses selection target drop >= CCI target drop - epsilon; WF maximizes selection target drop - selection max foil over that set",
        })
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_old(new_frames: dict[str, pd.DataFrame], old_root: Path | None, old_results: Path | None, out_dir: Path) -> None:
    if old_root is None:
        return
    rows = []
    for setting, frame in new_frames.items():
        old_path = old_root / setting / "per_image.csv"
        if not old_path.exists():
            rows.append({"dataset_model": setting, "status": "missing_old_per_image", "path": str(old_path)})
            continue
        old = pd.read_csv(old_path)
        old["image_id_canonical"] = old.image_id.map(canonical_id)
        new = frame.copy()
        old = old.sort_values("sample_index").reset_index(drop=True)
        new = new.sort_values("sample_index").reset_index(drop=True)
        fields = ["cci_region", "wf_region", "mean_region", "max_0_1_region", "feasible_set_size", "cci_bbox_precision", "cci_margin_norm", "cci_pmean_norm", "cci_target_drop_raw", "cci_target_drop_norm"]
        for field in fields:
            if field not in old or field not in new:
                continue
            if field.endswith("region") or field == "feasible_set_size":
                changed = int(np.sum(old[field].to_numpy() != new[field].to_numpy()))
                maxdiff = np.nan
            else:
                diff = np.abs(old[field].to_numpy(float) - new[field].to_numpy(float))
                changed = int(np.sum(diff > 1e-5))
                maxdiff = float(np.nanmax(diff))
            rows.append({"dataset_model": setting, "field": field, "status": "aligned_by_sample_index", "old_n": len(old), "new_n": len(new), "changed_count": changed, "new_minus_old_max_abs": maxdiff})
        for label, col in (("A", None), ("Aplus", None), ("B", None), ("repair", None), ("joint_J", None)):
            if col:
                continue
            if label == "A":
                old_val = ((old.cci_bbox_precision >= .5) & (old.cci_pmean_norm > 0)).sum()
            elif label == "Aplus":
                old_val = ((old.cci_bbox_precision >= .5) & (old.cci_pmean_norm > 0) & (old.cci_target_drop_norm > 0)).sum()
            elif label == "B":
                old_val = ((old.cci_bbox_precision >= .5) & (old.cci_pmean_norm > 0) & (old.cci_margin_norm < 0)).sum()
            elif label == "repair":
                old_val = (((old.cci_bbox_precision >= .5) & (old.cci_pmean_norm > 0) & (old.cci_margin_norm < 0)) & (old.wf_margin_norm >= 0)).sum()
            else:
                old_val = np.nan
            new_val = int(frame[label].sum())
            rows.append({"dataset_model": setting, "field": label, "status": "summary_old_vs_new", "old_value": old_val, "new_value": new_val, "difference_new_minus_old": new_val - old_val if np.isfinite(old_val) else np.nan})
    pd.DataFrame(rows).to_csv(out_dir / "new_vs_old_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    args.trace_root = args.trace_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "per_image").mkdir(exist_ok=True)
    all_frames: dict[str, pd.DataFrame] = {}
    paired_rows, paired_summary, screening_rows, restricted_rows, controls, plus_rows, mc_rows = [], [], [], [], [], [], []
    switch_rows, switch_summaries, strategy_rows, budget_rows, completion = [], [], [], [], []
    input_rows = []
    for setting, _, _ in SETTINGS:
        setting_dir = args.trace_root / setting
        frame = load_batches(setting_dir)
        all_frames[setting] = frame
        frame.to_csv(args.out_dir / "per_image" / f"{setting}.csv", index=False)
        paired, psummary = paired_analysis(frame, setting, args.bootstrap_count, args.bootstrap_seed)
        paired.to_csv(args.out_dir / "per_image" / f"{setting}_paired_wide.csv", index=False)
        paired_long = []
        for row in paired.itertuples(index=False):
            for right in ("mean", "max_0_1"):
                for metric in METRICS:
                    col = f"wf_minus_{right}_{metric}"
                    paired_long.append({"status": "local_rerun_traceable", "dataset_model": setting, "sample_index": int(row.sample_index), "image_id": row.image_id, "comparison": f"wf_minus_{right}", "metric": metric, "difference": float(getattr(row, col))})
        pd.DataFrame(paired_long).to_csv(args.out_dir / "per_image" / f"{setting}_paired_long.csv", index=False)
        paired_summary.extend(psummary)
        screening_rows.extend(screening_analysis(frame, setting, args.bootstrap_count, args.bootstrap_seed))
        r, c, p, m = restricted_foil_analysis(frame, setting, args.bootstrap_count, args.bootstrap_seed, args.mc_seed_count)
        restricted_rows.extend(r); controls.extend(c); plus_rows.extend(p); mc_rows.extend(m)
        sw, sws = switch_analysis(frame, setting, args.bootstrap_count, args.bootstrap_seed)
        switch_rows.append(sw); switch_summaries.append(sws)
        strategy_rows.extend(strategy_summary(frame, setting, args.bootstrap_count, args.bootstrap_seed))
        budget_rows.append(budget_analysis(frame, setting))
        completion.append({"status": "completed", "dataset_model": setting, "n_images": len(frame), "A": int(frame.A.sum()), "Aplus": int(frame.Aplus.sum()), "B": int(frame.B.sum()), "C0": int(frame.C0.sum()), "C1": int(frame.C1.sum()), "C2": int(frame.C2.sum()), "r": int(frame.repair.sum()), "j": int(frame.joint_J.sum()), "switch_count": int((frame.wf_region != frame.cci_region).sum()), "source_dir": str(setting_dir)})
        for path in sorted(setting_dir.rglob("*")):
            if path.is_file():
                input_rows.append({"dataset_model": setting, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    pd.concat(switch_rows, ignore_index=True).to_csv(args.out_dir / "switch_subset_changes_ci.csv", index=False)
    pd.DataFrame(paired_summary).to_csv(args.out_dir / "paired_difference_ci_10000.csv", index=False)
    pd.DataFrame(screening_rows).to_csv(args.out_dir / "screening_failure_rates.csv", index=False)
    pd.DataFrame(restricted_rows).to_csv(args.out_dir / "restricted_foil_failure_rates.csv", index=False)
    pd.DataFrame(controls).to_csv(args.out_dir / "matched_cardinality_control.csv", index=False)
    pd.DataFrame(plus_rows).to_csv(args.out_dir / "positive_target_subset.csv", index=False)
    pd.DataFrame([{
        "status": "not_run_in_compact_analyzer", "dataset_model": setting,
        "mc_seed_count_requested": args.mc_seed_count,
        "note": "Exact matched-cardinality q is the primary result. A seeded Monte Carlo check was not claimed because this compact export does not retain each absent class position; the traceable batch NPZs remain the source for such a check.",
    } for setting in all_frames]).to_csv(args.out_dir / "matched_cardinality_sanity_check.csv", index=False)
    pd.DataFrame(switch_summaries).to_csv(args.out_dir / "switch_margin_improvement_vs_sign_repair.csv", index=False)
    pd.DataFrame(strategy_rows).to_csv(args.out_dir / "strategy_summary.csv", index=False)
    pd.concat(budget_rows, ignore_index=True).to_csv(args.out_dir / "budget_scan.csv", index=False)
    pd.DataFrame(completion).to_csv(args.out_dir / "completed_and_missing.csv", index=False)
    compare_old(all_frames, args.old_runs_root.expanduser().resolve() if args.old_runs_root else None, args.old_results_root.expanduser().resolve() if args.old_results_root else None, args.out_dir)
    (args.out_dir / "input_hashes.csv").write_text(pd.DataFrame(input_rows).to_csv(index=False))
    git_bin = "/Users/von/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git"
    git_head = subprocess.check_output([git_bin, "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True).strip()
    metadata = {
        "status": "complete_traceable_analysis", "settings": [s[0] for s in SETTINGS],
        "bootstrap_count": args.bootstrap_count, "bootstrap_seed_base": args.bootstrap_seed,
        "bootstrap_interval": "percentile_2.5_97.5", "bootstrap_unit": "image",
        "epsilon": 0.02, "mc_seed_count": args.mc_seed_count,
        "authority": "same-generation traceable batch NPZs; old local-rerun files used only for comparison",
        "python": platform.python_version(), "platform": platform.platform(),
        "git_head": git_head,
    }
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "complete_traceable_analysis", "completion": completion}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
