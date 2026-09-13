#!/usr/bin/env python3
"""Independent verification of the four-setting local rerun.

This script deliberately reads the per-image outputs directly instead of
importing the original analysis module.  It recomputes paired image-level
statistics, screening ratios, feasible-set distributions, switch-subset
statistics, strategy choices, and the available frozen/local comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SETTINGS = (
    "coco_openai_b16",
    "coco_openai_b32",
    "voc2007_openai_b16",
    "voc2007_openai_b32",
)
METRICS = ("margin_norm", "target_drop_raw", "target_drop_norm", "bbox_precision")
METHODS = ("cci", "wf", "mean", "max_0_1")
FROZEN_SWITCH_COUNTS = {
    "coco_openai_b16": 663,
    "coco_openai_b32": 613,
    "voc2007_openai_b16": 50,
    "voc2007_openai_b32": 38,
}
BASELINE_COMMIT = "59fc00e3948015379c8dcc563283a5bdda7f1226"
EPSILON = 0.02
CLUSTERS = 8
BOOTSTRAP_COUNT = 10000
BOOTSTRAP_BASE_SEED = 1701
INTERVAL = "percentile_2.5_97.5"
TOLERANCE = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_COUNT)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_BASE_SEED)
    return parser.parse_args()


def seed_for(seed: int, *parts: str) -> int:
    return int(seed + sum(map(ord, "".join(parts))))


def bootstrap_mean(values: np.ndarray, seed: int, count: int) -> tuple[float, float, float, int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, math.nan, 0
    rng = np.random.default_rng(seed)
    draws = np.empty(count, dtype=np.float64)
    for start in range(0, count, 64):
        size = min(64, count - start)
        idx = rng.integers(0, len(values), size=(size, len(values)))
        draws[start:start + size] = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        int(len(values)),
    )


def bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    seed: int,
    count: int,
) -> tuple[float, float, float, int]:
    """Bootstrap a ratio by resampling numerator and denominator together."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator)
    numerator, denominator = numerator[valid], denominator[valid]
    if not len(numerator):
        return math.nan, math.nan, math.nan, 0
    rng = np.random.default_rng(seed)
    draws = np.empty(count, dtype=np.float64)
    for start in range(0, count, 64):
        size = min(64, count - start)
        idx = rng.integers(0, len(numerator), size=(size, len(numerator)))
        den = denominator[idx].sum(axis=1)
        draws[start:start + size] = np.divide(
            numerator[idx].sum(axis=1),
            den,
            out=np.full(size, np.nan),
            where=den > 0,
        )
    draws = draws[np.isfinite(draws)]
    point = float(numerator.sum() / denominator.sum()) if denominator.sum() else math.nan
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), int(len(numerator))


def read_frame(project: Path, setting: str) -> pd.DataFrame:
    path = project / "rerun_workspace" / "runs" / setting / "per_image.csv"
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty per-image file: {path}")
    if frame["image_id"].duplicated().any():
        raise ValueError(f"duplicate image_id in {path}")
    return frame


def pairwise_tables(
    frame: pd.DataFrame,
    setting: str,
    seed: int,
    count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict] = []
    summary_rows: list[dict] = []
    for left, right in (("wf", "mean"), ("wf", "max_0_1")):
        for metric in METRICS:
            field = f"{left}_{metric}"
            right_field = f"{right}_{metric}"
            values = frame[field].to_numpy(float) - frame[right_field].to_numpy(float)
            comparison = f"{left}_minus_{right}"
            actual_seed = seed_for(seed, setting, left, right, metric)
            estimate, low, high, n_valid = bootstrap_mean(values, actual_seed, count)
            summary_rows.append({
                "status": "local_rerun_verified",
                "dataset_model": setting,
                "comparison": comparison,
                "metric": metric,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "bootstrap_count": count,
                "bootstrap_seed": actual_seed,
                "bootstrap_interval": INTERVAL,
                "bootstrap_unit": "image",
                "difference_definition": f"{left}_{metric} - {right}_{metric}; same image and candidate set",
                "n_valid": n_valid,
                "raw_or_normalized": "raw" if metric == "target_drop_raw" else "normalized" if metric in ("margin_norm", "target_drop_norm") else "unit_interval",
            })
            for sample_index, image_id, value in zip(frame["sample_index"], frame["image_id"], values):
                long_rows.append({
                    "status": "local_rerun_verified",
                    "dataset_model": setting,
                    "sample_index": int(sample_index),
                    "image_id": int(image_id),
                    "comparison": comparison,
                    "metric": metric,
                    "difference": float(value),
                    "bootstrap_seed": actual_seed,
                })
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def screening_table(frame: pd.DataFrame, setting: str, method: str, seed: int, count: int) -> dict:
    screen = (frame[f"{method}_bbox_precision"] >= 0.5) & (frame[f"{method}_pmean_norm"] > 0)
    failure = frame[f"{method}_margin_norm"] < 0
    joint = screen & failure
    base_seed = seed_for(seed, setting, method, "screen")

    def rate(mask: pd.Series, actual_seed: int) -> tuple[float, float, float, int]:
        return bootstrap_mean(mask.astype(float).to_numpy(), actual_seed, count)

    screen_seed = base_seed + int(screen.sum())
    failure_seed = base_seed + int(failure.sum())
    joint_seed = base_seed + int(joint.sum())
    conditional_seed = base_seed + 23
    sp, sp_low, sp_high, sp_valid = rate(screen, screen_seed)
    fr, fr_low, fr_high, fr_valid = rate(failure, failure_seed)
    jr, jr_low, jr_high, jr_valid = rate(joint, joint_seed)
    cond, cond_low, cond_high, ratio_valid = bootstrap_ratio(
        joint.astype(float).to_numpy(), screen.astype(float).to_numpy(), conditional_seed, count
    )
    return {
        "status": "local_rerun_verified",
        "dataset_model": setting,
        "method": method,
        "n_images": len(frame),
        "screen_definition": "bbox_precision >= 0.5 AND normalized aggregate foil-mean margin > 0",
        "failure_definition": "normalized aggregate worst-foil margin < 0",
        "screen_pass_count": int(screen.sum()),
        "failure_count": int(failure.sum()),
        "joint_failure_count": int(joint.sum()),
        "screen_pass_rate": sp,
        "screen_pass_ci_low": sp_low,
        "screen_pass_ci_high": sp_high,
        "failure_rate": fr,
        "failure_ci_low": fr_low,
        "failure_ci_high": fr_high,
        "joint_failure_rate": jr,
        "joint_failure_ci_low": jr_low,
        "joint_failure_ci_high": jr_high,
        "failure_given_screen_pass": cond,
        "conditional_ci_low": cond_low,
        "conditional_ci_high": cond_high,
        "bootstrap_count": count,
        "bootstrap_base_seed": base_seed,
        "screen_pass_bootstrap_seed": screen_seed,
        "failure_bootstrap_seed": failure_seed,
        "joint_failure_bootstrap_seed": joint_seed,
        "conditional_bootstrap_seed": conditional_seed,
        "bootstrap_interval": INTERVAL,
        "bootstrap_unit": "image; numerator and denominator jointly resampled",
        "screen_valid_images": sp_valid,
        "failure_valid_images": fr_valid,
        "joint_valid_images": jr_valid,
        "conditional_ratio_valid_images": ratio_valid,
    }


def feasible_table(frame: pd.DataFrame, setting: str) -> pd.DataFrame:
    counts = frame["feasible_set_size"].value_counts().reindex(range(1, CLUSTERS + 1), fill_value=0)
    return pd.DataFrame([
        {
            "status": "local_rerun_verified",
            "dataset_model": setting,
            "feasible_set_size": int(k),
            "image_count": int(v),
            "image_fraction": float(v / len(frame)),
        }
        for k, v in counts.items()
    ])


def switch_tables(frame: pd.DataFrame, setting: str, seed: int, count: int) -> tuple[pd.DataFrame, dict]:
    switched = frame["wf_region"].to_numpy(int) != frame["cci_region"].to_numpy(int)
    subset = frame.loc[switched]
    rows: list[dict] = []
    for metric in METRICS:
        values = subset[f"wf_{metric}"].to_numpy(float) - subset[f"cci_{metric}"].to_numpy(float)
        actual_seed = seed_for(seed, setting, "switch", metric)
        estimate, low, high, n_valid = bootstrap_mean(values, actual_seed, count)
        quantiles = {f"p{q}": float(np.quantile(values, q / 100)) for q in (10, 25, 50, 75, 90)}
        rows.append({
            "status": "local_rerun_verified",
            "dataset_model": setting,
            "subset": "cci_to_wf_switched",
            "metric": metric,
            "n_images": len(subset),
            "estimate": estimate,
            "ci_low": low,
            "ci_high": high,
            **quantiles,
            "bootstrap_count": count,
            "bootstrap_seed": actual_seed,
            "bootstrap_interval": INTERVAL,
            "bootstrap_unit": "switched image",
            "difference_definition": f"wf_{metric} - cci_{metric}; evaluated only where wf_region != cci_region",
            "n_valid": n_valid,
            "raw_or_normalized": "raw" if metric == "target_drop_raw" else "normalized" if metric in ("margin_norm", "target_drop_norm") else "unit_interval",
        })
    wf_margin = frame["wf_margin_norm"].to_numpy(float)
    cci_margin = frame["cci_margin_norm"].to_numpy(float)
    delta = wf_margin - cci_margin
    improvement = delta > 0
    repair = (cci_margin < 0) & (wf_margin >= 0)
    summary = {
        "status": "local_rerun_verified",
        "dataset_model": setting,
        "n_images": len(frame),
        "switch_count": int(switched.sum()),
        "switch_rate": float(switched.mean()),
        "switch_reference_frozen": FROZEN_SWITCH_COUNTS[setting],
        "switch_count_difference_vs_frozen": int(switched.sum()) - FROZEN_SWITCH_COUNTS[setting],
        "delta_margin_improvement_count": int((switched & improvement).sum()),
        "delta_margin_improvement_fraction_among_switched": float(improvement[switched].mean()),
        "sign_repair_count": int((switched & repair).sum()),
        "sign_repair_fraction_among_switched": float(repair[switched].mean()),
        "margin_improvement_definition": "WF normalized aggregate worst-foil margin > CCI normalized aggregate worst-foil margin",
        "sign_repair_definition": "CCI margin < 0 and WF margin >= 0; counted separately from all positive margin improvements",
    }
    return pd.DataFrame(rows), summary


def parse_array(value: object) -> np.ndarray:
    try:
        parsed = json.loads(str(value))
        return np.asarray(parsed, dtype=np.float64)
    except (TypeError, ValueError, json.JSONDecodeError):
        return np.asarray([], dtype=np.float64)


def strategy_validation(frame: pd.DataFrame, setting: str) -> dict:
    array_fields = (
        "candidate_selection_target",
        "candidate_selection_mean_foil",
        "candidate_selection_max_foil",
        "candidate_eval_margin_norm",
        "candidate_eval_pmean_norm",
        "candidate_eval_target_drop_raw",
        "candidate_eval_target_drop_norm",
        "candidate_bbox_precision",
    )
    counts = {key: 0 for key in (
        "array_length_failures", "array_nonfinite_failures", "cci_region_mismatches",
        "feasible_count_mismatches", "wf_region_mismatches", "mean_region_mismatches",
        "max_0_1_region_mismatches", "summary_value_mismatches",
    )}
    max_summary_abs_diff = 0.0
    for row in frame.itertuples(index=False):
        arrays = {field: parse_array(getattr(row, field)) for field in array_fields}
        if any(len(values) != CLUSTERS for values in arrays.values()):
            counts["array_length_failures"] += 1
            continue
        if any(not np.isfinite(values).all() for values in arrays.values()):
            counts["array_nonfinite_failures"] += 1
            continue
        target = arrays["candidate_selection_target"]
        cci = int(np.argmax(target))
        feasible = target >= target[cci] - EPSILON
        wf = int(np.argmax(np.where(feasible, target - arrays["candidate_selection_max_foil"], -np.inf)))
        mean = int(np.argmax(np.where(feasible, target - arrays["candidate_selection_mean_foil"], -np.inf)))
        max_01 = int(np.argmax(np.where(feasible, target - 0.1 * arrays["candidate_selection_max_foil"], -np.inf)))
        counts["cci_region_mismatches"] += int(cci != int(row.cci_region))
        counts["feasible_count_mismatches"] += int(int(feasible.sum()) != int(row.feasible_set_size))
        counts["wf_region_mismatches"] += int(wf != int(row.wf_region))
        counts["mean_region_mismatches"] += int(mean != int(row.mean_region))
        counts["max_0_1_region_mismatches"] += int(max_01 != int(row.max_0_1_region))
        for method, region in (("cci", cci), ("wf", wf), ("mean", mean), ("max_0_1", max_01)):
            for metric in ("margin_norm", "pmean_norm", "target_drop_raw", "target_drop_norm", "bbox_precision"):
                observed = float(getattr(row, f"{method}_{metric}"))
                candidate_field = "candidate_bbox_precision" if metric == "bbox_precision" else f"candidate_eval_{metric}"
                expected = float(arrays[candidate_field][region])
                difference = abs(observed - expected)
                max_summary_abs_diff = max(max_summary_abs_diff, difference)
                counts["summary_value_mismatches"] += int(difference > TOLERANCE)
    rows_checked = len(frame)
    passed = all(value == 0 for value in counts.values())
    return {
        "status": "local_rerun_verified",
        "dataset_model": setting,
        "rows_checked": rows_checked,
        **counts,
        "max_summary_abs_diff": max_summary_abs_diff,
        "tolerance": TOLERANCE,
        "passed": passed,
        "strategy_definition_source": "rerun_workspace/scripts/run_local_rerun.py: CCI argmax; WF/Mean/Max-.1 argmax over epsilon-feasible target drop minus max/mean/0.1*max selection foil drop",
    }


def compare_tables(project: Path, out: Path, recomputed: dict[str, pd.DataFrame]) -> pd.DataFrame:
    existing_names = {
        "paired_difference_ci_10000": "paired_difference_ci_10000.csv",
        "screening_failure_rates": "screening_failure_rates.csv",
        "feasible_set_size_distribution": "feasible_set_size_distribution.csv",
        "switch_subset_changes_ci": "switch_subset_changes_ci.csv",
        "switch_margin_improvement_vs_sign_repair": "switch_margin_improvement_vs_sign_repair.csv",
    }
    comparisons: list[dict] = []
    for table_name, new_table in recomputed.items():
        existing_path = project / "rerun_workspace" / "results" / existing_names[table_name]
        if not existing_path.exists():
            comparisons.append({"table": table_name, "field": "__file__", "existing": "missing", "recomputed": "present", "match": False})
            continue
        old = pd.read_csv(existing_path)
        key_cols = [c for c in ("dataset_model", "comparison", "metric", "method", "subset", "feasible_set_size") if c in new_table.columns and c in old.columns]
        value_cols = [c for c in new_table.columns if c in old.columns and c not in key_cols and c not in ("status", "bootstrap_base_seed", "screen_pass_bootstrap_seed", "failure_bootstrap_seed", "joint_failure_bootstrap_seed", "conditional_bootstrap_seed")]
        for _, new_row in new_table.iterrows():
            mask = np.ones(len(old), dtype=bool)
            for key in key_cols:
                mask &= old[key].astype(str).to_numpy() == str(new_row[key])
            matches = old.loc[mask]
            if len(matches) != 1:
                comparisons.append({"table": table_name, "field": "row_match", "existing": len(matches), "recomputed": 1, "match": False})
                continue
            old_row = matches.iloc[0]
            for field in value_cols:
                new_value, old_value = new_row[field], old_row[field]
                try:
                    numeric = np.isfinite(float(new_value)) and np.isfinite(float(old_value))
                    difference = abs(float(new_value) - float(old_value)) if numeric else None
                    match = bool(numeric and difference <= TOLERANCE) if numeric else str(new_value) == str(old_value)
                except (TypeError, ValueError):
                    difference = None
                    match = str(new_value) == str(old_value)
                comparisons.append({
                    "table": table_name,
                    "keys": ";".join(f"{key}={new_row[key]}" for key in key_cols),
                    "field": field,
                    "existing": old_value,
                    "recomputed": new_value,
                    "abs_diff": difference,
                    "match": match,
                    "difference_class": "definition_correction" if field in ("difference_definition", "sign_repair_definition") else "statistical_value",
                })
    return pd.DataFrame(comparisons)


def frozen_paths(project: Path) -> tuple[Path, Path, Path]:
    root = project / "verification_2026-09-13" / "frozen"
    b16 = root / "derived" / "reviewer_cci_preserving_v1" / "01_original_cci_per_sample.csv"
    b16_target = root / "derived" / "reviewer_cci_preserving_v1" / "03_target_preserving_per_sample.csv"
    b32 = root / "results" / "disjoint_foil_v17_2_clean" / "replay_current_1" / "coco_openai_b32_n27708" / "sufficient_statistics.npz"
    return b16, b16_target, b32


def frozen_local_comparison(project: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frozen_path, _, _ = frozen_paths(project)
    local_path = project / "rerun_workspace" / "runs" / "coco_openai_b16" / "per_image.csv"
    frozen = pd.read_csv(frozen_path).set_index("image_id", drop=False)
    local = pd.read_csv(local_path).set_index("image_id", drop=False)
    if set(frozen.index) != set(local.index):
        raise ValueError("COCO B/16 frozen/local image_id sets do not align")
    local = local.loc[frozen.index]
    frozen_screen = (frozen["bbox_precision"] >= 0.5) & (frozen["aggregate_pmean_foil"] > 0)
    frozen_failure = frozen["aggregate_pmax"] < 0
    frozen_joint = frozen_screen & frozen_failure
    local_screen = (local["cci_bbox_precision"] >= 0.5) & (local["cci_pmean_norm"] > 0)
    local_failure = local["cci_margin_norm"] < 0
    local_joint = local_screen & local_failure
    states = pd.DataFrame({
        "frozen_screen": frozen_screen,
        "local_screen": local_screen,
        "frozen_failure": frozen_failure,
        "local_failure": local_failure,
        "frozen_joint_failure": frozen_joint,
        "local_joint_failure": local_joint,
    }, index=frozen.index)
    changed = states.ne(states.shift(axis=1))
    changed_any = (
        (frozen_screen != local_screen)
        | (frozen_failure != local_failure)
        | (frozen_joint != local_joint)
    )
    rows: list[dict] = []
    for image_id in frozen.index[changed_any]:
        f = frozen.loc[image_id]
        l = local.loc[image_id]
        changes = []
        if bool(frozen_screen.loc[image_id] != local_screen.loc[image_id]): changes.append("screen")
        if bool(frozen_failure.loc[image_id] != local_failure.loc[image_id]): changes.append("failure")
        if bool(frozen_joint.loc[image_id] != local_joint.loc[image_id]): changes.append("joint_failure")
        local_cci = int(l["cci_region"])
        frozen_cci = int(f["cci_region"])
        local_target = parse_array(l["candidate_selection_target"])
        rows.append({
            "image_id": int(image_id),
            "sample_index_frozen": int(f["sample_index"]),
            "sample_index_local": int(l["sample_index"]),
            "change_types": ",".join(changes),
            "frozen_cci_region": frozen_cci,
            "local_cci_region": local_cci,
            "local_wf_region": int(l["wf_region"]),
            "local_mean_region": int(l["mean_region"]),
            "local_max_0_1_region": int(l["max_0_1_region"]),
            "frozen_selection_score": float(f["selection_score"]),
            "local_cci_selection_target_at_local_region": float(local_target[local_cci]),
            "local_cci_selection_target_at_frozen_region": float(local_target[frozen_cci]),
            "selection_score_delta_at_local_region": float(local_target[local_cci] - f["selection_score"]),
            "selection_score_delta_at_frozen_region": float(local_target[frozen_cci] - f["selection_score"]),
            "frozen_bbox_precision": float(f["bbox_precision"]),
            "local_cci_bbox_precision": float(l["cci_bbox_precision"]),
            "bbox_delta_local_minus_frozen": float(l["cci_bbox_precision"] - f["bbox_precision"]),
            "frozen_aggregate_pmean_foil": float(f["aggregate_pmean_foil"]),
            "local_cci_pmean_norm": float(l["cci_pmean_norm"]),
            "pmean_delta_local_minus_frozen": float(l["cci_pmean_norm"] - f["aggregate_pmean_foil"]),
            "frozen_aggregate_pmax": float(f["aggregate_pmax"]),
            "local_cci_margin_norm": float(l["cci_margin_norm"]),
            "margin_delta_local_minus_frozen": float(l["cci_margin_norm"] - f["aggregate_pmax"]),
            "bbox_threshold": 0.5,
            "pmean_threshold": 0.0,
            "margin_threshold": 0.0,
            "frozen_screen": bool(frozen_screen.loc[image_id]),
            "local_screen": bool(local_screen.loc[image_id]),
            "frozen_failure": bool(frozen_failure.loc[image_id]),
            "local_failure": bool(local_failure.loc[image_id]),
            "frozen_joint_failure": bool(frozen_joint.loc[image_id]),
            "local_joint_failure": bool(local_joint.loc[image_id]),
            "frozen_candidate_arrays_available": False,
            "frozen_candidate_arrays_note": "not available in frozen 01_original_cci_per_sample.csv",
            **{field: l[field] for field in (
                "candidate_selection_target", "candidate_selection_mean_foil", "candidate_selection_max_foil",
                "candidate_eval_margin_norm", "candidate_eval_pmean_norm", "candidate_eval_target_drop_raw",
                "candidate_eval_target_drop_norm", "candidate_bbox_precision",
            )},
        })
    changed_table = pd.DataFrame(rows).sort_values("image_id")
    changed_table.to_csv(out / "frozen_local_state_changes_coco_b16.csv", index=False)
    summary = pd.DataFrame([
        {"state": "screen_pass", "frozen_count": int(frozen_screen.sum()), "local_count": int(local_screen.sum()), "local_minus_frozen": int(local_screen.sum() - frozen_screen.sum()), "changed_images": int((frozen_screen != local_screen).sum())},
        {"state": "failure", "frozen_count": int(frozen_failure.sum()), "local_count": int(local_failure.sum()), "local_minus_frozen": int(local_failure.sum() - frozen_failure.sum()), "changed_images": int((frozen_failure != local_failure).sum())},
        {"state": "joint_failure", "frozen_count": int(frozen_joint.sum()), "local_count": int(local_joint.sum()), "local_minus_frozen": int(local_joint.sum() - frozen_joint.sum()), "changed_images": int((frozen_joint != local_joint).sum())},
        {"state": "cci_region_label", "frozen_count": math.nan, "local_count": math.nan, "local_minus_frozen": math.nan, "changed_images": int((frozen["cci_region"] != local["cci_region"]).sum())},
    ])
    summary.to_csv(out / "frozen_local_state_change_summary.csv", index=False)

    region_rows: list[dict] = []
    for image_id in frozen.index[frozen["cci_region"] != local["cci_region"]]:
        f, l = frozen.loc[image_id], local.loc[image_id]
        region_rows.append({
            "image_id": int(image_id),
            "sample_index": int(l["sample_index"]),
            "frozen_cci_region": int(f["cci_region"]),
            "local_cci_region": int(l["cci_region"]),
            "frozen_selection_score": float(f["selection_score"]),
            "local_selection_target_at_local_cci": float(parse_array(l["candidate_selection_target"])[int(l["cci_region"])]),
            "local_selection_target_at_frozen_cci": float(parse_array(l["candidate_selection_target"])[int(f["cci_region"])]),
            "frozen_bbox_precision": float(f["bbox_precision"]),
            "local_cci_bbox_precision": float(l["cci_bbox_precision"]),
            "frozen_aggregate_pmean_foil": float(f["aggregate_pmean_foil"]),
            "local_cci_pmean_norm": float(l["cci_pmean_norm"]),
            "frozen_aggregate_pmax": float(f["aggregate_pmax"]),
            "local_cci_margin_norm": float(l["cci_margin_norm"]),
            "candidate_arrays_available_local": True,
            "candidate_arrays_available_frozen": False,
        })
    pd.DataFrame(region_rows).sort_values("image_id").to_csv(out / "frozen_local_cci_region_changes_coco_b16.csv", index=False)
    details = {
        "n_aligned": len(frozen),
        "same_image_id_order": bool(np.array_equal(frozen["image_id"].to_numpy(), local["image_id"].to_numpy())),
        "screen_frozen": int(frozen_screen.sum()), "screen_local": int(local_screen.sum()),
        "failure_frozen": int(frozen_failure.sum()), "failure_local": int(local_failure.sum()),
        "joint_frozen": int(frozen_joint.sum()), "joint_local": int(local_joint.sum()),
        "screen_changed": int((frozen_screen != local_screen).sum()),
        "failure_changed": int((frozen_failure != local_failure).sum()),
        "joint_changed": int((frozen_joint != local_joint).sum()),
        "cci_region_changed": int((frozen["cci_region"] != local["cci_region"]).sum()),
        "changed_image_ids": [int(x) for x in changed_table["image_id"]],
    }
    return changed_table, summary, details


def availability_table(project: Path, out: Path) -> pd.DataFrame:
    b16, b16_target, b32 = frozen_paths(project)
    rows = [
        {"dataset_model": "coco_openai_b16", "frozen_per_image_file": str(b16), "per_image_available": b16.exists(), "candidate_arrays_available": False, "regions_available": True, "aggregate_switch_count_available": True, "frozen_switch_count": 663, "limitation": "per-image aggregate/selected-region fields exist; candidate response arrays are absent"},
        {"dataset_model": "coco_openai_b32", "frozen_per_image_file": "not found in frozen package", "per_image_available": False, "candidate_arrays_available": False, "regions_available": False, "aggregate_switch_count_available": True, "frozen_switch_count": 613, "limitation": "only sufficient_statistics.npz/replay_summary.json are available; no image_id-aligned region or candidate fields"},
        {"dataset_model": "voc2007_openai_b16", "frozen_per_image_file": "not found in frozen package", "per_image_available": False, "candidate_arrays_available": False, "regions_available": False, "aggregate_switch_count_available": True, "frozen_switch_count": 50, "limitation": "no frozen per-image fields available for alignment"},
        {"dataset_model": "voc2007_openai_b32", "frozen_per_image_file": "not found in frozen package", "per_image_available": False, "candidate_arrays_available": False, "regions_available": False, "aggregate_switch_count_available": True, "frozen_switch_count": 38, "limitation": "no frozen per-image fields available for alignment"},
    ]
    table = pd.DataFrame(rows)
    table.to_csv(out / "frozen_artifact_availability.csv", index=False)
    return table


def runtime_table(project: Path, out: Path) -> pd.DataFrame:
    rows = []
    for setting in SETTINGS:
        metadata_path = project / "rerun_workspace" / "runs" / setting / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        resumed = int(metadata.get("completed_before", 0)) > 0
        rows.append({
            "dataset_model": setting,
            "device": metadata.get("device_last"),
            "elapsed_seconds_reported": metadata.get("elapsed_seconds_this_run"),
            "images_per_second_reported": metadata.get("images_per_second_this_run"),
            "completed_before": metadata.get("completed_before"),
            "completed_this_run": metadata.get("completed_this_run"),
            "completed_total": metadata.get("completed_total"),
            "time_scope": "resumed portion only; not complete experiment elapsed time" if resumed else "full local run",
            "source_metadata": str(metadata_path),
        })
    table = pd.DataFrame(rows)
    table.to_csv(out / "corrected_runtime_summary.csv", index=False)
    return table


def write_corrections(out: Path) -> None:
    (out / "corrected_config_notes.md").write_text(
        "# Corrections to configuration wording\n\n"
        "The original `rerun_workspace/config/local_rerun_config.json` is preserved.\n"
        "This verification directory supplies the corrected interpretation used by\n"
        "the independent recomputation.\n\n"
        "## Direct paired baselines\n\n"
        "`WF-Mean` and `WF-Max-.1` are direct, same-image comparisons on the same\n"
        "epsilon-feasible candidate set and held-out evaluation: `WF(metric) -\n"
        "baseline(metric)`. They are not `candidate - CCI`. The two comparisons\n"
        "are computed separately, and each CI resamples the per-image difference\n"
        "vector directly.\n\n"
        "## Switch-subset comparison\n\n"
        "The switch-subset analysis is a different estimand: `WF(metric) -\n"
        "CCI(metric)` restricted to images with `wf_region != cci_region`. Its\n"
        "bootstrap unit is the switched image, and its CI is calculated from the\n"
        "subset difference vector.\n\n"
        "## Screening seeds\n\n"
        "Each screening rate has its own recorded seed in\n"
        "`corrected_screening_failure_rates.csv`: screen-pass, failure, joint\n"
        "failure, and conditional failure. The conditional CI jointly resamples\n"
        "the joint-failure numerator and screen-pass denominator.\n\n"
        "## Runtime wording\n\n"
        "The COCO B/16 and B/32 elapsed values are resumed portions only; they are\n"
        "not complete-experiment elapsed times. See\n"
        "`corrected_runtime_summary.csv`.\n",
        encoding="utf-8",
    )


def file_hash(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def git_revision(path: Path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_report(
    out: Path,
    manifests: list[dict],
    strategy: pd.DataFrame,
    comparisons: pd.DataFrame,
    changed_table: pd.DataFrame,
    frozen_details: dict,
    availability: pd.DataFrame,
    runtime: pd.DataFrame,
) -> None:
    mismatch_mask = comparisons.get("match", pd.Series(dtype=bool)) == False
    statistical_mismatch_mask = mismatch_mask & comparisons.get("difference_class", pd.Series(dtype=str)).eq("statistical_value")
    definition_correction_count = int(mismatch_mask.sum() - statistical_mismatch_mask.sum())
    statistical_mismatch_count = int(statistical_mismatch_mask.sum())
    strategy_failures = int((~strategy["passed"]).sum())
    lines = [
        "# Local rerun verification report",
        "",
        f"Verification date: 2026-09-13  |  baseline commit: `{BASELINE_COMMIT}`",
        "",
        "## Scope",
        "",
        "This is an independent read-only recomputation from the four local `per_image.csv` files. It does not modify the frozen package, the existing local-rerun results, paper sources, PDF, LaTeX, or figures. No model rerun was performed during this verification pass.",
        "",
        "The bootstrap algorithm is the archive algorithm: NumPy `default_rng`, image-level resampling in chunks of 64, 10,000 draws, and percentile 2.5/97.5 intervals. Conditional screening intervals jointly resample the numerator and denominator and recompute their ratio.",
        "",
        "## Input completion",
        "",
        "| setting | rows | status |",
        "|---|---:|---|",
    ]
    for item in manifests:
        lines.append(f"| {item['dataset_model']} | {item.get('n_images', '')} | {item['status']} |")
    lines += [
        "",
        "## Recomputed tables",
        "",
        "Delivered files include `corrected_paired_difference_ci_10000.csv`, `corrected_screening_failure_rates.csv`, `corrected_switch_subset_changes_ci.csv`, `corrected_switch_margin_improvement_vs_sign_repair.csv`, `corrected_feasible_set_size_distribution.csv`, and the corresponding per-image difference CSV.",
        "",
        f"Comparison with the existing local summary tables found {statistical_mismatch_count} statistical-value mismatches under tolerance {TOLERANCE:g}; {definition_correction_count} textual definition corrections are listed separately in the comparison CSV. Seed columns that were absent or mislabeled in the existing screening table are supplied explicitly in the corrected table.",
        "",
        "## Strategy and candidate-array validation",
        "",
        f"Rows with any strategy/array validation failure: {strategy_failures}. The check reconstructs CCI argmax and WF/Mean/Max-.1 argmax over the epsilon-feasible candidates, then verifies feasible-set size and every selected held-out summary value against its candidate array entry.",
        "",
        "| setting | rows | array/strategy failures | max summary abs diff |",
        "|---|---:|---:|---:|",
    ]
    for row in strategy.itertuples(index=False):
        failure_count = sum(int(getattr(row, field)) for field in (
            "array_length_failures", "array_nonfinite_failures", "cci_region_mismatches",
            "feasible_count_mismatches", "wf_region_mismatches", "mean_region_mismatches",
            "max_0_1_region_mismatches", "summary_value_mismatches",
        ))
        lines.append(f"| {row.dataset_model} | {row.rows_checked} | {failure_count} | {row.max_summary_abs_diff:.3g} |")
    lines += [
        "",
        "## Frozen versus local COCO B/16",
        "",
        f"The image_id sets align exactly ({frozen_details['n_aligned']} images, same order). Frozen counts are screen={frozen_details['screen_frozen']}, failure={frozen_details['failure_frozen']}, joint={frozen_details['joint_frozen']}; local counts are screen={frozen_details['screen_local']}, failure={frozen_details['failure_local']}, joint={frozen_details['joint_local']}. Changed-image counts are screen={frozen_details['screen_changed']}, failure={frozen_details['failure_changed']}, joint={frozen_details['joint_changed']}. The complete state-change rows are in `frozen_local_state_changes_coco_b16.csv`.",
        "",
        "| image_id | change types | frozen CCI | local CCI | frozen pmax | local margin | frozen bbox | local bbox |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in changed_table.itertuples(index=False):
        lines.append(
            f"| {row.image_id} | {row.change_types} | {row.frozen_cci_region} | {row.local_cci_region} | "
            f"{row.frozen_aggregate_pmax:.6f} | {row.local_cci_margin_norm:.6f} | "
            f"{row.frozen_bbox_precision:.6f} | {row.local_cci_bbox_precision:.6f} |"
        )
    lines += [
        "",
        "The 135671 screen change is accompanied by CCI region 6→1 and bbox precision 0.078947→0.583333; the 185335 and 213809 failure changes keep the CCI region but change the normalized worst-foil sign. Their frozen-to-local worst-foil differences are 0.777786 and 0.091734 respectively, so these are not explainable as threshold rounding alone. The frozen candidate arrays and intermediate tensors needed to identify whether the source is model arithmetic, clustering, preprocessing, or another protocol difference are unavailable.",
        "",
        f"CCI region labels changed for {frozen_details['cci_region_changed']} images. These are listed separately in `frozen_local_cci_region_changes_coco_b16.csv`; frozen candidate arrays are not present, so the file cannot by itself distinguish a changed response from changed clustering/region assignment for those label changes.",
        "",
        "For the three screen/failure/joint state-change images, the report compares the frozen selected-region scalar fields with the local candidate array and local CCI/held-out scalars. It does not attribute the discrepancies to MPS solely from their size: frozen candidate arrays and the complete frozen B/16 protocol-level intermediate tensors are unavailable.",
        "",
        "## Frozen artifact limits",
        "",
        "The frozen B/32 artifact contains aggregate sufficient statistics and replay metadata but no image_id-aligned per-image region or candidate-response fields. Therefore the frozen 613-to-local-614 switch change cannot be enumerated or causally decomposed from the available frozen files. This is a data-availability limit, not a claim that the two runs are identical.",
        "",
        availability.to_markdown(index=False),
        "",
        "## Runtime wording correction",
        "",
        runtime[["dataset_model", "elapsed_seconds_reported", "images_per_second_reported", "completed_before", "completed_this_run", "completed_total", "time_scope"]].to_markdown(index=False),
        "",
        "## Delivered provenance",
        "",
        "`verification_metadata.json` records input paths, SHA256 values, Python/package versions, the baseline commit, and the exact recomputation script. The original configuration wording remains untouched; the corrected interpretation is in `corrected_config_notes.md`.",
        "",
    ]
    (out / "verification_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project = args.project_root.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_corrections(out)

    manifests: list[dict] = []
    pair_tables: list[pd.DataFrame] = []
    pair_summaries: list[pd.DataFrame] = []
    screen_rows: list[dict] = []
    feasible_tables: list[pd.DataFrame] = []
    switch_tables_all: list[pd.DataFrame] = []
    switch_summaries: list[dict] = []
    strategy_rows: list[dict] = []
    for setting in SETTINGS:
        path = project / "rerun_workspace" / "runs" / setting / "per_image.csv"
        frame = read_frame(project, setting)
        pair, pair_summary = pairwise_tables(frame, setting, args.seed, args.bootstrap)
        pair_tables.append(pair)
        pair_summaries.append(pair_summary)
        for method in ("cci", "wf"):
            screen_rows.append(screening_table(frame, setting, method, args.seed, args.bootstrap))
        feasible_tables.append(feasible_table(frame, setting))
        switch_table, switch_summary = switch_tables(frame, setting, args.seed, args.bootstrap)
        switch_tables_all.append(switch_table)
        switch_summaries.append(switch_summary)
        strategy_rows.append(strategy_validation(frame, setting))
        manifests.append({"status": "completed", "dataset_model": setting, "path": str(path), "n_images": len(frame), "rerun_status": "local_rerun"})

    pair_per_image = pd.concat(pair_tables, ignore_index=True)
    pair_summary = pd.concat(pair_summaries, ignore_index=True)
    screening = pd.DataFrame(screen_rows)
    feasible = pd.concat(feasible_tables, ignore_index=True)
    switches = pd.concat(switch_tables_all, ignore_index=True)
    switch_summary = pd.DataFrame(switch_summaries)
    strategy = pd.DataFrame(strategy_rows)
    pair_per_image.to_csv(out / "corrected_paired_differences_per_image.csv", index=False)
    pair_summary.to_csv(out / "corrected_paired_difference_ci_10000.csv", index=False)
    screening.to_csv(out / "corrected_screening_failure_rates.csv", index=False)
    feasible.to_csv(out / "corrected_feasible_set_size_distribution.csv", index=False)
    switches.to_csv(out / "corrected_switch_subset_changes_ci.csv", index=False)
    switch_summary.to_csv(out / "corrected_switch_margin_improvement_vs_sign_repair.csv", index=False)
    strategy.to_csv(out / "strategy_choice_validation.csv", index=False)
    pd.DataFrame(manifests).to_csv(out / "completed_and_missing.csv", index=False)

    recomputed = {
        "paired_difference_ci_10000": pair_summary,
        "screening_failure_rates": screening,
        "feasible_set_size_distribution": feasible,
        "switch_subset_changes_ci": switches,
        "switch_margin_improvement_vs_sign_repair": switch_summary,
    }
    comparison = compare_tables(project, out, recomputed)
    comparison.to_csv(out / "recomputed_vs_existing_statistics.csv", index=False)
    changed_table, _, frozen_details = frozen_local_comparison(project, out)
    availability = availability_table(project, out)
    runtime = runtime_table(project, out)

    input_paths = [
        project / "rerun_workspace" / "config" / "local_rerun_config.json",
        project / "rerun_workspace" / "scripts" / "run_local_rerun.py",
        project / "rerun_workspace" / "scripts" / "analyze_local_rerun.py",
        project / "verification_2026-09-13" / "frozen" / "derived" / "reviewer_cci_preserving_v1" / "01_original_cci_per_sample.csv",
        project / "verification_2026-09-13" / "frozen" / "derived" / "reviewer_cci_preserving_v1" / "03_target_preserving_per_sample.csv",
        project / "verification_2026-09-13" / "frozen" / "results" / "disjoint_foil_v17_2_clean" / "replay_current_1" / "coco_openai_b32_n27708" / "sufficient_statistics.npz",
    ]
    input_paths.extend(project / "rerun_workspace" / "results" / name for name in (
        "paired_difference_ci_10000.csv", "screening_failure_rates.csv",
        "feasible_set_size_distribution.csv", "switch_subset_changes_ci.csv",
        "switch_margin_improvement_vs_sign_repair.csv",
    ))
    input_paths.extend(project / "rerun_workspace" / "runs" / setting / name for setting in SETTINGS for name in ("per_image.csv", "run_metadata.json"))
    hashes = [file_hash(path) for path in input_paths if path.exists()]
    verifier_hash = file_hash(Path(__file__).resolve())
    pd.DataFrame(hashes + [verifier_hash]).to_csv(out / "source_sha256.csv", index=False)
    packages = {}
    for package in ("numpy", "pandas", "torch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    metadata = {
        "verification_status": "completed",
        "baseline_commit": BASELINE_COMMIT,
        "project_root": str(project),
        "verification_output": str(out),
        "bootstrap": {"count": args.bootstrap, "base_seed": args.seed, "interval": INTERVAL, "rng": "numpy.default_rng (PCG64)", "unit": "image"},
        "protocol_constants": {"epsilon": EPSILON, "clusters": CLUSTERS},
        "git_revisions": {"conway_upload": git_revision(project / "conway_upload"), "cci_archive": git_revision(project / "cci_archive")},
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "input_sha256": hashes,
        "verifier_script_sha256": verifier_hash,
        "frozen_local_details": frozen_details,
        "notes": ["Independent recomputation; existing frozen and local-rerun files were not overwritten.", "No full model rerun performed in verification pass."],
    }
    (out / "verification_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out, manifests, strategy, comparison, changed_table, frozen_details, availability, runtime)
    print(json.dumps({
        "status": "completed",
        "output": str(out),
        "settings": {setting: len(read_frame(project, setting)) for setting in SETTINGS},
        "strategy_failures": int((~strategy["passed"]).sum()),
        "summary_statistical_mismatches": int(((comparison.get("match", pd.Series(dtype=bool)) == False) & comparison.get("difference_class", pd.Series(dtype=str)).eq("statistical_value")).sum()),
        "summary_definition_corrections": int(((comparison.get("match", pd.Series(dtype=bool)) == False) & comparison.get("difference_class", pd.Series(dtype=str)).eq("definition_correction")).sum()),
        "frozen_local_changed_image_ids": frozen_details["changed_image_ids"],
        "frozen_local_details": frozen_details,
    }, indent=2))


if __name__ == "__main__":
    main()
