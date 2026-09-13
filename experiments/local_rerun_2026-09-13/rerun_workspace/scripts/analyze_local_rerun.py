#!/usr/bin/env python3
"""Compute image-level paired analyses from local_rerun per-image CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SETTINGS = (
    ("coco", "openai_b16"),
    ("coco", "openai_b32"),
    ("voc2007", "openai_b16"),
    ("voc2007", "openai_b32"),
)
METRICS = ("margin_norm", "target_drop_raw", "target_drop_norm", "bbox_precision")


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1701)
    return p.parse_args()


def bootstrap_mean(values: np.ndarray, seed: int, count: int) -> tuple[float, float, float, int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    draws = np.empty(count, dtype=np.float64)
    for start in range(0, count, 64):
        size = min(64, count - start)
        draws[start:start + size] = values[rng.integers(0, len(values), size=(size, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), int(len(values))


def bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, seed: int, count: int) -> tuple[float, float, float, int]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator)
    numerator, denominator = numerator[valid], denominator[valid]
    if not len(numerator):
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    draws = np.empty(count, dtype=np.float64)
    for start in range(0, count, 64):
        size = min(64, count - start)
        idx = rng.integers(0, len(numerator), size=(size, len(numerator)))
        den = denominator[idx].sum(1)
        draws[start:start + size] = np.divide(numerator[idx].sum(1), den, out=np.full(size, np.nan), where=den > 0)
    draws = draws[np.isfinite(draws)]
    return float(numerator.sum() / denominator.sum()) if denominator.sum() else np.nan, float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), int(len(numerator))


def comparison_rows(frame: pd.DataFrame, setting: str, left: str, right: str, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    diff = frame[["sample_index", "image_id"]].copy()
    rows = []
    for metric in METRICS:
        values = frame[f"{left}_{metric}"].to_numpy(float) - frame[f"{right}_{metric}"].to_numpy(float)
        diff[f"{left}_minus_{right}_{metric}"] = values
        mean, low, high, n = bootstrap_mean(values, args.seed + sum(map(ord, setting + left + right + metric)), args.bootstrap)
        rows.append({
            "status": "local_rerun", "dataset_model": setting,
            "comparison": f"{left}_minus_{right}", "metric": metric,
            "estimate": mean, "ci_low": low, "ci_high": high,
            "bootstrap_count": args.bootstrap, "bootstrap_seed": args.seed + sum(map(ord, setting + left + right + metric)),
            "bootstrap_interval": "percentile_2.5_97.5", "bootstrap_unit": "image",
            "difference_definition": f"{left}_{metric} - {right}_{metric}", "n_valid": n,
        })
    return diff, pd.DataFrame(rows)


def screening_rows(frame: pd.DataFrame, setting: str, method: str, args) -> dict:
    screen = (frame[f"{method}_bbox_precision"] >= .5) & (frame[f"{method}_pmean_norm"] > 0)
    failure = frame[f"{method}_margin_norm"] < 0
    joint = screen & failure
    n = len(frame)
    rng_seed = args.seed + sum(map(ord, setting + method + "screen"))
    def rate(mask):
        est, low, high, valid = bootstrap_mean(mask.astype(float).to_numpy(), rng_seed + int(mask.sum()), args.bootstrap)
        return est, low, high, valid
    sp, sp_lo, sp_hi, _ = rate(screen)
    fr, fr_lo, fr_hi, _ = rate(failure)
    jr, jr_lo, jr_hi, _ = rate(joint)
    cond, cond_lo, cond_hi, valid = bootstrap_ratio(joint.astype(float).to_numpy(), screen.astype(float).to_numpy(), rng_seed + 23, args.bootstrap)
    return {
        "status": "local_rerun", "dataset_model": setting, "method": method, "n_images": n,
        "screen_definition": "bbox_precision >= 0.5 AND normalized aggregate foil-mean margin > 0",
        "failure_definition": "normalized aggregate worst-foil margin < 0",
        "screen_pass_count": int(screen.sum()), "failure_count": int(failure.sum()), "joint_failure_count": int(joint.sum()),
        "screen_pass_rate": sp, "screen_pass_ci_low": sp_lo, "screen_pass_ci_high": sp_hi,
        "failure_rate": fr, "failure_ci_low": fr_lo, "failure_ci_high": fr_hi,
        "joint_failure_rate": jr, "joint_failure_ci_low": jr_lo, "joint_failure_ci_high": jr_hi,
        "failure_given_screen_pass": cond, "conditional_ci_low": cond_lo, "conditional_ci_high": cond_hi,
        "bootstrap_count": args.bootstrap, "bootstrap_seed": rng_seed,
        "bootstrap_interval": "percentile_2.5_97.5", "bootstrap_unit": "image; numerator and denominator jointly resampled",
        "valid_images": valid,
    }


def feasible_distribution(frame: pd.DataFrame, setting: str) -> pd.DataFrame:
    counts = frame["feasible_set_size"].value_counts().reindex(range(1, 9), fill_value=0)
    return pd.DataFrame([{"status": "local_rerun", "dataset_model": setting, "feasible_set_size": int(k), "image_count": int(v), "image_fraction": float(v / len(frame))} for k, v in counts.items()])


def switch_rows(frame: pd.DataFrame, setting: str, args) -> tuple[pd.DataFrame, dict]:
    switched = frame["wf_region"].to_numpy(int) != frame["cci_region"].to_numpy(int)
    subset = frame.loc[switched].copy()
    rows = []
    for metric in METRICS:
        values = subset[f"wf_{metric}"].to_numpy(float) - subset[f"cci_{metric}"].to_numpy(float)
        seed = args.seed + sum(map(ord, setting + "switch" + metric))
        mean, low, high, n = bootstrap_mean(values, seed, args.bootstrap)
        rows.append({
            "status": "local_rerun", "dataset_model": setting, "subset": "cci_to_wf_switched",
            "metric": metric, "n_images": len(subset), "estimate": mean, "ci_low": low, "ci_high": high,
            "p10": float(np.quantile(values, .10)) if len(values) else np.nan,
            "p25": float(np.quantile(values, .25)) if len(values) else np.nan,
            "p50": float(np.quantile(values, .50)) if len(values) else np.nan,
            "p75": float(np.quantile(values, .75)) if len(values) else np.nan,
            "p90": float(np.quantile(values, .90)) if len(values) else np.nan,
            "bootstrap_count": args.bootstrap, "bootstrap_seed": seed,
            "bootstrap_interval": "percentile_2.5_97.5", "bootstrap_unit": "switched image",
        })
    wf_margin = frame["wf_margin_norm"].to_numpy(float)
    cci_margin = frame["cci_margin_norm"].to_numpy(float)
    delta = wf_margin - cci_margin
    sign_repair = (cci_margin < 0) & (wf_margin >= 0)
    return pd.DataFrame(rows), {
        "status": "local_rerun", "dataset_model": setting, "n_images": len(frame),
        "switch_count": int(switched.sum()), "switch_rate": float(switched.mean()),
        "switch_reference_frozen": {"coco_openai_b16": 663, "coco_openai_b32": 613, "voc2007_openai_b16": 50, "voc2007_openai_b32": 38}.get(setting),
        "delta_margin_improvement_count": int((switched & (delta > 0)).sum()),
        "delta_margin_improvement_fraction_among_switched": float((delta[switched] > 0).mean()) if switched.any() else np.nan,
        "sign_repair_count": int(sign_repair[switched].sum()),
        "sign_repair_fraction_among_switched": float(sign_repair[switched].mean()) if switched.any() else np.nan,
        "margin_improvement_definition": "WF normalized aggregate worst-foil margin > CCI normalized aggregate worst-foil margin",
        "sign_repair_definition": "CCI margin < 0 and WF margin >= 0",
    }


def main():
    a = args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    all_pairs, all_summary, all_screen, all_dist, all_switch, all_switch_summary = [], [], [], [], [], []
    manifest = []
    for dataset, model in SETTINGS:
        key = f"{dataset}_{model}"
        path = a.runs_root / key / "per_image.csv"
        if not path.exists():
            manifest.append({"status": "missing", "dataset_model": key, "path": str(path), "reason": "local rerun has not completed"})
            continue
        frame = pd.read_csv(path)
        if not len(frame):
            manifest.append({"status": "missing", "dataset_model": key, "path": str(path), "reason": "empty per_image.csv"})
            continue
        for left, right in (("wf", "mean"), ("wf", "max_0_1")):
            diff, summary = comparison_rows(frame, key, left, right, a)
            diff.to_csv(a.out_dir / f"{key}_{left}_minus_{right}_per_image.csv", index=False)
            long_rows = []
            for metric in METRICS:
                field = f"{left}_minus_{right}_{metric}"
                for row in diff[["sample_index", "image_id", field]].itertuples(index=False):
                    long_rows.append({
                        "dataset_model": key, "sample_index": int(row.sample_index), "image_id": int(row.image_id),
                        "comparison": f"{left}_minus_{right}", "metric": metric, "difference": float(getattr(row, field)),
                    })
            all_pairs.append(pd.DataFrame(long_rows))
            all_summary.append(summary)
        for method in ("cci", "wf"):
            all_screen.append(screening_rows(frame, key, method, a))
        all_dist.append(feasible_distribution(frame, key))
        switch_frame, switch_summary = switch_rows(frame, key, a)
        all_switch.append(switch_frame)
        all_switch_summary.append(switch_summary)
        manifest.append({"status": "completed", "dataset_model": key, "path": str(path), "n_images": len(frame), "rerun_status": "local_rerun"})
    if all_pairs:
        pd.concat(all_pairs, ignore_index=True).to_csv(a.out_dir / "paired_differences_per_image.csv", index=False)
    if all_summary:
        pd.concat(all_summary, ignore_index=True).to_csv(a.out_dir / "paired_difference_ci_10000.csv", index=False)
    if all_screen:
        pd.DataFrame(all_screen).to_csv(a.out_dir / "screening_failure_rates.csv", index=False)
    if all_dist:
        pd.concat(all_dist, ignore_index=True).to_csv(a.out_dir / "feasible_set_size_distribution.csv", index=False)
    if all_switch:
        pd.concat(all_switch, ignore_index=True).to_csv(a.out_dir / "switch_subset_changes_ci.csv", index=False)
    if all_switch_summary:
        pd.DataFrame(all_switch_summary).to_csv(a.out_dir / "switch_margin_improvement_vs_sign_repair.csv", index=False)
    pd.DataFrame(manifest).to_csv(a.out_dir / "completed_and_missing.csv", index=False)
    (a.out_dir / "analysis_metadata.json").write_text(json.dumps({"status": "local_rerun", "bootstrap_count": a.bootstrap, "bootstrap_seed": a.seed, "settings": SETTINGS}, indent=2) + "\n")
    print(pd.DataFrame(manifest).to_string(index=False))


if __name__ == "__main__":
    main()
