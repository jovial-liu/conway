#!/usr/bin/env python3
"""Validate row completeness and CPU/MPS smoke consistency for the local rerun."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "rerun_workspace" / "runs"
OUT = ROOT / "rerun_workspace" / "results" / "validation_summary.csv"
EXPECTED = {"coco_openai_b16": 27708, "coco_openai_b32": 27708, "voc2007_openai_b16": 1943, "voc2007_openai_b32": 1943}
METRICS = [
    "cci_margin_norm", "wf_margin_norm", "mean_margin_norm", "max_0_1_margin_norm",
    "cci_target_drop_raw", "wf_target_drop_raw", "mean_target_drop_raw", "max_0_1_target_drop_raw",
    "cci_bbox_precision", "wf_bbox_precision", "mean_bbox_precision", "max_0_1_bbox_precision",
]


def main() -> None:
    rows: list[dict] = []
    for key, expected in EXPECTED.items():
        frame = pd.read_csv(RUNS / key / "per_image.csv")
        numeric = frame.select_dtypes("number").to_numpy()
        rows.append({
            "check": "full_run",
            "dataset_model": key,
            "passed": bool(len(frame) == expected and frame.sample_index.nunique() == expected and np.isfinite(numeric).all()),
            "n_rows": len(frame),
            "expected_rows": expected,
            "unique_indices": frame.sample_index.nunique(),
            "max_metric_abs_difference": "",
            "note": "expected row count, unique indices, finite numeric values",
        })
        dataset = "coco" if key.startswith("coco") else "voc2007"
        model = key.split("_")[-2] + "_" + key.split("_")[-1]
        mps = pd.read_csv(RUNS / f"{key}_smoke_mps" / "per_image.csv").sort_values("sample_index").reset_index(drop=True)
        cpu = pd.read_csv(RUNS / f"{key}_smoke_cpu" / "per_image.csv").sort_values("sample_index").reset_index(drop=True)
        max_diff = max(float(np.max(np.abs(mps[c].to_numpy() - cpu[c].to_numpy()))) for c in METRICS)
        regions_equal = all(mps[c].equals(cpu[c]) for c in ("cci_region", "wf_region", "mean_region", "max_0_1_region"))
        rows.append({
            "check": "mps_cpu_smoke",
            "dataset_model": key,
            "passed": bool(mps[["sample_index", "image_id"]].equals(cpu[["sample_index", "image_id"]]) and regions_equal),
            "n_rows": len(mps),
            "expected_rows": len(cpu),
            "unique_indices": mps.sample_index.nunique(),
            "max_metric_abs_difference": max_diff,
            "note": f"same images; regions_equal={regions_equal}",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    if not all(row["passed"] for row in rows):
        raise SystemExit("validation failed")


if __name__ == "__main__":
    main()
