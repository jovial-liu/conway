#!/usr/bin/env python3
"""Independent verification of restricted-foil outputs.

This file deliberately does not import the primary analysis module.  It reads
only the response NPZ files, the fixed local-rerun CSVs, and the diagnostics CSV.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = {
    "coco_openai_b16": (17726, 11410),
    "coco_openai_b32": (17847, 11561),
    "voc2007_openai_b16": (1209, 511),
    "voc2007_openai_b32": (1227, 505),
}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--fixed-root", type=Path, required=True)
    return p.parse_args()


def as_id(v):
    s = str(v)
    return int(s) if s.isdigit() else s


def canonical_image_ids(values):
    return np.asarray([str(int(str(v))) if str(v).isdigit() else str(v) for v in values], dtype=str)


def exact_q(n, k, m):
    if k == 0 or m == 0:
        return 0.0
    if k > n - m:
        return 1.0
    return 1.0 - math.comb(n - m, k) / math.comb(n, k)


def main():
    a = args()
    root = a.root.expanduser().resolve()
    fixed_root = a.fixed_root.expanduser().resolve()
    diag = pd.read_csv(root / "results/per_image_restricted_foil_diagnostics.csv", low_memory=False)
    rows = []
    for name, (expected_a, expected_full) in EXPECTED.items():
        npz = np.load(root / "responses" / name / f"{name}_cci_full_class_responses.npz", allow_pickle=False)
        fixed = pd.read_csv(fixed_root / "runs" / name / "per_image.csv")
        norm = np.asarray(npz["response_norm"], dtype=float)
        image_ids = canonical_image_ids(npz["image_id"])
        categories = [as_id(x) for x in npz["category_ids"].tolist()]
        cat_index = {v: i for i, v in enumerate(categories)}
        target = [as_id(x) for x in npz["target_id"].tolist()]
        target_pos = np.asarray([cat_index[x] for x in target])
        cci = fixed.cci_region.to_numpy(dtype=int)
        target_response = norm[np.arange(len(norm))[:, None], np.arange(8)[None, :], target_pos[:, None]]
        full_mask = np.ones((len(norm), len(categories)), dtype=bool)
        full_mask[np.arange(len(norm)), target_pos] = False
        full_worst = np.where(full_mask[:, None, :], norm, -np.inf).max(axis=2)
        full_failure = target_response[np.arange(len(norm)), cci] - full_worst[np.arange(len(norm)), cci] < 0
        annotated = [{as_id(v) for v in json.loads(s)} for s in npz["annotated_category_ids"].astype(str)]
        absent_failure = []
        q_values = []
        full_worst_ids = []
        absent_worst_ids = []
        for i, t in enumerate(target):
            non_target = [c for c in categories if c != t]
            absent = [c for c in non_target if c not in annotated[i]]
            pos = np.asarray([cat_index[c] for c in absent], dtype=int)
            c = cci[i]
            target_value = target_response[i, c]
            vals_full = norm[i, c, [cat_index[x] for x in non_target]]
            m = int(np.sum(vals_full > target_value))
            q_values.append(exact_q(len(non_target), len(absent), m))
            fw = np.asarray([cat_index[x] for x in non_target])[int(np.argmax(vals_full))]
            full_worst_ids.append(categories[int(fw)])
            if len(pos):
                vals_absent = norm[i, c, pos]
                aw = pos[int(np.argmax(vals_absent))]
                absent_worst_ids.append(categories[int(aw)])
                absent_failure.append(bool(target_value - vals_absent.max() < 0))
            else:
                absent_worst_ids.append(None)
                absent_failure.append(False)
        absent_failure = np.asarray(absent_failure, dtype=bool)
        q_values = np.asarray(q_values, dtype=float)
        A = (fixed.cci_bbox_precision.to_numpy(float) >= 0.5) & (fixed.cci_pmean_norm.to_numpy(float) > 0)
        Aplus = A & (fixed.cci_target_drop_norm.to_numpy(float) > 0)
        d = diag[diag["dataset"] + "_" + diag["model"] == name].sort_values("sample_index")
        diag_q = d.exact_matched_random_failure_probability.to_numpy(float)
        diag_full = d.full_failure.to_numpy(int).astype(bool)
        diag_absent = d.annotation_absent_failure.to_numpy(int).astype(bool)
        checks = {
            "rows_match": len(d) == len(norm) == len(fixed),
            "image_order_match": np.array_equal(image_ids, canonical_image_ids(fixed.image_id.to_numpy())),
            "finite_responses": bool(np.isfinite(norm).all()),
            "A_count_expected": int(A.sum()) == expected_a,
            "A_full_failure_count_expected": int((A & full_failure).sum()) == expected_full,
            "Aplus_count": int(Aplus.sum()),
            "Aplus_absent_failure_count": int((Aplus & absent_failure).sum()),
            "diagnostics_full_failure_match": bool(np.array_equal(full_failure.astype(int), diag_full)),
            "diagnostics_absent_failure_match": bool(np.array_equal(absent_failure.astype(int), diag_absent)),
            "diagnostics_q_max_abs_diff_le_1e-12": bool(np.max(np.abs(q_values - diag_q)) <= 1e-12),
            "full_worst_class_match": bool(np.array_equal(np.asarray(full_worst_ids, dtype=str), d.full_worst_class_id.astype(str).to_numpy())),
            "absent_worst_class_match": bool(np.array_equal(np.asarray(["None" if x is None else str(x) for x in absent_worst_ids]), d.absent_worst_class_id.fillna("None").astype(str).to_numpy())),
        }
        rows.append({
            "setting": name,
            "status": "PASS" if all(checks.values()) else "FAIL",
            "A_count": int(A.sum()),
            "A_full_failure_count": int((A & full_failure).sum()),
            "A_absent_failure_count": int((A & absent_failure).sum()),
            "Aplus_count": int(Aplus.sum()),
            "Aplus_absent_failure_count": int((Aplus & absent_failure).sum()),
            "exact_random_A": float(q_values[A].mean()),
            "max_q_diag_abs_diff": float(np.max(np.abs(q_values - diag_q))),
            "checks": json.dumps(checks, sort_keys=True),
        })
    out = pd.DataFrame(rows)
    out.to_csv(root / "results/independent_verification.csv", index=False)
    print(out.to_string(index=False))
    if not (out.status == "PASS").all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
