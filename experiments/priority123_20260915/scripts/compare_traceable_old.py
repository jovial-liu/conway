#!/usr/bin/env python3
"""Compare new traceable per-image exports with old local-rerun CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SETTINGS = ("coco_openai_b16", "coco_openai_b32", "voc2007_openai_b16", "voc2007_openai_b32")
STATE_FIELDS = ("cci_region", "wf_region", "mean_region", "max_0_1_region", "feasible_set_size", "A", "Aplus", "B", "C0", "C1", "C2", "repair")
METRIC_FIELDS = ("cci_bbox_precision", "cci_pmean_norm", "cci_margin_norm", "cci_target_drop_raw", "cci_target_drop_norm")
ARRAY_FIELDS = ("candidate_selection_target", "candidate_selection_mean_foil", "candidate_selection_max_foil", "candidate_eval_margin_norm", "candidate_eval_pmean_norm", "candidate_eval_target_drop_raw", "candidate_eval_target_drop_norm", "candidate_bbox_precision")


def states(row: pd.Series) -> dict:
    cci, wf = int(row.cci_region), int(row.wf_region)
    selection = np.asarray(json.loads(row.candidate_selection_target), dtype=float)
    max_foil = np.asarray(json.loads(row.candidate_selection_max_foil), dtype=float)
    margin = np.asarray(json.loads(row.candidate_eval_margin_norm), dtype=float)
    feasible = selection >= selection[cci] - 0.02
    passing = margin >= 0
    a = bool(row.cci_bbox_precision >= 0.5 and row.cci_pmean_norm > 0)
    ap = bool(a and row.cci_target_drop_norm > 0)
    b = bool(a and row.cci_margin_norm < 0)
    return {
        "cci_region": cci, "wf_region": wf, "mean_region": int(row.mean_region),
        "max_0_1_region": int(row.max_0_1_region), "feasible_set_size": int(row.feasible_set_size),
        "A": int(a), "Aplus": int(ap), "B": int(b),
        "C0": int(b and not passing.any()),
        "C1": int(b and passing.any() and not (passing & feasible).any()),
        "C2": int(b and (passing & feasible).any() and not passing[wf]),
        "repair": int(b and passing[wf]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--new-root", type=Path, required=True, help="directory containing per-image/<setting>.csv")
    p.add_argument("--old-root", type=Path, required=True, help="directory containing <setting>/per_image.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for setting in SETTINGS:
        new = pd.read_csv(args.new_root / "per_image" / f"{setting}.csv", low_memory=False).sort_values("sample_index").reset_index(drop=True)
        old = pd.read_csv(args.old_root / setting / "per_image.csv", low_memory=False).sort_values("sample_index").reset_index(drop=True)
        if len(new) != len(old):
            raise ValueError(f"{setting}: row count differs ({len(old)} vs {len(new)})")
        old_states = [states(old.iloc[i]) for i in range(len(old))]
        changed_rows = []
        state_counts = {field: 0 for field in STATE_FIELDS}
        numeric_counts = {field: 0 for field in METRIC_FIELDS}
        numeric_max = {field: 0.0 for field in METRIC_FIELDS}
        for i in range(len(new)):
            nr, old_row, old_state = new.iloc[i], old.iloc[i], old_states[i]
            change_types = []
            for field in STATE_FIELDS:
                if int(nr[field]) != int(old_state[field]):
                    change_types.append(field); state_counts[field] += 1
            for field in METRIC_FIELDS:
                diff = abs(float(nr[field]) - float(old_row[field]))
                numeric_max[field] = max(numeric_max[field], diff)
                if diff > 1e-5:
                    numeric_counts[field] += 1
            if not change_types:
                continue
            out = {"dataset_model": setting, "sample_index": int(nr.sample_index), "image_id": str(nr.image_id), "target_id": str(nr.target_id), "change_types": ";".join(change_types), "epsilon": 0.02, "bbox_threshold": 0.5, "pmean_threshold": 0.0, "margin_threshold": 0.0}
            for field in STATE_FIELDS:
                out[f"old_{field}"] = old_state[field]; out[f"new_{field}"] = nr[field]
            for field in METRIC_FIELDS:
                out[f"old_{field}"] = old_row[field]; out[f"new_{field}"] = nr[field]
            for field in ARRAY_FIELDS:
                out[f"old_{field}"] = old_row[field]; out[f"new_{field}"] = nr[field]
            changed_rows.append(out)
        pd.DataFrame(changed_rows).to_csv(args.out_dir / f"{setting}.csv", index=False)
        summary.append({"dataset_model": setting, "n_images": len(new), "state_changed_rows": len(changed_rows), **{f"{k}_changes": v for k, v in state_counts.items()}, **{f"{k}_numeric_changes_gt_1e-5": v for k, v in numeric_counts.items()}, **{f"{k}_max_abs_diff": v for k, v in numeric_max.items()}})
    pd.DataFrame(summary).to_csv(args.out_dir / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
