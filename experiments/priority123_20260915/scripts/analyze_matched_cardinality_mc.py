#!/usr/bin/env python3
"""Check exact matched-cardinality foil probabilities with seeded draws."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SETTINGS = ("coco_openai_b16", "coco_openai_b32", "voc2007_openai_b16", "voc2007_openai_b32")


def exact_q(n: int, k: int, m: int) -> float:
    if k == 0 or m == 0:
        return 0.0
    if k > n - m:
        return 1.0
    return 1.0 - math.comb(n - m, k) / math.comb(n, k)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--full-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed-count", type=int, default=100)
    args = p.parse_args()
    rows = []
    for setting in SETTINGS:
        failure_by_seed = []
        exact_values = []
        n_a = 0
        for path in sorted((args.full_root / setting / "traceable_batches").glob("*.npz")):
            with np.load(path, allow_pickle=False) as x:
                norm = np.asarray(x["response_norm"], dtype=float)
                cci = np.asarray(x["cci_region"], dtype=int)
                targets = [str(v) for v in x["target_id"].tolist()]
                classes = [str(v) for v in x["category_ids"].tolist()]
                for i, sample_index in enumerate(x["sample_index"].astype(int)):
                    target = targets[i]
                    target_pos = classes.index(target)
                    foil = np.asarray([j for j, value in enumerate(classes) if value != target], dtype=int)
                    target_value = float(norm[i, cci[i], target_pos])
                    bbox = float(x["bbox"][i, cci[i]])
                    pmean = target_value - float(norm[i, cci[i], foil].mean())
                    if bbox < 0.5 or pmean <= 0:
                        continue
                    n_a += 1
                    present = {str(v) for v in json.loads(str(x["annotated_category_ids"][i]))}
                    absent_count = sum(value not in present for value in classes if value != target)
                    n = len(foil)
                    m = int(np.sum(norm[i, cci[i], foil] > target_value))
                    exact_values.append(exact_q(n, absent_count, m))
                    failure_by_seed.append([])
                    for seed in range(args.seed_count):
                        rng = np.random.default_rng(np.random.SeedSequence([seed, int(sample_index)]))
                        chosen = rng.choice(foil, size=absent_count, replace=False)
                        failure_by_seed[-1].append(bool(np.any(norm[i, cci[i], chosen] > target_value)))
        by_seed = np.asarray(failure_by_seed, dtype=float).mean(axis=0)
        exact = np.asarray(exact_values, dtype=float)
        rows.append({
            "status": "local_rerun_traceable", "dataset_model": setting, "subset": "A",
            "n_A": n_a, "mc_seed_count": args.seed_count,
            "mc_mean_failure_rate_across_seeds": float(by_seed.mean()),
            "mc_sd_across_seeds": float(by_seed.std(ddof=1)),
            "exact_mean_q": float(exact.mean()), "mc_minus_exact": float(by_seed.mean() - exact.mean()),
            "max_abs_seed_level_mc_minus_exact": float(np.max(np.abs(by_seed - exact.mean()))),
            "mc_seed_rule": "SeedSequence([seed, sample_index])",
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
