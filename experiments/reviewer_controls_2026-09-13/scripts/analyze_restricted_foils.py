#!/usr/bin/env python3
"""Analyze fixed-region annotation-absent foils from full class responses."""
from __future__ import annotations

import argparse
import csv
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
SUBSETS = ("A", "Aplus")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--fixed-root", type=Path, required=True)
    p.add_argument("--bootstrap-count", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=1701)
    p.add_argument("--mc-seeds", type=int, default=100)
    return p.parse_args()


def percentile_ci(values: np.ndarray, count: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    # Chunk the image resamples so COCO's ~18k-image A set does not allocate a
    # multi-gigabyte (draws x images) integer matrix.
    boot = np.empty(count, dtype=float)
    chunk = 128
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        draws = rng.integers(0, len(values), size=(stop - start, len(values)))
        boot[start:stop] = values[draws].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, count: int, seed: int) -> tuple[float, float, float]:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = percentile_ci(diff, count, seed)
    return float(diff.mean()), lo, hi


def exact_q(n: int, k: int, m: int) -> float:
    if k == 0 or m == 0:
        return 0.0
    if k > n - m:
        return 1.0
    return 1.0 - math.comb(n - m, k) / math.comb(n, k)


def load_npz_and_fixed(root: Path, fixed_root: Path, name: str):
    npz_path = root / "responses" / name / f"{name}_cci_full_class_responses.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    archive = np.load(npz_path, allow_pickle=False)
    x = {key: archive[key] for key in archive.files}
    # Older completed response archives contain only the response/category
    # arrays; the per-image CCI metadata is equivalently recorded in the
    # adjacent metadata.csv.  Enrich the in-memory view without rewriting the
    # large archive or rerunning model inference.
    metadata_path = root / "responses" / name / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    for key in (
        "cci_region", "cci_bbox_precision", "cci_target_drop_raw",
        "cci_target_drop_norm", "cci_margin_norm", "cci_pmean_norm",
        "feasible_set_size",
    ):
        if key not in x:
            if key not in metadata.columns:
                raise KeyError(f"{name}: missing {key} in NPZ and metadata.csv")
            x[key] = metadata[key].to_numpy()
    fixed_path = fixed_root / "runs" / name / "per_image.csv"
    fixed = pd.read_csv(fixed_path)
    return x, fixed, npz_path, fixed_path


def get_category_index(x):
    return {int(v) if str(v).isdigit() else str(v): i for i, v in enumerate(x["category_ids"].tolist())}


def as_id(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v)
    return int(s) if s.isdigit() else s


def canonical_image_ids(values) -> np.ndarray:
    out = []
    for value in values:
        s = str(value)
        out.append(str(int(s)) if s.isdigit() else s)
    return np.asarray(out, dtype=str)


def calculate_setting(name: str, x, fixed: pd.DataFrame, bootstrap_count: int, bootstrap_seed: int, mc_seed_count: int):
    norm = np.asarray(x["response_norm"], dtype=np.float64)
    raw = np.asarray(x["response_raw"], dtype=np.float64)
    image_ids = canonical_image_ids(x["image_id"])
    sample_index = x["sample_index"].astype(np.int64)
    target_ids = np.asarray([as_id(v) for v in x["target_id"].tolist()], dtype=object)
    category_ids = [as_id(v) for v in x["category_ids"].tolist()]
    cat_index = {v: i for i, v in enumerate(category_ids)}
    if norm.ndim != 3 or norm.shape[1] != 8 or norm.shape[2] != len(category_ids):
        raise ValueError(f"{name}: unexpected response shape {norm.shape}")
    if len(fixed) != len(norm):
        raise ValueError(f"{name}: fixed rows {len(fixed)} != response rows {len(norm)}")
    if not np.array_equal(image_ids, canonical_image_ids(fixed.image_id.to_numpy())):
        raise ValueError(f"{name}: image_id order differs from fixed local rerun")
    if not np.array_equal(sample_index, fixed.sample_index.to_numpy(dtype=np.int64)):
        raise ValueError(f"{name}: sample_index order differs from fixed local rerun")

    target_pos = np.asarray([cat_index[t] for t in target_ids], dtype=np.int64)
    target_resp = norm[np.arange(len(norm))[:, None], np.arange(8)[None, :], target_pos[:, None]]
    full_mask = np.ones((len(norm), len(category_ids)), dtype=bool)
    full_mask[np.arange(len(norm)), target_pos] = False
    full_resp = np.where(full_mask[:, None, :], norm, -np.inf)
    full_worst_pos = full_resp.argmax(axis=2)
    full_worst = full_resp.max(axis=2)
    full_margin = target_resp - full_worst
    full_pmean = target_resp - np.where(full_mask[:, None, :], norm, 0.0).sum(axis=2) / (len(category_ids) - 1)
    generated_cci_region = np.asarray(x["cci_region"], dtype=np.int64)
    cci_region = fixed.cci_region.to_numpy(dtype=np.int64)
    row_idx = np.arange(len(norm))
    cci_margin = full_margin[row_idx, cci_region]
    cci_pmean = full_pmean[row_idx, cci_region]
    cci_target_norm = target_resp[row_idx, cci_region]
    cci_target_raw = raw[row_idx, cci_region, target_pos]
    bbox = fixed.cci_bbox_precision.to_numpy(dtype=float)
    # The original A is defined by the fixed full-foil CCI summary, not by the
    # restricted foil analysis.  We assert the stored summary agrees first.
    fixed_margin = fixed.cci_margin_norm.to_numpy(dtype=float)
    fixed_pmean = fixed.cci_pmean_norm.to_numpy(dtype=float)
    fixed_target_norm = fixed.cci_target_drop_norm.to_numpy(dtype=float)
    fixed_target_raw = fixed.cci_target_drop_raw.to_numpy(dtype=float)
    region_summary_diff = {
        "cci_margin_norm_max_abs_diff": float(np.max(np.abs(cci_margin - fixed_margin))),
        "cci_pmean_norm_max_abs_diff": float(np.max(np.abs(cci_pmean - fixed_pmean))),
        "cci_target_drop_norm_max_abs_diff": float(np.max(np.abs(cci_target_norm - fixed_target_norm))),
        "cci_target_drop_raw_max_abs_diff": float(np.max(np.abs(cci_target_raw - fixed_target_raw))),
        "cci_margin_norm_n_gt_1e-5": int(np.sum(np.abs(cci_margin - fixed_margin) > 1e-5)),
        "cci_pmean_norm_n_gt_1e-5": int(np.sum(np.abs(cci_pmean - fixed_pmean) > 1e-5)),
        "cci_target_drop_norm_n_gt_1e-5": int(np.sum(np.abs(cci_target_norm - fixed_target_norm) > 1e-5)),
        "cci_target_drop_raw_n_gt_1e-5": int(np.sum(np.abs(cci_target_raw - fixed_target_raw) > 1e-5)),
        "cci_region_mismatch": int(np.sum(generated_cci_region != cci_region)),
    }
    # The source runner's candidate margin summary is also checked where it is
    # available in the fixed CSV; this catches class-order or target-index errors.
    old_candidate_margin = fixed.candidate_eval_margin_norm.map(json.loads).to_list()
    candidate_margin_diff = []
    for i, vals in enumerate(old_candidate_margin):
        candidate_margin_diff.append(np.max(np.abs(np.asarray(vals, dtype=float) - full_margin[i])))
    region_summary_diff["candidate_margin_max_abs_diff"] = float(np.max(candidate_margin_diff))
    region_summary_diff["candidate_margin_n_gt_1e-5"] = int(np.sum(np.asarray(candidate_margin_diff) > 1e-5))
    for key, fixed_col in (
        ("cci_bbox_precision", "cci_bbox_precision"),
        ("cci_target_drop_raw", "cci_target_drop_raw"),
        ("cci_target_drop_norm", "cci_target_drop_norm"),
        ("cci_margin_norm", "cci_margin_norm"),
        ("cci_pmean_norm", "cci_pmean_norm"),
        ("feasible_set_size", "feasible_set_size"),
    ):
        d = np.abs(np.asarray(x[key], dtype=float) - fixed[fixed_col].to_numpy(dtype=float))
        region_summary_diff[f"generated_{key}_max_abs_diff"] = float(np.max(d))
        region_summary_diff[f"generated_{key}_n_gt_1e-5"] = int(np.sum(d > 1e-5))
    if any(region_summary_diff[k] > 1e-5 for k in region_summary_diff if k.endswith("max_abs_diff")):
        raise ValueError(f"{name}: generated full responses do not reproduce fixed full-foil summaries: {region_summary_diff}")

    annotated = []
    for value in x["annotated_category_ids"].astype(str).tolist():
        annotated.append({as_id(v) for v in json.loads(value)})
    absent_positions = []
    n_full, k_absent, m_outrank = [], [], []
    absent_worst_pos, absent_worst = [], []
    for i, target in enumerate(target_ids):
        non_target = [c for c in category_ids if c != target]
        absent = [c for c in non_target if c not in annotated[i]]
        positions = np.asarray([cat_index[c] for c in absent], dtype=np.int64)
        absent_positions.append(positions)
        n = len(non_target)
        k = len(absent)
        m = int(np.sum(norm[i, cci_region[i], [cat_index[c] for c in non_target]] > target_resp[i, cci_region[i]]))
        n_full.append(n)
        k_absent.append(k)
        m_outrank.append(m)
        if len(positions):
            vals = norm[i, cci_region[i], positions]
            j = int(np.argmax(vals))
            absent_worst_pos.append(int(positions[j]))
            absent_worst.append(float(vals[j]))
        else:
            absent_worst_pos.append(-1)
            absent_worst.append(float("nan"))
    n_full = np.asarray(n_full, dtype=np.int64)
    k_absent = np.asarray(k_absent, dtype=np.int64)
    m_outrank = np.asarray(m_outrank, dtype=np.int64)
    q = np.asarray([exact_q(int(n), int(k), int(m)) for n, k, m in zip(n_full, k_absent, m_outrank)], dtype=float)
    absent_worst = np.asarray(absent_worst, dtype=float)
    absent_failure = np.where(k_absent > 0, target_resp[row_idx, cci_region] - absent_worst < 0, False)
    full_failure = cci_margin < 0
    A = (bbox >= 0.5) & (fixed_pmean > 0)
    Aplus = A & (fixed_target_norm > 0)

    # Check the fixed counts before any restricted-foil result is emitted.
    expected = {
        "coco_openai_b16": (17726, 11410),
        "coco_openai_b32": (17847, 11561),
        "voc2007_openai_b16": (1209, 511),
        "voc2007_openai_b32": (1227, 505),
    }[name]
    actual = (int(A.sum()), int((A & full_failure).sum()))
    if actual != expected:
        raise ValueError(f"{name}: fixed A/full-failure count {actual} != expected {expected}")

    ids = category_ids
    names = [str(v) for v in x["category_names"].tolist()]
    diag = fixed[["status", "dataset", "model", "sample_index", "image_id", "path", "target_id", "target_name", "cci_region", "cci_bbox_precision", "cci_target_drop_raw", "cci_target_drop_norm", "cci_margin_norm", "cci_pmean_norm", "feasible_set_size"]].copy()
    diag["annotated_category_ids"] = [json.dumps(sorted(a, key=str), ensure_ascii=False) for a in annotated]
    diag["annotated_non_target_count"] = [len(a - {t}) for a, t in zip(annotated, target_ids)]
    diag["full_non_target_foil_count"] = n_full
    diag["annotation_absent_foil_count"] = k_absent
    diag["full_outranking_foil_count"] = m_outrank
    diag["exact_matched_random_failure_probability"] = q
    diag["full_failure"] = full_failure.astype(int)
    diag["annotation_absent_failure"] = absent_failure.astype(int)
    diag["A"] = A.astype(int)
    diag["Aplus"] = Aplus.astype(int)
    diag["full_worst_class_id"] = [ids[int(p)] for p in full_worst_pos[row_idx, cci_region]]
    diag["full_worst_class_name"] = [names[int(p)] for p in full_worst_pos[row_idx, cci_region]]
    diag["full_worst_response_norm"] = full_worst[row_idx, cci_region]
    diag["full_margin_recomputed_norm"] = cci_margin
    diag["absent_worst_class_id"] = [None if p < 0 else ids[p] for p in absent_worst_pos]
    diag["absent_worst_class_name"] = [None if p < 0 else names[p] for p in absent_worst_pos]
    diag["absent_worst_response_norm"] = absent_worst
    diag["absent_margin_recomputed_norm"] = target_resp[row_idx, cci_region] - absent_worst

    summary_rows = []
    control_rows = []
    plus_rows = []
    for subset_name, subset in (("A", A), ("Aplus", Aplus)):
        idx = np.flatnonzero(subset)
        f = full_failure[idx].astype(float)
        a = absent_failure[idx].astype(float)
        qq = q[idx]
        f_rate = float(f.mean())
        a_rate = float(a.mean())
        f_lo, f_hi = percentile_ci(f, bootstrap_count, bootstrap_seed)
        a_lo, a_hi = percentile_ci(a, bootstrap_count, bootstrap_seed + 1)
        diff_mean, diff_lo, diff_hi = paired_bootstrap(a, f, bootstrap_count, bootstrap_seed + 2)
        random_rate = float(qq.mean())
        random_lo, random_hi = percentile_ci(qq, bootstrap_count, bootstrap_seed + 3)
        delta_mean, delta_lo, delta_hi = paired_bootstrap(a, qq, bootstrap_count, bootstrap_seed + 4)
        row = {
            "setting": name, "subset": subset_name, "n": int(len(idx)),
            "full_failure_count": int(f.sum()), "annotation_absent_failure_count": int(a.sum()),
            "Pr_F_full_given_subset": f_rate, "Pr_F_full_CI_lo": f_lo, "Pr_F_full_CI_hi": f_hi,
            "Pr_F_absent_given_subset": a_rate, "Pr_F_absent_CI_lo": a_lo, "Pr_F_absent_CI_hi": a_hi,
            "absent_minus_full": diff_mean, "absent_minus_full_CI_lo": diff_lo, "absent_minus_full_CI_hi": diff_hi,
            "bootstrap_count": bootstrap_count, "bootstrap_unit": "image", "bootstrap_interval": "percentile 2.5/97.5",
            "bootstrap_seed_full": bootstrap_seed, "bootstrap_seed_absent": bootstrap_seed + 1,
            "bootstrap_seed_paired_absent_minus_full": bootstrap_seed + 2,
        }
        summary_rows.append(row)
        control_rows.append({
            "setting": name, "subset": subset_name, "n": int(len(idx)),
            "full_non_target_foil_n_mean": float(n_full[idx].mean()),
            "annotation_absent_foil_n_mean": float(k_absent[idx].mean()),
            "full_outranking_n_mean": float(m_outrank[idx].mean()),
            "exact_matched_random_failure_rate": random_rate,
            "exact_matched_random_CI_lo": random_lo, "exact_matched_random_CI_hi": random_hi,
            "absent_minus_matched_random": delta_mean,
            "absent_minus_matched_random_CI_lo": delta_lo,
            "absent_minus_matched_random_CI_hi": delta_hi,
            "bootstrap_count": bootstrap_count, "bootstrap_unit": "image", "bootstrap_interval": "percentile 2.5/97.5",
            "bootstrap_seed_random": bootstrap_seed + 3,
            "bootstrap_seed_paired_absent_minus_random": bootstrap_seed + 4,
        })
        plus_rows.append({
            "setting": name, "n_Aplus": int(len(idx)),
            "Aplus_full_failure_count": int(f.sum()), "Aplus_absent_failure_count": int(a.sum()),
            "Aplus_Pr_F_full": f_rate, "Aplus_Pr_F_full_CI_lo": f_lo, "Aplus_Pr_F_full_CI_hi": f_hi,
            "Aplus_Pr_F_absent": a_rate, "Aplus_Pr_F_absent_CI_lo": a_lo, "Aplus_Pr_F_absent_CI_hi": a_hi,
            "Aplus_exact_matched_random_rate": random_rate,
            "Aplus_absent_minus_random": delta_mean,
            "Aplus_absent_minus_random_CI_lo": delta_lo,
            "Aplus_absent_minus_random_CI_hi": delta_hi,
        })

    # Exact 100-seed sanity check, restricted to A as the prespecified primary set.
    mc_rates = []
    mc_seeds = []
    a_idx = np.flatnonzero(A)
    for seed in range(mc_seed_count):
        failure = []
        for i in a_idx:
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(sample_index[i])]))
            positions = np.flatnonzero(full_mask[i])
            chosen = rng.choice(positions, size=int(k_absent[i]), replace=False)
            failure.append(bool(np.any(norm[i, cci_region[i], chosen] > target_resp[i, cci_region[i]])))
        mc_rates.append(float(np.mean(failure)))
        mc_seeds.append(seed)
    mc_rates = np.asarray(mc_rates)
    mc_row = {
        "setting": name, "subset": "A", "mc_seed_count": mc_seed_count,
        "mc_seed_first": 0, "mc_seed_last": mc_seed_count - 1,
        "mc_mean_failure_rate_across_seeds": float(mc_rates.mean()),
        "mc_sd_across_seeds": float(mc_rates.std(ddof=1)),
        "exact_mean_q": float(q[a_idx].mean()),
        "mc_minus_exact": float(mc_rates.mean() - q[a_idx].mean()),
        "max_abs_seed_level_mc_minus_exact": float(np.max(np.abs(mc_rates - q[a_idx].mean()))),
    }
    return diag, summary_rows, control_rows, plus_rows, mc_row, region_summary_diff


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(rows, path: Path):
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    fixed_root = args.fixed_root.expanduser().resolve()
    results = root / "results"
    manifest = root / "manifest"
    results.mkdir(parents=True, exist_ok=True)
    manifest.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    failure_rows, control_rows, plus_rows, mc_rows, reproduction_rows = [], [], [], [], []
    input_paths = []
    for name, _, _ in SETTINGS:
        x, fixed, npz_path, fixed_path = load_npz_and_fixed(root, fixed_root, name)
        input_paths.extend([npz_path, fixed_path])
        out = calculate_setting(name, x, fixed, args.bootstrap_count, args.bootstrap_seed, args.mc_seeds)
        diag, s_rows, c_rows, p_rows, mc_row, rep = out
        diagnostics.append(diag)
        failure_rows.extend(s_rows)
        control_rows.extend(c_rows)
        plus_rows.extend(p_rows)
        mc_rows.append(mc_row)
        reproduction_rows.append({"setting": name, **rep})

    diag_df = pd.concat(diagnostics, ignore_index=True)
    write_csv(diag_df, results / "per_image_restricted_foil_diagnostics.csv")
    write_csv(failure_rows, results / "restricted_foil_failure_rates.csv")
    write_csv(control_rows + mc_rows, results / "matched_cardinality_control.csv")
    write_csv(plus_rows, results / "positive_target_subset.csv")

    strata = []
    diag_df["setting"] = diag_df["dataset"] + "_" + diag_df["model"]
    for setting, grp in diag_df[diag_df["A"] == 1].groupby("setting"):
        for label, mask in (("2_classes", grp["annotated_non_target_count"] == 1),
                            ("3_classes", grp["annotated_non_target_count"] == 2),
                            ("4plus_classes", grp["annotated_non_target_count"] >= 3)):
            if len(mask) == 0:
                continue
            strata.append({
                "setting": setting, "stratum": label, "n": int(mask.sum()),
                "Pr_F_full_given_A_stratum": float(grp.loc[mask, "full_failure"].mean()) if mask.sum() else float("nan"),
                "Pr_F_absent_given_A_stratum": float(grp.loc[mask, "annotation_absent_failure"].mean()) if mask.sum() else float("nan"),
                "exact_matched_random_rate": float(grp.loc[mask, "exact_matched_random_failure_probability"].mean()) if mask.sum() else float("nan"),
            })
    write_csv(strata, results / "annotated_class_count_strata.csv")
    write_csv(reproduction_rows, results / "full_summary_reproduction.csv")

    # Restore the setting label in the on-disk diagnostics after using it for strata.
    diag_df.to_csv(results / "per_image_restricted_foil_diagnostics.csv", index=False)
    cfg = {
        "analysis": "fixed full-foil CCI-top1 region and A; restricted annotation-absent foils",
        "bootstrap_count": args.bootstrap_count,
        "bootstrap_seed_base": args.bootstrap_seed,
        "bootstrap_interval": "percentile 2.5/97.5",
        "bootstrap_unit": "image",
        "mc_seed_count": args.mc_seeds,
        "mc_seed_rule": "np.random.default_rng(np.random.SeedSequence([seed, sample_index]))",
        "generated_on": platform.platform(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root.parent.parent, text=True).strip(),
    }
    (root / "analysis_metadata.json").write_text(json.dumps(cfg, indent=2) + "\n")

    # Hash inputs and every output except the manifest file being written.
    input_rows = [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in input_paths]
    write_csv(input_rows, manifest / "input_hashes.csv")
    large_rows = []
    for p in sorted((root / "responses").glob("*/*")):
        if p.is_file() and p.suffix in {".npz", ".npy"}:
            large_rows.append({
                "path": str(p.relative_to(root)), "sha256": sha256(p), "bytes": p.stat().st_size,
                "github_upload": False,
                "reason": "response tensor retained locally; portable per-image diagnostics are uploaded",
            })
    write_csv(large_rows, manifest / "large_local_artifacts.csv")
    output_files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "output_hashes.csv")
    output_rows = [{"path": str(p.relative_to(root)), "sha256": sha256(p), "bytes": p.stat().st_size} for p in output_files]
    write_csv(output_rows, manifest / "output_hashes.csv")
    (manifest / "environment.txt").write_text(
        subprocess.check_output([str(Path("/Users/von/Projects/cci-local-recovery/.venv-local-rerun/bin/python")), "-m", "pip", "freeze"], text=True)
    )
    print(json.dumps({
        "status": "complete_analysis",
        "settings": [x[0] for x in SETTINGS],
        "diagnostic_rows": len(diag_df),
        "results_dir": str(results),
        "reproduction": reproduction_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
