"""Audit raw versus normalized responses on the fixed original normalized A.

This audit never infers per-template or per-candidate feasibility from an
average response. It uses complete response arrays and the original per-image
diagnostics; if an archive omits duplicate CCI metadata, the diagnostics are
used explicitly and the provenance records that fact.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SETTINGS = [
    "coco_openai_b16",
    "coco_openai_b32",
    "voc2007_openai_b16",
    "voc2007_openai_b32",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_id(value) -> str:
    text = str(value)
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return str(int(text)) if text.isdigit() else text


def canonical_ids(values) -> np.ndarray:
    return np.asarray([canonical_id(value) for value in values], dtype=str)


def metric_arrays(values, target_pos, regions):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if values.ndim != 3 or values.shape[1] != 8:
        raise ValueError(f"unexpected response shape {values.shape}")
    rows = np.arange(n)
    target = np.stack([values[i, :, target_pos[i]] for i in rows], axis=0)
    foil = values.copy()
    foil[rows, :, target_pos] = -np.inf
    margin = target - foil.max(axis=2)
    worst_pos = foil.argmax(axis=2)
    cci_margin = margin[rows, regions]
    cci_worst_pos = worst_pos[rows, regions]
    return target, margin, worst_pos, cci_margin, cci_worst_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-root", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    diagnostics = pd.read_csv(args.diagnostics, low_memory=False)
    manifest = pd.read_csv(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    audit_rows = []
    provenance = {
        "status": "computed",
        "scope": "fixed original normalized A and original CCI region",
        "oracle_scope": "unrestricted candidate pass capacity evaluated on fixed original normalized B",
        "feasible_set_size_used_as_per_candidate_record": False,
        "inputs": {
            "diagnostics": {"path": str(args.diagnostics), "sha256": sha(args.diagnostics)},
            "manifest": {"path": str(args.manifest), "sha256": sha(args.manifest)},
        },
        "settings": {},
    }
    input_hash_rows = [
        {"path": str(args.diagnostics), "sha256": sha(args.diagnostics), "bytes": args.diagnostics.stat().st_size,
         "expected_sha256": "", "match": True},
        {"path": str(args.manifest), "sha256": sha(args.manifest), "bytes": args.manifest.stat().st_size,
         "expected_sha256": "", "match": True},
    ]

    for setting in SETTINGS:
        filename = f"{setting}_cci_full_class_responses.npz"
        path = args.responses_root / setting / filename
        if not path.exists():
            raise FileNotFoundError(path)
        expected_rows = manifest[manifest.path.astype(str).str.endswith("/" + filename)]
        if len(expected_rows) != 1:
            raise ValueError(f"manifest must contain exactly one row for {filename}")
        expected_sha = str(expected_rows.iloc[0].sha256)
        actual_sha = sha(path)
        if actual_sha != expected_sha:
            raise ValueError(f"SHA mismatch for {setting}: {actual_sha} != {expected_sha}")
        input_hash_rows.append({
            "path": str(path), "sha256": actual_sha, "bytes": path.stat().st_size,
            "expected_sha256": expected_sha, "match": True,
        })

        d = diagnostics[diagnostics.setting == setting].sort_values("sample_index").reset_index(drop=True)
        with np.load(path, allow_pickle=False) as archive:
            raw = np.asarray(archive["response_raw"], dtype=np.float64)
            norm = np.asarray(archive["response_norm"], dtype=np.float64)
            if raw.shape != norm.shape:
                raise ValueError(f"{setting}: raw/norm shape mismatch")
            ids = canonical_ids(archive["image_id"])
            if len(d) != len(ids) or not np.array_equal(d.sample_index.to_numpy(dtype=int), np.arange(len(ids))):
                raise ValueError(f"{setting}: sample index mismatch")
            if not np.array_equal(canonical_ids(d.image_id), ids):
                raise ValueError(f"{setting}: image order mismatch")

            categories = [canonical_id(value) for value in archive["category_ids"].tolist()]
            category_index = {value: i for i, value in enumerate(categories)}
            targets = canonical_ids(archive["target_id"])
            diag_targets = canonical_ids(d.target_id)
            if not np.array_equal(targets, diag_targets):
                raise ValueError(f"{setting}: target order mismatch")
            target_pos = np.asarray([category_index[value] for value in targets], dtype=int)

            if "cci_region" in archive.files:
                regions = np.asarray(archive["cci_region"], dtype=int)
                region_source = "response_npz"
                if not np.array_equal(regions, d.cci_region.to_numpy(dtype=int)):
                    raise ValueError(f"{setting}: CCI region mismatch between NPZ and diagnostics")
            else:
                regions = d.cci_region.to_numpy(dtype=int)
                region_source = "diagnostics_csv_fallback_npz_field_absent"

            raw_target, raw_margin, raw_worst, raw_cci_margin, raw_cci_worst = metric_arrays(raw, target_pos, regions)
            norm_target, norm_margin, norm_worst, norm_cci_margin, norm_cci_worst = metric_arrays(norm, target_pos, regions)
            if not np.allclose(norm_cci_margin, d.full_margin_recomputed_norm.to_numpy(float), atol=1e-5, rtol=0):
                raise ValueError(f"{setting}: normalized CCI margin does not reproduce diagnostics")

            A = d.A.to_numpy(dtype=int) == 1
            raw_failure = raw_cci_margin < 0
            norm_failure = norm_cci_margin < 0
            B = A & norm_failure
            raw_any_pass = (raw_margin >= 0).any(axis=1)
            norm_any_pass = (norm_margin >= 0).any(axis=1)
            summary = {
                "setting": setting,
                "n_images": int(len(ids)),
                "n_A": int(A.sum()),
                "raw_mean_failure_pct": float(raw_failure[A].mean() * 100),
                "normalized_mean_failure_pct": float(norm_failure[A].mean() * 100),
                "sign_flip_pct": float((raw_failure[A] != norm_failure[A]).mean() * 100),
                "raw_pass_normalized_fail_pct": float((~raw_failure[A] & norm_failure[A]).mean() * 100),
                "raw_fail_normalized_pass_pct": float((raw_failure[A] & ~norm_failure[A]).mean() * 100),
                "hardest_foil_agreement_pct": float((raw_cci_worst[A] == norm_cci_worst[A]).mean() * 100),
                "fixed_normalized_B_n": int(B.sum()),
                "raw_unrestricted_oracle_pass_pct_of_fixed_normalized_B": float(raw_any_pass[B].mean() * 100) if B.any() else None,
                "normalized_unrestricted_oracle_pass_pct_of_fixed_normalized_B": float(norm_any_pass[B].mean() * 100) if B.any() else None,
                "cci_region_source": region_source,
                "response_sha256": actual_sha,
            }
            summaries.append(summary)
            provenance["settings"][setting] = {
                "file": filename, "sha256": actual_sha, "rows": len(ids),
                "category_count": len(categories), "category_order": categories,
                "cci_region_source": region_source,
            }

            category_names = [str(value) for value in archive["category_names"].tolist()]
            for i in range(len(ids)):
                audit_rows.append({
                    "setting": setting, "sample_index": int(i), "image_id": str(d.image_id.iloc[i]),
                    "target_id": str(d.target_id.iloc[i]), "target_name": str(d.target_name.iloc[i]),
                    "cci_region": int(regions[i]), "A": int(A[i]), "B_fixed_normalized": int(B[i]),
                    "raw_failure": int(raw_failure[i]), "normalized_failure": int(norm_failure[i]),
                    "raw_pass_normalized_fail": int((not raw_failure[i]) and norm_failure[i]),
                    "raw_fail_normalized_pass": int(raw_failure[i] and (not norm_failure[i])),
                    "raw_cci_margin": float(raw_cci_margin[i]), "normalized_cci_margin": float(norm_cci_margin[i]),
                    "raw_worst_class_id": categories[int(raw_cci_worst[i])],
                    "raw_worst_class_name": category_names[int(raw_cci_worst[i])],
                    "normalized_worst_class_id": categories[int(norm_cci_worst[i])],
                    "normalized_worst_class_name": category_names[int(norm_cci_worst[i])],
                    "hardest_foil_agreement": int(raw_cci_worst[i] == norm_cci_worst[i]),
                    "raw_unrestricted_oracle_pass": int(raw_any_pass[i]),
                    "normalized_unrestricted_oracle_pass": int(norm_any_pass[i]),
                })

    pd.DataFrame(summaries).to_csv(args.out / "normalization_fixed_A.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.out / "normalization_per_image_audit.csv", index=False)
    pd.DataFrame(input_hash_rows).to_csv(args.out / "input_hashes.csv", index=False)
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# 归一化敏感性分析结论",
        "",
        "分析固定原始 normalized A、图像集合与 CCI 区域；未按 raw 结果重新筛选人群。",
        "raw 与 normalized 的失败率、符号翻转方向、hardest-foil 一致率及固定 normalized B 上的 unrestricted oracle pass capacity 均由完整逐图响应直接计算。",
        "没有将 feasible_set_size 解释为逐候选可行性，也没有从平均响应倒推逐模板结果。",
        "",
    ]
    for row in summaries:
        lines.append(
            f"- {row['setting']}: A={row['n_A']}; raw failure={row['raw_mean_failure_pct']:.4f}%; "
            f"normalized failure={row['normalized_mean_failure_pct']:.4f}%; sign flip={row['sign_flip_pct']:.4f}% "
            f"(raw pass→norm fail {row['raw_pass_normalized_fail_pct']:.4f}%, raw fail→norm pass {row['raw_fail_normalized_pass_pct']:.4f}%); "
            f"hardest-foil agreement={row['hardest_foil_agreement_pct']:.4f}%; "
            f"fixed normalized B={row['fixed_normalized_B_n']}, "
            f"raw/norm oracle pass={row['raw_unrestricted_oracle_pass_pct_of_fixed_normalized_B']}/"
            f"{row['normalized_unrestricted_oracle_pass_pct_of_fixed_normalized_B']}%."
        )
    (args.out / "conclusion_zh.md").write_text("\n".join(lines) + "\n")
    print(args.out / "normalization_fixed_A.csv")


if __name__ == "__main__":
    main()
