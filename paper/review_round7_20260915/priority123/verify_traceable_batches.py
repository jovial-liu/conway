"""Independently verify same-generation traceable response batches.

The verifier does not import the capture or analysis module. It checks the
stored masks, geometry, selection scores, held-out template aggregation,
normalization, candidate choices, screening strata, and complete batch order.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def canonical_id(value) -> str:
    text = str(value)
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return str(int(text)) if text.isdigit() else text


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exact_q(n, k, m):
    if k == 0 or m == 0:
        return 0.0
    if k > n - m:
        return 1.0
    return 1.0 - math.comb(n - m, k) / math.comb(n, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.setting_dir.expanduser().resolve()
    out = args.out.expanduser().resolve()

    sums = root / "SHA256SUMS.txt"
    if not sums.exists():
        raise FileNotFoundError(sums)
    for line in sums.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        actual = digest(root / relative)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")

    meta = pd.read_csv(root / "metadata.csv", low_memory=False).sort_values("sample_index").reset_index(drop=True)
    npz_paths = sorted(root.glob("*_cci_full_class_responses.npz"))
    if len(npz_paths) != 1:
        raise ValueError(f"expected one complete NPZ, found {len(npz_paths)}")
    with np.load(npz_paths[0], allow_pickle=False) as z:
        required = {"response_raw", "response_norm", "image_id", "sample_index", "target_id", "category_ids", "category_names"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"NPZ missing fields: {sorted(missing)}")
        raw_full = np.asarray(z["response_raw"])
        norm_full = np.asarray(z["response_norm"])
        classes = [canonical_id(v) for v in z["category_ids"].tolist()]
        category_names = [str(v) for v in z["category_names"].tolist()]
        category_index = {value: i for i, value in enumerate(classes)}
        npz_ids = [canonical_id(v) for v in z["image_id"].tolist()]
        npz_indices = np.asarray(z["sample_index"], dtype=int)

        rows = []
        seen = []
        for batch_path in sorted((root / "traceable_batches").glob("*.npz")):
            with np.load(batch_path, allow_pickle=False) as batch:
                ids = np.asarray(batch["sample_index"], dtype=int)
                seen.extend(ids.tolist())
                masks = np.asarray(batch["masks"])
                inside = np.asarray(batch["inside"], dtype=bool)
                bbox = np.asarray(batch["bbox"], dtype=float)
                raw = np.asarray(batch["response_raw"], dtype=float)
                norm = np.asarray(batch["response_norm"], dtype=float)
                per_prompt = np.asarray(batch["response_per_prompt"], dtype=float)
                text_norm = np.asarray(batch["text_norm"], dtype=float)
                selection_all = np.asarray(batch["selection_all"], dtype=float)
                selection_target = np.asarray(batch["selection_target"], dtype=float)
                if not np.isin(masks, [0, 1]).all() or not np.all(masks.sum(axis=1) == 1) or not np.all(masks.sum(axis=2) > 0):
                    raise ValueError(f"invalid patch partition in {batch_path.name}")
                if masks.shape[2] != inside.shape[1]:
                    raise ValueError(f"inside/mask patch count mismatch in {batch_path.name}")
                bbox_recomputed = (masks * inside[:, None, :]).sum(axis=2) / masks.sum(axis=2)
                if not np.allclose(bbox_recomputed, bbox, atol=1e-6, rtol=0):
                    raise ValueError(f"bbox mismatch in {batch_path.name}")
                if not np.allclose(per_prompt.mean(axis=2), raw, atol=1e-5, rtol=0):
                    raise ValueError(f"held-out prompt aggregation mismatch in {batch_path.name}")
                if not np.allclose(raw / text_norm[None, None, :], norm, atol=1e-5, rtol=0):
                    raise ValueError(f"class normalization mismatch in {batch_path.name}")
                if not np.array_equal(raw, raw_full[ids]) or not np.array_equal(norm, norm_full[ids]):
                    raise ValueError(f"batch/full NPZ response mismatch in {batch_path.name}")
                if not np.array_equal(npz_indices[ids], ids):
                    raise ValueError(f"NPZ sample index mismatch in {batch_path.name}")
                if "category_ids" in batch.files and [canonical_id(v) for v in batch["category_ids"].tolist()] != classes:
                    raise ValueError(f"category order mismatch in {batch_path.name}")

                for j, image_index in enumerate(ids.tolist()):
                    row = meta.iloc[image_index]
                    if canonical_id(row.image_id) != canonical_id(batch["image_id"][j]):
                        raise ValueError(f"image ID mismatch at sample {image_index}")
                    target_id = canonical_id(row.target_id)
                    target_pos = category_index[target_id]
                    if "target_id" in batch.files and canonical_id(batch["target_id"][j]) != target_id:
                        raise ValueError(f"target ID mismatch at sample {image_index}")
                    # The capture intentionally follows the archived runner:
                    # target selection uses elementwise multiplication while
                    # the full selection matrix uses einsum. They are
                    # mathematically identical but are evaluated by two
                    # different float32 reduction paths.  The relative term
                    # is needed for large scores (one observed VOC B/16
                    # value is 9.53 and differs by 1.14e-5); the check remains
                    # tight enough to detect class/order or response errors.
                    if not np.isclose(selection_target[j], selection_all[j, :, target_pos], atol=2e-5, rtol=2e-6).all():
                        raise ValueError(f"selection target mismatch at sample {image_index}")

                    foil = np.arange(len(classes)) != target_pos
                    target_response = norm[j, :, target_pos]
                    margin = target_response - norm[j][:, foil].max(axis=1)
                    mean_margin = target_response - norm[j][:, foil].mean(axis=1)
                    cci = int(np.argmax(selection_target[j]))
                    feasible = selection_target[j] >= selection_target[j, cci] - 0.02
                    selection_max_foil = selection_all[j].copy()
                    selection_max_foil[:, target_pos] = -np.inf
                    selection_mean_foil = selection_all[j].copy()
                    selection_mean_foil[:, target_pos] = 0.0
                    wf = int(np.argmax(np.where(feasible, selection_target[j] - selection_max_foil.max(axis=1), -np.inf)))
                    mean = int(np.argmax(np.where(feasible, selection_target[j] - selection_mean_foil.sum(axis=1) / (len(classes) - 1), -np.inf)))
                    max_0_1 = int(np.argmax(np.where(feasible, selection_target[j] - 0.1 * selection_max_foil.max(axis=1), -np.inf)))
                    A = bool(bbox[j, cci] >= 0.5 and mean_margin[cci] > 0)
                    Aplus = bool(A and target_response[cci] > 0)
                    B = bool(A and margin[cci] < 0)
                    passing = margin >= 0
                    C0 = bool(B and not passing.any())
                    C1 = bool(B and passing.any() and not (passing & feasible).any())
                    C2 = bool(B and (passing & feasible).any() and not passing[wf])
                    repair = bool(B and passing[wf])

                    present = {canonical_id(v) for v in json.loads(str(row.annotated_category_ids))}
                    absent = np.asarray([(c not in present and c != target_id) for c in classes], dtype=bool)
                    if not absent.any():
                        raise ValueError(f"no annotation-absent foils at sample {image_index}")
                    absent_margin = target_response[cci] - norm[j, cci, absent].max()
                    n = int(foil.sum())
                    k = int(absent.sum())
                    m = int((norm[j, cci, foil] > target_response[cci]).sum())
                    q = exact_q(n, k, m)

                    stored = {
                        "cci_region": cci, "wf_region": wf, "mean_region": mean, "max_0_1_region": max_0_1,
                        "A": int(A), "Aplus": int(Aplus), "B": int(B), "C0": int(C0), "C1": int(C1), "C2": int(C2), "repair": int(repair),
                    }
                    for key, value in stored.items():
                        if key in row.index and int(row[key]) != value:
                            raise ValueError(f"stored {key} mismatch at sample {image_index}")
                    if "feasible_mask" in row.index and not np.array_equal(np.asarray(json.loads(row.feasible_mask), dtype=int), feasible.astype(int)):
                        raise ValueError(f"stored feasible mask mismatch at sample {image_index}")
                    if "cci_region" in batch.files and int(batch["cci_region"][j]) != cci:
                        raise ValueError(f"batch CCI mismatch at sample {image_index}")
                    if "feasible_mask" in batch.files and not np.array_equal(batch["feasible_mask"][j], feasible.astype(np.uint8)):
                        raise ValueError(f"batch feasible mask mismatch at sample {image_index}")
                    if B and sum(map(int, [C0, C1, C2, repair])) != 1:
                        raise ValueError(f"failure partition mismatch at sample {image_index}")
                    rows.append({
                        "sample_index": image_index, "image_id": row.image_id, "target_id": target_id,
                        "A": int(A), "Aplus": int(Aplus), "B": int(B), "C0": int(C0), "C1": int(C1), "C2": int(C2), "r": int(repair),
                        "full_failure": int(margin[cci] < 0), "absent_failure": int(absent_margin < 0),
                        "exact_matched_random_failure_probability": q, "cci_region": cci,
                        "wf_region": wf, "mean_region": mean, "max_0_1_region": max_0_1,
                        "cci_bbox_precision": bbox[j, cci], "wf_bbox_precision": bbox[j, wf],
                        "feasible_set_size": int(feasible.sum()), "n_full_foils": n, "n_absent_foils": k, "m_outranking_foils": m,
                    })

    if seen != list(range(len(meta))):
        raise ValueError("missing, duplicate, or reordered capture batches")
    d = pd.DataFrame(rows).sort_values("sample_index").reset_index(drop=True)
    if len(d) != len(meta) or not np.array_equal(d.sample_index.to_numpy(), np.arange(len(meta))):
        raise ValueError("diagnostic row count/order mismatch")
    screened = d[d.A == 1]
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "same_record_diagnostics.csv", index=False)
    result = {
        "status": "PASS", "n": len(d), "A": len(screened),
        "Aplus": int(d.Aplus.sum()), "B": int(d.B.sum()),
        "C0": int(d.C0.sum()), "C1": int(d.C1.sum()), "C2": int(d.C2.sum()), "r": int(d.r.sum()),
        "full_failure_pct_given_A": float(screened.full_failure.mean() * 100) if len(screened) else None,
        "annotation_absent_failure_pct_given_A": float(screened.absent_failure.mean() * 100) if len(screened) else None,
        "exact_matched_random_failure_pct_given_A": float(screened.exact_matched_random_failure_probability.mean() * 100) if len(screened) else None,
        "scope": "same-generation traceable records; complete CIs and cross-table analysis are produced separately",
    }
    (out / "same_record_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
