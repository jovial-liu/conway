#!/usr/bin/env python3
"""Regenerate fixed local-rerun responses with all held-out class values.

The protocol implementation is imported from the pinned local-rerun runner.  This
script only adds storage for the per-candidate, per-class held-out responses that
the original per-image CSV did not retain.  It never writes into the frozen or
local-rerun result directories.
"""
from __future__ import annotations

import hashlib
import argparse
import csv
import gc
import importlib.util
import json
import logging
import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve()
REPO = next(p for p in HERE.parents if (p / 'experiments/local_rerun_2026-09-13').exists())
BASE_PATH = REPO / "experiments/local_rerun_2026-09-13/rerun_workspace/scripts/run_local_rerun.py"
SPEC = importlib.util.spec_from_file_location("pinned_local_rerun", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import pinned runner: {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("coco", "voc2007"), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model", choices=("openai_b16", "openai_b32"), required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--samples", type=int, default=0, help="0 means all eligible samples")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def json_line(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def open_memmap(path: Path, shape: tuple[int, ...], dtype=np.float32):
    if path.exists():
        arr = np.lib.format.open_memmap(path, mode="r+")
        if tuple(arr.shape) != tuple(shape) or arr.dtype != np.dtype(dtype):
            raise RuntimeError(f"shape/dtype mismatch for {path}: {arr.shape}/{arr.dtype}, expected {shape}/{dtype}")
        return arr
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    arr[:] = np.nan
    arr.flush()
    return arr


def response_batch(model, device, samples, category_index, text_sel, text_eval, logit_scale,
                   indices, images, cluster_masks, inside, original):
    try:
        after = BASE.encode_masked(model, images, cluster_masks.to(device))
    except RuntimeError as exc:
        if device.type != "mps":
            raise
        logging.warning("MPS masked attention failed; retrying batch on CPU: %s", exc)
        model.to("cpu")
        device = torch.device("cpu")
        images = images.to(device)
        cluster_masks = cluster_masks.to(device)
        original, patches = BASE.visual_patch_tokens(model, images)
        after = BASE.encode_masked(model, images, cluster_masks)

    scale = logit_scale.to(after.device)
    target_ids = torch.tensor(
        [category_index[samples[int(i)]["target_id"]] for i in indices.tolist()],
        dtype=torch.long,
        device=after.device,
    )
    eval_emb = text_eval.to(after.device)
    diff = original[:, None, :] - after
    eval_cat_raw = scale * torch.einsum("bkd,pcd->bkpc", diff, eval_emb).mean(2)
    class_norm = eval_emb.mean(0).norm(dim=-1)
    eval_cat_norm = eval_cat_raw / class_norm[None, None, :]

    sel_target = text_sel[target_ids]
    sel_drop = scale * (diff * sel_target[:, None, :]).sum(2)
    cci = sel_drop.argmax(1)
    rows = torch.arange(len(indices), device=after.device)
    cci_target = sel_drop[rows, cci]
    feasible = sel_drop >= (cci_target[:, None] - 0.02)

    target_mask = F.one_hot(target_ids, num_classes=eval_cat_norm.shape[2]).bool()[:, None, :]
    foil_norm = eval_cat_norm.masked_fill(target_mask, float("-inf"))
    margin_norm = eval_cat_norm.gather(2, target_ids[:, None, None].expand(-1, eval_cat_norm.shape[1], 1)).squeeze(2) - foil_norm.max(2).values
    foil_sum = eval_cat_norm.masked_fill(target_mask, 0.0)
    pmean_norm = eval_cat_norm.gather(2, target_ids[:, None, None].expand(-1, eval_cat_norm.shape[1], 1)).squeeze(2) - foil_sum.sum(2) / (eval_cat_norm.shape[2] - 1)
    bbox = inside[indices].to(after.device).float()[:, None, :].mul(cluster_masks.to(after.device).float()).sum(2)
    bbox /= cluster_masks.to(after.device).float().sum(2).clamp_min(1)

    rows_out = []
    for local, global_i in enumerate(indices.tolist()):
        sample = samples[global_i]
        r = int(cci[local].item())
        rows_out.append({
            "sample_index": global_i,
            "image_id": sample["image_id"],
            "path": sample["path"],
            "target_id": sample["target_id"],
            "target_name": sample["target_name"],
            "annotated_category_ids": json_line([sample["target_id"], *sample["distractor_ids"]]),
            "cci_region": r,
            "cci_bbox_precision": float(bbox[local, r].detach().cpu()),
            "cci_target_drop_raw": float(eval_cat_raw[local, r, target_ids[local]].detach().cpu()),
            "cci_target_drop_norm": float(eval_cat_norm[local, r, target_ids[local]].detach().cpu()),
            "cci_margin_norm": float(margin_norm[local, r].detach().cpu()),
            "cci_pmean_norm": float(pmean_norm[local, r].detach().cpu()),
            "feasible_set_size": int(feasible[local].sum().item()),
        })
    record = dict(inside=inside[indices].detach().cpu().numpy(), masks=cluster_masks.detach().cpu().numpy().astype(np.uint8),
                  bbox=bbox.detach().cpu().numpy(), selection_target=sel_drop.detach().cpu().numpy(),
                  selection_all=(scale * torch.einsum('bkd,cd->bkc', diff, text_sel.to(after.device))).detach().cpu().numpy(),
                  response_per_prompt=(scale * torch.einsum('bkd,pcd->bkpc',diff,eval_emb)).detach().cpu().numpy(),
                  text_norm=class_norm.detach().cpu().numpy())
    return rows_out, eval_cat_raw.detach().cpu().numpy(), eval_cat_norm.detach().cpu().numpy(), device, record


def run(args: argparse.Namespace) -> None:
    if args.resume:
        raise RuntimeError('Traceable capture requires a fresh complete run; resume disabled.')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError('Use a new empty output directory; archived runs are immutable.')
    args.data_root = args.data_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.out_dir / "run_full_class_responses.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    samples, categories = BASE.load_samples(args.dataset, args.data_root)
    if args.samples:
        samples = samples[:args.samples]
    if not samples:
        raise RuntimeError("no eligible samples; dataset annotations/images are incomplete")

    requested_device = args.device
    device = BASE.device_for(requested_device)
    model, tokenizer, preprocess, openclip_name = BASE.load_model(args.model, device, args.weights_dir)
    logit_scale = model.logit_scale.exp().float().detach().to(device)
    category_ids = sorted(categories)
    category_index = {category_id: i for i, category_id in enumerate(category_ids)}
    sel_text = BASE.text_features(model, tokenizer, [f"a photo of a {categories[c]}" for c in category_ids], device)
    eval_prompts = [p.format(category=categories[c]) for p in BASE.EVAL_PROMPTS for c in category_ids]
    eval_text = BASE.text_features(model, tokenizer, eval_prompts, device).reshape(3, len(category_ids), -1)

    patch_grid = 14 if args.model == "openai_b16" else 7
    inside = BASE.transformed_masks(samples, patch_grid * patch_grid)
    n, class_count = len(samples), len(category_ids)
    raw_path = args.out_dir / "response_raw.npy"
    norm_path = args.out_dir / "response_norm.npy"
    raw = open_memmap(raw_path, (n, 8, class_count))
    norm = open_memmap(norm_path, (n, 8, class_count))
    metadata_path = args.out_dir / "metadata.csv"
    progress_path = args.out_dir / "progress.json"
    completed = 0
    if args.resume and progress_path.exists():
        completed = int(json.loads(progress_path.read_text()).get("completed", 0))
    if completed > n:
        raise RuntimeError(f"progress {completed} exceeds sample count {n}")

    mode = "a" if completed and metadata_path.exists() else "w"
    fieldnames = [
        "sample_index", "image_id", "path", "target_id", "target_name",
        "annotated_category_ids", "cci_region", "cci_bbox_precision",
        "cci_target_drop_raw", "cci_target_drop_norm", "cci_margin_norm",
        "cci_pmean_norm", "feasible_set_size",
    ]
    current_device = device
    remaining = samples[completed:]
    loader = DataLoader(
        BASE.ImageDataset(remaining, preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=BASE.collate,
    )
    started = time.perf_counter()
    total_new = 0
    with metadata_path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for batch_no, (local_indices, images) in enumerate(loader):
            global_indices = local_indices + completed
            images = images.to(current_device)
            original, patches = BASE.visual_patch_tokens(model, images)
            with warnings.catch_warnings():
                # The pinned MPS path emits repetitive sklearn matmul warnings
                # while converting patch features to CPU; the generated values
                # are checked against the fixed local run below.
                warnings.simplefilter("ignore", RuntimeWarning)
                cluster_masks = BASE.cluster_patch_rows(patches, global_indices.tolist())
            rows_out, batch_raw, batch_norm, current_device, record = response_batch(
                model, current_device, samples, category_index, sel_text, eval_text,
                logit_scale, global_indices, images, cluster_masks, inside,
                original,
            )
            capture = args.out_dir / 'traceable_batches'
            capture.mkdir(exist_ok=True)
            np.savez_compressed(capture / f'{int(global_indices[0]):08d}.npz',
                sample_index=global_indices.numpy(), image_id=np.array([samples[int(i)]['image_id'] for i in global_indices]),
                response_raw=batch_raw,response_norm=batch_norm,**record)
            idx = global_indices.numpy()
            raw[idx] = batch_raw
            norm[idx] = batch_norm
            raw.flush()
            norm.flush()
            writer.writerows(rows_out)
            handle.flush()
            total_new += len(rows_out)
            completed_now = completed + total_new
            progress_path.write_text(json.dumps({"completed": completed_now}, indent=2) + "\n")
            if batch_no % 10 == 0:
                elapsed = time.perf_counter() - started
                logging.info(
                    "completed=%d new=%d elapsed_s=%.3f images_per_s=%.6f device=%s",
                    completed_now, total_new, elapsed, total_new / max(elapsed, 1e-9), current_device,
                )
            del images, cluster_masks, rows_out, batch_raw, batch_norm
            gc.collect()

    raw.flush()
    norm.flush()
    if completed + total_new != n:
        raise RuntimeError(f"incomplete run: {completed + total_new}/{n}")

    metadata = {
        "status": "reviewer_controls_response_regeneration",
        "dataset": args.dataset,
        "model": args.model,
        "openclip_name": openclip_name,
        "device_requested": requested_device,
        "device_last": str(current_device),
        "samples_available": n,
        "completed_before": completed,
        "completed_this_run": total_new,
        "completed_total": completed + total_new,
        "class_count": class_count,
        "category_ids": category_ids,
        "category_names": [categories[c] for c in category_ids],
        "elapsed_seconds_this_run": time.perf_counter() - started,
        "images_per_second_this_run": total_new / max(time.perf_counter() - started, 1e-9),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "command": " ".join(os.sys.argv),
        "epsilon": 0.02,
        "clusters": 8,
        "kmeans_n_init": 3,
        "kmeans_max_iter": 50,
        "kmeans_seed": "1701 + sample_index",
        "held_out_prompts": list(BASE.EVAL_PROMPTS),
        "raw_response_definition": "mean over the three held-out prompts of logit-scale times original-minus-masked image feature dot class text feature",
        "normalized_response_definition": "raw aggregate divided by norm of mean held-out class text feature",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    metadata_df = pd.read_csv(metadata_path)
    npz_path = args.out_dir / f"{args.dataset}_{args.model}_cci_full_class_responses.npz"
    np.savez_compressed(
        npz_path,
        response_raw=np.asarray(raw),
        response_norm=np.asarray(norm),
        image_id=np.asarray([s["image_id"] for s in samples]),
        sample_index=np.arange(n, dtype=np.int64),
        target_id=np.asarray([s["target_id"] for s in samples]),
        target_name=np.asarray([s["target_name"] for s in samples]),
        annotated_category_ids=np.asarray([json_line([s["target_id"], *s["distractor_ids"]]) for s in samples]),
        cci_region=metadata_df.cci_region.to_numpy(dtype=np.int64),
        cci_bbox_precision=metadata_df.cci_bbox_precision.to_numpy(dtype=np.float32),
        cci_target_drop_raw=metadata_df.cci_target_drop_raw.to_numpy(dtype=np.float32),
        cci_target_drop_norm=metadata_df.cci_target_drop_norm.to_numpy(dtype=np.float32),
        cci_margin_norm=metadata_df.cci_margin_norm.to_numpy(dtype=np.float32),
        cci_pmean_norm=metadata_df.cci_pmean_norm.to_numpy(dtype=np.float32),
        feasible_set_size=metadata_df.feasible_set_size.to_numpy(dtype=np.int64),
        category_ids=np.asarray(category_ids),
        category_names=np.asarray([categories[c] for c in category_ids]),
    )
    files = sorted(p for p in args.out_dir.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt')
    def checksum(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
        return h.hexdigest()
    (args.out_dir / 'SHA256SUMS.txt').write_text(''.join(checksum(p)+'  '+str(p.relative_to(args.out_dir))+'\n' for p in files))
    print(json.dumps({**metadata, "npz_path": str(npz_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run(parse_args())
