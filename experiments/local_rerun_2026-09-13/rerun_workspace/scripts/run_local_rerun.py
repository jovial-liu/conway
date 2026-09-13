#!/usr/bin/env python3
"""Local CCI rerun for the four archived OpenAI CLIP settings.

This runner regenerates patch clusters and masked responses from public images and
OpenAI CLIP weights.  It never writes to the frozen evidence directories.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import platform
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


VOC_CATEGORIES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
EVAL_PROMPTS = (
    "a picture of the {category}",
    "an image containing a {category}",
    "the {category}",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("coco", "voc2007"), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model", choices=("openai_b16", "openai_b32"), required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--weights-dir", type=Path, default=None)
    p.add_argument("--samples", type=int, default=0, help="0 means all eligible samples")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-only", action="store_true", help="run only the first batch and parity checks")
    return p.parse_args()


def device_for(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("requested mps but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_key: str, device: torch.device, weights_dir: Path | None = None):
    import open_clip

    name = "ViT-B-16" if model_key == "openai_b16" else "ViT-B-32"
    weight_path = None
    if weights_dir is not None:
        candidate = weights_dir / ("ViT-B-16.pt" if model_key == "openai_b16" else "ViT-B-32.pt")
        if candidate.exists():
            weight_path = str(candidate)
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained=weight_path or "openai", force_quick_gelu=True, weights_only=False
    )
    tokenizer = open_clip.get_tokenizer(name)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, preprocess, name


def text_features(model, tokenizer, prompts: list[str], device: torch.device) -> torch.Tensor:
    tokens = tokenizer(prompts).to(device)
    with torch.inference_mode():
        return F.normalize(model.encode_text(tokens).float(), dim=-1)


def visual_patch_tokens(model, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same manual OpenCLIP ViT path as the archived source."""
    visual = model.visual
    if hasattr(visual, "trunk"):
        trunk = visual.trunk
        x = trunk.patch_embed(images)
        x = trunk._pos_embed(x)
        x = trunk.patch_drop(x)
        x = trunk.norm_pre(x)
        x = trunk.blocks(x)
        x = trunk.norm(x)
        cls = visual.head(trunk.forward_head(x))
        patches = x[:, trunk.num_prefix_tokens:]
    else:
        x = visual._embeds(images)
        x = visual.transformer(x)
        cls, patches = visual._pool(x)
        if getattr(visual, "proj", None) is not None:
            cls = cls @ visual.proj
            patches = patches @ visual.proj
    return F.normalize(cls.float(), dim=-1), F.normalize(patches.float(), dim=-1)


def build_attention_mask(cluster_masks: torch.Tensor, heads: int, dtype, device) -> torch.Tensor:
    batch, clusters, patches = cluster_masks.shape
    tokens = patches + 1
    mask = torch.zeros((batch, clusters, tokens, tokens), dtype=dtype, device=device)
    for row in range(batch):
        for cluster in range(clusters):
            columns = torch.where(cluster_masks[row, cluster])[0].to(device) + 1
            mask[row, cluster, :, columns] = float("-inf")
    return mask.reshape(batch * clusters, tokens, tokens).repeat_interleave(heads, dim=0)


@torch.inference_mode()
def encode_masked(model, images: torch.Tensor, cluster_masks: torch.Tensor) -> torch.Tensor:
    visual = model.visual
    if hasattr(visual, "trunk"):
        raise RuntimeError("timm visual path is not part of the archived OpenAI protocol")
    clusters = cluster_masks.shape[1]
    repeated = images.repeat_interleave(clusters, dim=0)
    x = visual._embeds(repeated)
    heads = visual.transformer.resblocks[0].attn.num_heads
    mask = build_attention_mask(cluster_masks, heads, x.dtype, x.device)
    transformer = visual.transformer
    if transformer.batch_first:
        x = transformer(x, attn_mask=mask)
    else:
        x = transformer(x, attn_mask=mask)
    pooled, _ = visual._pool(x)
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return F.normalize(pooled.float(), dim=-1).reshape(len(images), clusters, -1)


def load_coco_samples(root: Path) -> tuple[list[dict], dict]:
    metadata = json.loads((root / "annotations" / "instances_val2014.json").read_text())
    categories = {row["id"]: row["name"] for row in metadata["categories"]}
    images = {row["id"]: row for row in metadata["images"]}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in metadata["annotations"]:
        if not row.get("iscrowd", 0):
            grouped[row["image_id"]].append(row)
    samples = []
    for image_id in sorted(grouped):
        image = images[image_id]
        path = root / "val2014" / image["file_name"]
        if not path.exists():
            continue
        annotations = sorted(grouped[image_id], key=lambda row: row["area"], reverse=True)
        target = annotations[0]
        distractor = next((row for row in annotations[1:] if row["category_id"] != target["category_id"]), None)
        area_fraction = target["area"] / (image["width"] * image["height"])
        if distractor is None or not 0.02 <= area_fraction <= 0.8:
            continue
        samples.append({
            "image_id": image_id, "path": str(path), "width": image["width"], "height": image["height"],
            "target_id": target["category_id"], "target_name": categories[target["category_id"]],
            "distractor_id": distractor["category_id"], "distractor_name": categories[distractor["category_id"]],
            "distractor_ids": sorted({row["category_id"] for row in annotations if row["category_id"] != target["category_id"]}),
            "target_boxes": [row["bbox"] for row in annotations if row["category_id"] == target["category_id"]],
            "distractor_boxes": [row["bbox"] for row in annotations if row["category_id"] != target["category_id"]],
        })
    np.random.default_rng(1701).shuffle(samples)
    return samples, categories


def load_voc_samples(root: Path) -> tuple[list[dict], dict]:
    base = root / "VOCdevkit" / "VOC2007"
    ids = (base / "ImageSets" / "Main" / "test.txt").read_text().split()
    categories = {name: name for name in VOC_CATEGORIES}
    samples = []
    for image_id in sorted(ids):
        xml = ET.parse(base / "Annotations" / f"{image_id}.xml").getroot()
        width, height = int(xml.findtext("size/width")), int(xml.findtext("size/height"))
        annotations = []
        for obj in xml.findall("object"):
            name = obj.findtext("name")
            box = obj.find("bndbox")
            xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
            xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
            annotations.append({"category_id": name, "bbox": [xmin, ymin, xmax - xmin, ymax - ymin], "area": (xmax - xmin) * (ymax - ymin)})
        annotations.sort(key=lambda row: row["area"], reverse=True)
        if not annotations:
            continue
        target = annotations[0]
        distractor = next((row for row in annotations[1:] if row["category_id"] != target["category_id"]), None)
        area_fraction = target["area"] / (width * height)
        if distractor is None or not 0.02 <= area_fraction <= 0.8:
            continue
        samples.append({
            "image_id": image_id, "path": str(base / "JPEGImages" / f"{image_id}.jpg"),
            "width": width, "height": height, "target_id": target["category_id"],
            "target_name": target["category_id"], "distractor_id": distractor["category_id"],
            "distractor_name": distractor["category_id"],
            "distractor_ids": sorted({row["category_id"] for row in annotations if row["category_id"] != target["category_id"]}),
            "target_boxes": [row["bbox"] for row in annotations if row["category_id"] == target["category_id"]],
            "distractor_boxes": [row["bbox"] for row in annotations if row["category_id"] != target["category_id"]],
        })
    np.random.default_rng(1701).shuffle(samples)
    return samples, categories


def load_samples(dataset: str, root: Path) -> tuple[list[dict], dict]:
    return load_voc_samples(root) if dataset == "voc2007" else load_coco_samples(root)


def transformed_masks(samples: list[dict], patch_count: int, boxes_key: str = "target_boxes") -> torch.Tensor:
    grid = round(patch_count ** 0.5)
    if grid * grid != patch_count:
        raise ValueError(f"non-square patch grid {patch_count}")
    rows = []
    for sample in samples:
        scale = 224.0 / min(sample["width"], sample["height"])
        resized_w, resized_h = sample["width"] * scale, sample["height"] * scale
        crop_x, crop_y = (resized_w - 224.0) / 2, (resized_h - 224.0) / 2
        boxes = []
        for x, y, w, h in sample[boxes_key]:
            boxes.append((x * scale - crop_x, y * scale - crop_y, w * scale, h * scale))
        row = []
        for r in range(grid):
            for c in range(grid):
                x, y = (c + 0.5) * 224.0 / grid, (r + 0.5) * 224.0 / grid
                row.append(any(bx <= x <= bx + bw and by <= y <= by + bh for bx, by, bw, bh in boxes))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.bool)


class ImageDataset(Dataset):
    def __init__(self, samples: list[dict], preprocess):
        self.samples, self.preprocess = samples, preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = Image.open(self.samples[index]["path"]).convert("RGB")
        return index, self.preprocess(image)


def collate(rows):
    return torch.tensor([r[0] for r in rows], dtype=torch.long), torch.stack([r[1] for r in rows])


def cluster_patch_rows(patches: torch.Tensor, sample_indices: list[int]) -> torch.Tensor:
    out = []
    for sample_index, values in zip(sample_indices, patches.cpu().numpy()):
        labels = KMeans(n_clusters=8, n_init=3, max_iter=50, random_state=1701 + sample_index).fit_predict(values)
        out.append(np.stack([labels == cluster for cluster in range(8)]))
    return torch.tensor(np.stack(out), dtype=torch.bool)


def json_array(values) -> str:
    return json.dumps(np.asarray(values).tolist(), separators=(",", ":"))


def per_batch_rows(args, model, device, samples, categories, category_index, text_sel, text_eval, logit_scale, indices, images, cluster_masks, inside, original=None):
    if original is None:
        original, _ = visual_patch_tokens(model, images)
    try:
        after = encode_masked(model, images, cluster_masks.to(device))
    except RuntimeError as exc:
        if device.type != "mps":
            raise
        logging.warning("MPS masked attention failed; retrying this batch on CPU: %s", exc)
        model.to("cpu")
        cpu_images, cpu_masks = images.to("cpu"), cluster_masks.to("cpu")
        original, _ = visual_patch_tokens(model, cpu_images)
        after = encode_masked(model, cpu_images, cpu_masks)
        device = torch.device("cpu")
        images = cpu_images
    logit_scale = logit_scale.to(after.device)
    target_ids = torch.tensor([category_index[samples[int(i)]["target_id"]] for i in indices.tolist()], dtype=torch.long)
    target_sel = text_sel[target_ids].to(after.device)
    sel_drop = logit_scale * ((original[:, None, :] - after) * target_sel[:, None, :]).sum(2)
    cat_sel = logit_scale * ((original[:, None, :] - after) @ text_sel.to(after.device).T)
    target_mask = F.one_hot(target_ids, num_classes=cat_sel.shape[2]).bool()[:, None, :].to(after.device)
    mean_foil = (cat_sel.masked_fill(target_mask, 0.0).sum(2) / (cat_sel.shape[2] - 1))
    max_foil = cat_sel.masked_fill(target_mask, float("-inf")).max(2).values
    cci = sel_drop.argmax(1)
    rows_local = torch.arange(len(indices))
    cci_target = sel_drop[rows_local, cci]
    feasible = sel_drop >= (cci_target[:, None] - 0.02)
    wf_score = torch.where(feasible, sel_drop - max_foil, torch.tensor(float("-inf"), device=after.device))
    mean_score = torch.where(feasible, sel_drop - mean_foil, torch.tensor(float("-inf"), device=after.device))
    max_score = torch.where(feasible, sel_drop - 0.1 * max_foil, torch.tensor(float("-inf"), device=after.device))
    wf, mean, max01 = wf_score.argmax(1), mean_score.argmax(1), max_score.argmax(1)
    eval_emb = text_eval.to(after.device)
    diff = original[:, None, :] - after
    eval_target = logit_scale * torch.einsum("bkd,pbd->bkp", diff, eval_emb[:, target_ids, :]).mean(2)
    eval_cat = logit_scale * torch.einsum("bkd,pcd->bkpc", diff, eval_emb).mean(2)
    class_norm = eval_emb.mean(0).norm(dim=-1)
    eval_target_norm = eval_target / class_norm[target_ids][:, None]
    eval_cat_norm = eval_cat / class_norm[None, None, :]
    eval_target_mask = F.one_hot(target_ids, num_classes=eval_cat_norm.shape[2]).bool()[:, None, :].to(after.device)
    foil_eval_max = eval_cat_norm.masked_fill(eval_target_mask, float("-inf"))
    foil_eval_sum = eval_cat_norm.masked_fill(eval_target_mask, 0.0)
    margin_norm = eval_target_norm - foil_eval_max.max(2).values
    # Replace the target class by zero and divide by the fixed number of
    # non-target classes, matching the archived foil-mean definition.
    pmean_norm = eval_target_norm - foil_eval_sum.sum(2) / (eval_cat_norm.shape[2] - 1)
    target_raw = eval_target
    target_norm = eval_target_norm
    candidate_bbox = inside[indices].to(after.device).float()[:, None, :].mul(cluster_masks.to(after.device).float()).sum(2)
    candidate_bbox /= cluster_masks.to(after.device).float().sum(2).clamp_min(1)
    out = []
    names = (("cci", cci), ("wf", wf), ("mean", mean), ("max_0_1", max01))
    for local, global_i in enumerate(indices.tolist()):
        sample = samples[global_i]
        row = {
            "status": "local_rerun", "dataset": args.dataset, "model": args.model,
            "sample_index": global_i, "image_id": sample["image_id"], "path": sample["path"],
            "target_id": sample["target_id"], "target_name": sample["target_name"],
            "distractor_id": sample["distractor_id"], "distractor_name": sample["distractor_name"],
            "cci_region": int(cci[local]), "wf_region": int(wf[local]),
            "mean_region": int(mean[local]), "max_0_1_region": int(max01[local]),
            "feasible_set_size": int(feasible[local].sum().item()),
            "candidate_selection_target": json_array(sel_drop[local].detach().cpu().numpy()),
            "candidate_selection_mean_foil": json_array(mean_foil[local].detach().cpu().numpy()),
            "candidate_selection_max_foil": json_array(max_foil[local].detach().cpu().numpy()),
            "candidate_eval_margin_norm": json_array(margin_norm[local].detach().cpu().numpy()),
            "candidate_eval_pmean_norm": json_array(pmean_norm[local].detach().cpu().numpy()),
            "candidate_eval_target_drop_raw": json_array(target_raw[local].detach().cpu().numpy()),
            "candidate_eval_target_drop_norm": json_array(target_norm[local].detach().cpu().numpy()),
            "candidate_bbox_precision": json_array(candidate_bbox[local].detach().cpu().numpy()),
        }
        for method, region in names:
            r = int(region[local])
            row[f"{method}_margin_norm"] = float(margin_norm[local, r].detach().cpu())
            row[f"{method}_pmean_norm"] = float(pmean_norm[local, r].detach().cpu())
            row[f"{method}_target_drop_raw"] = float(target_raw[local, r].detach().cpu())
            row[f"{method}_target_drop_norm"] = float(target_norm[local, r].detach().cpu())
            row[f"{method}_bbox_precision"] = float(candidate_bbox[local, r].detach().cpu())
        out.append(row)
    return out, device


def run(args: argparse.Namespace) -> None:
    args.data_root = args.data_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.out_dir / "run.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    samples, categories = load_samples(args.dataset, args.data_root)
    if args.samples:
        samples = samples[:args.samples]
    if not samples:
        raise RuntimeError("no eligible samples; dataset annotations/images are incomplete")
    device = device_for(args.device)
    model, tokenizer, preprocess, openclip_name = load_model(args.model, device, args.weights_dir)
    scale = model.logit_scale.exp().float().detach().to(device)
    category_ids = sorted(categories)
    category_index = {category_id: i for i, category_id in enumerate(category_ids)}
    sel_text = text_features(model, tokenizer, [f"a photo of a {categories[c]}" for c in category_ids], device)
    eval_prompts = [p.format(category=categories[c]) for p in EVAL_PROMPTS for c in category_ids]
    eval_text = text_features(model, tokenizer, eval_prompts, device).reshape(3, len(category_ids), -1)
    patch_grid = 14 if args.model == "openai_b16" else 7
    inside = transformed_masks(samples, patch_grid * patch_grid)
    out_csv = args.out_dir / "per_image.csv"
    completed = 0
    if args.resume and out_csv.exists():
        completed = sum(1 for _ in out_csv.open()) - 1
        if completed < 0:
            completed = 0
    if completed >= len(samples):
        logging.info("already complete: %d rows", completed)
        return
    remaining = samples[completed:]
    loader = DataLoader(ImageDataset(remaining, preprocess), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)
    fieldnames = None
    mode = "a" if completed else "w"
    start_time = time.perf_counter()
    total_new = 0
    current_device = device
    for batch_no, (local_indices, images) in enumerate(tqdm(loader, desc=f"local rerun {args.dataset}/{args.model}")):
        global_indices = local_indices + completed
        images = images.to(current_device)
        original, patches = visual_patch_tokens(model, images)
        cluster_masks = cluster_patch_rows(patches, global_indices.tolist())
        rows, current_device = per_batch_rows(args, model, current_device, samples, categories, category_index, sel_text, eval_text, scale, global_indices, images, cluster_masks, inside, original=original)
        if fieldnames is None:
            fieldnames = list(rows[0])
        with out_csv.open(mode, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            writer.writerows(rows)
        mode = "a"
        total_new += len(rows)
        if args.smoke_only:
            break
        if batch_no % 10 == 0:
            elapsed = time.perf_counter() - start_time
            logging.info("completed=%d new=%d elapsed_s=%.3f images_per_s=%.6f device=%s", completed + total_new, total_new, elapsed, total_new / max(elapsed, 1e-9), current_device)
        del images, cluster_masks, rows
        gc.collect()
    elapsed = time.perf_counter() - start_time
    metadata = {
        "status": "local_rerun", "dataset": args.dataset, "model": args.model,
        "openclip_name": openclip_name, "device_requested": args.device, "device_last": str(current_device),
        "samples_available": len(samples), "completed_before": completed,
        "completed_this_run": total_new, "completed_total": completed + total_new,
        "elapsed_seconds_this_run": elapsed, "images_per_second_this_run": total_new / max(elapsed, 1e-9),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "mps_available": bool(torch.backends.mps.is_available()), "mps_built": bool(torch.backends.mps.is_built()),
        "command": " ".join(os.sys.argv), "epsilon": 0.02, "clusters": 8, "bootstrap_count": 10000,
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    run(parse_args())
