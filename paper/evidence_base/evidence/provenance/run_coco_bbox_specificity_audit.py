#!/usr/bin/env python3
"""COCO bbox audit for semantic localization versus patch-deletion specificity."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from run_waterbirds_mvp import load_open_clip, text_features, visual_patch_tokens


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("outputs_coco_bbox_specificity"))
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--gelu", choices=["quick", "standard"], default="quick")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset", choices=["coco", "voc2007"], default="coco")
    p.add_argument("--coco-split", choices=["train2014", "val2014"], default="val2014")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--topk-grid", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--extract-only", action="store_true")
    return p.parse_args()


def select_samples(coco_root, count, split="val2014"):
    metadata = json.loads((coco_root / "annotations" / f"instances_{split}.json").read_text())
    categories = {row["id"]: row["name"] for row in metadata["categories"]}
    images = {row["id"]: row for row in metadata["images"]}
    grouped = defaultdict(list)
    for row in metadata["annotations"]:
        if not row.get("iscrowd", 0):
            grouped[row["image_id"]].append(row)
    samples = []
    for image_id in sorted(grouped):
        image = images[image_id]
        path = coco_root / split / image["file_name"]
        if not path.exists():
            continue
        annotations = sorted(grouped[image_id], key=lambda row: row["area"], reverse=True)
        target = annotations[0]
        distractor = next((row for row in annotations[1:] if row["category_id"] != target["category_id"]), None)
        area_fraction = target["area"] / (image["width"] * image["height"])
        if distractor is None or not .02 <= area_fraction <= .8:
            continue
        target_boxes = [row["bbox"] for row in annotations if row["category_id"] == target["category_id"]]
        distractor_boxes = [row["bbox"] for row in annotations if row["category_id"] != target["category_id"]]
        distractor_ids = sorted({
            row["category_id"] for row in annotations if row["category_id"] != target["category_id"]
        })
        samples.append({
            "image_id": image_id,
            "path": str(path),
            "width": image["width"],
            "height": image["height"],
            "target_id": target["category_id"],
            "target_name": categories[target["category_id"]],
            "distractor_id": distractor["category_id"],
            "distractor_name": categories[distractor["category_id"]],
            "distractor_ids": distractor_ids,
            "target_boxes": target_boxes,
            "distractor_boxes": distractor_boxes,
        })
    generator = np.random.default_rng(1701)
    generator.shuffle(samples)
    return samples[:count], categories


VOC_CATEGORIES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def select_voc_samples(voc_root, count):
    base = voc_root / "VOCdevkit" / "VOC2007"
    image_ids = (base / "ImageSets" / "Main" / "test.txt").read_text().split()
    categories = {name: name for name in VOC_CATEGORIES}
    samples = []
    for image_id in sorted(image_ids):
        root = ET.parse(base / "Annotations" / f"{image_id}.xml").getroot()
        width = int(root.findtext("size/width"))
        height = int(root.findtext("size/height"))
        annotations = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            box = obj.find("bndbox")
            xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
            xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
            annotations.append({
                "category_id": name,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
            })
        annotations.sort(key=lambda row: row["area"], reverse=True)
        if not annotations:
            continue
        target = annotations[0]
        distractor = next(
            (row for row in annotations[1:] if row["category_id"] != target["category_id"]),
            None,
        )
        area_fraction = target["area"] / (width * height)
        if distractor is None or not .02 <= area_fraction <= .8:
            continue
        target_boxes = [row["bbox"] for row in annotations if row["category_id"] == target["category_id"]]
        distractor_boxes = [row["bbox"] for row in annotations if row["category_id"] != target["category_id"]]
        distractor_ids = sorted({
            row["category_id"] for row in annotations if row["category_id"] != target["category_id"]
        })
        samples.append({
            "image_id": image_id,
            "path": str(base / "JPEGImages" / f"{image_id}.jpg"),
            "width": width,
            "height": height,
            "target_id": target["category_id"],
            "target_name": target["category_id"],
            "distractor_id": distractor["category_id"],
            "distractor_name": distractor["category_id"],
            "distractor_ids": distractor_ids,
            "target_boxes": target_boxes,
            "distractor_boxes": distractor_boxes,
        })
    generator = np.random.default_rng(1701)
    generator.shuffle(samples)
    return samples[:count], categories


def select_dataset_samples(dataset, root, count, coco_split="val2014"):
    if dataset == "voc2007":
        return select_voc_samples(root, count)
    return select_samples(root, count, coco_split)


class CocoAuditDataset(Dataset):
    def __init__(self, samples, preprocess):
        self.samples, self.preprocess = samples, preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        image = Image.open(row["path"]).convert("RGB")
        return index, self.preprocess(image)


def collate(rows):
    return torch.tensor([row[0] for row in rows]), torch.stack([row[1] for row in rows])


@torch.no_grad()
def extract_features(args, samples, model, preprocess, device):
    cache = args.out_dir / f"{args.model.replace('/', '_')}_features.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu")
    loader = DataLoader(
        CocoAuditDataset(samples, preprocess),
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        collate_fn=collate,
    )
    images, patches = None, None
    offset = 0
    for _, batch in tqdm(loader, desc=f"extract {args.dataset} patches"):
        image_features, patch_features = visual_patch_tokens(model, batch.to(device))
        image_features = image_features.cpu()
        patch_features = patch_features.cpu()
        if images is None:
            images = torch.empty((len(samples), *image_features.shape[1:]), dtype=image_features.dtype)
            patches = torch.empty((len(samples), *patch_features.shape[1:]), dtype=patch_features.dtype)
        end = offset + len(image_features)
        images[offset:end].copy_(image_features)
        patches[offset:end].copy_(patch_features)
        offset = end
    if images is None or patches is None:
        raise ValueError("Cannot extract features from an empty dataset")
    output = {"image_features": images, "patch_features": patches}
    torch.save(output, cache)
    return output


def patch_inside_bbox(samples, patch_count):
    grid = round(patch_count ** .5)
    if grid * grid != patch_count:
        raise ValueError(f"Expected square patch grid, got {patch_count}")
    result = []
    for sample in samples:
        inside = []
        for row in range(grid):
            for col in range(grid):
                x = (col + .5) * sample["width"] / grid
                y = (row + .5) * sample["height"] / grid
                inside.append(any(bx <= x <= bx + bw and by <= y <= by + bh for bx, by, bw, bh in sample["target_boxes"]))
        result.append(inside)
    return torch.tensor(result, dtype=torch.bool)


def deletion_metrics(features, selectors, inside, target_text, distractor_text, scale, beta, topk):
    patches, images = features["patch_features"].float(), features["image_features"].float()
    idx = selectors.topk(k=min(topk, patches.shape[1]), dim=1).indices
    removed = torch.gather(patches, 1, idx[:, :, None].expand(-1, -1, patches.shape[-1])).mean(1)
    after = F.normalize(images - beta * F.normalize(removed, dim=-1), dim=-1)
    original_target = scale * (images * target_text).sum(1)
    after_target = scale * (after * target_text).sum(1)
    original_margin = scale * ((images * target_text).sum(1) - (images * distractor_text).sum(1))
    after_margin = scale * ((after * target_text).sum(1) - (after * distractor_text).sum(1))
    hit = torch.gather(inside, 1, idx).float().mean(1)
    return {
        "bbox_precision_at_k": hit,
        "target_logit_drop": original_target - after_target,
        "target_vs_distractor_margin_drop": original_margin - after_margin,
    }, idx


def summarize(method, seed, topk, metrics):
    return {
        "method": method,
        "seed": seed,
        "topk": topk,
        **{name: float(value.mean()) for name, value in metrics.items()},
    }


def bootstrap(real, controls, count=1000):
    rng, rows = np.random.default_rng(1701), []
    for control_name, control in controls.items():
        for metric in real:
            differences = []
            for _ in range(count):
                idx = rng.integers(0, len(real[metric]), len(real[metric]))
                differences.append(float(real[metric][idx].mean() - control[metric][idx].mean()))
            rows.append({
                "comparison": f"real_minus_{control_name}",
                "metric": metric,
                "mean": np.mean(differences),
                "ci_low": np.quantile(differences, .025),
                "ci_high": np.quantile(differences, .975),
            })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.coco_root = args.coco_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples, categories = select_dataset_samples(args.dataset, args.coco_root, args.samples, args.coco_split)
    if len(samples) < args.samples:
        raise RuntimeError(f"Requested {args.samples} samples but found {len(samples)} eligible images")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tokenizer, preprocess = load_open_clip(
        args.model, args.pretrained, args.checkpoint, device, force_quick_gelu=args.gelu == "quick"
    )
    scale = float(model.logit_scale.exp().detach().cpu())
    category_ids = sorted(categories)
    category_names = [categories[category_id] for category_id in category_ids]
    category_index = {category_id: index for index, category_id in enumerate(category_ids)}
    category_text = text_features(model, tokenizer, [f"a photo of a {name}" for name in category_names], device).cpu()
    features = extract_features(args, samples, model, preprocess, device)
    if args.extract_only:
        print(f"Wrote {len(samples)} feature rows to {args.out_dir}")
        return
    inside = patch_inside_bbox(samples, features["patch_features"].shape[1])
    target_index = torch.tensor([category_index[row["target_id"]] for row in samples])
    distractor_index = torch.tensor([category_index[row["distractor_id"]] for row in samples])
    target_text, distractor_text = category_text[target_index], category_text[distractor_index]
    patches = features["patch_features"].float()

    real_selector = torch.einsum("npd,nd->np", patches, target_text)
    real_by_k, real_indices, rows = {}, {}, []
    for topk in args.topk_grid:
        values, idx = deletion_metrics(features, real_selector, inside, target_text, distractor_text, scale, args.beta, topk)
        real_by_k[topk], real_indices[topk] = values, idx
        rows.append(summarize("real_target_concept", None, topk, values))

    controls_by_k = defaultdict(list)
    generator = torch.Generator()
    for seed in range(args.seeds):
        generator.manual_seed(1701 + seed)
        shuffled_index = target_index[torch.randperm(len(target_index), generator=generator)]
        same = shuffled_index == target_index
        shuffled_index[same] = (shuffled_index[same] + 1) % len(category_text)
        random_text = F.normalize(torch.randn(target_text.shape, generator=generator), dim=-1)
        selectors = {
            "shuffled_category_name": torch.einsum("npd,nd->np", patches, category_text[shuffled_index]),
            "random_text_direction": torch.einsum("npd,nd->np", patches, random_text),
            "random_patch": torch.rand((len(samples), patches.shape[1]), generator=generator),
        }
        for method, selector in selectors.items():
            for topk in args.topk_grid:
                values, _ = deletion_metrics(features, selector, inside, target_text, distractor_text, scale, args.beta, topk)
                rows.append(summarize(method, seed, topk, values))
                controls_by_k[(method, topk)].append(values)

    frame = pd.DataFrame(rows)
    control_summary = (
        frame[frame["seed"].notna()]
        .groupby(["method", "topk"])
        .agg(["mean", "std"])
        .reset_index()
    )
    control_summary.columns = ["_".join(str(part) for part in col if part != "").rstrip("_") for col in control_summary.columns]
    boot_controls = {}
    best_topk = max(args.topk_grid)
    for method in ["shuffled_category_name", "random_text_direction", "random_patch"]:
        values = controls_by_k[(method, best_topk)]
        boot_controls[method] = {
            metric: torch.stack([row[metric] for row in values]).mean(0)
            for metric in real_by_k[best_topk]
        }
    bootstrap_frame = bootstrap(real_by_k[best_topk], boot_controls)
    per_sample = pd.DataFrame({
        "image_id": [row["image_id"] for row in samples],
        "path": [row["path"] for row in samples],
        "target_name": [row["target_name"] for row in samples],
        "distractor_name": [row["distractor_name"] for row in samples],
        "target_boxes": [json.dumps(row["target_boxes"]) for row in samples],
        f"real_top{best_topk}_patch_indices": [json.dumps(row) for row in real_indices[best_topk].tolist()],
        f"real_bbox_precision_at_{best_topk}": real_by_k[best_topk]["bbox_precision_at_k"],
        f"real_target_logit_drop_at_{best_topk}": real_by_k[best_topk]["target_logit_drop"],
        f"real_margin_drop_at_{best_topk}": real_by_k[best_topk]["target_vs_distractor_margin_drop"],
    })

    frame.to_csv(args.out_dir / "all_seed_results.csv", index=False)
    control_summary.to_csv(args.out_dir / "control_summary.csv", index=False)
    bootstrap_frame.to_csv(args.out_dir / "bootstrap_ci.csv", index=False)
    per_sample.to_csv(args.out_dir / "per_sample.csv", index=False)
    report = f"""# COCO BBox Intervention-Specificity Audit

## Protocol

For each COCO image, select the largest annotated target object and a different
annotated distractor object. Rank CLIP patches using the target category text,
a shuffled category name, a random text direction, or random patch scores.
Measure bbox precision@k, target-logit drop, and target-versus-distractor margin
drop after deleting selected patch directions.

## Real Target Concept

{frame[frame["method"] == "real_target_concept"].to_markdown(index=False)}

## Matched Controls

{control_summary.to_markdown(index=False)}

## Bootstrap at Top-{best_topk}

{bootstrap_frame.to_markdown(index=False)}

No parameters were trained.
"""
    (args.out_dir / "report.md").write_text(report)
    print(frame[frame["method"] == "real_target_concept"].to_string(index=False))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()

