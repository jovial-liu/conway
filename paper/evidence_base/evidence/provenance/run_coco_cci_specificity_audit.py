#!/usr/bin/env python3
"""Audit semantic specificity of CCI-style CLIP cluster interventions."""

from __future__ import annotations

import argparse
import gc
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

from run_coco_bbox_specificity_audit import select_dataset_samples
from run_coco_context_decomposition_audit import transformed_bbox_masks
from run_waterbirds_mvp import load_open_clip, text_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--cluster-cache", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="ViT-B-16")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--gelu", choices=["quick", "standard"], default="quick")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", choices=["coco", "voc2007"], default="coco")
    parser.add_argument("--coco-split", choices=["train2014", "val2014"], default="val2014")
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--foil-lambdas", default="0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--hybrid-mean-lambdas", default="0.25,0.5,1.0")
    parser.add_argument("--hybrid-max-lambdas", default="0.1,0.25,0.5")
    parser.add_argument("--cvar-topks", default="1,3,5,10")
    parser.add_argument("--risk-lambdas", default="0.05,0.1,0.25")
    parser.add_argument("--block-layers", choices=["all", "last", "first_half", "last_half"], default="all")
    parser.add_argument("--intervention", choices=["attention_block", "token_zero"], default="attention_block")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Keep only the fixed paper comparison for memory-efficient full-scale confirmation.",
    )
    parser.add_argument(
        "--compact-risk-grid",
        action="store_true",
        help="Add the small paper-facing CVaR grid while retaining compact full-scale execution.",
    )
    return parser.parse_args()


class CocoDataset(Dataset):
    def __init__(self, samples, preprocess):
        self.samples, self.preprocess = samples, preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return index, self.preprocess(Image.open(self.samples[index]["path"]).convert("RGB"))


def collate(rows):
    return torch.tensor([row[0] for row in rows]), torch.stack([row[1] for row in rows])


def cluster_patches(patches, clusters):
    output = []
    for index, values in enumerate(tqdm(patches.numpy(), desc="cluster patches")):
        labels = KMeans(n_clusters=clusters, n_init=3, max_iter=50, random_state=1701 + index).fit_predict(values)
        output.append(np.stack([labels == cluster for cluster in range(clusters)]))
    return torch.tensor(np.stack(output), dtype=torch.bool)


def load_or_cluster_patches(args, patches):
    if args.cluster_cache is None:
        return cluster_patches(patches, args.clusters)
    feature_stat = args.feature_cache.stat()
    metadata = {
        "feature_cache": str(args.feature_cache.resolve()),
        "feature_cache_size": feature_stat.st_size,
        "feature_cache_mtime_ns": feature_stat.st_mtime_ns,
        "sample_offset": args.sample_offset,
        "samples": len(patches),
        "clusters": args.clusters,
        "patches": patches.shape[1],
    }
    if args.cluster_cache.exists():
        cached = torch.load(args.cluster_cache, map_location="cpu")
        if cached.get("metadata") != metadata:
            raise ValueError(f"Cluster cache metadata mismatch: {args.cluster_cache}")
        print(f"Loaded cluster cache: {args.cluster_cache}")
        return cached["cluster_masks"]
    cluster_masks = cluster_patches(patches, args.clusters)
    args.cluster_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "cluster_masks": cluster_masks}, args.cluster_cache)
    print(f"Wrote cluster cache: {args.cluster_cache}")
    return cluster_masks


def build_attention_mask(cluster_masks, heads, dtype, device):
    batch, clusters, patches = cluster_masks.shape
    tokens = patches + 1
    mask = torch.zeros((batch, clusters, tokens, tokens), dtype=dtype, device=device)
    for row in range(batch):
        for cluster in range(clusters):
            columns = torch.where(cluster_masks[row, cluster])[0].to(device) + 1
            mask[row, cluster, :, columns] = float("-inf")
    return mask.reshape(batch * clusters, tokens, tokens).repeat_interleave(heads, dim=0)


def build_timm_attention_mask(cluster_masks, dtype, device):
    batch, clusters, patches = cluster_masks.shape
    mask = torch.zeros((batch, clusters, 1, patches, patches), dtype=dtype, device=device)
    for row in range(batch):
        for cluster in range(clusters):
            columns = torch.where(cluster_masks[row, cluster])[0].to(device)
            mask[row, cluster, :, :, columns] = float("-inf")
    return mask.reshape(batch * clusters, 1, patches, patches)


def masked_layer_indices(count, mode):
    if mode == "last":
        return {count - 1}
    if mode == "first_half":
        return set(range(count // 2))
    if mode == "last_half":
        return set(range(count // 2, count))
    return set(range(count))


def zero_cluster_tokens(x, cluster_masks, prefix_tokens):
    selected = cluster_masks.reshape(-1, cluster_masks.shape[-1]).to(x.device)
    x = x.clone()
    x[:, prefix_tokens:].masked_fill_(selected[:, :, None], 0.0)
    return x


@torch.inference_mode()
def encode_with_attention_masks(model, images, cluster_masks, block_layers="all", intervention="attention_block"):
    visual = model.visual
    clusters = cluster_masks.shape[1]
    repeated = images.repeat_interleave(clusters, dim=0)
    if hasattr(visual, "trunk"):
        trunk = visual.trunk
        x = trunk.patch_embed(repeated)
        x = trunk._pos_embed(x)
        x = trunk.patch_drop(x)
        x = trunk.norm_pre(x)
        if intervention == "token_zero":
            x = zero_cluster_tokens(x, cluster_masks, trunk.num_prefix_tokens)
        mask = build_timm_attention_mask(cluster_masks, x.dtype, x.device)
        selected_layers = masked_layer_indices(len(trunk.blocks), block_layers)
        for index, block in enumerate(trunk.blocks):
            use_mask = intervention == "attention_block" and index in selected_layers
            x = block(x, attn_mask=mask if use_mask else None)
        x = trunk.norm(x)
        pooled = visual.head(trunk.forward_head(x))
        return F.normalize(pooled.float(), dim=-1).reshape(len(images), clusters, -1)
    x = visual._embeds(repeated)
    if intervention == "token_zero":
        x = zero_cluster_tokens(x, cluster_masks, 1)
    heads = visual.transformer.resblocks[0].attn.num_heads
    mask = build_attention_mask(cluster_masks, heads, x.dtype, x.device)
    if intervention == "token_zero":
        x = visual.transformer(x)
    elif block_layers == "all":
        x = visual.transformer(x, attn_mask=mask)
    else:
        transformer = visual.transformer
        if not transformer.batch_first:
            x = x.transpose(0, 1)
        selected_layers = masked_layer_indices(len(transformer.resblocks), block_layers)
        for index, block in enumerate(transformer.resblocks):
            x = block(x, attn_mask=mask if index in selected_layers else None)
        if not transformer.batch_first:
            x = x.transpose(0, 1)
    pooled, _ = visual._pool(x)
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return F.normalize(pooled.float(), dim=-1).reshape(len(images), clusters, -1)


@torch.inference_mode()
def encode_with_attention_scores(model, images):
    """Encode images and collect text-free CLS attention baselines."""
    maps, handles = [], []

    def capture_openclip_attention(module, inputs):
        query, key = inputs[:2]
        width, heads = module.embed_dim, module.num_heads
        head_width = width // heads
        q = F.linear(query, module.in_proj_weight[:width], module.in_proj_bias[:width])
        k = F.linear(key, module.in_proj_weight[width:2 * width], module.in_proj_bias[width:2 * width])
        q = q.reshape(len(query), -1, heads, head_width).transpose(1, 2)
        k = k.reshape(len(key), -1, heads, head_width).transpose(1, 2)
        maps.append((q @ k.transpose(-2, -1) * (head_width ** -0.5)).softmax(-1).mean(1).cpu())

    def capture_timm_attention(module, inputs):
        x = inputs[0]
        batch, tokens, width = x.shape
        qkv = module.qkv(x).reshape(batch, tokens, 3, module.num_heads, module.head_dim)
        q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)
        maps.append((q @ k.transpose(-2, -1) * module.scale).softmax(-1).mean(1).cpu())

    visual = model.visual
    if hasattr(visual, "trunk"):
        blocks = visual.trunk.blocks
        capture = capture_timm_attention
    else:
        blocks = visual.transformer.resblocks
        capture = capture_openclip_attention
    for block in blocks:
        handles.append(block.attn.register_forward_pre_hook(capture))
    try:
        original = F.normalize(model.encode_image(images).float(), dim=-1).cpu()
    finally:
        for handle in handles:
            handle.remove()
    tokens = maps[0].shape[-1]
    identity = torch.eye(tokens).expand(len(images), -1, -1)
    rollout = identity
    for attention in maps:
        attention = attention + identity
        attention = attention / attention.sum(-1, keepdim=True)
        rollout = attention @ rollout
    if hasattr(visual, "trunk"):
        return original, maps[-1].mean(1), rollout.mean(1)
    return original, maps[-1][:, 0, 1:], rollout[:, 0, 1:]


def aggregate_patch_scores(scores, cluster_masks):
    return (scores[:, None, :] * cluster_masks).sum(2) / cluster_masks.sum(2).clamp_min(1)


def category_reciprocal_rank(category_drop, target_drop, category_foil_mask):
    outranking_foils = (
        (category_drop > target_drop[:, :, None])
        & category_foil_mask[:, None, :]
    )
    return (1 + outranking_foils.sum(2)).float().reciprocal()


def choose_metrics(scores, values, cluster_masks, inside, distractor_inside):
    selected = scores.argmax(1)
    row = torch.arange(len(selected))
    chosen_mask = cluster_masks[row, selected]
    bbox_precision = (chosen_mask & inside).sum(1).float() / chosen_mask.sum(1).clamp_min(1)
    distractor_bbox_precision = (
        (chosen_mask & distractor_inside).sum(1).float() / chosen_mask.sum(1).clamp_min(1)
    )
    return {
        metric: tensor[row, selected]
        for metric, tensor in values.items()
    } | {
        "selected_cluster": selected.float(),
        "selection_score": scores[row, selected],
        "bbox_precision": bbox_precision,
        "distractor_bbox_precision": distractor_bbox_precision,
        "target_vs_distractor_bbox_precision": bbox_precision - distractor_bbox_precision,
        "cluster_fraction": chosen_mask.float().mean(1),
    }


def rank_percentile(scores):
    if scores.shape[1] == 1:
        return torch.ones_like(scores)
    order = scores.argsort(1)
    ranks = torch.empty_like(scores)
    values = torch.arange(scores.shape[1], dtype=scores.dtype, device=scores.device)
    values = values.expand_as(scores)
    ranks.scatter_(1, order, values)
    return ranks / float(scores.shape[1] - 1)


def lexicographic_score(primary, *ties):
    score = primary
    base = float(primary.shape[1] + 1)
    for tie in ties:
        score = score * base + tie
    return score


def choose_rank_balanced_cci(
    mean_score,
    strongest_foil_score,
    same_image_score,
    values,
    cluster_masks,
    inside,
    distractor_inside,
    mode,
):
    mean_rank = rank_percentile(mean_score)
    strongest_rank = rank_percentile(strongest_foil_score)
    same_image_rank = rank_percentile(same_image_score)
    ranks = torch.stack([mean_rank, strongest_rank, same_image_rank], dim=0)
    if mode == "min":
        primary = ranks.amin(0)
    elif mode == "geo":
        primary = ranks.prod(0).clamp_min(0).pow(1.0 / 3.0)
    else:
        raise ValueError(f"unknown rank-balanced CCI mode: {mode}")
    scores = lexicographic_score(primary, same_image_rank, mean_rank)
    return choose_metrics(scores, values, cluster_masks, inside, distractor_inside)


def choose_mean_feasible_strongest_foil(
    mean_score, strongest_foil_score, values, cluster_masks, inside, distractor_inside
):
    feasible = mean_score >= 0
    constrained_score = strongest_foil_score.masked_fill(~feasible, float("-inf"))
    has_feasible = feasible.any(1, keepdim=True)
    scores = torch.where(has_feasible, constrained_score, mean_score)
    return choose_metrics(scores, values, cluster_masks, inside, distractor_inside)


def choose_tail_safe_primary(
    primary_score, strongest_foil_score, values, cluster_masks, inside, distractor_inside
):
    tail_safe = strongest_foil_score >= 0
    safe_primary = primary_score.masked_fill(~tail_safe, float("-inf"))
    has_safe = tail_safe.any(1, keepdim=True)
    scores = torch.where(has_safe, safe_primary, primary_score)
    return choose_metrics(scores, values, cluster_masks, inside, distractor_inside)


def choose_pareto_tail_safe(
    mean_score,
    strongest_foil_score,
    same_image_score,
    values,
    cluster_masks,
    inside,
    distractor_inside,
):
    row = torch.arange(mean_score.shape[0])
    baseline = mean_score.argmax(1)
    baseline_tail_safe = strongest_foil_score[row, baseline] >= 0
    baseline_same_image = same_image_score[row, baseline][:, None]
    feasible = (strongest_foil_score >= 0) & (same_image_score >= baseline_same_image)
    constrained_score = mean_score.masked_fill(~feasible, float("-inf"))
    has_feasible = feasible.any(1)
    repaired = constrained_score.argmax(1)
    selected = torch.where(baseline_tail_safe | ~has_feasible, baseline, repaired)
    score = mean_score[row, selected]
    chosen_mask = cluster_masks[row, selected]
    bbox_precision = (chosen_mask & inside).sum(1).float() / chosen_mask.sum(1).clamp_min(1)
    distractor_bbox_precision = (
        (chosen_mask & distractor_inside).sum(1).float() / chosen_mask.sum(1).clamp_min(1)
    )
    return {
        metric: tensor[row, selected]
        for metric, tensor in values.items()
    } | {
        "selected_cluster": selected.float(),
        "selection_score": score,
        "bbox_precision": bbox_precision,
        "distractor_bbox_precision": distractor_bbox_precision,
        "target_vs_distractor_bbox_precision": bbox_precision - distractor_bbox_precision,
        "cluster_fraction": chosen_mask.float().mean(1),
    }


def summarize(method, seed, metrics):
    return {"method": method, "seed": seed, **{key: float(value.mean()) for key, value in metrics.items()}}


def lambda_method(prefix, value):
    return f"{prefix}_lambda_{value:g}"


def hybrid_method(mean_lambda, max_lambda):
    return f"hybrid_category_foil_cci_mean_{mean_lambda:g}_max_{max_lambda:g}"


def risk_method(topk, risk_lambda):
    return f"risk_category_foil_cci_top_{topk}_lambda_{risk_lambda:g}"


def bootstrap(left, right, comparison, count=1000):
    rng, rows = np.random.default_rng(1701), []
    for metric in left:
        delta = (left[metric] - right[metric]).numpy()
        means = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(count)])
        rows.append({
            "comparison": comparison,
            "metric": metric,
            "mean": float(means.mean()),
            "ci_low": float(np.quantile(means, .025)),
            "ci_high": float(np.quantile(means, .975)),
        })
    return rows


def main():
    args = parse_args()
    foil_lambdas = [float(value) for value in args.foil_lambdas.split(",")]
    hybrid_mean_lambdas = [float(value) for value in args.hybrid_mean_lambdas.split(",")]
    hybrid_max_lambdas = [float(value) for value in args.hybrid_max_lambdas.split(",")]
    cvar_topks = [int(value) for value in args.cvar_topks.split(",")]
    risk_lambdas = [float(value) for value in args.risk_lambdas.split(",")]
    if args.compact_risk_grid and not args.compact:
        raise ValueError("--compact-risk-grid requires --compact")
    if args.compact:
        args.seeds = 0
        foil_lambdas = []
        hybrid_mean_lambdas = [1.0]
        hybrid_max_lambdas = [0.1]
        if args.compact_risk_grid:
            cvar_topks = [1, 3, 5]
            risk_lambdas = [0.1, 0.25]
        else:
            cvar_topks = []
            risk_lambdas = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples, categories = select_dataset_samples(
        args.dataset, args.coco_root, args.samples + args.sample_offset, args.coco_split
    )
    samples = samples[args.sample_offset:]
    features = torch.load(args.feature_cache, map_location="cpu")
    end = args.sample_offset + len(samples)
    if len(features["patch_features"]) < end:
        raise ValueError("Feature cache is smaller than requested deterministic sample window")
    patches = features["patch_features"][args.sample_offset:end].float()
    cluster_masks = load_or_cluster_patches(args, patches)
    inside = transformed_bbox_masks(samples, patches.shape[1])["inside"]
    distractor_inside = transformed_bbox_masks(
        samples, patches.shape[1], boxes_key="distractor_boxes"
    )["inside"]
    if args.compact:
        del features, patches
        gc.collect()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tokenizer, preprocess = load_open_clip(
        args.model, args.pretrained, args.checkpoint, device, force_quick_gelu=args.gelu == "quick"
    )
    scale = float(model.logit_scale.exp().detach().cpu())
    category_ids = sorted(categories)
    category_index = {category_id: index for index, category_id in enumerate(category_ids)}
    selection_category_text = text_features(
        model, tokenizer, [f"a photo of a {categories[category_id]}" for category_id in category_ids], device
    ).cpu()
    evaluation_prompts = [
        f"a picture of the {categories[category_id]}"
        for category_id in category_ids
    ] + [
        f"an image containing a {categories[category_id]}"
        for category_id in category_ids
    ] + [
        f"the {categories[category_id]}"
        for category_id in category_ids
    ]
    evaluation_category_text_by_prompt = text_features(model, tokenizer, evaluation_prompts, device).cpu()
    evaluation_category_text_by_prompt = evaluation_category_text_by_prompt.reshape(3, len(category_ids), -1)
    evaluation_category_text = F.normalize(
        evaluation_category_text_by_prompt.mean(0), dim=-1
    )
    target_index = torch.tensor([category_index[row["target_id"]] for row in samples])
    distractor_index = torch.tensor([category_index[row["distractor_id"]] for row in samples])
    same_image_foil_mask = torch.zeros((len(samples), len(category_ids)), dtype=torch.bool)
    for row, sample in enumerate(samples):
        for category_id in sample["distractor_ids"]:
            same_image_foil_mask[row, category_index[category_id]] = True
    category_foil_mask = torch.ones((len(samples), len(category_ids)), dtype=torch.bool)
    category_foil_mask[torch.arange(len(samples)), target_index] = False
    selection_target_text = selection_category_text[target_index]
    target_text = evaluation_category_text[target_index]
    distractor_text = evaluation_category_text[distractor_index]
    shuffled_lookup = []
    for seed in range(args.seeds):
        generator = torch.Generator().manual_seed(1701 + seed)
        lookup = torch.randperm(len(category_ids), generator=generator)
        while torch.any(lookup == torch.arange(len(category_ids))):
            lookup = torch.randperm(len(category_ids), generator=generator)
        shuffled_lookup.append(lookup)
    loader = DataLoader(CocoDataset(samples, preprocess), batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
    collected = defaultdict(list)
    for indices, images in tqdm(loader, desc=f"CCI-style interventions {args.model}"):
        images = images.to(device)
        if args.compact:
            original = F.normalize(model.encode_image(images).float(), dim=-1).cpu()
        else:
            original, cls_attention, attention_rollout = encode_with_attention_scores(model, images)
        after = encode_with_attention_masks(
            model, images, cluster_masks[indices].to(device), args.block_layers, args.intervention
        ).cpu()
        target, distractor = target_text[indices], distractor_text[indices]
        target_drop = scale * ((original[:, None, :] - after) * target[:, None, :]).sum(2)
        distractor_drop = scale * ((original[:, None, :] - after) * distractor[:, None, :]).sum(2)
        category_drop = scale * ((original[:, None, :] - after) @ evaluation_category_text.T)
        category_mean_drop = category_drop.mean(2)
        same_image_foil_drop = category_drop.masked_fill(
            ~same_image_foil_mask[indices, None, :], float("-inf")
        ).amax(2)
        category_foil_drop = category_drop.masked_fill(
            ~category_foil_mask[indices, None, :], float("-inf")
        ).amax(2)
        category_foil_cvar_drop = {
            topk: category_drop.masked_fill(
                ~category_foil_mask[indices, None, :], float("-inf")
            ).topk(topk, dim=2).values.mean(2)
            for topk in cvar_topks
        }
        values = {
            "target_logit_drop": target_drop,
            "target_vs_distractor_margin_drop": target_drop - distractor_drop,
            "target_vs_category_mean_drop": target_drop - category_mean_drop,
            "target_vs_same_image_foil_max_drop": target_drop - same_image_foil_drop,
            "target_vs_category_max_drop": target_drop - category_foil_drop,
            "target_category_reciprocal_rank": category_reciprocal_rank(
                category_drop, target_drop, category_foil_mask[indices]
            ),
        }
        for topk, foil_drop in category_foil_cvar_drop.items():
            values[f"target_vs_category_cvar_{topk}_drop"] = target_drop - foil_drop
        for prompt_index, prompt_text in enumerate(evaluation_category_text_by_prompt):
            prompt_target = prompt_text[target_index[indices]]
            prompt_distractor = prompt_text[distractor_index[indices]]
            prompt_target_drop = scale * ((original[:, None, :] - after) * prompt_target[:, None, :]).sum(2)
            prompt_distractor_drop = scale * (
                (original[:, None, :] - after) * prompt_distractor[:, None, :]
            ).sum(2)
            prompt_category_drop = scale * ((original[:, None, :] - after) @ prompt_text.T)
            prompt_category_mean_drop = prompt_category_drop.mean(2)
            prompt_same_image_foil_drop = prompt_category_drop.masked_fill(
                ~same_image_foil_mask[indices, None, :], float("-inf")
            ).amax(2)
            prompt_category_foil_drop = prompt_category_drop.masked_fill(
                ~category_foil_mask[indices, None, :], float("-inf")
            ).amax(2)
            prompt_category_foil_cvar_drop = {
                topk: prompt_category_drop.masked_fill(
                    ~category_foil_mask[indices, None, :], float("-inf")
                ).topk(topk, dim=2).values.mean(2)
                for topk in cvar_topks
            }
            values[f"prompt_{prompt_index}_target_vs_distractor_margin_drop"] = (
                prompt_target_drop - prompt_distractor_drop
            )
            values[f"prompt_{prompt_index}_target_vs_category_mean_drop"] = (
                prompt_target_drop - prompt_category_mean_drop
            )
            values[f"prompt_{prompt_index}_target_vs_same_image_foil_max_drop"] = (
                prompt_target_drop - prompt_same_image_foil_drop
            )
            values[f"prompt_{prompt_index}_target_vs_category_max_drop"] = (
                prompt_target_drop - prompt_category_foil_drop
            )
            values[f"prompt_{prompt_index}_target_category_reciprocal_rank"] = (
                category_reciprocal_rank(
                    prompt_category_drop,
                    prompt_target_drop,
                    category_foil_mask[indices],
                )
            )
            for topk, foil_drop in prompt_category_foil_cvar_drop.items():
                values[f"prompt_{prompt_index}_target_vs_category_cvar_{topk}_drop"] = (
                    prompt_target_drop - foil_drop
                )
        if not args.compact:
            generic_drift = 1.0 - torch.einsum("nkd,nd->nk", after, original)
        selection_target_drop = scale * (
            (original[:, None, :] - after) * selection_target_text[indices, None, :]
        ).sum(2)
        selection_category_drop = scale * (
            (original[:, None, :] - after) @ selection_category_text.T
        )
        selection_category_mean_drop = selection_category_drop.mean(2)
        selection_same_image_foil_drop = selection_category_drop.masked_fill(
            ~same_image_foil_mask[indices, None, :], float("-inf")
        ).amax(2)
        selection_category_foil_drop = selection_category_drop.masked_fill(
            ~category_foil_mask[indices, None, :], float("-inf")
        ).amax(2)
        selection_category_foil_mean_drop = selection_category_drop.masked_fill(
            ~category_foil_mask[indices, None, :], 0.0
        ).sum(2) / category_foil_mask[indices].sum(1, keepdim=True)
        selection_category_foil_cvar_drop = {
            topk: selection_category_drop.masked_fill(
                ~category_foil_mask[indices, None, :], float("-inf")
            ).topk(topk, dim=2).values.mean(2)
            for topk in cvar_topks
        }
        # Including the target in the category mean only multiplies the
        # target-minus-foil-mean score by (C - 1) / C, so the region ranking
        # is exactly the same as the foil-only mean selector in the paper.
        contrastive_selection_drop = selection_target_drop - selection_category_mean_drop
        category_mean_foil_selection_drop = (
            selection_target_drop - selection_category_foil_mean_drop
        )
        same_image_foil_selection_drop = selection_target_drop - selection_same_image_foil_drop
        category_foil_selection_drop = selection_target_drop - selection_category_foil_drop
        tail_debt_selection_drop = selection_category_foil_drop - selection_category_foil_mean_drop
        debt_cci_selection_drop = same_image_foil_selection_drop - tail_debt_selection_drop
        robust_cci_selection_drop = torch.minimum(
            same_image_foil_selection_drop,
            category_foil_selection_drop,
        )
        mean_feasible_strongest_foil_selected = choose_mean_feasible_strongest_foil(
            category_mean_foil_selection_drop,
            category_foil_selection_drop,
            values,
            cluster_masks[indices],
            inside[indices],
            distractor_inside[indices],
        )
        tail_safe_mean_foil_selected = choose_tail_safe_primary(
            category_mean_foil_selection_drop,
            category_foil_selection_drop,
            values,
            cluster_masks[indices],
            inside[indices],
            distractor_inside[indices],
        )
        tail_safe_same_image_foil_selected = choose_tail_safe_primary(
            same_image_foil_selection_drop,
            category_foil_selection_drop,
            values,
            cluster_masks[indices],
            inside[indices],
            distractor_inside[indices],
        )
        pareto_tail_safe_selected = choose_pareto_tail_safe(
            category_mean_foil_selection_drop,
            category_foil_selection_drop,
            same_image_foil_selection_drop,
            values,
            cluster_masks[indices],
            inside[indices],
            distractor_inside[indices],
        )
        pareto_rank_selected = choose_rank_balanced_cci(
            category_mean_foil_selection_drop,
            category_foil_selection_drop,
            same_image_foil_selection_drop,
            values,
            cluster_masks[indices],
            inside[indices],
            distractor_inside[indices],
            mode="min",
        )
        geo_rank_selected = choose_rank_balanced_cci(
            category_mean_foil_selection_drop,
            category_foil_selection_drop,
            same_image_foil_selection_drop,
            values,
            cluster_masks[indices],
            inside[indices],
            distractor_inside[indices],
            mode="geo",
        )
        img_cci_selected = choose_metrics(
            same_image_foil_selection_drop,
            values,
            cluster_masks[indices],
            inside[indices],
            distractor_inside[indices],
        )
        debt_cci_selected = choose_metrics(
            debt_cci_selection_drop,
            values,
            cluster_masks[indices],
            inside[indices],
            distractor_inside[indices],
        )
        robust_cci_selected = choose_metrics(
            robust_cci_selection_drop,
            values,
            cluster_masks[indices],
            inside[indices],
            distractor_inside[indices],
        )
        target_selected = choose_metrics(
            selection_target_drop, values, cluster_masks[indices], inside[indices], distractor_inside[indices]
        )
        contrastive_selected = choose_metrics(
            contrastive_selection_drop, values, cluster_masks[indices], inside[indices], distractor_inside[indices]
        )
        selected_methods = [
            ("target_cci", target_selected),
            ("contrastive_cci", contrastive_selected),
            ("mean_feasible_strongest_foil_cci", mean_feasible_strongest_foil_selected),
            ("tail_safe_mean_foil_cci", tail_safe_mean_foil_selected),
            ("tail_safe_same_image_foil_cci", tail_safe_same_image_foil_selected),
            ("pareto_tail_safe_cci", pareto_tail_safe_selected),
            ("pareto_rank_cci", pareto_rank_selected),
            ("geo_rank_cci", geo_rank_selected),
            ("img_cci", img_cci_selected),
            ("debt_cci", debt_cci_selected),
            ("robust_cci", robust_cci_selected),
        ]
        if not args.compact:
            patch_text_score = torch.einsum(
                "npd,nd->np", patches[indices], selection_target_text[indices]
            )
            similarity_score = aggregate_patch_scores(patch_text_score, cluster_masks[indices])
            category_mean_foil_selected = choose_metrics(
                category_mean_foil_selection_drop, values, cluster_masks[indices], inside[indices], distractor_inside[indices]
            )
            same_image_foil_selected = choose_metrics(
                same_image_foil_selection_drop, values, cluster_masks[indices], inside[indices], distractor_inside[indices]
            )
            category_foil_selected = choose_metrics(
                category_foil_selection_drop, values, cluster_masks[indices], inside[indices], distractor_inside[indices]
            )
            drift_selected = choose_metrics(
                generic_drift, values, cluster_masks[indices], inside[indices], distractor_inside[indices]
            )
            similarity_selected = choose_metrics(
                similarity_score, values, cluster_masks[indices], inside[indices], distractor_inside[indices]
            )
            cls_attention_selected = choose_metrics(
                aggregate_patch_scores(cls_attention, cluster_masks[indices]), values, cluster_masks[indices],
                inside[indices], distractor_inside[indices]
            )
            rollout_selected = choose_metrics(
                aggregate_patch_scores(attention_rollout, cluster_masks[indices]), values, cluster_masks[indices],
                inside[indices], distractor_inside[indices]
            )
            selected_methods.extend([
                ("category_mean_foil_cci", category_mean_foil_selected),
                ("same_image_foil_cci", same_image_foil_selected),
                ("category_foil_cci", category_foil_selected),
                ("patch_text_similarity", similarity_selected),
                ("text_free_embedding_drift", drift_selected),
                ("cls_patch_attention", cls_attention_selected),
                ("attention_rollout", rollout_selected),
            ])
        for method, metrics in selected_methods:
            for metric, tensor in metrics.items():
                collected[(method, None, metric)].append(tensor)
        for foil_lambda in foil_lambdas:
            for prefix, foil_drop in [
                ("same_image_foil_cci", selection_same_image_foil_drop),
                ("category_foil_cci", selection_category_foil_drop),
            ]:
                method = lambda_method(prefix, foil_lambda)
                selected = choose_metrics(
                    selection_target_drop - foil_lambda * foil_drop,
                    values,
                    cluster_masks[indices],
                    inside[indices],
                    distractor_inside[indices],
                )
                for metric, tensor in selected.items():
                    collected[(method, None, metric)].append(tensor)
        for mean_lambda in hybrid_mean_lambdas:
            for max_lambda in hybrid_max_lambdas:
                method = hybrid_method(mean_lambda, max_lambda)
                selected = choose_metrics(
                    selection_target_drop
                    - mean_lambda * selection_category_foil_mean_drop
                    - max_lambda * selection_category_foil_drop,
                    values,
                    cluster_masks[indices],
                    inside[indices],
                    distractor_inside[indices],
                )
                for metric, tensor in selected.items():
                    collected[(method, None, metric)].append(tensor)
        for topk, foil_drop in selection_category_foil_cvar_drop.items():
            for risk_lambda in risk_lambdas:
                method = risk_method(topk, risk_lambda)
                selected = choose_metrics(
                    selection_target_drop
                    - selection_category_foil_mean_drop
                    - risk_lambda * foil_drop,
                    values,
                    cluster_masks[indices],
                    inside[indices],
                    distractor_inside[indices],
                )
                for metric, tensor in selected.items():
                    collected[(method, None, metric)].append(tensor)
        for seed in range(args.seeds):
            generator = torch.Generator().manual_seed(1701 + seed + int(indices[0]) * 1009)
            shuffled = shuffled_lookup[seed][target_index[indices]]
            shuffled_drop = scale * (
                (original[:, None, :] - after) * selection_category_text[shuffled, None, :]
            ).sum(2)
            random_score = torch.rand(shuffled_drop.shape, generator=generator)
            for method, score in [("shuffled_target", shuffled_drop), ("random_cluster", random_score)]:
                selected = choose_metrics(
                    score, values, cluster_masks[indices], inside[indices], distractor_inside[indices]
                )
                for metric, tensor in selected.items():
                    collected[(method, seed, metric)].append(tensor)

    metrics_by_method = defaultdict(dict)
    for (method, seed, metric), tensors in collected.items():
        metrics_by_method[(method, seed)][metric] = torch.cat(tensors)
    rows = [summarize(method, seed, metrics) for (method, seed), metrics in metrics_by_method.items()]
    control_mean = {}
    if not args.compact:
        for method in ["shuffled_target", "random_cluster"]:
            control_mean[method] = {
                metric: torch.stack([metrics_by_method[(method, seed)][metric] for seed in range(args.seeds)]).mean(0)
                for metric in metrics_by_method[(method, 0)]
            }
    target = metrics_by_method[("target_cci", None)]
    boot_rows = bootstrap(metrics_by_method[("contrastive_cci", None)], target, "contrastive_cci_minus_target_cci")
    if not args.compact:
        boot_rows.extend(bootstrap(metrics_by_method[("category_mean_foil_cci", None)], target, "category_mean_foil_cci_minus_target_cci"))
        boot_rows.extend(bootstrap(metrics_by_method[("same_image_foil_cci", None)], target, "same_image_foil_cci_minus_target_cci"))
        boot_rows.extend(bootstrap(metrics_by_method[("category_foil_cci", None)], target, "category_foil_cci_minus_target_cci"))
    for foil_lambda in foil_lambdas:
        for prefix in ["same_image_foil_cci", "category_foil_cci"]:
            method = lambda_method(prefix, foil_lambda)
            boot_rows.extend(
                bootstrap(metrics_by_method[(method, None)], target, f"{method}_minus_target_cci")
            )
    for mean_lambda in hybrid_mean_lambdas:
        for max_lambda in hybrid_max_lambdas:
            method = hybrid_method(mean_lambda, max_lambda)
            boot_rows.extend(
                bootstrap(metrics_by_method[(method, None)], target, f"{method}_minus_target_cci")
            )
    boot_rows.extend(
        bootstrap(
            metrics_by_method[("mean_feasible_strongest_foil_cci", None)],
            target,
            "mean_feasible_strongest_foil_cci_minus_target_cci",
        )
    )
    for method in [
        "tail_safe_mean_foil_cci",
        "tail_safe_same_image_foil_cci",
        "pareto_tail_safe_cci",
        "pareto_rank_cci",
        "geo_rank_cci",
        "img_cci",
        "debt_cci",
        "robust_cci",
    ]:
        boot_rows.extend(
            bootstrap(metrics_by_method[(method, None)], target, f"{method}_minus_target_cci")
        )
    for topk in cvar_topks:
        for risk_lambda in risk_lambdas:
            method = risk_method(topk, risk_lambda)
            boot_rows.extend(
                bootstrap(metrics_by_method[(method, None)], target, f"{method}_minus_target_cci")
            )
    if not args.compact:
        boot_rows.extend(bootstrap(metrics_by_method[("patch_text_similarity", None)], target, "patch_text_similarity_minus_target_cci"))
        boot_rows.extend(bootstrap(metrics_by_method[("text_free_embedding_drift", None)], target, "text_free_embedding_drift_minus_target_cci"))
        boot_rows.extend(bootstrap(metrics_by_method[("cls_patch_attention", None)], target, "cls_patch_attention_minus_target_cci"))
        boot_rows.extend(bootstrap(metrics_by_method[("attention_rollout", None)], target, "attention_rollout_minus_target_cci"))
    for method, metrics in control_mean.items():
        boot_rows.extend(bootstrap(metrics, target, f"{method}_mean_minus_target_cci"))
    frame, boot = pd.DataFrame(rows), pd.DataFrame(boot_rows)
    rank_divergence = frame[frame["seed"].isna()].copy()
    rank_divergence["bbox_precision_rank"] = rank_divergence["bbox_precision"].rank(
        method="min", ascending=False
    )
    rank_divergence["category_foil_purity_rank"] = rank_divergence[
        "target_vs_category_max_drop"
    ].rank(method="min", ascending=False)
    rank_divergence["rank_divergence"] = (
        rank_divergence["bbox_precision_rank"] - rank_divergence["category_foil_purity_rank"]
    )
    rank_divergence = rank_divergence[[
        "method",
        "bbox_precision",
        "distractor_bbox_precision",
        "target_vs_distractor_bbox_precision",
        "target_vs_same_image_foil_max_drop",
        "target_vs_category_max_drop",
        "target_category_reciprocal_rank",
        "bbox_precision_rank",
        "category_foil_purity_rank",
        "rank_divergence",
    ]]
    sample_path = args.out_dir / "per_sample_metrics.csv"
    first_sample_frame = True
    sample_index = np.arange(len(samples))
    target_ids = [sample["target_id"] for sample in samples]
    distractor_ids = [sample["distractor_id"] for sample in samples]
    for (method, seed), metrics in metrics_by_method.items():
        sample_frame = pd.DataFrame({
            "sample_index": sample_index,
            "method": method,
            "seed": np.nan if seed is None else seed,
            "target_id": target_ids,
            "distractor_id": distractor_ids,
            **{metric: values.numpy() for metric, values in metrics.items()},
        })
        sample_frame.to_csv(
            sample_path,
            index=False,
            mode="w" if first_sample_frame else "a",
            header=first_sample_frame,
        )
        first_sample_frame = False
    frame.to_csv(args.out_dir / "selector_summary.csv", index=False)
    boot.to_csv(args.out_dir / "bootstrap_ci.csv", index=False)
    rank_divergence.to_csv(args.out_dir / "selector_rank_divergence.csv", index=False)
    (args.out_dir / "report.md").write_text(
        f"# {args.dataset.upper()} CCI-Style Semantic Specificity Audit\n\n"
        f"Intervention: `{args.intervention}`. Attention blocking layers: `{args.block_layers}`.\n\n"
        "Final CLIP patch embeddings are clustered per image. For each cluster, attention access to its key/value positions is blocked in every visual transformer layer. Cluster rankings are compared using target-text score drop, text-free embedding drift, shuffled target text, and random cluster controls.\n\n"
        + frame.to_markdown(index=False)
        + "\n\n## Localization Versus Category-Foil Purity Rank\n\n"
        + rank_divergence.to_markdown(index=False)
        + "\n\n## Bootstrap\n\n"
        + boot.to_markdown(index=False)
        + "\n"
    )
    print(frame.to_string(index=False))
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()

