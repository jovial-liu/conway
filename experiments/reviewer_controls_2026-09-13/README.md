# Fixed-region restricted-foil controls

This directory contains the reviewer-control experiment for the four local-rerun
settings. It is experiment and statistics output only. No paper source, LaTeX,
PDF, figure, or TeX environment is edited by this package.

## Scientific scope

The fixed input is the local rerun recorded at commit
`1817e1666161fa902869affe61b55f7c98c45668`. For every image, the primary
analysis keeps the CCI-top1 region and the screening set `A` defined under the
full-foil condition:

`A = (cci_bbox_precision >= 0.5) AND (full-foil normalized mean-foil margin > 0)`.

The full foil set is all dataset classes except the target. The restricted set is
all dataset classes except the target and categories annotated as present for
that image. For COCO, the source protocol's non-crowd annotation handling is
retained. “Annotation-absent” means not annotated as present; it does not mean
semantic absence.

The restricted foil experiment never reselects a region, redefines `A`, or runs
a new Mean screen. Restricted-foil failure rates are therefore not conditional
rates from a newly screened restricted dataset.

## Response data

`responses/<setting>/<setting>_cci_full_class_responses.npz` stores, for every
image and all eight candidates, raw and class-normalized aggregate responses for
every dataset class. The three held-out prompts are averaged before class
normalization. Target/annotation IDs and the per-image CCI selection metadata
are also retained in the adjacent `metadata.csv`; the analysis joins that file
when an older NPZ archive omits duplicate metadata fields. The response arrays
are generated from the pinned OpenAI CLIP
ViT-B/16 or ViT-B/32 runner, with the original preprocessing, center crop,
K=8, KMeans `n_init=3`, `max_iter=50`, seed `1701 + sample_index`, attention
key/value masking, and epsilon `.02`.

## Statistics

For each fixed `A` and `Aplus = A AND (cci_target_drop_norm > 0)`, the package
reports full-foil failure, annotation-absent failure, their paired difference,
and the exact cardinality-matched random expectation

`q_i = 1 - C(N_i - m_i, K_i) / C(N_i, K_i)`.

Here `N_i` is the full non-target foil count, `K_i` is the annotation-absent
foil count, and `m_i` is the number of full foils whose response exceeds the
target response. Exact `q_i` is primary; 100 fixed-seed random draws are only a
sanity check. All confidence intervals use 10,000 image-level bootstrap draws,
NumPy `default_rng`, percentile 2.5/97.5 limits, and pointwise intervals with no
multiplicity correction. Paired quantities are formed per image before
resampling.

## Completed local-rerun results

The primary rows are in `results/restricted_foil_failure_rates.csv`; the
matched-cardinality rows and 100-seed Monte Carlo sanity check are in
`results/matched_cardinality_control.csv`. For the primary `A` subset, the
full-failure rate, annotation-absent rate, exact matched-random rate, and
annotation-absent minus full paired difference are:

| setting | n | full | annotation-absent | exact random | absent minus full |
|---|---:|---:|---:|---:|---:|
| COCO B/16 | 17,726 | 0.643687 | 0.631671 | 0.640915 | -0.012016 |
| COCO B/32 | 17,847 | 0.647784 | 0.636409 | 0.645016 | -0.011374 |
| VOC B/16 | 1,209 | 0.422663 | 0.409429 | 0.414619 | -0.013234 |
| VOC B/32 | 1,227 | 0.411573 | 0.397718 | 0.403216 | -0.013855 |

The table values are point estimates; the CSV contains the percentile 95% CIs
and the seed for every interval. `Aplus` counts are 16,855, 17,330, 1,186,
and 1,215 in the same setting order. The independent verifier reports PASS
for all four settings. Fixed A and full-failure counts reproduce exactly;
summary differences against the frozen CSV are all zero above the 1e-5
threshold, with maximum absolute metric differences below 5.5e-6.

All raw response tensors (`.npz` and `.npy`) remain locally with their SHA256
values in `manifest/large_local_artifacts.csv`; the two COCO NPZ archives also
exceed GitHub's 100 MB single-file limit. The compressed per-image diagnostics,
statistics, scripts, metadata, and run logs are the portable GitHub outputs.
The response tensors are not deleted and can be regenerated from the saved run
metadata and adjacent memmaps.

## Independent checks and limitations

`verify_restricted_foils.py` is independent of the main analysis module. It
recomputes full/absent worst foils, all requested fixed counts, `Aplus`, exact
matched-cardinality probabilities, and hardest-foil identities.

The package does not claim to validate the entire CCI heatmap, a new K, another
intervention, or another model. It does not interpret annotation absence as
semantic absence. Any nonzero region or summary mismatch against the pinned
local rerun is reported in `results/full_summary_reproduction.csv`; the analysis
does not silently call such a run an exact replay.

## Reproduction commands

The full-response command is resumable. It imports only the pinned local-rerun
protocol implementation and writes under this new directory:

```bash
python scripts/run_full_class_responses.py \
  --dataset coco --data-root /path/to/rerun_workspace/data/coco \
  --model openai_b16 --weights-dir /path/to/rerun_workspace/model_weights \
  --out-dir responses/coco_openai_b16 --device mps --resume
```

Run it once for each setting, then run:

```bash
python scripts/analyze_restricted_foils.py \
  --fixed-root ../local_rerun_2026-09-13/rerun_workspace
python scripts/verify_restricted_foils.py \
  --fixed-root ../local_rerun_2026-09-13/rerun_workspace
```

The source datasets and model weights are not copied into this directory.
Their paths and SHA256 values are inherited from the pinned local-rerun
manifest and are recorded in `manifest/input_hashes.csv` after generation.
