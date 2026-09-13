# ccf0-round2-20260913: exact repair-capacity reanalysis

This entry describes the second review revision, not the earlier `paper/local_rerun_revision/generated/paper_final.pdf`. The revised PDF and executable source/evidence capsule are delivered as ChatGPT attachments named `ccf0_round2_20260913.pdf` and `ccf0_round2_20260913_source.zip`; they have not been committed to the old generated-paper directory. Do not label the old PDF as this revision.

## Immutable input provenance

All analyses read the four local `per_image.csv` files in commit `1817e1666161fa902869affe61b55f7c98c45668`, under `experiments/local_rerun_2026-09-13/rerun_workspace/runs/`. No CLIP inference was run in this revision. No frozen records were substituted for local records.

| Setting | Rows | Input SHA256 |
|---|---:|---|
| coco_openai_b16 | 27708 | bece4e746855dea84bb9135c6bec41edbfbe9b8dcbb26afdcc66e59ebf2ca07f |
| coco_openai_b32 | 27708 | 674a6fb2a83e229bc0bebe9029967f48db960268d880cad9613911d08d6634e3 |
| voc2007_openai_b16 | 1943 | 85832f21c935f736256b0bb9ef1f42d29f4cd91b118a375230aaeb4eb35a877a |
| voc2007_openai_b32 | 1943 | d6ba22e6912f57093dcf733a59e16c77d116e7a99ee1d284ab034dfca352fef1 |

The read-only input packaging workflow is `.github/workflows/package-review-inputs.yml`, run 34750052221. Its artifact is only a transport convenience; the committed CSVs above remain the permanent inputs.

## Main-table map

The accompanying capsule contains `analysis/reanalyze.py`, `analysis/independent_count_check.py`, `analysis/make_tables.py`, the five-page manuscript, and the following result files.

| Paper item | Capsule evidence |
|---|---|
| Table 1: original and positive-target screens | `results/audit.csv` |
| Equation 7 and Table 2: exact oracle decomposition | `results/oracle_decomposition.csv` |
| Table 3: five-tolerance opportunity/cost sweep | `results/tolerance_frontier.csv`, `results/costs.csv` |
| Table 4: paired baselines and disagreement counts | `results/paired_comparisons.csv`, `results/selection_agreement.csv` |
| Table 5: switched-image costs | `results/costs.csv`, population=`switched`, epsilon=0.02 |
| Row-level audit and repair classification | `results/*_per_image_diagnostics.csv` |
| Independent row-loop verification | `results/independent_count_check.json` |

## New results and denominators

`A` is the original CCI-top1 locality/mean-foil screen. `Aplus` additionally requires positive held-out target drop. `F` is negative held-out normalized aggregate worst-foil margin. The primary repair population is the fixed set `B=A AND F`.

| Setting | Failure count / Aplus count | Exact feasible-oracle repairs in B at epsilon=.02 | Actual WF repairs in B |
|---|---:|---:|---:|
| COCO B/16 | 10558/16855 = 62.64% | 18/11410 = 0.1578% | 18 |
| COCO B/32 | 11054/17330 = 63.79% | 31/11561 = 0.2681% | 30 |
| VOC B/16 | 491/1186 = 41.40% | 5/511 = 0.9785% | 5 |
| VOC B/32 | 495/1215 = 40.74% | 3/505 = 0.5941% | 2 |

For all baseline failures instead of B, oracle repair counts are 43/50/11/7 and WF repairs are 42/48/11/6. These denominators are not interchangeable.

The exact mutually exclusive decomposition of B is: no passing candidate among all K; passing candidate excluded by tolerance; feasible passing candidate missed by WF; actual WF repair. The four setting counts are respectively:

- COCO B/16: 10697, 695, 0, 18 (total 11410).
- COCO B/32: 10655, 875, 1, 30 (total 11561).
- VOC B/16: 371, 135, 0, 5 (total 511).
- VOC B/32: 347, 155, 1, 2 (total 505).

WF/Max-.1 disagreements are 84/89/5/5 images. All five tolerances (.01,.02,.05,.10,.20) are re-evaluated from the same stored candidates. On COCO B/16, the oracle repair fraction of B rises from 0.1578% at .02 to 1.5688% at .20; at .20, all-image held-out raw target and bbox changes are -0.013972 and -1.626277 percentage points respectively.

## Computation and limits

The reanalysis reconstructs all four selectors using the original float32 arithmetic and validates every selected summary against its candidate entry. A second stdlib CSV/JSON row-loop implementation independently checks all 60 oracle rows and 20 tolerance rows; zero count mismatches. All 59302 image-model records are used, not 59302 independent images.

Intervals use 10000 percentile image bootstraps and deterministic metric-specific seeds in `results/analysis_manifest.json`. Ratios use exact multinomial grouping of jointly resampled image indicators; continuous means use image-index resampling. These are post-hoc, pointwise intervals with no multiple-comparison adjustment. New bootstrap seeds mean some CI endpoints differ from the earlier revision, although original point estimates and input rows are unchanged.

The capsule does NOT contain new annotation-absent baseline failure rates or foil-cardinality-matched controls. Stored candidate summaries do not recover full per-class responses or masks. Raw worst-foil, segmentation/region-size, and new-K controls remain uncompleted. Exact oracles use held-out information only as diagnostic upper limits, never for the deployed WF selector.
