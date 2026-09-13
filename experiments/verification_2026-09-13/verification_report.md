# Local rerun verification report

Verification date: 2026-09-13  |  baseline commit: `59fc00e3948015379c8dcc563283a5bdda7f1226`

## Scope

This is an independent read-only recomputation from the four local `per_image.csv` files. It does not modify the frozen package, the existing local-rerun results, paper sources, PDF, LaTeX, or figures. No model rerun was performed during this verification pass.

The bootstrap algorithm is the archive algorithm: NumPy `default_rng`, image-level resampling in chunks of 64, 10,000 draws, and percentile 2.5/97.5 intervals. Conditional screening intervals jointly resample the numerator and denominator and recompute their ratio.

## Input completion

| setting | rows | status |
|---|---:|---|
| coco_openai_b16 | 27708 | completed |
| coco_openai_b32 | 27708 | completed |
| voc2007_openai_b16 | 1943 | completed |
| voc2007_openai_b32 | 1943 | completed |

## Recomputed tables

Delivered files include `corrected_paired_difference_ci_10000.csv`, `corrected_screening_failure_rates.csv`, `corrected_switch_subset_changes_ci.csv`, `corrected_switch_margin_improvement_vs_sign_repair.csv`, `corrected_feasible_set_size_distribution.csv`, and the corresponding per-image difference CSV.

Comparison with the existing local summary tables found 0 statistical-value mismatches under tolerance 1e-05; 36 textual definition corrections are listed separately in the comparison CSV. Seed columns that were absent or mislabeled in the existing screening table are supplied explicitly in the corrected table.

## Strategy and candidate-array validation

Rows with any strategy/array validation failure: 0. The check reconstructs CCI argmax and WF/Mean/Max-.1 argmax over the epsilon-feasible candidates, then verifies feasible-set size and every selected held-out summary value against its candidate array entry.

| setting | rows | array/strategy failures | max summary abs diff |
|---|---:|---:|---:|
| coco_openai_b16 | 27708 | 0 | 1.78e-15 |
| coco_openai_b32 | 27708 | 0 | 1.78e-15 |
| voc2007_openai_b16 | 1943 | 0 | 4.44e-16 |
| voc2007_openai_b32 | 1943 | 0 | 1.78e-15 |

## Frozen versus local COCO B/16

The image_id sets align exactly (27708 images, same order). Frozen counts are screen=17725, failure=20516, joint=11411; local counts are screen=17726, failure=20514, joint=11410. Changed-image counts are screen=1, failure=2, joint=3. The complete state-change rows are in `frozen_local_state_changes_coco_b16.csv`.

| image_id | change types | frozen CCI | local CCI | frozen pmax | local margin | frozen bbox | local bbox |
|---:|---|---:|---:|---:|---:|---:|---:|
| 135671 | screen,joint_failure | 6 | 1 | -0.866144 | -0.277032 | 0.078947 | 0.583333 |
| 185335 | failure,joint_failure | 3 | 3 | -0.773712 | 0.004074 | 1.000000 | 1.000000 |
| 213809 | failure,joint_failure | 2 | 2 | -0.069674 | 0.022060 | 0.764706 | 0.775000 |

The 135671 screen change is accompanied by CCI region 6→1 and bbox precision 0.078947→0.583333; the 185335 and 213809 failure changes keep the CCI region but change the normalized worst-foil sign. Their frozen-to-local worst-foil differences are 0.777786 and 0.091734 respectively, so these are not explainable as threshold rounding alone. The frozen candidate arrays and intermediate tensors needed to identify whether the source is model arithmetic, clustering, preprocessing, or another protocol difference are unavailable.

CCI region labels changed for 10 images. These are listed separately in `frozen_local_cci_region_changes_coco_b16.csv`; frozen candidate arrays are not present, so the file cannot by itself distinguish a changed response from changed clustering/region assignment for those label changes.

For the three screen/failure/joint state-change images, the report compares the frozen selected-region scalar fields with the local candidate array and local CCI/held-out scalars. It does not attribute the discrepancies to MPS solely from their size: frozen candidate arrays and the complete frozen B/16 protocol-level intermediate tensors are unavailable.

## Frozen artifact limits

The frozen B/32 artifact contains aggregate sufficient statistics and replay metadata but no image_id-aligned per-image region or candidate-response fields. Therefore the frozen 613-to-local-614 switch change cannot be enumerated or causally decomposed from the available frozen files. This is a data-availability limit, not a claim that the two runs are identical.

| dataset_model      | frozen_per_image_file                                                                                                                   | per_image_available   | candidate_arrays_available   | regions_available   | aggregate_switch_count_available   |   frozen_switch_count | limitation                                                                                                       |
|:-------------------|:----------------------------------------------------------------------------------------------------------------------------------------|:----------------------|:-----------------------------|:--------------------|:-----------------------------------|----------------------:|:-----------------------------------------------------------------------------------------------------------------|
| coco_openai_b16    | /Users/von/Projects/cci-local-recovery/verification_2026-09-13/frozen/derived/reviewer_cci_preserving_v1/01_original_cci_per_sample.csv | True                  | False                        | True                | True                               |                   663 | per-image aggregate/selected-region fields exist; candidate response arrays are absent                           |
| coco_openai_b32    | not found in frozen package                                                                                                             | False                 | False                        | False               | True                               |                   613 | only sufficient_statistics.npz/replay_summary.json are available; no image_id-aligned region or candidate fields |
| voc2007_openai_b16 | not found in frozen package                                                                                                             | False                 | False                        | False               | True                               |                    50 | no frozen per-image fields available for alignment                                                               |
| voc2007_openai_b32 | not found in frozen package                                                                                                             | False                 | False                        | False               | True                               |                    38 | no frozen per-image fields available for alignment                                                               |

## Runtime wording correction

| dataset_model      |   elapsed_seconds_reported |   images_per_second_reported |   completed_before |   completed_this_run |   completed_total | time_scope                                                 |
|:-------------------|---------------------------:|-----------------------------:|-------------------:|---------------------:|------------------:|:-----------------------------------------------------------|
| coco_openai_b16    |                   8435.14  |                      3.28104 |                 32 |                27676 |             27708 | resumed portion only; not complete experiment elapsed time |
| coco_openai_b32    |                    565.149 |                      5.54897 |              24572 |                 3136 |             27708 | resumed portion only; not complete experiment elapsed time |
| voc2007_openai_b16 |                    430.092 |                      4.51764 |                  0 |                 1943 |              1943 | full local run                                             |
| voc2007_openai_b32 |                    190.303 |                     10.21    |                  0 |                 1943 |              1943 | full local run                                             |

## Delivered provenance

`verification_metadata.json` records input paths, SHA256 values, Python/package versions, the baseline commit, and the exact recomputation script. The original configuration wording remains untouched; the corrected interpretation is in `corrected_config_notes.md`.
