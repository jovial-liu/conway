# Local rerun experiment package

Status: complete local rerun. All four requested settings were run on the MacBook M4 Pro with PyTorch MPS. New outputs are marked `local_rerun` and are independent of the accepted frozen package; no paper, LaTeX, PDF, or figure was modified.

## Completed settings

| setting | images | device | measured time | measured speed |
|---|---:|---|---:|---:|
| COCO OpenAI ViT-B/16 | 27,708 | MPS | 8,435.14 s | 3.281 img/s |
| COCO OpenAI ViT-B/32 | 27,708 | MPS | 565.15 s for the resumed portion | 5.549 img/s |
| VOC2007 OpenAI ViT-B/16 | 1,943 | MPS | 430.09 s | 4.518 img/s |
| VOC2007 OpenAI ViT-B/32 | 1,943 | MPS | 190.30 s | 10.210 img/s |

The COCO B/32 run resumed a previously completed local prefix; its metadata records the resumed portion and the final row count is independently checked at 27,708. Every full-run `per_image.csv` has consecutive unique sample indices, the expected number of rows, and finite numeric metrics.

## Protocol retained

The local runner follows the archived protocol and selector source:

- OpenAI CLIP ViT-B/16 and ViT-B/32 weights; COCO val2014 and VOC2007 test.
- Selection prompt `a photo of a {category}`; held-out prompts `a picture of the {category}`, `an image containing a {category}`, and `the {category}`.
- Resize the short side to 224, center crop to 224, and use patch-center geometry for bbox precision. The patch grids are 14x14 for B/16 and 7x7 for B/32.
- K-means candidate regions use K=8, `n_init=3`, `max_iter=50`, and per-image seed `1701 + sample_index`.
- Intervention masks the selected patch key/value columns with `-inf` in the visual attention blocks/heads while retaining CLS.
- Epsilon is `.02`; ties use the first argmax. Selection and held-out evaluation remain separate.

The strategy formulas were read from the archived source, rather than inferred from display names. With `st` as selection target drop and `sf` as the foil vector, CCI-top1 is `argmax(st)`. The epsilon-feasible set is `st >= CCI_target - .02`. WF is `argmax(st - max(sf))`, Mean is `argmax(st - mean(sf))`, and the display shorthand `Max-.1` is `argmax(st - .1 * max(sf))`. Thus `WF - Max-.1` is a comparison of two strategies, not a strategy name.

## Statistics

`results/paired_difference_ci_10000.csv` uses direct same-image differences and 10,000 image-level bootstrap draws. It reports both `target_drop_raw` and `target_drop_norm`, plus normalized aggregate worst-foil margin and bbox precision. CI fields state the seed, percentile interval, bootstrap unit, and valid sample count. The screening conditional failure CI jointly resamples the numerator and denominator. Switch-subset CIs are computed from the switched-image vectors themselves, not from a full-sample CI divided by a switch rate.

Observed local switch counts are `663 / 614 / 50 / 38` for COCO B/16, COCO B/32, VOC B/16, and VOC B/32. The frozen references are `663 / 613 / 50 / 38`; the COCO B/32 difference of one is retained as an observed rerun difference and was not corrected or forced to match. Local sign-repair counts are `42 / 48 / 11 / 6` in the same order. Margin improvement and sign repair are separate fields and counts.

## Contents

- `runs/<setting>/per_image.csv`: one row per image, including all candidate selection/held-out response arrays, selected regions, feasible-set size, raw and normalized target drops, normalized aggregate worst-foil margin, and bbox precision for CCI, WF, Mean, and Max-.1.
- `runs/<setting>/run_metadata.json` and `run.log`: actual command, device, versions, timing, and progress.
- `results/`: paired per-image differences, 10,000-bootstrap CIs, screening/failure rates, feasible-set distributions, switch-subset summaries and quantiles, and `completed_and_missing.csv`.
- `config/local_rerun_config.json`: scientific and runtime configuration.
- `manifest/source_and_hashes.csv`: SHA256 for scripts, protocol/source records, model files, downloaded archives where retained, extracted COCO annotations, and the frozen package reference.
- `manifest/dependency_versions.txt`: Python/platform/MPS information and `pip freeze`.
- `scripts/run_local_rerun.py`, `scripts/analyze_local_rerun.py`, and `scripts/write_provenance.py`.

The large public datasets and model weight files are not duplicated in this results package. Their official URLs, local paths, and SHA256 values are recorded in the manifest. The original `experiment_results.zip` was treated as read-only and was not overwritten.

## Validation notes

Small CPU/MPS checks used the same images and configuration before each full setting. Region choices were identical. Maximum metric differences were approximately `1.10e-5` (COCO B/16), `1.94e-5` (COCO B/32), `2.05e-5` (VOC B/16), and `3.22e-5` (VOC B/32); bbox precision was identical in each check. The terminal emitted PyTorch TorchScript-loading and sklearn K-means runtime warnings; no full-run output contained missing rows or non-finite numeric metrics. These warnings are preserved in the run context and should be considered when reproducing.

No unresolved input is required for the four completed settings.
