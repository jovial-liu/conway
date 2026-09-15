# GPT-readable Round-4 source index

Use the expanded repository files below directly. **Do not rely on ZIP archives.**

Branch: `paper/verified-rerun-integration`

## Primary editable manuscript

- `paper/review_round4_20260914/source/main.tex`
- `paper/review_round4_20260914/source/authors.tex`
- `paper/review_round4_20260914/source/references.tex`
- `paper/review_round4_20260914/source/icassp2027_paperkit.sty`
- `paper/review_round4_20260914/source/tables/`

The matching expanded arXiv tree is:

- `paper/review_round4_20260914/arxiv_source/`

## Current figures

- `paper/review_round4_20260914/source/figures/figure1_method_clean.pdf`
- `paper/review_round4_20260914/source/figures/figure1_method_clean.svg`
- `paper/review_round4_20260914/source/figures/figure2_identification_final.pdf`
- `paper/review_round4_20260914/source/figures/figure2_identification_final.svg`

Current Round-4 Figure 3 is implemented directly in `source/main.tex` as inline LaTeX; there is no external current Figure-3 image file.

## Previous Figure 2 / Figure 3 assets retained as ordinary GitHub files

These are kept outside ZIP archives so GPT can inspect them directly:

- `paper/review_round4_20260914/reference_figures/figure2_previous_round3.pdf`
- `paper/review_round4_20260914/reference_figures/figure2_previous_round3.svg`
- `paper/review_round4_20260914/reference_figures/figure3_previous_round3.pdf`
- `paper/review_round4_20260914/reference_figures/figure3_previous_round3.svg`
- `paper/review_round4_20260914/reference_figures/figure3_previous_round3_pixels.png`

## Evidence/results

- `experiments/local_rerun_2026-09-13/`
- `experiments/verification_2026-09-13/`
- `experiments/reviewer_controls_2026-09-13/`
- `paper/review_round4_20260914/results/` when present in the current branch

## Rule for future GPT sessions

Read and edit the expanded files above. Do not ask for, depend on, or treat any `.zip` file as the source of truth. ZIP files in repository history are packaging artifacts only.
