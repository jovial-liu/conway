# ccf0-round4-20260914: endpoint-aligned manuscript finalization

This fourth revision supersedes the round-3 manuscript. The current compile-ready source is now materialized directly in GitHub under `source/` and mirrored under `arxiv_source/`; the ZIPs under `generated/` are convenience packages, not the only copy. The older PDF under `paper/local_rerun_revision/generated/` remains obsolete.

## Current GitHub source delivery

| File / directory | Current status |
|---|---|
| `source/main.tex` | editable Round-4 manuscript; SHA256 `02f4f36a6dbf441e835314acf5ef69b8c6c5fb27eecd8b1c363208792ed3973a` |
| `source/figures/` | current Figure 1 and Figure 2 as visible SVG/PDF files |
| `reference_figures/` | previous Figure 2 / Figure 3 PDF/SVG assets as ordinary GitHub files |
| `arxiv_source/` | expanded compile-ready source mirror |
| `generated/ccf0_round4_20260914_source.zip` | 6,335,106 bytes; SHA256 `07be9187a3069cb5e1626702f4f6e8fa888a4ff105e27e61cf912d01a4e4f459` |
| `generated/ccf0_round4_20260914_arxiv.zip` | 270,533 bytes; SHA256 `89c278a7a6b5065b407d438672302a12d30a210222eecb52c08e98de7ec86f99` |
| `generated/SOURCE_SHA256SUMS.txt` | authoritative current source-package hashes |

The repository materializer compiled the expanded source before packaging and verified both ZIPs with `unzip -t`. Current Round-4 Figure 3 is an inline response-audit panel in `main.tex`, so there is no external current Figure-3 image. The previous photographic Figure 3 is retained under `reference_figures/` specifically so future ChatGPT sessions can inspect it without opening a ZIP.

The previously delivered conversation PDF `ccf0_round4_20260914.pdf` was 308,084 bytes with SHA256 `3a38c8d0243413bee18a1a1107d5c73cd797512563e80612d481d6bdabc8baea`. It is a separate conversation artifact; do not substitute the obsolete `paper/local_rerun_revision/generated/paper_final.pdf` for it.

## Inputs and unchanged primary evidence

Original candidate CSVs remain pinned to `1817e1666161fa902869affe61b55f7c98c45668`, under `experiments/local_rerun_2026-09-13/rerun_workspace/runs/`. Restricted-foil evidence remains pinned to `6a52e8f3d39f343d8483390882dcb5e57003d904`, under `experiments/reviewer_controls_2026-09-13/`. No model inference, new K, intervention, raw-normalizer, segmentation, or region-size experiment was run in this revision.

Tables 1–4 retain the prior fixed-A foil controls, exact B=A AND F decomposition, five-budget sweep, and paired selector comparisons. C1 explicitly means at least one passing candidate exists but NO passing candidate is feasible. The discussion quantifies the COCO/VOC difference in unrestricted candidate capacity and the subsequent budget exclusion.

## New Table 5: sign repair versus retained endpoints

All counts below condition on ORIGINAL B=A_CCI AND F_CCI and epsilon=.02. J requires P(WF)>=0, bbox(WF)>=.5, full mean-foil margin(WF)>0, and positive held-out target contribution. Applying the screen to a new region is an outcome, never a redefinition of original B.

| Setting | Sign repairs in B | Repairs also satisfying J | Feasible sign oracle | Feasible J oracle |
|---|---:|---:|---:|---:|
| COCO B/16 | 18 | 12 | 18 | 12 |
| COCO B/32 | 30 | 25 | 31 | 26 |
| VOC B/16 | 5 | 3 | 5 | 3 |
| VOC B/32 | 2 | 1 | 3 | 2 |

Table 5 reports raw target and bbox changes WITHIN these actual B sign repairs, with 10,000-draw within-subset percentile intervals. It is not the earlier all-switched table. The prior 42/48/11/6 repair counts refer to ALL baseline failures and are explicitly distinguished in the text. `ENDPOINT_SUPPLEMENT.md` and `results/repair_subset_costs.csv` separate B-switches, B-sign-repairs, and all-failure sign repairs for all five budgets. Small VOC repair counts are not characterized as stable population effects.

## Figure and PDF changes

Figure 3 now shows the real local Aplus AND F case, image 254807, as an absolute-endpoint audit card. It names the original CCI foil tie and reports each region's own worst-foil margin, mean contrast, target drop, and bbox. No missing spatial masks were fabricated. The old photographic examples remain intact under `reference_figures/` rather than occupying the main figure slot.

Figures 1 and 2 retain their original drawing geometry and colors. Their SVG source files and standalone PDFs are directly visible in `source/figures/`. Figure 1 is re-exported from the visible SVG rather than the old full-page PDF object that carried hidden manuscript text.

The previously delivered final PDF was five US Letter pages, references only on page five, 122 whitespace-delimited abstract tokens, all 25 font resources embedded AND subset, no Type 3 fonts, no undefined references, overfull boxes, out-of-page text or hidden old methods. Main and arXiv sources were separately compiled and had identical rendered pages. All five final Poppler pages were visually reviewed.

## Verification and portable candidate evidence

All 59,302 original image-model records were read after SHA256 checks. A separate stdlib CSV/JSON row-loop implementation, using struct-rounded scalar float32 selection and math.fsum, checked all 60 new endpoint rows and 180 cost means, plus independently replayed the 12 epsilon=.02 B-repair CI rows. Maximum discrepancies: zero. Other new CI rows were computed by the main script, not individually replayed by the second implementation.

Four compact NPZ capsules preserve all eight candidates' selection target/max/mean-foil and held-out margin/mean/target/bbox summaries, image IDs and sample indices. Every array is losslessly round-trip checked. A no-image/no-checkpoint verifier recomputes all 60 original oracle rows and 60 new endpoint rows from these capsules, with no mismatches.

These are candidate SUMMARY capsules, not full per-class tensors or masks. They improve independent reconstruction of Table 2 and endpoint counts; selected-region full-class vectors are still in the local response archives. Arbitrary restricted-foil reconstruction from public diagnostics alone remains unsupported. The inherited round-3 source-metadata notes and verification outputs are preserved.

The package maps all five tables to input files, scripts, seeds and checks. No arXiv submission, main merge, force push, or experimental input overwrite was performed.
