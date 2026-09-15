# Verification and scope

Input: author-corrected, layout-reviewed Round-5 expanded source at branch commit f1345f7. Evidence lineage remains local rerun 1817e1666161fa902869affe61b55f7c98c45668 and restricted-foil controls 6a52e8f3d39f343d8483390882dcb5e57003d904.

`generated/robustness_checks.json` records hashes and per-setting outputs for 59,302 stored image-model records. `scripts/robustness.py` generates Table 3 and Table 6. Thresholds retain original A, budgets retain original B, and all-image target/bbox averages are separate. These are post-hoc descriptive checks, not new model inference, independent samples across checkpoints, or new confidence intervals. The original Figure 2 and Table 1/2/4/5 are retained. The old single-case panel and single-setting budget table are no longer in the manuscript.

All five budget points are cross-checked against the independently generated Round-5 evidence report. Threshold counts are additionally checked by scalar CSV analysis. Raw worst-foil values cannot be reconstructed from stored raw target values alone. Original spatial identities cannot be proved without original masks.

Build gates: five pages, four technical pages and references on page five; embedded/subset fonts, no Type 3 fonts, undefined references or overfull boxes. Editable and expanded arXiv trees are compiled separately and all rendered pages must match. Visual inspection is recorded separately. This is not conference document-checker certification.

## Bibliography spot-check (2026-09-15)

- CCI title/authors/year and method description confirmed in https://arxiv.org/abs/2511.12978 . Citation now uses the verified 2025 preprint; this pass did not establish the inherited CVPR 2026 page range.
- CASE title/authors and contrastive-saliency description confirmed in https://arxiv.org/abs/2506.07327 .
- CDA-CLIP publisher page retrieved: https://link.springer.com/article/10.1007/s44443-026-00779-3 .
- Access to the MGA-CLIP DOI and the Contrastive Concept Importance arXiv page failed; those inherited entries still require direct metadata verification. Failed access is not proof that a paper does not exist.
- Remaining entries were inherited; a complete per-DOI bibliography audit is outstanding.
