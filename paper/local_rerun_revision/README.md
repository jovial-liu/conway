# Completed manuscript integration

Revised five-page paper: four pages of manuscript plus one references page.
Both source/ and arxiv_source/ were compiled with pdfLaTeX + BibTeX and compared
by rendered-page hashes. Five rendered pages were reviewed. See generated/build_report.json
for the actual current build and release status; only status=passed denotes a completed build.

## Downloads in generated/
- paper_final.pdf: revised complete paper
- arxiv_preview.pdf: PDF from arxiv_source/
- arxiv_source.zip: compilation inputs only; not submitted to arXiv
- paper_complete.zip: source, final PDFs, arXiv package, corrected statistical summaries and provenance/verification notes
- build_report.json: compile, page count, warning, rendered-equivalence and release-file hashes

## Scientific provenance
Original source and figures: 17c6ca5fb254e2eea0c98d479509aaf38af558c5.
Local model rerun: 59fc00e3948015379c8dcc563283a5bdda7f1226.
Statistical verification: 1cdffa7abdca1f84c1120362fbaf04d9fb52d13b.
Full per-image records remain under experiments/local_rerun_2026-09-13/rerun_workspace
at the immutable experiment commit; the complete paper bundle includes corrected
summary evidence and verification code, not a duplicate of the large response CSVs.

## Changes
Tables 2, 3 and 6 now report local-rerun screening, direct paired margin differences
and switch-conditioned target/locality changes with CIs.
Feasible-set distributions and separate margin-improvement/sign-repair counts
were added. Frozen Table 1 and foil/tolerance controls retain their original
provenance. All three figures, six-author file, bibliography including IDEA,
and mathematical definitions were preserved.
Introduction and repeated archived-result prose were condensed to meet five pages;
font sizes and figure assets were not reduced.

## Limits retained in the paper
COCO B/16 has three changed screen/failure-status images and ten changed CCI
region labels. COCO B/32 has 614 local switches versus frozen 613. Missing
frozen candidate/per-image data prevent causal explanation. These are separate
runs, not exact reproduction or demonstrated MPS-only errors.
Direct margin intervals are pointwise, without multiplicity correction;
gains over Max-.1 are small and locality is not uniformly preserved.
Restricted-foil conditional failure rates and new K/intervention robustness
are not supplied by the local run.

No manuscript was edited by the local experimental Codex for this release.
Nothing has been merged into main or submitted to arXiv.
