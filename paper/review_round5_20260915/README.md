# Current manuscript: Round-5, 2026-09-15

This revision supersedes Round-4 for manuscript editing. All inputs are ordinary files; no ZIP is required or produced.

- PDF: `generated/ccf0_round5_20260915.pdf`
- Editable manuscript: `source/main.tex`
- Author information: `source/authors.tex` (Qihang Wu is the fifth author)
- Expanded arXiv compilation tree: `arxiv_source/`
- SHA256 manifest: `generated/SHA256SUMS.txt` (paths relative to this directory)
- Compilation checks: `generated/build_checks.json`
- Existing-data count/mean checks: `generated/evidence_checks.json`
- Revision decisions and unresolved experiment requests: `CHANGELOG_zh.md`
- Provenance and verification limits: `VERIFICATION.md`

## Build

Run `bash scripts/build.sh` from this directory or any working directory. Requires TeX Live (`latexmk`, PDFLaTeX and the manuscript packages), Poppler and Python with PyMuPDF. It synchronizes the expanded arXiv tree from the editable source, compiles both separately, checks five pages, embedded/subset fonts, references and rendered equivalence, then writes relative SHA256 checksums. Source and arXiv files use an inline bibliography; BibTeX is not needed.

Optional existing-data verification and plot regeneration: `python scripts/verify_and_plot.py` (NumPy and Matplotlib). This reads existing repository CSVs; it does not run CLIP or estimate new confidence intervals. The submitted figure PDFs are already present, so no Python step or experimental data is needed to compile the paper.

## Delivery boundary

Submit only the paper PDF to the conference under the applicable instructions. The scripts, checks and handoff documents are working/reproducibility materials, not asserted to be an accepted supplementary-file submission. The expanded arXiv source is prepared but has not been submitted.
