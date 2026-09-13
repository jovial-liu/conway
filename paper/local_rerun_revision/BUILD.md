# Build and release gates — pending

No LaTeX compiler or local command executor is available in the editing session.
No compilation has been attempted here; no new PDF, five-page claim, or arXiv
platform compatibility claim is made.

In an environment with pdfLaTeX, BibTeX and the package dependencies:
1. Run `sh build.sh` from source/.
2. In arxiv_source/, run pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX on main.tex.
3. Check both logs for undefined references/citations and overfull boxes.
4. Confirm five total pages with references on page five; adjust prose/float
   placement if needed while retaining the agreed font sizes and figures.
5. Render and inspect all pages for clipping, figure placement, table width,
   overlapping text and six-author information.
6. Compare the two rendered PDFs page by page.
7. Only after these checks, export paper_final.pdf and arxiv_preview.pdf and
   prepare an arXiv source archive. Do not submit automatically.

The existing main.bbl and references.bib are retained because citation keys and
order are unchanged. Rebuild the bibliography as part of verification.
