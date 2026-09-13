# Compilation and validation

The paper is compiled on the isolated paper/verified-rerun-integration branch
using .github/workflows/verify-paper.yml. No local Mac TeX installation is required.

Commands for each source directory:
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex

Dependencies: TeX Live latex-extra, fonts-recommended, science; PyMuPDF for
rendering checks. See verify_build.py and the Actions logs for actual execution.

The release gate requires successful builds, five pages with references on page
five, no undefined-reference/citation or overfull warnings, no text outside page
bounds, matching rendered pages, and exact matches to reviewed page hashes.
Five pages have been visually reviewed; visual_review.json records those hashes.
Final bundles are made only when all gates pass. ZIP integrity is also checked.

pdfLaTeX compatibility was tested in this GitHub environment. This is not an arXiv
server-side compilation or a submission; inspect arXiv's generated PDF when uploading.
