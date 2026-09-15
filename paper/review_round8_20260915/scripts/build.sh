#!/usr/bin/env bash
set -euo pipefail
paper_root="$(cd "$(dirname "$0")/.." && pwd)"
export SOURCE_DATE_EPOCH=1789430400
export FORCE_SOURCE_DATE=1
mkdir -p "$paper_root/generated/build" "$paper_root/arxiv_source"
# Keep the arXiv submission tree expanded, with exactly the current compilation inputs.
cp "$paper_root/source/"*.tex "$paper_root/source/icassp2027_paperkit.sty" "$paper_root/arxiv_source/"
mkdir -p "$paper_root/arxiv_source/tables" "$paper_root/arxiv_source/figures"
cp "$paper_root/source/tables/"*.tex "$paper_root/arxiv_source/tables/"
cp "$paper_root/source/figures/figure1_method_clean.pdf" "$paper_root/source/figures/figure2_repair_capacity.pdf" "$paper_root/arxiv_source/figures/"
for tree in source arxiv_source; do
  (
    cd "$paper_root/$tree"
    latexmk -g -pdf -interaction=nonstopmode -halt-on-error \
      -outdir="$paper_root/generated/build/$tree" main.tex \
      > "$paper_root/generated/${tree}_build.log" 2>&1
  )
done
cp "$paper_root/generated/build/source/main.pdf" "$paper_root/generated/ccf0_round8_20260915.pdf"
pdfinfo "$paper_root/generated/ccf0_round8_20260915.pdf" > "$paper_root/generated/pdfinfo.txt"
pdffonts "$paper_root/generated/ccf0_round8_20260915.pdf" > "$paper_root/generated/pdffonts.txt"
pdftotext -layout "$paper_root/generated/ccf0_round8_20260915.pdf" "$paper_root/generated/paper_text.txt"
python "$paper_root/scripts/check_build.py"
(
  cd "$paper_root"
  sha256sum generated/ccf0_round8_20260915.pdf source/*.tex source/*.sty source/tables/*.tex source/figures/*.pdf \
    arxiv_source/*.tex arxiv_source/*.sty arxiv_source/tables/*.tex arxiv_source/figures/*.pdf \
    > generated/SHA256SUMS.txt
)

python - "$paper_root" <<'PYTHON'
from pathlib import Path
import sys
root = Path(sys.argv[1]) / 'generated'
for name in ['source_build.log', 'arxiv_source_build.log', 'pdfinfo.txt']:
    path = root / name
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + '\n')
PYTHON
