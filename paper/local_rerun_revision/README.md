# Verified local-rerun manuscript integration — source draft

Status: edited source, NOT compiled or submission-ready. No new PDF is included.
The five-page constraint remains a required release gate; page count, overflow,
visual layout and PDF equivalence have NOT been checked in this session.

## Provenance
- Untouched manuscript base: 17c6ca5fb254e2eea0c98d479509aaf38af558c5, paper/evidence_base/source.
- Experiment run: 59fc00e3948015379c8dcc563283a5bdda7f1226.
- Statistical verification: 1cdffa7abdca1f84c1120362fbaf04d9fb52d13b, experiments/verification_2026-09-13.
- Verification report and corrected CSVs remain at that immutable commit.

## Changes
- Table 2 now gives all four local-rerun CCI screening pass, joint failure and conditional failure rates; conditional 95% intervals are shown.
- Table 3 now reports direct paired WF-minus-Mean and WF-minus-Max-.1 normalized margin differences and pointwise intervals, scaled by 1000.
- Table 6 now reports local switch counts and raw target/bbox changes with switch-subset intervals. Bbox is expressed in percentage points.
- Added local feasible-set distributions, distinct improvement/sign-repair counts, resampling definitions, multiple-comparison limitation, and observed frozen/rerun differences.
- Updated abstract and conclusion to distinguish positive margin gains from loss-free superiority.
- Original Table 1, foil/tolerance controls and figure data remain frozen-archive evidence. New CIs are never combined with frozen point estimates.
- Original descriptive baseline table remains recoverable in evidence_base.
- All six authors, bibliography (including IDEA), equations and three figure assets are retained. Captions add source labels as needed.
- source/ contains the complete editable project; arxiv_source/ contains matching main.tex and required compilation inputs.
- Original source/main.pdf was deliberately excluded from this draft to avoid presenting an old PDF as the revised manuscript.

## Remaining scientific limits
COCO B/16 screen-state changes involve images 135671, 185335 and 213809; ten CCI
region labels change. Some score differences exceed rounding. Missing frozen
candidate arrays prevent causal attribution. Region labels alone do not identify
matching masks across runs. COCO B/32 switches 614 versus frozen 613; missing
frozen per-image fields prevent identifying the changed samples. These are not
claimed to be MPS-only effects or exact reproduction.

Restricted-foil conditional failure rates and new K/intervention robustness
experiments remain outside the completed local-rerun evidence.

## Verification performed here
Read the uploaded base main.tex, checked references and citation keys, checked
matching LaTeX begin/end environments, and generated table values from the
corrected CSVs. Source/arXiv main.tex contents are identical by construction.
Figures, author file and bibliography are reused by their existing Git blob IDs.
This is source-level checking only; it does not establish successful compilation.
