# Round-8: traceable experiment integration

Current manuscript: `submission/liu_round8_traceable.pdf` (same bytes as `submission/liu.pdf`). Five pages: four technical pages plus references/ethics. Six author ORCIDs retained; no acknowledgments; original Figure 1 preserved.

Primary numerical source: experiments/priority123_20260915 at experiment commit 7ee2b6080aa6752790e17c910c5c8a3111235d4d; handoff commit 78387d76f271d84b508373c603a5eff75c09299b. Tables 1–3,5–6 and Figure 2 use new same-generation records. Table 4 alone fixes the older regions and normalized A, explicitly labeled A_old. Original-logit versus intervention-drop foil distinction and NeurIPS 2022 reference retained.

Run from repository root:

```bash
python paper/review_round8_20260915/scripts/integrate_traceable.py
bash paper/review_round8_20260915/scripts/build.sh
python paper/review_round8_20260915/scripts/prepare_submission.py
```

The integration checks CSV-level counts, choices, normalization rows and supplied paired estimates. Masks/full tensors remain on the author's machine; same-record PASS reports are supplied by the experiment runner, not independently re-executed here. Table 5 CIs are recomputed from new per-image paired endpoint differences with 10,000 bootstrap resamples; seeds documented in generated/evidence_checks.json. The supplied WF–Mean/Max-.1 paired CI file is retained in generated/ for review, outside the six manuscript tables.

Remaining submission facts: conflicts-of-interest confirmation, final author approval and submission category. No conference/arXiv submission or main merge has been performed. No ZIP is needed.
