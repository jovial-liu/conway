最新审阅修正版：submission/liu_review_fixes_20260917.pdf。已合并人群定义、明确候选域和百分点单位、补充置信区间解释并改写相关工作措辞。

最新公式说明版：submission/liu_formulas_explained_20260917.pdf。每个编号公式前先说明含义与符号，计算定义不变。

最新叙述重写版（2026-09-17）：submission/liu_prose_revised_20260917.pdf。摘要至结论已重写，作者、图件及表格数值不变。

当前修订：2026-09-17。PDF 为 submission/liu_readability_20260917.pdf；Zhenghao Wang 单位为南京理工大学泰州科技学院；已完成可读性、图表分析和 IDEA 删除修订。

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
