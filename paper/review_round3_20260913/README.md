# ccf0-round3-20260913: fixed-population restricted-foil audit

This is the third review revision. It integrates the completed restricted-foil baseline controls into the five-page manuscript. The delivered files are ChatGPT conversation attachments, NOT the older PDF still stored under `paper/local_rerun_revision/generated/`. That older PDF has not been overwritten.

## Exact delivered files

| File | Bytes | SHA256 |
|---|---:|---|
| ccf0_round3_20260913.pdf | 4807615 | 76692364c7b106dcc5c76fa65932c428c7bac45dca437e269e59dc0361c00a3e |
| ccf0_round3_20260913_source.zip | 33292870 | 69ea8b77a615bf3dd09b933e9d2a070577dd71b8d348a576a09c6d3e4e08716e |
| ccf0_round3_20260913_arxiv.zip | 4564816 | 59692b459bee0c6a154799e296e4ed648d472cb12bfd11f78b5ae1463aaa9576 |

The source/evidence ZIP has 137 files. Its `source/main.tex` SHA256 is `f9acf27a7522e3916cb2068e1bed9f84017a2e734ee9163a271fad3738acf46f`.

## Provenance and main-table map

Original local candidate inputs: commit `1817e1666161fa902869affe61b55f7c98c45668`, under `experiments/local_rerun_2026-09-13/rerun_workspace/runs/`.
Restricted-foil controls: commit `6a52e8f3d39f343d8483390882dcb5e57003d904`, under `experiments/reviewer_controls_2026-09-13/`.
The read-only transfer workflow ran as 34764597239. It did not rerun models or modify experimental results.

| Paper item | Capsule evidence |
|---|---|
| Eq. 8, Table 1: full / absent / exact matched-cardinality failure | `evidence/reviewer_controls/results/restricted_foil_failure_rates.csv`, `matched_cardinality_control.csv`; `analysis/integrate_controls.py` |
| Positive-target and scene-count checks | Explicit subset=Aplus rows in the primary CSVs; `results/positive_target_subset_verified.csv`, `strata_recheck.csv` |
| Eq. 7, Table 2: exact repair decomposition of original B=A AND F | `evidence/round2/results/oracle_decomposition.csv` |
| Table 3: five-budget opportunity/cost sweep | `evidence/round2/results/tolerance_frontier.csv`, `costs.csv` |
| Table 4: paired margin differences and disagreements | `evidence/round2/results/paired_comparisons.csv`, `selection_agreement.csv` |
| Table 5: switch-conditioned target and bbox costs | `evidence/round2/results/costs.csv` |
| Portable independent control recheck | `analysis/verify_portable_controls.py`, `results/portable_verification_report.json`, `bootstrap_recheck.csv` |

## New primary result

Keep each CCI region and the original full-foil screen A fixed. Annotation-absent excludes categories present in the source loader's retained annotations. Random foils are sampled uniformly without replacement with each image's absent-set size; exact combinatorial expectations, not Monte Carlo estimates, are primary.

| Setting | n_A | Full % | Absent % | Exact random % | Absent minus random, pp [95% CI] |
|---|---:|---:|---:|---:|---:|
| COCO B/16 | 17726 | 64.37 | 63.17 | 64.09 | -0.92 [-1.08,-0.77] |
| COCO B/32 | 17847 | 64.78 | 63.64 | 64.50 | -0.86 [-1.01,-0.72] |
| VOC B/16 | 1209 | 42.27 | 40.94 | 41.46 | -0.52 [-1.16,+0.06] |
| VOC B/32 | 1227 | 41.16 | 39.77 | 40.32 | -0.55 [-1.19,+0.04] |

In Aplus, annotation-absent failure is 61.38/62.61/40.05/39.34%. The exact-oracle population remains ORIGINAL full-foil B=A AND F, not restricted failures. COCO matched-control intervals exclude zero; VOC intervals include zero. Intervals are pointwise and unadjusted. Annotation absence is not verified semantic absence; COCO retains the non-crowd convention.

## Checks actually performed in this revision

- All 40 transported evidence files checked against the pinned transport manifest.
- All 59302 diagnostic rows joined against SHA256-verified original local CSVs; labels and full-foil states match. Regenerated selected summaries differ by at most 5.7220459e-6, below 1e-5.
- Exact random probabilities independently recomputed by a product formula; all 40 reported intervals (120 estimates/endpoints) recomputed with 10000 image-index bootstrap draws and recorded seeds; differences below 1e-12.
- The separate stdlib row-loop checker rerun on all 60 oracle rows and 20 tolerance rows; no count mismatches.
- Five pages, references on page 5; all 26 PDF font resources embedded; abstract 122 whitespace-delimited words; no undefined references, overfull boxes, or page-boundary violations. Standalone source and arXiv inputs actually compiled with identical rendered pages. All five final Poppler page renders visually inspected.
- All three figure PDF assets, author file and bibliography preserved from the second revision. Figure-1 hidden old text remains removed; figure-3 labels remain embedded.

## Source issues, handled without changing original evidence

The source `positive_target_subset.csv` contains both A and Aplus rows but no subset label. The paper uses explicit subset=Aplus rows in the two main result tables; a correctly filtered view is included in the capsule. Metadata prose says masked-minus-original although executed code uses original-minus-masked. The source README's below-5.5e-6 statement overlooks a 5.722e-6 generated pmean discrepancy, still below the declared 1e-5 tolerance. The source output manifest has one stale `analysis.log` hash (recorded while empty). All are documented in `VERIFICATION_NOTES.md`; main statistics were not altered.

Full class-response NPZ/NPY tensors remain local with recorded hashes. The supplied local verifier checked tensor-level results; this cloud session checked portable diagnostics, not unprovided tensors or old candidate masks, and performed no CLIP inference. Public diagnostics do not recover arbitrary class subsets. Raw/normalizer sensitivity, segmentation/region-size and new-K/intervention controls remain outside the completed task.

No arXiv submission, main-branch merge or force push was performed. The binary paper/source packages are conversation downloads; this file is their permanent version/evidence index, not a claim that those binaries were committed here.
