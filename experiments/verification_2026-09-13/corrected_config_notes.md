# Corrections to configuration wording

The original `rerun_workspace/config/local_rerun_config.json` is preserved.
This verification directory supplies the corrected interpretation used by
the independent recomputation.

## Direct paired baselines

`WF-Mean` and `WF-Max-.1` are direct, same-image comparisons on the same
epsilon-feasible candidate set and held-out evaluation: `WF(metric) -
baseline(metric)`. They are not `candidate - CCI`. The two comparisons
are computed separately, and each CI resamples the per-image difference
vector directly.

## Switch-subset comparison

The switch-subset analysis is a different estimand: `WF(metric) -
CCI(metric)` restricted to images with `wf_region != cci_region`. Its
bootstrap unit is the switched image, and its CI is calculated from the
subset difference vector.

## Screening seeds

Each screening rate has its own recorded seed in
`corrected_screening_failure_rates.csv`: screen-pass, failure, joint
failure, and conditional failure. The conditional CI jointly resamples
the joint-failure numerator and screen-pass denominator.

## Runtime wording

The COCO B/16 and B/32 elapsed values are resumed portions only; they are
not complete-experiment elapsed times. See
`corrected_runtime_summary.csv`.
