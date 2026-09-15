# Verification and provenance

## Manuscript input

- Read root HANDOFF.md and Round-4 GPT_SOURCE_INDEX.md at b731a8b58ab5dbb57fce9f02f4d4cb4da5e70884.
- Used only the expanded Round-4 `source/` as the editable starting point.
- User attachment `paper_final(2).pdf` has SHA256 48a6983583028da3e6aa6357abdb7553577f711ab75054bd47124d283952dee7, exactly matching the obsolete `paper/local_rerun_revision/generated/paper_final.pdf`. It was inspected only as historical context.
- Earlier user attachment `v30.pdf` has SHA256 3a38c8d0243413bee18a1a1107d5c73cd797512563e80612d481d6bdabc8baea and matches the historical Round-4 PDF identity.
- Figure 1 is the clean standalone asset from current Round-4. No full-page crop or hidden old manuscript was introduced.

## Evidence lineage

- Original local rerun: 1817e1666161fa902869affe61b55f7c98c45668, `experiments/local_rerun_2026-09-13/rerun_workspace/`.
- Restricted-foil controls: 6a52e8f3d39f343d8483390882dcb5e57003d904, `experiments/reviewer_controls_2026-09-13/`.
- Figure 2 and case verification use existing per-image candidate summaries in the current repository; input SHA256 values are recorded in `generated/evidence_checks.json`.
- No experimental input, stored response, or existing confidence interval was modified. This pass verified counts and means, not all inherited bootstrap intervals. Restriction-control raw full-class tensors and original spatial masks remain incomplete in public artifacts.

## Formula and implementation check

`build_attention_mask` in the local rerun script sets additive attention-mask columns to negative infinity and passes that mask to the transformer. The manuscript wording was aligned with this implementation. The reported margin remains the normalized aggregate margin; dropping the unused prompt-mean definition did not change any reported values.

## Compilation gates

See `generated/build_checks.json`. Both expanded trees are separately compiled and their five rendered pages compared. All fonts must be embedded/subset, no Type 3 fonts, no overfull boxes or undefined references. Only references appear on page five. Text bounds are checked, then all pages are visually reviewed. The existing custom Paper-Kit-compatible style is retained, with text height reduced to 226mm to keep glyphs clear of the bottom boundary; this does not constitute certification by the conference's document checker.

## Reference and policy scope

The ICASSP 2027 Paper Kit was consulted: https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php . No permission for extra review attachments is assumed. The source includes a convenience hyperlink to public code/diagnostics. Bibliography content was inherited, except removal of the nonessential CVaR entry; this pass is not a full bibliographic verification of every DOI and publication metadata field.
