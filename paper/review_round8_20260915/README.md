# Current paper: five-author revision, 2026-09-23

Read `GITHUB_HANDOFF_20260923.md` for the current state and submission caveats. The editable source is `source/`; the expanded arXiv mirror is `arxiv_source/`. Current PDFs are `submission/liu_five_authors_20260923.pdf` and its identical submission alias `submission/liu.pdf`.

The paper has five pages: four technical pages and a fifth page of references/ethical-compliance text. All five authors have affiliation 1, Taizhou Institute of Science and Technology, Nanjing University of Science and Technology. The main result tables and Figure 2 derive from the same-generation experiment records at `experiments/priority123_20260915/`. Table 4 alone is explicitly an earlier-run fixed-population normalization check. Figure 1 is the original selection/held-out schematic.

Run from repository root:

```bash
bash paper/review_round8_20260915/scripts/build.sh
python paper/review_round8_20260915/scripts/prepare_submission.py
```

`generated/build_checks.json` checks the page limit, font embedding, references, overfull boxes, and source/arXiv render identity. `generated/SHA256SUMS.txt` and `submission/SHA256SUMS.txt` cover current build and submission files. Funding is recorded in `submission/submission_metadata.json`; COI and final author approvals remain open in `submission/DECLARATIONS_TO_CONFIRM.md`. No ZIP or actual conference/arXiv submission was made.
