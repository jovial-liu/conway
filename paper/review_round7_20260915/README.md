# Round-7: ICASSP submission preparation

Start with `submission/START_HERE_zh.md`. Current PDF: `submission/liu.pdf`, identical to `generated/ccf0_round7_20260915.pdf`. Author information is incomplete for submission: three ORCIDs and the conflicts statement must still be supplied. No submission has been made.

All source files are expanded under `source/`; `arxiv_source/` compiles identically. No ZIP is produced. Run `bash scripts/build.sh`, then `python scripts/prepare_submission.py`. Existing figure PDFs are included. To rebuild Figure 1, compile `source/figures/figure1_method.tex` with PDFLaTeX and copy its output to `figure1_method_clean.pdf`; Figure 2 is reproducible with `scripts/verify_and_plot.py`. Statistical reanalysis: `scripts/robustness.py`.

Read `CHANGES_zh.md`, `VERIFICATION.md`, and `LOCAL_EXPERIMENT_REQUEST.md`. Historical Round-6 and earlier PDFs are superseded for editing. Qihang Wu remains fifth author with user-confirmed Taizhou affiliation.
