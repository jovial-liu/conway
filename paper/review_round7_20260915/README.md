# Round-7: ICASSP submission preparation

Start with `submission/START_HERE_zh.md`. Current PDF: `submission/liu.pdf`, identical to `generated/ccf0_round7_20260915.pdf`. Author information is incomplete for submission: all five ORCIDs are recorded; the conflicts statement still requires author confirmation. No submission has been made.

All source files are expanded under `source/`; `arxiv_source/` compiles identically. No ZIP is produced. Run `bash scripts/build.sh`, then `python scripts/prepare_submission.py`. Existing figure PDFs are included. Figure 1 is based on Round-6 with only masking terminology corrected; its PDF/SVG are the authoritative artwork. Figure 2 is reproducible with `scripts/verify_and_plot.py`. Statistical reanalysis: `scripts/robustness.py`.

Read `CHANGES_zh.md`, `VERIFICATION.md`, and `LOCAL_EXPERIMENT_REQUEST.md`. Historical Round-6 and earlier PDFs are superseded for editing. Qihang Wu remains fifth author with user-confirmed Taizhou affiliation.
