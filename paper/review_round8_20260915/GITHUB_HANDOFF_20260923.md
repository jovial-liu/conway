# 2026-09-23 论文交接：五作者版

仓库 `jovial-liu/conway`，工作分支 `paper/verified-rerun-integration`。当前唯一有效的论文工作目录是 `paper/review_round8_20260915/`。直接读取普通 GitHub 文件，不需要 ZIP，也不要从 main 或历史稿续改。

## 当前入口

- 可读 PDF：`submission/liu_five_authors_20260923.pdf`；`submission/liu.pdf` 是相同内容的投稿文件名。
- 可编辑 LaTeX：`source/main.tex`、`source/authors.tex`、`source/references.tex`、`source/tables/`、`source/figures/` 和 `source/icassp2027_paperkit.sty`。
- 展开的同版本 arXiv 源码：`arxiv_source/`。
- 投稿表单：`submission/authors.csv`、`submission/submission_metadata.json`、`submission/title.txt`、`submission/abstract.txt`、`submission/keywords.txt`。
- 核验：`generated/SHA256SUMS.txt`、`submission/SHA256SUMS.txt`、`generated/build_checks.json`、`generated/visual_review.json`。

## 当前作者

1. Kaixin Liu — 单位 1，ORCID 0009-0005-5213-8081；共同第一作者。
2. Zhipeng Ye — 单位 1，ORCID 0000-0002-3384-2779；共同第一作者、通讯作者。
3. Feng Jiang — 单位 1，ORCID 0000-0001-5362-3234。
4. Zhenghao Wang — 单位 1，ORCID 0009-0002-8768-4937。
5. Qihang Wu — 单位 1，ORCID 0009-0009-6082-0223。

单位 1 为 Taizhou Institute of Science and Technology, Nanjing University of Science and Technology, Taizhou 225300, Jiangsu, China。Qiufeng Wang 已从当前论文署名、单位、邮箱、ORCID 和投稿表单移除。旧六作者 PDF 别名已从当前分支移除；历史提交未改写。

## 资助与尚待作者确认的声明

资助原文保存在 `submission/submission_metadata.json`：

> This work was supported by the Young Scientific and Technological Talent Support Program under the Taizhou Fengcheng Talent Plan.

按先前要求，论文 PDF 中没有 ACKNOWLEDGMENTS。`conflicts_of_interest` 保持 `null`，`ready_to_submit` 保持 `false`；未替作者推定无利益冲突。投稿类别、伦理说明和最终作者审核也待作者确认。具体见 `submission/DECLARATIONS_TO_CONFIRM.md`。

## 实验与重建

主表和图 2 来源为 `experiments/priority123_20260915/`（证据提交 `7ee2b6080aa6752790e17c910c5c8a3111235d4d`）。表 4 的旧固定人群单独标明 earlier-run。当前更改仅涉及署名与交接，实验数值、原 Figure 1、Figure 2 和表格内容保持不变。

从仓库根目录运行：

```bash
bash paper/review_round8_20260915/scripts/build.sh
python paper/review_round8_20260915/scripts/prepare_submission.py
```

`build.sh` 同步并分别编译 `source/` 与 `arxiv_source/`，`check_build.py` 检查五页、字体、引用、溢出和两份源码的渲染一致性。仅在确实需要重新整合实验时再运行 `scripts/integrate_traceable.py`。没有向会议或 arXiv 实际提交，也没有合并 main。
