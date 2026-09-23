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

## 资助与利益冲突声明（已按用户提供的老师原文加入）

This work was supported by the Young Scientific and Technological Talent Support Program under the Taizhou Fengcheng Talent Plan.

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

两段均写入 PDF 第 5 页 Acknowledgments，利益冲突仅保留一次，并同步到投稿元数据。现有伦理声明保留；投稿类别和最终作者审核未在本次操作中确认，ready_to_submit 仍为 false。

## 实验与重建

主表和图 2 来源为 `experiments/priority123_20260915/`（证据提交 `7ee2b6080aa6752790e17c910c5c8a3111235d4d`）。表 4 的旧固定人群单独标明 earlier-run。当前更改涉及署名、声明及实验说明澄清；实验数值、原 Figure 1、Figure 2 和表格内容保持不变。

从仓库根目录运行：

```bash
bash paper/review_round8_20260915/scripts/build.sh
python paper/review_round8_20260915/scripts/prepare_submission.py
```

`build.sh` 同步并分别编译 `source/` 与 `arxiv_source/`，`check_build.py` 检查五页、字体、引用、溢出和两份源码的渲染一致性。仅在确实需要重新整合实验时再运行 `scripts/integrate_traceable.py`。没有向会议或 arXiv 实际提交，也没有合并 main。

## 本次声明更新的编译与实验历史核对

本次使用 Tectonic 0.16.9（XeTeX）编译，两份源码渲染一致；原 pdflatex 的 PDF 版本设置增加条件保护。为保持四页技术正文，微调表 5 行距及表 4/5 附近留白，字号、页边距、表格数据和图件不变。

可使用 `tectonic --keep-logs --outdir generated/build/source source/main.tex` 编译（从本论文目录运行；先创建输出目录），arxiv_source 同理。完整检查见 generated/build_checks.json 与 generated/visual_review.json。

旧运行与新完整重跑使用相同种子规则；没有发现多组 base seed 完整重复实验的证据。详见 EXPERIMENT_HISTORY_REVIEW_20260923.md，内含给老师的准确回复。

## 投稿前表述修订

- 用明确事件 $E_i^{abs}$ 定义 annotation-absent 失败判定，再用于指示函数；原稿普通 F 已有失败判定说明，本次消除其与花体 foil 集合的视觉歧义，并非更改统计计算。
- 主结果明确来自沿用原种子规则的一次完整重跑，不声称多组独立 base seed 重复实验；bootstrap 区间不反映 base-seed 变异。
- 表 4 固定早期运行区域和图像，以隔离归一化影响，并明确不与重跑结果混算。
- 重新编译及逐页视觉检查通过，仍为五页。没有运行新实验或更改表格数据。
