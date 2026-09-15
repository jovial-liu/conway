# ccf0：当前 Round-7 展开源码交接

仓库 `jovial-liu/conway`；工作分支 `paper/verified-rerun-integration`。

当前唯一可编辑论文目录：`paper/review_round7_20260915/`。直接读取普通文件，不读取或要求 ZIP。

- 论文：`generated/ccf0_round7_20260915.pdf`
- 源码：`source/main.tex`、`source/authors.tex`、`source/references.tex`、`source/tables/`、`source/figures/`
- 展开 arXiv 树：`arxiv_source/`
- 入口：`GPT_SOURCE_INDEX.md`
- 修改：`CHANGES_zh.md`
- 核验：`VERIFICATION.md`、`generated/SHA256SUMS.txt`、`generated/build_checks.json`、`generated/visual_review.json`
- 新分析：`scripts/robustness.py`、`generated/robustness_checks.json`
- 剩余实验：`LOCAL_EXPERIMENT_REQUEST.md`

以上路径除仓库和分支外均相对当前论文目录。

## 本轮状态

- Qihang Wu 为第五作者。用户已确认其单位为南京理工大学泰州科技学院；邮箱 24107880128@nustti.edu.cn，ORCID 0009-0009-6082-0223。其他作者及顺序不变。
- Table 3 已改为四设置 .02/.20 预算端点。五档预算的完整再分析保存在 JSON。
- Table 6 已改为固定 A 的阈值敏感性，不再是 image 254807 的单案例面板。
- Figure 2 为双端点容量图，沿用已核验数值。不存在 Figure 3。
- failure 仅指未通过指定的 worst-foil 判据；不能写成解释必然语义错误。
- 预算放宽时 bbox 全图均值在 COCO 降、VOC 升，不能写成普遍定位损失。
- COCO/VOC 的 foil 数、目标选择和标注不同；不能直接推断数据集本质难度。
- 汇总值一致不证明空间 masks 身份一致。原始 masks 缺失，区域固定控制仍依赖忠实重建；完整逐类响应也缺失，归一化敏感性未完成。
- 参考文献仅部分核实，未验证条目详见 VERIFICATION.md。不可宣称全部文献已核实。

## 证据与权限边界

原始 local rerun：1817e1666161fa902869affe61b55f7c98c45668。
Restricted-foil controls：6a52e8f3d39f343d8483390882dcb5e57003d904。
权威实验目录继续为 experiments/local_rerun_2026-09-13/、experiments/verification_2026-09-13/、experiments/reviewer_controls_2026-09-13/。

本轮仅再分析现有候选 CSV，没有运行 CLIP、补造 masks 或新增置信区间。Table 1/2/4/5 的数值和已有置信区间保留；不得把 59,302 image-model records 写成独立图像数。

本地 Codex 负责必要实验；云端 GPT 负责论文和现有证据再分析。不要修改历史实验，不提交 arXiv，不 merge main，不 force push。

Round-4、Round-5 和旧 paper/local_rerun_revision/generated/paper_final.pdf 均是历史版本。不要根据同名 PDF 或旧图恢复当前稿。

## 投稿准备状态

先读当前目录 submission/START_HERE_zh.md 和 FILE_INDEX.md。submission/liu.pdf 为投稿文件命名副本，但利益冲突声明尚待作者确认，另有三位作者 ORCID 缺失；ready_to_submit=false。原实验和统计结果不变。Figure 1 已按用户要求恢复为 Round-6 原图，Figure 2 标签至少 9pt。第 5 页包含伦理和 AI 辅助说明。已移除 MGA-CLIP 与 Contrastive Concept Importance 两条未能直接核验的外围引用。当前共 13 条引用，不宣称全部出版元数据均已独立核实。

用户已补充 Kaixin Liu 的 ORCID 0009-0005-5213-8081，已加入姓名链接和投稿表；已按原文加入 Young Scientific and Technological Talent Support Program under the Taizhou Fengcheng Talent Plan 资助。仍缺 Zhipeng Ye/Feng Jiang/Qiufeng Wang 的 ORCID 和利益冲突确认。

最新作者变更：按用户要求删除 Xihang Zhou，当前作者共五人；源码、PDF 元数据及投稿表已同步。

Figure 1 最新状态：使用重画前的原 PDF/SVG，已移除替代的 TikZ 源文件，勿再次自动替换。
