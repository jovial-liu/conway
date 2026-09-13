# GitHub 归档补证据版 / 2026-09-12

## 下载内容
- paper_evidence_revised.pdf：五页论文，Figure 1/2/3 保持上一确认版本。
- source/：可编译 LaTeX，运行 `sh build.sh`，不依赖实验室或重新生成图片。
- arxiv_source.zip：仅编译所需文件；选择 main.tex / pdfLaTeX。arxiv_preview.pdf 是本地编译预览。未提交或发布。
- evidence/recovered/：从 GitHub 已存档 CSV 恢复的分析输入。
- analyze_recovered.py：运行 `python analyze_recovered.py` 重新计算本轮增加的统计，需要 Python、NumPy、pandas。
- recovered_statistics.json / four_setting_conditional.csv：未舍入计算结果。
- RECOVERY_AND_QA.md：来源、与当前稿的对齐检查，以及仍不能补齐的项目。

本轮增加 COCO B/16 筛查条件失败率及置信区间、可行集大小分布、切换子集的代价区间，以及四设置的切换条件均值。四设置的条件均值由未舍入的归档总体均值除以精确切换率得到，明确标注为代数推导，不把它们当成新的模型实验或带 CI 的测量。

所有作者信息与图件不变。保留 IDEA 引用与 Results 合并，未恢复 Acknowledgment。主表及已有增益数字不改。为保证五页，将原来的模板稳定性细表从正文移出，正文保留平均/全模板一致性结果，原始稳定性 CSV 仍在 evidence/provenance/。

## 尚未补齐
其它三个设置的筛查条件失败率及 CI；限制 foil 后的基线条件失败率；直接 WF-minus-alternative CI；其它三个设置的切换代价分位数和 CI；独立 K/干预实验。没有重新运行模型，没有宣称完成完整复现。
