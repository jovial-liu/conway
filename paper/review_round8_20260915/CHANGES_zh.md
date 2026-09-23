## 2026-09-23 五作者交接

Qiufeng Wang 已从本轮 `source/authors.tex`、PDF 元数据、作者 CSV 与 JSON 删除，同时移除其西交利物浦大学单位、邮箱与 ORCID。当前五位作者及顺序见 `GITHUB_HANDOFF_20260923.md`。旧六作者 PDF 别名从当前 HEAD 清理，历史提交保留。生成脚本只输出当前五作者 PDF 与 `liu.pdf`。表格数值、图 1/图 2 和实验来源不变；资助事实保留在 JSON，利益冲突待作者确认。

## 历史修订记录（下文所述旧 PDF/作者状态已过时）

## 2026-09-17 剩余审阅问题修正

最新 PDF：submission/liu_review_fixes_20260917.pdf。
- A、F、B、A+ 及图像索引说明集中在第一页同一不可拆分段落。
- 以自然语言解释候选集合、oracle 和修复计数，减少结果段中的符号复述。
- 式 (5) 显式加入 R 属于候选集合；表 1 表头和说明统一为 100 Delta_A，单位为百分点，数值不变。
- 表 5 区分均值方向与置信区间，明确两个 B/16 目标变化区间和 VOC B/32 重叠变化区间跨零，以及 VOC 仅 5/2 个修复样本。
- 相关工作按方法与区别描述，删除 Wang and Wang's 人名式句首，保留相关引用。
- 五页编译和视觉检查通过。作者、原图与六张表的数值保持不变。归一化旧批次事实与待作者确认的投稿声明不变；未开展新实验。

## 2026-09-17 公式前置解释版

最新 PDF：submission/liu_formulas_explained_20260917.pdf。八个编号公式均先说明用途、量的含义与运算，再给公式。补齐归一化长度、图像索引、指示函数、组合数、集合基数与修复比例解释，纠正先用 P_i 后定义的顺序。编号公式本身逐字未改，实验数值、作者与图件不变。五页编译通过，无未定义引用或溢出。

## 2026-09-17 叙述重写版

最新 PDF：submission/liu_prose_revised_20260917.pdf。
重写摘要、引言、方法解释、结果分析及结论：从问题出发引入指标，先解释再给符号；用图表支持具体发现，减少连续比例和重复术语。简化联合条件 J 的表达，保留其三个判定条件、固定人群及统计口径。作者、两幅图及所有表格数据未变。全文五页，前四页正文，编译无溢出及未定义引用。

## 2026-09-17 可读性与作者单位修订

当前 PDF：submission/liu_readability_20260917.pdf。
- Zhenghao Wang 与 Qihang Wu 同属南京理工大学泰州科技学院（单位 1）；删除多余单位 3，保留作者次序、ORCID 和邮箱。
- 补充图像、类别、候选区域、选择提示、分数差、归一化、筛选人群、oracle、修复计数、预算及表格变化量定义。
- Results 改为 Results and Analysis；正文解释 Figure 1 的三个环节，结合 Figure 2 和 Tables 1–6 分析结果。
- 删除独立 Interpretation and Limitations 标题、Scope 段落及相关局限性讨论；保留数据版本和统计方法说明。
- 删除 IDEA 正文和 ye2026idea 参考文献，剩余 13 篇引用匹配。
- 表格行距与上下间距收紧；五页、前四页正文，无溢出、未定义引用；source 与 arxiv_source 渲染一致。六张数据表和原有图件未修改。

## 投稿前五处文字修正

删除未使用的 Mean、Max-.1 定义；残余失败比例改引表 1；主结果统一 fixed baseline，历史归一化数据使用 earlier-run；修复分号后 The 的句法；引言收窄为 quantify worst-foil failure and decompose constraints on sign repair。实验数值不变，编译仍为五页。

## 第四页留白调整

调整表 4、表 5 行距与相邻段落间距，两栏底部差约 0.4 pt。正文及数据不变；维持五页，无溢出，其余四页渲染一致。最新文件为 submission/liu_round8_page4_balanced.pdf。

# Round-8 更新

1. 完成论文中的归一化敏感性表（Table 4）：固定旧 A 和旧区域，四设置 raw/normalized 失败率、符号翻转率；hardest-foil 一致率写入正文。
2. 主表与 Figure 2 统一切换到新 traceable 记录；COCO B/32 的 A=17848、B=11562，VOC B/32 的 B=504。原 Table 4 的小幅 baseline 增益不再占用论文表格，直接配对 CI 保留在可读材料中。
3. 原始 masks 未保存的旧说明改为新版本同批保存 masks、响应、选择分数和 bbox 记录；正文固定链接指向实验内容提交。大张量仍在本机，未声称已公开。
4. 保留 NeurIPS 2022 直接相关文献与 foil 定义区别、六位作者 ORCID、原 Figure 1，ACKNOWLEDGMENTS 未恢复。
5. 数值变化贯穿摘要、结果、预算表、条件代价、阈值表和 Figure 2；归一化旧 A 不与新 A 混算。

核验：逐图数据独立重算主计数、可行选择、分解、归一化表；核对实验侧四设置 PASS。原始 masks/张量未下载，空间核验依据实验侧记录。全文五页，字体嵌入，无溢出或未定义引用，源码与 arXiv 展开树渲染一致。

作者更新：新增 Zhenghao Wang（0009-0002-8768-4937，wangzhenghao2002@outlook.com），位于 Qihang Wu 前，单列南京理工大学。正文、实验数据和图表不变。
