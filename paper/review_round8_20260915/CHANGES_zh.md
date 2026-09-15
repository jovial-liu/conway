# Round-8 更新

1. 完成论文中的归一化敏感性表（Table 4）：固定旧 A 和旧区域，四设置 raw/normalized 失败率、符号翻转率；hardest-foil 一致率写入正文。
2. 主表与 Figure 2 统一切换到新 traceable 记录；COCO B/32 的 A=17848、B=11562，VOC B/32 的 B=504。原 Table 4 的小幅 baseline 增益不再占用论文表格，直接配对 CI 保留在可读材料中。
3. 原始 masks 未保存的旧说明改为新版本同批保存 masks、响应、选择分数和 bbox 记录；正文固定链接指向实验内容提交。大张量仍在本机，未声称已公开。
4. 保留 NeurIPS 2022 直接相关文献与 foil 定义区别、五位作者 ORCID、原 Figure 1，ACKNOWLEDGMENTS 未恢复。
5. 数值变化贯穿摘要、结果、预算表、条件代价、阈值表和 Figure 2；归一化旧 A 不与新 A 混算。

核验：逐图数据独立重算主计数、可行选择、分解、归一化表；核对实验侧四设置 PASS。原始 masks/张量未下载，空间核验依据实验侧记录。全文五页，字体嵌入，无溢出或未定义引用，源码与 arXiv 展开树渲染一致。
