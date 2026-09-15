# Round-7 修改

- Figure 1 重绘为可编辑 TikZ 流程图，明确 attention-key columns 的 masking；移除旧 K/V masking 的含糊标签和容易被当成真实响应数据的示意散点。字体 9.2pt。
- Figure 2 的设置标签从 8.5pt 调至 9pt，保留精确数据、共享线性坐标与形状区分。
- 删除两条未能直接核验且非核心论证必需的外围引用（MGA-CLIP、Contrastive Concept Importance），重新编号为 13 条。无法访问不等于文献不存在；删除不作此判断。其余文献核验范围见 VERIFICATION.md。
- 第 5 页加入与已知研究方法一致的伦理说明和实际 AI 辅助说明；没有编造资助、利益冲突或伦理审批信息。
- 提供 liu.pdf、ASCII 摘要、标题、关键词、作者 CSV、JSON、声明待确认说明和哈希；没有创建 ZIP 或代办投稿。
- 沿用 Round-6 阈值与四设置预算分析，不改变实验数值，不声称补齐原始 masks 或逐类响应。

剩余的学术风险：归一化敏感性、原始区域身份、语义 hardest-foil 审核、跨设置校准与空间控制。新增的投稿信息缺口：其余三位作者 ORCID、利益冲突声明。当前文件是可审阅的投稿准备版，不是手续已齐全的提交证明。

## 资助及 ORCID 更新

按用户提供原文加入 Taizhou Fengcheng Talent Plan 下的资助说明；加入 Kaixin Liu 的 ORCID 姓名链接及表单记录。上传 DCI PDF 的全文、259 个链接、PDF 对象和 XML 元数据中未检出 ORCID。仍缺其他三人 ORCID 和利益冲突确认。

## 作者更新

按用户明确要求移除 Xihang Zhou 的署名、邮箱、多伦多大学单位和 PDF 作者元数据；当前共五位作者。剩余 ORCID 缺口为 Zhipeng Ye、Feng Jiang、Qiufeng Wang。

## 恢复原 Figure 1

按用户明确要求恢复 Round-6 的 Figure 1 原始 PDF/SVG；撤下后续 TikZ 重绘稿。当前五位作者、资助说明、ORCID 和 Figure 2 保持不变。

## ORCID 与致谢更新

依据用户上传 DOCX 补齐三位作者 ORCID；五位作者编号齐全。按用户要求删除 ACKNOWLEDGMENTS 整节。原 Figure 1 及五位作者保留。

## 表述优化

原 Figure 1 仅修正 masking 与 class-token 标签，对应归档实验脚本 build_attention_mask；图形结构不变。引言明确审计问题和三项贡献；精简重复数值，明确 A、B 与全部 eligible images 的分母；统一 sign repair 的限定，分层组织局限性。所有表格数据与 Figure 2 均未改变。五位作者 ORCID 和无 ACKNOWLEDGMENTS 版本保留。

## 语言修订

按用户要求改为直接陈述发现与证据，删除重复防御性措辞，将实质边界集中在局限性部分；结论总结贡献与评价意义。图表、实验数值及作者信息不变。

## 引言补强

补充平均非目标响应掩盖单个强竞争类别的直觉解释；具体区分 COCOA 的 corpus/foil 归因、CASE 的类别敏感性与对比显著图，以及本文固定 CCI 区域的控制和枚举审计。文献依据：https://arxiv.org/abs/2210.00107 ，https://arxiv.org/abs/2506.07327 ，https://arxiv.org/abs/2511.12978 。未加入未经验证的首创声明。第 3–5 页与上一版渲染相同。

## 材料表述与排版优化

依据 large_local_artifacts.csv 更正数据可用性：重生成 raw/norm 响应记录为本机保留、未公开；原始 masks 未保存；归一化敏感性尚未完成。表格行距增加，置信区间更易阅读；参考文献页两栏平衡，技术部分仍四页。图表数据、原 Figure 1 样式和五位作者保持。
