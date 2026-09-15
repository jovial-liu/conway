# ccf0 最终交接：GPT 直接读取展开源码

仓库：`jovial-liu/conway`  
工作分支：`paper/verified-rerun-integration`  
论文：**Beyond Mean Foils: Auditing Worst-Foil Specificity in Frozen CLIP Region Explanations**

## 0. 最重要规则：不要用 ZIP

**GPT/插件后续不要读取、依赖或要求任何 ZIP。**

当前源码、图件和证据都应从普通 GitHub 文件/目录直接读取。历史提交中的 ZIP 只是旧包装记录，不是当前工作入口；当前分支已移除主要 ZIP 交付物。

新对话首先读取：

1. `HANDOFF.md`
2. `paper/review_round4_20260914/GPT_SOURCE_INDEX.md`
3. `paper/review_round4_20260914/source/main.tex`
4. `paper/review_round4_20260914/source/figures/`
5. `paper/review_round4_20260914/reference_figures/`

## 1. 当前唯一可编辑源码

主稿：

`paper/review_round4_20260914/source/main.tex`

配套普通文件：

- `paper/review_round4_20260914/source/authors.tex`
- `paper/review_round4_20260914/source/references.tex`
- `paper/review_round4_20260914/source/icassp2027_paperkit.sty`
- `paper/review_round4_20260914/source/tables/`
- `paper/review_round4_20260914/source/figures/`

与之对应的展开 arXiv 树：

`paper/review_round4_20260914/arxiv_source/`

以后修改、编译、交付新版时，直接从这些展开文件开始，不要回退到：

- `paper/local_rerun_revision/`
- round2 / round3 旧稿
- `paper/evidence_base/`
- 用户早期上传的 `liu_submission_compliance_final(1).pdf`

## 2. Figure 1 / Figure 2 / Figure 3

当前 Figure 1：

- `paper/review_round4_20260914/source/figures/figure1_method_clean.pdf`
- `paper/review_round4_20260914/source/figures/figure1_method_clean.svg`

当前 Figure 2：

- `paper/review_round4_20260914/source/figures/figure2_identification_final.pdf`
- `paper/review_round4_20260914/source/figures/figure2_identification_final.svg`

当前 Round-4 Figure 3 **不是外部图片**，而是直接写在：

`paper/review_round4_20260914/source/main.tex`

中的 inline LaTeX response-audit panel。

此前的照片版 Figure 2 / Figure 3 已作为普通 GitHub 文件保留，GPT 可直接检查，不需要 ZIP：

- `paper/review_round4_20260914/reference_figures/figure2_previous_round3.pdf`
- `paper/review_round4_20260914/reference_figures/figure2_previous_round3.svg`
- `paper/review_round4_20260914/reference_figures/figure3_previous_round3.pdf`
- `paper/review_round4_20260914/reference_figures/figure3_previous_round3.svg`
- `paper/review_round4_20260914/reference_figures/figure3_previous_round3_pixels.png`

不要擅自把旧 Figure 3 当成当前 Figure 3；它只是可直接读取的历史参考图件。

## 3. 作者与格式约束

- 保持五页：前四页技术正文，第五页参考文献。
- 六作者顺序固定：Kaixin Liu、Zhipeng Ye、Feng Jiang、Qiufeng Wang、Hao Li、Xihang Zhou。
- 机构、共同一作、通讯作者按当前 `authors.tex`。
- 保留 IDEA 引用 `ye2026idea`。
- 不恢复已删除的 Acknowledgment。
- 不因切换对话退回旧版排版。
- Figure 1 旧版隐藏正文对象的问题已经修复，不要恢复裁剪整页旧稿的做法。
- 最终 PDF 应检查 5 页、字体嵌入/子集化、无 Type 3、无越界和隐藏旧正文。

## 4. 实验与证据版本

### 4.1 local rerun

固定实验提交：

`1817e1666161fa902869affe61b55f7c98c45668`

目录：

`experiments/local_rerun_2026-09-13/rerun_workspace/`

四个 setting：

- `coco_openai_b16`：27,708 records
- `coco_openai_b32`：27,708 records
- `voc2007_openai_b16`：1,943 records
- `voc2007_openai_b32`：1,943 records

共 59,302 个 image-model records；同数据集两个模型共享图像，不能写成 59,302 张独立图像。

### 4.2 restricted-foil controls

最终 reviewer-control 实验提交：

`6a52e8f3d39f343d8483390882dcb5e57003d904`

目录：

`experiments/reviewer_controls_2026-09-13/`

该实验固定原始 CCI-top1 区域和 full-foil 下定义的筛选集合 `A`，比较：

- full foils；
- annotation-absent foils；
- 每图 foil 数量匹配的 exact random subset expectation。

四设置独立核验均 PASS。

## 5. 当前主要结果

### 5.1 固定 A 的 worst-foil failure

`A = (CCI bbox precision >= .5) AND (full-foil normalized aggregate mean-foil margin > 0)`。

| Setting | Full F|A | Annotation-absent F|A | Exact matched-random | Absent - random, pp [95% CI] |
|---|---:|---:|---:|---:|
| COCO B/16 | 64.37% | 63.17% | 64.09% | -0.92 [-1.08,-0.77] |
| COCO B/32 | 64.78% | 63.64% | 64.50% | -0.86 [-1.01,-0.72] |
| VOC B/16 | 42.27% | 40.94% | 41.46% | -0.52 [-1.16,+0.06] |
| VOC B/32 | 41.16% | 39.77% | 40.32% | -0.55 [-1.19,+0.04] |

结论边界：annotation-absent 仅代表“未被标注存在”，不等于人工验证的语义缺席。

### 5.2 正目标贡献检查

`Aplus = A AND (CCI held-out normalized target drop > 0)`。

Full-foil failure：62.64%、63.79%、41.40%、40.74%。  
Annotation-absent failure：61.38%、62.61%、40.05%、39.34%。

因此主失败现象不是主要由负目标贡献样本造成。

### 5.3 精确 repair decomposition，固定原始 B=A AND F

在 `epsilon=.02` 下：

- `C0`：全部 K=8 候选都没有 passing candidate；
- `C1`：存在 passing candidate，但没有任何 passing candidate 满足目标预算；
- `C2`：存在 feasible passing candidate，但 WF 未选中；
- `r`：WF 实际 sign repair。

| Setting | B | C0 | C1 | C2 | r |
|---|---:|---:|---:|---:|---:|
| COCO B/16 | 11,410 | 10,697 | 695 | 0 | 18 |
| COCO B/32 | 11,561 | 10,655 | 875 | 1 | 30 |
| VOC B/16 | 511 | 371 | 135 | 0 | 5 |
| VOC B/32 | 505 | 347 | 155 | 1 | 2 |

正确解释：当前低修复率主要来自候选空间有限和紧目标预算，而不是 WF 漏掉大量已经可行的修复。

### 5.4 sign repair 不等于全部 endpoint 都保持

Round-4 定义 `J`：WF 新区域同时满足：

- worst-foil margin >= 0；
- bbox precision >= .5；
- full mean-foil margin > 0；
- held-out target contribution > 0。

在原始 `B`、`epsilon=.02` 下：

| Setting | WF sign repairs | Repairs also satisfying J |
|---|---:|---:|
| COCO B/16 | 18 | 12 |
| COCO B/32 | 30 | 25 |
| VOC B/16 | 5 | 3 |
| VOC B/32 | 2 | 1 |

不要把 sign repair 写成“全面修复”。

### 5.5 tolerance 与代价

当前候选数据已分析 `.01/.02/.05/.10/.20` 五档 tolerance。放宽预算会增加多候选比例和 oracle 修复机会，同时带来 target/locality 代价。不要声称低修复率对所有 tolerance 都不变。

### 5.6 WF vs Max-.1

两策略仅在 84/89/5/5 个 image-model records 上选择不同区域。全样本增益很小主要因为绝大多数样本选同一区域；WF 的角色仍是诊断性 reranking probe，不是新 SOTA explanation algorithm。

## 6. 统计规则

- 10,000 image-level bootstrap draws。
- NumPy `default_rng`。
- percentile 2.5/97.5。
- paired quantities 先逐图形成差值再重采样。
- intervals 为 pointwise，未做 multiplicity correction。
- matched-cardinality random 主结果用 exact combinatorial expectation；Monte Carlo 只作 sanity check。

不要混淆这些总体：

- A；
- Aplus；
- B=A AND F；
- all baseline failures；
- switched subset；
- actual repaired subset。

## 7. 仍然不能宣称完成的内容

当前证据不支持：

- 对完整 CCI heatmap 的普遍结论；
- 对所有 CLIP explanation 方法的普遍结论；
- annotation absence = 语义缺席；
- 新 K、其他 intervention、其他模型已经验证；
- raw worst-foil / normalizer sensitivity 已完成；
- segmentation / region-size sensitivity 已完成；
- 全量 MPS 与 CPU 严格等价；
- 任意历史 frozen 中间张量均可恢复。

## 8. 分工

- 本地 Mac Codex：只负责必要的实验、统计核验和数据搬运。
- 云端 GPT：论文修改、Figure 3、内部版本记录、外部材料依赖、编译和交付。
- 没有新实验需求时，不要默认重跑 CLIP。
- 不自动提交 arXiv。
- 不 merge main。
- 不 force push。

## 9. 新对话直接使用的提示

> 请先读 `HANDOFF.md` 和 `paper/review_round4_20260914/GPT_SOURCE_INDEX.md`，接手 ccf0。工作分支是 `paper/verified-rerun-integration`。不要读取或要求 ZIP；所有源码和 Figure 2/3 参考图都已经展开为普通 GitHub 文件。当前主稿是 `paper/review_round4_20260914/source/main.tex`。直接继续修改、编译和交付，不要回退到旧稿。
