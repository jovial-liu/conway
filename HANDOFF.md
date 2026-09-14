# ccf0 最终交接：ICASSP 2027 Round-4 定稿

交接日期：2026-09-14  
仓库：`jovial-liu/conway`  
**工作分支：`paper/verified-rerun-integration`。不要只读 main。**

论文：**Beyond Mean Foils: Auditing Worst-Foil Specificity in Frozen CLIP Region Explanations**

## 1. 当前唯一有效的论文入口

当前定稿为 Round-4 endpoint-aligned finalization。GitHub 物化前的父提交：`2698b71079266a66c280fdd73f89b4efff9fafde`。

以后不要再把以下旧文件当作当前定稿：

- `paper/local_rerun_revision/generated/paper_final.pdf`
- `paper/review_round2_20260913/` 对应的旧对话交付物
- `paper/review_round3_20260913/` 对应的旧对话交付物
- 用户早期上传的 `liu_submission_compliance_final(1).pdf`

当前定稿目录：

| 内容 | 仓库相对路径 |
|---|---|
| 最终 PDF | `paper/review_round4_20260914/generated/ccf0_round4_20260914.pdf` |
| 可编辑主稿 | `paper/review_round4_20260914/source/main.tex` |
| 作者文件 | `paper/review_round4_20260914/source/authors.tex` |
| 当前图件 | `paper/review_round4_20260914/source/figures/` |
| 当前表格片段 | `paper/review_round4_20260914/source/tables/` |
| arXiv 展开源码 | `paper/review_round4_20260914/arxiv_source/` |
| GitHub 构建的源码包 | `paper/review_round4_20260914/generated/ccf0_round4_20260914_source.zip` |
| GitHub 构建的 arXiv 包 | `paper/review_round4_20260914/generated/ccf0_round4_20260914_arxiv.zip` |
| 生成文件校验 | `paper/review_round4_20260914/generated/SHA256SUMS.txt` |
| PDF 字体检查 | `paper/review_round4_20260914/generated/pdffonts.txt` |
| PDF 页面信息 | `paper/review_round4_20260914/generated/pdfinfo.txt` |
| Round-4 版本说明 | `paper/review_round4_20260914/README.md` |
| Round-4 修改记录 | `paper/review_round4_20260914/CHANGELOG.md` |
| Round-4 核验说明 | `paper/review_round4_20260914/VERIFICATION_NOTES.md` |
| endpoint 补充说明 | `paper/review_round4_20260914/ENDPOINT_SUPPLEMENT.md` |
| GitHub handoff 包 | `handoff/round4/ccf0_round4_github_handoff.zip` |
| 最终总交接包 | `handoff/ccf0_final_handoff.zip` |

**交付时以 `generated/SHA256SUMS.txt` 中的 GitHub 构建哈希为准。** PDF 的元数据可能导致不同环境构建的字节哈希与先前 ChatGPT 会话附件不同；内容和格式判断必须基于当前 GitHub 定稿目录与当前构建结果，不要拿旧附件哈希反向覆盖新构建。

## 2. 论文格式与作者约束

- 保持五页：前四页技术正文，第五页参考文献。
- 六作者顺序固定：Kaixin Liu、Zhipeng Ye、Feng Jiang、Qiufeng Wang、Hao Li、Xihang Zhou。
- 机构、共同一作、通讯作者以当前 `authors.tex` 为准。
- 保留 IDEA 引用（`ye2026idea`）。
- 不恢复已删除的 Acknowledgment。
- 当前提交为单匿名作者信息保留版。
- 最终 PDF 要求所有字体 `emb=yes` 且 `sub=yes`，不得出现 Type 3 字体。
- Figure 1 旧版隐藏正文对象的问题已经修复；不要恢复裁剪整页旧稿的图件。
- 不因切换对话重新回退到 evidence_base 或旧 round2/round3 排版。

## 3. 科学证据版本链

### 3.1 原始 local rerun

不可变基线提交：

`1817e1666161fa902869affe61b55f7c98c45668`

主要目录：

`experiments/local_rerun_2026-09-13/rerun_workspace/`

四个 setting：

- `coco_openai_b16`：27,708 records
- `coco_openai_b32`：27,708 records
- `voc2007_openai_b16`：1,943 records
- `voc2007_openai_b32`：1,943 records

共 59,302 个 image-model records；同数据集两模型使用同一批图像，**不能称为 59,302 张独立图像**。

### 3.2 restricted-foil reviewer controls

最终实验提交：

`6a52e8f3d39f343d8483390882dcb5e57003d904`

目录：

`experiments/reviewer_controls_2026-09-13/`

该实验固定原始 CCI-top1 区域和 full-foil 下定义的筛选集合 `A`，不重新选区域、不重新定义 `A`，比较：

1. full foils；
2. annotation-absent foils；
3. 每图 foil 数量匹配的 exact random subset expectation。

四设置独立验证均 PASS。完整逐类别响应大张量因体积限制留在本地，SHA256 记录在：

`experiments/reviewer_controls_2026-09-13/manifest/large_local_artifacts.csv`

公开仓库包含可复查的统计、逐图诊断、脚本、metadata、日志和独立核验结果；不要把公开诊断文件描述成“包含全部逐类别原始张量”。

## 4. 当前论文的主要实证结论

### 4.1 固定原始筛选集合下的 worst-foil failure

`A = (CCI bbox precision >= .5) AND (full-foil normalized aggregate mean-foil margin > 0)`。

| Setting | Full F|A | Annotation-absent F|A | Exact matched-random | Absent - random, pp [95% CI] |
|---|---:|---:|---:|---:|
| COCO B/16 | 64.37% | 63.17% | 64.09% | -0.92 [-1.08,-0.77] |
| COCO B/32 | 64.78% | 63.64% | 64.50% | -0.86 [-1.01,-0.72] |
| VOC B/16 | 42.27% | 40.94% | 41.46% | -0.52 [-1.16,+0.06] |
| VOC B/32 | 41.16% | 39.77% | 40.32% | -0.55 [-1.19,+0.04] |

解释边界：

- 高失败率在排除“已标注存在”的非目标类别后仍基本保留，因此已标注共现类别不能单独解释主现象。
- COCO 上定向排除相对 matched-random 的差值区间低于零；VOC 对应区间包含零。
- `annotation-absent` 仅指未被数据集标注为存在，**不等于人工验证的语义缺席**。
- 不能把这一结果写成“共现不重要”或“所有语义共现都被排除”。

### 4.2 正目标贡献检查

`Aplus = A AND (CCI held-out normalized target drop > 0)`。

Full-foil failure rates：

- COCO B/16 62.64%
- COCO B/32 63.79%
- VOC B/16 41.40%
- VOC B/32 40.74%

Annotation-absent failure rates：

- COCO B/16 61.38%
- COCO B/32 62.61%
- VOC B/16 40.05%
- VOC B/32 39.34%

因此“mean-foil screen 本身没有保证正目标贡献”已经通过本地 `Aplus` 稳健性检查处理；不要再用 frozen archive 的 90.40% 代替这套本地结果。

### 4.3 精确 repair oracle：固定原始 `B=A AND F`

在 epsilon=.02 下，互斥分解为：

- `C0`：全部 K=8 候选中没有任何 passing candidate；
- `C1`：至少存在 passing candidate，但 **没有任何 passing candidate 满足目标预算**；
- `C2`：存在 feasible passing candidate，但 WF 未选中；
- `r`：WF 实际 sign repair。

| Setting | B | C0 | C1 | C2 | r |
|---|---:|---:|---:|---:|---:|
| COCO B/16 | 11,410 | 10,697 | 695 | 0 | 18 |
| COCO B/32 | 11,561 | 10,655 | 875 | 1 | 30 |
| VOC B/16 | 511 | 371 | 135 | 0 | 5 |
| VOC B/32 | 505 | 347 | 155 | 1 | 2 |

由此可见：

- COCO 中约 92%–94% 的 `B` 在当前八个候选中完全没有 passing candidate；VOC 为约 69%–73%。
- 即便全部候选中存在 passing candidate，epsilon=.02 的目标预算又排除了绝大多数机会。
- 紧预算下 WF 漏掉的 feasible sign repair 很少：四设置 C2 为 0/1/0/1。
- 这只针对当前 K=8 候选，不等于图像中不存在其他更合理区域。

### 4.4 sign repair 与 endpoint retention 不等价

Round-4 新增 `J`：WF 新区域同时满足：

- worst-foil margin >= 0；
- bbox precision >= .5；
- full mean-foil margin > 0；
- held-out target contribution > 0。

在原始 `B`、epsilon=.02 下：

| Setting | WF sign repairs | Repairs also satisfying J |
|---|---:|---:|
| COCO B/16 | 18 | 12 |
| COCO B/32 | 30 | 25 |
| VOC B/16 | 5 | 3 |
| VOC B/32 | 2 | 1 |

所以不要把“margin 转正”写成“解释全面修复”。对所有 baseline failures 的 sign repair 计数仍是 42/48/11/6；这是不同分母，不要与 `B` 内 18/30/5/2 混用。

### 4.5 tolerance frontier 与代价

当前论文已经使用同一 stored-candidate 数据对 `.01/.02/.05/.10/.20` 五档 tolerance 进行重分析。

COCO B/16 的代表性结果：

- epsilon=.02：多候选比例约 4.71%，`B` 内 exact feasible oracle repair 约 0.16%。
- epsilon=.20：多候选比例约 35.78%，`B` 内 oracle repair 约 1.57%。
- epsilon=.20 时全样本 held-out raw target response 约 -0.01397，bbox precision 约 -1.63 percentage points，报告区间均低于零。

因此正确表述是：放宽目标预算增加修复机会，但会产生可测目标/locality 代价；不要声称低修复率在所有 tolerance 下不变。

### 4.6 WF vs Max-.1

WF 与 Max-.1 仅在 84/89/5/5 个 image-model records 上选择不同区域。全样本均值差很小，主要因为绝大多数样本选择相同；分歧子集上的差异更大且可能伴随 target/locality 代价。

WF 在论文中的角色仍是**诊断性受约束 reranking probe**，不是新 SOTA explanation algorithm。

## 5. 统计与复现约束

- 主要 interval：10,000 image-level bootstrap draws。
- NumPy `default_rng`。
- percentile 2.5/97.5。
- paired quantities 先在同一 image 上形成差异再重采样。
- 条件比例按对应固定总体联合重采样。
- intervals 为 pointwise，未做 multiplicity correction。
- matched-cardinality random 主结果使用 exact combinatorial expectation，而不是只依赖 Monte Carlo。
- Monte Carlo 仅作为 sanity check。

不要把：

- full-foil A；
- Aplus；
- original B=A AND F；
- all baseline failures；
- switched subset；
- actual repaired subset

混为同一个统计总体。

## 6. 当前证据可以和不可以支持什么

可以支持：

- CCI-top1 的 ontology-relative worst-foil audit；
- full / annotation-absent / exact matched-cardinality fixed-population comparison；
- positive-target robustness；
- current K=8 candidate set 下的 exact sign-repair capacity decomposition；
- epsilon tolerance 下机会与代价的变化；
- WF/Mean/Max-.1 的同可行集比较；
- sign repair 与 target/locality/screen-retention 是不同 endpoint。

不能支持：

- 对完整 CCI heatmap 的普遍结论；
- 对所有 CLIP explanation 方法的普遍结论；
- annotation absence = 人工语义缺席；
- 新 K、其他 intervention、其他模型的稳健性；
- raw worst-foil / normalizer sensitivity 已完成；
- segmentation/region-size sensitivity 已完成；
- MPS 与 CPU 在全数据上严格等价；
- 任意历史 frozen 中间张量均可恢复。

## 7. Frozen 与 local rerun 仍须严格分开

旧 frozen archive 和当前 local rerun 是不同证据集。此前已记录的差异包括：

- frozen/local COCO B/16 screen/failure 状态有少量图像不同；
- COCO B/32 frozen switches=613，local=614；
- frozen candidate arrays / 中间量并不完整。

不能把这些差异归因于“已经证实的 MPS 误差”。当前最终论文优先使用材料更完整的 local rerun 和 reviewer controls；历史 archive 仅在明确标注来源的位置作为补充证据。

## 8. 分工与操作边界

用户确认的分工：

- 本地 MacBook M4 Pro Codex：**只负责实验、统计核验、数据搬运**。
- 云端助手：负责论文修改、图表排版、编译、最终打包和交接。
- 没有新任务时不要默认让 Mac Codex 修改论文。

禁止默认执行：

- 不自动提交 arXiv；
- 不自动 merge main；
- 不 force push；
- 不覆盖 main；
- 不为了匹配旧数字而改实验结果；
- 不把 frozen 与 local 混算；
- 不默认追加全量模型推理。

## 9. GitHub 目录说明

现有完整实验已经在本分支历史与当前树中，不需要在 `handoff/` 再复制一份几十/几百 MB 数据：

- 原始 local rerun：`experiments/local_rerun_2026-09-13/`
- 统计 verification：`experiments/verification_2026-09-13/`
- restricted-foil controls：`experiments/reviewer_controls_2026-09-13/`
- 当前论文：`paper/review_round4_20260914/`

`handoff/round4/ccf0_round4_github_handoff.zip` 和 `handoff/ccf0_final_handoff.zip` 是便携交付包；**完整实验数据仍以仓库中的 experiment directories 为权威来源**。

## 10. 新对话启动指令

切换新对话后直接发：

> 请先完整阅读仓库根目录 `HANDOFF.md`，接手 ccf0 项目。工作分支是 `paper/verified-rerun-integration`，不要只读 main。当前唯一有效定稿目录是 `paper/review_round4_20260914/`，不要使用旧 `paper/local_rerun_revision/generated/paper_final.pdf`、round2 或 round3 旧稿。本地 Codex 只负责实验，论文由你处理。先核对 `generated/SHA256SUMS.txt`、当前 PDF/source/arXiv 包和 `handoff/ccf0_final_handoff.zip`，再继续。

如果没有新的实验或修改要求，**不要重新做已经完成的实验或从旧稿重新开始。**
