# ccf0 优先项 1–3 实验报告

## 结论先行

四个设置均已在本机 MacBook M4 Pro 上用 PyTorch MPS 完成真实全量生成，输出来自同一次生成的 traceable batch 记录。新结果统一标记为 `local_rerun_traceable`，不覆盖 frozen 或旧 `local_rerun`，也没有修改论文、LaTeX、PDF 或图稿。

四个设置的结果文件、逐图数据和核验记录均已生成；完整原始张量约 1.4 GB，仅保留在本机并在哈希清单中记录，GitHub 上传内容为展开的可读文件、脚本、日志、配置和 SHA256，不上传 ZIP、原始数据集或模型权重。

## 1. 四设置重跑与主计数

| 设置 | 图像 | MPS 耗时（本次） | 速度 img/s | A | A+ | B | C0 | C1 | C2 | r | j | CCI→WF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO B/16 | 27,708 | 4,573.71 s | 6.058 | 17,726 | 16,855 | 11,410 | 10,697 | 695 | 0 | 18 | 12 | 663 |
| COCO B/32 | 27,708 | 2,139.51 s | 12.951 | 17,848 | 17,331 | 11,562 | 10,655 | 876 | 1 | 30 | 25 | 613 |
| VOC2007 B/16 | 1,943 | 244.85 s | 7.936 | 1,209 | 1,186 | 511 | 371 | 135 | 0 | 5 | 3 | 50 |
| VOC2007 B/32 | 1,943 | 95.11 s | 20.430 | 1,227 | 1,215 | 504 | 347 | 154 | 1 | 2 | 1 | 38 |

总实测生成时间为 7,053.17 s（约 1 h 57 min 33 s）。四行均满足 `B=C0+C1+C2+r`。完整计数在 `traceable_results/completed_and_missing.csv`，耗时在 `traceable_results/runtime_summary.csv`。

配置沿用归档协议：OpenAI CLIP ViT-B/16、B/32，原始 OpenAI 预处理，K=8，KMeans `n_init=3`、`max_iter=50`，selection/held-out 模板、foil 集合、attention 干预和 `epsilon=.02` 均固定。策略公式以 `config/traceable_capture_config.json` 为准：`WF−Max-.1` 是 WF 与 Max-.1 两个策略的逐图直接差值，不是单独策略名。

## 2. 归一化敏感性（固定原 normalized A）

| 设置 | A | raw mean 失败率 | normalized mean 失败率 | 符号翻转 | raw→fail | fail→pass | hardest foil 一致率 | 固定 B 上 raw / normalized oracle capacity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO B/16 | 17,726 | 64.318% | 64.369% | 0.107% | 0.079% | 0.028% | 99.520% | 6.380% / 6.249% |
| COCO B/32 | 17,847 | 64.762% | 64.778% | 0.073% | 0.045% | 0.028% | 99.468% | 7.897% / 7.837% |
| VOC2007 B/16 | 1,209 | 42.184% | 42.266% | 0.083% | 0.083% | 0 | 99.421% | 27.397% / 27.397% |
| VOC2007 B/32 | 1,227 | 41.157% | 41.157% | 0 | 0 | 0 | 99.837% | 31.089% / 31.287% |

这里的 COCO B/32 固定 A 为 17,847、固定 B 为 11,561，不能与同批新 traceable A/B（17,848/11,562）混算。完整逐图审计、输入 SHA 和来源在 `normalization/`。

## 3. 直接配对比较

`traceable_results/paired_difference_ci_10000.csv` 是唯一应替换主表的配对结果：每个图像先计算 WF−Mean 或 WF−Max-.1，再做 10,000 次图像级 percentile bootstrap（2.5/97.5），没有用两个独立 CI 相减。`target_drop_raw`、`target_drop_norm`、`margin_norm` 和 `bbox_precision` 分列保存。

关键的 normalized aggregate worst-foil margin 差值如下（估计值 [95% CI]）：

| 设置 | WF−Mean | WF−Max-.1 |
|---|---:|---:|
| COCO B/16 | +0.012461 [0.010889, 0.014159] | +0.000245 [0.000127, 0.000370] |
| COCO B/32 | +0.009343 [0.007974, 0.010818] | +0.000242 [0.000083, 0.000417] |
| VOC2007 B/16 | +0.004480 [0.002330, 0.007013] | +0.000507 [0.000046, 0.001120] |
| VOC2007 B/32 | +0.003184 [0.001462, 0.005129] | +0.000418 [0.000058, 0.000912] |

该 CSV 同时给出 raw/normalized target drop、bbox precision 的完整估计值、区间、种子和有效样本数；`traceable_results/main_table_comparison.csv` 给出与旧统计表逐项的新旧差值。

## 4. 筛查、切换和可行集统计

四设置的 CCI 筛查通过数为 17,726 / 17,848 / 1,209 / 1,227；联合失败数为 11,410 / 11,562 / 511 / 504。完整失败率、联合失败率、条件失败率及每项实际 bootstrap seed 在 `traceable_results/screening_failure_rates.csv`；条件率是联合重采样分子和分母后再取比值。

切换子集的 `margin improvement` 与 `sign repair` 分开统计：

| 设置 | 切换数 | margin improvement | sign repair | j |
|---|---:|---:|---:|---:|
| COCO B/16 | 663 | 611（92.157%） | 42（6.335%） | 12 |
| COCO B/32 | 613 | 532（86.786%） | 48（7.830%） | 25 |
| VOC2007 B/16 | 50 | 46（92.000%） | 11（22.000%） | 3 |
| VOC2007 B/32 | 38 | 35（92.105%） | 6（15.789%） | 1 |

切换子集的 target、bbox、margin 均值、10,000 次 CI 和 P10/P25/P50/P75/P90 在 `traceable_results/switch_subset_changes_ci.csv`；负转非负的定义是 CCI margin < 0 且 WF margin ≥ 0，不能与单纯 margin 变大混同。完整 epsilon-feasible set 分布在逐图 CSV 的 `feasible_set_size` 及 `traceable_results/matched_cardinality_control.csv`。

## 5. 同批可追溯记录与验证

四个设置的 `verification/<setting>/same_record_checks.json` 均为 `PASS`。核验覆盖批次完整性、patch 分区、mask、bbox 重算、类别和图像顺序、selection/held-out 全类别响应、raw/normalized 聚合、归一化系数、可行集、四策略选择和 C0/C1/C2/r 分解。100 个 seeded matched-cardinality 抽样也已运行；结果在 `matched_cardinality_mc.csv`，使用 `SeedSequence([seed, sample_index])` 并与 exact q 对照。

小样本 CPU/MPS 对照和四设置全量日志在 `manifest/logs/`。full 运行最终均在 MPS 完成；日志里的 `elapsed_seconds_this_run` 是本次续跑/采集段耗时，不应解释为旧实验的完整历史耗时。

## 6. 新旧差异、权威版本与限制

- COCO B/16、VOC2007 B/16：没有状态变化；对齐逐图的数值差异约在 `7e-6` 以内。
- COCO B/32：两个状态变化。sample 20045/image 203257 的 CCI 区域 `6→1`，由非 A 进入 A/B/C1；sample 24978/image 390130 的四个策略区域 `5→7`，仍不进入 A/B。相对旧 local_rerun，A/A+/B/C1 各增加 1；旧 local_rerun 的切换数为 614，frozen 对照清单为 613，本次新记录为 613。
- VOC2007 B/32：sample 1118/image 2189（diningtable）的四个策略区域 `3→7`；旧 B/C1 变为新非 B，因此 B 从 505 变为 504。
- 旧 masks 和完整旧候选中间量没有保留，故不能仅凭差异把原因归为 MPS 浮点误差。新生成的同批 traceable 记录是本轮实验的权威版本；旧 frozen/local_rerun 只作对照，未被覆盖。
- `source/main.tex` 和 references 的 Wang & Wang, NeurIPS 2022、IDEA 引用已只读核验；没有恢复旧论文，也没有改动三框版 Figure 1。本报告不替论文编辑端做任何文本替换。

仍存在的限制是：大张量未上传 GitHub，需按 `traceable_results/input_hashes.csv` 指向本机路径核验；旧 masks 无法恢复，因此新旧候选区域不能声明为逐 mask 相同。除此之外，四设置的真实全量生成、统计和同批验证均已完成，没有待运行的实验项目。

## 7. 交付索引

- 逐图与候选摘要：`traceable_results/per_image/`
- 配对 CI：`traceable_results/paired_difference_ci_10000.csv`
- 筛查 CI：`traceable_results/screening_failure_rates.csv`
- 切换子集 CI：`traceable_results/switch_subset_changes_ci.csv`
- 新旧差异：`traceable_results/state_changes/`、`traceable_results/main_table_comparison.csv`
- 归一化敏感性：`normalization/`
- 脚本：`scripts/`
- 配置、依赖、输入来源和哈希：`config/`、`manifest/`
- 全量运行记录：`runs/`、`verification/`

上传提交 SHA 在 GitHub 推送完成后写入本报告的末尾；当前工作分支为 `paper/verified-rerun-integration`。
