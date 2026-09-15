# ccf0 优先项 1–3：本机真实重跑与核验

本目录只包含实验、统计和验证材料；本轮没有修改论文源码、LaTeX、PDF 或图稿。新结果标记为 `local_rerun_traceable`，与旧 frozen 和旧 local_rerun 分开保存。

## 完成状态

四个设置均已在 MacBook M4 Pro 上用 PyTorch MPS 实际运行，并保存了同一次生成的逐批记录：

| 设置 | 图像数 | 采集耗时（本次） | 实测速度 | A | A+ | B | C0 | C1 | C2 | r | j | CCI→WF 切换 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO B/16 | 27,708 | 4,573.71 s | 6.058 img/s | 17,726 | 16,855 | 11,410 | 10,697 | 695 | 0 | 18 | 12 | 663 |
| COCO B/32 | 27,708 | 2,139.51 s | 12.951 img/s | 17,848 | 17,331 | 11,562 | 10,655 | 876 | 1 | 30 | 25 | 613 |
| VOC2007 B/16 | 1,943 | 244.85 s | 7.936 img/s | 1,209 | 1,186 | 511 | 371 | 135 | 0 | 5 | 3 | 50 |
| VOC2007 B/32 | 1,943 | 95.11 s | 20.430 img/s | 1,227 | 1,215 | 504 | 347 | 154 | 1 | 2 | 1 | 38 |

四行都满足 `B = C0 + C1 + C2 + r`。`j` 是 r 中同时满足 WF 区域 bbox、正 normalized foil-mean margin 和正 normalized target drop 的数量。

## 归一化敏感性（固定原 normalized A）

结果在 `normalization/normalization_fixed_A.csv`。它固定原始 normalized A、图像集合和 CCI 区域，只比较 raw mean 与 normalized mean，不用 raw 结果重新筛选：

| 设置 | A | raw 失败率 | normalized 失败率 | 符号翻转 | raw→fail | fail→pass | hardest-foil 一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| COCO B/16 | 17,726 | 64.318% | 64.369% | 0.107% | 0.079% | 0.028% | 99.520% |
| COCO B/32 | 17,847 | 64.762% | 64.778% | 0.073% | 0.045% | 0.028% | 99.468% |
| VOC2007 B/16 | 1,209 | 42.184% | 42.266% | 0.083% | 0.083% | 0 | 99.421% |
| VOC2007 B/32 | 1,227 | 41.157% | 41.157% | 0 | 0 | 0 | 99.837% |

该分析还记录了固定 normalized B 上的 unrestricted oracle capacity。COCO B/16 的旧 B 为 11,410，COCO B/32 为 11,561；归一化敏感性分析使用固定原 A，不能把这张表与新生成的 A/B 混算。

## 统计定义

- CCI-top1、WF、Mean、Max-.1 使用同一图像、同一 K=8 候选集和同一 `epsilon=.02` 可行集。
- `WF−Max-.1` 是两个策略的直接配对比较；Max-.1 的实际公式是 `selection target drop - .1 * selection max-foil drop`，不是一个叫作 “WF−Max-.1” 的策略。
- 配对结果在 `traceable_results/paired_difference_ci_10000.csv`，逐图结果在 `traceable_results/per_image/`；`target_drop_raw` 与 `target_drop_norm` 分开列出。
- 所有 CI 使用 10,000 次图像级 percentile 2.5/97.5 bootstrap；直接比较先在每张图内相减，再重采样。screening 的条件失败率联合重采样分子和分母，切换子集 CI 只使用切换图像向量。
- `traceable_results/matched_cardinality_control.csv` 给出 exact matched-random 的逐图组合概率；`matched_cardinality_sanity_check.csv` 是 100 个 `SeedSequence([seed, sample_index])` 随机抽样核验。

## 新旧差异

新旧逐图按 `sample_index` 和 `image_id` 对齐，详情在 `traceable_results/state_changes/` 和 `traceable_results/new_vs_old_summary.csv`。

- COCO B/16 与 VOC2007 B/16 没有状态变化；数值差异均在约 `7e-6` 以内。
- COCO B/32 有两个状态变化：sample 20045 / image 203257 的 CCI 区域 `6→1`，由非 A 变为 A/B/C1；sample 24978 / image 390130 的四个策略区域 `5→7`，仍不进入 A/B。新 A、A+、B、C1 各比旧 local_rerun 多 1；旧 local_rerun 的切换数为 614，frozen 对照清单为 613，本次同批 traceable 记录为 613，不强行对齐这两个旧值。
- VOC2007 B/32 有一个状态变化：sample 1118 / image 2189（diningtable）的四个策略区域 `3→7`，旧 B/C1 变为新非 B；新 B 比旧值少 1。
- 新旧中旧 masks 和完整旧候选中间量没有保留，因此不能仅凭这些差异把原因归为 MPS 浮点误差。新记录保存了本次完整 masks、bbox、逐候选 selection/held-out 响应，作为本次 local rerun 的权威版本；旧 frozen 结果没有被覆盖。

## 可追溯记录与大文件

`verification/<setting>/same_record_checks.json` 全部为 `PASS`，并附有逐图核验 CSV。核验覆盖批次完整性、patch 分区、bbox 重算、held-out 聚合、文本归一化、图像/类别顺序、可行集、CCI/WF/Mean/Max-.1 选择和 C0/C1/C2/r 分解。

完整批次 NPZ、`response_raw.npy`、`response_norm.npy` 和 masks 保留在本机：

`/Users/von/Projects/cci-local-recovery/priority123_traceable_20260915/full/`

每个设置的 `SHA256SUMS.txt` 已展开复制到 `manifest/run_sha256/`；所有本机大文件的绝对路径、字节数和 SHA256 在 `traceable_results/input_hashes.csv` 中。因体量约 1.4 GB，本 GitHub 目录不上传这些张量，也不上传 ZIP；可读的逐图 CSV、汇总、核验结果、脚本、配置、日志和哈希清单均展开保存。

## 第 3 项只读核验

当前论文底稿中已存在 Wang & Wang, NeurIPS 2022 的引用，并明确区分：该文 max contrast 取原图非目标 logit 最大类别；本文 worst foil 取区域干预后 score drop 最大的非目标类别。`IDEA` 引用仍在当前底稿 references 中。本轮没有恢复旧论文或修改这些文件。

## 运行入口

脚本位于 `scripts/`；其中 `run_traceable_capture.py` 和 `verify_traceable_batches.py` 的源文件也保留在 `paper/review_round7_20260915/priority123/`。实际运行配置、依赖、输入来源和 SHA 位于 `manifest/`、`runs/<setting>/` 与 `traceable_results/analysis_metadata.json`。

旧 frozen/local_rerun 仅作为对照，论文编辑、排版和编译不属于本目录任务。
