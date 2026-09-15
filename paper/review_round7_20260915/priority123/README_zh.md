# 优先项 1–3：当前状态与本机执行交接

## 当前真实状态

- 3：已完成论文修改。核对 NeurIPS 2022 Wang & Wang 第 3.3 节，补入引用，区分原图 logit 的 argmax 与区域干预下降量的 argmax，并明确固定人群、等基数对照、候选/预算分解三项增量。
- 1：分析脚本已写好并通过合成符号翻转测试；**尚未读取真实响应、未生成四设置归一化结果**。
- 2：同批记录采集入口及核验脚本已写好；核验器通过合成空间/响应记录测试。**尚未运行完整模型采集，尚未冻结新的权威数据版本，尚未统一重算全部论文表格**。

当前工作区没有用户本机的 raw/norm NPZ、数据集图片及模型权重。清单中的 SHA256 不是数据本身。不要把脚本准备完成等同于实验完成。新论文只完成第 3 项，不包含推测的第 1、2 项结果。

## 1：固定原始 A 的归一化比较

在仓库根目录执行，RESPONSE_ROOT 是本机包含四个 setting 子目录的 responses 路径：

```bash
python paper/review_round7_20260915/priority123/analyze_normalization.py \
  --responses-root /absolute/path/to/responses \
  --diagnostics experiments/reviewer_controls_2026-09-13/results/per_image_restricted_foil_diagnostics.csv \
  --manifest experiments/reviewer_controls_2026-09-13/manifest/large_local_artifacts.csv \
  --out /absolute/path/to/normalization_results
```

必须提供四个 `<setting>_cci_full_class_responses.npz`，setting 为 coco_openai_b16、coco_openai_b32、voc2007_openai_b16、voc2007_openai_b32。脚本使用 manifest 校验每个 NPZ 的 SHA256，逐行验证图像 ID、类别、CCI 区域，并验证重算 normalized margin 与原诊断一致。固定原始 A，不按 raw 重筛。

输出 `normalization_fixed_A.csv`：raw/normalized 失败率、符号翻转率和两个方向、hardest-foil 一致率，以及固定原 normalized B 上的 unrestricted pass capacity。`provenance.json` 记录输入哈希及分母。未报告可行 oracle，因为旧 NPZ 的 feasible_set_size 不能替代逐候选 selection scores；逐模板结果也不能由平均响应倒推。

真实结果通过校验后，再将四行表加入论文。若敏感性大，应据实调整结论。

## 2：同一次生成保存 masks 和响应

从新的空目录采集，沿用原 runner 的环境、模型权重、数据和预处理。下面只是单设置命令模板，需对四设置运行。禁止写入旧 frozen/reviewer_controls 输出目录。

```bash
python paper/review_round7_20260915/priority123/run_traceable_capture.py \
  --dataset coco --model openai_b16 \
  --data-root /absolute/path/to/data \
  --weights-dir /absolute/path/to/weights \
  --out-dir /absolute/path/to/new_release/responses/coco_openai_b16 \
  --device mps --batch-size 2
```

先用 `--samples 8` 和另一个全新 smoke 目录检查本机环境；通过后使用空 full 目录且不指定 samples。此入口为独立副本，不修改原 runner。它禁用 resume，避免新记录与旧无 masks 的进度混合。

每个批次在完成同一次响应计算后保存：sample/image ID、候选 patch membership、目标 bbox patch membership、逐候选 bbox precision、selection 全类分数和 target 分数、逐模板 held-out 响应、raw/normalized 聚合响应、文本均值归一化系数。另保留原 metadata、运行配置和输出文件 SHA256。其响应说明已修正为 original-minus-masked，与实际代码一致。

```bash
python paper/review_round7_20260915/priority123/verify_traceable_batches.py \
  --setting-dir /absolute/path/to/new_release/responses/coco_openai_b16 \
  --out /absolute/path/to/new_release/verified/coco_openai_b16
```

核验器检查哈希、无缺失/重排的批次、patch 分区、bbox 重算、逐模板聚合和归一化、批次与完整 NPZ 对应，再用同一套记录计算 A、full/absent/random failure 和 C0/C1/C2/r。输出仅是新记录的核心计数，**不是全部主表的最终替代**。

完成第 2 项还需：四设置全量采集通过；记录模型权重哈希与输入图片/标注版本；从新记录统一重算预算、baseline、条件代价与 CI；对照旧表并解释差异；冻结完整记录及汇总的版本；论文所有数值统一切换后才能标记完成。不能把旧 A 与新 masks 混合后称为同一次生成；旧固定 A 比较与新权威人群必须分别记录。

## 回传材料

优先回传第 1 项四行 CSV 和 provenance；第 2 项回传四设置核验 JSON、逐图诊断、权威记录 manifest 和模型/输入来源。大张量可提供可访问存储链接，无须塞进论文附件。原始 masks 若确实未保存，只能用新完整记录建立新的可追溯版本，不能追认旧 masks 已获证明。

## 本轮验证范围

Python 语法检查、已知符号翻转的合成用例、含可行机会被 WF 漏选的同批空间记录合成用例通过。没有运行真实 CLIP 推理；本机 smoke 和全量验证仍是必需步骤。
