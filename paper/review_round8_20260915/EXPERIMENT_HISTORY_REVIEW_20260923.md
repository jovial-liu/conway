# 实验历史核对（2026-09-23）

本次只读检查 GitHub 分支的旧运行配置、新运行元数据、新旧对照 CSV 和核验报告，没有重新运行模型。

## 结论

- 仓库保存旧 `local_rerun_2026-09-13` 与新 `priority123_20260915` 的运行记录；不能说整个研究只运行过一次。
- 最新主结果来自一轮完整重跑，包含 COCO/VOC × OpenAI CLIP B/16、B/32 四个设置。每个 COCO 设置 27,708 张，每个 VOC 设置 1,943 张。
- 两版记录的聚类规则都是 `1701 + sample_index`。没有发现使用多组 base seed 独立完整重复实验并汇总均值/标准差的证据。
- K-means `n_init=3` 是单次聚类中的初始化次数；10,000 次 bootstrap 是统计重采样；100 个 Monte Carlo seeds 是 foil 子集概率核验。三者都不等于完整实验多种子重复。
- 新旧 CCI 选择在 COCO B/16、VOC B/16 上一致，在 COCO B/32 上有两张变化，VOC B/32 上有一张变化。旧 masks 和完整中间量未保留，不应擅自归因于 MPS 浮点误差。
- 表 4 明确使用旧运行固定人群，而其他主结果使用新完整记录；本次没有更改实验数值或这一数据来源安排。

## 可追溯文件

- `experiments/local_rerun_2026-09-13/rerun_workspace/config/local_rerun_config.json`
- `experiments/priority123_20260915/config/traceable_capture_config.json`
- `experiments/priority123_20260915/runs/*/run_metadata.json`
- `experiments/priority123_20260915/traceable_results/new_vs_old_summary.csv`
- `experiments/priority123_20260915/README_zh.md`

## 给老师的准确回复

老师，不是整个实验只跑过一次，之前有一版运行，后面又对 COCO/VOC 和 B/16、B/32 四个设置做了完整重跑。文中的 one run 是指主表统一取自同一轮完整保存记录的结果，避免混用不同运行的数据。不过这两版采用同一套随机种子规则，目前没有做多组随机种子的完整重复实验，因此主表也不是多种子平均结果。

## 稿件已同步澄清

已在正文明确完整重跑与独立 base-seed 重复实验的区别、bootstrap 的范围，以及表 4 的单独旧记录比较。失败事件符号改为明确的 E，统计数据未变。
