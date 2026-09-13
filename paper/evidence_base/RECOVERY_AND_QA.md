# 来源核对与统计说明

## 固定来源
仓库：jovial-liu/cci
分支：reviewer-identification-round2-v1
固定提交：b70a6e18d7db52af3750635e76a5b457693eb546

1. `derived/reviewer_cci_preserving_v1/01_original_cci_per_sample.csv`
   Git blob SHA: de8bfdfb4e4c7ba9ad3539fc9ac8e367f07bb46f
   已读取完整 27,708 行；本包 coco_b16_baseline.csv 保留本轮统计所需原列，未舍入。
2. `derived/reviewer_cci_preserving_v1/03_target_preserving_per_sample.csv`
   Git blob SHA: 531fa15b2b8b63fa3f2cf2f5a6e9ebbf2da9e8f6
   已读取原文件；本包 coco_b16_repair_e002.csv 保留 epsilon=.02 的全部 27,708 行、全部列，未舍入。
3. `paper_handoff/reviewer_round2/12_feasible_baselines.csv`
   含四设置 full_foil/rtp 的未舍入总体变化和精确切换数。
4. `paper_handoff/reviewer_round2/14_sparse_repair_summary.csv`
   提供四设置切换数、改善比例、符号修复比例。与上表逐设置核对一致。

固定版本目录：https://github.com/jovial-liu/cci/tree/b70a6e18d7db52af3750635e76a5b457693eb546

## 防止历史版本混用
旧 multisetting/04_multisetting_frontier.csv 中部分 setting 的切换数为 626/47/46，和当前 round-2 的 613/50/38 不符。因此不从那份旧 frontier 移植其它设置的新统计。
当前分析使用 round-2 full_foil/rtp 与 sparse summary，两者的切换数和条件 margin 增益通过恒等式逐项一致。COCO B/16 逐图文件同时核对了 image_id 对齐、基线 margin 和 bbox，未切换图像三个 delta 均为零。

## 新增统计
COCO B/16 的四格计数 (A且F, A且非F, 非A且F, 非A且非F)=(11411,6314,9105,878)。A 是 bbox precision≥.5 且 aggregate_pmean_all>0；F 是 aggregate_pmax<0。
条件失败率=11411/17725。使用 10,000 次图像 bootstrap、seed=1701。对四格计数进行 multinomial 重采样与逐图重采样这些二元事件的统计分布严格等价。CI 使用百分位数方法；不是从舍入百分数恢复的计数。
COCO B/16 的条件均值区间在观测到的 663 张切换图像内进行 10,000 次配对图像重采样，seed=1702；不表示参数选择或模型训练的不确定性。
其它设置的条件均值使用未舍入的 full-sample delta / switch_rate；未把总体 delta CI 除以切换率来伪造条件 CI。

VOC B/16 的 bbox 条件均值=-.1319895678，不是评审依据舍入数反算的 -.1323。两者差异来自舍入。VOC B/32 的 raw target 条件均值=-.0348642265；不要与归一化 target 响应混用。

## 保留的局限
原始 frozen_drop_table 与 normalized_text_cache 在 LARGE_ARTIFACT_MANIFEST.csv 中仍标记 not_copied=True。本包并非完整原始响应张量备份。历史 sufficient_statistics.npz 属于另一条分析链，未未经对齐直接使用。
尚未确认仓库能否被无认证的审稿人访问，不将本次连接成功等同于公开复现入口。

## 验收
主稿与 arXiv 包均须独立编译为五页；第 5 页为参考文献。未定义引用、overfull 警告均应为零。作者文件和全部图件与上一版逐字节一致。统计重算入口为 analyze_recovered.py，不依赖实验室。
