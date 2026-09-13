# ccf0 论文与实验交接：新对话先读这里

交接日期：2026-09-13。仓库：jovial-liu/conway。
**工作分支：paper/verified-rerun-integration。不要只读 main。**

论文：Beyond Mean Foils: Auditing Worst-Foil Specificity in Frozen CLIP Region Explanations。

## 1. 当前状态与唯一入口

已完成论文整合、pdfLaTeX/BibTeX 编译、五页检查、引用/溢出检查、逐页视觉检查、主稿/arXiv 渲染一致性检查、ZIP 完整性检查。
**尚未向 arXiv 提交，尚未合并 main。**

最终论文与实验快照：**1817e1666161fa902869affe61b55f7c98c45668**。
本交接是该提交之后的文档更新，不改变论文或实验数据。

| 内容 | 仓库相对路径 |
|---|---|
| 最新可编辑主稿 | paper/local_rerun_revision/source/main.tex |
| 六作者 | paper/local_rerun_revision/source/authors.tex |
| 定稿图件 | paper/local_rerun_revision/source/figures/ |
| 最终论文 PDF | paper/local_rerun_revision/generated/paper_final.pdf |
| 完整论文 ZIP | paper/local_rerun_revision/generated/paper_complete.zip |
| arXiv 展开源码 | paper/local_rerun_revision/arxiv_source/ |
| arXiv 上传 ZIP | paper/local_rerun_revision/generated/arxiv_source.zip |
| arXiv 预览 PDF | paper/local_rerun_revision/generated/arxiv_preview.pdf |
| 实际发布校验 | paper/local_rerun_revision/generated/build_report.json |
| 逐页 PNG、编译日志 | paper/local_rerun_revision/generated/ |
| 修改/编译说明 | paper/local_rerun_revision/README.md、BUILD.md |
| 原始底稿，仅供对照 | paper/evidence_base/ |

[最终 PDF](https://github.com/jovial-liu/conway/blob/1817e1666161fa902869affe61b55f7c98c45668/paper/local_rerun_revision/generated/paper_final.pdf)
· [当前主稿](https://github.com/jovial-liu/conway/blob/1817e1666161fa902869affe61b55f7c98c45668/paper/local_rerun_revision/source/main.tex)

不要把 evidence_base/source/main.pdf、旧上传 liu_submission_compliance_final(1).pdf 或 Mac 未完成编辑副本当作当前定稿。

## 2. 用户确认的分工和约束

- 本地 MacBook M4 Pro Codex **只负责实验、统计核验和数据搬运**。不要再让它编辑论文、排版或安装 TeX。
- 接手的云端助手负责论文修改、图表排版、编译和最终打包。
- 保持五页：前四页论文，第五页参考文献。
- 六作者顺序：Kaixin Liu、Zhipeng Ye、Feng Jiang、Qiufeng Wang、Hao Li、Xihang Zhou。机构、共同一作、通讯作者按 authors.tex。
- 保留 IDEA 引用（ye2026idea）。
- Figure 1 钢蓝、砖红、橄榄绿配色已定稿，标题下三条线已删。三张图件均保留原 Git blob，不擅自重画。
- 不恢复已移除的 Acknowledgment，不因切换对话重做旧版。
- Frozen 与 local_rerun 分别标明，不能混算、覆盖或为了匹配旧数调数据。
- 实验已完成，无需默认追加全量模型推理。
- 不自动提交 arXiv、不强推、不覆盖 main。不要反复确认已授权的工作。

## 3. 版本链

| 版本 | 提交 |
|---|---|
| 实际 MPS 推理及原统计 | 59fc00e3948015379c8dcc563283a5bdda7f1226 |
| 独立统计核验、定义与种子修正 | 1cdffa7abdca1f84c1120362fbaf04d9fb52d13b |
| 原始展开论文包上传 | 17c6ca5fb254e2eea0c98d479509aaf38af558c5 |
| 首轮未编译源码，已被取代 | 12cfab20e4fc533419b8ea68feae95719ffd1b68 |
| 最终五页论文与生成文件快照 | 1817e1666161fa902869affe61b55f7c98c45668 |

此前“无法编译”“六页”“尚无 PDF”均已过时；之后用 GitHub Actions 实际编译、修正数学符号、精简文字完成了五页版。

## 4. 实验数据目录

前缀 R = experiments/local_rerun_2026-09-13/rerun_workspace/。

- R/runs/<setting>/per_image.csv：每图一行，image_id/sample_index、目标、CCI/WF/Mean/Max-.1 选择、可行集大小、候选 selection target/mean-foil/max-foil 数组、候选 held-out margin/mean/target/bbox 数组及选中指标。
- R/runs/<setting>/run.log、run_metadata.json：实际命令、MPS 设备、进度、样本数、版本、耗时。
- R/runs/*_smoke_cpu/、*_smoke_mps/：小样本设备对照。
- R/scripts/run_local_rerun.py：推理；analyze_local_rerun.py：原统计。
- R/scripts/validate_outputs.py、write_provenance.py：检查与溯源。
- R/config/local_rerun_config.json：配置。
- R/manifest/dependency_versions.txt、source_and_hashes.csv：依赖、输入来源、已记录哈希。
- R/results/：原统计、八份逐图直接差值。最终报告优先用 corrected 统计。

| setting | 记录数 | per_image.csv 字节数 | local WF 切换 | local sign repair |
|---|---:|---:|---:|---:|
| coco_openai_b16 | 27,708 | 49,338,958 | 663 | 42 |
| coco_openai_b32 | 27,708 | 47,876,646 | 614 | 48 |
| voc2007_openai_b16 | 1,943 | 3,504,239 | 50 | 11 |
| voc2007_openai_b32 | 1,943 | 3,399,748 | 38 | 6 |

共 59,302 个图像—模型记录。同数据集两模型使用相同图像，不能称为 59,302 张独立图像。
候选数组是当前策略所需的**汇总响应，不等于完整逐模板×逐类别张量或原始候选掩码**，不能据此宣称任意 foil 子集实验都可直接重算。

## 5. 最终统计与核验

前缀 V = experiments/verification_2026-09-13/。

必读 V/verification_report.md 和：
- results/corrected_paired_difference_ci_10000.csv
- results/corrected_screening_failure_rates.csv
- results/corrected_switch_subset_changes_ci.csv
- results/corrected_switch_margin_improvement_vs_sign_repair.csv
- results/corrected_feasible_set_size_distribution.csv
- results/corrected_runtime_summary.csv
- results/recomputed_vs_existing_statistics.csv
- results/strategy_choice_validation.csv
- scripts/verify_local_rerun.py
- manifest/verification_metadata.json
- logs/verification_run.log

记录：策略/候选数组校验失败 0；768 项比较中统计数值不一致 0（报告阈值 1e-5），36 项为定义文字修正；screening 各项实际种子已列出。
独立脚本直接读逐图数据，不导入原分析模块。完整重算在本地完成；云端核对报告、代码并复算八份逐图差值均值，不要误称云端又跑了模型。

算法：10,000 次 NumPy default_rng 图像 bootstrap，percentile 2.5/97.5，base seed 1701 加记录的 metric-specific seeds。
直接比较先同图相减再重采样；条件失败率联合重采样分子/分母；切换 CI 从子集向量计算，不能将总体 CI 除以切换率。区间为逐项区间，未校正多重比较。
旧 config 中 candidate−CCI 的泛称不是直接基线比较定义；看 V/corrected_config_notes.md 和 corrected 表实际列定义。

## 6. 科学定义、结果和表格来源

CCI-top1 = selection target drop 最大的单区域，不是完整 CCI heatmap。
epsilon=.02 可行集：selection target >= CCI target−.02。
WF 最大化 target−max(non-target foil)；Mean 最大化 target−mean(foil)；
Max-.1 最大化 target−.1*max(foil)。WF−Max-.1 是比较，不是策略。
K=8，K-means n_init=3、max_iter=50、seed=1701+sample_index。
selection prompt 与三个 held-out prompts 分开；raw 和 normalized target 分开。
筛查 A：bbox>=.5 且 normalized aggregate mean-foil margin>0；
失败 F：normalized aggregate worst-foil margin<0。
all-class mean 与 non-target mean 正号筛查数学等价，但数值不同。

| CCI local rerun | 条件失败率 P(F|A) | 95% CI |
|---|---:|---|
| COCO B/16 | 64.37% | [63.67%,65.05%] |
| COCO B/32 | 64.78% | [64.07%,65.47%] |
| VOC B/16 | 42.27% | [39.45%,45.05%] |
| VOC B/32 | 41.16% | [38.40%,43.91%] |

- Table 1：frozen CCI→WF 全样本结果。
- Table 2：local rerun 四设置筛查。
- Table 3：local WF−Mean/WF−Max-.1 直接配对 margin，显示值乘 1000。
- Table 4/5、Figure 2/3：frozen foil/tolerance/归档图证据。
- Table 6：local 切换子集 raw target 和 bbox CI；bbox 乘 100，为百分点。
- Figure 1 为定稿方法图，不含新实验测量。

四设置两种直接 margin 比较的逐项 CI 均大于零；相对 Max-.1 增益很小，不应称全面/无代价优越。
切换 bbox：COCO B/16 −4.95 个百分点 [−8.33,−1.45]；
VOC B/16 −13.20 个百分点 [−24.62,−1.80]。
Margin improvement、sign repair、target response、locality 是不同结局。

## 7. 差异及未解决项目

Frozen 切换 663/613/50/38；local 为 663/614/50/38。
Frozen COCO B/16 通过 17,725、联合失败 11,411；local 为 17,726、11,410。
状态变化图像 135671、185335、213809；CCI label 变化共 10 张。
部分响应差异明显大于舍入；缺 frozen 候选数组/中间量，无法确定聚类、算术、预处理等原因。不能归因于已证实的 MPS 误差。
同一 region 编号不保证候选掩码相同。
B/32 frozen 缺对齐逐图字段，无法枚举 613→614 图像。
local COCO B/16 多可行样本 1,304（frozen 1,303），不可混用。
COCO 两模型耗时都是续跑部分，不是完整耗时。
CPU/MPS 检查仅 1–2 图/设置，不证明全量后端等价。
限制 foil 后的条件失败率、新 K/干预鲁棒性仍未补齐；不宣称全部历史实验完整复现。

## 8. Frozen 来源与未上传输入

本仓库 paper/evidence_base/evidence/ 有旧稿依赖的恢复 CSV、协议、代码及图件数据。
更多历史归档位于 jovial-liu/cci：
- reviewer-identification-round2-v1 分支的 derived/reviewer_cci_preserving_v1/：
  01_original_cci_per_sample.csv、03_target_preserving_per_sample.csv。
- 提交 0ec6831cca419d106ffe6ba6aef255819898b7a6 的 experiment_results.zip。
这些是 frozen 来源，不是本次 local_rerun。

**COCO/VOC 原图、模型权重和本机完整缓存未上传 GitHub。**
公开下载 URL 和已有 SHA256 在 R/manifest/source_and_hashes.csv。
部分早期张量没有归档，不能保证从仓库恢复所有历史控制实验。
本次逐图结果足够重算当前固定策略的主要统计。
paper_complete.zip 是论文交付包，不含全部大 CSV；完整实验应克隆本分支。

## 9. 新会话启动与可复现命令

先读本文、当前 source/main.tex、corrected 统计、build_report.json。
没有新增任务时不要默认重做论文/实验。

从独立目录克隆：
    git clone --branch paper/verified-rerun-integration --single-branch https://github.com/jovial-liu/conway.git

最终 release 精确快照可检出 1817e1666161fa902869affe61b55f7c98c45668；该快照本身早于本交接文件。

从仓库根目录重算原 local 统计到独立输出：
    python experiments/local_rerun_2026-09-13/rerun_workspace/scripts/analyze_local_rerun.py --runs-root experiments/local_rerun_2026-09-13/rerun_workspace/runs --out-dir /tmp/cci-recomputed --bootstrap 10000 --seed 1701

独立 verify_local_rerun.py 的 --project-root 采用原 Mac 布局
（rerun_workspace、verification_2026-09-13/frozen 等）。仓库路径不同：
先读脚本、显式映射，不盲目执行或伪造缺 frozen 输入。
历史 /Users/von 路径是来源记录，不是云端可访问路径。

handoff/FILE_INDEX.json 和 CSV 枚举 release 的全部论文/实验/编译流程文件。
包含 Git blob SHA1、大小和部分已有 SHA256；两类哈希不混称。
运行 python handoff/verify_inventory.py --repo-root . 可检查 checkout 文件完整性。
该脚本仅检查完整性，不重算统计或推理；本次未执行此新脚本。

## 10. 编译与后续编辑注意事项

编译流程：.github/workflows/verify-paper.yml，限定修订分支。
修改 source、arxiv_source 或 verify_build.py 会触发编译；新增本交接不触发。
依赖 TeX Live + PyMuPDF，pdfLaTeX/BibTeX 四步。
两处 main.tex 必须同步，五页图件和作者约束保持。

verify_build.py 检查编译、五页、引用、溢出、页面边界、主稿/arXiv
渲染及 visual_review.json 的已审阅哈希。内容变化后必须实际看新 PNG
再更新审阅哈希，不自动把新哈希标为“已审阅”。
失败时 generated 中可能留旧 PDF，应对齐当前 report/Actions commit，
不能只看 PDF 文件名。当前 release report.status=passed。
两 PDF SHA256 可以不同（元数据），已验证页面一致。

成功构建：https://github.com/jovial-liu/conway/actions/runs/34744468616

本次环境无本地 shell，但已通过 GitHub 读写、Actions 编译和 base64
页面 PNG 完成审阅。新会话无 shell 时优先沿用，不转交论文给 Mac Codex。
