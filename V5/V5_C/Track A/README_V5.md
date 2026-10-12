# Track A V5 Test Pack

生成时间：2026-05-15

这个 V5 包现在有两个用途：

1. 作为 0.20 目标下 patched rerun 冲刺的分发包；
2. 作为 rerun 失败时的 V5 fallback 安全网。

当前基线是 public score 0.130199999 的 600 行提交：

`submission_600_diag_lora_boundary_resubmit.csv`

## 目录

- `RERUN_SPRINT_PLAN.md`：当前主线，给 A/B/C/D 的 0.20 冲刺计划。
- `BCD_QUICK_START.md`：发给 B/C/D 的短版执行清单。
- `V5_EXPERIMENT_LOGIC.md`：详细解释赛题目标、当前系统、低分原因、V5 测试目的和拿到分数后的优化闭环。
- `TEAM_SPLIT.md`：按 B/C/D 分工写的执行清单。
- `submissions/`：可直接上传平台测试的 600 行 CSV。
- `reports/repair_report.md`：每个候选文件的改动数、F1 分布、动作转移。
- `reports/repair_summary.json`：机器可读的完整摘要。
- `reports/A_environment_api_check.md`：A 侧环境、API/token、fallback CSV 格式检查记录。
- `reports/C_f2_pair_analysis.md`：F2 pair 结构违规和 10 行修复说明。
- `phase2_run.py`：patched rerun 的主入口，B 直接在 V5 里修改。
- `skills/`：完整主流水线 skill modules，包括 candidate generator、feature engine、API gatherer。
- `rag_retriever_v2.py`：结构化 RAG / KNN 检索模块。
- `scripts/f1_distribution_repair.py`：离线复现候选文件的脚本。
- `scripts/C_f2_pair_repair.py`：离线复现 F2 10 行 pair 修复的脚本。
- `results/phase2/...`：脚本所需的 diagnostics、KNN votes 和生成结果。
- `data/Phase_1/train.json`：Phase1 offline smoke test 数据。
- `data/Phase_2/test.json`：Phase2 rerun / action parsing / 格式校验数据。
- `phase2_audit.py`：提交前格式检查脚本。

## 当前两次提交策略

全队只剩 2 次正式提交机会，所以 `submissions/` 里的文件不是提交顺序，而是候选池。不要按列表逐个上传。

注意：V5 现在已经包含完整 rerun 主流水线。B/C/D 不需要再切到其他版本目录工作。

```text
/home/huangxiaolin/AITO/V5/Track A
```

B 的 patch 目标就是 V5 里的 `phase2_run.py`、`skills/candidate_generator.py` 和新增 guard / smoke scripts。A 也从 V5 启动 patched rerun。

如果目标是 public score 0.20，默认建议：

1. 第 1 次正式提交：优先 patched rerun。
   - 必须包含 candidate filter、prompt/cardinality constraint、post-output guard。
   - 必须重放 2 个 audited LoRA overrides。
   - 如果 C 的 F2 calibration 在 Day 1 noon 前交付，也并入 Sub1。

2. 第 2 次正式提交：根据 patched rerun public score 决定修补或回退。
   - `>=0.180`：Sub1 + LLM F1 boundary expansion。
   - `0.150-0.180`：Sub1 + C 的 F2 patches + V5 calibrated KNN60。
   - `0.130-0.150`：Sub1 + B attribution 标出的 targeted manual fixes。
   - `<0.130`：回退 `submission_600_v5_11_baseline_plus_f2_pair_fix.csv`。

如果 B patch、Phase1 smoke、cloud API/token 任一失败，则切回 V5 fallback：

```text
Sub1 = submissions/submission_600_v5_07_adj_only.csv
Sub2 = submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv
```

详细执行看：

```text
RERUN_SPRINT_PLAN.md
TEAM_SPLIT.md
```

候选文件说明：

1. `submissions/submission_600_v5_07_adj_only.csv`
   - 主假设：只验证 F1/F1_5BS 的 `adjust_azimuth` 是否应被替换。
   - 改动：60 行。
   - 主要转移：`adjust_azimuth -> add_neighbor 36`，`adjust_azimuth -> decrease_a3 24`。
   - F1 分布：`decrease_a3 78 / lift_tilt 71 / add_neighbor 49 / insufficient_data 45 / pdcch_2sym 26 / adjust_azimuth 0 / check_server 0`。

2. `submissions/submission_600_v5_01_knn_conf50.csv`
   - 主假设：KNN vote share >= 0.50 的中等强度修复。
   - 改动：84 行。
   - F1 分布：`decrease_a3 61 / insufficient_data 54 / add_neighbor 53 / lift_tilt 59 / pdcch_2sym 30 / adjust_azimuth 10 / check_server 2`。

3. `submissions/submission_600_v5_02_hardblock_adjust.csv`
   - 主假设：F1/F1_5BS 禁止 `adjust_azimuth`，并叠加 KNN 非 adjust 修复。
   - 改动：107 行。
   - F1 分布：`decrease_a3 67 / insufficient_data 64 / add_neighbor 57 / pdcch_2sym 30 / lift_tilt 47 / check_server 4 / adjust_azimuth 0`。

4. `submissions/submission_600_v5_03_knn_all.csv`
   - 主假设：完整 KNN vote 更接近 Phase1 均衡分布。
   - 改动：101 行。
   - F1 分布：`decrease_a3 61 / insufficient_data 64 / add_neighbor 57 / pdcch_2sym 30 / lift_tilt 47 / adjust_azimuth 6 / check_server 4`。

辅助候选：

- `submissions/submission_600_v5_04_knn_conf60.csv`：更保守，主要验证高置信 KNN 是否稳定。
- `submissions/submission_600_v5_05_knn_conf40.csv`：介于 conf50 和 all 之间。
- `submissions/submission_600_v5_06_knn_conf50_plus_check.csv`：只比 conf50 多放开全部 check_server KNN vote，用来专测 check_server 是否有微弱收益。
- `submissions/submission_600_v5_08_adj_only_tau70_reference.csv`：只按 candidate score ratio >= 0.70 改 6 行，太窄，主要作为离线参考。
- `submissions/submission_600_v5_09_adj_only_plus_calibrated_knn60.csv`：`adj_only` 加上 Phase1 holdout 支持的 KNN60 transition，适合第 2 次整合参考，不建议第 1 次直接交。
- `submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv`：`v5_07` 加 F2 10 行结构修复，首提交涨分后的默认第 2 次候选。
- `submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv`：只在 0.1302 baseline 上加 F2 修复，首提交持平或跌分时的 fallback。
- `submissions/submission_600_v5_12_adj_only_plus_knn60_plus_f2_pair_fix.csv`：`v5_09` 加 F2 修复，只有 B 认为 KNN60 值得带入时才考虑。

Phase1 holdout 校准结果：

- `share 0.50-0.59` 的 `lift_tilt -> insufficient_data` 精度低于旧规则，不能直接信任 conf50。
- `share >= 0.60` 的 `lift_tilt -> insufficient_data` 和 `decrease_a3 -> add_neighbor` 更有正信号，可作为第 2 次整合候选。

## 复现命令

在本目录运行：

```bash
/home/huangxiaolin/AITO/bin/python scripts/f1_distribution_repair.py
```

格式检查示例：

```bash
/home/huangxiaolin/AITO/bin/python scripts/check_600_format.py submissions/submission_600_v5_07_adj_only.csv
```

注意：平台需要 600 行 ID 模板。不要提交 500 行 raw result。

## 当前判断

V5 CSV 不是当前 0.20 主攻，而是 rerun 失败或 Sub2 回退时的安全网。

- 0.20 主攻：patched rerun。
- V5 fallback：`v5_07 -> v5_10`。
- rerun catastrophic fallback：`v5_11`。
- KNN60 可选：`v5_12`，只有 B 建议时使用。
