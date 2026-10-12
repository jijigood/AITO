# B/C/D Quick Start

生成时间：2026-05-15

这份是给 B/C/D 的短版执行清单。完整背景看 `RERUN_SPRINT_PLAN.md`，细节看 `TEAM_SPLIT.md`。

## 总目标

如果目标是 public score 0.20，主线不是 V5-only，而是 patched rerun：

```text
Sub1 = patched rerun
     + candidate / prompt / post-output guards
     + 2 audited LoRA overrides
     + C 的 F2 calibration 如果能赶上
```

V5 CSV 是安全网，不是默认主攻。

## 代码位置说明

V5 包现在已经包含完整 rerun 主流水线。B/C/D 直接在 V5 里工作：

```text
/home/huangxiaolin/AITO/V5/Track A
```

## B

任务：修 rerun contract，给 A 一个能安全全量 rerun 的代码版本。

必须改：

```text
/home/huangxiaolin/AITO/V5/Track A/skills/candidate_generator.py
/home/huangxiaolin/AITO/V5/Track A/phase2_run.py
post-output guard module/script
Phase1 smoke script
```

相关文件：

```text
skills/candidate_generator.py
phase2_run.py
skills/feature_engine.py
skills/scenario_router.py
skills/api_data_gatherer.py
rag_retriever_v2.py
data/Phase_1/train.json
data/Phase_2/test.json
```

必须保证：

```text
F1/F1_5BS：single-answer，不允许 adjust_azimuth 成为最终答案
F2：exactly 2 codes，只允许合法 pair
F2：禁止 add_neighbor / decrease_a3
F4M：exactly 2 decrease_interfreq
```

交给 A 的 smoke 结果：

```text
F1 top1 acc >= 0.80
F2 legal-pair top1 acc >= 0.50
F4 top1 acc >= 0.95
invalid F2 outputs = 0
```

Day 2：根据 Sub1 public score 写 `reports/B_rerun_attribution.md`，推荐 Sub2。

## C

任务：做 F2 pair calibration，不再主攻 `check_server`。

已有基础：

```text
scripts/C_f2_pair_repair.py
reports/C_f2_pair_analysis.md
submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv
submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv
submissions/submission_600_v5_12_adj_only_plus_knn60_plus_f2_pair_fix.csv
```

Day 1 noon 前尽量交：

```text
scripts/C_f2_pair_calibration.py
reports/C_f2_pair_calibration.md
ranked F2 flip list
可复现 patch CSV/script
```

规则：

```text
只改 confidence >= 0.70 的 F2 pair flip
不要为了配成 62/38 硬改
不要全量 API / LLM
赶不上 Day 1 noon 就转为 Sub2 候选
```

## D

任务：人工 spot-check，不造大提交。

Day 1 AM 审：

```text
v5_07 的 60 个 F1 flips
24 个 adjust_azimuth -> decrease_a3
36 个 adjust_azimuth -> add_neighbor
```

Day 1 PM 审：

```text
v5_10 的 10 个 F2 fixes
```

交付：

```text
reports/D_spotcheck_v5_07.md
scenario_id | old_action | new_action | keep/suspicious/drop | reason
```

不要：

```text
不要全量重跑 API
不要自己生成大提交
不要把人工直觉直接混进 Sub1
```

## 回退规则

如果 B patch / smoke / cloud API 任一失败：

```text
Sub1 = submissions/submission_600_v5_07_adj_only.csv
Sub2 = submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv
```

如果 patched rerun Sub1 分数低于 0.130：

```text
Sub2 = submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv
```
