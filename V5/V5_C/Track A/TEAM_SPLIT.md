# V5 / Rerun 冲刺四人分工

生成时间：2026-05-15

## 当前主线

如果目标是 public score 0.20，V5-only 后处理方案上限不够。现在主线改为：

```text
Sub1 = patched rerun
     + candidate / prompt / post-output guards
     + 2 个 audited LoRA overrides
     + C 的 F2 pair calibration patches 如果 Day 1 noon 前交付

Sub2 = 根据 Sub1 public score 选择修复或回退
```

V5 CSV 不作废，但它们现在是安全网：

```text
如果 B patch 失败、Phase1 smoke 不过、cloud API/token 不稳：
Sub1 = submission_600_v5_07_adj_only.csv
Sub2 = submission_600_v5_10_adj_only_plus_f2_pair_fix.csv
```

## V5 文件关系

V5 现在是自包含的分发、实验、fallback 和 patched rerun 包。B/C/D 不需要再去其他版本目录找主流水线。

```text
/home/huangxiaolin/AITO/V5/Track A
```

关键文件已经放在 V5 根目录：

```text
phase2_run.py
rag_retriever_v2.py
skills/candidate_generator.py
skills/feature_engine.py
skills/scenario_router.py
skills/api_data_gatherer.py
data/Phase_1/train.json
data/Phase_2/test.json
```

B 的 patch 直接改 V5 里的文件。A 从 V5 运行 patched rerun。

## 两个必须先确认的决策

1. F1_5BS guard：按 F1 single-answer 处理。
   - exactly 1 code
   - 禁止 `adjust_azimuth`
   - 只允许 F1 safe actions：`add_neighbor/check_server/decrease_a3/insufficient_data/lift_tilt/pdcch_2sym`

2. F4M guard：保守约束。
   - exactly 2 codes
   - 两个 action 都必须是 `decrease_interfreq`
   - 当前 28 行已经满足，guard 只是防 rerun 发散

## Go / No-Go 门槛

在 A 启动 cloud API 全量 rerun 前，必须跑 Phase1 train smoke test。这个 smoke test 是离线 deterministic / candidate 层检查，不是最终 LLM 分数预测。

通过门槛：

```text
F1 single-answer top1 accuracy >= 0.80
F2 legal-pair top1 accuracy >= 0.50
F4 top1 accuracy >= 0.95，最好接近 1.00
```

失败处理：

```text
任何关键门槛失败 -> abort rerun -> 走 V5 fallback
不要临场调参硬跑 cloud API
```

## A：总控、API、集成、提交


A 的任务不是再造规则，而是控制风险和节奏。

Day 1 AM：

- 确认 public score turnaround 是分钟级；
- 确认 cloud API/token 可用；
- review B 的 rerun contract patch；
- 跑 Phase1 smoke test；
- 如果 smoke 失败，立刻切 V5 fallback。

Cloud API health check 示例：

```bash
curl -k https://120.46.145.77/no/health -H "Authorization: Bearer <TRACK_A_API_KEY>"
```

Day 1 PM：

- smoke 通过后，启动 patched `phase2_run.py` 全量 rerun；
- 观察 tool-call timeout、API error、空数据；
- rerun 完成后生成 600 行 submission；
- post-process：
  - 应用 post-output guard；
  - 重放 2 个 audited LoRA overrides；
  - 如果 C 的 F2 calibration 已交付，则合并；
  - 跑格式检查和 action distribution audit。

Day 1 evening：

- 提交 Sub1；
- 记录 public score；
- 交给 B 做 attribution。

Day 2：

- 按 Sub2 decision table 选最终提交；
- 提交 Sub2；
- 不再保留试错空间。

## B：rerun contract patch + attribution

B 的任务不是继续造 V5 KNN CSV，而是修主流水线，让 rerun 有可能打到 0.20。

### B Day 1 AM 交付

Patch 必须覆盖三层：

```text
1. candidate filter
2. LLM prompt/cardinality constraint
3. post-output guard
```

修改位置：

```text
/home/huangxiaolin/AITO/V5/Track A/skills/candidate_generator.py
/home/huangxiaolin/AITO/V5/Track A/phase2_run.py
新增 post-output guard module 或 audit script
新增 Phase1 smoke script
```

### 必修 bug

F1 / F1_5BS：

- single-answer top candidates 不允许 `adjust_azimuth` 成为最终 top1；
- F1_5BS 也按 F1 single-answer guard 处理；
- 保留合法候选用于 LLM 仲裁可以，但最终 deterministic/top guard 必须把 top1 拉回 safe F1 action。

F2：

- 不允许输出 4 codes；
- 不允许 `add_neighbor` / `decrease_a3` 进入 F2 multi-answer；
- F2 最终必须 exactly 2 codes；
- F2 最终 action pair 只能是：

```text
{adjust_azimuth, increase_power}
{decrease_power, press_down_tilt}
```

F4 / F4M：

- F4 single-answer 必须保持 `decrease_interfreq`；
- F4M exactly 2 codes，且两个 action 都是 `decrease_interfreq`。

### B smoke test

B 需要提供一份 3-5 行 smoke result 给 A：

```text
F1 top1 acc: <value>  pass/fail
F2 legal-pair top1 acc: <value>  pass/fail
F4 top1 acc: <value>  pass/fail
invalid F2 outputs: <count>
```

如果 smoke 不过，B 不要继续 debug 到晚上。A 直接切回 V5 fallback。

### B Day 2 交付

Sub1 public score 回来后，B 做 attribution：

- rerun 是否达到预期区间；
- F1 分布是否更接近 Phase1；
- F2 是否还有非法 pair/cardinality；
- 是否有明显 regression family；
- Sub2 应该修补还是回退。

交付文件建议：

```text
reports/B_rerun_attribution.md
```

## C：F2 pair calibration


C 的主线不是 `check_server`。当前 `check_server` 没有高置信证据，C 应该集中做 F2 pair selection。

### C Day 1 noon 前交付

目标：给 patched rerun 的 Sub1 提供可合并的 F2 pair calibration patch。如果赶不上 Day 1 noon，就转为 Sub2 候选。

任务：

1. 在 Phase1 F2 train 上跑 feature pipeline；
2. 统计 `top_issue -> legal pair` 条件分布；
3. 可选增强：统计 `(top_issue, dominant_feature_score_bucket) -> legal pair`；
4. 对 Phase2 card-2 F2 行计算 pair confidence；
5. 找出当前 pair 和 Phase1 majority pair 不一致、且 conditional confidence >= 0.70 的 10-25 行；
6. 输出 ranked flip list 和 patch CSV/script。

重点桶：

```text
Phase2 里 18 个 top_issue = late_handover 但当前为 tilt pair 的 F2 行
late_handover 更像 handover/mobility，不一定应该是 tilt pair
```

交付文件建议：

```text
scripts/C_f2_pair_calibration.py
reports/C_f2_pair_calibration.md
submissions/submission_600_C_f2_calibrated_*.csv
```

已有基础文件：

```text
scripts/C_f2_pair_repair.py
reports/C_f2_pair_analysis.md
reports/C_f2_pair_repair.json
submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv
submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv
submissions/submission_600_v5_12_adj_only_plus_knn60_plus_f2_pair_fix.csv
```

禁止：

- 不要只为了把 F2 pair 比例配成 62/38 而硬改；
- 不要扩大到 F3/F4/F4M，除非先证明结构违规；
- 不要用 LLM 或 API 全量重跑；
- 不要再把 `check_server` 当主线。

## D：spot-check / audit

难度：执行型。

D 的任务是帮 A 防止明显错改，不需要写复杂脚本。

### D Day 1 AM

审 `v5_07` 的 60 个 F1 翻转：

```text
24 个 adjust_azimuth -> decrease_a3
36 个 adjust_azimuth -> add_neighbor
```

重点：

- `adjust_azimuth -> decrease_a3` 是否有 late_handover / A3 / handover 证据；
- `adjust_azimuth -> add_neighbor` 是否有 missing_neighbor、未配置邻区、neighbor advantage 证据；
- 标出 weak-evidence flips。

交付：

```text
reports/D_spotcheck_v5_07.md
```

格式：

```text
scenario_id | old_action | new_action | keep/suspicious/drop | reason
```

### D Day 1 PM

审 F2 10 行 repair：

- 新答案是否 exactly 2 codes；
- pair 是否为合法 pair；
- 有没有 action 和 top_issue 明显矛盾；
- 标出可疑行。

### D Day 2 PM

对 A 最终 Sub2 候选做 30 分钟 changed-row sanity sweep。

禁止：

- 不要全量重跑 API；
- 不要自己生成大提交；
- 不要把人工直觉改动直接混进 Sub1；
- 只交 `keep/suspicious/drop` 列表。

## Sub1 内容

如果 B patch 和 smoke 通过，Sub1 是最大化 rerun：

```text
patched rerun output
+ post-output guard
+ 2 audited LoRA overrides
+ C's F2 pair calibration patches if delivered by Day 1 noon
```

2 个 audited LoRA overrides 必须重放：

```text
6060dca7: C23 -> C22
1784b64c: C21 -> C20
```

这两个被视为 ground-truth-equivalent，不随 rerun 改动。

## Sub2 decision table

```text
Sub1 >= 0.180:
  Sub2 = Sub1 + LLM F1 boundary expansion on highest-risk 80 rows

0.150 <= Sub1 < 0.180:
  Sub2 = Sub1 + C's F2 patches if not already in Sub1 + V5 calibrated KNN60

0.130 <= Sub1 < 0.150:
  Sub2 = Sub1 + targeted manual fixes only on flagged regression rows

Sub1 < 0.130:
  Sub2 = submission_600_v5_11_baseline_plus_f2_pair_fix.csv
  abandon rerun direction
```

## V5 fallback

如果 Day 1 noon 前发生任一情况：

- B patch 未完成；
- Phase1 smoke gate 失败；
- cloud API/token 不通；
- 全量 rerun 出现大量空数据或 timeout；

则不赌 rerun：

```text
Sub1 = submissions/submission_600_v5_07_adj_only.csv
Sub2 = submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv
```

如果还想带 KNN60：

```text
Sub2 = submissions/submission_600_v5_12_adj_only_plus_knn60_plus_f2_pair_fix.csv
```

但只有 B 明确建议时才带。
