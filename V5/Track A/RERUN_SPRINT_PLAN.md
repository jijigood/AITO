# Track A 0.20 Rerun Sprint Plan

生成时间：2026-05-15

## 结论

如果目标是 public score 0.20，主线必须从 V5-only postprocess 切到 patched rerun。

V5-only 的优势是稳、可归因；缺点是改动面太小，预计很难从 0.1302 推到 0.20。patched rerun 风险更高，但上限更高。

当前策略：

```text
主攻：patched rerun
安全网：V5 v5_07 / v5_10 / v5_11 / v5_12
```

## 代码位置说明

V5 现在已经包含完整 rerun 主流水线。B/C/D 不需要再切到其他版本目录。

```text
/home/huangxiaolin/AITO/V5/Track A
```

关键 patch 目标：

```text
phase2_run.py
skills/candidate_generator.py
skills/feature_engine.py
skills/scenario_router.py
skills/api_data_gatherer.py
rag_retriever_v2.py
data/Phase_1/train.json
data/Phase_2/test.json
```

B 直接改 V5 里的这些文件，A 也从 V5 启动 patched rerun。

## Sub1 推荐内容

```text
patched phase2_run.py rerun output
+ candidate_generator guards
+ phase2_run prompt/cardinality constraints
+ post-output guard
+ 2 audited LoRA overrides
+ C's F2 pair calibration patches if delivered before Day 1 noon
```

Sub1 不追求小而干净。因为 public score turnaround 是分钟级，Sub1 可以承担一次最大化 rerun 风险，Sub2 再修补或回退。

## 必须修的三层 guard

### 1. Candidate Filter

位置：

```text
/home/huangxiaolin/AITO/V5/Track A/skills/candidate_generator.py
```

目标：

- F1 / F1_5BS single-answer 不让 `adjust_azimuth` 成为 deterministic top1；
- F2 不产生 4-code answer set；
- F2 不产生含 `add_neighbor/decrease_a3` 的 answer set；
- F2 answer set 只保留两类合法 pair：

```text
{adjust_azimuth, increase_power}
{decrease_power, press_down_tilt}
```

### 2. LLM Prompt Constraint

位置：

```text
/home/huangxiaolin/AITO/V5/Track A/phase2_run.py
```

目标：

- F2 prompt 从 “2 or 4 codes” 改成 “exactly 2 codes”；
- 明确 F2 只能选合法 pair；
- F1_5BS prompt 明确按 F1 single-answer 处理；
- F4M prompt 明确 exactly 2 `decrease_interfreq` codes。

### 3. Post-Output Guard

位置：

```text
新增 module/script，或集成到 rerun submission audit 阶段
```

目标：

- 如果 LLM 仍输出非法 F2，强制 rewrite 为最高分合法 F2 pair；
- 如果 F1/F1_5BS 输出 `adjust_azimuth`，强制 rewrite 为安全候选；
- 如果 F4M 输出非 `decrease_interfreq` pair，强制 rewrite；
- 最后再重放 2 个 audited LoRA overrides。

## 固定重放的 2 个 overrides

这两个不要再交给 rerun 或 LLM 判断：

```text
6060dca7: C23 -> C22
1784b64c: C21 -> C20
```

它们来自已审计的 Phase2 closeout，按 ground-truth-equivalent 处理。

## Phase1 Smoke Gate

在 A 启动 cloud API rerun 前，B 必须跑 Phase1 train offline smoke。

门槛：

```text
F1 single-answer top1 accuracy >= 0.80
F2 legal-pair top1 accuracy >= 0.50
F4 top1 accuracy >= 0.95
invalid F2 outputs = 0
```

如果失败：

```text
abort rerun
Sub1 = submissions/submission_600_v5_07_adj_only.csv
Sub2 = submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv
```

## 3-Day Schedule

### Day 1 AM

A：

- 确认 cloud API/token 可用；
- review B patch；
- 跑 Phase1 smoke；
- 决定 go/no-go。

B：

- 修改 `candidate_generator.py`；
- 修改 `phase2_run.py` prompt/cardinality；
- 写 post-output guard；
- 写 smoke script；
- 交 smoke result。

C：

- 做 Phase1 F2 pair calibration；
- Day 1 noon 前交 ranked flip list / patch script。

D：

- spot-check `v5_07` 的 60 个 F1 flips；
- noon 前交 weak-evidence list。

### Day 1 PM

A：

- smoke 通过则 cloud rerun；
- 生成 rerun submission；
- 跑 post guard；
- 重放 2 overrides；
- 合并 C F2 calibration 如果已经交付；
- 提交 Sub1。

B：

- standby 修 patch；
- Sub1 score 回来后开始 attribution。

C：

- standby；
- 若 Day 1 noon 未交，转为 Sub2 patch。

D：

- audit F2 10 行 fix；
- 标出明显矛盾行。

### Day 2

A：

- 根据 Sub1 score 和 B attribution 选 Sub2；
- 提交 Sub2。

B：

- 提交 `reports/B_rerun_attribution.md`；
- 推荐 Sub2 内容。

C：

- 如果 F2 calibration 没进 Sub1，则交给 A 做 Sub2 候选。

D：

- 对最终 Sub2 changed rows 做 30 分钟 sanity sweep。

### Day 3

Buffer。只有 Sub2 失败或格式问题时才启用。

## Sub2 Decision Table

```text
Sub1 >= 0.180
  Sub2 = Sub1 + LLM F1 boundary expansion on highest-risk 80 rows

0.150 <= Sub1 < 0.180
  Sub2 = Sub1 + C's F2 patches if not already in Sub1 + V5 calibrated KNN60

0.130 <= Sub1 < 0.150
  Sub2 = Sub1 + targeted manual fixes only on flagged regression rows

Sub1 < 0.130
  Sub2 = submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv
  abandon rerun direction
```

## V5 Fallback Files

```text
submissions/submission_600_v5_07_adj_only.csv
submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv
submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv
submissions/submission_600_v5_12_adj_only_plus_knn60_plus_f2_pair_fix.csv
```

用途：

- `v5_07`：rerun 失败时的 Sub1；
- `v5_10`：V5 fallback 的 Sub2 默认；
- `v5_11`：patched rerun 低于 0.130 时的安全回退；
- `v5_12`：只有 B 建议带 KNN60 时使用。

## 不要做

- 不要把 V5 候选按顺序逐个提交；
- 不要在 smoke 不过时强行 cloud rerun；
- 不要让 C 继续主攻 `check_server`；
- 不要把 D 的人工直觉直接混入 Sub1；
- 不要忘记 600 行格式和 Track B 空行。
