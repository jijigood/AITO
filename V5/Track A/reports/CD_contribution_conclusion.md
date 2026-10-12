# C/D 工作贡献、测试结论与后续提分策略

生成时间：2026-05-17

本文汇总 C、D 两部分工作的真实贡献、已验证测试结果、当前结论，以及它们对后续拿更高分的实际价值。结论基于本地源码、提交 CSV、Phase2 test 结构、真实 diagnostics JSONL，以及对 C 原版、C v2、D spot-check 的复核。

## 1. 审计范围

已检查的主要材料：

- D 工作：`/home/huangxiaolin/AITO/V5/D_spotcheck_v5_07.md`
- C 原版：`/home/huangxiaolin/AITO/V5/V5_C`
- C v2：`scripts/C_f2_pair_calibration_v2.py`
- C safe 复核版：`scripts/C_f2_pair_calibration_safe.py`
- 关键提交文件：
  - `submissions/submission_600_v5_00_base_01302.csv`
  - `submissions/submission_600_v5_07_adj_only.csv`
  - `submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv`
  - `submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv`
  - `submissions/submission_600_C_f2_calibrated_v2.csv`
  - `submissions/submission_600_C_f2_calibrated_safe.csv`
- 真实 diagnostics：
  - `results/phase2/sub2_lora_boundary/result_diag_lora_boundary_only.diagnostics.jsonl`

## 2. Phase2 家族结构

Phase2 Track A 共 500 行，按结构拆分如下：

| Family | Rows | 正常答案形态 |
|---|---:|---|
| F1 | 241 | 1 code |
| F1_5BS | 28 | 1 code |
| F2 | 145 | 2 codes，合法 pair |
| F2_5BS | 29 | multi-code，当前多为 4 codes |
| F4 | 29 | 1 code，`decrease_interfreq` |
| F4M | 28 | 2 codes，`decrease_interfreq` pair |
| Total | 500 | - |

说明：真实 diagnostics 中 5-BS multi 有时会标成 `F3`，本文按 test 结构称为 `F2_5BS`。

F4 + F4M 共 57 行，baseline 和 v5_11 的 `decrease_interfreq` 计数都是 85，正好等于 `29 * 1 + 28 * 2 = 85`。这说明 F4/F4M 已经基本是确定性高置信部分，不是主要提分缺口。

## 3. D 部分的贡献与结论

D 的最大价值是：用真实 diff 和真实 diagnostics 审计已有提交，帮助团队识别哪些改动是高风险、哪些改动可以保留。

### 3.1 D 对 v5_07 F1/F1_5BS 改动的审计

D 审计的核心对象是 `v5_07` 相对 baseline 的 60 个 F1/F1_5BS single-answer 改动。

复核结果：

- `v5_07` 确实修改了 60 个 F1/F1_5BS single-answer 行。
- 转换方向主要是：
  - `adjust_azimuth -> add_neighbor`: 36 行
  - `adjust_azimuth -> decrease_a3`: 24 行
- D 的结论是：
  - drop: 44
  - suspicious: 16
  - keep: 0

进一步复核 diagnostics 后，可以确认：

- 这 60 行的原始问题集中在 single-answer F1/F1_5BS 上。
- Phase1 的 F1 single 真值分布里没有 `adjust_azimuth`，因此 Phase2 这 60 个 single-answer `adjust_azimuth` 是强结构性可疑点。
- 但 `v5_07` 对这些行的替换并没有足够证据支撑：
  - 36 个 `add_neighbor` 替换中，真实 diagnostics 候选支持很弱。
  - 24 个 `decrease_a3` 替换里，只有部分存在较弱 A3 支持，不能整体视为可靠修复。

因此 D 对 v5_07 的实际贡献是：证明 `v5_07` 不是稳健主线，不能因为它修了 60 个可疑 `adjust_azimuth` 就直接提交。

重要措辞：D 并不能离线证明这 60 个隐藏真值全部错误；但它证明了这些行是最大、最明确的结构性风险区，并证明 v5_07 的替换方案本身不可靠。

### 3.2 D 对 v5_10 F2 structural fix 的审计

D 同时审计了 `v5_10` 相对 `v5_07` 的 10 个 F2 修复。

复核结果：

- 这 10 个修复集中在 F2。
- 主要目标是把非法或不合理的 F2 答案修成合法 2-code pair。
- 修复后 F2 pair 合法性明显改善。
- 这部分比 v5_07 的 F1 大规模改动更可靠。

这直接支持后续生成 `v5_11`：保留 baseline，不带 v5_07 的 F1 激进改动，只叠加 F2 structural fix。

### 3.3 D 的最终贡献

D 的贡献不是直接生成一个更高分 CSV，而是建立了正确的提交决策边界：

- 不应把 `v5_07` 当主提交。
- 不应把 `v5_10` 当主提交，因为它继承了 v5_07 的 F1 风险。
- `v5_11 = baseline + F2 structural fix` 是更干净的 fallback。
- 后续真正冲高分，应围绕 60 个 F1/F1_5BS single-answer `adjust_azimuth` 做 patched rerun，而不是离线硬改。

## 4. C 部分的贡献与结论

C 的方向是研究 F2 pair calibration。这个方向有价值，但当前 C 原版和 C v2 都不能直接作为提交文件使用。

### 4.1 C 原版的问题

C 原版的核心问题：

1. 使用 Phase2 `data/Phase_2/test.json` 的占位字段做诊断。
   - Phase2 test 中很多真实 API 数据字段是占位文本。
   - 在这些占位字段上跑 `parse_scenario` / `run_all_skills` 会产生不可靠的 `top_issue=uncertain`。

2. 生成 F2 target pair 的方式不安全。
   - 原逻辑只是选择 action type 属于目标集合的前两个 code。
   - 这不能保证一个 code 是 `decrease_power`，另一个 code 是 `press_down_tilt`。
   - 也不能保证一个 code 是 `adjust_azimuth`，另一个 code 是 `increase_power`。

3. 生成的答案缺少真实 diagnostics candidate 支持。
   - 很多新答案不是 diagnostics 中已有打分候选。
   - 因此不能判断它们是否比当前答案更优。

结论：C 原版不能提交。

### 4.2 C v2 的进步

C v2 相比 C 原版有两个明显进步：

- base CSV 改为 `v5_11`，避开了 `v5_07/v5_10` 的 F1 激进风险。
- 排除了 6 个 D-keep 冲突行，避免直接覆盖 D 已经指出有反证的样本。

这两个方向是正确的，说明 C v2 在提交策略上比 C 原版更接近可用状态。

### 4.3 C v2 的关键错误

尽管 v2 有进步，但它仍然不能提交。

本地复核结果：

- `submission_600_C_f2_calibrated_v2.csv` 通过 600 行格式检查：
  - rows: 600
  - Track A answered: 500
  - Track B non-empty: 0
- 但是语义合法性审计失败：
  - F2 bad_count: 26
  - 典型非法答案：
    - 两个 code 都是 `decrease_power`
    - 两个 code 都是 `press_down_tilt`

根因是 v2 使用 `frozenset` 判断 pair。`frozenset` 会吞掉重复 action，导致两个同类动作也可能被误判成可接受结构。生成答案时仍然只是取前两个匹配 code，没有强制一类动作各取一个。

对 v2 的 70 个 flip 用真实 diagnostics 复核后，结果如下：

| 指标 | 结果 |
|---|---:|
| v2 相对 v5_11 flip 数 | 70 |
| 非 F2 flip | 0 |
| F2 语义非法 pair | 26 |
| 新答案不在 diagnostics candidates 中 | 55 |
| 新答案在 candidates 中但比分更低 | 15 / 15 |

这 70 个 flip 的真实 diagnostics top_issue 分布：

| real top_issue | rows |
|---|---:|
| overshoot_overlap | 49 |
| late_handover | 12 |
| uncertain | 7 |
| backhaul_server | 2 |

因此，v2 报告中“70 flips applied, all valid uncertain-azimuth F2 rows”这个说法不成立。它主要仍然依赖占位数据产生的 offline `uncertain` 判断，而不是真实 diagnostics。

结论：C v2 不能提交。

### 4.4 C safe 复核版的意义

为了验证 C 方向能否安全推进，额外实现了一个保守版脚本：

- `scripts/C_f2_pair_calibration_safe.py`

它的原则是：

- 使用真实 diagnostics，而不是 Phase2 占位字段。
- 新答案必须来自 diagnostics scored candidates。
- F2 pair 必须严格合法。
- 目标候选分数不能明显低于当前答案。
- 输出后执行 F2 合法性审计。

测试结果：

- 编译通过。
- 600 行格式检查通过。
- F2 bad_count: 0。
- 在默认参数下只找到 3 个可应用候选。
- Phase1 crossval smoke 没有触发可评估增益。

因此 safe 版的意义是提供正确的安全判据，而不是证明 C calibration 当前已经有大幅增分能力。

如果后续继续做 C，应形成 C v3：

- base 使用 `v5_11`。
- 保留 D exclude。
- 使用 safe 版的真实 diagnostics + scored candidate + 严格合法 pair 逻辑。
- 不做 70 行这种大规模无候选分数支持的 flip。

## 5. C 与 D 共同建立的判断

### 5.1 F2 不是主要瓶颈

F2 共 145 行，占 Track A 的 29%。它确实重要，但目前不是最大缺口。

当前可靠结论：

- F2 structural fix 值得保留。
- `v5_11` 中 F2 已全部是 2-code 合法 pair。
- 大规模 F2 calibration 没有可靠证据支持。
- C v2 的 70 行 flip 不可信，且存在 26 个非法 pair。

因此，F2 的后续策略应该是小规模、强证据、低风险，而不是全量基于 Phase1 prior 迁移。

### 5.2 最大缺口在 F1/F1_5BS 的 60 个 single-answer adjust_azimuth

Phase1 F1 single 真值分布：

| action | Phase1 F1 count |
|---|---:|
| check_server | 239 |
| decrease_a3 | 231 |
| lift_tilt | 254 |
| insufficient_data | 252 |
| pdcch_2sym | 240 |
| add_neighbor | 240 |
| adjust_azimuth | 0 |

Phase2 当前 `v5_11` 中：

- F1 的 `adjust_azimuth`: 54
- F1_5BS 的 `adjust_azimuth`: 6
- 合计：60

这说明 60 个 single-answer `adjust_azimuth` 是最集中的结构性异常。它们很可能来自模型/候选生成逻辑允许 F1 single 输出 `adjust_azimuth`，而 Phase1 规律并不支持这一点。

但注意：不能简单把这 60 行离线硬改成某一种动作。D 已经证明 v5_07 的硬改方案风险很高。

正确方向是 patched rerun：

- 禁止或强降权 F1/F1_5BS single 的 `adjust_azimuth`。
- 让 LLM/API/候选生成重新在合法 F1 action 中选择。
- 对 smoke 结果做 F1 top-1 准确率和分布审计。

### 5.3 F4/F4M 已接近上限

F4/F4M 共 57 行。

当前 `decrease_interfreq` 计数已经是 85，正好匹配：

- F4: 29 行，每行 1 个 `decrease_interfreq`
- F4M: 28 行，每行 2 个 `decrease_interfreq`

这部分不是主要冲分空间。

## 6. 对分数提升的贡献评估

假设 public score = Track A mean IOU / 2，baseline public 约为 0.1302，则 Track A mean IOU 约为 0.2604。

### 6.1 v5_11 的贡献

`v5_11` 相对 baseline 只改了 10 个 F2 行，目标是修复 F2 structural violations。

预期贡献：

- 这是可靠的小幅提升。
- public gain 大约在 0.003 左右量级。
- 更重要的是，它提供了一个干净 fallback。

因此 `v5_11` 的定位是：保底提交，而不是冲 0.20 的主力。

### 6.2 D 的贡献

D 的贡献主要是防止掉分和定位主缺口：

- 排除 `v5_07/v5_10` 作为主提交的风险。
- 保留 `v5_11` 的 clean baseline 策略。
- 确认 60 个 F1/F1_5BS `adjust_azimuth` 是最大可恢复区域。

它对分数的贡献不是直接加分，而是避免把提交带到错误方向。

### 6.3 C 的贡献

C 的当前贡献主要是研究性和负面验证：

- 证明 F2 pair calibration 需要严格合法性审计。
- 证明不能使用 Phase2 占位字段做 top_issue 判断。
- 证明大规模 `uncertain -> tilt` 迁移不可靠。
- 提供了后续 C v3 的安全设计方向。

C 当前没有形成可提交的高置信增分 CSV。

### 6.4 patched rerun 的潜在贡献

如果 patched rerun 能正确修复 F1/F1_5BS 的 60 个 single-answer `adjust_azimuth`，理论 public gain 上限约为：

- 修对 60 行：`60 / 500 / 2 = +0.060`
- 修对 45 行：`45 / 500 / 2 = +0.045`
- 修对 39 行：`39 / 500 / 2 = +0.039`

因此：

| 情况 | 估计 public score |
|---|---:|
| baseline | ~0.130 |
| v5_11 fallback | ~0.133-0.134 |
| patched rerun 修对 65% 的 60 行 | ~0.172-0.175 |
| patched rerun 修对 75% 的 60 行 | ~0.178-0.180 |
| patched rerun 接近修满 60 行并带少量其他收益 | ~0.190-0.200 |

0.20 目标并非完全不可能，但必须高度依赖 patched rerun 对 F1/F1_5BS 的真实修复效果。单靠 C/D 的离线 CSV patch 不足以从 0.134 推到 0.20。

## 7. 当前提交策略建议

### 7.1 推荐提交优先级

1. 若 patched rerun 通过 smoke gate：
   - Sub1 提交 patched rerun CSV。
   - Sub2 提交 `v5_11` 作为 fallback 或对照。

2. 若 patched rerun 不稳定或来不及：
   - Sub1 提交 `submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv`。
   - 不提交 C 原版、C v2、v5_07、v5_10 作为主线。

### 7.2 明确不推荐提交

- `submission_600_C_f2_calibrated_v2.csv`
  - 有 26 个非法 F2 pair。
  - 70 个 flip 不是真实 diagnostics 支持的 uncertain rows。
  - 55 个新答案不在 diagnostics candidates 中。

- C 原版输出
  - 使用 Phase2 占位数据。
  - 会生成非法 pair。

- `v5_07`
  - 60 个 F1/F1_5BS 激进替换没有可靠证据。

- `v5_10`
  - 虽然包含 F2 fix，但继承了 v5_07 的 F1 风险。

### 7.3 smoke gate 建议

patched rerun 必须至少通过以下检查：

- 600 行格式检查通过。
- Track A 恰好 500 个非空答案。
- Track B 全空。
- F2 全部合法 pair。
- F1/F1_5BS single 中 `adjust_azimuth` 显著减少或归零。
- F1 action 分布接近 Phase1 合理分布。
- 对抽样 rows 做 diagnostics 复核，不能出现 v5_07 那种大面积无证据替换。

## 8. 最终结论

D 是当前最有效的工作部分。它的贡献是审计、止损、定位主缺口，并帮助确定 `v5_11` 是安全 fallback。

C 的方向有研究价值，但当前原版和 v2 都不能提交。C 的最大贡献是暴露了 F2 calibration 的关键风险，并为后续 C v3 提供了正确约束：真实 diagnostics、scored candidates、严格合法 pair、小规模强证据。

整体来看，当前冲高分的核心路径只有一条：

1. 用 `v5_11` 保底。
2. 对 F1/F1_5BS single-answer `adjust_azimuth` 做 patched rerun。
3. 用 D 的审计方法检查 rerun 是否真的修复，而不是产生新的无证据替换。
4. C 只作为小规模 F2 安全补丁研究，不作为主力冲分方案。

一句话总结：

`v5_11` 负责保底，D 负责防错和定位，C 负责提供 F2 风险教训；真正决定能否接近 0.20 的，是 patched rerun 能否正确修复那 60 个 F1/F1_5BS single-answer `adjust_azimuth`。
