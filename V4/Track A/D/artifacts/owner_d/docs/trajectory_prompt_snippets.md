# Trajectory Prompt Snippets

本文件提供按 branch 分组的 prompt-safe trajectory snippets，供 C 在 prompt 设计时直接复用。所有片段只描述可迁移诊断轨迹，不包含训练答案编号、`answer_labels`、`gt_c_code` 或“正确答案”提示。

## Source Policy

- `phase1_train`：来自训练样本的稳定轨迹记忆，覆盖面广，但工具调用顺序通常是“推荐/抽象顺序”，不一定是实际 probe 中执行过的顺序。
- `phase2_probe`：来自 Phase 2 diagnostics 的轨迹记忆，优先表达真实工具使用模式和证据比较方式；若当前仓库未落盘 probe diagnostics，则该来源暂时为空。
- 当同一 branch 同时存在 train/probe 两类记忆时，C 在高风险场景里应优先参考 probe 的工具使用顺序，再用 train 记忆补泛化证据模式。

## Coverage Status

- 当前仓库内可用 memory 以 `phase1_train` 为主。
- `lift_tilt_vs_insufficient_data`、`add_neighbor_vs_decrease_a3`、`inter_freq_load`、`pdcch_issue`、`backhaul_server`、`overshoot_overlap` 已有可引用片段。
- 当前工作区未包含实际 probe `.diagnostics.jsonl`，因此本文件中的 probe-specific 示例只能保留占位说明，待 A 的 probe diagnostics 落盘后补齐真实片段。

## lift_tilt_vs_insufficient_data

### Pattern A

- 场景类型：`F1` 单选覆盖/证据不足边界
- 来源：`phase1_train`，可被 `phase2_probe` 强化
- 关键证据：serving RSRP 偏弱，MR 样本偏少，主瓣判断和 MR 采样共同决定是天线问题还是证据不足
- 工具顺序：`judge_mainlobe`, `get_mr_data`
- 轨迹摘要：先确认是否存在明确的主瓣外证据，再看 MR 数量是否足够支撑覆盖根因；若证据不足，降低 coverage_hole 权重并保守转向 `insufficient_data`

### Pattern B

- 场景类型：`F1` 单选覆盖问题
- 来源：`phase1_train`
- 关键证据：serving RSRP 弱且邻区可见较多，MR 样本不稀疏
- 工具顺序：`judge_mainlobe`, `get_mr_data`
- 轨迹摘要：覆盖证据充分时保留 `coverage_hole -> lift_tilt`，目标角色通常是 serving cell 对应角色

## add_neighbor_vs_decrease_a3

### Pattern A

- 场景类型：`F1` 单选移动性边界
- 来源：`phase1_train`，优先等待 `phase2_probe` 补充真实工具使用次序
- 关键证据：最佳邻区更强、A3 margin 为负、邻区已在配置中
- 工具顺序：`get_signaling_events`, `calculate_overlap_ratio`
- 轨迹摘要：优先判断这是“切换门限过严”还是“邻区关系缺失”；若邻区已配置且 HO 条件未真正触发，轨迹更偏向 `late_handover -> decrease_a3`

### Pattern B

- 场景类型：`F1` 单选邻区关系缺失
- 来源：`phase1_train`
- 关键证据：最佳邻区更强且不在配置内
- 工具顺序：`get_signaling_events`, `calculate_overlap_ratio`
- 轨迹摘要：当强邻区不可被 serving cell 直接利用时，优先保留 `missing_neighbor -> add_neighbor`

## inter_freq_load

### Pattern A

- 场景类型：`F4` 13 基站负载/异频问题
- 来源：`phase1_train`
- 关键证据：DL PRB 利用率高，13-BS family 先验强
- 工具顺序：`get_kpi_data`
- 轨迹摘要：将高 PRB 和家族先验联合解释为异频分流需求，保留 `inter_freq_load -> decrease_interfreq`

## pdcch_issue

### Pattern A

- 场景类型：`F1` 单选容量控制问题
- 来源：`phase1_train`
- 关键证据：CCE 利用率高，Serving radio 指标不一定差
- 工具顺序：`get_kpi_data`
- 轨迹摘要：通过 KPI 细化确认控制信道拥塞，而不是把问题误归因到 backhaul 或 coverage

## backhaul_server

### Pattern A

- 场景类型：`F1` 单选服务器/回传问题
- 来源：`phase1_train`
- 关键证据：RSRP/SINR 健康，但吞吐仍显著下跌
- 工具顺序：`get_kpi_data`, `get_mr_data`
- 轨迹摘要：先确认 radio 健康，再用 KPI/MR 排除无线证据不足，保留 `backhaul_server -> check_server`

## overshoot_overlap

### Pattern A

- 场景类型：`F2` / `F3` 多动作重叠干扰类场景
- 来源：`phase1_train`
- 关键证据：serving RSRP 较强但 SINR 一般，强邻区同时存在
- 工具顺序：`calculate_overlap_ratio`, `judge_mainlobe`
- 轨迹摘要：关注覆盖重叠与干扰边界，优先抽象为可迁移的 `overshoot_overlap` 证据，而不是复用具体训练答案组合

## Probe Placeholders

### 待补 `phase2_probe` 片段

- `lift_tilt_vs_insufficient_data`
- `add_neighbor_vs_decrease_a3`
- `backhaul_server`

说明：当前仓库未包含实际 probe diagnostics JSONL，因此这里只保留待补列表，不伪造 probe-derived few-shot。
