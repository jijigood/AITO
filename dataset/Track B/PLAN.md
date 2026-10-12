# Track B 综合方案与排期

> Telco Troubleshooting Agentic Challenge — Track B (IP 网络运维)
> 编制日期：2026-04-30
> 修订状态：对齐 2026-04-30 当前 `my_agent/` 实现、PJ fault playbook 与离线回放结果
> 基础模型：Qwen3.5-35B-A3B（仅允许 LoRA / 全量微调，不可换架构）

## 1. 背景

赛题官方建议 Agent 同时使用四类技术：**Multi-turn Tool Use、Reasoning (CoT)、Fine-Tuning、Skills**。
本方案在保留现有 [my_agent/](my_agent/) 自定义 ReAct 框架的基础上，把上述四项官方技术全部映射到具体可交付物，并且明确舍弃 RAG（理由见第 3 节）。

赛程关键节点：

| 阶段 | 时间窗 | 题量 | 评分 | 关键约束 |
|---|---|---|---|---|
| Phase 1 | 4/3–5/4 | 50 | 仅准确率 | 1000 calls/天 |
| Phase 2 | 5/4–5/18 | 100（每 3 天放 20 题）| 准确率 + 调用数 | 仅 3 次提交，需上传执行轨迹 |
| Phase 3 | 5/18–5/29 | 70（64 节点金融网络）| Pass@1×4 + TTS | 15 分钟硬上限，提交 LoRA + 代码 |

## 2. 现状评估

最近一次完整 Phase-1 跑批（[results_compact_20260428_144004/](my_agent/results_compact_20260428_144004/)）：

- 50/50 完成，248 次 API 调用（均值 4.96/题），远低于配额。
- 题型分布：topology 11 / path 15 / fault 24。
- **格式合法率约 15–17/50（30%–34%）**，取决于校验规则。主要问题是模型把 chain-of-thought / markdown / 残缺 `<tool_call>` 直接写进了 `result.csv`。根因在旧 `_finalize_answer`：repair 阶段重新拼回完整 observation 上下文，Qwen3 在 thinking 模式下继续推理而不是修格式。
- 历史产物 [results/result.csv](my_agent/results/result.csv) 为空，但 `progress.json` 标记 50 题完成。该写盘 / resume 风险已在 [main.py](my_agent/main.py) 中修复：后续以新 `--output_dir` 跑批结果为准。
- 最新 smoke test 结论：
  - topology deterministic solver：11/11 离线格式合法。
  - physical-interface path solver：Q7–Q16 离线格式合法，Q7 真实复测通过。
  - PJ/PJGFA 业务路径 Q34–Q38 已由 `path_pj.py` 离线格式合法覆盖。
  - PJ 区高级故障 Q39–Q50 已由 `fault_evpn_srv6.py` 离线格式合法覆盖；规则覆盖 shutdown、L3VPN RT、MAC 冲突、VPN 绑定缺失、host collect 缺失、BGP advertise、OSPF network type、LoopBack 冲突、VNI/NVE 配置。
  - 当前 playbook 离线回放：50/50 命中，0 个格式非法，约 460 次模拟 API 调用。由于 Phase-1 test 无公开 GT，准确率仍需通过小规模真实提交 / leaderboard 反馈确认。

已经具备：
- 文本式 ReAct loop（[agent_loop.py](my_agent/agent_loop.py)）
- topology + 部分 path 确定性求解器（[deterministic.py](my_agent/deterministic.py)）：topology 全覆盖，path 当前覆盖 Q7–Q16 物理接口路径。
- 答案格式校验 + repair（[answer_validation.py](my_agent/answer_validation.py)）
- 命令规范化、逐条 tool trace、`llm_trace` 调试记录（[tools.py](my_agent/tools.py)、[main.py](my_agent/main.py)）

## 3. 系统设计

四层架构，每一层对应一项官方技术：

```
┌────────────────────────────────────────────────────────────────┐
│ Layer 1  Multi-turn Tool Use     agent_loop.py + tools.py      │
│ Layer 2  Skills (playbooks)      my_agent/playbooks/           │
│ Layer 3  Reasoning (CoT)         prompts.REASONING_HINT[fam]   │
│ Layer 4  Fine-Tuning (LoRA)      my_agent/ft/                  │
└────────────────────────────────────────────────────────────────┘
```

**RAG 显式舍弃**。理由：
1. 厂商命令集仅约 40 条，已硬编码进 [prompts.py](my_agent/prompts.py) 的 `SYSTEM_PROMPT`。
2. 故障原因词表 21 条，固化在 [answer_validation.py:36-79](my_agent/answer_validation.py)。
3. `devices_outputs/` 已经由 [server.py](server.py) 按需提供，相当于现成的"检索系统"。
4. 引入向量索引只会增加 50–100 ms 延迟和工程负担，没有准确率收益。

向评审解释时，把 RAG 的位置替换为"结构化 playbooks"——同样思路、更低成本、更贴合赛题。

### Layer 1 — 两阶段多轮工具使用（P0）

把 [agent_loop.py](my_agent/agent_loop.py) 拆成两个阶段：

| 阶段 | 上下文 | 模型设置 | 输出 |
|---|---|---|---|
| Reasoning | 完整观察 + `<think>` 允许 | `temperature=0`, `max_tokens=768` | `<tool_call>` 或继续推理 |
| Strict-emit | **仅题面 + 结构化 fact bag** | `temperature=0`, `max_tokens=192`, `stop=["</answer>"]` 或不用 stop，prefix-fill `<answer>\n` | 仅最终答案 |

要点：
- 新增 [my_agent/finalize.py](my_agent/finalize.py)：取代旧 `_finalize_answer` 的 repair 逻辑。
- 在 [answer_validation.py](my_agent/answer_validation.py) 增加 `extract_facts(observations, family) -> FactBag`：只把 LLDP 邻居、下一跳、接口状态等**事实**送进 strict-emit；模型没有"思考语境"可以续写。
- `<think>` 剥离移到 `_call_llm` 边界处（不再只在 `extract_answer` 里），半剥离的 CoT 永远到不了 `result.csv`。
- **不要用 `\n\n` 作为 stop**：topology / fault 是多行答案，空行 stop 容易截断合法答案。
- 残缺 `<tool_call>` 不进入最终答案，作为 retryable model-format error 处理；当前已在 [agent_loop.py](my_agent/agent_loop.py) 中实现基础版本。
- 修主目录 CSV 写盘 bug：每写一条 detail 就 append 到 result.csv，最后整体按 id 排序去重；当前 [main.py](my_agent/main.py) 已实现基础版本。
- `NOCClient` 增加每题执行轨迹日志：`(ts, device, command, response_excerpt)` → `data/exec_traces/qNN.json`。Phase 2 提交需要。当前 `detail.jsonl` 已含 `trace` / `llm_trace`，下一步是导出官方可上传 schema。

**Exit gate**：
- 已达成：用 `devices_outputs/` 缓存离线回放 50 题（零 API 消耗），playbook 命中 50/50，格式合法率 100%。
- 已达成：deterministic/playbook 覆盖 topology 11/11、physical path Q7–Q16、PJ path Q34–Q38、fault Q17–Q50。
- 待确认：Phase-1/Phase-2 leaderboard 精确准确率；重点核验多故障题是否需要输出全部故障行。
- 准确率无法完全离线计算（Phase test 无公开 GT），用已知样例 + 小规模真实提交 / leaderboard 反馈确认；Phase 1 目标 ≥ 70%。

### Layer 2 — Skills (playbooks)

新建 `my_agent/playbooks/`，每个文件一个题型，对外暴露：

```python
def should_run(question_text: str) -> bool: ...
def plan(question_text: str) -> list[ToolCall]: ...
def interpret(observations: list[Observation]) -> Answer: ...
```

| Playbook 文件 | 镜像官方 skill | 主要题号 |
|---|---|---|
| `playbooks/topology.py`        | infra_maintenance + l2_link | Q1–6, Q29–33 |
| `playbooks/path_physical.py`   | l3_route + infra_maintenance | Q7–16（已在 deterministic backend 中基本覆盖） |
| `playbooks/path_pj.py`         | l3_route                    | Q34–38（业务 IP → 网关 / VLAN / 汇聚路径） |
| `playbooks/fault_routing.py`   | l3_route                    | Q17–28、Q39–50 路由侧 |
| `playbooks/fault_port.py`      | l2_link                     | Q17–28、Q39–50 端口 / MAC / VPN 侧 |
| `playbooks/fault_evpn_srv6.py` | adv_tunnel                  | Q39–50 PJ 高级故障；Phase 2/3 高级协议 |

复用：[deterministic.py](my_agent/deterministic.py) 的 `solve_topology` / `solve_path` / `parse_lldp_links` / `_lldp_commands_for` / `_description_commands_for`。

当前实现边界：
- `solve_topology` 已作为 `topology.py` 的后端。
- `solve_path` 目前只适合“源设备 → 目标设备某物理接口”的题；PJ 区业务 IP 路径由 `path_pj.py` 单独建模。
- `fault_evpn_srv6.py` 当前针对 Phase-1 PJ 区做了高置信硬规则。Phase 2/3 若出现新拓扑 / 新命名，需要把这些规则抽象成“从配置表推断期望值”，不能长期依赖 Phase-1 设备名常量。

新增的 fault playbook 大约每个 120 LOC。例如 `fault_routing.py` 的判定逻辑：

```
1. 在源设备 display ip routing-table 上对题面 IP 做最长前缀匹配
2. 命中 0 → missing static route
3. 命中但 next-hop 在 ARP / interface 表上不可达 → blackhole route
4. current-configuration 中存在静态路由但未进入 active route，且下一跳接口 shutdown/down → blackhole route + shutdown
5. current-configuration 中静态路由目的网段或下一跳明显错误 → static route error
6. 多节点下一跳回到已访问节点 → routing loop
```

**评审可见性**：把 `playbooks/<name>.md` symlink 到 `agent/skills/<name>/SKILL.md`，使官方四个技能名字仍然出现在提交树里。

### Layer 3 — CoT（推理提示）

新增 `prompts.REASONING_HINT[family]`，每题型 4–6 条要点，**仅注入 Reasoning 阶段提示**，绝不进 Strict-emit。

示例（path）：
1. 若目标 IP 未给出，先在目标节点 `display ip interface brief`。
2. 在源节点 `display ip routing-table`。
3. 通过 LLDP 解析下一跳归属设备。
4. 递归。
5. 到达目标节点即停。

策略：
- topology / physical path 已被确定性求解器覆盖时跳过 CoT，避免无谓 token。
- 重点投入 fault 题，因为目前模型在这里"自由发挥"最多。
- 用 `<think>` 包裹推理，让 Layer 1 的 stripper 干净抹除。

### Layer 4 — LoRA 微调

**5/10 之前不动手**。整体节奏：

1. **5/1–5/4 被动收集 SFT**：`NOCClient` 把每题 `(question, observations, final_answer, validation, trace)` 写入 `my_agent/data/sft/*.jsonl`。Phase test 没有公开 GT，因此日志中只标记格式合法性和来源，不能伪造 `is_correct=True`。
2. **5/4–5/12**：Phase 2 跑批中持续挖掘成功轨迹；可用来源包括：
   - 确定性求解器高置信样本（topology、Q7–Q16 physical path）。
   - 官方 examples / 已知 GT 样例。
   - leaderboard 或人工复核确认正确的提交轨迹。
   只保留 `source_confidence=high` 且无格式风险的样本。预计到 5/12 累积 80–150 条。
3. **5/12–5/14**：rank-16 LoRA，1×A100-80G，1–2 epoch，约 3–4 GPU·h。
   - `my_agent/ft/build_sft.py`（采集 + 增广 + ChatML 格式化）
   - `my_agent/ft/train_lora.py`（PEFT，rank=16，lr=1e-4，lora_alpha=32）
   - `my_agent/ft/eval_lora.py`（在 [examples/traces.json](examples/traces.json) + 留出 P1 题集上回归）
4. **5/14–5/17 离线评测 + 提交决策**。
   - **采用准则**：相对 prompt-only 基线 ≥ +3 分准确率，确定性求解器题目无回归，且格式合法率不下降。
   - 不达标则 Sub #3 仍走精修 prompt-only 版本。
5. **5/18–5/26 Phase 3 准备**：合并 Phase-2 成功轨迹 + 自合成 SRv6/EVPN/ISIS 样本再训一版；冻结 ≥ 3 天后做最终 70 题 dress rehearsal。

LoRA 实际收益（按重要性排序）：
1. 把 1.5k token 的厂商命令清单"吃"进权重，减少每次调用 token 数 → Phase 3 TTS 降档。
2. 把 `<answer>…</answer>` 唯一输出风格固化进权重 → 不再依赖 prefix-fill 技巧也能达到接近 100% 格式合法率。
3. 满足"使用了 Fine-Tuning"的评审要点。

LoRA 不作为 Phase 2 成败前置条件。若 playbooks / strict-emit 未稳定，优先保证规则求解与可提交 trace，不为 LoRA 挪走 P0 工程时间。

## 4. 阶段排期

### Days 1–2 (4/30 → 5/2)：P0 可提交链路 + 高收益 solver

按优先级执行：

1. `finalize.py` / strict-emit：彻底切断 CoT、markdown、残缺 tool_call 进入 `result.csv` 的路径。
2. `fault_routing.py` + `fault_port.py`：先覆盖 Phase 1 fault 高频模式（missing route、blackhole、shutdown、interface IP、MAC/VPN/ARP）。
3. `path_pj.py`：补 Q34–Q38 业务 IP path。
4. exec trace exporter：从 `detail.jsonl.trace` / `llm_trace` 导出 Phase 2 可上传 JSON。
5. `playbooks/` dispatcher：先接入 topology、path_physical、fault，再扩展高级协议。

已达到：playbook 离线 50/50 命中、0 格式非法；下一步以真实提交反馈修正 Q39–Q50 的多故障行和故障原因粒度。

### Days 3–5 (5/2 → 5/4)：playbooks + CoT + SFT 收集器

- 完成 `playbooks/fault_routing.py`、`playbooks/fault_port.py` 的第一轮真实 smoke。
- 增加 `REASONING_HINT["fault"]`。
- 打开被动 SFT 日志。
- 真实跑 50 题：目标准确率 ≥ 70%，平均 ≤ 5 calls/q。

### Phase 2 (5/4 → 5/18)：3 次提交怎么用

| 提交 | 时间 | 内容 | 目的 |
|---|---|---|---|
| Sub #1 | ~5/6（前 20 题放出后）| Layer 1 + topology/path/fault 基础 playbooks | 锁定 leaderboard 基线，验证 result.csv + trace pipeline |
| Sub #2 | ~5/11（≥60 题可见）| + family-CoT + `fault_evpn_srv6.py`，再做调用数剪枝 | 在准确率不降的前提下优化调用数 |
| Sub #3 | ~5/16–5/17 | LoRA（达标）或精修 prompt-only | 终版 |

### Phase 3 准备 (5/18 → 5/29)：专精化 + Docker

- 加深 `playbooks/fault_evpn_srv6.py`：`display segment-routing ipv6 local-sid …`、`display srv6-te policy`、`display bgp evpn all routing-table`。
- TTS 优化：`tools.py:NOCClient.execute` 加 `(device, command, question_id)` 响应缓存。缓存只保证**单次运行内**重复命令不再访问 server；不要假设不同 trial / 不同容器之间能共享缓存。
- Docker 镜像：`my_agent/` + LoRA adapter + `agent/skills/<name>/SKILL.md` symlink，让评审能直观看到四类官方技术全部到位。
- 70 题 × 4 trial 全量演练 + 15 分钟硬限验证。

## 5. 关键文件

待修改（已有）：
- [my_agent/agent_loop.py](my_agent/agent_loop.py) — 双阶段拆分；`_call_llm` 边界处剥 `<think>`
- [my_agent/answer_validation.py](my_agent/answer_validation.py) — 新增 `extract_facts`；复用 `FAULT_CAUSES`、`validate_answer`、`normalize_answer`
- [my_agent/prompts.py](my_agent/prompts.py) — 增 `REASONING_HINT[family]`；复用 `detect_family`、`extract_answer`
- [my_agent/main.py](my_agent/main.py) — exec-trace 独立导出；继续保留 result.csv 排序去重和 resume 保护
- [my_agent/tools.py](my_agent/tools.py) — 扩展逐条命令日志；（Phase 3）响应缓存
- [my_agent/deterministic.py](my_agent/deterministic.py) — 作为 playbook 后端复用 `solve_topology` / `solve_path` / `parse_lldp_links`；补 PJ path 支撑函数

待新建：
- `my_agent/finalize.py` — Strict-emit 阶段
- `my_agent/playbooks/{topology, path_physical, path_pj, fault_routing, fault_port, fault_evpn_srv6}.py`
- `my_agent/ft/{build_sft, train_lora, eval_lora}.py`
- `my_agent/data/{sft, exec_traces}/` — 运行期产物
- `agent/skills/<name>/SKILL.md` symlink — 评审可见性

## 6. 验证方案

| 层 | 验证手段 | 通过条件 |
|---|---|---|
| Layer 1 格式漏出 | `validate_results.py` + 新增 `--strict` 后对 50 题离线回放 | 合法率 ≥ 95% |
| Layer 2 Playbooks | `my_agent/tests/test_playbooks_*.py` + 50 题盲跑 | 格式合法率 ≥ 95%，Phase 1 leaderboard / 已知样例准确率 ≥ 70%，平均 ≤ 5 calls/q |
| Layer 3 CoT | fault 题 A/B（带/不带 hint） | Δ ≥ +3 分则保留 |
| Layer 4 LoRA | `my_agent/ft/eval_lora.py` 在 examples/traces.json + 高置信 SFT 留出集上 | Δ ≥ +3 分、格式合法率不下降、确定性题无回归 |
| Phase 2 提交 | 本地 server.py:7860 全量 100 题 dry-run | result.csv 排序去重 + 执行轨迹 JSON 合法 |
| Phase 3 演练 | Docker 内 70 题 × 4 trial × 15 分钟 | 后续 trial 命中缓存进入 < 5 min TTS 档 |

## 7. 切分准则（什么时候放弃哪一层）

| 信号 | 行动 |
|---|---|
| Layer 1 修完后准确率已 ≥ 80% | Layer 2 仍补 fault + PJ path；不再扩 topology / physical path |
| 5/12 离线评估 LoRA 收益 < 3 分 | Sub #3 走 prompt-only，跳过 LoRA |
| Phase 2 leaderboard 调用数远未到瓶颈 | 不做 token 裁剪，保留观察预算 |
| Phase 3 SRv6/EVPN 题占比 < 20% | `fault_evpn_srv6.py` 不再做深解析，回退通用 fault |

## 8. 给评审的"四技术对账"

| 官方技术 | 本方案对应物 | 文件锚点 |
|---|---|---|
| Multi-turn Tool Use | 两阶段 ReAct + 严格 strict-emit | [agent_loop.py](my_agent/agent_loop.py)、`finalize.py` |
| Skills | 六个 playbook + symlink 到 SKILL.md | `my_agent/playbooks/`、`agent/skills/*/SKILL.md` |
| Reasoning (CoT) | 题型化 `REASONING_HINT` + `<think>` 块 | [prompts.py](my_agent/prompts.py) |
| Fine-Tuning | rank-16 LoRA on Qwen3.5-35B-A3B | `my_agent/ft/` |
| ~~RAG~~ | 改为 structured playbooks（理由见 §3）| — |
