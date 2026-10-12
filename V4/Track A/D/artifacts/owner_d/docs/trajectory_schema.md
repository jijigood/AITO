# Trajectory Schema

本文件定义 Owner D 在 Phase 2 使用的稳定 trajectory record schema。后续 `trajectory_examples.jsonl`、检索索引、prompt-safe context 都应以这里的字段语义为准。

## 设计原则

- 核心 record 描述的是诊断轨迹，不是训练答案拷贝。
- 核心字段名固定，允许追加派生字段，但不能改名、删字段或改变基础语义。
- prompt 侧只能消费安全字段，不能暴露训练答案或选项编号。
- 同一条 record 可以来自 Phase 1 train case，也可以来自 Phase 2 probe trace，但核心字段含义必须一致。

## Core Record

```json
{
  "scenario_id": "str",
  "family": "str",
  "tag": "str",
  "branch": "str",
  "serving_sinr": "float",
  "serving_rsrp": "float",
  "best_neighbor_advantage": "float",
  "neighbor_in_config": "bool",
  "mainlobe_outside": "bool|null",
  "mr_count": "int",
  "tool_sequence": ["str"],
  "key_evidence": ["str"],
  "root_cause": "str",
  "action": "str",
  "target_role": "str|null",
  "confidence": "high|medium|low"
}
```

## Field Semantics

| Field | Type | Null / empty rule | Semantics |
|---|---|---|---|
| `scenario_id` | `str` | 不允许空字符串 | 原始场景唯一标识。必须稳定，可用于去重、tie-break 和回溯。 |
| `family` | `str` | 不允许为空 | 场景族，当前取值与 `PHASE2_TEAM_DIVISION` 对齐，通常是 `F1`/`F2`/`F3`/`F4`。 |
| `tag` | `str` | 不允许为空 | 原始题型标签，例如 `single-answer`、`multiple-answer`。保留原始任务视角，不替代 `family`。 |
| `branch` | `str` | 不允许为空；未知时写 `unknown` | B 侧定义的诊断分支名。用于 branch filter、few-shot 分组和适用边界控制。不是自由文本摘要。 |
| `serving_sinr` | `float` | 必须给数值；缺失时用约定默认值并在派生字段记录来源 | 代表检索时使用的 serving SINR 标量，通常取 worst-time 或 branch-relevant timestamp。 |
| `serving_rsrp` | `float` | 必须给数值；缺失时用约定默认值并在派生字段记录来源 | 代表检索时使用的 serving RSRP 标量。 |
| `best_neighbor_advantage` | `float` | 必须给数值；没有邻区时可为 `0.0` 或负值 | 最强邻区相对 serving 的优势值，通常是 `best_neighbor_rsrp - serving_rsrp`。 |
| `neighbor_in_config` | `bool` | 不允许为空 | 是否已有配置内邻区可支撑当前判断。用于区分 `add_neighbor` 与 `decrease_a3` 等近邻歧义。 |
| `mainlobe_outside` | `bool|null` | 可为 `null` | 是否存在主瓣落点偏离服务区的明确证据。`null` 表示未调用工具或证据不足，不等价于 `false`。 |
| `mr_count` | `int` | 必须为 `>=0` 整数 | 当前轨迹可用 MR 样本量或等价测量条目数，用于区分证据充分与 `insufficient_data`。 |
| `tool_sequence` | `list[str]` | 可为空列表，不可为 `null` | 诊断中实际使用或推荐的工具调用顺序，按执行顺序排列，例如 `["rf_summary", "judge_mainlobe"]`。 |
| `key_evidence` | `list[str]` | 至少 1 条，不能为空列表 | 用自然语言短句或规范化 evidence phrase 表达的关键证据，不包含答案编号。 |
| `root_cause` | `str` | 不允许为空 | 诊断根因的规范化描述，例如 `coverage_hole`、`missing_neighbor`、`backhaul_server`。 |
| `action` | `str` | 不允许为空 | 规范化动作类型，例如 `lift_tilt`、`add_neighbor`、`check_server`。 |
| `target_role` | `str|null` | 可为 `null` | 动作目标的角色，而不是训练场景中的具体 option 编号。建议取值包括 `primary`、`secondary`、`neighbor`、`source`、`target`。 |
| `confidence` | `enum` | 仅允许 `high`/`medium`/`low` | 该轨迹作为可复用诊断记忆时的置信等级，表示适用边界和证据强弱，不表示模型概率。 |

## 关键语义约束

### `branch`

- `branch` 是 D 侧检索和 few-shot 组织的硬过滤键之一。
- `branch` 的命名和触发语义由 B 提供，D 不重新定义其业务含义。
- 在 B 的正式 `branch_spec` 到位前，可临时使用兼容值，但必须保证后续可映射，不要混入自由文本。

### `key_evidence`

- `key_evidence` 记录“为什么这条轨迹可复用”。
- 应优先写可迁移证据，例如：
  - `serving_sinr < 0 dB while rsrp remains acceptable`
  - `best neighbor stronger but missing from config`
  - `mainlobe judged outside service area`
- 不应写：
  - `correct answer is C7`
  - `choose lift_tilt because training label says so`

### `target_role`

- `target_role` 只表达动作作用对象的角色，不绑定具体 cell ID、PCI 或选项号。
- 允许 `null`，适用于 `check_server`、`insufficient_data` 这类无明确目标小区的动作。
- 如需保留具体目标小区，应放在派生字段，不应污染核心 schema。

### `confidence`

- `high`: 证据充分、边界清晰，适合作为强 few-shot 或高优先级检索命中。
- `medium`: 主要证据成立，但存在未核实前提或分支边界重叠。
- `low`: 只适合弱提示或覆盖补齐，不应在 prompt 中作为强示例。

## Source Rules

### 来自 Phase 1 training cases 的字段

- 必填核心字段原则上都可以从 train case 或其派生特征中构造。
- `action` 常来自 ground-truth option label 的规范化解析。
- `target_role` 可来自 `skills/scenario_parser.py` 中的 `cell_roles` 映射。
- `tool_sequence`、`mainlobe_outside` 允许为空或 `null`，因为训练集未必包含真实工具执行痕迹。

### 来自 Phase 2 probe traces 的字段

- probe traces 可以补更真实的 `tool_sequence`、`mainlobe_outside`、`key_evidence`。
- probe traces 不得因为缺失训练答案而改变 `action` / `root_cause` 的定义方式。
- 若 probe 只提供部分证据，仍需输出完整 core record；缺失处按本 schema 的空值规则填写。

## Retrieval vs Prompt Display

### 检索主字段

以下字段默认用于索引、过滤、排序或 tie-break：

- `family`
- `tag`
- `branch`
- `serving_sinr`
- `serving_rsrp`
- `best_neighbor_advantage`
- `neighbor_in_config`
- `mainlobe_outside`
- `mr_count`
- `confidence`

### Prompt-safe 展示字段

以下字段允许被格式化后注入给 C：

- `family`
- `branch`
- `tool_sequence`
- `key_evidence`
- `root_cause`
- `action`
- `target_role`
- `confidence`

数值字段如 `serving_sinr`、`serving_rsrp` 可按需要摘要展示，但不要求原样暴露。

## Unsafe Fields

以下字段不是 trajectory core schema 的一部分；如果内部调试或缓存里存在，也不得进入 prompt-safe context：

- `gt_c_code`
- `answer`
- `answer_labels`
- `options`
- `C1` 到 `C22` 的任何直接编号提示
- 任何“正确答案是某个选项”的文本

当前 `rag_retriever_v2.py` 中的 `RAGHit.gt_c_code` 和 `prompt_snippet` 仅可视为内部兼容输出，不可当作对 C 的安全接口。

## Compatibility Mapping

当前仓库中的相关字段与 trajectory schema 的关系如下：

| Current field | Source | Schema status | Notes |
|---|---|---|---|
| `RAGHit.scenario_id` | `rag_retriever_v2.py` | 对应 `scenario_id` | 保持稳定。 |
| `RAGHit.gt_action_type` / `case["action_type"]` | `rag_retriever_v2.py` | 对应 `action` | 后续应统一命名为 `action`。 |
| `meta.target_role` | `build_sft_v3.py` | 对应 `target_role` | 当前来自 `cell_roles` 推断。 |
| `RAGHit.gt_c_code` / `case["answer"]` | `rag_retriever_v2.py` | 非 schema | 仅内部调试，禁止进入 prompt。 |
| `case["answer_labels"]` | `rag_retriever_v2.py` | 非 schema | 明确禁止进入 prompt-safe context。 |
| `RAGHit.prompt_snippet` | `rag_retriever_v2.py` | 非 schema | 当前格式含答案信息，后续任务需替换为安全文本。 |

## Extension Policy

- 可以追加派生字段，例如 `source_origin`、`record_version`、`target_cell_id`、`retrieval_text`。
- 派生字段不得覆盖 core field 名称。
- 新增字段如果会被 A/C 直接消费，必须补文档并说明是否 prompt-safe。

