# Track A Phase 2 四人分工说明

**竞赛**：Telco Troubleshooting Agentic Challenge — Track A  
**Phase 2 截止时间**：2026-05-18  
**参赛方向**：只参加 Track A；所有提交文件中的 `Track B` 列保持空白。

**English version**：`PHASE2_TEAM_DIVISION.md`

**AI 任务指引**：`docs/PHASE2_AI_TASK_GUIDE.md`

**Owner D 任务包**：`docs/owner_d/README.md`

这一版按 **agent 系统能力** 分工，而不是按测试/反馈分工。B/C/D 各自负责一个相对独立的子系统，A 负责集成、live run、微调和提交。


---

## 1. 角色总览

| Owner | Agent 子系统 | 核心产物 |
|---|---|---|
| **A** | Integration + Fine-tuning + Submission | 可运行主流程、SFT/LoRA、live probe、audit、submission |
| **B** | Skills + Tools | RF skills、branch rules、tool menus、evidence schema |
| **C** | Prompt + CoT + Agentic Loop | system prompt、step prompt、诊断 JSON、tool-use reasoning |
| **D** | Trajectory RAG + Memory | trajectory dataset、retrieval index、few-shot/tool-use examples |

最小集成接口：

```text
B -> A/C/D: branch_spec + evidence_schema + tool_menu
C -> A: prompt/CoT/loop package
D -> A/C: retrieve_trajectory_context() + trajectory examples
A -> all: 必要的 probe/full-run diagnostics
```

---

## 2. A：集成、微调与提交

### 使命

A 负责完整可运行系统。A 集成 B/C/D 的产物，跑 live server，构建 SFT 数据集，训练和评估 LoRA，审计 CSV，并管理 3 次提交。

### 主要文件

| 文件/路径 | 用途 |
|---|---|
| `phase2_run.py` | Phase 2 主运行器和 CLI |
| `skills/api_data_gatherer.py` | 云端 API 数据桥 |
| `phase2_audit.py` | 提交审计 |
| `phase2_select_probe.py` | probe ID 选择 |
| `build_sft_v3.py` | SFT 数据构建 |
| `curate_sft.py` | SFT 清洗、平衡、hard negatives |
| `eval_ckpt.py` | heldout checkpoint 评估 |
| 外部本地归档 `checkpoints/` | LoRA checkpoints，保留在共享目录外 |
| `results/phase2/current_submission/` | 当前候选 CSV、diagnostics、audit |

### 职责

- 维护可端到端运行的 pipeline。
- 把 B 的 skills/tools spec 集成到 feature/tool/candidate 逻辑。
- 把 C 的 prompt/loop 改动集成到 `agentic_overlay.py`。
- 把 D 的 trajectory retriever 集成到 prompt/RAG context。
- 构建最终 SFT JSONL，并训练/评估 LoRA checkpoint。
- LoRA 未过 heldout gate 前保持 `audit`。
- 跑 representative probe、high-risk probe、full run、audit 和 submission。

### LoRA Gate

LoRA 只有在 heldout eval 通过后才能超过 `audit`：

| Class | 当前 IOU | Gate |
|---|---:|---:|
| `lift_tilt` | 0.00 | >= 0.50 |
| `insufficient_data` | 0.00 | >= 0.50 |
| `add_neighbor` | 0.68 | >= 0.60 |
| `check_server` | 1.00 | >= 0.95 |
| `decrease_a3` | 0.94 | >= 0.90 |
| `pdcch_2sym` | 1.00 | >= 0.95 |

两个失败类必须同时通过。

---

## 3. B：Skills + Tools

### 使命

B 负责 agent 的电信诊断能力：应该看什么指标、应该调用什么工具、如何解释工具输出、证据如何映射到 root cause 和 action。

### 主要文件

| 文件 | 范围 |
|---|---|
| `skills/feature_engine.py` | RF/KPI 特征提取和 issue scores |
| `skills/candidate_generator.py` | root-cause/action 到 C-code 候选排序 |
| `skills/tool_caller.py` | tool evidence 和 feature augmentation |
| `skills/agentic_overlay.py` | `compute_risk_level()` 和 `BRANCH_TOOL_MENUS` 的 spec，由 A 实现 |

### 职责

- 定义每个 family/branch 的 RF skills。
- 定义 branch trigger、tool menu 和 evidence schema。
- 定义工具结果如何影响 issue score 和 action selection。
- 优化 F1/F1_5BS deterministic 行为，重点包括：
  - `F1_5BS` 全部 collapse 到 `press_down_tilt`
  - `coverage_hole` -> `increase_power` vs `lift_tilt`
  - `missing_neighbor` vs `decrease_a3`
  - `lift_tilt` vs `insufficient_data`
- 给出 mapper 语义：root cause/action/target/relation 如何匹配选项。

### Skill 开发优先级

目标不是堆很多松散规则，而是补强 Phase 2 最容易丢分的歧义边界。B 开发的 skill 必须能给 C 的 prompt 和 D 的 trajectory memory 提供清晰证据。

| 优先级 | Skill | 目标歧义 |
|---|---|---|
| P0 | `coverage_vs_insufficient_data_skill` | 区分 `lift_tilt`、`increase_power`、`insufficient_data` |
| P0 | `neighbor_handover_skill` | 区分 `add_neighbor` 和 `decrease_a3` |
| P0 | `f1_5bs_overlap_skill` | 降低 `F1_5BS` 全部 collapse 到 `press_down_tilt` 的风险 |
| P0 | `multi_answer_pair_skill` | 提升 F2/F3 多答案配对，避免只选对一半 |
| P1 | `inter_freq_load_skill_v2` | 强化 F4/F4M 的 load / inter-frequency threshold 判断 |
| P1 | `evidence_confidence_skill` | 判断证据是否足够强，决定是否 override 或保守保留 |
| P1 | `target_cell_resolution_skill` | 统一 PCI、Cell ID、option label、target role 的匹配 |

每个 skill 不应只输出 final score，还要输出结构化 evidence fields。这些字段就是 C 和 D 消费的接口。

### 交付物

| Artifact | 内容 |
|---|---|
| `branch_spec.md/json` | branch name、trigger condition、family/tag scope、risk level |
| `tool_menu_spec.md/json` | 每个 branch 的 allowed tools、required args、expected result fields |
| `evidence_schema.md/json` | C/D 可以在 prompt/RAG 中使用的证据字段 |
| `skill_priority_spec.md/json` | P0/P1 skill 列表、输入字段、输出证据字段、预期 score 影响 |
| code patches | 对 `feature_engine.py`、`candidate_generator.py`、`tool_caller.py` 的聚焦修改 |

B 不负责 prompt wording，也不负责 RAG 检索。B 提供领域诊断契约，C/D 消费这个契约。

---

## 4. C：Prompt + CoT + Agentic Loop

### 使命

C 负责 agent 的语言侧智能：如何提示 LLM、如何让 LLM 比较证据、何时请求工具、如何输出稳定 diagnosis JSON。

### 主要文件

| 文件 | 范围 |
|---|---|
| `skills/agentic_overlay.py` | `_STEP1_SYSTEM`、`_STEP2_SYSTEM`、message builders、diagnosis JSON expectations |
| prompt artifacts | prompt v2/v3、branch prompt snippets、CoT templates |

### 职责

- 写 telecom expert system prompt。
- 写 branch-specific Step 1 / Step 2 prompts。
- 定义展示给模型的 CoT/trajectory 格式。
- 确保 LLM 输出 diagnosis JSON，而不是直接猜 C-code。
- 必要时加入 `disagreement_reason` 和 evidence comparison 字段。
- 控制 JSON 稳定性，避免幻觉工具调用。
- 决定 D 的 trajectory snippets 如何注入 prompt。

### 交付物

| Artifact | 内容 |
|---|---|
| `prompt_v2.md` | system prompt、Step 1 prompt、Step 2 prompt |
| `diagnosis_schema.json` | LLM 输出的精确 keys 和 allowed values |
| `cot_template.md` | evidence comparison 和 diagnosis reasoning 模板 |
| prompt patch | 可直接更新 `agentic_overlay.py` message builders 的补丁 |

C 不定义 RF branch 语义，那属于 B。C 不构建 trajectory index，那属于 D。

---

## 5. D：Trajectory RAG + Memory

### 使命

D 负责 agent 的 memory 层。D 把 Phase 1 训练样本和 Phase 2 probe traces 转成可复用的诊断轨迹，供 C 注入 prompt、供 A 集成。

### 主要文件和产物

| 文件/产物 | 范围 |
|---|---|
| `rag_retriever_v2.py` | 检索逻辑 |
| `artifacts/owner_d/data/trajectory_examples.jsonl` | 标准化 trajectory records |
| `artifacts/owner_d/index/trajectory_index.faiss` | 如使用 FAISS，则为向量索引 |
| `artifacts/owner_d/docs/trajectory_schema.md/json` | trajectory record 的稳定字段 |
| `artifacts/owner_d/docs/trajectory_prompt_snippets.md` | 给 C prompt 使用的紧凑 few-shot examples |

### Trajectory Record

每条记录描述诊断行为，不让模型复制训练样本的 C-code：

```python
{
    "scenario_id": str,
    "family": str,
    "tag": str,
    "branch": str,
    "serving_sinr": float,
    "serving_rsrp": float,
    "best_neighbor_advantage": float,
    "neighbor_in_config": bool,
    "mainlobe_outside": bool | None,
    "mr_count": int,
    "tool_sequence": list[str],
    "key_evidence": list[str],
    "root_cause": str,
    "action": str,
    "target_role": str | None,
    "confidence": "high|medium|low"
}
```

### 职责

- 从 Phase 1 train cases 抽取 trajectory records。
- 基于 branch + RF/evidence features 建检索。
- 提供 `retrieve_trajectory_context(sf, branch, k=3) -> str`。
- 给 C 提供紧凑 tool-use few-shot examples。
- 确保检索样例不会诱导模型复制训练 C-code。
- 保持检索确定性和低延迟。

### 交付物

| Artifact | 内容 |
|---|---|
| `artifacts/owner_d/docs/trajectory_schema.md/json` | record fields 和 allowed values |
| `artifacts/owner_d/data/trajectory_examples.jsonl` | Phase 1 trajectory memory |
| `artifacts/owner_d/index/trajectory_index.faiss` 或等价索引 | retrieval index |
| `retrieve_trajectory_context()` | 可调用的文本注入函数 |
| `artifacts/owner_d/docs/trajectory_prompt_snippets.md` | 按 branch 分组的 few-shot tool-use examples |

D 不负责 prompt wording，那属于 C。D 不负责模型微调，那属于 A。

---

## 6. 时间计划

| 日期 | A | B | C | D |
|---|---|---|---|---|
| 5/7-5/8 | 保持当前候选可审计；准备集成点 | 起草 branch/tool/evidence specs | 起草 prompt/CoT schema | 起草 trajectory schema |
| 5/9-5/10 | 集成 B/C/D 最小版本；跑 probes | 交付 F1_5BS 和 high-risk branch specs | 交付 prompt v2 和 diagnosis schema | 交付 trajectory examples + retriever prototype |
| 5/11-5/12 | full run；如果准备好则构建/训练/eval SFT | 交付高影响 skills/tools 代码补丁 | 根据 probe traces 优化 prompt | 优化 retrieval quality 和 snippets |
| 5/13 | Submission 1 | 复查 leaderboard 相关症状 | 复查 prompt/RAG 行为 | 扩展失败 branch 的 memory |
| 5/13-5/15 | 定向 follow-up probes 和训练 | patch 最高影响 skills/tools | patch prompt/loop failures | patch retrieval/few-shot coverage |
| 5/16-5/17 | freeze，final full audit | 只做低风险修补 | 只做低风险 prompt 修补 | 只做低风险 retrieval 修补 |
| 5/18 | Final submission | - | - | - |

---

## 7. 三次提交预算

| 提交 | 目标日期 | 目的 |
|---|---|---|
| Submission 1 | 2026-05-13 | 稳定候选或干净的小幅改进，获取 leaderboard 信号 |
| Submission 2 | 2026-05-15 | 基于反馈的定向修复 |
| Submission 3 | 2026-05-18 | 保守最终候选 |

不要把提交机会花在未审计的 model-only experiment 上。
