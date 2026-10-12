# V3 团队分工

**竞赛**：电信故障排查 Agentic Challenge — Track A（无线）
**第一阶段截止日期**：2026-05-04（以 D-day = 2026-04-20 计，剩余约 2 周）
**基础模型**：Qwen3.5-35B-A3B（仅可使用 LoRA；不可更换基座模型）

### 当前基线

| 版本 | 总体 | F1 | F2 | F3 | F4 | 说明 |
|---|---|---|---|---|---|---|
| V2 提交版 | 0.3665 | 0.17 | 0.85 | 1.00 | 0.90 | 仅做微调，没有特征引擎 |
| V3 确定性版（当前） | 0.6750 | 0.5752 | 1.00 | 1.00 | 1.00 | 特征引擎 + 无模型 |
| **V3 目标** | **≥ 0.85** | **≥ 0.78** | **1.00** | **1.00** | **1.00** | 重训 LoRA + 工具证据 |

### 为什么要拆成四个方向并行

V3 流水线可以自然拆成四层、且耦合度较低：

1. 特征/候选层（本地、确定性）
2. 工具证据层（服务端、权威）
3. 检索层（对 2000 条训练样本做 RAG）
4. 模型层（Qwen LoRA 微调）

在接口冻结之后，这四层拥有各自独立的优化轴，共享状态很少，因此适合在剩下两周里由四个人并行推进，而不会互相踩踏。

---

## 1. 架构到负责人映射

```text
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  原始场景                                                          │
│        │                                                           │
│        ▼                                                           │
│  scenario_parser ──► ParsedScenario                                │
│        │                                                           │
│        ▼                                                           │
│  scenario_router ──► Family {F1|F2|F3|F4}                          │
│        │                                                           │
│        ▼                                                           │
│  feature_engine ──► ScenarioFeatures                 [Owner B]     │
│        │                                                           │
│        ▼                                                           │
│  candidate_generator ──► [(c_code, score, reason), ...][Owner B]  │
│        │                                                           │
│        ├─────────────┬──────────────┐                              │
│        │             │              │                              │
│      F2/F3/F4       F1 (~75%)      F1 (fallback)                   │
│        │             │              │                              │
│        ▼             ▼              ▼                              │
│   top_cand()    tool_caller    top_cand()           [Owner C]     │
│                      │                                             │
│                      ▼                                             │
│                 ToolEvidence                                       │
│                      │                                             │
│                      ▼                                             │
│                 ScenarioFeatures++                                 │
│                      │                                             │
│                      ▼                                             │
│                 rag_retriever_v2 ──► RAGHit[k=6]     [Owner D]     │
│                      │                                             │
│                      ▼                                             │
│                 _build_f1_prompt()                                 │
│                      │                                             │
│                      ▼                                             │
│                 Qwen LoRA (vLLM)                     [Owner A]     │
│                      │                                             │
│                      ▼                                             │
│                 answer_parser ──► c_code_list        [Owner B]     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 汇总表

| 负责人 | 层 | 文件 | 可量化目标 |
|---|---|---|---|
| **A（我）** | 微调 + SFT 数据整理 + 提交 | `build_sft_v3.py`、`curate_sft.py`、LoRA 配置、`main_v3.py` 集成 | 借助 LoRA 使 F1 IOU ≥ 0.78 |
| **B** | 特征 + 候选 + 解析 | `skills/feature_engine.py`、`skills/candidate_generator.py`、`skills/scenario_parser.py`、`skills/scenario_router.py`、`skills/answer_parser.py` | F1 确定性版 ≥ 0.70；F2/F3/F4 ≥ 0.98 |
| **C** | 服务端工具集成 | `skills/tool_caller.py` | `lift_tilt` ≥ 0.60，`insufficient_data` ≥ 0.60 |
| **D** | 推理期 RAG + 数据分析工具 | `rag_retriever_v2.py`、`skills/data_tools.py` | 在 `lift_tilt` / `insufficient_data` 探针上，RAG 命中率 ≥ 60% |

---

## 2. Owner A — 微调与提交（我）

### 2.1 职责说明

负责从“原始训练场景”一直到“最终结果提交”的完整链路。这包括**SFT 数据整理**（借助 Owner D 提供的数据分析工具）、数据集构建、LoRA 训练、checkpoint 选择、与 `main_v3.py` 的集成，以及最终上传到竞赛服务器。

**为什么整理归 Owner A，而不是 D**：所谓 curation，本质上是在决定哪些样本进入 `sft_v3.jsonl`。这属于建模决策，会影响类别平衡、hard negative 和验证集卫生，因此应由 A 负责。D 提供的是工具库（去重分数、异常分数、标签错误打分），A 调用这些工具并设定阈值。

### 2.2 开始前的先决条件（我需要其他人先交付什么）

| 来自 | 内容 | 截止时间 |
|---|---|---|
| Owner B | 冻结后的 `ScenarioFeatures` schema | D+2（2026-04-22） |
| Owner B | `feature_engine.py` 稳定到足以保证特征向量不再漂移 | D+2 |
| Owner C | 初版 `ToolEvidence` schema（第一轮允许是空值） | D+2 |
| Owner D | `retrieve(sf, k=6) → list[RAGHit]` 这个接口签名 | D+2 |
| Owner D | `data_tools.py`：去重、异常、标签错误打分函数 | D+3 |

### 2.3 交付物

| # | 工件 | 详细规范 | ETA |
|---|---|---|---|
| A1a | `curate_sft.py` | 消费 D 的 `data_tools.py` 分数，做样本纳入决策：去掉重复样本（cosine < 0.02）、去掉明显标签错误样本、降低模糊样本权重。输出 `curated_train_ids.txt` 与 `curation_report.md`，统计每种原因删掉多少。 | D+3 |
| A1b | `build_sft_v3.py` | 遍历 `curated_train_ids.txt`，对每个 F1 场景构建 `(system, user, assistant)`，调用 `_build_f1_prompt()`，写出 `sft_v3.jsonl`。按第 2.5 节做类别平衡上采样，并输出直方图。 | D+3 |
| A2 | LoRA 训练配置 | `llama-factory` YAML：r=32，alpha=64，dropout=0.05，target_modules={q,k,v,o,gate,up,down}，lr=1e-4 cosine，3% warmup，bs=32，epochs=3，max_len=4096，DeepSpeed zero-2。 | D+3 |
| A3 | 第一版 SFT checkpoint | 基于当前特征集（无工具证据）训练。目标：F1 ≥ 0.70。每 500 step 存一次 checkpoint，在验证集上挑最佳。 | D+4 |
| A4 | 加入工具证据后的重训 | 待 Owner C 交付 C4 后进行。预期 F1 从 0.70 提升到 0.75。 | D+6 |
| A5 | 加入完整 RAG + confusion pairs 后重训 | 待 Owner D 交付 D3 后进行。预期 F1 从 0.75 提升到 0.78。 | D+7 |
| A6 | 提交流水线 | `submit.py`：对评测服务器运行 `main_v3.py`，写出 `result_final.csv`，并上传到竞赛平台。需支持 dry-run。 | D+8 |
| A7 | 消融报告 | 生成 CSV：`{checkpoint × per-action IOU × wall-time}`，明确每个组件究竟提升了哪些指标。 | D+9 |

### 2.4 训练数据构建规范

```python
# build_sft_v3.py 伪代码
for scenario_id in train_ids:
    ps = parse(scenario_id)
    family = route(ps)
    if family != Family.F1:
        continue                           # 仅使用 F1
    sf = run_all_skills(ps)
    candidates = generate_single_candidates(ps, sf)[:6]
    rag_hits = rag.retrieve(sf, k=6)

    system = QWEN_V3_SYSTEM_PROMPT         # 冻结系统提示词
    user   = _build_f1_prompt(ps, sf, candidates, rag_hits)
    gt     = ground_truth[scenario_id]     # F1 只有一个 C-code
    rationale = _build_rationale(sf, gt)   # 例如 "SINR=6.2, PRB=82% → coverage_hole → lift_tilt/secondary"
    assistant = f"\\boxed{{{gt}}}\n# {rationale}"

    write_jsonl({system, user, assistant})
```

理由性 token 有助于学习；推理阶段如果需要追求延迟，可以去掉。

### 2.5 类别均衡采样

F1 的动作分布并不均衡（21/19/18/16/15/11）。重点上采样确定性流水线最难搞定的三类：

| 动作 | 原始占比 % | 采样后占比 % | 原因 |
|---|---|---|---|
| `lift_tilt` | 19 | **30** | 确定性版是 0%，必须重点强化 |
| `insufficient_data` | 21 | **25** | 确定性版也是 0% |
| `add_neighbor` | 15 | **18** | 确定性版只有 35%，适度上采样 |
| `pdcch_2sym` | 18 | 12 | 确定性版已是 100%，没必要浪费容量 |
| `check_server` | 16 | 10 | 确定性版已是 100% |
| `decrease_a3` | 11 | 5 | 确定性版已达 97% |

Hard-negative 设计：每个 `lift_tilt` 正样本都配 2 个 `insufficient_data` 负样本（反之亦然），逼模型学会这两类的边界。

### 2.6 评估协议

一个 checkpoint 只有同时通过以下门槛，才允许晋升：

1. 在 200 样本 held-out 集上，F1 IOU ≥ 0.75（与训练集无重叠）。
2. 按动作看：`lift_tilt` ≥ 0.60，`insufficient_data` ≥ 0.60，`add_neighbor` ≥ 0.65。
3. F2/F3/F4 不变（因为这些家族不走模型，应维持 1.0）。
4. F1 单场景平均 wall-time ≤ 90 秒。

工具命令：

```bash
python eval_v3.py --full --per-action --model http://localhost:8001/v1 --lora ckpt_xxx
```

### 2.7 非目标

- 不修改 `feature_engine.py`、`candidate_generator.py`、`tool_caller.py`、`rag_retriever_v2.py`。如果发现 bug，只提交带最小复现的 issue 给对应 Owner。
- 不动 F2/F3/F4 的确定性路径，它们已经达到 1.0。
- 不尝试更换基座模型，也不探索非 LoRA 适配器，避免范围失控。

### 2.8 我承担的风险

| 风险 | 缓解手段 |
|---|---|
| SFT 对 RAG 示例过拟合 | 分层 train/val 划分；按 val perplexity 早停 |
| 训练与推理 prompt 漂移 | `build_sft_v3.py` 与 `main_v3.py` 共用同一个 `_build_f1_prompt()` |
| 特征 schema 频繁变动导致数据集失效 | D+2 冻结；之后任何变化都必须阻断并重新评估 |
| LoRA 推理太慢 | D+3 先测 tokens/s；若每场景 >30 秒，则切到 vLLM + AWQ 量化 |

---

## 3. Owner B — 特征工程与确定性流水线

### 3.1 职责说明

负责主干链路：从原始场景一直到候选列表。这一层既决定了确定性基线，也决定了 Owner A 训练 LoRA 时看到的输入分布。只要这里做好，F2/F3/F4 基本就是白送的 100%，而 F1 也会被压缩成一个干净的重排序问题。

### 3.2 负责文件

| 文件 | 当前状态 | 关键函数 |
|---|---|---|
| `skills/feature_engine.py` | ~1200 行，最近做过 V3 的 4BS 修复 | `run_all_skills()`、`_get_max_dl_cce()`、各类 `skill_*` 函数 |
| `skills/candidate_generator.py` | ~600 行，已加入 V3 的 serving-cell boost | `generate_single_candidates()`、`generate_multi_candidates()`、`_find_best_option()`、`_upsert_candidate()` |
| `skills/scenario_parser.py` | 稳定 | `parse()`、`ParsedScenario`、`CodedOption` |
| `skills/scenario_router.py` | 稳定 | `route()`、`Family` 枚举 |
| `skills/answer_parser.py` | 稳定 | `extract_boxed()`、`compute_iou()` |

### 3.3 最高优先级问题

| 优先级 | 问题 | 当前 → 目标 |
|---|---|---|
| P0 | `add_neighbor` 的目标小区经常选错，65% 的案例把正确的 tertiary 选成了 primary | 0.35 → 0.75 |
| P0 | F2 在 SINR 分组修复后从 1.0 回落到 0.84，说明多答案路径被误伤 | 0.84 → 1.00 |
| P1 | 只靠本地数据无法把 `lift_tilt` 与 `insufficient_data` 区分开 | 0% → 40%（剩下部分交给工具） |
| P2 | 必须尽快冻结特征 schema，否则 Owner A 的数据集会一直失效 | D+2 冻结 |

### 3.4 交付物

| # | 工件 | 详细规范 | ETA |
|---|---|---|---|
| B1 | Schema freeze | 锁定 `ScenarioFeatures` dataclass。任何新字段都必须经过团队签字并触发 Owner A 重训。将规范写入 `SCHEMA.md`。 | D+2 |
| B2 | 修复 `add_neighbor` 的目标选择 | 在 `_find_best_option` 中交叉比对 `option.target_cell_id` 与 `mr_data.best_neighbor_cell`。若两者匹配，则施加 1.15× boost。 | D+3 |
| B3 | 修复 F2 回归 | 根因是 `is_single` 的保护条件漏掉了一些标签变体。改成直接解析答案数：`n_answers = len(gt_answers); is_single = (n_answers == 1)`。并补全所有 F2 标签变体的测试夹具。 | D+3 |
| B4 | 加入几何型 `lift_tilt` 启发式 | 在 `feature_engine.py` 中新增规则：若 serving RSRP ∈ (-105, -90)，且 500m 内没有任何站点与 serving 方位角的差值在 30° 以内，则令 `coverage_hole_tilt = 0.6`（与 `coverage_hole_power` 区分）。 | D+5 |
| B5 | 单元测试 | Pytest 覆盖 6 个 F1 动作类别，每类至少 3 个场景（至少 18 个测试）。每次提交都要跑。 | D+5 |
| B6 | 重构：抽取 `_apply_4bs_overhaul()` | 目前内联逻辑太难审查，提炼成带 docstring 的单独函数，不改变行为。 | D+6 |

### 3.5 成功判定

- F2/F3/F4 在全量训练集上均达到 ≥ 0.98
- F1 确定性基线达到 ≥ 0.70（当前是 0.5752）
- `eval_v3.py --full` 可以完整跑通，不抛异常
- B5 测试在主分支全部通过

### 3.6 测试预期

`skills/tests/test_feature_engine.py` 至少应包含：

```python
def test_f1_lift_tilt_tertiary_target():
    ps = load_fixture("f1_lift_tilt_tertiary_001")
    sf = run_all_skills(ps)
    candidates = generate_single_candidates(ps, sf)
    assert candidates[0][0] == ps.options["lift_tilt/tertiary"].c_code

def test_f2_regression_not_masked_by_is_single():
    ps = load_fixture("f2_multi_drops")
    sf = run_all_skills(ps)
    assert sf.score_inter_freq_load > 0.3  # not capped to 0.08
```

### 3.7 我承担的风险

| 风险 | 缓解手段 |
|---|---|
| 为了解决 `add_neighbor` 引入新的回归 | 修改逻辑前先补上覆盖所有通过动作路径的回归测试 |
| schema 冻结阻碍了明显的改进 | 冻结只是软规则；若确有必要，可经团队共识 + A 的重训预算解锁 |
| 几何启发式对训练集过拟合 | 额外留出 100 个场景作为 B 的私有验证集，开发中不接触 |

---

## 4. Owner C — 工具集成（19 个服务端工具）

### 4.1 职责说明

负责把本地流水线与竞赛服务端的 19 个工具对接起来。确定性流水线直接读取 CSV 字段，而 `tool_caller.py` 负责补充 CSV 中没有、模型自己也推不出的**几何与因果证据**。

### 4.2 这 19 个工具

| Tool | 作用 | 何时调用 |
|---|---|---|
| `judge_mainlobe` | 判断 UE 是否位于某个小区的主瓣内 | 对所有 SINR>5 且 PRB 低的场景都调用，用于区分 `lift_tilt` 与 `check_server` |
| `calculate_tilt_angle` | 返回天线当前俯仰角 | 当 `judge_mainlobe=False` 时调用，用来决定是否需要调整 tilt |
| `calculate_overlap_ratio` | 计算小区间覆盖重叠比例 | 用于 SINR<0 的切换类场景 |
| `get_mr_data` | 返回原始 MR 样本 | 当本地 MR 很稀疏时调用，用于识别 `insufficient_data` |
| `get_kpi_data` | 返回某小区的历史 KPI | 判断问题是长期性的还是瞬时性的 |
| `get_cell_config` | 返回邻区配置和 A3 参数 | 用于校验 `add_neighbor` / `decrease_a3` |
| `get_signaling_trace` | 返回原始信令事件 | 只有在其他工具解释不清时才调用 |
| `get_traffic_pattern` | 返回流量时间曲线 | 区分拥塞与故障 |
| ...（其余 11 个，见 `server.py`） | ... | 机会式调用 |

### 4.3 负责文件

`skills/tool_caller.py`。目前还是骨架模块，只有 `ToolEvidence` dataclass 与 `augment_features()` 的 TODO，必须把它补成可生产使用的模块。

### 4.4 为什么这项工作杠杆最高

确定性流水线在 F1 的两个类别上失败得最彻底：`lift_tilt` 和 `insufficient_data`。根本原因相同：**本地特征无法区分它们**。它们都会表现成高 SINR + 中等 PRB + 中等吞吐。只有工具才能给出真正有区分度的信号：

- `lift_tilt`：问题小区的 `judge_mainlobe=False`
- `insufficient_data`：`get_mr_data` 返回的样本数 < N

没有这些工具，再好的特征工程和微调都无从补救。

### 4.5 交付物

| # | 工件 | 详细规范 | ETA |
|---|---|---|---|
| C1 | 冻结 `ToolEvidence` schema | Dataclass 至少包含 `{mainlobe_flags, tilt_angles, overlap_ratio, kpi_data_raw, signaling_events, mr_sample_counts}`，每个字段都要写明定义。 | D+2 |
| C2 | 面向 SINR<0 组的 `call_targeted()` | 当 `top_issue ∈ {late_handover, missing_neighbor}` 时，调用 `get_cell_config`，并对 serving + top-2 邻区调用 `calculate_overlap_ratio`。 | D+3 |
| C3 | 面向 SINR>5+低 PRB 组的 `call_targeted()` | 当 `top_issue ∈ {coverage_hole, backhaul_server}` 时，对每个候选目标调用 `judge_mainlobe`，并用 `get_mr_data` 检测 `insufficient_data`。 | D+4 |
| C4 | `augment_features()` | 从 `ToolEvidence` 读取信息并更新 `ScenarioFeatures`：<br>• `mainlobe=False` → `score_coverage_hole += 0.3`，目标设为该小区<br>• `mr_sample_count < 10` → `score_insufficient_data = 0.7`（新字段）<br>• `overlap_ratio > 0.7` → `score_late_handover += 0.2` | D+4 |
| C5 | 工具调用预算控制 | 每场景维护 wall-time 计时器；超预算后停止继续调用工具，回退到原始特征。默认预算：90 秒。 | D+5 |
| C6 | 集成测试 | 基于真实服务 `--server_url` 的集成测试；每个工具至少覆盖一次。用 `@pytest.mark.integration` 标记，保证无服务器时单元测试仍可通过。 | D+6 |
| C7 | 响应缓存 | 使用 `(tool_name, scenario_id, args_hash)` 作为 key 的 `@lru_cache`。这样在反复评估时可以明显加速。 | D+7 |

### 4.6 约束

- **时间预算**：每个场景的总工具调用时延 ≤ 90 秒（剩余约 210 秒留给 Qwen 与解析）。由 C5 强制执行。
- **失败处理**：任何工具超时、HTTP 错误或 schema 不匹配，都必须回退到原始特征，绝不能让流水线崩溃。
- **限流**：服务器可能会返回 429，必须进行指数退避。

### 4.7 成功判定

- 在任何微调前，`lift_tilt` recall ≥ 0.60
- 在任何微调前，`insufficient_data` recall ≥ 0.60
- F1 场景平均工具调用时延 ≤ 90 秒，p95 ≤ 150 秒
- 在 200 样本评估中，由工具导致的流水线崩溃次数为 0

### 4.8 我承担的风险

| 风险 | 缓解手段 |
|---|---|
| 批量评估时被服务端限流 | C7 缓存 + 指数退避 + 并发上限 4 |
| 工具响应 schema 与预期不一致 | 在边界处用 Pydantic 做校验；异常时软失败 |
| 工具调用耗尽时间折扣预算 | C5 强制预算；必要时退化成确定性模式 |
| 阻塞 Owner A 的重训 | 尽早交付 C1 + C4，就算证据还是部分空值也要先打通接口 |

---

## 5. Owner D — 推理期 RAG 与数据分析工具

### 5.1 职责说明

负责两类相互关联、但又不同的产物：

1. **推理期 RAG**：每次调用 Qwen 时都实时执行的检索系统。这是主产品，直接影响 F1 IOU 和 prompt token 数。
2. **数据分析工具**：提供给 Owner A 的一个打分函数库（去重、异常、标签错误），A 在 `curate_sft.py` 中调用，用来决定哪些样本进入 `sft_v3.jsonl`。

**D 不负责的内容**：具体的整理决策。D 只算分，不拍板。是否丢掉哪些样本、阈值设多少，这些会影响类别平衡、验证集卫生与 hard-negative mining，属于 A 的建模职责。

### 5.2 负责文件

- `rag_retriever_v2.py`：`StructuredCaseRAG`。在 13 维特征向量上构建带家族硬过滤的 FAISS 索引。
- `skills/data_tools.py`：新的工具库。对训练集做纯函数式打分，不产生副作用，也不替 A 做决策。

### 5.3 当前缺口

| 优先级 | 问题 | 行动 |
|---|---|---|
| P0 | 检索结果在动作类型上严重失衡，top-k 经常被主流类别占满 | 做分层检索，每类动作分配 k/6 |
| P0 | 近重复场景霸占 top-k | 用特征向量余弦相似度做去重 |
| P1 | 没有为 `lift_tilt` / `insufficient_data` 做混淆对挖掘 | 显式构建 confusion index |
| P1 | 特征向量目前还没包含工具证据 | 待 C1 落地后与 Owner C 协调，扩展到 16 维 |
| P1 | A 还没有工具能自动发现训练标签问题 | 交付 `data_tools.py` 打分库 |

### 5.4 交付物

| # | 工件 | 详细规范 | ETA |
|---|---|---|---|
| D1 | 特征向量 FAISS 索引 | 基于 `_sf_to_feature_vec()` 的 13 维向量重建索引，缓存到 `.rag_cache_v2/`。触发重建的条件是 `train_data` 的 hash 改变。 | D+3 |
| D2 | 分层 top-k 检索 | `retrieve(sf, k=6) → list[RAGHit]`：先按动作类型分桶，每个桶取 k/6，不足时再回退到桶内纯相似度排序。 | D+4 |
| D3 | Confusion-pair 索引 | 对每个 `lift_tilt` 样本预计算最近的 2 个 `insufficient_data` 邻居，反过来也一样。把它们和普通 top-k 一起返回，帮助模型显式看见边界。 | D+5 |
| D4 | 冻结检索签名 | `retrieve(sf: ScenarioFeatures, k: int = 6) → list[RAGHit]`。D+2 后不允许再改，除非团队批准。 | D+2 |
| D5 | `skills/data_tools.py` | 提供纯函数打分：<br>• `duplicate_score(sid) → float`：与其他场景的最大余弦相似度<br>• `outlier_score(sid) → float`：到类别中心的距离<br>• `label_confidence(sid) → float`：若 gt 在选项中且无歧义则接近 1，否则更低<br>• `ambiguity_score(sid) → float`：第二优选与第一优选有多接近<br>只输出分数，不输出阈值和裁决。 | D+3 |
| D6 | 扩展到 16 维 | 在 Owner C 完成 C4 后，把 `{mainlobe_hit_count, tilt_angle_mean, overlap_ratio_max}` 加入特征向量，并重建索引。 | D+6 |

### 5.5 成功判定

- 在 50 个 F1 `lift_tilt` 探针样本上，top-6 RAG 结果中 ≥ 60% 也应为 `lift_tilt`
- 在 50 个 F1 `insufficient_data` 探针样本上，top-6 RAG 结果中 ≥ 60% 也应为 `insufficient_data`
- 单次端到端 RAG 查询耗时 ≤ 200ms
- `data_tools.py` 在全部 2000 个场景上的打分耗时 ≤ 30 秒
- Owner A 在应用自己设定的筛选阈值后，验证集 IOU 至少提升 0.02

### 5.6 我承担的风险

| 风险 | 缓解手段 |
|---|---|
| 分层检索把验证集样本泄漏进训练侧 | 接受 A 提供的 val-set ID 列表；在构建 RAG 索引时显式排除 |
| 扩展到 16 维会触发 A 重训 | 把 D6 的上线时间与 A4 的重训窗口协调起来 |
| `data_tools.py` 的打分噪声太大，导致 A 的整理策略失真 | 记录分数分布，先在 50 个手工标注样本上做 sanity check |

---

## 6. 冻结接口（如需改动，必须团队批准）

这些接口在 **D+2（2026-04-22）** 之后必须保持稳定，这样四位 Owner 才能并行推进。任何变更都需要提前 24 小时在团队频道预告，并获得 Owner A 同意，因为所有接口变化最终都会波及 SFT 数据。

### 6.1 `ScenarioFeatures`（Owner B）

```python
@dataclass
class ScenarioFeatures:
    mobility: MobilityFeatures            # serving_rsrp_dbm, serving_sinr_db,
                                          # neighbor_advantage_db, a3_margin_db
    throughput: ThroughputFeatures        # worst_throughput_mbps, drop_magnitude,
                                          # drop_duration_s
    inter_freq: InterFreqFeatures         # max_dl_prb_util, max_ul_prb_util

    # 6 个问题分数，范围都在 [0, 1]
    score_late_handover: float
    score_missing_neighbor: float
    score_coverage_hole: float
    score_pdcch_issue: float
    score_backhaul_server: float
    score_inter_freq_load: float

    # 元数据
    top_issue: str                        # 最高分字段名
    serving_cell_id: str
    serving_cell_role: Optional[str]      # "primary" | "secondary" | "tertiary"

    # 由 tool_caller 扩展（可为空）
    tool_evidence: Optional["ToolEvidence"] = None
```

### 6.2 `ToolEvidence`（Owner C）

```python
@dataclass
class ToolEvidence:
    mainlobe_flags:       dict[str, bool]    # cell_id → 是否位于主瓣内
    tilt_angles:          dict[str, float]   # cell_id → 角度
    overlap_ratio:        dict[str, float]   # cell_id → [0, 1]
    kpi_data_raw:         dict               # 原始 KPI blob
    signaling_events:     list               # 事件 dict 列表
    mr_sample_counts:     dict[str, int]     # cell_id → MR 样本数
    tool_call_latency_ms: float              # 用于预算追踪

def augment_features(
    sf: ScenarioFeatures,
    ev: ToolEvidence,
) -> ScenarioFeatures: ...
```

### 6.3 `RAGHit` 与 `retrieve()`（Owner D）

```python
@dataclass
class RAGHit:
    scenario_id: str
    features: ScenarioFeatures
    gt_c_code: str
    gt_action_type: str                   # lift_tilt / insufficient_data / ...
    similarity: float                     # cosine，相似度，范围 [0, 1]
    prompt_snippet: str                   # 两行可读摘要

def retrieve(
    sf: ScenarioFeatures,
    k: int = 6,
    stratify_by_action: bool = True,
    include_confusion_pairs: bool = True,
) -> list[RAGHit]: ...
```

### 6.4 `_build_f1_prompt()`（Owner A）

```python
def _build_f1_prompt(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    candidates: list[tuple[str, float, str]],   # (c_code, score, reason)
    rag_hits: list[RAGHit],
) -> str: ...
```

这个函数由 `build_sft_v3.py` 与 `main_v3.py` 共同使用，确保训练和推理的输入分布保持一致。

---

## 7. 每周节奏

### 7.1 同步时间表

| 日期 | 时间 | 仪式 | 时长 | 必须参加者 |
|---|---|---|---|---|
| 周一 | 10:00 | 规划会：锁定本周优先级 | 30 分钟 | 全员 |
| 周三 | 10:00 | 周中检查：打通跨人依赖 | 20 分钟 | 全员 |
| 周五 | 16:00 | 周末评测 + 演示 | 60 分钟 | 全员 |
| 周五 | 17:00 | 满足门槛时尝试提交 | — | A + 旁观者 |

### 7.2 每日异步更新

每位 Owner 每天 10:00 前都要在团队频道发一条：

```text
[Owner X — Day N]
- Done: <1 line>
- Doing: <1 line>
- Blocked on: <1 line, or "nothing">
```

### 7.3 跨 Owner 沟通规则

- **PR 审查**：任何冻结接口的变更都必须经过跨 Owner 审查。若只改自己负责的文件，可以自合并。
- **事故处理协议**：如果一次合并让评估分数下降超过 0.05，先 `git revert`，再开会讨论。
- **问题路由**：在频道里直接 @ 对应 Owner；工作时间内响应 SLA 是 2 小时。

## 7.5 集成职责（跨层事项）

有些事情不属于任何一个 Owner 的单独职责，但必须有人认领。默认由 A 负责，因为最终是 A 把所有人的成果装配成一次可提交的系统。

### 7.5.1 端到端集成测试（Owner A）

- **产物**：`tests/test_pipeline_integration.py`，对 5 个夹具场景跑通 `parse → route → features → tools → rag → candidates → prompt → deterministic-fallback`，覆盖 4 个家族。
- **触发方式**：只要某个 PR 改到了任何一位 Owner 的责任文件，就要跑这个测试。如果某个阶段报错或输出形状变了，PR 必须失败。
- **ETA**：D+3，在后续重训依赖此链路前先打通。

### 7.5.2 `_build_f1_prompt()` 的维护权（Owner A）

- prompt 模板是训练/推理对齐的唯一真源，Owner A 负责维护。
- **规则**：`build_sft_v3.py` 与 `main_v3.py` 必须 import 同一个 `_build_f1_prompt` 符号，并由集成测试做静态检查。
- **从 Day 1 就支持 nullable**：`ScenarioFeatures`、`ToolEvidence`、`RAGHit` 中被 prompt 消费的每个字段，都必须在 `None` 或空列表时也能正常渲染。这样 A3（无工具）与 A4（有工具）就能共享同一个模板。

### 7.5.3 基础设施：Qwen 服务、GPU 分配、checkpoint 存储（Owner A）

- 通过 `serve_model.py` 在 `http://localhost:8001/v1` 拉起 vLLM + LoRA adapter。
- checkpoint 存在 `checkpoints/v3/ckpt_YYYYMMDD_stepN/`，只保留最近 3 个，老的归档后删除。
- **ETA**：D+2，在 A3 启动前完成。

### 7.5.4 wall-time 预算归属（Owner A，总控；C/D 汇报）

此前没人真正对 5 分钟总时长负责；现在 A 统一持有预算：

| 预算 | 阶段 | 向谁汇报 | 上限 |
|---|---|---|---|
| 90 秒 | 工具调用（C） | Owner C 通过 `ToolEvidence.tool_call_latency_ms` 回报 | C5 |
| 200 ms | RAG 检索（D） | Owner D 通过检索计时器回报 | D 的成功标准 |
| 180 秒 | Qwen 推理（A） | A 自行测量 | A4 的上线门槛 |
| 30 秒 | 解析 + 打分（B） | 几乎可忽略 | — |
| **< 300 秒总计** | 端到端流水线 | A 在每次 eval 里测 p50 和 p95 | A7 |

如果 p95 > 300 秒，A 要立刻发起 wall-time review 会议，而不是让某个 Owner 单独背锅。

### 7.5.5 schema freeze 与 bug 修复的协调（Owner B）

B 在 D+2 冻结 schema，但 B2/B3 要在 D+3 才修复。协调原则如下：

- **冻结的是 dataclass 形状，不是字段取值**。也就是说，后续仍可修改已有字段的打分逻辑。
- **冻结后新增字段**必须满足三件事：默认值为 `None`；Owner A 签字；自动排队一次重训。
- `score_insufficient_data`（C4 引入的新字段）属于预先批准的唯一例外，必须在 D+2 前落地。

### 7.5.6 RAG 索引重建如何联动 SFT 重建（由 Owner A 在 Owner D 发信号后触发）

当 D6 把特征向量从 13 维扩展到 16 维时：

1. D 在版本化路径下发布新的 `.rag_cache_v2/`。
2. D 在频道里通知 A 新的缓存路径。
3. A 用新索引重跑 `build_sft_v3.py`（新增的 3 维会出现在 `RAGHit.features` 中，并流入 `_build_f1_prompt`）。
4. A 只在下一轮既定重训（A4 或 A5）里吸收这个变化，不额外开一次单独重训。

### 7.5.7 prompt 调整流程（Owner A 提案，全员评审）

`QWEN_V3_SYSTEM_PROMPT` 虽然按约定冻结，但并非完全不能改。如果 A 想改：

- 必须提交一个 PR，附上改前改后在 200 样本评估上的 IOU 对比，包括旧 prompt 的 LoRA 和按新 prompt 重训后的 LoRA。
- 四位 Owner 都要审查，SLA 为 72 小时。
- 批准条件：任一家族都不能出现 > 0.02 的回归。

### 7.5.8 文档维护责任（共享）

- `README_V3.md`：Owner A 主负责，但谁都可以提 PR
- `TEAM_DIVISION.md`（本文档）：Owner A
- `SCHEMA.md`：Owner B（冻结 dataclass 的参考文档）
- 每位 Owner 都要维护自己负责文件顶部的 docstring，让它保持最新

---

## 8. 风险登记表

| 风险 | 发生概率 | 影响 | Owner | 缓解手段 |
|---|---|---|---|---|
| 工具调用耗尽时间折扣预算 | 中 | 高 | C | 每场景预算 90 秒；超时回退到特征；响应缓存 |
| 特征 schema 频繁震荡，迫使 A 反复重训 | 中 | 高 | B | D+2 硬冻结；任何变更都需 A 签字 |
| 微调对 RAG 示例过拟合 | 中 | 中 | A | 分层 train/val；早停；打乱 RAG 顺序 |
| F2 回归迟迟修不掉 | 低 | 中 | B | D+3 专门排查；回归测试套件 |
| 评测期服务端限流 | 中 | 中 | C | 响应缓存；指数退避；并发上限 4 |
| 提交脚本本身有 bug | 低 | 严重 | A | D+7 用 mock server dry-run；D+8 再真提 |
| RAG 去重过度 | 低 | 中 | D | 使用容差；人工复核；跟踪训练集规模变化 |
| A 的 LoRA 推理太慢 | 低 | 高 | A | D+3 做基准测试；必要时回退到 AWQ |
| Qwen 返回非法 C-code | 低 | 高 | A | 后处理先剥掉非法 token，再用更严格 prompt 重试一次 |
| 集成时跨 Owner 接口不匹配 | 中 | 高 | 全员 | `tests/test_pipeline_integration.py` 每天在 CI 上跑 |

---

## 9. 完成定义（团队级）

只有当下面所有门槛都通过，团队工作才算真正完成：

### 9.1 定量指标

- ✅ 离线 200 样本评估的 Overall mean IOU ≥ **0.85**
- ✅ 分家族：F1 ≥ 0.78，F2 ≥ 0.98，F3 ≥ 0.98，F4 ≥ 0.98
- ✅ F1 关键动作最低线：`lift_tilt` ≥ 0.60，`insufficient_data` ≥ 0.60，`add_neighbor` ≥ 0.65
- ✅ 平均单场景 wall-clock ≤ 5 分钟（才能吃满时间折扣）
- ✅ p95 wall-clock ≤ 8 分钟

### 9.2 定性要求

- ✅ 从 `v3/` 目录出发可以一键复现（`python main_v3.py ...`）
- ✅ 所有接口都记录在 `SCHEMA.md`
- ✅ 集成测试覆盖每一位 Owner 的交接边界
- ✅ 有消融报告解释每个人的工作到底贡献了什么
- ✅ 在 2026-05-04 前把提交送上排行榜

### 9.3 最终交接工件

- `result_final.csv`（排行榜提交文件）
- `ABLATION.md`（各组件贡献说明）
- `sft_v3.jsonl`（训练数据归档）
- LoRA adapter 权重（checkpoint）
- `.rag_cache_v2/` 索引快照

## 9.5 每位 Owner 的回传包（最终要交给 A 的东西）

所有人都在同一个 Git 仓库内、用各自 feature branch 开发。下表规定了**每个人具体修改哪些文件，以及最终怎样把东西交回给 Owner A 做总装与提交**。

### 9.5.1 Owner A（我）——我端到端负责的文件

| 文件 | 状态 | 要做什么 |
|---|---|---|
| `curate_sft.py` | **stub** | 实现 A1a：调用 D 的 `data_tools.py` 分数，写出 `curated_train_ids.txt` 与 `curation_report.md` |
| `build_sft_v3.py` | **stub** | 实现 A1b：消费整理后的 id，调用 `_build_f1_prompt`，写出 `sft_v3.jsonl` |
| `submit.py` | **stub** | 实现 A6：运行 `main_v3.py`，校验 C-code 合法性，再上传排行榜 |
| `main_v3.py` | 现有文件 | 加入 `_build_f1_prompt()`，并落实第 7.5.2 节的 nullable-from-day-1 规则；除此之外不做多余改动 |
| `tests/test_pipeline_integration.py` | **stub** | 补齐夹具与断言 |
| LoRA 训练配置 | 待创建 | 按第 2.3 节 A2 写 `llama-factory` YAML |
| `checkpoints/v3/` | 待创建 | LoRA checkpoints 目录 |

**最终我要装配出的工件**：

- `sft_v3.jsonl`（A1b 产出）
- LoRA adapter 权重（A3/A4/A5）
- `result_final.csv`（A6）
- `ABLATION.md`（A7）
- 来自 Owner D 的 `.rag_cache_v2/` 打包文件

### 9.5.2 Owner B —— 通过 PR 回传

| 文件 | 动作 | 门槛 |
|---|---|---|
| `skills/feature_engine.py` | 修改：修复 `add_neighbor`（B2）、修复 F2 回归（B3）、增加 `lift_tilt` 启发式（B4）、做 B6 重构 | D+2 后不得改 `ScenarioFeatures` dataclass 的形状（见 `SCHEMA.md` §1） |
| `skills/candidate_generator.py` | 修改：修复 `add_neighbor` 的目标小区选择（B2） | 不允许改接口 |
| `skills/scenario_parser.py` | 只做必要轻修 | `ParsedScenario` 形状已冻结 |
| `skills/scenario_router.py` | 理论上无需改 | `Family` 枚举已冻结 |
| `skills/answer_parser.py` | 理论上无需改 | 评分函数已冻结 |
| `skills/rf_analysis.py` | Owner B 范围内的已有文件 | 不允许改接口 |
| `tests/test_features.py` | **新建** | B5：覆盖 6 个 F1 动作类别的单测 |
| `SCHEMA.md` §1 | 仅在 dataclass 新增预批准字段时修改 | 需 A 签字 + 重训排队 |

**交回给 A 的方式**：直接提 PR 即可，不需要额外传文件，但 CI 必须是绿的。

### 9.5.3 Owner C —— 通过 PR + fixture tarball 回传

| 文件 | 动作 | 门槛 |
|---|---|---|
| `skills/tool_caller.py` | 实现 C1–C7：补全 `ToolEvidence`、`call_targeted`、`augment_features`、预算、缓存 | `ToolEvidence` 的形状在 D+2 冻结（见 `SCHEMA.md` §2） |
| `tests/test_tools.py` | **新建** | C6：集成测试，打 `@pytest.mark.integration` |

**交回给 A 的东西**：

1. 合并到主分支的 PR（代码本身）
2. `tool_evidence_fixtures.json`：按 `scenario_id` 索引的 10 个 `ToolEvidence` 样例。这样即使真实服务器暂时不可用，A 也能先构建 `sft_v3.jsonl`

### 9.5.4 Owner D —— 通过 PR + cache tarball 回传

| 文件 | 动作 | 门槛 |
|---|---|---|
| `rag_retriever_v2.py` | 实现 D1–D4 与 D6：FAISS 索引、分层检索、混淆对、13→16 维扩展 | `retrieve()` 签名在 D+2 冻结（见 `SCHEMA.md` §3） |
| `skills/data_tools.py` | 实现 D5：四个纯函数式打分函数 | 仓库里已有 stub，需要填掉 `NotImplementedError` |

**交回给 A 的东西**：

1. 合并到主分支的 PR（代码本身）
2. `rag_cache_v2.tar.gz`：构建好的 FAISS 索引。它是二进制工件，不进 Git。放到 `/home/huangxiaolin/AITO/dataset/Track A/v3/artifacts/rag_cache_v2.tar.gz`

### 9.5.5 编译装配顺序（由 Owner A 依次执行）

```bash
# 1. 拉取所有人的 PR，先跑 CI
pytest tests/ -v

# 2. 解压 D 的 RAG 缓存
tar -xzf artifacts/rag_cache_v2.tar.gz -C .

# 3. 运行 D 的打分库，再由我做整理决策
python curate_sft.py --train_dir train_data --out_ids curated_train_ids.txt

# 4. 构建 SFT 数据集
python build_sft_v3.py --ids curated_train_ids.txt --train train_data --out sft_v3.jsonl --upsample

# 5. 训练 LoRA
llamafactory-cli train configs/lora_v3.yaml

# 6. 带 LoRA 跑离线评估
python eval_v3.py --full --per-action --model http://localhost:8001/v1 --lora checkpoints/v3/ckpt_latest

# 7. Dry-run 提交
python submit.py --server_url ... --model_url ... --lora checkpoints/v3/ckpt_latest --dry-run

# 8. 真正提交（只有 dry-run 通过后才允许）
python submit.py --server_url ... --model_url ... --lora checkpoints/v3/ckpt_latest
```

---

## 10. 术语表

| 术语 | 定义 |
|---|---|
| **C-code** | C1..C22 中的一个编码选项，是场景多选题的答案单位 |
| **F1/F2/F3/F4** | 场景家族，见 README §2.2 |
| **ScenarioFeatures** | Owner B 的输出，即本地确定性分析结果 |
| **ToolEvidence** | Owner C 的输出，即来自服务端工具的权威证据 |
| **RAGHit** | Owner D 的输出，即一个被检索到的相似训练样例 |
| **SFT** | Supervised Fine-Tuning，即有监督微调（Owner A 的工作） |
| **IOU** | 预测 C-code 集合与真实集合的交并比 |
| **Time discount** | 依据 wall-time 施加在 IOU 上的折扣（<5min=1.0，<10min=0.5，≥10min=0） |
| **Top issue** | `ScenarioFeatures` 中得分最高的问题字段名 |
| **Main lobe** | 天线主辐射方向（由 `judge_mainlobe` 判定） |
| **A3 event** | 3GPP 定义的切换测量触发事件 |
| **MR data** | Measurement Reports，即 UE 侧无线测量数据 |
| **CCE utilization** | 下行控制信道元素利用率，高时意味着 PDCCH 拥塞 |
| **PRB utilization** | 物理资源块利用率，高时意味着无线拥塞 |

---

## 11. 参考资料

- 架构规范：`v3/README_V3.md`
- 特征 schema：`skills/feature_engine.py:ScenarioFeatures`
- 候选逻辑：`skills/candidate_generator.py:generate_single_candidates`
- 评分函数：`skills/answer_parser.py:compute_iou`
- 服务端工具：`server.py`（请勿修改）
- 历史 V2：`../main_v2.py`
