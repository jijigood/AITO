# V3 框架 — Track A 电信故障排查

**竞赛**：电信故障排查 Agentic Challenge — Track A（无线网络）
**第一阶段截止日期**：2026-05-04
**基础模型**：Qwen3.5-35B-A3B（不可替换，仅允许使用 LoRA）
**评分**：`IOU(predicted_codes, ground_truth) × time_discount`

---

## 1. 执行摘要

V3 是一条**模块化、按场景家族优化的流水线**，将确定性特征工程与微调后的 Qwen LoRA 结合起来，取代了 V2 那种“把所有内容一股脑交给模型”的单体式方案。核心洞见是：**F2/F3/F4 场景（约占 25% 流量）可以通过确定性规则以 100% 准确率解决；而 F1 场景（约占 75%）也不再需要从零生成，而是变成 LoRA 的一个干净的重排序任务。**

### 分数演进

| 阶段 | 方法 | 总体 IOU | F1 | F2 | F3 | F4 | 说明 |
|---|---|---|---|---|---|---|---|
| V1 基线 | 仅原始 Qwen | ~0.25 | — | — | — | — | LoRA 之前 |
| V2（已提交） | LoRA + 最少特征 | **0.3665** | 0.17 | 0.85 | 1.00 | 0.90 | 频间负载被过度触发 |
| V3 确定性版 | 固定特征，无模型 | **0.6750** | 0.5752 | 1.00 | 1.00 | 1.00 | 当前离线结果 |
| V3 目标 | + 重训 LoRA + 工具证据 | **≥ 0.85** | 0.78 | 1.00 | 1.00 | 1.00 | 提交目标 |

### 当前运行基线（2026-04-25）

- `deterministic + tools` 经过 2026-04-25 的映射修复后，已经成为新的**heldout 最佳参考**：F1 overall IOU `0.8500`，`lift_tilt=1.0000`，`insufficient_data=0.9062`，`check_server=1.0000`，`add_neighbor=0.3421`，平均时延 `0.0s`。
- 触发这次跃迁的关键探针是那 58 个卡点样本：overall IOU `0.9655`，其中 `lift_tilt=1.0000`（`n=26`），`insufficient_data=0.9375`（`n=32`）。
- `ckpt-477 fixed` 现在保留为**接工具前的 LoRA-only 时延参考**：F1 overall IOU `0.6000`，`add_neighbor=0.4737`，平均时延 `19.4s`，fallback `16/200 = 8%`。
- `corrected fixed (246)` 现在保留为**接工具前的 LoRA-only 精度参考**：F1 overall IOU `0.6050`，`add_neighbor=0.5000`，平均时延 `40.8s`，fallback `36/200 = 18%`。

### 相比 V2 的关键提升

1. **F2/F3/F4 完全绕过模型**。确定性候选本来就是对的，Qwen 反而会引入噪声（0.85 → 1.00）。
2. **4BS 场景的频间负载上限**。V2 在约 80% 的 F1 场景里把 `inter_freq_load` 打到 >0.5，淹没了其他所有信号。V3 对 ≤5BS 场景把它封顶到 0.08。
3. **基于 SINR 的问题分组**。SINR<0 → 切换类问题；SINR>5 且 PRB 高 → 覆盖/回传；SINR>5 且 PRB 低 → PDCCH/服务器。
4. **为服务小区候选增加权重**。V2 假设“primary 一定是目标”，V3 会读取真实的 serving-cell 角色并施加 1.05× 加权。
5. **离线评估框架**。`eval_v3.py` 无需服务器也无需 Qwen 即可运行，让我们可以在秒级完成迭代。

---

## 2. 竞赛背景

### 2.1 任务

给定一个故障排查场景（CSV 日志 + 文本描述 + 多项选择选项，标记为 C1..C22），预测正确的 C-code 集合。场景描述的是无线网络异常（如吞吐低、掉话、切换失败），选项则是与具体小区绑定的候选缓解动作。

### 2.2 场景家族

| 家族 | 描述 | 占比 | 答案类型 | 典型基站数 |
|---|---|---|---|---|
| **F1** | 4 基站部署，单一根因 | ~75% | 1 个 C-code | 4 |
| **F2** | 4 基站部署，多根因（两项动作） | ~9% | 2 个 C-code | 4 |
| **F3** | 5 基站部署，多根因（固定 4 项动作） | ~4% | 4 个 C-code | 5 |
| **F4** | 13 基站部署，单一根因 | ~12% | 1 个 C-code | 13 |

### 2.3 F1 动作类型分布

F1 是最难的场景家族，因为要在 6 个类别里选出 1 个动作，而这些类别的特征分布高度重叠：

| 动作 | 占比 | 区分信号 |
|---|---|---|
| `insufficient_data` | 21% | 某些小区缺少 MR 样本；只看本地信息时与 `lift_tilt` 难以区分 |
| `lift_tilt` | 19% | 天线主瓣方向错误；需要 `judge_mainlobe` 的几何检查 |
| `pdcch_2sym` | 18% | 下行 CCE 利用率 > 0.5，且 PRB 低 |
| `check_server` | 16% | PRB 高、RSRP 高、SINR>5（说明不是无线问题） |
| `add_neighbor` | 15% | 邻区 RSRP 更优，但没有被配置 |
| `decrease_a3` | 11% | 邻区更优，但切换触发过晚 |

### 2.4 评分方式

```text
IOU = |pred ∩ gt| / |pred ∪ gt|
time_discount = 1.0  if wall_time < 5 min
              = 0.5  if 5 ≤ wall_time < 10 min
              = 0.0  if wall_time ≥ 10 min
final_score  = IOU × time_discount
```

这意味着**过度预测和漏预测受到的惩罚一样重**（因为分母是并集），并且**任何单个场景耗时超过 10 分钟，该场景直接记 0 分**。

---

## 3. 流水线架构

```text
┌──────────────────────────────────────────────────────────────────┐
│  原始场景（CSV 文件 + 问题文本 + C1..C22 选项）                  │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. scenario_parser.ParsedScenario                               │
│     • traffic.csv → DataFrame                                     │
│     • mr.csv      → DataFrame                                     │
│     • site.json   → dict（小区坐标、方位角、俯仰角）              │
│     • options     → list[CodedOption]                             │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. scenario_router.route() → Family enum {F1, F2, F3, F4}       │
│     基于基站数、标签与答案个数的决策树                            │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. feature_engine.run_all_skills() → ScenarioFeatures            │
│     Skill A: 移动性信号（RSRP、SINR、邻区优势）                   │
│     Skill B: 吞吐/掉线检测                                        │
│     Skill C: 频间负载（PRB 利用率）                               │
│     Skill D: 信令事件（切换尝试、失败）                           │
│     Skill E: PDCCH / CCE 利用率                                   │
│     Skill F: 覆盖空洞几何检查                                     │
│     Skill G: 问题分数聚合 + V3 的 4BS 改造                        │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. candidate_generator.generate_*_candidates()                  │
│     将问题分数映射为 (c_code, score, reason) 三元组              │
│     应用 serving-cell 加权（V3）与选项可用性掩码                 │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
                   ┌───────────┴────────────┐
                   │                        │
              (F2/F3/F4)                 (F1)
                   │                        │
                   ▼                        ▼
          top_candidate()          tool_caller.augment()
                                           │
                                           ▼
                                  rag_retriever_v2.retrieve(k=6)
                                           │
                                           ▼
                                  通过 serve_model.py 调用 Qwen LoRA
                                           │
                                           ▼
                                  answer_parser.extract()
                                           │
                   └───────────┬────────────┘
                               ▼
                      (c_code_list, confidence)
                               │
                               ▼
                     result.csv + result_final.csv
```

---

## 4. 快速开始

### 4.1 环境

项目默认使用位于 `/home/huangxiaolin/AITO/` 的 `uv` 虚拟环境：

```bash
# 激活环境
source /home/huangxiaolin/AITO/bin/activate

# 或直接调用
/home/huangxiaolin/AITO/bin/python main_v3.py ...
```

依赖包：`pandas`、`numpy`、`faiss-cpu`、`sentence-transformers`、`requests`、`openai`（用于 OpenAI 兼容的 vLLM 端点）。

### 4.2 离线评估（不需要服务器，也不需要 Qwen）

```bash
# 完整 200 样本离线评估
python eval_v3.py --full --per-action

# 输出错误分析 CSV（每个失败场景记录 top-6 候选）
python eval_v3.py --full --per-action --errors errors.csv

# 只评估单一家族
python eval_v3.py --full --per-action --family F1
```

### 4.3 连接竞赛服务器并运行 Qwen

```bash
# 在另一个终端启动 Qwen 服务（vLLM + LoRA 适配器）
bash launch_qwen.sh   # 暴露 http://localhost:8001/v1

# 运行端到端流水线
python main_v3.py \
    --server_url http://localhost:7860 \
    --model_url  http://localhost:8001/v1 \
    --output_dir results/v3_run_$(date +%Y%m%d_%H%M%S)
```

### 4.4 常用参数

| 参数 | 作用 |
|---|---|
| `--deterministic` | 跳过 Qwen；所有场景都直接采用 top candidate（适合作为回退方案） |
| `--use-tools` | 为 F1 启用 `tool_caller.augment_features()`（待 Owner C 的工作落地后使用） |
| `--no_rag` | 关闭 RAG 检索（用于验证某些分布下 RAG 是否在拖后腿） |
| `--limit N` | 仅处理前 N 个场景，便于快速迭代 |
| `--verbose` | 打印逐场景特征、候选和 Qwen 提示词 |

---

## 5. 文件清单

### 5.1 顶层文件

| 文件 | 行数 | 用途 |
|---|---|---|
| `main_v3.py` | ~650 | V3 主调度器。包含 `V3Runner` 类、`solve()`、`benchmark()`，也是 CLI 入口。 |
| `eval_v3.py` | ~220 | 离线评估框架。从 `train_data/` 加载场景，完整运行到候选选择阶段，输出按家族和按动作划分的 IOU。 |
| `serve_model.py` | ~180 | 与 vLLM 兼容的 Qwen 服务，加载 LoRA 适配器，并提供 OpenAI Chat Completions API。 |
| `rag_retriever_v2.py` | ~332 | `StructuredCaseRAG`。在训练集上构建 13 维特征向量的 FAISS 索引，并按家族名称做硬过滤。 |
| `TEAM_DIVISION.md` | — | 四人分工说明。开始任务前先读它。 |
| `README_V3.md` | — | 本文档。 |

### 5.2 `skills/` 目录

| 文件 | 用途 | 负责人 |
|---|---|---|
| `scenario_parser.py` | 解析 CSV 与选项，生成 `ParsedScenario` dataclass | B |
| `scenario_router.py` | 家族分类器（F1/F2/F3/F4），依据基站数与标签分类 | B |
| `feature_engine.py` | 7 段 skill 流水线，输出 `ScenarioFeatures`；这里承载了 V3 的核心改进 | B |
| `candidate_generator.py` | 将问题分数映射为带原因的 C-code 候选列表 | B |
| `answer_parser.py` | 从 Qwen 输出中提取 `\boxed{C1,C5}`，并计算 IOU | B |
| `tool_caller.py` | 包装 19 个服务器工具；输出 `ToolEvidence` | C |

### 5.3 数据

| 路径 | 用途 |
|---|---|
| `train_data/` | 2000 条训练场景。供 RAG 与 SFT 使用的只读输入。 |
| `.rag_cache_v2/` | 序列化后的 FAISS 索引和特征矩阵。首次运行时自动构建。 |

---

## 6. 特征引擎深度解析

`feature_engine.run_all_skills()` 是 V3 的核心。每个 skill 都会产出信号，供候选生成器消费。

### 6.1 Skills 一览

| Skill | 产出 | 核心逻辑 |
|---|---|---|
| A（mobility） | `serving_rsrp_dbm`、`serving_sinr_db`、`neighbor_advantage_db`、`a3_margin_db` | 解析 MR CSV；通过多数投票识别 serving cell；计算最佳邻区相对 serving 的增益 |
| B（throughput） | `worst_throughput_mbps`、`drop_magnitude`、`drop_duration_s` | 通过滚动最小值识别持续吞吐塌陷 |
| C（inter-frequency） | `max_dl_prb_util`、`max_ul_prb_util` | 对所有小区取最大值，用于识别拥塞 |
| D（signaling） | 切换尝试次数、失败率 | 统计 `HO_Request` 与 `HO_Failure` 事件 |
| E（PDCCH） | `max_dl_cce` | 下行 CCE 最大利用率，是 `pdcch_2sym` 的关键信号 |
| F（coverage） | `coverage_hole_risk` | 利用站点几何关系构建覆盖空洞启发式 |
| G（aggregator） | 全部 `score_*` 字段 | 将前面各项 skill 聚合为 6 个问题分数 |

### 6.2 V3 的 4BS 改造（关键修复）

```python
# 在 run_all_skills() 末尾、13BS 家族增强之后
if ps.num_base_stations <= 5 and ps.num_base_stations != 13:
    sf.score_inter_freq_load = min(sf.score_inter_freq_load, 0.08)

if ps.num_base_stations == 4 and is_single:
    if sinr < 0:
        # 切换类问题
        if neighbor_advantage > 8 and margin > 2:
            score_missing_neighbor = 0.75
            score_late_handover    = 0.35
        elif neighbor_advantage > 3:
            score_late_handover    = 0.70
            score_missing_neighbor = 0.30
        # 压制无关分数
        score_coverage_hole    = min(..., 0.15)
        score_pdcch_issue      = min(..., 0.15)
        score_backhaul_server  = min(..., 0.15)

    elif sinr > 5:
        if max_prb > 70:
            score_coverage_hole    = max(..., 0.45)
            score_backhaul_server  = max(..., 0.40)
            # 压制切换相关分数
        else:
            if max_dl_cce > 0.5:
                score_pdcch_issue     = max(..., 0.50)
                score_backhaul_server = max(..., 0.40)
            else:
                score_backhaul_server = max(..., 0.50)
                score_pdcch_issue     = max(..., 0.40)
            # 压制切换相关分数
```

### 6.3 问题分数到动作的映射

| 最高问题分数 | 映射动作（按优先级） |
|---|---|
| `score_late_handover` | `decrease_a3` |
| `score_missing_neighbor` | `add_neighbor` |
| `score_coverage_hole` | `lift_tilt`、`increase_power` |
| `score_pdcch_issue` | `pdcch_2sym` |
| `score_backhaul_server` | `check_server`、`insufficient_data` |

---

## 7. 候选生成器深度解析

`candidate_generator.generate_single_candidates()` 会生成传给 Qwen 的排序候选列表：

1. 对每个问题分数 ≥ 0.3 的项，查表找到对应动作。
2. 在 `ps.options` 中找到与 (`action_type`, `target_cell_role`) 匹配的选项。
3. 如果目标是 serving cell，则施加 `score × 1.05` 的加权（这是 V3 的修复点）。
4. 调用 `_upsert_candidate()`：若新分数高于已有分数则更新，否则跳过。
5. 返回按分数降序排序的 `[(c_code, score, reason), ...]`。

### 7.1 为什么 V2 在 `decrease_a3` 上失败

V2 假设每个动作的 `target_cell_role="primary"`。但在真实数据里，serving cell 经常是 `secondary` 或 `tertiary`。当 `primary` 与 `secondary` 都被赋予同一个 `late_handover` 分数时，V2 会按列表顺序把 `primary` 放在前面，从而选错。

### 7.2 V3 的修复方式

```python
if serving_role:
    if sf.score_late_handover >= 0.3:
        hit = _find_best_option(
            ps, "decrease_a3", serving_role,
            boost=sf.score_late_handover * 1.05,
        )
        if hit:
            _upsert_candidate(scored, reasons, hit[0], hit[1], reason)
```

1.05× 的加权幅度不大，但在两个候选原本打平时，足以让真正的 serving-cell 选项赢得 tie-break。

---

## 8. 评估框架（`eval_v3.py`）

它会在带有已知标签的训练场景上运行整条确定性流水线，并输出：

```text
Overall mean IOU: 0.6750
F1: 0.5752 (n=150)
F2: 0.8421 (n=19)
F3: 1.0000 (n=8)
F4: 1.0000 (n=23)

Per-action breakdown (F1 only):
  check_server       1.00 (n=24)
  pdcch_2sym         1.00 (n=27)
  decrease_a3        0.97 (n=17)
  add_neighbor       0.35 (n=23)
  lift_tilt          0.00 (n=28)
  insufficient_data  0.00 (n=31)
```

使用 `--errors errors.csv` 时，会把每个失败样本写成一行，包含：

- `scenario_id`、`family`、`gt_c_code`、`pred_c_code`
- `gt_rank`：正确答案在候选列表中的排名
- `candidates`：带分数的 top-6 候选
- `top_issue`：最高分的问题类型

**推荐迭代闭环**：修改 `feature_engine.py` → 运行 `python eval_v3.py --full --per-action` → 检查指标变化 → 提交。

---

## 9. 剩余优化目标

### 9.1 `lift_tilt` — 当前为 0%

**失败原因**：正确选项通常针对的是一个非 serving 的小区（通常是 `tertiary`），因为问题出在天线方向模式而不是 serving cell 本身。仅看本地 CSV 时，这种情况与 `insufficient_data` 很像，都会表现为高 SINR + 中等 PRB + 中等吞吐。

**修复路径**：

- 短期（Owner B）：加一个几何启发式。如果 serving cell 的 RSRP 处于中等区间（-105 到 -90），并且在主瓣方向附近没有相邻小区，则把 `lift_tilt` 的分数提到高于 `insufficient_data`。
- 长期（Owner C）：对每个候选目标小区调用 `judge_mainlobe` + `calculate_tilt_angle`，把主瓣标记作为硬判别依据。

**目标**：从 0% 提升到 50%（确定性），加上 LoRA 后达到 70%。

### 9.2 `insufficient_data` — 当前为 0%

**失败原因**：真正的关键信号是“某个小区的 MR 数据缺失或已经过时”。这类模式在有合适提示词时，Qwen 能识别，但当前本地启发式看不到。

**修复路径**：

- Owner D：构建一个 RAG 索引，在特征签名匹配时返回 `insufficient_data` 的示例（典型特征是 PRB 高但几乎没有切换活动）。
- Owner A：把这一模式显式写入 CoT/rationale token 中，对 Qwen 进行微调。

**目标**：从 0% 提升到 30%（纯确定性），加上 LoRA 后达到 60%。

### 9.3 `add_neighbor` — 当前为 35%

**失败原因**：当 `neighbor_advantage_db > 8` 且 `a3_margin_db > 2` 时，候选生成器会正确给出 `missing_neighbor = 0.75`，但 `_find_best_option` 有时会选错目标小区，因为 MR 数据中的“最佳邻区”不一定就是选项文本里缺失的那个邻区。

**修复路径**（Owner B）：交叉校验 `option.target_cell` 与 `mr_data.best_neighbor_cell`；两者不一致时，优先采用与 MR 信号一致的选项。

**目标**：从 35% 提升到 75%。

### 9.4 F2 回归 — 1.0 → 0.84

**失败原因**：为 F1 引入的 SINR 分组修复同时影响到了 F2，因为 F2 也是 4BS 场景。`is_single` 这个保护条件挡住了大多数情况，但并没有覆盖所有标签变体，尤其是一些非规范写法里也包含 “single”。

**修复路径**（Owner B）：让 `is_single` 对标签变体更鲁棒，并补上覆盖所有 F2 标签模式的单元测试。

**目标**：从 0.84 恢复到 1.0。

---

## 10. 微调整合

LoRA 在 `sft_v3.jsonl` 上训练，训练数据由 Owner A 基于这条 V3 流水线构建：

1. 对每个训练场景，依次调用 `parse` → `route` → `feature_engine` → `candidate_generator`。
2. 如果场景家族是 F1，就用 `_build_f1_prompt(ps, sf, candidates, rag_hits)` 序列化出提示词。
3. assistant 回合写成 `\boxed{gt_c_code}` + 一段简短理由。
4. 保存为 `{system, user, assistant}` 三元组。

**为什么这很重要**：训练与推理时的 prompt 分布保持完全一致。LoRA 学的是“如何对候选列表重排序”，而不是“如何从零生成 C-code”。这比自由生成简单得多，也更容易迁移。

Owner A 的完整计划见 `TEAM_DIVISION.md` 第 2 节。

---

## 11. 故障排查

| 症状 | 可能原因 | 修复方式 |
|---|---|---|
| `ModuleNotFoundError: pandas` | 正在使用系统 Python | 执行 `source /home/huangxiaolin/AITO/bin/activate` |
| RAG 没有返回结果 | `.rag_cache_v2/` 不存在 | 删除缓存目录；首次运行时会自动重建 |
| 修改特征后 F2 分数掉到 1.0 以下 | F1 的 SINR 分组修复泄漏到了多答案路径 | 检查 `feature_engine.py` 中的 `is_single` 保护条件 |
| Qwen 幻觉出 C1..C22 之外的 C-code | 系统提示词缺少选项约束 | 使用 `QWEN_V3_SYSTEM_PROMPT`，不要自行覆盖 |
| wall-time > 5 分钟 | 工具调用没有被限流 | 检查 `tool_caller.py` 中的工具预算，并为每个工具加超时 |

---

## 12. 变更记录

| 日期 | 变更 | 文件 |
|---|---|---|
| 2026-04-20 | 创建 V3 框架 | `main_v3.py`、`eval_v3.py`、特征/候选修复 |
| 2026-04-20 | 添加 `tool_caller.py` 骨架 | `skills/tool_caller.py` |
| 2026-04-20 | 形成自包含的 `v3/` 目录分发版 | 整个目录 |
| 2026-04-20 | 从 v3 中移除旧版 `rag_retriever.py` | 仅保留 V2 的 `StructuredCaseRAG` |

---

## 13. 参考资料

- Track A 数据 schema：`data/Phase_1/train.json`
- 竞赛服务器 API：`server.py`（请勿修改）
- 评分函数：`skills/answer_parser.py:compute_iou()`
- V2 架构（历史版本）：`../main_v2.py`
