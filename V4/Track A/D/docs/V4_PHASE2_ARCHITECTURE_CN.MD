## 1. 系统架构

### 1.1 总体流程

```text
Phase 2 test.json
  500 个 scenario；题面和选项可用，数据字段是 stub
        |
        v
APIDataGatherer  (skills/api_data_gatherer.py)
  默认 4 个核心云端 API：
    get_user_plane_data  -> user_plane_data CSV，并提取 worst_time
    get_config_data      -> network_configuration_data CSV
    get_kpi_data         -> traffic_data CSV
    get_mr_data          -> mr_data CSV
  可选：
    get_signaling_plane_event_log，目前 live 返回为空，默认关闭
  条件工具：
    judge_mainlobe_or_not
    calculate_overlap_ratio
        |
        v
ScenarioParser  (skills/scenario_parser.py)
  将 API CSV 重构为 ParsedScenario
  解析 C-code 选项，得到 action_type、target_cell、cells
  建立 pci <-> cell_id 映射
        |
        v
ScenarioRouter + Phase 2 override
  BS=4,  single-answer   -> F1
  BS=4,  multiple-answer -> F2
  BS=5,  single-answer   -> F1_5BS
  BS=5,  multiple-answer -> F3
  BS=13, single-answer   -> F4
  BS=13, multiple-answer -> F4M
        |
        v
FeatureEngine  (skills/feature_engine.py)
  mobility / throughput / inter-freq / interference / PDCCH /
  geometry / aggregator 共同输出 ScenarioFeatures
        |
        v
Phase2ToolChecks + ToolCaller
  phase2_run.py::_apply_phase2_tool_checks() 构造 ToolEvidence
  ToolCaller.augment_features() 复用已有增强逻辑
        |
        v
CandidateGenerator  (skills/candidate_generator.py)
  将 issue score 映射为 C-code 候选
        |
        +-----------------------------+
        |                             |
        v                             v
F2/F3/F4/F4M                    F1/F1_5BS
模板/先验候选                    高风险题可触发 agentic overlay
        |                             |
        +-------------+---------------+
                      v
FineTunedVerifier，可选
  audit: 只记录 LoRA 答案，最终仍保留 backbone 答案
  gate/always: 当前不推荐
                      |
                      v
result.csv
  ID, Track A, Track B
  本队只参加 Track A，Track B 留空
```

### 1.2 Phase 2 家族路由

| BS 数量 | tag | Family | 答案数 | 当前处理 |
|---|---|---|---|---|
| 4 | single-answer | F1 | 1 | 特征分数 + 高风险 agentic overlay |
| 4 | multiple-answer | F2 | 2 | 确定性 pair template |
| 5 | single-answer | F1_5BS | 1 | 使用 F1 逻辑；当前候选预测 `press_down_tilt`，属于证据支持而非 GT 验证 |
| 5 | multiple-answer | F3 | 4 | 固定 4-action 多选模板 |
| 13 | single-answer | F4 | 1 | `decrease_interfreq` 先验 |
| 13 | multiple-answer | F4M | 2 | 13-BS inter-frequency pair 模板 |

### 1.3 F1 issue score 到动作的映射

| Score 字段 | 主要触发信号 | 映射动作 |
|---|---|---|
| `score_late_handover` | SINR < 0，邻区优势 > 3 dB | `decrease_a3` |
| `score_missing_neighbor` | 最优邻区未在配置中，邻区优势 > 3 dB | `add_neighbor` |
| `score_coverage_hole` | 低 SINR + 低 RSRP，或几何/功率证据支持；忽略常量 `mainlobe=False` | `lift_tilt` / `increase_power` |
| `score_overshoot_overlap` | serving 与 neighbor overlap ratio > 0.3 | `press_down_tilt` / `decrease_power` |
| `score_pdcch_issue` | 下行 CCE 利用率高 | `pdcch_2sym` |
| `score_backhaul_server` | KPI 异常但无线故障证据不足 | `check_server` |
| `score_insufficient_data` | MR 样本数 < 7，或 `Outside_Mainlobe_Flag=True` | `insufficient_data` |
| `score_inter_freq_load` | 13-BS 场景中 PRB 利用率高 | `decrease_interfreq` |

### 1.4 Agentic Overlay

`skills/agentic_overlay.py` 是受控 agentic 层。它不是完全开放的 22 工具 ReAct，而是在确定性 backbone 之后，只对高风险题触发：

- 最多 2 次 LLM 调用。
- 最多 1 次额外工具调用。
- 每个 branch 有工具白名单。
- LLM 只输出 diagnosis JSON，不直接输出 C-code。
- 本地 mapper 将 diagnosis 映射回当前题目的 C-code。
- 任意失败都回退到 deterministic answer。

当前覆盖两个 branch：

```text
lift_tilt_vs_insufficient_data:
  coverage_hole > 0.20 AND insufficient_data > 0.20
  AND (abs(diff) < 0.25
       OR single-answer AND insufficient_data >= 0.65 AND coverage_hole >= 0.20)

add_neighbor_vs_decrease_a3:
  missing_neighbor > 0.20 AND late_handover > 0.20
  AND (abs(diff) < 0.25
       OR single-answer AND missing_neighbor >= 0.25 AND late_handover >= 0.35)
```

工具菜单：

```python
BRANCH_TOOL_MENUS = {
    "lift_tilt_vs_insufficient_data": [
        "judge_mainlobe_or_not",
        "calculate_tilt_angle",
        "get_user_location",
    ],
    "add_neighbor_vs_decrease_a3": [
        "get_neighboring_cells_pci",
        "get_neighboring_cell_rsrp",
        "get_cell_info",
    ],
}
```

### 1.5 2026-05-07 关键修复：mainlobe 极性问题

问题：Phase 2 probe 中 `judge_mainlobe_or_not` 对 coverage-like 场景几乎恒定返回 `Outside_Mainlobe_Flag=False`。旧逻辑把 `False` 当作 `coverage_hole -> lift_tilt` 的强信号，导致 `lift_tilt` 被过度选择。

修复原则：

```python
# 旧逻辑：False 会 boost coverage_hole，Phase 2 中会被常量输出误导
if serving_flag is False:
    sf.score_coverage_hole = max(sf.score_coverage_hole, 0.74)

# 新逻辑：只把 True 当作非恒定信号
if serving_flag is True:
    sf.score_insufficient_data = max(current_insufficient, 0.78)
    sf.score_coverage_hole = min(sf.score_coverage_hole, 0.25)

# False 不再调整，让原始特征分数决定
```

MR 样本数 guard 也同步修复：

```python
if evidence.mr_sample_count < self.mr_sample_threshold \
        and serving_flag is not True:
    sf.score_insufficient_data = max(..., 0.82)
```

28 个高风险 probe 显示该修复稳定：之前被强推 coverage 的样本现在回到 `insufficient_data/0.780` 路径。外部归档中的 `result_full_uncertainfix.csv` 使用了该修复。

---

## 2. 关键文件

| 文件 | Owner | 用途 |
|---|---|---|
| `phase2_run.py` | A | Phase 2 主入口，串联 API、特征、候选、overlay、审计日志 |
| `phase2_audit.py` | A | 提交前审计，检查格式、覆盖率、cardinality、诊断异常 |
| `phase2_select_probe.py` | A | 选择代表性 probe IDs |
| `main.py` | 官方框架 | Phase 2 Environment 和 HTTP 层 |
| `skills/api_data_gatherer.py` | A | 云端 API 数据桥接，默认 4 个核心数据源 |
| `skills/agentic_overlay.py` | A/C | 高风险题受控 2-call LLM loop，消费 B/D 产物 |
| `skills/scenario_parser.py` | B | 原始 JSON/CSV 转 `ParsedScenario` |
| `skills/scenario_router.py` | B | 根据 tag 和 BS 数量路由 family |
| `skills/feature_engine.py` | B | 输出 `ScenarioFeatures` 和 issue scores |
| `skills/candidate_generator.py` | B | 将特征分数映射为 C-code 候选 |
| `skills/tool_caller.py` | B | 工具证据增强 `ScenarioFeatures` |
| `rag_retriever_v2.py` | D | Phase 1/probe 轨迹 RAG 与 memory 检索 |
| `build_sft_v3.py` / `curate_sft.py` / `eval_ckpt.py` | A | SFT 数据构建、LoRA 训练数据准备与 heldout 评估 |

---

## 3. 运行流程

### 3.1 启动模型服务


```bash
CUDA_VISIBLE_DEVICES=0,1 \
NO_PROXY=127.0.0.1,localhost \
/home/huangxiaolin/dify_vllm_uv310/bin/vllm serve /home/huangxiaolin/AITO/model \
  --host 0.0.0.0 --port 8001 --served-model-name qwen \
  --tensor-parallel-size 2 --enable-lora \
  --lora-modules v4=/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track\ A/checkpoints/v4_fast_balanced \
  --max-lora-rank 32 --gpu-memory-utilization 0.82 \
  --max-model-len 8192 --reasoning-parser qwen3 --language-model-only
```

### 3.2 全量 Phase 2 运行

如果 shell 环境里设置了代理变量，运行 live server 请求前应确认请求会走预期网络路径。

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    AGENT_API_KEY=<AGENT_API_KEY> \
    NO_PROXY=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
    no_proxy=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
python phase2_run.py \
  --server_url  https://120.46.145.77/no \
  --test_json   data/Phase_2/test.json \
  --out         results/phase2/current_submission/result_full_uncertainfix_next.csv \
  --model_url   http://127.0.0.1:8001/v1 \
  --llm_policy  audit \
  --agentic_policy high_risk \
  --save_freq   10
```

### 3.3 代表性 probe

```bash
python phase2_select_probe.py \
  --test_json data/Phase_2/test.json \
  --out_ids   results/phase2/probe_selection/probe_ids.txt \
  --per_group 3

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    AGENT_API_KEY=<AGENT_API_KEY> \
    NO_PROXY=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
    no_proxy=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
python phase2_run.py \
  --server_url https://120.46.145.77/no \
  --test_json  data/Phase_2/test.json \
  --ids        results/phase2/probe_selection/probe_ids.txt \
  --out        results/phase2/agentic_overlay_tests/probe_agentic_representative_next.csv \
  --model_url  http://127.0.0.1:8001/v1 \
  --llm_policy audit \
  --agentic_policy high_risk
```

### 3.4 高风险 probe

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    AGENT_API_KEY=<AGENT_API_KEY> \
    NO_PROXY=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
    no_proxy=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
python phase2_run.py \
  --server_url https://120.46.145.77/no \
  --test_json  data/Phase_2/test.json \
  --ids        results/phase2/agentic_overlay_tests/high_risk_ids.txt \
  --out        results/phase2/agentic_overlay_tests/high_risk_agentic_probe_next.csv \
  --model_url  http://127.0.0.1:8001/v1 \
  --llm_policy audit \
  --agentic_policy high_risk
```

### 3.5 提交前审计

```bash
PYTHONDONTWRITEBYTECODE=1 /home/huangxiaolin/AITO/bin/python phase2_audit.py \
  --test_json data/Phase_2/test.json \
  --result_csv results/phase2/current_submission/result_full_uncertainfix_next.csv \
  --diagnostics_jsonl results/phase2/current_submission/result_full_uncertainfix_next.diagnostics.jsonl \
  --out_report results/phase2/current_submission/result_full_uncertainfix_next.audit.json
```

### 3.6 关键 CLI 参数

| 参数 | 作用 |
|---|---|
| `--agentic_policy high_risk` | 只对高风险 F1 分支启用 agentic overlay |
| `--llm_policy audit` | 记录 LoRA/LLM 答案，但最终保留确定性答案 |
| `--llm_policy gate` | 低风险可被模型覆盖；当前不推荐 |
| `--llm_policy always` | 总是使用模型答案；仅 debug |
| `--fetch_signaling` | 拉取 signaling，当前 live 基本为空，默认关闭 |
| `--ids file.txt` | 只运行指定 scenario IDs，用于 probe |
| `--save_freq N` | 每 N 题保存一次中间结果 |
| `--token TOKEN` | 也可使用环境变量 `AGENT_API_KEY`，文档和日志不要明文记录真实 token |

---

## 4. Owner 接口摘要

详细四人分工、时间表和交付物见 `PHASE2_TEAM_DIVISION_CN.MD`。本架构文档只保留技术集成接口：

```text
B -> A/C/D: branch_spec + evidence_schema + tool_menu
C -> A: prompt/CoT/loop package
D -> A/C: retrieve_trajectory_context() + trajectory examples
A -> all: 必要的 probe/full-run diagnostics
```

---

## 5. 当前提交状态

### 5.1 候选归档快照

```text
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/results/phase2/current_submission/result_full_uncertainfix.csv
```

| 指标 | 值 |
|---|---|
| 行数 | 500 / 500 |
| 已回答 | 500 / 500 |
| Hard errors | 0 |
| API errors | 0 |
| Top-candidate score floor | 0.30 |
| 平均延迟 | 约 11 秒/题 |
| 覆盖 family | F1, F1_5BS, F2, F3, F4, F4M |

### 5.2 当前 F1 分布

| Top issue | 数量 | 主要动作 |
|---|---|---|
| overshoot_overlap | 89 | `press_down_tilt` |
| coverage_hole | 102 | `increase_power` 73 + `lift_tilt` 29 |
| insufficient_data | 30 | `insufficient_data` |
| late_handover | 18 | `decrease_a3` |
| missing_neighbor | 2 | `add_neighbor` |
| F1_5BS 独立组 | 28 | `press_down_tilt` |

### 5.3 剩余风险

| 风险 | 判断 | 当前决策 |
|---|---|---|
| F1_5BS 全部 `press_down_tilt` | probe 稳定且有证据支持，但无 Phase 2 GT | 首次提交可接受，leaderboard 后复查 |
| 73 个 coverage_hole -> `increase_power` | 无 mainlobe 证据偏向 `lift_tilt`，Phase 2 GT 未知 | 保留，等待反馈 |
| 7 个 F2 all-zero 本地分数 | probe overlap 低，强改属于猜测 | 保留 |
| LoRA 对 `lift_tilt/insufficient_data` 失败 | heldout 两类 IOU 为 0.0 | 只保留 `audit` |

### 5.4 Agentic Overlay 状态

- 28 个高风险 probe：0 errors，0 overrides。
- 这对当前候选是预期行为：修复后的 deterministic backbone 已经把这些题推到 `insufficient_data`，LLM 也确认相同 diagnosis。
- 当前 overlay 主要是验证层，还不是强纠错层。
- 下一步要加 trajectory RAG 和更多 branch，让它在真正模糊题上有纠错能力。

---

## 6. 相关文档

| 文档 | 用途 |
|---|---|
| `README_CN.MD` | 文档索引和阅读顺序 |
| `PHASE2_PROJECT_CN.MD` | 项目概览和当前候选摘要 |
| `PHASE2_TEAM_DIVISION_CN.MD` | A/B/C/D 职责、时间表和提交预算 |

## 7. 术语

| 术语 | 含义 |
|---|---|
| C-code | 当前 Phase 2 数据中的 `C1` 到 `C24` 选项代码 |
| F1/F2/F3/F4/F1_5BS/F4M | 场景 family，见路由表 |
| ScenarioFeatures | `feature_engine.run_all_skills()` 输出的结构化特征 |
| ToolEvidence | 条件工具调用结果构成的证据对象 |
| ParsedScenario | `scenario_parser` 输出，包含 DataFrames、选项解析和 cell 映射 |
| RankedCandidate | `candidate_generator` 输出的候选 C-code、分数和理由 |
| IOU | 交并比：`|pred ∩ gt| / |pred ∪ gt|` |
| Time discount | 时间折扣：越晚提交/越慢可能得分越低，按官方规则执行 |
| Agentic overlay | 高风险题受控 2-call LLM 诊断层 |
| Audit mode | 记录 LLM 答案但最终不采用 |
| MR data | Measurement Reports，UE 侧无线测量数据 |
| Outside_Mainlobe_Flag | 工具返回：True 表示 UE 在天线主瓣外；False 表示在主瓣内，但 Phase 2 probe 中 False 近似常量，需要谨慎 |
| Worst time | 吞吐最低的时间点，作为后续工具查询锚点 |
| PRB utilization | Physical Resource Block 使用率 |
| CCE utilization | 下行控制信道 CCE 使用率 |
| A3 event | 邻区信号超过服务小区 A3 margin 后触发的切换事件 |
