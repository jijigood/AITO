# Track A Phase 2 项目说明

本文档说明 V4/Track A 在第二阶段的工作展开方式、系统组成、运行流程、当前产物和后续改进方向。该项目只参加 Track A，提交文件中 `Track B` 列保留为空，仅用于满足平台 CSV 格式。

English version: `PHASE2_PROJECT.md`

## 1. 阶段目标

Phase 2 的目标是在 500 个无线网络故障诊断题上，通过主办方云端 Agent Tool Server 获取实时沙箱数据，输出每题的 Track A 选项代码。评分使用答案集合 IOU，并带有时间折扣，因此系统需要同时满足：

- 答案格式稳定：`ID,Track A,Track B`，多答案使用 `|`。
- API 访问稳定：用 bearer token 调云端工具，不直接依赖本地占位数据。
- 推理稳定：低风险题快速确定，高风险题保留工具证据和诊断 trace。
- 时间稳定：尽量保持每题在 5 分钟满分折扣窗口内，实际当前 full run 约 10-12 秒/题。

## 2. 当前目录结构

整理后的关键目录如下：

```text
V4/Track A/
├── phase2_run.py                         # Phase 2 主入口
├── phase2_audit.py                       # 提交前审计
├── phase2_select_probe.py                # 代表样本选择
├── main.py / server.py / utils.py         # 官方 Phase 2 框架与本地参考服务
├── skills/
│   ├── api_data_gatherer.py              # 云端 API 数据桥接
│   ├── agentic_overlay.py                # 高风险题受控 agentic overlay
│   ├── scenario_parser.py                # CSV/选项解析
│   ├── feature_engine.py                 # RF/KPI/配置特征工程
│   ├── candidate_generator.py            # C-code 候选生成
│   ├── tool_caller.py                    # 工具证据增强
│   └── ...
├── rag_retriever_v2.py                   # Phase 1 结构化 RAG
├── results/phase2/
│   ├── current_submission/               # 后续重新生成的运行结果；本地快照已外部归档
│   ├── agentic_overlay_tests/            # 后续重新生成的 overlay probes
│   └── probe_selection/                  # 代表 probe ID
└── archive/                               # 空占位；本地归档在 V4 外部
```

本地运行产物已移动到共享目录外：

```text
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/
```

该归档包含 checkpoints、SFT JSONL、运行结果、diagnostics、旧实验和旧 V3/Phase 1 设计文档。当前 `V4/Track A` 目录用于源码和文档交付。

## 3. 系统组成

### 3.1 数据桥接层

Phase 2 的 `test.json` 中数据字段是占位文本，真实数据来自云端工具。`skills/api_data_gatherer.py` 默认调用 4 个核心 API：

- `get_user_plane_data`
- `get_config_data`
- `get_kpi_data`
- `get_mr_data`

从 user-plane 数据中提取 `worst_time`、`last_time` 和 serving PCI。`--fetch_signaling` 可选调用 `get_signaling_plane_event_log`，但 live probe 显示 signaling 默认为空，所以当前 full run 不启用。

### 3.2 解析与特征层

`scenario_parser.py` 将原始 JSON 和管道分隔 CSV 转为 `ParsedScenario`：

- 解析 22+ 个选项，得到 `action_type`、`target_cell`、`cells`。
- 构建 `pci_to_cell_id`、`cell_id_to_pci`。
- 推断 cell role：primary / secondary / tertiary。

`feature_engine.py` 输出 `ScenarioFeatures`：

- throughput 定位低速窗口。
- mobility 判断 serving/neighbor、A3 margin、missing neighbor。
- inter-frequency、capacity、transport、geometry、power 等技能给出 issue score。
- 合成 `late_handover`、`missing_neighbor`、`coverage_hole`、`insufficient_data` 等分数。

### 3.3 候选生成层

`candidate_generator.py` 将诊断分数映射到具体 C-code：

- 单选题输出若干 `RankedCandidate`。
- 多选题输出 `AnswerSet`。
- F3/F4/F4M 使用强 family prior。
- F1/F1_5BS 保留 `insufficient_data`、`lift_tilt`、`decrease_a3`、`add_neighbor` 等高风险候选。

### 3.4 工具证据增强层

`phase2_run.py` 中 `_apply_phase2_tool_checks()` 会在规则特征后补充条件工具证据：

- `judge_mainlobe_or_not`：覆盖/主瓣边界。
- `calculate_overlap_ratio`：overlap/interference 边界。
- MR 样本数：`insufficient_data` 判断。

这些证据会写入 diagnostics，供审计和后续 agentic overlay 使用。

### 3.5 受控 Agentic Overlay

`skills/agentic_overlay.py` 是当前新增的高风险受控 agent loop。它不是全量 ReAct，而是只在 high-risk 分支触发：

- `lift_tilt_vs_insufficient_data`
- `add_neighbor_vs_decrease_a3`

每题最多 2 次 LLM 调用，最多 1 次额外工具调用。LLM 不能自由调用全部工具，只能从 branch 白名单中提出 JSON 格式工具请求。工具参数会由本地 evidence 兜底或覆盖，避免 copied PCI 或无效参数。LLM 最终输出 diagnosis JSON，本地 mapper 再映射到 C-code。

当前测试结论：overlay 工程稳定，28 个 high-risk 样本全部完成、无 JSON/LLM/API 错误；但 agentic answer 与 deterministic answer 全部一致，目前更像确认层，还不是纠错层。

### 3.6 LLM / RAG / 微调

当前模型服务：

- Endpoint: `http://127.0.0.1:8001/v1`
- Served name: `qwen`
- Backing path: `/home/huangxiaolin/AITO/model`
- Verified model: `Qwen3.5-35B-A3B`

LoRA `v4_fast_balanced` 可以服务，但 heldout 显示 `lift_tilt` 和 `insufficient_data` 失败，因此不作为无条件 reranker。当前策略：

- deterministic backbone 是 authority。
- LoRA 只适合 audit/gate。
- 后续应建设 trajectory RAG：检索“证据模式 + 诊断动作 + 目标小区”，而不是检索旧 C-code。

## 4. Phase 2 主运行流程

每个 scenario 的主流程：

```text
test.json scenario
  -> APIDataGatherer.gather()
  -> parse_scenario()
  -> run_all_skills()
  -> phase2 MR/config fallback
  -> targeted tool checks
  -> generate_candidates()
  -> optional agentic_overlay(high_risk)
  -> optional verifier/arbitrator
  -> CSV row + diagnostics JSONL
```

## 5. 当前推荐提交

当前推荐提交文件位于：

```text
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/results/phase2/current_submission/result_full_uncertainfix.csv
```

配套文件：

```text
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/results/phase2/current_submission/result_full_uncertainfix.audit.json
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/results/phase2/current_submission/result_full_uncertainfix.diagnostics.jsonl
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/results/phase2/current_submission/presubmission_risk_report.md
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/results/phase2/current_submission/lora_vllm_eval_report.md
```

归档前这份候选通过 500/500 full audit，0 hard errors，0 API errors，0 low-confidence rows。`Track B` 列保持空白。候选快照现在位于 V4 外部本地归档，不在共享目录内。

## 6. Agentic Overlay 测试产物

```text
results/phase2/agentic_overlay_tests/agentic_overlay_test_report.md
results/phase2/agentic_overlay_tests/high_risk_agentic_probe.csv
results/phase2/agentic_overlay_tests/high_risk_agentic_probe.diagnostics.jsonl
results/phase2/agentic_overlay_tests/probe_agentic_representative.csv
results/phase2/agentic_overlay_tests/probe_agentic_representative.diagnostics.jsonl
```

这些文件证明 overlay 已经能稳定运行，但尚未证明能提升分数。后续优化的重点不是继续扩大覆盖，而是让 overlay 在证据明确时有能力纠错。


