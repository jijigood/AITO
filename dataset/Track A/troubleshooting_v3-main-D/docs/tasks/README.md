# Owner D 任务文档索引（已细分版）

本目录包含 Owner D（RAG 检索与数据分析工具）的所有子任务文档，每个文档对应 **1–2 个函数/方法**，可直接作为独立 agent 工作指令。

## 使用建议

- `D4` 到 `D6` 中，除 `D5a/D5b` 外，几乎都修改 `rag_retriever_v2.py`，**不适合盲并行委托**。
- 推荐串行主链：
  `D4 → D1a → D1b → D2a → D2b → D3a → D3b → D6`
- `D5a → D5b` 是独立支线，改的是 `skills/data_tools.py`，可以在 `D1b` 之后并行推进。
- 当前 `main_v3.py` 仍依赖 `_retrieve_raw()` / `retrieve_and_format()` 旧路径；D 系列任务默认**不迁移主流程消费方**。

---

## 依赖关系图

```
D4（新接口桩，保留旧路径）
  │
  └─► D1a（action_type 字段）
        │
        └─► D1b（hash 校验 + FAISS）
              │
              ├─► D2a（4 个辅助方法）
              │     │
              │     └─► D2b（retrieve 新路径实现）
              │            │
              │            └─► D3b（confusion pair 追加）
              │
              ├─► D3a（confusion index 构建与缓存）
              │     │
              │     └─► D3b
              │
              ├─► D5a（单例 + duplicate/outlier）
              │     │
              │     └─► D5b（confidence/ambiguity/batch）
              │
              └─► D6（16 维扩展，需等 Owner C 的 C4）
```

---

## 任务列表

| 文档 | 修改内容 | 优先级 | 前置依赖 |
|---|---|---|---|
| [D4_raghit_interface_stub.md](D4_raghit_interface_stub.md) | `RAGHit` dataclass + `retrieve()` 空桩 + `_retrieve_raw` 重命名 | 🔴 最高 | 无 |
| [D1a_action_type_field.md](D1a_action_type_field.md) | 复用 `parse_option_label()` + `build_index()` 循环补充 `action_type` + `_action_indices` | 🔴 高 | 无（建议在 D4 后串行合入） |
| [D1b_hash_cache_faiss.md](D1b_hash_cache_faiss.md) | `_sha256_file()` + `_build_faiss_index()` + `build_index()` 缓存命中逻辑 | 🟠 中 | D1a |
| [D2a_utility_methods.md](D2a_utility_methods.md) | `_cosine_sim_to_subset` + `_sf_to_feature_vec` + `_deduplicate` + `_case_to_raghit` | 🔴 高 | D1b |
| [D2b_stratified_retrieve.md](D2b_stratified_retrieve.md) | `_flat_retrieve` + `_stratified_retrieve` + `retrieve()` 新路径实现 | 🔴 高 | D2a |
| [D3a_confusion_build_index.md](D3a_confusion_build_index.md) | `_build_confusion_index()` + `build_index()` 缓存集成 | 🟠 高 | D1b |
| [D3b_confusion_retrieve.md](D3b_confusion_retrieve.md) | `_get_confusion_pairs()` + `retrieve()` 触发条件 | 🟠 高 | D2b + D3a |
| [D5a_singleton_dup_outlier.md](D5a_singleton_dup_outlier.md) | 单例管理框架 + `duplicate_score` + `outlier_score` | 🟠 高 | D1b |
| [D5b_confidence_ambiguity_batch.md](D5b_confidence_ambiguity_batch.md) | `label_confidence` + `ambiguity_score` + `score_all_scenarios` | 🟠 高 | D5a |
| [D6_16dim_feature_extension.md](D6_16dim_feature_extension.md) | `FEATURE_DIM=16` + 工具证据维度扩展 | 🟡 低 | D2a + Owner C 的 C4 |

---

## 修改文件汇总

| 文件 | 涉及任务 |
|---|---|
| `rag_retriever_v2.py` | D4 → D1a → D1b → D2a → D2b → D3a → D3b → D6 |
| `skills/data_tools.py` | D5a → D5b |

> ⚠️ **禁止修改**：`main_v3.py`、`build_sft_v3.py`、`skills/feature_engine.py`、`skills/scenario_parser.py`

---

## 测试文件规范

每个 agent 应在自己的任务文档末尾"验收检查"部分的代码中创建或追加对应的测试函数：

| 测试文件 | 覆盖任务 |
|---|---|
| `tests/test_rag_retriever.py` | D4、D1a、D1b、D2a、D2b、D3a、D3b、D6 |
| `tests/test_data_tools.py` | D5a、D5b |

测试应**优先使用合成索引**（不依赖 `train.json`）。  
只有缓存命中、维度重建、性能预算这类问题，才额外加一条基于真实 `train.json` 的 smoke check。

---

## 成功指标

| 指标 | 目标 |
|---|---|
| 50 个 `lift_tilt` probe 的 top-6 中同类占比 | ≥ 60% |
| 50 个 `insufficient_data` probe 的 top-6 中同类占比 | ≥ 60% |
| 单次 `retrieve()` 端到端时延 | ≤ 200 ms |
| `score_all_scenarios` 2000 条耗时 | ≤ 30 秒 |
| `pytest tests/` 全部通过 | 0 failures |
