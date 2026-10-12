# Owner D Task 03: Retrieval And Index

## 任务目标

在现有 `rag_retriever_v2.py` 基础上实现或修正稳定检索逻辑，包括 branch filter、排序规则、可选 index 和 fallback。

## 开始前先读

1. `docs/PHASE2_TEAM_DIVISION.md`
2. `docs/PHASE2_AI_TASK_GUIDE.md`
3. `docs/owner_d/OWNER_D_SKILL.md`
4. `docs/owner_d/TASK_01_TRAJECTORY_SCHEMA.md`
5. `docs/owner_d/TASK_02_BUILD_TRAJECTORY_EXAMPLES.md`
6. `rag_retriever_v2.py`
7. `phase2_run.py`

## 环境要求

- 所有调试、性能检查、索引构建都在 conda 环境 `tele` 中执行。
- 如需 FAISS 或其它依赖，只能装到 `tele`。

## 输入

- 已有的 `StructuredCaseRAG`
- trajectory examples
- B 提供或约定中的 branch / evidence schema

## 输出

- 对 `rag_retriever_v2.py` 的兼容性补丁
- 如适用，索引构建产物或缓存说明

## 主要工作

- 保持现有调用链可用，不要只设计新接口。
- 检索必须 deterministic。
- 支持 branch-aware 或 family-aware 过滤。
- 明确 tie-breaker，避免同输入返回不同结果。
- 如果 FAISS 不可用，必须保留稳定 fallback。

## 文件边界

- 主要修改 `rag_retriever_v2.py`
- 必要时小改 `phase2_run.py` 的调用适配
- 不要在本任务中把重点放到 prompt 文本润色

## 完成标准

- 相同输入返回相同 top-k。
- 不破坏 A 侧当前 `rag.retrieve(...)` 调用方式。
- fallback 可用，异常路径明确。

## 建议验证

- 在 `tele` 环境对同一输入重复检索至少两次，结果应一致。
- 验证 FAISS 有无两种路径都能工作。
- 对高风险 branch 做最小样例检查。
