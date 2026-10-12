# Owner D Task 01: Trajectory Schema

## 任务目标

定义并固化 Owner D 的 trajectory schema，确保后续样本构建、检索、prompt 注入都使用同一组稳定字段。

## 开始前先读

1. `docs/PHASE2_TEAM_DIVISION.md`
2. `docs/PHASE2_AI_TASK_GUIDE.md`
3. `docs/owner_d/OWNER_D_SKILL.md`
4. `rag_retriever_v2.py`
5. `skills/scenario_parser.py`

## 环境要求

- 所有检查和调试都在 conda 环境 `tele` 中执行。
- 如果需要装依赖，只能装到 `tele`。

## 输入

- Phase 2 分工文档中定义的 D 侧 trajectory record 字段
- 当前 `rag_retriever_v2.py` 已存在的 feature / hit / prompt 字段
- B 侧未来会提供的 branch / evidence schema

## 输出

- `trajectory_schema.md` 或 `trajectory_schema.json`
- 必要时补充字段映射说明

## 主要工作

- 固化 record 字段名、类型、允许空值规则。
- 明确 `branch`、`key_evidence`、`target_role`、`confidence` 的语义。
- 明确哪些字段来自训练样本，哪些字段允许来自 probe traces。
- 明确 schema 中哪些字段用于检索，哪些字段只用于 prompt 展示。

## 文件边界

- 优先修改或新增文档文件。
- 不要在这个任务里大幅重写检索逻辑。

## 完成标准

- schema 和 `PHASE2_TEAM_DIVISION` 一致。
- 字段语义足够清楚，让后续 AI 不需要猜。
- 明确哪些字段不能在 prompt 中泄露训练答案。

## 建议验证

- 人工检查 schema 是否覆盖 `OWNER_D_SKILL.md` 中要求的全部字段。
- 如果实现了 JSON schema，至少做一次基本格式验证。
