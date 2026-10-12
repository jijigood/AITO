# Owner D Task 05: Probe Memory And Coverage

## 任务目标

把 Phase 2 probe traces 纳入 memory，并用它补足高风险 branch 的 few-shot 覆盖和边界案例。

## 开始前先读

1. `docs/PHASE2_TEAM_DIVISION.md`
2. `docs/PHASE2_AI_TASK_GUIDE.md`
3. `docs/owner_d/OWNER_D_SKILL.md`
4. `docs/owner_d/TASK_01_TRAJECTORY_SCHEMA.md`
5. `docs/owner_d/TASK_04_PROMPT_SAFE_CONTEXT.md`
6. 与 probe diagnostics 相关的结果文件

## 环境要求

- 所有脚本和检查都在 conda 环境 `tele` 中执行。
- 新依赖只能装到 `tele`。

## 输入

- Phase 2 probe traces / diagnostics
- 高风险 branch 列表
- 当前 trajectory examples 和 snippets

## 输出

- 补充后的 trajectory memory
- 按 branch 分组的 `trajectory_prompt_snippets.md`
- 覆盖改进说明

## 主要工作

- 从 probe traces 提取可复用诊断轨迹。
- 优先补充混淆最严重的 branch。
- 让 snippet 展示工具使用模式和证据比较方式，而不是答案拷贝。
- 明确 probe memory 与训练 memory 的来源差异。

## 文件边界

- 可修改 RAG 相关文档与数据产物
- 必要时小改 `rag_retriever_v2.py`
- 不要在本任务中替代 A 的 full-run 评估职责

## 完成标准

- 高风险 branch 覆盖更完整。
- snippet 可被 C 直接引用。
- 新增 memory 不泄露训练答案映射。

## 建议验证

- 统计每个重点 branch 是否已有对应 snippet。
- 抽查 probe-derived 记录是否符合 schema。
- 检查新增内容是否仍遵守 prompt-safe 约束。
