# Owner D Task 04: Prompt-Safe Context

## 任务目标

把检索命中结果格式化成可供 C 注入 prompt 的安全文本，明确禁止泄露训练答案编号或“正确答案”提示。

## 开始前先读

1. `docs/PHASE2_TEAM_DIVISION.md`
2. `docs/PHASE2_AI_TASK_GUIDE.md`
3. `docs/owner_d/OWNER_D_SKILL.md`
4. `phase2_run.py`
5. `rag_retriever_v2.py`

## 环境要求

- 所有调试都在 conda 环境 `tele` 中执行。
- 新依赖只能装到 `tele`。

## 输入

- retrieval hits
- 当前 `phase2_run.py` 的 RAG prompt 格式化逻辑
- trajectory schema 中允许对外展示的字段

## 输出

- `retrieve_trajectory_context()` 或等价兼容接口
- 对 `phase2_run.py` / `rag_retriever_v2.py` 的安全格式化补丁
- 如需要，`trajectory_prompt_snippets.md`

## 主要工作

- 清理 prompt 文本中的 `gt_c_code`、`answer`、`answer_labels`、`C1-C22`。
- 输出的重点是证据、工具顺序、root cause、action、target role、confidence。
- 保留调试字段给内部日志可以接受，但不应进入 LLM prompt。
- 如新增格式化函数，必须兼容现有调用链。

## 文件边界

- 主要修改 `phase2_run.py` 和 `rag_retriever_v2.py`
- 不要在这个任务里大幅重构检索排序

## 完成标准

- prompt 注入文本不包含训练答案泄露项。
- C 可以直接消费格式化文本。
- A 当前流程仍能调用。

## 建议验证

- 搜索输出文本，确认不包含 `gt_c_code`、`answer_labels`、`Correct answer`、`C1` 等模式。
- 用最小样例检查格式化文本是否仍有诊断价值。
