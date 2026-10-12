# Owner D Task 06: Integration And Validation

## 任务目标

从 A/C 的接入视角检查 Owner D 产物是否真的可用，包括接口兼容性、最小运行验证和残余风险说明。

## 开始前先读

1. `docs/PHASE2_TEAM_DIVISION.md`
2. `docs/PHASE2_AI_TASK_GUIDE.md`
3. `docs/owner_d/OWNER_D_SKILL.md`
4. `phase2_run.py`
5. `build_sft_v3.py`
6. `eval_ckpt.py`
7. `docs/owner_d/TASK_03_RETRIEVAL_AND_INDEX.md`
8. `docs/owner_d/TASK_04_PROMPT_SAFE_CONTEXT.md`

## 环境要求

- 所有运行验证都在 conda 环境 `tele` 中执行。
- 新依赖只能装到 `tele`。

## 输入

- D 的 schema、examples、retriever、prompt-safe context
- A 的调用链
- C 对 prompt 注入的消费方式

## 输出

- 集成兼容性说明
- 最小验证结果
- 风险与未验证项清单

## 主要工作

- 检查 `phase2_run.py`、`build_sft_v3.py`、`eval_ckpt.py` 是否仍能消费 D 的接口。
- 确认新增接口没有破坏旧调用方式。
- 确认 prompt 注入文本是安全版本。
- 给 A/C 总结可直接使用的入口和限制条件。

## 文件边界

- 可小幅修改调用适配代码
- 不要在此任务中重做 schema 或大规模重写检索算法

## 完成标准

- A/C 侧可以明确知道怎么接入。
- 已做最小运行验证，或明确说明为什么无法验证。
- 风险清单具体，不笼统。

## 建议验证

- 在 `tele` 环境运行最小检索或最小 prompt 组装链路。
- 检查日志或输出中是否仍出现训练答案泄露字段。
- 记录未覆盖的分支和待补数据。
