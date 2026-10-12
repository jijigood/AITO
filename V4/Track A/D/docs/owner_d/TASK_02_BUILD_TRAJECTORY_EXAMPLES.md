# Owner D Task 02: Build Trajectory Examples

## 任务目标

从 Phase 1 training cases 中抽取标准化 trajectory records，生成首版 `trajectory_examples.jsonl`。

## 开始前先读

1. `docs/PHASE2_TEAM_DIVISION.md`
2. `docs/PHASE2_AI_TASK_GUIDE.md`
3. `docs/owner_d/OWNER_D_SKILL.md`
4. `docs/owner_d/TASK_01_TRAJECTORY_SCHEMA.md`
5. `rag_retriever_v2.py`
6. `skills/scenario_parser.py`

## 环境要求

- 所有脚本运行都在 conda 环境 `tele` 中执行。
- 新依赖只能装到 `tele`。

## 输入

- Phase 1 训练数据
- 已定义的 trajectory schema
- 当前 `rag_retriever_v2.py` 中可复用的 feature 提取逻辑

## 输出

- `trajectory_examples.jsonl`
- 如有必要，附带生成脚本或生成逻辑补丁

## 主要工作

- 为每个样本构建 trajectory record，而不是复制训练答案选项文本。
- 尽量复用现有 feature 提取逻辑，避免再造一套不兼容实现。
- 为缺失字段提供稳定默认值或空值策略。
- 重点优先覆盖高风险 branch 和混淆边界。

## 文件边界

- 可修改 `rag_retriever_v2.py`
- 可新增专用构建脚本或文档
- 不要在本任务中同时大改 prompt 格式化

## 完成标准

- 输出文件符合 schema。
- 记录内容描述诊断行为，而不是训练答案 C-code。
- 对缺失值和异常样本有清楚处理。

## 建议验证

- 随机抽样检查若干条 JSONL 记录。
- 检查是否出现 `C1-C22`、`answer_labels`、`gt_c_code` 等泄露字段。
- 如有脚本，至少在 `tele` 环境跑一次最小生成流程。
