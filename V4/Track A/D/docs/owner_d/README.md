# Owner D Task Pack

本目录是 Owner D 的仓库内工作入口，供后续 AI 直接读取并执行任务。

## 阅读顺序

1. `docs/PHASE2_TEAM_DIVISION.md`
2. `docs/PHASE2_AI_TASK_GUIDE.md`
3. `docs/owner_d/OWNER_D_SKILL.md`
4. 下面对应的任务文档
5. 相关代码文件

## 环境要求

- 所有命令都在 conda 环境 `tele` 中执行。
- 推荐命令前缀：
  `source /home/jett/miniconda3/etc/profile.d/conda.sh && conda activate tele && <command>`
- 新依赖只能装到 `tele` 环境。

## 任务拆分

- `TASK_01_TRAJECTORY_SCHEMA.md`
  目标：定义稳定 schema 和字段语义，作为后续所有 D 任务的基础。
- `TASK_02_BUILD_TRAJECTORY_EXAMPLES.md`
  目标：从 Phase 1 训练样本生成标准化 `trajectory_examples.jsonl`。
- `TASK_03_RETRIEVAL_AND_INDEX.md`
  目标：在 `rag_retriever_v2.py` 内实现或修正 deterministic retrieval、branch filter 和 index/fallback。
- `TASK_04_PROMPT_SAFE_CONTEXT.md`
  目标：输出给 C 的安全 prompt context，禁止泄露训练答案。
- `TASK_05_PROBE_MEMORY_AND_COVERAGE.md`
  目标：纳入 Phase 2 probe traces，补高风险 branch 覆盖与 few-shot 质量。
- `TASK_06_INTEGRATION_AND_VALIDATION.md`
  目标：验证 A/C 的接入面，检查调用链、验证结果和残余风险。

## 派发建议

- 如果任务是“先把地基打稳”，优先派发 `TASK_01` 和 `TASK_02`。
- 如果任务是“修当前 RAG 行为”，优先派发 `TASK_03` 和 `TASK_04`。
- 如果任务是“补 leaderboard 高风险分支”，优先派发 `TASK_05`。
- 如果任务是“准备交给 A 集成”，优先派发 `TASK_06`。
