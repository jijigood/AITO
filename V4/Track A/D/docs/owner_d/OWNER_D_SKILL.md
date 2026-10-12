# Skill: Phase 2 Track A - Owner D (Trajectory RAG + Memory)

你是 Phase 2 的 Owner D，负责轨迹检索与记忆层，不负责 prompt wording、RF 规则定义或 LoRA 训练。

## 开始前必须先读

1. `docs/PHASE2_TEAM_DIVISION.md`
2. `docs/PHASE2_TEAM_DIVISION_CN.MD`
3. `docs/PHASE2_AI_TASK_GUIDE.md`
4. `docs/owner_d/README.md`
5. 与 D 相关的现有实现：
   - `rag_retriever_v2.py`
   - `phase2_run.py`
   - `build_sft_v3.py`
   - `eval_ckpt.py`
6. B 提供的 branch / evidence 契约文档；如果还不存在，先按 `PHASE2_TEAM_DIVISION` 中的 D 职责做兼容设计。

## 环境要求

- 所有调试、测试、索引构建、脚本运行都必须在 conda 环境 `tele` 中执行。
- 推荐命令前缀：
  `source /home/jett/miniconda3/etc/profile.d/conda.sh && conda activate tele && <command>`
- 如果需要安装新依赖，必须安装到 `tele` 环境，不要装到 `base` 或系统 Python。
- 安装新依赖后，必须在开发日志或说明中记录安装的包名、用途和安装方式。

## 你的主职责

- 将 Phase 1 training cases 和 Phase 2 probe traces 转成可复用的 trajectory memory。
- 在现有 `rag_retriever_v2.py` 基础上演进，不要平行再造一套独立 RAG 栈，除非现有实现无法兼容且你明确说明原因。
- 为 A/C 提供稳定、可复现、低延迟的检索结果与 prompt 注入文本。
- 产出的是“诊断行为轨迹”，不是训练答案 C-code 的拷贝。

## 现有仓库的真实集成点

- 当前主检索实现位于 `rag_retriever_v2.py`，核心类是 `StructuredCaseRAG`。
- 当前 Phase 2 运行流程优先通过 `rag.retrieve(...)` 拿回命中结果，再在 `phase2_run.py` 中格式化 prompt 文本。
- 如果你实现 `retrieve_trajectory_context(sf, branch, k=3) -> str`，必须满足以下之一：
  - 作为现有 `StructuredCaseRAG` 的兼容封装；
  - 或同步更新调用方，确保 A 可以无缝集成。
- 不允许只定义新接口而不处理现有调用链兼容性。

## 必须遵守的输出契约

### 1. Trajectory record schema

轨迹记录必须与下面字段对齐，允许补充派生字段，但不能破坏这些基础字段的稳定性：

```json
{
  "scenario_id": "str",
  "family": "str",
  "tag": "str",
  "branch": "str",
  "serving_sinr": "float",
  "serving_rsrp": "float",
  "best_neighbor_advantage": "float",
  "neighbor_in_config": "bool",
  "mainlobe_outside": "bool|null",
  "mr_count": "int",
  "tool_sequence": ["str"],
  "key_evidence": ["str"],
  "root_cause": "str",
  "action": "str",
  "target_role": "str|null",
  "confidence": "high|medium|low"
}
```

### 2. Prompt-safe retrieval output

- 检索结果可以保留内部训练标识供调试，但给 C 的 prompt 文本必须是安全版本。
- 以下内容不得出现在 prompt 注入文本中：
  - `gt_c_code`
  - `answer`
  - `answer_labels`
  - `C1` 到 `C22` 这类训练答案编号
  - 任何“正确答案是某个选项”的直接提示
- prompt 文本应描述：
  - 场景类型
  - 关键证据
  - 工具调用顺序
  - root cause / action / target_role
  - 置信度与适用边界

### 3. Deterministic retrieval

- 相同输入必须返回相同结果。
- 如果相似度相同，必须有稳定 tie-breaker，例如：
  - branch/filter 匹配优先
  - similarity 降序
  - scenario_id 升序
- FAISS 是可选实现，不是强制要求；如果 FAISS 不可用，必须有可工作的稳定 fallback。

## 与其他 Owner 的边界

- D 不定义 RF branch 语义，branch / evidence schema 由 B 提供。
- D 不决定 prompt wording，prompt 如何注入由 C 决定。
- D 不负责模型训练、full run、submission，由 A 负责。
- D 需要保证自己的产物足够清晰，让 A 能接、C 能用。

## 交付物

- `artifacts/owner_d/docs/trajectory_schema.md` 或 `artifacts/owner_d/docs/trajectory_schema.json`
- `artifacts/owner_d/data/trajectory_examples.jsonl`
- `artifacts/owner_d/index/trajectory_index.faiss` 或等价索引缓存
- `artifacts/owner_d/docs/trajectory_prompt_snippets.md`
- `retrieve_trajectory_context()` 或兼容现有调用链的等价接口
- 必要时对以下文件提交补丁：
  - `rag_retriever_v2.py`
  - `phase2_run.py`
  - `build_sft_v3.py`
  - `eval_ckpt.py`

## 推荐执行顺序

### P0

- 定义并固化 `trajectory_schema`
- 从 Phase 1 数据生成首版 `artifacts/owner_d/data/trajectory_examples.jsonl`
- 补齐 branch 过滤和 evidence 对齐的检索逻辑
- 输出 prompt-safe 的 trajectory context
- 生成按 branch 分组的 `artifacts/owner_d/docs/trajectory_prompt_snippets.md`

### P1

- 引入或优化 probe trace memory
- 优化排序与加权逻辑
- 扩大高风险 branch 覆盖
- 增加 tool-use few-shot 的边界案例

## 实施要求

- 每次任务先说明本次要修改的目标文件和兼容点。
- 改动前先检查是否会影响 A 的调用链。
- 如果发现现有实现仍向 prompt 暴露训练答案，要优先修复。
- 增加新依赖前，先确认能否用现有库完成；确实需要新增时，只安装到 `tele` 环境。
- 保留简短开发日志到 `results/phase2/trajectory_dev.log` 是推荐做法；如果本次任务没有落盘日志，需在结果说明里交代原因。

## 完成定义

- 轨迹记录与 schema 对齐。
- 检索输出 deterministic。
- prompt 注入文本不泄露训练 C-code。
- A 可以接入，C 可以直接消费。
- 运行、调试、依赖安装均遵守 `tele` 环境约束。
