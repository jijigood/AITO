# Track A 赛题要求整理（中文）

## 1. 文档说明

本文件基于以下信息整理：

- 官方 `Track A/README.md`
- 官方 `Track A/server.py`
- 官方 `Track A/main.py`、`_types.py`、`utils.py`
- 你在对话中提供的比赛文字说明

如果后续你还想把图片里的额外细节单独补进来，可以继续在本文件上追加。

---

## 2. 赛题目标

Track A 的目标是构建一个能够理解并优化 5G 无线网络的 AI Agent。系统需要根据 drive test 数据，诊断网络故障或性能瓶颈，并从固定的 22 个优化动作选项中选出正确答案。

核心任务不是自由问答，而是：

1. 理解题目给出的网络上下文与遥测数据
2. 调用官方提供的工具接口进行分析或仿真
3. 在 `C1 ~ C22` 中选出 1 个或多个正确动作

---

## 3. Track A 的官方约束

### 3.1 模型约束

- 官方指定基础模型为 `Qwen3.5-35B-A3B`
- 允许微调，包括 `LoRA`、全量微调等
- 不允许替换为其他架构
- 不允许换成不同参数规模的模型

### 3.2 工程约束

- 允许修改：`main.py`
- 可选新增：`skills`、辅助模块、RAG、微调权重
- 不允许修改：`server.py`
- 最终系统必须能输出 `result.csv`

### 3.3 工具与数据约束

- 只能使用公开、开源的语言和工具
- 只能使用比赛提供的数据集
- 不允许使用 AutoML

### 3.4 代码审核约束

- 公榜前 30 名会被要求提交代码
- 收到邮件后需在 48 小时内提交代码
- 账号间不得私下共享代码或信息

---

## 4. 比赛阶段与提交要求

### Phase 1

- 训练集：`2000` 题，有答案
- 测试集：`500` 题，无答案
- 可持续提交 `result.csv`
- 你提供的说明中提到：提交入口在 `4 月 17 日` 开放

### Phase 2

- 新测试集：`500` 题
- 最多提交 `3` 次
- 每个 Track 公榜前 `30` 名进入最终阶段

### Phase 3

- 再使用新的 `500` 题做最终评估
- 需要提交 Agent 设计与微调权重
- 官方会运行你的 `main.py`
- 最终分数同时考虑准确率与用时

---

## 5. 评分规则

### 5.1 答案格式

- 单选题：输出一个选项，例如 `C9`
- 多选题：多个选项之间用 `|` 分隔，例如 `C3|C7`
- 多选答案要求按编号升序排列

### 5.2 官方准确率

Track A 按 `IOU` 评分：

`accuracy = intersection(prediction, ground_truth) / union(prediction, ground_truth)`

这意味着：

- 多报会掉分
- 少报也会掉分
- 多选题不能随便猜很多项

### 5.3 Phase 3 时间折扣

- `< 5 分钟`：100%
- `5 ~ 10 分钟`：80%
- `10 ~ 15 分钟`：60%
- `> 15 分钟`：0%

结论：最终系统不仅要准，还要稳定且足够快。

---

## 6. 官方代码与目录结构的含义

### 6.1 核心文件

- `server.py`：官方工具服务器与仿真接口定义
- `main.py`：官方 baseline agent 入口
- `data/Phase_1/train.json`：训练集
- `data/Phase_1/test.json`：测试集
- `utils.py`：答案抽取、日志、打分辅助
- `_types.py`：Scenario、Task、ToolCall 等结构定义

### 6.2 官方架构含义

从官方 README 可以看出，Track A 的标准运行方式是：

1. `server.py` 暴露数据与仿真接口
2. `main.py` 作为 Agent 编排入口
3. 你可以扩展 `skills`
4. 你可以使用微调后的 Qwen 权重

这说明官方期望的是“Agent + Tools + 可选 Skills + 可选微调”的方案，而不是纯规则系统或纯静态分类器。

---

## 7. 题目数据结构

每个 scenario 通常包含以下部分：

- `scenario_id`
- `tag`
  - `single-answer`
  - `multiple-answer`
- `task`
  - `description`
  - `options`（固定 22 个动作）
- `context`
  - 网络类型
  - 基站数量
  - 场景信息
- `data`
  - `user_plane_data`
  - `network_configuration_data`
  - `signaling_plane_data`
  - `traffic_data`
  - `mr_data`
- `tools`
  - `description`
  - `capabilities`

从本地代码和数据可见：

- `task.allowed_tools = "all"`
- `tools.capabilities = ["data retrieval"]`

虽然样本元数据里写的是 `data retrieval`，但 `server.py` 实际还提供了几何与仿真类工具。

---

## 8. 官方工具接口整理

官方通过 `/tools` 暴露可调用工具。按功能可分为三类。

### 8.1 数据查询类

- `get_throughput_logs`
- `get_cell_info`
- `get_gnodeb_location`
- `get_user_location`
- `get_serving_cell_pci`
- `get_serving_cell_rsrp`
- `get_serving_cell_sinr`
- `get_rbs_allocated_to_user`
- `get_signaling_plane_event_log`
- `get_neighboring_cells_pci`
- `get_neighboring_cell_rsrp`
- `get_all_cells_pci`
- `get_kpi_data`
- `get_mr_data`

### 8.2 几何与覆盖分析类

- `judge_mainlobe_or_not`
- `calculate_horizontal_angle`
- `calculate_tilt_angle`
- `calculate_pathloss`
- `calculate_overlap_ratio`

### 8.3 仿真类

- `optimize_antenna_gain`

结论：

- 官方不仅允许读数，还允许做一定程度的几何分析和天线调整仿真
- 这对 azimuth、tilt、power、overlap 等题型非常重要

---

## 9. 官方 baseline 的特点与局限

### 9.1 官方 baseline 在做什么

官方 `main.py` 本质上是一个：

- 获取 scenario
- 获取 `/tools`
- 让 LLM 反复对话并调用工具
- 最后抽取 `\boxed{}` 答案

的通用 tool-calling loop。

### 9.2 官方 baseline 没有提供的东西

官方原始 Track A baseline 里没有现成的：

- 结构化 `skills/` 目录
- RAG 检索器
- 结构化 parser
- 家族路由器
- 特征工程层
- 候选裁剪器
- 微调数据生成脚本

也就是说，这些都需要参赛者自己设计。

### 9.3 一个重要工程注意点

官方文档说 Track A 评分按 `IOU`，但官方 baseline 自带的本地 `utils.py` 评估逻辑并不完全等价于正式 IOU 评估。

因此建议：

- 自己实现一份严格的本地 IOU 评估
- 不要完全依赖 baseline 的本地分数判断系统优劣

---

## 10. 从赛题要求推导出的系统设计结论

结合官方规则与代码，Track A 系统必须满足以下几点：

### 10.1 既要“像 Agent”，又不能过于自由

官方希望你用 Qwen + tools + skills，但这道题实际上是固定答案空间的结构化诊断问题。因此最优系统应当是“结构化诊断系统 + Qwen 最终决策”，而不是完全自由对话。

### 10.2 最终主路径必须快

因为 Phase 3 有严格时间折扣，系统应尽量减少：

- 多轮 LLM 循环
- 不必要的 HTTP 工具调用
- 重复查询

### 10.3 多选题要特别谨慎

由于评分是 IOU，多选题不是“多报多中”的逻辑，而是“精确组合更重要”。系统应明确控制答案个数和组合合法性。

### 10.4 本地可解释与可调试非常关键

由于前 30 可能要交代码，系统不仅要分高，还要：

- 结构清晰
- 可复现
- 可说明为什么这么选

---

## 11. 对 Track A 的一句话总结

Track A 的本质是：

**在官方规定的 Qwen 基础上，构建一个能够结合遥测数据、工具接口、无线网络知识和候选动作约束的高效诊断 Agent。**

这不是纯聊天任务，而是“结构化无线故障诊断 + 受约束决策”任务。
