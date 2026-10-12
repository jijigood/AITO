# Track A 源码与数据学习手册

这个目录是我为你整理的一套 **Track A 中文学习资料**。

目标不是替代官方代码，而是帮你更快建立下面这几个核心理解：

1. 官方 starter code 到底是怎么串起来运行的
2. Phase 1 数据集每条样本到底长什么样
3. 一道题应该先看哪张表、再看哪张表
4. 不同题型为什么会落到不同优化动作

这套资料适合你在真正改模型、写规则、做特征工程之前，先把“数据和流程心智模型”搭起来。

---

## 目录说明

### 1. 官方源码学习版

- [official_5_4_update_notes.md](/home/huangxiaolin/AITO/learnzhege1/official_5_4_update_notes.md)
  作用：对照 `/home/huangxiaolin/AITO/5.4`，说明 5.4 相比之前学习版真正需要补的内容。
  重点：Phase 2 云端 API、token header、`phase2_run.py`、`skills/` 规则流水线、`tag` 字段陷阱。

- [_types_md.py](/home/huangxiaolin/AITO/learnzhege1/_types_md.py)
  作用：讲清楚官方数据结构和 tool-calling 协议。
  重点：`Scenario`、`Task`、`ScenarioData`、`ToolCall` 到底分别在表达什么。

- [utils_md.py](/home/huangxiaolin/AITO/learnzhege1/utils_md.py)
  作用：讲清楚数据加载、时间切片、答案抽取、本地辅助评分。
  重点：`load_scenarios()`、`get_fields_at_time()`、`extract_answer()`、`compute_score()`。

- [server_md.py](/home/huangxiaolin/AITO/learnzhege1/server_md.py)
  作用：讲清楚官方工具服务器。
  重点：`X-Scenario-Id`、五张表如何被封装成 API、每个 endpoint 在查什么。

- [main_md.py](/home/huangxiaolin/AITO/learnzhege1/main_md.py)
  作用：讲清楚 agent 主循环。
  重点：模型如何拿 `/tools`、如何发 tool call、如何生成 `result.csv`。

- [phase2_run_md.py](/home/huangxiaolin/AITO/learnzhege1/phase2_run_md.py)
  作用：讲清楚 5.4 新增的 Phase 2 deterministic runner。
  重点：它如何绕过 LLM 主循环，按“拉数据 -> 解析 -> 特征 -> 候选 -> 提交”直接生成答案。

- [logger_md.py](/home/huangxiaolin/AITO/learnzhege1/logger_md.py)
  作用：讲清楚日志初始化。
  重点：stdout/stderr/file handler 怎么配、有哪些容易忽略的小细节。

### 2. 5.4 新增 skills 学习版

- [skills/README.md](/home/huangxiaolin/AITO/learnzhege1/skills/README.md)
  作用：作为 `skills/` 学习版目录总览。
  重点：每个 skill 文件在 Phase 2 流水线中的位置。

- [skills/scenario_router_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/scenario_router_md.py)
  作用：讲清楚 `tag + num_base_stations -> F1/F2/F3/F4` 的题型路由。

- [skills/scenario_parser_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/scenario_parser_md.py)
  作用：讲清楚五张 CSV、每题 option label、cell role、PCI 映射如何被标准化。

- [skills/api_data_gatherer_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/api_data_gatherer_md.py)
  作用：讲清楚 Phase 2 如何从云端 API 拉回原始表格，并重建成 Phase 1 风格 raw scenario。

- [skills/feature_engine_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/feature_engine_md.py)
  作用：讲清楚 A-G 七类 skill 如何产生 late handover、coverage、PDCCH、server 等 issue score。

- [skills/candidate_generator_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/candidate_generator_md.py)
  作用：讲清楚如何把 issue score 映射回每题动态变化的 C-code 候选。

- [skills/answer_parser_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/answer_parser_md.py)
  作用：讲清楚答案提取、合法化、排序和 IOU 评分。

### 3. 官方示例轨迹学习版

- [traces_md.md](/home/huangxiaolin/AITO/learnzhege1/traces_md.md)
  作用：讲清楚官方 `examples/traces.json` 在教什么。
  重点：两条示例轨迹分别在演示什么诊断链条，为什么它们对 feature / candidate / explanation 很重要。

### 4. Phase 1 数据集学习版

- [phase1_data_md.md](/home/huangxiaolin/AITO/learnzhege1/phase1_data_md.md)
  作用：系统讲清楚 `Track A/data/Phase_1` 的格式、内容和用途。
  重点：顶层字段、五张内嵌表、训练/测试差异、哪些结构规律最值得建模。

- [phase2_data_md.md](/home/huangxiaolin/AITO/learnzhege1/phase2_data_md.md)
  作用：讲清楚 5.4 `Track A/data/Phase_2` 和 Phase 1 的关键差异。
  重点：云端 API 占位数据、`C1-C24`、Phase 2 family 分布和 safety override。

- [track_a_family_overview.md](/home/huangxiaolin/AITO/learnzhege1/track_a_family_overview.md)
  作用：从全局角度讲清楚 `F1 / F2 / F3 / F4` 四个结构家族。
  重点：每个家族的硬分布、家族规则、动作模板和建模启发。

- [add_neighbor_caveat.md](/home/huangxiaolin/AITO/learnzhege1/add_neighbor_caveat.md)
  作用：解释 `Add neighbor relationship` 标签在这份数据集里的一个重要理解陷阱。
  重点：为什么不能把它机械理解成“配置表里缺少邻区项”。

### 5. 单条样本 walkthrough

- [phase1_sample_walkthrough.md](/home/huangxiaolin/AITO/learnzhege1/phase1_sample_walkthrough.md)
  作用：手把手读一条“不是无线参数问题”的真实样本。
  重点：如何通过排除法得出 `Check test server and transmission issues`。

- [phase1_sample_walkthrough_mobility.md](/home/huangxiaolin/AITO/learnzhege1/phase1_sample_walkthrough_mobility.md)
  作用：手把手读一条标准 `late handover / A3` 正例。
  重点：为什么 strongest neighbor 明显更强、serving stuck、不及时切换，最终应该选 `Decrease A3 Offset threshold`。

- [phase1_sample_walkthrough_pdcch.md](/home/huangxiaolin/AITO/learnzhege1/phase1_sample_walkthrough_pdcch.md)
  作用：手把手读一条 `PdcchOccupiedSymbolNum -> 2SYM` 正例。
  重点：为什么无线信号看起来健康时，CCE allocation success rate 可能成为决定性证据。

- [phase1_sample_walkthrough_coverage.md](/home/huangxiaolin/AITO/learnzhege1/phase1_sample_walkthrough_coverage.md)
  作用：手把手读一条 F2 覆盖组合修复正例。
  重点：为什么 `adjust_azimuth + increase_power` 是覆盖形状和覆盖强度的组合动作。

---

## 推荐学习顺序

如果你想尽快把 Track A 看明白，建议按这个顺序读：

1. [official_5_4_update_notes.md](/home/huangxiaolin/AITO/learnzhege1/official_5_4_update_notes.md)
   如果你是从 5.4 官方包开始看，先读这份，避免把 Phase 1 starter code 和 Phase 2 新 runner 混在一起。

2. [_types_md.py](/home/huangxiaolin/AITO/learnzhege1/_types_md.py)
   先建立“每条题长什么样”的概念。

3. [phase1_data_md.md](/home/huangxiaolin/AITO/learnzhege1/phase1_data_md.md)
   再建立“这份数据集到底给了我什么”的概念。

4. [phase2_data_md.md](/home/huangxiaolin/AITO/learnzhege1/phase2_data_md.md)
   接着看 5.4 Phase 2 为什么不能再把本地 JSON 当成自包含数据集。

5. [track_a_family_overview.md](/home/huangxiaolin/AITO/learnzhege1/track_a_family_overview.md)
   这一步先建立全局结构，不要一开始就陷进单题细节。

6. [skills/README.md](/home/huangxiaolin/AITO/learnzhege1/skills/README.md)
   先把 5.4 新 pipeline 的模块边界看清楚。

7. [skills/scenario_router_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/scenario_router_md.py)
   把 F1/F2/F3/F4 和 5.4 官方 router 对起来。

8. [skills/scenario_parser_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/scenario_parser_md.py)
   理解 5.4 新 pipeline 为什么要先解析 option label 和 cell role。

9. [skills/api_data_gatherer_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/api_data_gatherer_md.py)
   补上 Phase 2 云端 API 数据获取这层。

10. [skills/feature_engine_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/feature_engine_md.py)
   看 A-G 七类 skill 如何把表格信号变成 issue score。

11. [skills/candidate_generator_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/candidate_generator_md.py)
   看 issue score 如何映射成当前题的具体 C-code。

12. [phase2_run_md.py](/home/huangxiaolin/AITO/learnzhege1/phase2_run_md.py)
   最后串起来看 Phase 2 deterministic runner。

13. [utils_md.py](/home/huangxiaolin/AITO/learnzhege1/utils_md.py)
   这一步把“JSON 怎么变成可操作数据”补上。

14. [server_md.py](/home/huangxiaolin/AITO/learnzhege1/server_md.py)
   接着理解官方怎么把数据封装成工具。

15. [main_md.py](/home/huangxiaolin/AITO/learnzhege1/main_md.py)
   然后再看 agent 是怎么调用工具和输出结果的。

16. [traces_md.md](/home/huangxiaolin/AITO/learnzhege1/traces_md.md)
   这一步开始建立“官方认可的诊断思维链”。

17. [phase1_sample_walkthrough.md](/home/huangxiaolin/AITO/learnzhege1/phase1_sample_walkthrough.md)
   先学会“什么时候不要乱改无线参数”。

18. [phase1_sample_walkthrough_mobility.md](/home/huangxiaolin/AITO/learnzhege1/phase1_sample_walkthrough_mobility.md)
   再学会“什么时候可以有把握地判成 mobility / A3”。

19. [phase1_sample_walkthrough_pdcch.md](/home/huangxiaolin/AITO/learnzhege1/phase1_sample_walkthrough_pdcch.md)
   补上 PDCCH / 控制信道容量类问题。

20. [phase1_sample_walkthrough_coverage.md](/home/huangxiaolin/AITO/learnzhege1/phase1_sample_walkthrough_coverage.md)
   补上覆盖形状和功率组合类问题。

21. [add_neighbor_caveat.md](/home/huangxiaolin/AITO/learnzhege1/add_neighbor_caveat.md)
   这一步专门补一个很容易误导规则设计的数据陷阱。

22. [skills/answer_parser_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/answer_parser_md.py)
   最后补答案格式清洗和 IOU 评分。

23. [logger_md.py](/home/huangxiaolin/AITO/learnzhege1/logger_md.py)
   最后补上日志层，不影响主线，但有助于你后面调试。

---

## 读完之后你应该能掌握什么

如果这套资料你都顺下来，理想状态下你会清楚下面这些事：

### 关于官方代码

- `main.py` 不是解题逻辑本体，它只是一个很薄的 tool-using agent loop。
- `server.py` 不是“智能体”，它只是当前题目的工具 API 封装层。
- `utils.py` 既负责数据入口，也负责答案提取和本地辅助评分。
- `_types.py` 的价值不在算法，而在数据契约。

### 关于数据集

- 每条样本本质上是一个“自包含题目包”。
- 五张表都是直接嵌在 JSON 里的 `|` 分隔字符串。
- `task.options` 每题都不同，`C1~C22` 不是全局固定语义。
- `user_plane_data` 只有 `12` 个时间点，所以这是短窗口诊断，不是长时序预测。
- `network_configuration_data` 的行数和题目规模强相关，是很强的结构先验。
- 整个数据集可以非常稳定地拆成 `F1 / F2 / F3 / F4` 四个结构家族。
- `Add neighbor relationship` 标签不能被简单理解成“配置里没有邻区”。

### 关于做题思路

- 一般先看 `user_plane_data` 定位问题窗口，再看 serving / neighbor / 参数。
- 不是所有低吞吐题都应该改无线参数。
- 当 strongest neighbor 稳定强于 serving，且后续切换恢复时，A3 / mobility 才会非常可信。
- `traffic_data` 和 `mr_data` 很适合做排除法和交叉验证。
- `signaling_plane_data` 很有用，但不一定每题都能当主证据。

---

## 这套资料最适合什么用途

它最适合下面几类工作前的准备阶段：

- 开始写本地 parser 前
- 开始做 feature engineering 前
- 开始做 candidate generation 前
- 开始重写 prompt / verifier / reranker 前
- 带队友快速上手 Track A 时

如果你现在的目标是提分，这套学习资料的意义在于：

- 先让你搞清楚“哪些信号真有用”
- 再去做规则、训练和推理优化

---

## 和你当前项目的关系

你现在自己的工程实现已经明显比官方 starter code 更进一步了。  
所以这套资料不是为了让你回退到官方 baseline，而是为了让你：

1. 更清楚官方原始设计意图
2. 更清楚数据集真正支持哪些分析路线
3. 更容易判断你们现在的 V2 方案到底在哪些地方比官方更强

---

## 下一步最值得继续补的内容

如果你想继续把学习资料补全，我最推荐的后续方向是：

1. 再做一条 Phase 2 API 实战调用记录  
   原因：它有助于你把 `phase2_data_md.py`、`api_data_gatherer_md.py` 和真实云端工具调用串起来。

2. 再做一条 `insufficient_data` 正例  
   原因：它有助于理解“什么时候应该克制，不要强行选无线优化动作”。

---

## 一句话总结

这个目录不是“翻译版源码”，而是一套围绕 **Track A 源码、数据、样本、推理路径** 组织起来的中文学习地图。
