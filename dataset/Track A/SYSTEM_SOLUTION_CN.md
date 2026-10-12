# Track A 系统方案（中文）

## 1. 方案目标

本方案面向 Track A，目标是在官方约束下构建一个高分、可解释、可调试、可微调的无线网络诊断系统。

总体目标：

1. 在固定 22 个选项中尽可能准确地选出正确动作
2. 满足 Phase 3 的时延要求
3. 兼容 Qwen3.5-35B-A3B 基础模型与后续微调
4. 让系统具备工程可维护性，而不是只靠 prompt 碰运气

---

## 2. 方案核心思想

本题不适合做成“完全自由探索型 agent”。更适合的方式是：

**结构化诊断优先，Qwen 最终决策兜底。**

也就是：

1. 先把 scenario 解析成结构化数据
2. 再用无线诊断技能抽特征、定候选
3. 最后把缩小后的候选交给 Qwen 做最终选择

这样做的原因是：

- 题目模板固定
- 选项空间固定
- 训练分布高度重复
- 多轮工具调用成本较高
- 最终评分对速度敏感

---

## 3. 总体架构

```text
输入 Scenario
    │
    ▼
场景路由 Router
    │
    ▼
本地解析器 Scenario Parser
    │
    ▼
特征引擎 Feature Engine
    │
    ├── 技能层 Skills
    │       ├── 吞吐定位
    │       ├── Mobility/Handover 诊断
    │       ├── Inter-Frequency 诊断
    │       ├── Coverage/Geometry 诊断
    │       ├── Power/Interference 诊断
    │       ├── Capacity/PDCCH 诊断
    │       └── Server/Backhaul 诊断
    │
    ├── 结构化 Case RAG
    │
    ▼
候选生成器 Candidate Generator
    │
    ▼
Qwen 最终选择器 Final Selector
    │
    ▼
低置信度校验 Verifier（可选调用官方工具）
    │
    ▼
答案规范化 Answer Parser
    │
    ▼
result.csv
```

---

## 3.1 当前代码实现对应表

下面这张表说明的是：

- 总体架构里的每一步，在当前代码里是否已经有对应实现
- 对应到哪个文件
- 当前成熟度如何

需要特别说明：

- 当前系统不是“每一步都是一个独立 agent”
- 除了最终的 `Qwen Final Selector` 与低置信度 `Verifier` 以外，大多数步骤本质上是程序模块、规则模块或特征模块

| 架构步骤 | 当前代码位置 | 当前状态 | 说明 |
|---|---|---|---|
| Orchestrator | `main_v2.py` | 已实现 | 负责串联整条 V2 流水线 |
| Scenario Router | `skills/scenario_router.py` | 已实现 | 可将样本分到 `F1/F2/F3/F4` |
| Scenario Parser | `skills/scenario_parser.py` | 已实现 | 已支持本地解析 scenario 内嵌 CSV |
| Feature Engine | `skills/feature_engine.py` | 已实现 | 已实现吞吐、mobility、inter-frequency、geometry、power、capacity、transport 等特征 |
| Skills 层 | `skills/feature_engine.py` | 已实现 | 当前是函数式技能，不是多 agent 结构 |
| Candidate Generator | `skills/candidate_generator.py` | 已实现 | 已支持单选候选、多选组合、F2/F3/F4 的家族偏置 |
| Structured Case RAG | `rag_retriever_v2.py` | 已实现 | 已支持 family-aware 的结构化案例检索 |
| Qwen Final Selector | `main_v2.py` | 已实现 | 已可接入基础 Qwen 或微调 Qwen 做最终答案选择 |
| Verifier | `main_v2.py` | 部分实现 | 已可在低置信度时触发少量工具验证，但策略仍可增强 |
| Answer Parser | `skills/answer_parser.py` | 已实现 | 已支持 `\boxed{}` 提取、排序与 IOU 评估 |
| Fine-tuning Pipeline | `finetune/prepare_data.py`、`finetune/train.py` | 已实现 | 已有数据生成与训练脚本，但仍需继续验证与迭代 |

### 当前代码成熟度判断

如果从“是否存在代码实现”来看：

- 这套总体架构的大部分步骤都已经在代码中出现了

如果从“是否已经达到比赛高分强度”来看：

- 目前属于“主干已经搭好，部分 family 已经比较强，但还在持续优化”的状态

更准确地说：

#### 已经比较成熟的部分

- `Router`
- `Parser`
- `Feature Engine` 的主干
- `F2 / F3 / F4` 的候选逻辑
- `Structured RAG`
- `Answer Parser`

#### 已有实现但仍需继续强化的部分

- `F1` 单选主战场
- `Verifier` 的触发与工具选择策略
- 微调后的 Qwen 与当前结构化流水线的协同效果

### 当前代码与设计图的一个重要区别

虽然设计图里看起来是逐层流动的智能系统，但当前代码更接近：

**一个总编排器 + 一组结构化诊断模块 + 一个 Qwen 最终决策器**

而不是：

**很多独立 agent 分别对话、彼此协商的多 agent 系统**

这其实更适合 Track A，因为：

- 更快
- 更稳
- 更容易调试
- 更容易满足 Phase 3 的时间要求

---

## 4. 核心模块设计

## 4.1 Orchestrator

作用：

- 负责整条推理链路的编排
- 管理每个 scenario 的处理流程
- 汇总输出 `result.csv`

职责：

1. 读取 scenario
2. 调用 router 判断属于哪个家族
3. 调 parser 做本地结构化解析
4. 调 feature engine 和 skills
5. 调候选生成器生成小规模候选
6. 调 RAG 检索相似案例
7. 调 Qwen 输出最终答案
8. 必要时触发 verifier
9. 调 answer parser 做最终标准化

---

## 4.2 Scenario Router

Track A 的数据分布非常适合先分流，再求解。

建议按以下四类家族处理：

1. `F1 = 4 BS + single-answer`
2. `F2 = 4 BS + multiple-answer`
3. `F3 = 5 BS + multiple-answer`
4. `F4 = 13 BS + single-answer`

原因：

- 这四类问题的答案结构不同
- 根因分布不同
- 最优策略不同

当前经验上：

- `F3` 很像固定组合题
- `F4` 很像 inter-frequency 专项题
- `F1` 是最大、最复杂的主战场
- `F2` 主要是两动作组合问题

---

## 4.3 Scenario Parser

作用：

- 将官方 scenario 中嵌入的原始 CSV 字符串解析成 DataFrame 和辅助索引

输入来源：

- `user_plane_data`
- `network_configuration_data`
- `signaling_plane_data`
- `traffic_data`
- `mr_data`

输出内容：

- 统一 DataFrame 结构
- `PCI -> cell_id` 映射
- `cell_id -> PCI` 映射
- 解析后的 22 个 option
- 各 cell 的角色映射（primary / secondary / tertiary）
- 通用访问器，如：
  - 获取最差时间点数据
  - 获取 serving cell 配置
  - 获取邻区配置
  - 获取 traffic / MR 汇总

设计原则：

- 主路径尽量本地解析
- 不依赖大量 HTTP 工具调用

---

## 4.4 Feature Engine

作用：

- 将原始表格转换成可决策的无线诊断特征

建议抽取的核心特征：

### 吞吐特征

- 最差吞吐时间点
- 吞吐下降幅度
- 吞吐恢复情况

### Mobility 特征

- 最差时刻 serving PCI
- serving RSRP / SINR
- strongest neighbor PCI
- neighbor RSRP 相对 serving 的优势值
- A3 margin
- strongest neighbor 是否在配置中

### Inter-Frequency / Load 特征

- 最大下行 PRB 利用率
- 是否有 inter-frequency 负载或切换迹象

### Geometry / Coverage 特征

- 是否可能主瓣外
- 是否有 overlap / overshoot 迹象
- tilt / azimuth 潜在问题

### Capacity 特征

- CCE utilization
- 是否存在 PDCCH 配置瓶颈

### Backhaul / Server 特征

- 无线指标是否健康但吞吐仍很低

最终这些特征会被压缩成若干 issue score，例如：

- `late_handover`
- `missing_neighbor`
- `inter_freq_load`
- `interference`
- `coverage_hole`
- `overshoot_overlap`
- `pdcch_issue`
- `backhaul_server`

---

## 4.5 Skills 设计

Skills 不是独立 agent，而是稳定可复用的诊断模块。

建议至少包括：

### Skill A：吞吐定位

用途：

- 找到故障时间窗
- 找到最差吞吐点

### Skill B：Mobility / Handover

用途：

- 判断是否晚切换
- 判断是否缺邻区关系
- 判断 A3 阈值是否过高

### Skill C：Inter-Frequency

用途：

- 判断是否应触发 `decrease_interfreq`
- 识别 13 BS 场景中的跨频/负载迁移问题

### Skill D：Coverage / Geometry

用途：

- 判断 tilt / azimuth / overlap / overshoot 问题

### Skill E：Power / Interference

用途：

- 区分弱覆盖还是强干扰
- 决定应增功率还是降功率

### Skill F：Capacity / PDCCH

用途：

- 判断是否需要 `PdcchOccupiedSymbolNum = 2SYM`

### Skill G：Transport / Server

用途：

- 判断无线侧正常但业务吞吐异常的场景

### Skill H：Multi-Answer Composer

用途：

- 专门负责多选题的组合生成
- 避免让模型自由拼接多选答案

---

## 4.6 Candidate Generator

作用：

- 将诊断问题映射到少量最终候选动作

设计目标：

- 单选题只给 Qwen `2~5` 个候选
- 多选题只给 Qwen 少量合法组合

为什么重要：

- 如果直接让 Qwen 在 22 选项里硬选，误差会大
- 如果候选层足够强，Qwen 的职责就会从“自由搜索”变成“最终排序”

候选生成逻辑应结合：

- issue score
- target role
- family prior
- option label 解析结果

特别说明：

- `F4` 应单独走 inter-frequency 候选逻辑
- `F2` 应重点考虑常见双动作模板
- `F3` 应重点考虑固定四动作模板

---

## 4.7 Structured Case RAG

RAG 不建议只基于题面文本做。

因为：

- 题目描述高度模板化
- 单纯文本相似度区分度不够

更好的方式是：

### 存储内容

每条训练样本存为 case：

- family
- tag
- answer
- answer family
- 结构化特征摘要
- issue tags
- 简短证据文本

### 检索方式

1. 先按 family 过滤
2. 再按结构化特征向量找近邻

### RAG 的作用

- 提供同类案例先验
- 给 Qwen few-shot 参考
- 提升候选排序稳定性

---

## 4.8 Qwen Final Selector

Qwen 在本方案中的职责不是从零开始分析所有原始 CSV。

它的职责是：

1. 读取结构化证据
2. 读取候选项
3. 参考相似案例
4. 从小候选集里选出最终答案

建议输入内容：

- 当前 family
- 结构化 evidence
- 候选项或候选组合
- 2~3 个相似案例
- 严格输出格式要求

建议输出内容：

- 最终 `\boxed{}` 答案

这样可以减少：

- prompt 长度
- 模型搜索成本
- 自由发挥带来的不稳定性

### 微调后的 Qwen 在哪一步使用

微调后的 `Qwen3.5-35B-A3B` 主要用在这里，也就是：

- 在前面的结构化模块已经完成诊断、候选裁剪和案例检索之后
- 由 Qwen 负责做最终决策

在整条流水线中的位置是：

1. `Router`
2. `Parser`
3. `Feature Engine / Skills`
4. `Candidate Generator`
5. `Structured Case RAG`
6. **微调后的 Qwen Final Selector**
7. `Verifier`
8. `Answer Parser`

这意味着：

- 微调后的 Qwen 不是系统第一步
- 也不是负责原始 CSV 的底层解析
- 它主要负责“根据结构化证据，在少量候选中做最终选择”

### 微调后的 Qwen 接收什么输入

建议输入内容包括：

- 当前 scenario 的 family 信息
- 结构化 evidence
- 候选项或候选组合
- 相似案例检索结果
- 严格输出格式要求

也就是说，微调后的 Qwen 看到的不是“未经处理的全部原始数据”，而是“已经被前面模块整理好的决策上下文”。

### 微调后的 Qwen 输出什么

输出应尽量简洁，核心是最终答案，例如：

- `\boxed{C9}`
- `\boxed{C3|C7}`

后续再由 `answer_parser` 做标准化和合法性检查。

### 为什么把微调 Qwen 放在这一步

这样设计有几个优势：

1. Qwen 不需要从零摸索整个诊断路径
2. 前面的模块已经把问题空间缩小，模型更容易选对
3. 微调目标更清晰，可以直接学：
   `结构化证据 -> 正确答案`
4. 更适合比赛的时延要求，因为不会让模型在大上下文里反复探索

### 微调后的 Qwen 的第二个用途

除了主路径上的最终选择，微调后的 Qwen 还可以用于低置信度重判。

也就是：

1. 系统第一次给出候选与答案
2. 如果置信度低，则进入 `Verifier`
3. `Verifier` 额外调用少量官方工具，补充证据
4. 再把补充后的证据交给 Qwen 重新判断一次

因此，微调后的 Qwen 在系统中的定位可以概括为：

- **主用途：最终选择器**
- **次用途：低置信度复核器**

---

## 4.9 Verifier

作用：

- 在低置信度时，补充少量官方工具调用做验证

适合触发 verifier 的场景：

- geometry / overlap 模糊
- handover 信号不足
- inter-frequency 证据不够明确
- Qwen 未输出合法答案

适合使用的工具：

- `get_signaling_plane_event_log`
- `get_kpi_data`
- `get_mr_data`
- `judge_mainlobe_or_not`
- `calculate_horizontal_angle`
- `calculate_tilt_angle`
- `calculate_overlap_ratio`
- `optimize_antenna_gain`

设计原则：

- 只在低置信度时用
- 不把 verifier 变成主路径

---

## 5. 四类家族的建议解法

## 5.1 F1：4 BS + 单选

这是最大、最复杂的部分。

重点关注：

- late handover
- missing neighbor
- interference
- coverage hole
- PDCCH
- server/backhaul

建议策略：

- 用最完整的 skills 组合
- 候选层决定前 3~5 个可能动作
- Qwen 负责最终选择

## 5.2 F2：4 BS + 多选

本质上更像“二动作组合题”。

重点关注：

- `decrease_power + press_down_tilt`
- `adjust_azimuth + increase_power`

建议策略：

- 用 pair template 生成候选组合
- 禁止模型自由拼接多选答案

## 5.3 F3：5 BS + 多选

这类题分布高度规则。

建议策略：

- 明确构造四动作模板
- 候选组合层尽量直接给出合法四项组合

## 5.4 F4：13 BS + 单选

这类题应作为家族专用模块处理。

建议策略：

- 默认强 prior 指向 `decrease_interfreq`
- 再在两个候选 cell 之间判断谁是目标

---

## 6. 微调方案

建议采用分阶段策略。

### 阶段 1：先做强 baseline

先把以下模块做强：

- router
- parser
- feature engine
- candidate generator
- structured RAG

这一步做好后，即使不微调，系统也应该已经具有不错的候选质量。

### 阶段 2：结构化 SFT

微调数据建议不要只是“原始 CSV + 长工具轨迹”。

更好的形式是：

- 输入：family + evidence + candidates + retrieved cases
- 输出：最终答案

这样模型学的是决策，不是死记原始表格。

### 阶段 3：难例排序训练

可以进一步构造 hard negatives：

- 看起来合理但错误的动作
- 错误的目标 cell
- 错误的多选组合

这样能显著提升候选排序与最终判别能力。

---

## 7. 评估方案

建议本地评估分两层：

### 第一层：候选层评估

指标：

- top1 exact
- correct answer in candidates

意义：

- 先判断结构化链路是否靠谱

### 第二层：端到端评估

指标：

- IOU
- exact match
- 平均时延
- family 维度分数

建议一定做分家族统计：

- F1
- F2
- F3
- F4

---

## 8. 实施优先级

建议优先顺序如下：

1. 对齐官方 IOU 评估
2. 做 Router
3. 做 Parser
4. 做 Feature Engine
5. 做 Candidate Generator
6. 做 Structured RAG
7. 接入 Qwen Final Selector
8. 做 Verifier
9. 再做微调

---

## 9. 最终方案总结

Track A 的推荐系统方案不是“纯工具对话 agent”，而是：

**一个以结构化诊断为核心、以 Qwen 为最终选择器、以 family-aware 策略与 structured RAG 为增强的无线网络优化系统。**

一句话概括：

**先把问题做窄，再让 Qwen 做准。**
