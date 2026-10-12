# Track A `data/Phase_1` 学习说明

这个文档对应官方数据目录：

- [Track A/data/Phase_1](/home/huangxiaolin/.cache/huggingface/hub/datasets--netop--Telco-Troubleshooting-Agentic-Challenge/snapshots/1d6d3533ee03126f67e7f2f6bc23f1c66a192a13/Track%20A/data/Phase_1)

它不是源码，所以不能像 `*_md.py` 那样逐行注释。  
这份文档的目标是把它讲清楚到“你可以真正开始自己写 parser / feature / eval”的程度。

---

## 1. 这个目录里有什么

目录里只有两个核心文件：

- `train.json`
- `test.json`

它们在本地是软链接，但你使用时可以把它们当成普通 JSON 文件。

样本数量：

- `train.json`: `2000` 条
- `test.json`: `500` 条

这也和比赛 README 里的 Phase 1 设定一致：

- 训练集 `2000`
- 测试集 `500`

---

## 2. 顶层格式：每条样本长什么样

无论是 train 还是 test，每条样本顶层字段都一样：

- `scenario_id`
- `tag`
- `task`
- `answer`
- `context`
- `data`
- `tools`

也就是说，它本质上是一个“题目对象列表”。

如果你结合前面看的 [_types_md.py](/home/huangxiaolin/AITO/learnzhege1/_types_md.py) 一起理解，  
那就是：

- 每条 JSON 样本都会被 `Scenario.model_validate(...)` 校验成一个 `Scenario`

---

## 3. 每个顶层字段分别是什么意思

### `scenario_id`

每道题的唯一 ID。

它的作用主要有两个：

1. 在结果文件里和答案一一对应
2. 在 `server.py` 中通过 `X-Scenario-Id` 指定“当前题目”

---

### `tag`

题型标签。

在 Phase 1 中只有两种：

- `single-answer`
- `multiple-answer`

训练集分布：

- `single-answer`: `1701`
- `multiple-answer`: `299`

测试集分布：

- `single-answer`: `433`
- `multiple-answer`: `67`

这个字段非常重要，因为它直接告诉你：

- 这题应该输出一个答案
- 还是输出多个答案

---

### `task`

题目本身。

里面有三个字段：

- `description`
- `options`
- `allowed_tools`

#### `description`

题干。

整个训练集其实只有两种描述模板：

1. 单选版
2. 多选版

它们共同部分都是：

- 分析 5G drive test 数据
- 定位 throughput 明显下降问题
- 选择优化动作

不同点只在最后一句：

- 单选：`Select the most appropriate optimization solution`
- 多选：`Select two to four possible optimization solutions`

#### `options`

候选项列表。

每条样本都是 `22` 个选项，也就是：

- `C1 ~ C22`

但要特别注意：

- **22 个选项的具体文本不是全数据集固定不变的**
- 训练集中 `2000` 条样本，实际上有 `2000` 套不同的 options 组合

也就是说：

- `C1` 在不同题里不一定表示同一个具体动作对象
- 真正稳定的是“动作类型模式”，而不是“编号绝对语义”

举例：

- 某题的 `C9` 可能是“decrease A3 for 某个 cell”
- 另一题的 `C9` 可能是“modify PDCCH for 另一个 cell”

所以你们系统后面做动作抽象时，一定不能把：

- `C1~C22`

直接当成全局固定类别来理解，必须结合当题的 `options` 文本。

#### `allowed_tools`

这个字段在 Phase 1 全部都是：

- `all`

意思是：

- 这题允许使用 server 暴露的全部工具

---

### `answer`

标准答案。

这里 train 和 test 的差别非常关键：

#### 训练集

训练集里 `answer` 是真实标签，例如：

- `C9`
- `C3|C5`
- `C2|C8|C11|C16`

训练集答案基数分布：

- 单答案：`1701`
- 双答案：`228`
- 四答案：`71`

也就是说，训练集中没有三答案题，只有：

- `1`
- `2`
- `4`

#### 测试集

测试集里 `answer` 不是标签，而是占位符：

- `To be determined`

这点很重要，因为它意味着：

- 测试集 JSON 结构仍然和训练集一致
- 但你不能从 test.json 里拿到真实答案

这也是为什么你在本地跑 test 集时，不能用 `answer` 做真实评估。

---

### `context`

题目的背景信息。

里面有两个字段：

- `description`
- `wireless_network_information`

`wireless_network_information` 里又有：

- `mobility_scenario`
- `network_type`
- `num_base_stations`

在 Phase 1 中，这几个字段几乎是固定的：

- `mobility_scenario = vehicle-based drive test`
- `network_type = 5G`

真正有区分度的主要是：

- `num_base_stations`

它的分布：

训练集：

- `4`: `1684`
- `5`: `71`
- `13`: `245`

测试集：

- `4`: `421`
- `5`: `20`
- `13`: `59`

这个字段在名字上叫“base stations”，  
但从数据实际表现来看，它更接近：

- 当前题里需要考虑的“小区/候选 cell 数量”

因为它和 `network_configuration_data` 的行数是一一对应的。

---

### `data`

这是整个数据集最核心的部分。

里面有：

- `user_plane_data`
- `network_configuration_data`
- `signaling_plane_data`
- `traffic_data`
- `mr_data`
- `notes`
- `collection_method`

最重要的理解是：

- 这些不是文件路径
- 不是嵌套 JSON 数组
- 而是**直接嵌在 JSON 里的 `|` 分隔表格字符串**

也就是说，每条样本本身就是一个“自包含小数据包”。

你拿到一条样本后，不需要再去找别的 CSV 文件，  
直接：

- `io.StringIO(...)`
- `pd.read_csv(..., sep='|')`

就能把内部表格读出来。

#### `collection_method`

Phase 1 中全部都是：

- `Drive testing performed by an engineer`

#### `notes`

在 Phase 1 训练集中，我检查到：

- 非空 `notes` 数量是 `0`

所以这个字段目前基本可以忽略。

---

### `tools`

这是工具元信息，不是原始观测数据。

里面有：

- `description`
- `capabilities`

当前 Phase 1 基本固定为：

- `description = Analytical tools are provided to retrieve the collected data.`
- `capabilities = ['data retrieval']`

它更像元数据说明，不是主要建模信号。

---

## 4. 五张核心内嵌表到底是什么

`data` 里真正有分析价值的，是这五张表：

1. `network_configuration_data`
2. `user_plane_data`
3. `signaling_plane_data`
4. `traffic_data`
5. `mr_data`

---

## 5. `network_configuration_data`

这是配置表。

它是所有“动作选项”和“参数解释”的核心来源。

列一共 `28` 个：

1. `gNodeB ID`
2. `Cell ID`
3. `Longitude`
4. `Latitude`
5. `Mechanical Azimuth`
6. `Mechanical Downtilt`
7. `Digital Tilt`
8. `Height`
9. `Duplex Mode`
10. `PCI`
11. `Band`
12. `DL ARFCN`
13. `BW [MHz]`
14. `TxRx Mode`
15. `Transmission Power`
16. `ARFCN`
17. `InterFreqHoEventType`
18. `CovInterFreqA2RsrpThld [dBm]`
19. `InterFreqA2Hyst [0.5dB]`
20. `CovInterFreqA5RsrpThld1 [dBm]`
21. `CovInterFreqA5RsrpThld2 [dBm]`
22. `IntraFreqHoA3Offset [0.5dB]`
23. `IntraFreqHoA3Hyst [0.5dB]`
24. `IntraFreqHoA3TimeToTrig`
25. `PCell Neighbor Cell (gNodeBID_ARFCN_PCI)`
26. `PdcchOccupiedSymbolNum`
27. `Digital Azimuth`
28. `Max Transmit Power`

### 这张表的作用

它主要用来回答：

- serving / neighbor 小区的静态配置是什么
- handover 参数是什么
- 邻区关系是否存在
- PDCCH 当前是 `1SYM` 还是 `2SYM`
- 功率、方位角、下倾角是否可疑

### 行数规律

这张表的行数不是固定的，但有非常强的规律：

- 最少 `4` 行
- 最多 `13` 行

而且它和 `context.wireless_network_information.num_base_stations` 完全一致：

- `num_base_stations = 4` -> 配置表 `4` 行
- `num_base_stations = 5` -> 配置表 `5` 行
- `num_base_stations = 13` -> 配置表 `13` 行

### 一个很重要的理解

从实际数据看，这张表每一行更像是：

- 一个候选小区 / 扇区 / 诊断对象

而不太像传统意义上“一座基站包含多个 sector 后再嵌套”那种层级结构。

因为行内已经直接给了：

- `gNodeB ID`
- `Cell ID`
- `PCI`

也就是说，对建模来说，你基本可以直接把每一行当成：

- 一个可被选中、可被配置、可被比较的 cell

---

## 6. `user_plane_data`

这是用户面时序表，也是全数据集中最常用的一张表。

列一共 `32` 个。

核心列可以按几组来理解：

### 基础时空信息

- `Timestamp`
- `UE`
- `Longitude`
- `Latitude`
- `GPS Speed (km/h)`

### serving 小区观测

- `5G KPI PCell RF Serving PCI`
- `5G KPI PCell RF Serving ARFCN`
- `5G KPI PCell RF Serving SS-RSRP [dBm]`
- `5G KPI PCell RF Serving SS-SINR [dB]`
- `5G KPI PCell Layer2 MAC DL Throughput [Mbps]`
- `5G KPI PCell Layer1 DL RB Num (Including 0)`

### 调度/链路质量相关

- `CCE Fail Rate`
- `Avg Rank`
- `Grant`
- `Avg MCS`
- `Initial BLER(%)`
- `Residual BLER(%)`

### 邻区观测 Top1~Top5

每个 TopN 都有：

- `PCI`
- `ARFCN`
- `Filtered Tx BRSRP [dBm]`

### 这张表的作用

它主要用来回答：

- 吞吐在什么时候掉下去
- 问题时刻 serving 是谁
- 问题时刻 serving RSRP / SINR 怎么样
- strongest neighbor 是谁
- 邻区比 serving 强多少
- UE 是否一直停留在同一 serving 上

### 行数规律

这张表全数据集固定都是：

- `12` 行
- `32` 列

这意味着每道题都只给你一个很短的时序片段。  
也就是说，这不是长时间 KPI 监控数据，而是：

- 围绕一个问题窗口的短序列观测

这对建模影响很大：

- 你更适合做“局部时段异常诊断”
- 不适合做很长时间尺度的序列模型

---

## 7. `signaling_plane_data`

这是信令事件表。

列只有 `3` 个：

1. `Timestamp`
2. `Event Name`
3. `Event Content`

### 作用

它主要用来辅助判断：

- handover 是否发生
- 某个时刻之前发生了哪些关键事件
- 有没有失败、异常、迟滞之类的现象

### 行数规律

- 最少 `4` 行
- 最多 `22` 行
- 最常见大约 `10` 行

所以它更像：

- 简洁的事件日志

而不是很长的 signaling trace。

---

## 8. `traffic_data`

这是 cell-level KPI 表。

列一共 `14` 个：

1. `Day\Hour`
2. `gNodeB_ID`
3. `Cell_ID`
4. `Uplink PRB utilization(%)`
5. `Downlink PRB utilization(%)`
6. `Uplink PRB Interference(dBm)`
7. `User Uplink Throughput(Mbps)`
8. `User Downlink Throughput(Mbps)`
9. `Downlink Weak Coversge Ratio`
10. `TA>1KM Ratio`
11. `Uplink CCE utilization(%)`
12. `Downlink CCE utilization(%)`
13. `Uplink CCE Allocation Success Rate(%)`
14. `Downlink CCE Allocation Success Rate(%)`

### 作用

它更偏全局/小区级解释，例如：

- PRB 是否拥塞
- CCE 是否紧张
- 覆盖弱区比例高不高
- 干扰或远距离覆盖问题是否明显

### 行数规律

和配置表一样：

- `4 / 5 / 13` 行

也和 `num_base_stations` 一一对应。

所以可以理解为：

- 每个候选 cell 都有一行 KPI 概览

---

## 9. `mr_data`

这是测量报告摘要表。

列一共 `14` 个：

1. `Call_ID`
2. `Serving ARFCN`
3. `Serving PCI`
4. `Serving RSRP(dBm)`
5. `Neighbor 1 ARFCN`
6. `Neighbor 1 PCI`
7. `Neighbor 1 RSRP(dBm)`
8. `Neighbor 2 ARFCN`
9. `Neighbor 2 PCI`
10. `Neighbor 2 RSRP(dBm)`
11. `Neighbor 3 ARFCN`
12. `Neighbor 3 PCI`
13. `Neighbor 3 RSRP(dBm)`
14. `Throughput(Mbps)`

### 作用

它相当于：

- 把若干差体验样本的 serving / neighbor 强度关系做成摘要记录

比较适合用来辅助判断：

- weakest serving vs strongest neighbor 的格局
- 是否存在明显 missing neighbor / late handover / interference 场景

### 行数规律

- 最少 `5` 行
- 最多 `8` 行

不像 user plane 那样是定长 `12` 行。

---

## 10. 训练集和测试集的关键差异

最重要的不是字段差异，而是：

### 相同点

- 结构完全一样
- 五张内嵌表格式完全一样
- `task/options/context/tools` 结构一样

### 不同点

- 训练集 `answer` 是真实标签
- 测试集 `answer = To be determined`

所以：

- 训练集可以做监督训练、规则分析、离线评估
- 测试集只能做推理生成，不能本地验真

---

## 11. 这个数据集在功能上到底支持你做什么

从功能角度看，它其实同时支持三类方法。

### 第一类：纯 tool-using agent

也就是官方 baseline 那条路：

1. 先看 throughput logs
2. 再查 serving / neighbor / config
3. 最后选答案

这条路依赖：

- `server.py`
- `/tools`
- `main.py`

---

### 第二类：本地结构化 parser

因为所有表格都直接嵌在 JSON 中，  
所以你完全可以跳过在线工具循环，直接本地解析：

- `network_configuration_data`
- `user_plane_data`
- `traffic_data`
- `mr_data`

这就是你们后面 V2 路线的基础。

它的优势是：

- 更快
- 更稳定
- 更适合批量特征抽取

---

### 第三类：SFT / explanation-style training

因为每条题都有：

- 题干
- 22 个选项
- 标准答案

你可以把它做成：

- classification 风格数据
- instruction tuning 数据
- CoT / explanation 风格数据

尤其结合官方 `examples/traces.json`，  
还可以进一步构造：

- Evidence / Findings / Recommendations 风格的监督样本

---

## 12. 这个数据集最值得你注意的几个建模事实

### 事实 1：每题 options 都变

这意味着：

- 不能把 `C1~C22` 当成全局固定 label
- 需要先理解每题 options 文本

---

### 事实 2：user plane 只有 12 个时间点

这意味着：

- 问题不是长序列预测
- 本质是短窗口诊断

---

### 事实 3：配置表行数和题型/家族高度相关

训练集里有一个非常强的组合规律：

- `single-answer + 4 rows`: `1456`
- `multiple-answer + 4 rows`: `228`
- `multiple-answer + 5 rows`: `71`
- `single-answer + 13 rows`: `245`

这其实已经非常接近你们后来总结出的 family 结构。

也就是说：

- 题目规模本身就是一个很强的先验特征

---

### 事实 4：没有字段缺失

我检查的结果是：

- train/test 里这五张表都没有缺失

也就是说，在 Phase 1 里你基本不用担心：

- 某题没有 `mr_data`
- 某题没有 `traffic_data`

这对批量 parser 非常友好。

---

### 事实 5：答案分布并不严重塌缩

训练集中 22 个动作总体都出现得比较均匀，  
并不是极端长尾。

例如出现次数较高的有：

- `C14`: `136`
- `C12`: `126`
- `C15`: `125`
- `C11`: `123`

出现较少但也并不极低的有：

- `C8`: `91`
- `C5`: `91`

所以如果模型得分低，通常不是因为“类别极度不平衡”，  
更多还是：

- 候选缩减不准
- target cell / role 判断不准
- family 识别错

---

## 13. 如果你现在要开始真正使用这个数据集，最推荐的读法

建议你按下面顺序动手：

1. 先只读一条 train 样本  
   看清楚顶层结构和五张表怎么嵌在一起。

2. 把五张表都解析成 DataFrame  
   尤其是 `network_configuration_data` 和 `user_plane_data`。

3. 手工对照一条题的 `answer` 和 `options`  
   看正确动作到底落在哪个 cell、哪种参数、哪种方向。

4. 再写你自己的结构化特征抽取  
   比如：
   - throughput 最低时刻
   - serving 是否变化
   - strongest neighbor 与 serving 差值
   - A3 / A2 / A5 / PDCCH / tilt / power 特征

5. 最后再考虑：
   - 规则
   - rerank
   - SFT

---

## 14. 一句话总结

`Track A/data/Phase_1` 本质上不是“零散日志集合”，而是一个高度结构化的、按题目封装好的诊断数据集：

- 每条样本自带题干、选项、标签、背景和五张表
- 可以走官方 tool-using 路线
- 也非常适合你们走本地 parser + feature + candidate 的工程化路线
