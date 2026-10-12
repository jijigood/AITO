# Track A 学习版 `traces.json`

这个文件对应官方示例：

- 原文件：[traces.json](/home/huangxiaolin/.cache/huggingface/hub/datasets--netop--Telco-Troubleshooting-Agentic-Challenge/snapshots/1d6d3533ee03126f67e7f2f6bc23f1c66a192a13/Track%20A/examples/traces.json)

它不是源码，而是两条“官方示范解题轨迹”。

你可以把它理解成：

1. 官方给 agent 的参考解题样例
2. 官方认为“一个合格的 tool-using 诊断过程”应该长什么样
3. 官方在暗示你优先关注哪些观测量、如何把观测量映射到 22 个选项

---

## 1. 这个 JSON 的结构

`traces.json` 顶层是一个 `list`，当前一共只有 `2` 条样例。

每条样例都有这些字段：

- `question`
- `completion`
- `prediction`
- `ground_truth`
- `score`
- `tool_calls`

这几个字段分别在表达什么：

### `question`

就是实际给模型的题目文本。

它包含两部分：

1. 一段自然语言任务说明  
   例如：测速低于 `100 Mbps`，请选择最合适的优化方案，并满足“不能降低覆盖、不能增加过载小区负荷”等约束。

2. 候选项列表  
   也就是 `C1 ~ C22` 的动作选项。

这和 `main.py` 里的做法是一致的：

- `main.py` 会把 `task.description`
- 再加上 `options`
- 拼成最终给模型的 user prompt

---

### `completion`

这是模型最终输出的完整回答。

它通常按三段来写：

- `Evidence`
- `Findings`
- `Recommendations`

最后再给出：

- `\boxed{C...}`

这和 `_types.py` 里那个 `ExpectedOutput` 的设计思路是呼应的。  
也就是说，官方心里理想的 agent 并不是只吐一个 `C9`，而是：

1. 先列证据
2. 再做归因
3. 最后提出动作建议

只是比赛提交时，最终真正用来打分的是 boxed 里的选项编号。

---

### `prediction`

这是系统解析出来的预测答案。

这里有个小细节：

- 样例里写成 `['C9']`、`['C22']`
- 也就是“列表字符串”的样子

而不是直接的：

- `C9`
- `C22`

这说明官方生成这个示例文件时，中间可能走过一层额外包装逻辑。  
它不是特别重要，但你读这个字段时要知道：

- 真正有判别意义的是里面的选项编号

---

### `ground_truth`

标准答案。

这两条样例里分别是：

- 第一条：`C9`
- 第二条：`C22`

---

### `score`

这两条样例都是：

- `1.0`

意思是这两个示例都答对了。

---

### `tool_calls`

这是最值得你重点看的字段。

它不是结构化数组，而是一整段长字符串，里面按顺序串联了：

- 调了哪个工具
- 参数是什么
- 返回结果是什么

所以它本质上是在记录：

- 一条题从“先查什么”
- 到“再查什么”
- 最后“凭什么选这个答案”

的完整诊断轨迹。

这和你们后面做的：

- `tool trace`
- `feature extraction`
- `candidate generation`

这些工作在思路上是完全连起来的。

---

## 2. 这两条样例分别在教什么

### 样例 1

基本信息：

- `prediction = ['C9']`
- `ground_truth = C9`
- 调用了 `5` 次工具

工具顺序：

1. `get_throughput_logs`
2. `get_serving_cell_pci`
3. `get_serving_cell_pci`
4. `get_serving_cell_pci`
5. `get_cell_info`

这个样例非常重要，因为它在演示一种“极简诊断路径”：

1. 先看吞吐曲线，定位问题时间段  
   从大约 `300~400 Mbps` 掉到 `27~60 Mbps`

2. 在问题区间连续检查 serving PCI  
   发现 `21:33:47.5`、`21:33:49.5`、`21:33:51.5` 都还是同一个 PCI `328`

3. 再去查这个 serving cell 的配置  
   拿到：
   - `IntraFreqHoA3Offset = 10`，也就是 `5 dB`
   - `PdcchOccupiedSymbolNum = 1SYM`
   - 邻区关系是存在的

4. 最后得出结论  
   这更像是“晚切换 / cell edge 挂住”问题，而不是：
   - 邻区缺失
   - PDCCH 配置异常
   - 传输功率问题

5. 所以选：
   - `C9 = Decrease A3 Offset threshold for 3225568_1`

#### 这个样例真正想教你的东西

它在暗示：

- 有时你不需要把所有工具都查一遍
- 只要“吞吐掉点 + serving 不变 + A3 偏高 + 邻区已存在”这几个证据凑齐
- 就足够把问题归到 `late handover`

也就是说，这条样例展示的是一种：

- **最小证据闭环**

而不是“把所有观测量都查一遍再决定”。

这对你们后面做候选缩减很有启发：

- 某些 family，其实只靠少数关键特征就可以锁定

---

### 样例 2

基本信息：

- `prediction = ['C22']`
- `ground_truth = C22`
- 调用了 `10` 次工具

工具顺序：

1. `get_throughput_logs`
2. `get_serving_cell_pci`
3. `get_cell_info`
4. `get_user_location`
5. `get_serving_cell_rsrp`
6. `get_serving_cell_sinr`
7. `get_neighboring_cells_pci`
8. `get_neighboring_cell_rsrp`
9. `get_neighboring_cell_rsrp`
10. `get_neighboring_cell_rsrp`

这条样例比第一条更完整，也更“教科书式”。

它的分析链是：

1. 先看吞吐日志  
   `~500 Mbps` 突然掉到 `~40 Mbps`

2. 查问题时刻的 serving PCI  
   serving 是 `PCI 343`

3. 查 serving cell 配置  
   拿到：
   - `IntraFreqHoA3Offset = 10`
   - `IntraFreqHoA3Hyst = 2`
   - 邻区列表中已经包含 `PCI 505`

4. 查用户位置  
   这一步更多是补空间背景，虽然在最终结论里不是决定性证据

5. 查 serving RSRP 和 SINR  
   发现：
   - `RSRP = -83.07 dBm`
   - `SINR = -0.35 dB`

6. 查邻区 PCI 列表  
   得到：
   - `505`
   - `599`
   - `953`

7. 分别查邻区 RSRP  
   得到最强邻区：
   - `PCI 505 = -77 dBm`

8. 比较 serving 和 strongest neighbor  
   结论是：
   - 邻区比 serving 强 `6 dB`

9. 对照 A3 参数  
   当前门限相当于：
   - Offset `5 dB`
   - Hysteresis `1 dB`
   - 总门限 `6 dB`

10. 推出 handover 条件刚好卡住  
   邻区强度差“等于”门限，而不是“严格大于”门限  
   所以切换不触发，UE 被卡在更差的 serving cell 上，SINR 很差，吞吐掉下去。

11. 所以选：
   - `C22 = Decrease A3 Offset threshold for 3216262_3`

#### 这个样例真正想教你的东西

它在演示一条更完整的 mobility 诊断链：

1. 吞吐差
2. serving 识别
3. serving 配置读取
4. serving 质量读取
5. 邻区观测读取
6. strongest neighbor 对比
7. 与 A3 参数联动解释

这其实就是官方最希望 agent 学会的一类套路：

- **现象 -> serving/neighbor 对比 -> 参数解释 -> 动作选择**

---

## 3. 两条样例合起来告诉了你什么

这两个样例虽然都落到 “Decrease A3 Offset threshold” 这个方向，但它们强调的重点不一样。

### 样例 1强调

- 用最少工具快速确认“晚切换”
- serving 一直不变本身就是强证据
- 邻区关系存在，说明问题不是 missing neighbor

### 样例 2强调

- 不只是“怀疑晚切换”，而是把它算清楚
- serving RSRP、neighbor RSRP、SINR、A3 参数要能够互相解释
- 重点是证明“为什么没切过去”

换句话说：

- 样例 1更像快速 heuristics
- 样例 2更像完整论证

---

## 4. 它和 `main.py` / `server.py` 是怎么对应上的

这份 `traces.json` 本质上就是：

- `main.py` 里的多轮工具调用结果
- 再加上模型最终 `completion`

你可以把一条 trace 映射成下面这条链：

1. `main.py` 构造 `question`
2. 模型请求某个 tool
3. `Environment.execute()` 把 tool name 映射到 `server.py` 的 endpoint
4. `server.py` 从当前 scenario 中切数据返回
5. `main.py` 把 tool result 继续塞回 `messages`
6. 模型继续推理
7. 最终产出 `completion`
8. 再由 `extract_answer()` / `extract_answer_all()` 取出 boxed 答案

所以这份示例文件其实是：

- 官方 starter code 真正运行流程的“成品截图”

---

## 5. 这份示例对你们做系统优化有什么价值

价值很大，主要体现在三个层面。

### 第一层：它给了官方认可的分析顺序

你会发现两条样例都不是乱查工具，而是大致遵循：

1. 先定位问题时间点
2. 再看 serving
3. 再看配置或 neighbor
4. 最后做动作选择

这说明如果你们要保留 tool-using 路线，工具调度也最好沿这个顺序设计。

---

### 第二层：它暴露了官方最看重的判据

从这两条样例看，官方特别在意：

- throughput drop 的时间段
- serving cell 是否变化
- A3 threshold 是否偏高
- 邻区是否存在
- strongest neighbor 是否明显强于 serving
- SINR 是否因为 stuck on wrong serving cell 而恶化

这其实已经非常接近你们后面做的 `feature_engine` 要抽的核心信号。

---

### 第三层：它告诉你 baseline 的“解释模板”

两条 `completion` 都在用：

1. Evidence
2. Findings
3. Recommendations

这个模板很适合拿去做：

- SFT 数据构造
- verifier prompt
- explanation-style reranker

哪怕你最后提交只需要 `C9` / `C22`，  
这种结构化解释仍然对模型稳定性很有帮助。

---

## 6. 读这份文件时你最该记住的几个点

1. 这不是训练集样本，而是“官方示范推理轨迹”。
2. 它重点教的是工具调用顺序和证据组织方式。
3. 两条样例都在演示 mobility / handover 问题的诊断。
4. 第一条更偏快速判断，第二条更偏完整论证。
5. `tool_calls` 只是长字符串，不是结构化数组。
6. `completion` 的写法很适合你们后面做 explanation-style SFT。

---

## 7. 如果你后面想继续深挖，最值得做的两件事

1. 把这两条 trace 反向对照到对应 scenario 的原始数据，手工走一遍。  
   这样你会真正建立“工具返回值”和“结论”之间的因果感觉。

2. 把 `tool_calls` 解析成结构化列表，再和你们自己的 V2 feature 对齐。  
   这样你能直接看出：
   - 官方关注了哪些信号
   - 你们有没有漏掉
   - 哪些信号其实可以本地直接算，不必走完整 tool loop

---

## 8. 一句话总结

`traces.json` 的真正价值，不在于那两个答案本身，而在于它把官方认为“合理的诊断思维链”展示出来了：

- 先看吞吐掉点
- 再查 serving / neighbor / 参数
- 最后把无线现象解释成具体优化动作
