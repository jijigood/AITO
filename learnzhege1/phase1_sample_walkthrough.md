# Phase 1 单条样本手把手 walkthrough

这份文档选用一条真实训练集样本，目标不是“展示代码”，而是让你真正学会：

1. 一条题怎么从 JSON 读出来
2. 五张表各自该看什么
3. 怎么把表格里的证据一步步映射到正确答案

选用样本：

- `scenario_id = 0cd874c0-eab3-4963-a51f-28fc5e004ed8`
- `tag = single-answer`
- `ground_truth = C9`
- `C9 = Check test server and transmission issues`

我特意选这条，是因为它很适合教学：

- 它不是“某个参数明显错”的题
- 而是需要你先排除无线侧常见原因
- 最后得出“更像测试链路/外部问题”的结论

这类题非常适合练你对数据集的真正理解。

---

## 1. 先把这条样本读出来

最小读取方式：

```python
import json
import io
import pandas as pd
from pathlib import Path

base = Path("/home/huangxiaolin/.cache/huggingface/hub/datasets--netop--Telco-Troubleshooting-Agentic-Challenge/snapshots/1d6d3533ee03126f67e7f2f6bc23f1c66a192a13/Track A/data/Phase_1")
train = json.loads((base / "train.json").read_text())

item = next(x for x in train if x["scenario_id"] == "0cd874c0-eab3-4963-a51f-28fc5e004ed8")

cfg = pd.read_csv(io.StringIO(item["data"]["network_configuration_data"]), sep="|")
up = pd.read_csv(io.StringIO(item["data"]["user_plane_data"]), sep="|")
sig = pd.read_csv(io.StringIO(item["data"]["signaling_plane_data"]), sep="|")
traffic = pd.read_csv(io.StringIO(item["data"]["traffic_data"]), sep="|")
mr = pd.read_csv(io.StringIO(item["data"]["mr_data"]), sep="|")
```

如果你能把这一步熟练做好，后面大部分本地 parser 工作就已经入门了。

---

## 2. 先看顶层，不要急着看表

这条题的背景是：

- `4` 个候选 cell
- `5G`
- `vehicle-based drive test`

题型是：

- `single-answer`

所以你先知道一件事：

- 最终只需要选一个动作

再看选项，会发现这题的动作大致分成几组：

- 对 `3213751_2` 做功率、A3、PDCCH、tilt、azimuth 调整
- 对 `3244912_3` 做功率、A3、A2/A5、PDCCH、tilt、azimuth 调整
- 加邻区关系
- `Insufficient data`
- `Check test server and transmission issues`

也就是说，这题其实在问你：

- 到底是无线参数问题
- 邻区关系问题
- 还是根本不是无线侧问题

---

## 3. 第一步永远先看 `user_plane_data`

这是最重要的表，因为它告诉你：

- 什么时候掉速
- serving 是谁
- 掉速时 RSRP / SINR / MCS / RB 怎么样
- strongest neighbor 有没有明显更强

这条题里，吞吐序列很清楚：

- 前 3 个点很高：`466.35 / 569.87 / 410.34 Mbps`
- 中间 6 个点掉到 `100 Mbps` 以下：
  - `70.51`
  - `51.10`
  - `89.24`
  - `65.27`
  - `79.39`
  - `53.91`
- 后 3 个点又恢复到：
  - `426.80`
  - `431.26`
  - `471.50 Mbps`

这说明它是一个：

- **短时间窗口内的明显掉速事件**

而不是整段时间都持续很差。

---

## 4. 第二步：掉速时先看 serving cell 是否变了

在这 12 个时间点里，serving PCI 始终都是：

- `546`

也就是配置表里的：

- `gNodeB ID = 3213751`
- `Cell ID = 2`

这告诉你两件事：

1. 整个问题窗口里 UE 一直挂在同一个 serving cell 上
2. 如果真的要怀疑 mobility 问题，也得先看有没有更强邻区值得切

---

## 5. 第三步：掉速时看 serving 质量到底差不差

在吞吐低于 `100 Mbps` 的 6 个点里，serving 观测是这样的：

| Timestamp | Throughput | Serving PCI | Serving RSRP | Serving SINR | Avg MCS | RB |
|---|---:|---:|---:|---:|---:|---:|
| 2024-09-20 22:21:34.219 | 70.51 | 546 | -85.83 | 17.78 | 17.44 | 266 |
| 2024-09-20 22:21:35.683 | 51.10 | 546 | -90.33 | 15.15 | 18.81 | 255 |
| 2024-09-20 22:21:36.489 | 89.24 | 546 | -86.06 | 17.56 | 14.04 | 251 |
| 2024-09-20 22:21:37.633 | 65.27 | 546 | -93.11 | 7.92  | 15.47 | 255 |
| 2024-09-20 22:21:38.455 | 79.39 | 546 | -89.65 | 12.17 | 15.23 | 271 |
| 2024-09-20 22:21:39.806 | 53.91 | 546 | -91.70 | 8.88  | 14.95 | 267 |

最关键的观察是：

- `RSRP` 没有烂到离谱
- `SINR` 虽然有波动，但整体并不呈现“严重失真/强干扰”那种典型坏相
- `Avg MCS` 依然在 `14~19` 左右，不像已经掉到特别差的调制编码
- `RB` 数量也维持在 `251~271`，并没有资源给不上

这一步对判断非常关键。

因为如果是下面这些问题，通常你会看到更强的证据：

- 覆盖问题：RSRP 很差，且通常持续差
- 干扰问题：SINR 会非常糟糕而且有更明确的邻区/覆盖证据
- 调度资源问题：RB / CCE / PRB 之类会更异常

但这条样本里，这些都不明显。

---

## 6. 第四步：看 strongest neighbor 有没有明显比 serving 更强

在这条题里，Top1 邻区大多数时候是：

- `PCI 494`

对应的 Top1 邻区 RSRP 和 serving 的差值大致是：

- `-2.19 dB`
- `-4.57 dB`
- `-7.41 dB`
- `-7.17 dB`
- `-1.67 dB`
- `-3.94 dB`
- 后面有些点甚至出现了 Top1 记录成 `546` 自身的情况

这说明：

- 邻区并没有稳定地显著强于 serving

这点非常重要，因为它几乎直接排除了：

- “晚切换，应该降低 A3 offset”
- “存在明显更强邻区但没切过去”

也就是说，这题不像官方 `traces.json` 里那两条 late handover 示例。

---

## 7. 第五步：看 `network_configuration_data`

这张表里有 `4` 行，对应 `4` 个候选 cell：

- `3211532_1`，PCI `344`
- `3213751_2`，PCI `546`
- `3244912_3`，PCI `494`
- `3274217_4`，PCI `583`

这条题真正和选项关联最强的是两个 cell：

- `3213751_2`
- `3244912_3`

因为很多选项都在改它们。

先看 serving cell `3213751_2`：

- `Transmission Power = 29`
- `Mechanical Downtilt = 0`
- `IntraFreqHoA3Offset = 6`，也就是 `3 dB`
- `IntraFreqHoA3Hyst = 2`，也就是 `1 dB`
- `PdcchOccupiedSymbolNum = 1SYM`
- 邻区列表里已经包含：
  - `344`
  - `494`
  - `583`

再看 `3244912_3`：

- `Transmission Power = 28`
- `Mechanical Downtilt = 5`
- `A2/A5/A3` 参数和其他 cell 基本同一套模式
- `PdcchOccupiedSymbolNum = 1SYM`

### 这张表告诉你什么

1. 邻区关系已经存在  
   所以 `Add neighbor relationship` 这类动作缺乏证据支持。

2. A3 参数没有出现那种“异常偏大、明显卡住 handover”的强信号  
   而且前面你已经看到 strongest neighbor 并不稳定更强。

3. PDCCH 虽然是 `1SYM`，但你还不能因为这个字段存在就直接选 `2SYM`，  
   还要去看 traffic 里的 CCE 证据。

---

## 8. 第六步：看 `traffic_data`，排除拥塞和 CCE 问题

这条题的 cell-level KPI 很关键。

四个 cell 的下行 PRB 利用率只有：

- `16.45%`
- `5.90%`
- `12.01%`
- `10.29%`

这说明：

- 根本没有明显拥塞

再看下行 CCE 利用率：

- `0.2047`
- `0.1846`
- `0.2482`
- `0.2451`

再看下行 CCE 分配成功率：

- `97.732%`
- `93.088%`
- `95.728%`
- `97.758%`

这几项放在一起，结论很明确：

- 没有明显的 CCE / PDCCH 瓶颈证据

所以这就基本排除了：

- `Modify PdcchOccupiedSymbolNum to 2SYM for 3213751_2`
- `Modify PdcchOccupiedSymbolNum to 2SYM for 3244912_3`

也就是：

- `C10`
- `C2`

都不太合理。

---

## 9. 第七步：看 `mr_data`，再确认一遍邻区格局

`mr_data` 里 8 条记录都非常一致：

- serving 一直是 `PCI 546`
- 邻区 1 一直是 `PCI 494`
- 但邻区 1 的 RSRP 大多还是弱于 serving

例如：

- serving `-85.8`，neighbor1 `-93.8`
- serving `-87.3`，neighbor1 `-93.5`
- serving `-84.0`，neighbor1 `-93.4`

只有极个别记录 throughput 很低，但无线强度关系仍然没有显示：

- 一个明显更强的邻区正在等待切换

所以 `mr_data` 再次支持同一个判断：

- 这不像典型 mobility 参数问题

---

## 10. 第八步：`signaling_plane_data` 这题里怎么看

这条题的 `signaling_plane_data` 有一些事件：

- `NREventA2MeasConfig`
- `NREventA3MeasConfig`
- `NREventA3`
- `NRHandoverAttempt`

但这里有一个很值得你学习的现实点：

- 这张表不是每道题都一定是“最可靠主证据”

这条样本里，事件内容里出现的 `ServCellPCI:995`、`TargetPCI:995`，  
和配置表 / 用户面里看到的 PCI 体系并不直接对上。

所以真正成熟的读法不是：

- 看见 signaling 就死抓它不放

而是：

- 把它当辅助证据
- 如果它和 user plane / config 不一致，就降低它在这道题里的权重

这也是一个很真实的数据集使用经验。

---

## 11. 最终为什么答案是 `C9`

现在把前面所有证据合起来：

1. 吞吐确实出现了短时大幅下降
2. 但 serving cell 一直稳定
3. serving RSRP / SINR 并没有差到足以支撑“明显无线侧故障”
4. strongest neighbor 没有稳定显著强于 serving
5. 邻区关系已经存在
6. PRB 不拥塞
7. CCE / PDCCH 没有明显瓶颈
8. 覆盖也没有出现特别强的弱覆盖证据

这时候你就会发现：

- 所有“针对某个 cell 参数做调整”的动作都缺乏强证据

而题目又确实存在掉速现象。

于是最合理的结论反而变成：

- 这次问题更像测试链路、服务端、传输链路或非无线侧因素

所以答案是：

- `C9 = Check test server and transmission issues`

---

## 12. 这道题最有教学价值的地方

它教你的不是“怎么发现某个参数错了”，  
而是：

- **当无线证据不够时，不要硬选无线优化动作**

这是你后面做模型时非常重要的能力。

因为很多模型会犯一个典型错误：

- 只要看见 throughput 低，就强行在 power / A3 / tilt / PDCCH 里选一个

但这道题告诉你：

- 如果无线侧证据链没有闭环，正确答案可能就是“先查非无线侧问题”

---

## 13. 如果你要把这条题变成自己的分析模板，可以直接照这个顺序

1. 先从 `user_plane_data` 找掉速区间
2. 看掉速时 serving PCI 是否变化
3. 看掉速时 serving RSRP / SINR / MCS / RB
4. 看 strongest neighbor 是否明显更强
5. 去 `network_configuration_data` 查 serving / neighbor 配置
6. 去 `traffic_data` 排除拥塞和 CCE 问题
7. 用 `mr_data` 做补充确认
8. `signaling_plane_data` 只作为辅助，不要盲信
9. 最后再决定：
   - 是 mobility
   - 是 coverage
   - 是 capacity / PDCCH
   - 还是非无线侧问题

---

## 14. 一句话总结

这条样本最值得你学的地方是：

- **题目不是让你“总要改点什么无线参数”，而是让你根据证据判断“到底该不该改”**

这正是你们后面做高质量 candidate generation 和 verifier 时，最需要保留下来的判断习惯。
