# Phase 1 单条样本手把手 walkthrough：Mobility / A3 正例

这份文档补一条“正例”样本，用来和前一份
[phase1_sample_walkthrough.md](/home/huangxiaolin/AITO/learnzhege1/phase1_sample_walkthrough.md)
形成对照。

前一条样本教你的是：

- 不能一看到低吞吐就乱改无线参数

这一条样本教你的则是：

- 当证据链闭环时，为什么**确实应该调整 handover 参数**

---

## 1. 选用样本

本次样本：

- `scenario_id = 01511464-4a9e-4bfd-a019-b56847d3d394`
- `tag = single-answer`
- `ground_truth = C8`
- `C8 = Decrease A3 Offset threshold for 3257709_1`

为什么选它？

因为它具备一个非常标准的 late handover / A3 场景：

1. 掉速阶段 serving 一直不变
2. strongest neighbor 稳定比 serving 强很多
3. serving 的 SINR 在掉速阶段明显变差
4. 随后 serving 真的切到了 strongest neighbor
5. 切过去以后吞吐立刻恢复

这条题的教学价值非常高。

---

## 2. 先看顶层和选项

这题是：

- `4` 个候选 cell
- `single-answer`

关键选项里有：

- `C4 = Decrease A3 Offset threshold for 3262328_2`
- `C8 = Decrease A3 Offset threshold for 3257709_1`
- `C3 = Increase A3 Offset threshold for 3262328_2`
- `C18 = Increase A3 Offset threshold for 3257709_1`
- `C12 = Add neighbor relationship between 3257709_1 and 3262328_2`
- `C17 = Modify PdcchOccupiedSymbolNum to 2SYM for 3257709_1`
- `C22 = Modify PdcchOccupiedSymbolNum to 2SYM for 3262328_2`
- `C16 = Check test server and transmission issues`

也就是说，这题核心是在几个方向里做判断：

- 是不是 `3257709_1` 的 A3 设置太高
- 是不是 `3262328_2` 的 A3 设置有问题
- 邻区是不是缺失
- 是不是 PDCCH 问题
- 还是不是无线侧问题

---

## 3. 最先看 `user_plane_data`

这一步依然最重要。

### 吞吐整体走势

前 3 个点：

- `455.36`
- `516.32`
- `603.69 Mbps`

中间 5 个点突然掉到 `100 Mbps` 以下：

- `57.29`
- `55.28`
- `67.40`
- `78.66`
- `42.77 Mbps`

然后在 `22:21:39.828` 之后恢复：

- `228.98`
- `598.48`
- `537.90`
- `345.44 Mbps`

这说明：

- 有一个非常明确的“中间掉速窗口”

---

## 4. 掉速时先看 serving 是否变了

低吞吐这 5 个点里，serving PCI 一直都是：

- `385`

也就是配置表中的：

- `3257709_1`

这点很关键，因为它说明：

- 掉速期间 UE 一直 stuck 在 `3257709_1`

而不是在几个小区之间频繁摆动。

---

## 5. 掉速时看 serving 无线质量

低吞吐 5 个点的数据如下：

| Timestamp | Throughput | Serving PCI | Serving RSRP | Serving SINR | Avg MCS | RB |
|---|---:|---:|---:|---:|---:|---:|
| 2024-09-20 22:21:34.866 | 57.29 | 385 | -87.21 | -2.70 | 8.77 | 265 |
| 2024-09-20 22:21:35.149 | 55.28 | 385 | -90.35 | -2.21 | 9.62 | 272 |
| 2024-09-20 22:21:36.338 | 67.40 | 385 | -92.72 | -3.95 | 9.79 | 250 |
| 2024-09-20 22:21:37.491 | 78.66 | 385 | -88.85 | -3.48 | 8.48 | 254 |
| 2024-09-20 22:21:38.249 | 42.77 | 385 | -87.53 | -3.33 | 8.98 | 261 |

### 这一步最关键的观察

1. `RSRP` 已经明显比前面的好时段差了一截
2. `SINR` 掉到了 `-2 ~ -4 dB` 这个很差的区间
3. `Avg MCS` 也明显掉到了 `8~10` 左右
4. 但 `RB` 数量还在 `250+`

这说明问题更像：

- **无线质量恶化**

而不像：

- 调度资源不够
- RB 分配不上

也就是说：

- 这里是“链路质量问题”，不是“资源问题”

---

## 6. 再看 strongest neighbor

掉速阶段 Top1 邻区一直是：

- `PCI 821`

也就是配置表中的：

- `3262328_2`

更关键的是，Top1 邻区相对 serving 的优势非常大：

| Timestamp | Serving RSRP | Neighbor 821 RSRP | 邻区-Serving |
|---|---:|---:|---:|
| 22:21:34.866 | -87.21 | -81.0 | +6.21 dB |
| 22:21:35.149 | -90.35 | -82.0 | +8.35 dB |
| 22:21:36.338 | -92.72 | -85.0 | +7.72 dB |
| 22:21:37.491 | -88.85 | -81.0 | +7.85 dB |
| 22:21:38.249 | -87.53 | -79.0 | +8.53 dB |

这就是最典型的 mobility 证据之一：

- strongest neighbor 稳定明显强于 serving
- 但 UE 仍然挂在旧 serving 上

这时你就应该本能地想到：

- `late handover`
- `A3 threshold too high`

---

## 7. 然后看 `network_configuration_data`

四个 cell 分别是：

- `3234922_4`，PCI `470`
- `3257709_1`，PCI `385`
- `3262328_2`，PCI `821`
- `3274253_3`，PCI `477`

### 重点看 serving cell：`3257709_1`

它的关键配置是：

- `PCI = 385`
- `Transmission Power = 15`
- `Mechanical Downtilt = 13`
- `IntraFreqHoA3Offset = 10`
- `IntraFreqHoA3Hyst = 2`
- `PdcchOccupiedSymbolNum = 1SYM`
- 邻区列表里明确包含：
  - `3262328_504990_821`

### 把 A3 参数翻译成人能理解的量

- `IntraFreqHoA3Offset = 10`  
  这里单位是 `0.5 dB`，所以等于 `5 dB`

- `IntraFreqHoA3Hyst = 2`  
  等于 `1 dB`

合起来你可以粗略理解成：

- handover 触发门槛大约要邻区比 serving 强 `6 dB` 左右

而我们刚才看到，低吞吐阶段 strongest neighbor 的优势在：

- `+6.21 ~ +8.53 dB`

这就非常接近官方 `traces.json` 里那种典型套路了：

- 参数门限偏高
- UE 被卡在较差 serving 上
- 切换触发不够及时

### 为什么不是 `3262328_2` 的 A3 有问题

因为当前问题阶段的 serving 是：

- `3257709_1`

真正应该被优化的是：

- 当前 serving cell 的切换门限

而不是目标小区的 A3。

所以在 `C4` 和 `C8` 之间，方向应当指向：

- `3257709_1`

也就是 `C8`。

---

## 8. 排除“邻区关系缺失”

这题里 `3257709_1` 的配置表已经明确写着邻区：

- `3262328_504990_821`

所以：

- `C12 = Add neighbor relationship between 3257709_1 and 3262328_2`

缺乏证据支持。

这一步也非常重要，因为真实比赛里：

- `missing neighbor`
- `late handover`

这两个家族很容易被混淆。

它们的关键区别之一就是：

- 邻区在不在配置里

这题里：

- 邻区已经在配置里

所以更像：

- **有邻区，但门限不合理**

而不是：

- **根本没配邻区**

---

## 9. 排除“PDCCH / CCE 问题”

虽然配置表里 `PdcchOccupiedSymbolNum = 1SYM`，  
但不能只看见这个字段就去选：

- `C17`
- `C22`

还要看 `traffic_data`。

这条题的 cell-level KPI 里：

- Downlink PRB utilization 大约 `5% ~ 19%`
- Downlink CCE utilization 大约 `0.17 ~ 0.29`
- Downlink CCE Allocation Success Rate 大约 `91% ~ 95%`

它们没有呈现出那种非常典型的：

- CCE 紧张
- PDCCH 明显瓶颈

所以这题更不像：

- 调度/控制信道资源不足

而更像：

- 无线链路 stuck on wrong serving cell

---

## 10. `mr_data` 再做一次交叉确认

`mr_data` 很支持这个判断。

例如第一条 MR：

- serving `385 = -86.1 dBm`
- neighbor1 `821 = -80.7 dBm`
- throughput `7.3 Mbps`

还有另一条：

- serving `385 = -87.8 dBm`
- neighbor1 `821 = -83.1 dBm`
- throughput `86.2 Mbps`

也就是说，MR 摘要里也反复在说同一件事：

- `821` 往往比当前 serving `385` 更强

这和 user plane 的时序证据是对得上的。

---

## 11. 什么时候真正发生了 handover

最漂亮的证据在这里：

低吞吐窗口最后一个点是：

- `2024-09-20 22:21:38.249`
- serving 还是 `385`
- throughput `42.77 Mbps`

下一条就变成：

- `2024-09-20 22:21:39.828`
- serving 变成 `821`
- throughput 回升到 `228.98 Mbps`

再后面：

- `598.48 Mbps`
- `537.90 Mbps`

这几乎是教科书式现象：

1. 先 stuck 在较差 serving 上
2. strongest neighbor 明显更强
3. 直到后来终于切过去
4. 切过去以后吞吐恢复

这正是：

- **handover 太晚**

而不是：

- 永远切不过去
- 或者功率/方位角/下倾角本身就严重错误

---

## 12. `signaling_plane_data` 怎么看

这里的 signaling 事件也给了辅助证据：

- `NREventA3`
- `NRHandoverAttempt`

并且它们就发生在掉速窗口尾部附近。

虽然这个数据集里 signaling 的 PCI 编号有时和 user plane / config 不完全对齐，  
所以我不建议把它当唯一主证据，  
但在这条题里它至少在“现象层面”支持了同一个方向：

- 确实出现了 A3 / handover 相关事件

所以这张表在这道题里适合做：

- **辅助佐证**

而不是：

- 主判据

---

## 13. 为什么最终答案是 `C8`

现在把证据串起来：

1. 中间 5 个点吞吐突然降到 `100 Mbps` 以下
2. 掉速阶段 serving 一直卡在 `3257709_1`（PCI `385`）
3. 掉速阶段 strongest neighbor `3262328_2`（PCI `821`）稳定强于 serving `6~8 dB`
4. 掉速阶段 serving `SINR` 已经掉到负值，链路质量明显变差
5. 邻区关系已经存在，不是 missing neighbor
6. PRB / CCE 没有明显拥塞型证据，不像 PDCCH 问题
7. serving cell 的 A3 offset 是 `5 dB`，偏高，容易导致 handover 触发过晚
8. 后面 serving 真的切到 `821`，而且吞吐恢复

所以最合理的动作就是：

- 降低当前 serving cell `3257709_1` 的 A3 offset

也就是：

- `C8 = Decrease A3 Offset threshold for 3257709_1`

---

## 14. 这条题最值得你学的东西

这条题教你的核心不是“A3 是什么”，  
而是：

- **什么时候可以有把握地说，这是 late handover，而不是别的问题**

一个很实用的判断模板是：

1. throughput 明显掉
2. serving 不变
3. strongest neighbor 稳定明显更强
4. serving SINR 明显变差
5. 邻区已经存在
6. 后续真的切过去并恢复

如果这 6 条基本都满足，那么：

- `Decrease A3 Offset threshold`

通常就是一个非常有力的候选。

---

## 15. 和上一条“非无线侧问题”样本对照着看

上一条样本里：

- strongest neighbor 并不明显强
- 无线侧证据链不闭环
- PRB / CCE 也没明显异常
- 所以最后选的是 `Check test server and transmission issues`

这一条样本里：

- strongest neighbor 稳定更强
- serving stuck
- SINR 变差
- 后续切换后恢复
- 所以最后选的是 `Decrease A3 Offset threshold`

这两个样本一起看，会非常有帮助，因为你会真正理解：

- 不是所有低吞吐题都该往 A3 上靠
- 但当 A3 证据齐了，它又会非常明显

---

## 16. 一句话总结

这条样本是一个非常标准的“晚切换正例”：

- 旧 serving 挂太久
- 更强邻区早就出现
- UE 没及时切过去
- 切过去以后吞吐恢复

所以正确动作不是加功率、改 PDCCH、补邻区、查服务器，  
而是：

- **降低当前 serving cell 的 A3 Offset threshold**
