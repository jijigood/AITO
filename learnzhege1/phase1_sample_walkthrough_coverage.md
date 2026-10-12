# Phase 1 样本 Walkthrough：覆盖形状 / 方位角 + 功率正例

这份 walkthrough 读一条典型的 F2 多答案覆盖类样本。

标准答案是一个组合：

```text
Adjust the azimuth + Increase transmission power
```

它适合用来理解：

- 为什么多答案题不能当成“随便选两个”
- 覆盖类问题为什么常常需要组合动作
- 什么时候应该考虑方位角和功率，而不是 A3、PDCCH 或 server

---

## 1. 样本基本信息

样本：

- `scenario_id = 056addfa-6d17-4c13-a5cc-faabe3d44b80`
- `tag = multiple-answer`
- `num_base_stations = 4`
- family 视角：`F2 = 4 BS + multiple-answer`
- 标准答案：`C3|C14`

答案文本：

```text
C3  = Adjust the azimuth of 3273494_1 by 50 degrees
C14 = Increase transmission power for 3273494_1
```

目标 cell：

- `3273494_1`
- `PCI = 827`

---

## 2. F2 先验：这不是单动作题

这题的 `tag = multiple-answer`，所以从一开始就不应该只输出一个动作。

在 Track A 的 F2 家族里，常见组合之一就是：

```text
adjust_azimuth + increase_power
```

直觉是：

- 方位角解决“主瓣没对准 UE / 覆盖方向不合适”
- 功率解决“覆盖强度不够”

这比单独调一个参数更像覆盖修复组合。

---

## 3. 先看 throughput：低谷非常明显

关键时间点：

| Timestamp | Serving PCI | RSRP | SINR | Throughput |
|---|---:|---:|---:|---:|
| 22:21:31.904 | 827 | -86.42 | 14.22 | 303.97 |
| 22:21:32.616 | 827 | -86.80 | 15.87 | 573.21 |
| 22:21:34.610 | 827 | -108.80 | 1.56 | 32.72 |
| 22:21:36.111 | 827 | -108.66 | -1.42 | 40.74 |
| 22:21:37.409 | 827 | -104.48 | 0.28 | 31.09 |
| 22:21:38.703 | 827 | -103.64 | -1.23 | 20.14 |
| 22:21:40.531 | 827 | -86.80 | 14.47 | 305.39 |
| 22:21:41.129 | 827 | -91.29 | 13.97 | 576.16 |

这里的模式很有辨识度：

- 吞吐从 `573.21 Mbps` 掉到 `20.14 Mbps`
- 同一窗口里 RSRP 掉到 `-108 dBm` 左右
- SINR 也掉到 `0 dB` 附近甚至负数
- 后面 RSRP/SINR 恢复，吞吐也恢复

这比 PDCCH 题更像覆盖/波束方向问题。

---

## 4. 为什么不是 server / PDCCH

如果是 server 问题，通常会看到：

- RSRP 还不错
- SINR 还不错
- 但 throughput 低

而这题低吞吐时：

```text
RSRP = -108.80 dBm
SINR = 1.56 dB
```

甚至：

```text
RSRP = -108.66 dBm
SINR = -1.42 dB
```

无线信号本身已经明显变差，所以不应优先判 server。

如果是 PDCCH，则通常需要 CCE allocation success rate 明显异常。  
但这题目标 cell `3273494_1` 的 traffic 指标并不支持控制信道主因：

| 指标 | 值 |
|---|---:|
| Downlink PRB utilization | 5.9324 |
| Downlink CCE utilization | 0.1322 |
| Downlink CCE Allocation Success Rate | 96.051 |

PDCCH 看起来健康。

---

## 5. 再看目标 cell 配置

目标 cell `3273494_1 / PCI 827` 的配置：

| 字段 | 值 |
|---|---:|
| Mechanical Azimuth | 307 |
| Mechanical Downtilt | 4 |
| Height | 23.2 |
| Transmission Power | 25 |
| PdcchOccupiedSymbolNum | 1SYM |

答案要求：

```text
Adjust azimuth by 50 degrees
Increase transmission power
```

这说明官方认为主问题不是控制信道，而是覆盖形状和覆盖强度。

---

## 6. 邻区观测也支持“覆盖整体变弱”

在低吞吐窗口，不只是 serving RSRP 弱，邻区也整体弱。

例如：

| Timestamp | Serving RSRP | Top1 PCI | Top1 RSRP | Top2 PCI | Top2 RSRP |
|---|---:|---:|---:|---:|---:|
| 22:21:34.610 | -108.80 | 507 | -111 | 132 | -119 |
| 22:21:36.111 | -108.66 | 507 | -111 | 132 | -116 |
| 22:21:38.703 | -103.64 | 507 | -109 | 132 | -116 |

这和“有一个强邻区但切不过去”的 late handover 不一样。

如果是 late handover，你通常会看到：

- best neighbor 明显强于 serving
- A3 条件相关证据更突出
- 后续切换恢复更像主线

但这里更像 UE 走到了目标 cell 覆盖不理想的位置，整体信号质量变差。

---

## 7. 为什么是两个动作

单独增加功率可以提高覆盖强度。

但如果主瓣方向没有对准 UE，仅加功率可能只是粗暴扩大覆盖，不一定解决方向问题。

单独调方位角可以让主瓣更对准 UE。

但如果链路预算本身偏弱，仅调方向可能不够。

所以 F2 给出组合：

```text
C3  Adjust the azimuth of 3273494_1 by 50 degrees
C14 Increase transmission power for 3273494_1
```

这是典型的覆盖形状 + 覆盖强度双修复。

---

## 8. 为什么不是其它候选

### 不是 A3

选项里有：

- `Decrease A3 Offset threshold for 3273494_1`
- `Increase A3 Offset threshold for 3273494_1`

但这题没有体现“强邻区应该切换但没切”的主线。

低谷期间邻区也弱，不像单纯 A3 threshold 问题。

### 不是 add neighbor

目标 serving PCI `827` 的配置邻区里已经有：

```text
[3268508_504990_132,3213593_504990_1004,3231269_504990_507]
```

低谷期间可见邻区也主要就是这些 PCI。

所以缺邻区不是最自然解释。

### 不是 PDCCH

目标 cell 的 CCE allocation success rate 很高，PDCCH 不像瓶颈。

---

## 9. 这题的诊断链

可以压成这条链：

1. `tag = multiple-answer`，先按 F2 组合题处理。
2. `user_plane_data` 发现 throughput 出现深低谷。
3. 低谷窗口里 serving RSRP 和 SINR 同时变差。
4. 邻区 RSRP 也整体偏弱，不像 late handover。
5. traffic 里 PDCCH/CCE 不异常，不像控制信道问题。
6. 目标 cell 是当前 serving cell `3273494_1 / PCI 827`。
7. 覆盖方向和强度都需要修正。
8. 输出 `adjust_azimuth + increase_power`。

最终答案：

```text
C3|C14
```

---

## 10. 对 candidate generation 的启发

这类题很适合用 F2 模板：

```text
adjust_azimuth + increase_power
```

触发条件可以是：

- multiple-answer
- 4 BS
- throughput 深低谷
- RSRP/SINR 同时显著下降
- best neighbor 没有形成强 handover 证据
- PDCCH/CCE 不异常

然后 candidate generator 不应该从所有选项里乱组合，而应该优先找：

- 当前题中 `adjust_azimuth` 对应哪个 C-code
- 当前题中 `increase_power` 对应哪个 C-code
- 二者是否指向同一个合理 target cell

这正是 `scenario_parser.py + candidate_generator.py` 组合的价值。

---

## 一句话总结

这条样本不是“一个参数坏了”，而是典型 F2 复合覆盖修复题。  
当 throughput 深跌伴随 RSRP/SINR 同时变差，而控制信道和 server 证据不强时，`adjust_azimuth + increase_power` 是非常自然的组合候选。
