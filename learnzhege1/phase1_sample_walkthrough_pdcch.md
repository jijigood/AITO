# Phase 1 样本 Walkthrough：PDCCH / 控制信道容量正例

这份 walkthrough 读一条典型的：

```text
Modify PdcchOccupiedSymbolNum to 2SYM
```

正例。

目标不是复述答案，而是理解：

- 为什么低吞吐不一定是覆盖、切换、功率问题
- 什么时候 `PdcchOccupiedSymbolNum -> 2SYM` 会比改无线覆盖参数更合理
- `traffic_data` 里的 CCE / allocation success rate 为什么很关键

---

## 1. 样本基本信息

样本：

- `scenario_id = 88f1c45d-96a4-444e-ba6b-66658ec4ad48`
- `tag = single-answer`
- `num_base_stations = 4`
- 标准答案：`C15`

答案文本：

```text
Modify PdcchOccupiedSymbolNum to 2SYM for 3236632_3
```

目标 cell：

- `gNodeB ID = 3236632`
- `Cell ID = 3`
- `PCI = 566`

---

## 2. 先看吞吐：确实有一个短时间低谷

用户面里 serving PCI 一直是 `566`，也就是答案目标 cell。

关键时间点：

| Timestamp | Serving PCI | RSRP | SINR | Throughput |
|---|---:|---:|---:|---:|
| 22:21:31.350 | 566 | -91.91 | 16.43 | 469.72 |
| 22:21:33.294 | 566 | -88.24 | 16.75 | 551.02 |
| 22:21:34.522 | 566 | -90.45 | 14.40 | 94.65 |
| 22:21:35.748 | 566 | -90.97 | 14.32 | 64.82 |
| 22:21:37.881 | 566 | -89.48 | 7.95 | 56.22 |
| 22:21:40.109 | 566 | -90.03 | 9.74 | 462.45 |
| 22:21:42.437 | 566 | -91.60 | 11.77 | 376.34 |

现象很清楚：

- 吞吐从 `551.02 Mbps` 掉到 `56.22 Mbps`
- 后面又恢复到 `462.45 Mbps` 以上
- 这是短窗口退化，不是整段都差

---

## 3. 但无线覆盖本身不像主因

低吞吐期间，serving RSRP 大致在：

```text
-91 dBm ~ -87 dBm
```

SINR 大多数时间也不差：

```text
14 dB 左右，最差点还有 7.95 dB
```

所以它不像典型覆盖洞：

- 不是 RSRP 掉到 `-110 dBm` 以下
- 不是 SINR 长时间小于 `0 dB`
- serving PCI 没有明显切换来回跳

这一步很关键。

如果只看到 throughput 低就去选：

- increase power
- lift tilt
- adjust azimuth

就会把“无线侧看起来健康”的题误判成覆盖问题。

---

## 4. 再看配置：目标 cell 仍是 `1SYM`

目标 cell `3236632_3 / PCI 566` 的配置：

| 字段 | 值 |
|---|---:|
| Transmission Power | 18 |
| Mechanical Azimuth | 241 |
| Mechanical Downtilt | 1 |
| PdcchOccupiedSymbolNum | 1SYM |

`PdcchOccupiedSymbolNum = 1SYM` 说明当前 PDCCH 控制区域比较窄。

如果控制信道资源不足，用户面吞吐可能会掉，即使 RSRP/SINR 没有明显坏掉。

---

## 5. 关键证据在 `traffic_data`

目标 cell `3236632_3` 的 traffic 指标：

| 指标 | 值 |
|---|---:|
| Downlink PRB utilization | 17.2860 |
| Uplink CCE utilization | 0.8273 |
| Downlink CCE utilization | 0.7857 |
| Uplink CCE Allocation Success Rate | 41.458 |
| Downlink CCE Allocation Success Rate | 33.395 |

最醒目的不是 PRB。

PRB 利用率只有 `17.2860%`，不像普通数据面资源耗尽。

真正异常的是：

- 上行 CCE 分配成功率只有 `41.458%`
- 下行 CCE 分配成功率只有 `33.395%`

这正是 PDCCH / CCE 资源问题的味道。

---

## 6. 为什么不是其它选项

### 不是 `Check test server`

如果是 server / transmission 问题，通常会看到无线侧和小区侧指标都健康。

但这题目标 cell 的 CCE allocation success rate 明显异常，已经有很具体的无线控制信道证据。

所以不应该优先选：

```text
C11 Check test server and transmission issues
```

### 不是覆盖类动作

覆盖类动作包括：

- lift tilt
- press down tilt
- adjust azimuth
- increase/decrease power

这题 RSRP/SINR 不支持强覆盖问题。

### 不是 A3 / handover

serving PCI 一直是 `566`，没有明显 stuck serving 后切换恢复的结构。

而且答案目标正是当前 serving cell 的 PDCCH 参数。

---

## 7. 这题的诊断链

可以把这题压成一条很短的链：

1. `user_plane_data` 发现吞吐短时大幅下降。
2. 同一窗口里 RSRP/SINR 不差，不像覆盖洞。
3. serving PCI 一直是 `566`，对应 cell `3236632_3`。
4. 配置表里 `3236632_3` 的 `PdcchOccupiedSymbolNum = 1SYM`。
5. `traffic_data` 中该 cell 的 CCE allocation success rate 很低。
6. 所以应扩大 PDCCH 占用符号数到 `2SYM`。

最终答案：

```text
C15 = Modify PdcchOccupiedSymbolNum to 2SYM for 3236632_3
```

---

## 8. 对 feature engineering 的启发

这类题可以设计成一组规则：

- throughput 有明显低谷
- serving RSRP 不弱
- serving SINR 不差
- PRB utilization 不高
- CCE allocation success rate 明显低
- 目标 cell 的 `PdcchOccupiedSymbolNum = 1SYM`

满足这些条件时，`pdcch_2sym` 的优先级应该高于：

- coverage_hole
- late_handover
- check_server

这就是 `feature_engine.py` 里 capacity/control-channel 类 skill 应该抓住的边界。

---

## 一句话总结

PDCCH 题的核心不是“信号弱”，而是“无线信号看起来还行，但控制信道分配明显失败”。  
看到低吞吐时，不要只盯 RSRP/SINR，`traffic_data` 里的 CCE 成功率往往才是决定性证据。
