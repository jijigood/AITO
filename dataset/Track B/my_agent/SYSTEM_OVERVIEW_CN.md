# Track B Agent 系统说明

本文档描述 `my_agent` 的整体系统组成、工作流程、关键约束和后续扩展方向。它面向 Track B 的 IP 网络故障诊断赛题，目标是在遵守赛题规则的前提下，稳定生成可提交的 `result.csv`。

## 1. 系统目标

Track B 要求参赛 Agent 通过调用 Agent Tool Server 暴露的网络设备 CLI 仿真接口，完成以下三类任务：

- 拓扑恢复：补全某个节点 UP 接口对应的链路。
- 路径查询：给出源节点到目标 IP 或目标端口的逐跳路径。
- 故障定位：判断导致业务不可达的路由故障或端口故障，并按题目要求输出原因。

最终输出文件为：

```text
id,prediction
1,...
2,...
```

评分方式以 exact match 准确率为核心，因此系统必须特别重视格式、空格、故障原因词、节点名和端口名。

## 2. 系统组成

当前 `my_agent` 由四个核心 Python 模块组成：

| 文件 | 作用 |
|---|---|
| `main.py` | 批量入口，负责读取题目、并发调度、结果保存、进度记录 |
| `agent_loop.py` | ReAct 循环，负责让模型在“调用工具”和“输出答案”之间切换 |
| `tools.py` | NOC API 客户端，负责调用官方 `server.py` 的 `/api/agent/execute` 接口 |
| `prompts.py` | 系统提示词、题型识别、工具调用解析、答案抽取 |

官方组件保持不改：

| 官方组件 | 作用 |
|---|---|
| `server.py` | 本地 Agent Tool Server，模拟多厂商设备 CLI |
| `devices_outputs/` | 每道题、每台设备、每条命令对应的离线 CLI 输出 |
| `question_limits_config.json` | 定义部分题目的无权限命令限制 |
| `data/Phase_1/test.json` | Phase 1 的 50 道测试题 |

## 3. 架构关系

```text
test.json
   |
   v
main.py
   |
   | 逐题调度，最多并发 2 题
   v
agent_loop.py
   |
   | 发送题目和历史上下文
   v
OpenAI-compatible LLM
   |
   | 输出 <tool_call> 或最终答案
   v
tools.py / NOCClient
   |
   | POST /api/agent/execute
   v
official server.py
   |
   | 根据 question_number + device_name + command 读取 CLI 输出
   v
devices_outputs/
   |
   v
agent_loop.py
   |
   | 循环分析，直到得到最终答案
   v
result.csv + detail.jsonl + progress.json
```

## 4. 单题工作流程

每道题的处理流程如下：

1. `main.py` 从 `test.json` 读取题目，并根据题目 id 创建一个 `NOCClient`。
2. `prompts.detect_family()` 粗略判断题型：`topology`、`path` 或 `fault`。
3. `agent_loop.solve()` 将系统提示词和题目发送给 LLM。
4. 如果 LLM 输出 `<tool_call>`，系统解析出设备名和命令。
5. `NOCClient.execute()` 调用官方 server，并自动附带 `question_number`。
6. CLI 输出作为新的用户消息回填给 LLM。
7. LLM 继续调用工具或输出最终答案。
8. `prompts.extract_answer()` 根据题型抽取干净答案。
9. `main.py` 写入 `result.csv`，并在 `detail.jsonl` 中记录题型、调用数、耗时和答案。

## 5. 赛题约束映射

| 赛题约束 | 当前系统对应设计 |
|---|---|
| 单题内工具调用必须串行 | `agent_loop.py` 中每次只解析并执行一个 `<tool_call>` |
| 整体并发最多 2 题 | `main.py --concurrency` 默认值为 2 |
| 每次调用必须带题号 | `NOCClient` 自动发送 `question_number` |
| Phase 2 同分可能看调用数 | `detail.jsonl` 记录每题 `call_count` |
| 输出 exact match | `extract_answer()` 负责清洗模型输出 |
| 命令可能无权限 | `NOCClient` 将 403 包装为 `[PERMISSION DENIED]` |
| 命令可能语法错误 | `NOCClient` 将 422 包装为 `[COMMAND ERROR]` |

## 6. 三类题的解决策略

### 6.1 拓扑恢复

优先命令：

```text
display lldp neighbor brief
display interface brief
display interface description
display arp
```

常规路径：

1. 查询目标节点 LLDP。
2. 如果目标节点 LLDP 无权限，则查询候选邻居节点 LLDP。
3. 如果 LLDP 仍不可用，则用接口描述、ARP 和接口状态交叉验证。
4. 输出 `LocalNode(port)->RemoteNode(port)`。

### 6.2 路径查询

优先命令：

```text
display ip interface brief
display ip routing-table
display arp
```

常规路径：

1. 找到目标端口对应 IP 或目标 IP 所在网段。
2. 在源节点路由表中做最长匹配，找到下一跳。
3. 通过接口 IP 表识别下一跳属于哪台设备。
4. 重复逐跳查找，直到到达目标节点。
5. 输出 `NodeA->NodeB->NodeC`。

### 6.3 故障定位

优先命令：

```text
display ip routing-table
display interface brief
display current-configuration
display arp
display logbuffer
display alarm active
display bgp evpn all routing-table
display bgp vpnv4 all routing-table
display vxlan tunnel
display srv6-te policy
```

常规路径：

1. 先推断预期转发路径。
2. 检查每跳路由是否缺失、错误、黑洞或成环。
3. 检查相关接口是否 shutdown、IP 错误或带宽异常。
4. 检查 ARP、BGP、OSPF、VXLAN、L3VPN、L2VPN、SRv6 等配置状态。
5. 使用题目中列出的故障原因原文输出。

## 7. 当前风险点

当前版本仍有以下风险，需要在计划中优先处理：

- 默认 server 端口需要和官方 `server.py` 对齐。
- `--resume` 逻辑不能清空历史结果。
- 故障原因必须以题目原文为准，不能固定使用少量旧词表。
- `PSS` 是 Cisco 设备，需要补充 Cisco 命令指导。
- 长路由表和长配置直接送入 LLM，可能造成上下文膨胀。
- 纯 ReAct 对调用次数控制较弱，后续需要引入确定性解析和规则 solver。

## 8. 目标形态

理想的最终系统不是“纯 LLM 盲查”，而是：

```text
题目解析器
  + 确定性 CLI 解析器
  + 拓扑/路由图构建器
  + 规则化故障检测器
  + LLM 辅助裁决
  + 严格答案格式化器
```

这样可以同时提升准确率、稳定性和调用效率。
