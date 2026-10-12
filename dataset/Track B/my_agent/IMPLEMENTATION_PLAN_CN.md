# Track B Agent 开展计划

本文档描述 `my_agent` 后续实施计划。原则是先把系统修到稳定可跑，再逐步减少调用数、提升 exact match 准确率，最后沉淀成可复用的确定性网络推理框架。

## 总体原则

1. 稳定性优先：先保证所有题都能跑完并生成合法 `result.csv`。
2. 精确格式优先：Track B 是 exact match，格式错误等同于答错。
3. 少调用但不盲省：Phase 2 同分可能看 API 调用数，但准确率仍是第一优先级。
4. 规则和 LLM 结合：确定性网络事实由代码解析，复杂判断交给 LLM 辅助。
5. 每一步都要可验证：每个阶段都用小题集和全量 50 题回归。

## 第一步：修正基础运行问题

### 事先声明

这一步不追求大幅提升准确率，目标是让 baseline 可靠运行、可靠恢复、可靠提交。

### 具体工作

1. 将 `main.py` 默认 `server_url` 从 `http://localhost:7861` 改为 `http://localhost:7860`。
2. 修复 `--resume` 时清空 `result.csv` 和 `detail.jsonl` 的问题。
3. 输出 `result.csv` 前按 `id` 排序。
4. 增加启动前检查：
   - `test.json` 是否存在。
   - Agent Tool Server 是否连通。
   - LLM endpoint 是否连通。
5. 在 `NOCClient` 内增加单题缓存，避免重复调用同一条命令。

### 预期效果

- 运行入口和官方 server 默认配置一致。
- 中断后可以继续跑，不会丢失已完成答案。
- 提交文件顺序稳定。
- 重复命令调用减少。
- 这一步完成后，系统应能稳定跑完 50 题。

### 验证方式

```bash
python my_agent/main.py --questions 1,2,7 --server_url http://localhost:7860
python my_agent/main.py --resume --server_url http://localhost:7860
```

检查：

- `my_agent/results/result.csv` 是否存在。
- `id,prediction` 表头是否正确。
- `detail.jsonl` 是否记录调用数和耗时。
- resume 后旧结果是否仍保留。

## 第二步：修正提示词和输出词表

### 事先声明

这一步主要解决 exact match 风险。即使网络推理正确，如果故障原因词写错，也会被判错。

### 具体工作

1. 在 prompt 中声明：故障原因必须使用当前题目列出的原文。
2. 补充 Phase 1 题面出现过的故障原因：
   - `blackhole route`
   - `missing static route`
   - `static route error`
   - `ARP configuration error`
   - `routing loop`
   - `BGP configuration error`
   - `OSPF configuration error`
   - `loopback IP configuration conflict`
   - `VXLAN configuration error`
   - `L3VPN configuration error`
   - `L2VPN configuration error`
   - `IS-IS configuration error`
   - `SRV6-Policy tunnel planning error`
   - `shutdown`
   - `interface IP error`
   - `traffic congestion on port bandwidth`
   - `MAC address configuration error`
   - `VPN configuration missing`
   - `MTU value configuration error`
   - `host information collection function missing`
3. 补充 `PSS` 是 Cisco 设备，必须使用 `show ...` 命令。
4. 在 `extract_answer()` 之后增加轻量格式校验：
   - 拓扑题只允许 `A(port)->B(port)`。
   - 路径题只允许 `A->B->C`。
   - 故障题只允许 `node;target;cause`。

### 预期效果

- 故障题的原因词更贴近题面，减少“意思对但字符串错”的问题。
- 涉及 PSS 的题目减少命令语法错误。
- 模型输出带解释时，答案抽取更稳。

### 验证方式

优先跑覆盖三类题的样本：

```bash
python my_agent/main.py --questions 1,2,7,17,29,34,39 --server_url http://localhost:7860
```

检查：

- 故障原因是否完全来自题面原文。
- 多行答案是否没有空行。
- `PSS` 相关调用是否使用 Cisco 命令。

## 第三步：增加命令输出压缩层

### 事先声明

这一步的目标是减少上下文膨胀，不改变官方 server 返回内容，只在送给 LLM 前做信息压缩。

### 具体工作

在 `NOCClient.execute()` 后增加 `summarize_output(command, raw_output, question_context)`：

1. `display lldp neighbor brief`：解析为边表。
2. `display interface brief`：只保留接口、物理状态、协议状态、利用率和错误数。
3. `display ip interface brief`：只保留接口、IP、掩码、状态。
4. `display ip routing-table`：优先保留目标网段相关路由、默认路由、静态路由、黑洞路由。
5. `display current-configuration`：按关键词抽取 interface、ip route-static、bgp、ospf、vpn-instance、vxlan、srv6 段。
6. `display arp`：结构化为 IP、MAC、接口。

### 预期效果

- LLM 上下文更短，长题不容易遗忘前面信息。
- 单题耗时下降。
- 复杂故障题更容易定位关键配置。

### 验证方式

比较压缩前后的：

- 每题耗时。
- LLM 调用是否报上下文长度问题。
- `detail.jsonl` 中的平均 call_count。
- 关键样本题答案是否不退化。

## 第四步：拓扑恢复确定性 Solver

### 事先声明

拓扑恢复题最适合代码化。目标是让 1-6、29-33 尽量不依赖 LLM 推理。

### 具体工作

1. 从题目中抽取目标节点和候选邻居。
2. 优先调用目标节点 `display lldp neighbor brief`。
3. 如果目标节点无权限，则遍历候选邻居查 LLDP。
4. 如果 LLDP 不可用，再使用 interface description、interface brief、ARP 交叉验证。
5. 由代码直接生成 `LocalNode(port)->RemoteNode(port)`。

### 预期效果

- 拓扑题准确率更稳定。
- 调用次数可控。
- 输出格式基本不出错。

### 验证方式

```bash
python my_agent/main.py --questions 1,2,3,4,5,6,29,30,31,32,33 --server_url http://localhost:7860
```

检查：

- 每行是否严格为 `A(port)->B(port)`。
- 无权限题是否能自动走反向 LLDP。

## 第五步：路径查询确定性 Solver

### 事先声明

路径题可通过路由表逐跳推断。目标是让路径题主要由代码完成，LLM 只做兜底。

### 具体工作

1. 从题目抽取源节点、目标节点、目标端口或目标 IP。
2. 如果目标 IP 未给出，查询目标节点 `display ip interface brief`。
3. 在源节点 `display ip routing-table` 中做最长前缀匹配。
4. 根据 next-hop IP 反查设备。
5. 逐跳重复，直到目标节点。
6. 生成 `A->B->C`。

### 预期效果

- 路径题调用数下降。
- 路径输出格式稳定。
- 对已知无权限路由表题可以提前切换到邻居/接口 IP 推断。

### 验证方式

```bash
python my_agent/main.py --questions 7,8,9,10,11,12,13,14,15,16,34,35,36,37,38 --server_url http://localhost:7860
```

检查：

- 是否到达目标节点。
- 是否错误包含接口名或 IP。
- 是否存在多余空格。

## 第六步：故障定位规则增强

### 事先声明

故障题最复杂，不建议一次完全规则化。先做高置信规则，再用 LLM 做最后裁决。

### 具体工作

1. 静态路由类：
   - 缺路由：`missing static route`
   - 下一跳错误：`static route error`
   - 下一跳不可达或出接口 shutdown：`blackhole route` 或端口 `shutdown`
2. 接口类：
   - `*down` 或配置 `shutdown`：`shutdown`
   - 接口 IP 与对端/规划不一致：`interface IP error`
   - 利用率异常：`traffic congestion on port bandwidth`
3. ARP 类：
   - IP 到接口或 MAC 映射异常：`ARP configuration error`
4. 协议类：
   - BGP/OSPF/VXLAN/L3VPN/L2VPN/SRv6 先抽取关键配置和状态，再交给 LLM 判断。

### 预期效果

- 简单故障题不再依赖 LLM 自由发挥。
- 复杂协议题有结构化证据，降低误判。
- 故障原因词与题面一致。

### 验证方式

```bash
python my_agent/main.py --questions 17,18,19,20,21,22,23,24,25,26,27,28,39,40,41,42,43,44,45,46,47,48,49,50 --server_url http://localhost:7860
```

检查：

- 是否每行一个故障。
- 是否把路由故障和端口故障格式混淆。
- 是否输出题面未列出的原因词。

## 第七步：全量回归和调用数优化

### 事先声明

这一步不再大改策略，主要做回归、统计和微调。

### 具体工作

1. 跑全量 50 题。
2. 汇总每题：
   - 题型。
   - 调用数。
   - 耗时。
   - 是否空答案。
   - 是否格式不合法。
3. 对调用数异常高的题做专项优化。
4. 对格式不合法题增强 `extract_answer()`。
5. 固化一组 smoke test 题号。

### 预期效果

- 形成稳定 baseline。
- 知道哪些题是调用数瓶颈。
- Phase 2 迁移时有明确调试入口。

### 验证方式

```bash
python my_agent/main.py --input data/Phase_1/test.json --server_url http://localhost:7860 --concurrency 2
```

输出检查：

```text
my_agent/results/result.csv
my_agent/results/detail.jsonl
my_agent/results/progress.json
```

## 推荐执行顺序

优先级最高的五件事：

1. 修端口默认值。
2. 修 resume 不清空结果。
3. 故障原因改为题面原文优先。
4. 补充 PSS/Cisco 命令说明。
5. 增加单题命令缓存。

完成这五项后，再进入确定性 solver 的建设。这样可以尽快得到一个可跑、可恢复、可评估的 Track B baseline。

## 2026-04-29 加固进展

基于首轮 50 题结果，当前已经完成第一批加固：

1. 新增 `answer_validation.py`：
   - 按 topology/path/fault 三类题做 exact-match 格式校验。
   - 统一故障原因词表到题面原文。
   - 支持拓扑答案按目标节点视角去重和反向纠正。
2. `agent_loop.py` 增加最终答案校验与 repair：
   - LLM 输出不是合法答案时，不再直接写入 CSV。
   - 最多追加 2 次“只修格式、不调用工具”的 final-answer repair。
3. `tools.py` 增加命令规范化：
   - `display ip routing-table <ip>` 改为 `display ip routing-table`。
   - 去除 `| include` 这类不支持的管道过滤。
   - `display interface GE...` 改为 `display interface brief`。
4. 新增 `deterministic.py`：
   - topology 题优先用 LLDP / interface description 直接解析。
   - 离线 fixtures 验证 11 道 topology 均可生成格式合法答案。
5. 新增 `validate_results.py`：
   - 可对任意 `result.csv` 做离线格式审计。

下一步重点：

1. 小规模重跑高风险题，确认 repair 是否能显著减少解释文本。
2. 为 path/fault 增加路由表结构化摘要和候选下一跳解析。
3. 再做一次全量 50 题回归，比较格式合法率、空答案和调用数。

## 2026-04-29 二次赛题目标检查

本轮针对官方目标又补了几处低风险缺口：

1. 故障原因词表补齐 `host information collection function missing`，避免 Phase 1/2 题面合法原因被本地校验误判。
2. 故障答案归一化支持 `(3) cause` 编号前缀和 `cause1, cause2` 多原因拆行，减少 exact-match 格式损失。
3. H3C LLDP 优先使用官方 README 中的 `display lldp neighbor-information`，降低无效 API 调用。
4. 路由表摘要扩展到 Cisco `show ip route` / `show route`，为 Phase 2/3 多厂商路径题保留最长前缀匹配信息。
5. `detail.jsonl` 的 trace 增加 `question_number`、`started_at`、`elapsed_ms`，便于 Phase 2 上传执行轨迹和统计调用耗时。
6. `result.csv` 结束排序时去重；`validate_results.py` 增加重复 id 检查；`--resume` 会优先参考非空结果，避免空答案被当作已完成。

仍未完全解决的核心问题：

1. path/fault 还没有完整确定性 solver，准确率仍主要依赖 LLM 对路由表、接口表、配置和告警的推理。
2. Phase 2 execution trace 的最终上传格式如果官方另有 schema，需要在提交前再做一次适配。
3. Phase 3 的 SRv6/EVPN/ISIS/BGP 深层故障需要专门解析器，目前只是给 prompt 和命令集合预留了入口。

## 2026-04-29 测试记录

本轮实际测试结果：

1. 静态编译通过：`agent_loop.py`、`answer_validation.py`、`deterministic.py`、`main.py`、`prompts.py`、`tools.py`、`validate_results.py`。
2. 离线 fixtures：
   - topology 11/11 生成格式合法答案。
   - 物理接口 path 题 Q7-Q16 生成格式合法答案。
3. 第一次真实 smoke（Q1/Q2/Q7/Q17/Q34/Q39/Q42）：
   - Q1/Q2 topology 成功。
   - path/fault 暴露出模型反复发工具调用、最终空答案的问题。
4. 修复项：
   - 记录 `llm_trace`，用于定位模型 raw response。
   - 对残缺 `<tool_call>` 不再当成最终答案，而是提示模型重发。
   - 增加 deterministic path solver：按“路由表最长前缀匹配 + LLDP egress 端口”追踪物理路径。
5. 复测 Q7：
   - 走 deterministic path，不调用 LLM。
   - 输出合法：
     `Beta-Node-01->Beta-Axis-02->Beta-Portal-02->Alpha-Center-02->Gamma-Portal-02->Gamma-Axis-02->Gamma-Node-01`

测试结论：

- topology 与 Q7-Q16 这类物理接口 path 题已经可以优先确定性求解。
- fault 题和 PJ 区只给业务 IP 的 path 题仍是下一阶段重点。
