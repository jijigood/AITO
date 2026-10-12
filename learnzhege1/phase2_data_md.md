# Track A `data/Phase_2` 学习说明

这个文档对应官方数据目录：

- [5.4/Track A/data/Phase_2](/home/huangxiaolin/AITO/5.4/Track%20A/data/Phase_2)

它的目标是讲清楚 Phase 2 和 Phase 1 的差别。  
如果你已经读过 [phase1_data_md.md](/home/huangxiaolin/AITO/learnzhege1/phase1_data_md.md)，这份文档就是它的 Phase 2 补丁。

---

## 1. 这个目录里有什么

Phase 2 目录主要有：

- `README.md`
- `test.json`

和 Phase 1 不同，Phase 2 没有公开训练集标签，也没有把真实观测表直接放进 JSON。

官方 README 说得很直接：

- 必须使用官方云端服务访问 simulation sandbox
- 云端服务暴露的 endpoint 和 Phase 1 `server.py` 一致
- 请求时必须带 Zindi 提供的 Bearer token

两个工具服务地址：

- 中国区：`https://120.46.145.77/no`
- 海外区：`https://124.71.227.61/no`

请求头核心格式：

```text
Content-Type: application/json
Authorization: Bearer <Your-Token>
```

---

## 2. Phase 2 `test.json` 的规模

我统计了 `/home/huangxiaolin/AITO/5.4/Track A/data/Phase_2/test.json`：

| 项目 | 数量 |
|---|---:|
| 样本数 | `500` |
| `single-answer` | `298` |
| `multiple-answer` | `202` |
| `answer = To be determined` | `500` |
| 每题 option 数 | `24` |

这里最重要的变化是：

- Phase 1 常见是 `C1 ~ C22`
- Phase 2 是 `C1 ~ C24`

所以如果你的答案清洗、候选生成、合法选项集合仍然写死 `C1-C22`，到 Phase 2 会直接漏掉 `C23/C24`。

这也是为什么 5.4 的新增代码里，不能只靠旧的 `utils.py` 思维继续往前走，最好把 option label 动态解析作为第一步。

---

## 3. 顶层字段仍然像 Phase 1

Phase 2 每条样本顶层仍然有：

- `scenario_id`
- `tag`
- `task`
- `answer`
- `context`
- `data`
- `tools`

所以从“题目外壳”看，它仍然像 Phase 1。

但每个字段的含义有几个细节变化。

---

## 4. `task.options`：从 22 个变成 24 个

Phase 2 每题有 `24` 个候选项：

- `C1`
- `C2`
- ...
- `C24`

这件事非常重要，因为很多早期代码会默认：

```python
VALID_OPTIONS = {f"C{i}" for i in range(1, 23)}
```

这在 Phase 1 够用，但在 Phase 2 不够。

如果你要让 `answer_parser.py` 兼容 Phase 2，应该改成：

```python
VALID_OPTIONS = {f"C{i}" for i in range(1, 25)}
```

或者更稳一点：

```python
VALID_OPTIONS = {opt["id"] for opt in scenario["task"]["options"]}
```

后一种更适合 Track A，因为 option 数量和选项语义都不应该写死。

---

## 5. `data`：五张表变成 API 占位

Phase 1 的 `data` 里直接内嵌五张 pipe-delimited CSV：

- `user_plane_data`
- `network_configuration_data`
- `signaling_plane_data`
- `traffic_data`
- `mr_data`

Phase 2 的这五个字段仍然存在，但值通常是同一句占位文案：

```text
Please use the API to access the cloud server to retrieve data
```

也就是说：

- 本地 `test.json` 只给题干、选项、context、scenario_id
- 真正用于诊断的五张表要通过工具 API 拉回来

这就是 [api_data_gatherer_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/api_data_gatherer_md.py) 存在的原因。

---

## 6. `tag + num_base_stations` 分布

Phase 2 的 `num_base_stations` 分布：

| `num_base_stations` | 数量 |
|---|---:|
| `4` | `386` |
| `5` | `57` |
| `13` | `57` |

组合分布：

| 组合 | 数量 |
|---|---:|
| `4 BS + single-answer` | `241` |
| `4 BS + multiple-answer` | `145` |
| `5 BS + multiple-answer` | `29` |
| `5 BS + single-answer` | `28` |
| `13 BS + single-answer` | `29` |
| `13 BS + multiple-answer` | `28` |

这和 Phase 1 的训练分布不完全一样。

Phase 1 里我们总结过强规则：

- `5 BS` 基本就是多答案 F3
- `13 BS` 基本就是单答案 F4

但 Phase 2 里出现了：

- `5 BS + single-answer`
- `13 BS + multiple-answer`

所以 5.4 的 `phase2_run.py` 里有一个 safety override：

- 先按 `num_base_stations` 路由 family
- 再检查 family 的 `is_multiple` 是否和真实 `tag` 一致
- 如果不一致，优先尊重 `tag`

这说明：

- family 仍然是很强的先验
- 但 Phase 2 不能把 Phase 1 的 family 规则当成绝对硬规则

---

## 7. `answer`

Phase 2 全部都是：

```text
To be determined
```

这和 Phase 1 test 一样，不能用于本地真实评估。

所以 Phase 2 的本地开发更依赖：

- Phase 1 train 的统计规律
- feature/candidate 规则
- 少量可解释 verifier
- 提交后的 leaderboard 反馈

---

## 8. `tools`

Phase 2 的 `tools` 字段更像能力说明：

- `description = Analytical tools are provided to retrieve the collected data.`
- `capabilities = ["data retrieval"]`

真正的 OpenAI-compatible tool schema 仍然要从：

```text
GET /tools
```

获取。

这和 Phase 1 `server.py` 的设计一致。

---

## 9. Phase 2 和 `main.py` 的关系

你仍然可以用 `main.py` 的 tool-calling loop 去跑 Phase 2，但要注意：

1. `server_url` 要指向云端地址。
2. 请求头要带 Bearer token。
3. 本地 JSON 的 `data` 字段不能直接解析成 DataFrame。
4. 如果从 `/scenario/all` 拿 scenario，`tag` 可能会被官方 `Scenario.model_dump()` 丢掉。
5. 如果你的答案合法性检查仍然只允许 `C1-C22`，会漏掉 Phase 2 的 `C23/C24`。

5.4 新增的 `phase2_run.py` 之所以直接读原始 `test.json`，就是为了保留：

- `tag`
- `task.options`
- `context`

然后再通过 API 拉真实表格。

---

## 10. Phase 2 和 `phase2_run.py` 的关系

`phase2_run.py` 是专门为这个数据形态写的：

1. 读本地 `test.json`，拿题干和选项。
2. 用 `tag + num_base_stations` 做 family router。
3. 用 `APIDataGatherer` 从云端拉五张表。
4. 用 `parse_scenario()` 转成 ParsedScenario。
5. 用 `run_all_skills()` 做特征工程。
6. 用 candidate generator 输出 C-code。
7. 写出 `scenario_id, Track A` 两列 CSV。

所以 Phase 2 的关键不是“JSON 结构完全变了”，而是：

- 题目外壳仍在本地
- 诊断数据移到云端
- option 数从 22 扩到 24
- family 规则需要尊重 `tag` 做修正

---

## 11. 一句话总结

Phase 2 不是一份“本地自包含数据集”，而是一份“本地题目清单 + 云端数据 API”的评测集。

你读它时要把注意力放在三件事上：

- 本地 JSON 保留了哪些决策先验
- 云端 API 如何把五张表补回来
- 旧 Phase 1 的硬规则在 Phase 2 哪里需要放松
