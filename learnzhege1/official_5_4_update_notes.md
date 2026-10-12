# 5.4 官方更新对照笔记

这份笔记专门记录 `/home/huangxiaolin/AITO/5.4` 相比你之前的 `learnzhege1` 学习版，哪些地方值得补充。

结论先说：

- `_types.py`、`utils.py`、`server.py`、`logger.py` 的核心数据契约和工具逻辑没有大改，你原来的中文注释版仍然可以继续作为 Phase 1 starter code 学习资料。
- 真正需要补的是 5.4 的 Phase 2 运行方式、云端工具 API 鉴权、`main.py` 默认参数变化、以及新增的 `phase2_run.py` 和 `skills/` 规则流水线。
- 5.4 里还出现了一个很关键的小细节：原始 JSON 有 `tag`，但官方 `Scenario` Pydantic 模型没有声明它。走 `model_validate -> model_dump` 会丢掉 `tag`，而新增的 Phase 2 deterministic runner 正是直接读原始 JSON 来保留这个字段。

---

## 1. 旧五个源码文件

### `_types.py`

结构基本没变。

值得补充的不是字段新增，而是字段缺失：

- 原始样本顶层有 `tag`，例如 `single-answer` / `multiple-answer`。
- 官方 `Scenario` 没有声明 `tag`。
- `server.py` 通过 `load_scenarios()` 把 raw JSON 校验成 `Scenario` 后，再 `model_dump()` 返回时会丢掉 `tag`。
- 所以如果后续逻辑要做 F1/F2/F3/F4 路由，最好直接读原始 JSON，或者自己扩展类型模型。

我已经把这个点补进 [_types_md.py](/home/huangxiaolin/AITO/learnzhege1/_types_md.py) 的 `Scenario` 注释里。

### `main.py`

5.4 里有几处和 Phase 2 运行直接相关的变化：

- 默认 `server_url` 从本地 `http://localhost:7860` 改成中国区云端 `https://120.46.145.77/no`。
- Phase 2 README 要求请求头带 `Authorization: Bearer <Your-Token>`。
- 官方模板在工具请求里加了 token 相关 header，并对 HTTPS 请求设置了 `verify=False`。
- 默认 `max_samples` 从旧学习版里的 `130` 变成 `500`，对应 Track A Phase 1/2 测试规模。
- 保存结果时除了 `result.csv`，还新增了完整运行记录 `results.json`。
- 需要额外注意 Phase 2 option 数是 `24`，旧的 `C1-C22` 假设不再够用。

我已经把这些点补进 [main_md.py](/home/huangxiaolin/AITO/learnzhege1/main_md.py)。

### `server.py`

server 的工具集合和核心逻辑基本不需要补。

Phase 2 的重要变化不是本地 server 新增了很多 endpoint，而是官方提供了云端工具服务，并说明云端部署的是和 Phase 1 `server.py` 一样的 endpoint。也就是说，你理解 `server_md.py` 仍然很有用，只是调用对象从本地 FastAPI 换成了远程 API。

### `utils.py`

`utils.py` 本体基本不需要补。

但 5.4 新增的 `skills/answer_parser.py` 比 `utils.py` 里的 `compute_score()` 更贴近比赛指标：

- `utils.compute_score()` 是 starter code 的简化本地评分。
- `skills.answer_parser.iou_score()` 按答案集合算 IOU，更接近 README 里的 accuracy 定义。
- `validate_and_normalise()` 会过滤非法 C-code 并按数字升序输出；如果跑 Phase 2，需要注意合法范围不应只写死到 `C22`。

### `logger.py`

没有实质性更新，原学习版注释已经够用。

---

## 2. Phase 2 新增文件

### `data/Phase_2/README.md`

Phase 2 最大变化是工具 server 的使用方式：

- 不再要求只跑本地 server。
- 官方提供云端 endpoint。
- 请求必须带 Bearer token。

两个地址：

- 中国区：`https://120.46.145.77/no`
- 海外区：`https://124.71.227.61/no`

### `phase2_run.py`

这是 5.4 最值得单独学习的新入口。

它和原始 `main.py` 的思路完全不同：

- 原始 `main.py` 是 LLM tool-calling loop，让模型自己决定调什么工具。
- `phase2_run.py` 是 deterministic agent，基本绕过 LLM。
- 它按固定流程做：读 JSON、按 family 路由、从 API 拉数据、解析表格、跑特征、生成候选、写提交。

核心流程：

1. 从 `data/Phase_2/test.json` 读取原始 scenario。
2. 用 `tag + num_base_stations` 路由到 `F1/F2/F3/F4`。
3. 通过云端 API 拉回 user plane、config、signaling、traffic、MR 等原始 CSV。
4. 用 `scenario_parser.py` 转成结构化 `ParsedScenario`。
5. 用 `feature_engine.py` 跑 A-G 七类特征。
6. 低置信度时额外调用 `judge_mainlobe_or_not` 做 lift_tilt / insufficient_data 消歧。
7. 用 `candidate_generator.py` 直接输出 `C` 编号。
8. 写出 `scenario_id, Track A` 两列 CSV。

这说明 5.4 的新增重点已经从“理解官方 starter code 怎么调模型”转向“把 Track A 做成规则化、家族化、特征化的可控 pipeline”。

---

## 3. 新增 `skills/` 目录

5.4 的 `skills/` 可以按下面这张心智图理解。

| 文件 | 作用 | 学习重点 |
|---|---|---|
| `scenario_router.py` | 按 `tag + num_base_stations` 分 F1/F2/F3/F4 | 题型家族是强先验 |
| `scenario_parser.py` | 解析五张 CSV 和每题选项 | C-code 不是固定语义，必须按 label 动态解析 |
| `api_data_gatherer.py` | Phase 2 云端 API 数据桥 | Phase 2 JSON 不应被看成“全部数据都在本地” |
| `feature_engine.py` | A-G 七类特征工程 | 低吞吐定位、mobility、inter-freq、coverage、power、capacity、server |
| `candidate_generator.py` | 把 issue score 转成候选 C-code | 先缩小候选集，再输出答案 |
| `answer_parser.py` | 答案提取、合法化、IOU 评分 | 比 `utils.compute_score()` 更适合最终评估 |

你原来已经有 [track_a_family_overview.md](/home/huangxiaolin/AITO/learnzhege1/track_a_family_overview.md)，它和 `scenario_router.py` 是同一条主线；这次只需要把 “family 先验” 和 “skills pipeline” 接起来理解。

---

## 4. 最容易漏的 5.4 细节

### `tag` 会在 server 路径里丢失

这个点很关键：

- `phase2_run.py` 用的是原始 `test.json`，所以能读到 `scenario["tag"]`。
- `main.py` 从 `/scenario/all` 拿到的是 server 返回的 `model_dump()`，默认不会带 `tag`。

如果你以后想在 `main.py` 的 LLM agent loop 里也用 family router，有两个选择：

- 直接让 `main.py` 读取原始 `test.json` 补充 `tag`。
- 或者扩展 `_types.py` 的 `Scenario`，显式声明 `tag: Optional[str] = None`。

### Phase 2 API 数据仍然是那五张表

Phase 2 看起来是云端 API，但本质上仍然围绕这些表：

- `network_configuration_data`
- `user_plane_data`
- `signaling_plane_data`
- `traffic_data`
- `mr_data`

Phase 2 的 `test.json` 里这些字段通常只是占位文案：

```text
Please use the API to access the cloud server to retrieve data
```

所以你之前对 `server_md.py` 和 `phase1_data_md.md` 的学习没有过时，只是数据获取路径从“JSON 内嵌原表”换成了“通过同名工具 API 拉回原表”。

### `result.csv` 列名有两种上下文

原 `main.py` 写的是：

- `scenario_id`
- `answers`

`phase2_run.py` 写的是：

- `scenario_id`
- `Track A`

如果要和 Track B 合并成总提交文件，后者更贴近多 Track 提交格式。

### Track B 是另一个系统

`5.4` 里新增/包含了完整 Track B：IP 网络、CLI 仿真、OpenClaw agent 配置。

它和 `learnzhege1` 这套 Track A 学习资料不是同一个主题。除非你接下来要做 Track B，否则这里不用强行把 Track B 混进 Track A 注释版。

---

## 5. 建议补充顺序

如果你继续完善学习版，我建议按这个顺序补：

1. 先读 `official_5_4_update_notes.md`，明确哪些是 5.4 真变化。
2. 回到 [_types_md.py](/home/huangxiaolin/AITO/learnzhege1/_types_md.py)，记住 `tag` 这个类型模型外字段。
3. 看 [phase2_data_md.md](/home/huangxiaolin/AITO/learnzhege1/phase2_data_md.md)，明确 Phase 2 的本地 JSON 和云端数据边界。
4. 看 [main_md.py](/home/huangxiaolin/AITO/learnzhege1/main_md.py) 的 Phase 2 cloud API 相关变化。
5. 看 [track_a_family_overview.md](/home/huangxiaolin/AITO/learnzhege1/track_a_family_overview.md)，把 F1/F2/F3/F4 和 `scenario_router.py` 对上。
6. 继续看 [phase2_run_md.py](/home/huangxiaolin/AITO/learnzhege1/phase2_run_md.py) 和 [skills/README.md](/home/huangxiaolin/AITO/learnzhege1/skills/README.md)，因为它们对应的是另一种 agent 架构，不只是 starter code 注释。

---

## 一句话总结

`learnzhege1` 作为 Phase 1 源码学习版仍然有效；5.4 真正需要补的是 Phase 2 的云端调用方式和新增的 deterministic skills pipeline。
