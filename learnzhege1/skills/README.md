# Track A 5.4 Skills 学习目录

这个目录对应官方：

`/home/huangxiaolin/AITO/5.4/Track A/skills/`

这里不是官方可运行模块，而是中文学习版导读。

---

## 这组文件在做什么

你可以把 5.4 的 deterministic pipeline 理解成这条链：

1. `scenario_router.py`
   先按 `tag + num_base_stations` 分到 F1/F2/F3/F4。

2. `api_data_gatherer.py`
   从云端 API 把 Phase 2 的五张原始表拉回来。

3. `scenario_parser.py`
   把五张表和每题候选选项标准化成 `ParsedScenario`。

4. `feature_engine.py`
   把表格信号转换成 late handover、coverage、PDCCH、server 等 issue score。

5. `candidate_generator.py`
   把 issue score 映射回当前题的具体 C-code。

6. `answer_parser.py`
   最后做答案提取、合法化和 IOU 评估。

---

## 推荐阅读顺序

1. [scenario_router_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/scenario_router_md.py)
2. [scenario_parser_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/scenario_parser_md.py)
3. [api_data_gatherer_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/api_data_gatherer_md.py)
4. [feature_engine_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/feature_engine_md.py)
5. [candidate_generator_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/candidate_generator_md.py)
6. [answer_parser_md.py](/home/huangxiaolin/AITO/learnzhege1/skills/answer_parser_md.py)

最后再回到：

- [phase2_run_md.py](/home/huangxiaolin/AITO/learnzhege1/phase2_run_md.py)

这样会更容易看懂整条 Phase 2 runner 是怎么把这些模块串起来的。
