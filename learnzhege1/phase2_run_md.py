#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Track A 学习版 phase2_run.py

官方对应文件：
    /home/huangxiaolin/AITO/5.4/Track A/phase2_run.py

这个文件是 5.4 里最重要的新增入口之一。

它和原来的 `main.py` 不是同一种 agent：

- `main.py`
  让 LLM 自己看题、自己决定调用哪些工具、最后输出 `\\boxed{C...}`。

- `phase2_run.py`
  基本绕过 LLM，把 Track A 当成一个结构化诊断问题：
  先拉数据 -> 本地解析 -> 跑规则特征 -> 生成候选 -> 直接输出 C 编号。

所以你可以把它理解成：

    Phase 2 deterministic agent runner

它更像一个可控的专家系统，而不是一个自由对话式 tool-calling agent。
"""


# ------------------------------------------------------------------------------
# 1. 为什么 Phase 2 需要这个新入口
# ------------------------------------------------------------------------------
#
# Phase 1 的 test/train JSON 里直接内嵌五张表：
#
# - network_configuration_data
# - user_plane_data
# - signaling_plane_data
# - traffic_data
# - mr_data
#
# Phase 2 的 test.json 仍然保留 `data` 字段，但里面通常是占位文字：
#
#     "Please use the API to access the cloud server to retrieve data"
#
# 也就是说：
# - 题干、选项、context 在本地 JSON
# - 真正的观测表格要通过云端工具 API 拉
#
# `phase2_run.py` 就是为这个新运行方式写的。


# ------------------------------------------------------------------------------
# 2. 主流程总览
# ------------------------------------------------------------------------------
#
# 官方代码的主线可以按 8 步理解：
#
# 1. 读取 `data/Phase_2/test.json`
#    得到 scenario_id、tag、task.options、context 等轻量信息。
#
# 2. 创建 `Environment`
#    复用 `main.py` 里的 HTTP 工具调用封装。
#    区别是 server_url 默认指向云端：
#        https://120.46.145.77/no
#
# 3. 对每条 scenario 调 `_solve_scenario()`
#    这是真正的单题求解函数。
#
# 4. 用 `scenario_router.route(tag, num_base_stations)` 分家族
#    - F1: 4 BS + single-answer
#    - F2: 4 BS + multiple-answer
#    - F3: 5 BS + multiple-answer
#    - F4: 13 BS + single-answer
#
# 5. 用 `APIDataGatherer.gather()` 拉回五张原始表
#    这一步大约会调 5-6 个工具 API。
#
# 6. 用 `parse_scenario(raw)` 做本地解析
#    把 pipe-delimited CSV 字符串转成 DataFrame，
#    同时把每题 option label 解析成 action_type / target_cell。
#
# 7. 用 `run_all_skills(ps)` 做特征工程
#    产出 ScenarioFeatures，里面包含各种 issue score。
#
# 8. 用 `generate_single_candidates(...)` 生成候选并格式化答案
#    最终写出 CSV：
#        scenario_id, Track A


# ------------------------------------------------------------------------------
# 3. `_format_answer()`：把候选对象变成提交字符串
# ------------------------------------------------------------------------------
#
# 输入：
# - candidates: candidate_generator 产出的 RankedCandidate 列表
# - family: F1/F2/F3/F4
# - tag: single-answer / multiple-answer
#
# 输出：
# - 单选：`C7`
# - 多选：`C3|C7` 或 `C5|C9|C11|C20`
#
# 关键规则：
#
# - 如果题目不是 multiple-answer，只取 top1。
# - 如果是 F3，默认取前 4 个候选。
# - 如果是 F2，默认取前 2 个；只有第 3 个候选分数足够强时才扩到 4 个。
# - 多答案最终按 C 编号数字升序排序，保证提交格式稳定。


# ------------------------------------------------------------------------------
# 4. `_solve_scenario()`：单题求解核心
# ------------------------------------------------------------------------------
#
# 这是官方 `phase2_run.py` 中最值得读的函数。
#
# 它做的事情可以理解为：
#
# Step 0：取基本字段
#
#     sid = scenario["scenario_id"]
#     tag = scenario["tag"]
#     num_bs = context.wireless_network_information.num_base_stations
#
# Step 1：路由到 family
#
#     family = route(tag, num_bs)
#
# 这里还有一个 Phase 2 safety override：
# 如果 family 的 single/multiple 和原始 tag 不一致，就优先尊重 tag。
#
# 这说明官方自己也意识到：
# - family 规则是强先验
# - 但最终答案个数仍然要以题目 tag 为准
#
# Step 2：通过 API 拉数据
#
#     gatherer = APIDataGatherer(env, scenario)
#     raw = gatherer.gather()
#
# 这一步会把 Phase 2 云端数据重新拼成和 Phase 1 raw scenario 类似的结构。
#
# Step 3：本地解析
#
#     ps = parse_scenario(raw)
#
# 得到 ParsedScenario：
# - 五张 DataFrame
# - parsed_options
# - cell_roles
# - pci_to_cell_id
#
# Step 4：跑特征
#
#     sf = run_all_skills(ps)
#
# 这个 `sf` 是全局诊断特征汇总。
#
# Step 5：lift_tilt / insufficient_data 消歧
#
# 如果 coverage_hole 和 insufficient_data 分数很接近，
# 且二者都不太低，就调用云端工具：
#
#     judge_mainlobe_or_not(time, pci)
#
# 直觉是：
# - 用户在主瓣外：更像 coverage hole / lift_tilt
# - 用户在主瓣内：无线几何没问题，更像 insufficient_data
#
# Step 6：生成候选
#
#     candidates = generate_single_candidates(ps, sf, family, top_n=6)
#
# 注意这里官方 phase2_run.py 调的是 `generate_single_candidates`，
# 然后 `_format_answer()` 再根据 tag/family 截取 1、2 或 4 个。
#
# Step 7：输出答案字符串
#
#     return "C7" 或 "C3|C8"


# ------------------------------------------------------------------------------
# 5. `main()`：批处理入口
# ------------------------------------------------------------------------------
#
# CLI 参数重点：
#
# --server_url
#     Phase 2 云端工具服务地址。
#
# --token
#     Zindi 提供的 Bearer token。
#     如果传入，会写入环境变量 `AGENT_API_KEY`。
#
# --test_json
#     Phase 2 test.json 路径。
#
# --out
#     输出 CSV，默认 `result_phase2.csv`。
#
# --save_freq
#     每 N 题保存一次，防止中途失败后完全丢结果。
#
# --max_samples
#     调试时只跑前 N 题。


# ------------------------------------------------------------------------------
# 6. 这个文件真正教你的东西
# ------------------------------------------------------------------------------
#
# 5.4 的 `phase2_run.py` 不是简单改了 `main.py` 参数，
# 而是展示了一种 Track A 更强的解法方向：
#
# 1. 不要让模型直接从全量选项里自由猜。
# 2. 先按 family 拆题型。
# 3. 先把 option label 动态解析成结构化动作。
# 4. 用表格特征做 issue score。
# 5. 再把 issue score 映射回具体 C-code。
# 6. 只在低置信度时调用少量 verifier 工具。
#
# 这套思想比原始 tool-calling baseline 更稳定，也更容易 debug。


# ------------------------------------------------------------------------------
# 7. 建议阅读顺序
# ------------------------------------------------------------------------------
#
# 如果你想读懂官方 5.4 新增代码，我建议按这个顺序：
#
# 1. skills_scenario_router_md.py
# 2. skills_scenario_parser_md.py
# 3. skills_api_data_gatherer_md.py
# 4. skills_feature_engine_md.py
# 5. skills_candidate_generator_md.py
# 6. skills_answer_parser_md.py
# 7. 本文件
#
# 原因：
# - router 决定题型
# - parser 决定数据结构
# - gatherer 决定 Phase 2 数据从哪里来
# - feature_engine 决定诊断分数
# - candidate_generator 决定 C-code
# - answer_parser 决定提交答案是否合法
