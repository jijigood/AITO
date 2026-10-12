#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学习版：skills/candidate_generator.py

官方对应文件：
    /home/huangxiaolin/AITO/5.4/Track A/skills/candidate_generator.py

这个文件负责把 feature_engine 的 issue score 映射成具体 C-code。

它回答的问题是：

    已经知道这题像 late_handover / coverage_hole / pdcch_issue，
    那在当前题的候选 C-code 里，应该选哪一个？
"""


# ------------------------------------------------------------------------------
# 1. 为什么需要 Candidate Generator
# ------------------------------------------------------------------------------
#
# 直接让模型从全量选项里挑，容易出错。
#
# 5.4 的思路是：
#
# 1. parser 先把 option label 解析成结构化动作。
# 2. feature_engine 给出 issue score。
# 3. candidate_generator 把 issue 映射到动作类型和目标 cell。
# 4. 最后只保留少数候选。
#
# 这相当于把：
#
#     22 选 1 / 22 选多
#
# 变成：
#
#     从 2-5 个高质量候选里选。


# ------------------------------------------------------------------------------
# 2. `_ISSUE_ACTION_MAP`
# ------------------------------------------------------------------------------
#
# 这是最核心的映射表。
#
# 例子：
#
# late_handover:
#     decrease_a3 / primary
#     decrease_a3 / secondary
#     decrease_interfreq / primary 作为弱 fallback
#
# missing_neighbor:
#     add_neighbor / primary
#     add_neighbor / any
#     decrease_a3 / primary 作为弱 fallback
#
# coverage_hole:
#     increase_power / primary
#     lift_tilt / primary
#
# pdcch_issue:
#     pdcch_2sym / primary
#
# backhaul_server:
#     check_server / any
#
# 这里的 target_role 来自 scenario_parser 推断的 cell_roles。


# ------------------------------------------------------------------------------
# 3. option scoring
# ------------------------------------------------------------------------------
#
# `_score_option()` 会比较：
#
# - 当前 option 的 action_type 是否匹配
# - 当前 option 的 target_cell 是否属于目标 role
#
# 如果完全匹配，得分高。
# 如果 action_type 对但 role 错，只给部分分。
#
# 特别处理：
#
# - check_server / insufficient_data
#   不需要 target cell，action_type 对就行。
#
# - add_neighbor
#   label 里有两个 cell。
#   官方会检查 serving_cell_id 是否在 option 的 cells 列表中，
#   用来区分“添加哪个邻区关系”。


# ------------------------------------------------------------------------------
# 4. single-answer candidate
# ------------------------------------------------------------------------------
#
# `generate_single_candidates()` 的主流程：
#
# 1. 把所有 issue score 从高到低排序。
# 2. 对每个 issue 查 `_ISSUE_ACTION_MAP`。
# 3. 找到当前题 options 中最匹配的 C-code。
# 4. 用 `_upsert_candidate()` 去重并保留更高分。
# 5. 追加每个 skill 自己推荐的候选，做少量多样化。
# 6. 针对 serving cell role 做额外修正。
# 7. 永远加入 check_server / insufficient_data 低分 fallback。
# 8. 返回 top_n 个 RankedCandidate。
#
# `RankedCandidate` 包含：
#
# - c_code
# - score
# - action_type
# - target_cell
# - reason


# ------------------------------------------------------------------------------
# 5. F4 特化逻辑
# ------------------------------------------------------------------------------
#
# `F4 = 13 BS + single-answer`
#
# 官方训练分析认为：
#
# - F4 极强地对应 inter-frequency 阈值类动作
# - 正确动作几乎总是 decrease_interfreq
# - target role 通常是 secondary
#
# 所以 `_generate_f4_candidates()` 会优先找：
#
#     decrease_interfreq / secondary
#
# 这比通用 issue score 更强。


# ------------------------------------------------------------------------------
# 6. multi-answer candidate
# ------------------------------------------------------------------------------
#
# 官方也实现了 Skill H：multi-answer composer。
#
# 核心数据结构：
#
#     AnswerSet(c_codes, score, reason)
#
# F2：
# - 多数是 2 个动作组合。
# - 主要用 `_MULTI_PAIRS` 里的动作模板组合。
#
# F3：
# - 5 BS 多答案题。
# - 通常固定输出 4 个动作。
# - 用 `_F3_4ACTION_TEMPLATES` 找 4 个 C-code。
#
# 这再次说明：
#
#     多答案题不是任意多标签分类，而是模板化组合生成。


# ------------------------------------------------------------------------------
# 7. CandidateOutput
# ------------------------------------------------------------------------------
#
# 这是顶层输出容器：
#
# - family
# - is_multiple
# - single_candidates
# - multi_candidates
#
# 它提供：
#
# - format_for_prompt()
#   把候选格式化成可塞给 LLM 的文本。
#
# - top_candidate()
#   直接取最高分候选，适合作 deterministic fallback。


# ------------------------------------------------------------------------------
# 8. 学习时最该记住的点
# ------------------------------------------------------------------------------
#
# candidate_generator 是从“问题类型”到“当前题 C-code”的桥。
#
# 它依赖两个前提：
#
# 1. scenario_parser 必须正确理解 option label。
# 2. feature_engine 必须给出相对合理的 issue score。
#
# 任何一个前提错了，最后 C-code 都会错。
