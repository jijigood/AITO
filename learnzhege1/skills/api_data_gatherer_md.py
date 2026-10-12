#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学习版：skills/api_data_gatherer.py

官方对应文件：
    /home/huangxiaolin/AITO/5.4/Track A/skills/api_data_gatherer.py

这个文件是 Phase 2 的“云端数据桥”。

它把 Phase 2 云端工具 API 拉到的数据，重新拼成一个和 Phase 1 raw scenario
兼容的 dict，这样 `scenario_parser.parse_scenario()` 可以复用。
"""


# ------------------------------------------------------------------------------
# 1. 为什么需要 APIDataGatherer
# ------------------------------------------------------------------------------
#
# Phase 1：
#     scenario["data"]["user_plane_data"] 里直接就是 CSV 字符串。
#
# Phase 2：
#     scenario["data"]["user_plane_data"] 里只是占位文案。
#     真实 CSV 要通过工具 API 获取。
#
# APIDataGatherer 做的事情就是：
#
#     云端 API 返回 -> CSV 字符串 -> raw["data"] 字段
#
# 这样后面的 parser / feature_engine 不需要区分 Phase 1 和 Phase 2。


# ------------------------------------------------------------------------------
# 2. gather() 主流程
# ------------------------------------------------------------------------------
#
# 官方 gather() 大致做这些调用：
#
# 1. get_user_plane_data
#    拉用户面原表，并从中找：
#    - worst_time：吞吐最低的时刻
#    - last_time：最后一个时间点
#    - serving_pci：worst_time 的 serving PCI
#
# 2. get_config_data
#    拉网络配置表。
#
# 3. get_signaling_plane_event_log(last_time)
#    拉到最后一个时间点为止的信令事件。
#
# 4. get_kpi_data
#    拉 traffic / KPI 表。
#
# 5. get_mr_data
#    拉 MR 测量报告表。
#
# 最后构造：
#
#     raw = dict(self.scenario)
#     raw["data"] = {
#         "user_plane_data": ...,
#         "network_configuration_data": ...,
#         "signaling_plane_data": ...,
#         "traffic_data": ...,
#         "mr_data": ...,
#         ...
#     }
#
# 输出的 raw 就能直接送进 parse_scenario(raw)。


# ------------------------------------------------------------------------------
# 3. timing 提取
# ------------------------------------------------------------------------------
#
# `_extract_timing_from_user_plane()` 会从 user plane CSV 里找：
#
# - Timestamp 列
# - Throughput 列
# - Serving PCI 列
#
# 它用 throughput 最低点作为 worst_time。
#
# 这个 worst_time 后面非常重要：
#
# - mobility skill 要看 worst_time serving/neighbor
# - mainlobe verifier 要用 worst_time
# - signaling log 会拉到 last_time


# ------------------------------------------------------------------------------
# 4. disambiguate_mainlobe()
# ------------------------------------------------------------------------------
#
# 这是一个低置信度 verifier。
#
# 当 feature_engine 觉得：
#
#     coverage_hole 分数 和 insufficient_data 分数很接近
#
# 就额外调用：
#
#     judge_mainlobe_or_not(worst_time, serving_pci)
#
# 返回字段：
#
# - Outside_Mainlobe_Flag = True
#   用户在主瓣外，更像覆盖几何问题。
#   于是提高 score_coverage_hole，压低 score_insufficient_data。
#
# - Outside_Mainlobe_Flag = False
#   用户在主瓣内，无线几何不像主因。
#   于是提高 score_insufficient_data，压低 score_coverage_hole。


# ------------------------------------------------------------------------------
# 5. get_overlap_ratio()
# ------------------------------------------------------------------------------
#
# 这是另一个辅助 verifier：
#
#     calculate_overlap_ratio(serving, neighbor)
#
# 用来帮助 F2 / overshoot / overlap 类问题区分目标 cell。
#
# 官方实现会遍历返回 dict 的 value，尝试转成 float。
# 因为这个 endpoint 的 key 是动态字符串，不适合写死。


# ------------------------------------------------------------------------------
# 6. 学习时最该记住的点
# ------------------------------------------------------------------------------
#
# APIDataGatherer 的价值不是复杂算法，而是让 Phase 2 回到 Phase 1 的数据形态。
#
# 一旦 gather() 完成，后面的 parser / feature_engine 可以像处理本地 JSON
# 一样处理云端数据。
