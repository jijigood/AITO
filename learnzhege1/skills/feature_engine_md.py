#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学习版：skills/feature_engine.py

官方对应文件：
    /home/huangxiaolin/AITO/5.4/Track A/skills/feature_engine.py

这个文件是 5.4 deterministic pipeline 的“诊断大脑”。

它不直接输出 C-code，而是输出一组结构化诊断分数：

- score_late_handover
- score_missing_neighbor
- score_inter_freq_load
- score_interference
- score_coverage_hole
- score_overshoot_overlap
- score_pdcch_issue
- score_backhaul_server
- score_insufficient_data

后面的 candidate_generator 再把这些 issue score 映射成当前题的具体 C-code。
"""


# ------------------------------------------------------------------------------
# 1. 总体结构
# ------------------------------------------------------------------------------
#
# 官方把 feature_engine 拆成 A-G 七个 skill：
#
# A. throughput_localizer
# B. mobility_skill
# C. inter_freq_skill
# D. coverage_geometry_skill
# E. power_interference_skill
# F. capacity_control_skill
# G. transport_server_skill
#
# 主入口：
#
#     run_all_skills(ps) -> ScenarioFeatures
#
# 输入 `ps` 是 scenario_parser 产出的 ParsedScenario。
# 输出 `sf` 是 ScenarioFeatures，里面同时保存原始 feature 和合并后的 issue score。


# ------------------------------------------------------------------------------
# 2. Skill A：Throughput Localizer
# ------------------------------------------------------------------------------
#
# 目标：
#     找到吞吐最低点和退化窗口。
#
# 主要字段：
#
# - worst_time
# - worst_throughput_mbps
# - drop_start_time
# - drop_end_time
# - recovery_flag
# - throughput_drop_magnitude
#
# 关键直觉：
# Track A 的很多工具都需要一个 time 参数。
# 先找到 worst_time，后面才知道该查哪个时间点的 serving PCI、RSRP、SINR、邻区。


# ------------------------------------------------------------------------------
# 3. Skill B：Mobility Skill
# ------------------------------------------------------------------------------
#
# 目标：
#     判断 late handover / missing neighbor / stuck serving。
#
# 主要观察：
#
# - worst_time 的 serving PCI
# - serving RSRP / SINR
# - visible neighbor PCI 和 RSRP
# - best neighbor 比 serving 强多少
# - A3 threshold 大约是多少
# - best neighbor 是否已经在 serving cell 配置邻区里
#
# 典型判断：
#
# - best neighbor 很强，但不在配置邻区里：
#     missing_neighbor
#
# - best neighbor 强，但 A3 条件没触发：
#     late_handover / decrease_a3
#
# - A3 margin 明显大于 0，但 UE 还没切：
#     stuck_serving，分数较弱，更多作为辅助信号。


# ------------------------------------------------------------------------------
# 4. Skill C：Inter-Frequency Skill
# ------------------------------------------------------------------------------
#
# 目标：
#     判断 inter-frequency / load balancing 类问题。
#
# 最重要的 family 先验：
#
#     13 BS 场景高度规律，几乎就是 F4 inter-frequency 题。
#
# 所以官方对 num_base_stations == 13 给了强 prior：
#
# - recommended_action_family = decrease_interfreq
# - target_cell 倾向 secondary
# - score_inter_freq_load 至少给到较高分
#
# 此外也会看 traffic_data 中的 Downlink PRB utilization。
# PRB 很高时，会增加 inter_freq_load 分数。


# ------------------------------------------------------------------------------
# 5. Skill D：Coverage Geometry Skill
# ------------------------------------------------------------------------------
#
# 目标：
#     判断覆盖几何类问题，例如 tilt mismatch / overlap。
#
# 本地规则比较轻：
#
# - RSRP 很弱，且多个邻区可见：
#     可能是 coverage hole / lift_tilt。
#
# - RSRP 不弱，但 SINR 很差，且强邻区存在：
#     可能是 overlap / interference，考虑 press_down_tilt。
#
# 更精细的 mainlobe 判断没有放在这里，而是由 APIDataGatherer 的
# `judge_mainlobe_or_not` 在低置信度时调用。


# ------------------------------------------------------------------------------
# 6. Skill E：Power / Interference Skill
# ------------------------------------------------------------------------------
#
# 目标：
#     区分 coverage_hole 和 interference。
#
# 典型规则：
#
# - serving RSRP < -110：
#     覆盖洞，倾向 increase_power。
#
# - RSRP 还可以，但 SINR < 0：
#     干扰，倾向 decrease_power 或 press_down_tilt。
#
# - RSRP < -100：
#     中等覆盖问题，给 coverage_hole 一个较弱分数。


# ------------------------------------------------------------------------------
# 7. Skill F：Capacity Control Skill
# ------------------------------------------------------------------------------
#
# 目标：
#     识别 PDCCH / CCE 控制信道容量问题。
#
# 主要看 traffic_data：
#
# - CCE utilization 是否很高
# - PdcchOccupiedSymbolNum 是否是 1SYM
#
# 官方逻辑比较保守：
# 只有 CCE 压力明显时，才强推 `pdcch_2sym`。


# ------------------------------------------------------------------------------
# 8. Skill G：Transport / Server Skill
# ------------------------------------------------------------------------------
#
# 目标：
#     判断不是无线侧，而是 server / transmission / backhaul 问题。
#
# 典型规则：
#
# - RSRP > -100
# - SINR > 5
# - 但 throughput 仍然很低
#
# 这说明无线信号看起来健康，低吞吐可能不是 RF 参数问题。
# 候选动作会倾向：
#
#     Check test server and transmission issues


# ------------------------------------------------------------------------------
# 9. ScenarioFeatures
# ------------------------------------------------------------------------------
#
# 这是最终的特征汇总对象。
#
# 它保存：
#
# - 七个 skill 的原始 dataclass 输出
# - signaling_event_counts
# - 合并后的 issue score
# - top_issue()
# - as_evidence_text()
#
# `top_issue()` 用于看最高分问题类型。
#
# `as_evidence_text()` 可以把结构化证据压成几行文本，
# 适合塞给模型或写 debug 日志。


# ------------------------------------------------------------------------------
# 10. run_all_skills()
# ------------------------------------------------------------------------------
#
# 主入口流程：
#
# 1. 跑 Skill A，找到 worst_time。
# 2. 跑 Skill B，依赖 A 的 worst_time。
# 3. 跑 Skill C，依赖 B 的 serving cell。
# 4. 跑 Skill D，依赖 A+B。
# 5. 跑 Skill E，依赖 B。
# 6. 跑 Skill F，依赖 A+B。
# 7. 跑 Skill G，依赖 A+B。
# 8. 统计 signaling events。
# 9. 合并 issue score。
# 10. 按 family / SINR / PRB 等规则做二次修正。


# ------------------------------------------------------------------------------
# 11. 5.4 里最重要的后处理规则
# ------------------------------------------------------------------------------
#
# 13 BS：
# - 强化 inter_freq_load。
# - 压低 late_handover / backhaul / overlap 等干扰项。
#
# 4/5 BS：
# - 压低 inter_freq_load。
# - 因为训练分析显示 4BS/F1/F2 里基本不该选 decrease_interfreq。
#
# 4 BS single-answer：
# - 进一步按 SINR 分组：
#
#   SINR < 0：
#       更像 handover / neighbor / interference。
#
#   SINR > 5：
#       无线质量健康，更像 capacity / server / insufficient_data。
#
# 这些后处理非常像“数据集分布先验”，不是纯无线物理公式。


# ------------------------------------------------------------------------------
# 12. 学习时最该记住的点
# ------------------------------------------------------------------------------
#
# feature_engine 的输出不是答案，而是“问题类型概率/分数”。
#
# 它解决的是：
#
#     这题像什么问题？
#
# candidate_generator 解决的才是：
#
#     在这题的候选 option 里，哪个 C-code 对应这个问题？
