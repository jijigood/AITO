#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学习版：skills/scenario_parser.py

官方对应文件：
    /home/huangxiaolin/AITO/5.4/Track A/skills/scenario_parser.py

这个文件是 5.4 deterministic pipeline 的“数据入口”。

它负责把 raw scenario 变成 `ParsedScenario`：

- 五张 pipe-delimited CSV -> pandas DataFrame
- 每题 option label -> action_type / target_cell / cells / degrees
- network config -> PCI 和 cell_id 的双向映射
- 选项里出现的 cell -> primary / secondary / tertiary 角色
"""


# ------------------------------------------------------------------------------
# 1. 为什么不能直接使用固定 C-code 语义
# ------------------------------------------------------------------------------
#
# Track A 有一个最重要的事实：
#
#     C-code 的语义不是全局固定的。
#
# 例如：
# - 某题 C7 可能是 `Decrease A3 Offset`
# - 另一题 C7 可能是 `Decrease CovInterFreq...`
#
# 所以任何规则系统都不能写死：
#
#     C7 == 某个动作
#
# 正确做法是：
#
#     每题动态解析 task.options[*].label
#
# 这正是 `scenario_parser.py` 的第一层职责。


# ------------------------------------------------------------------------------
# 2. option label 解析
# ------------------------------------------------------------------------------
#
# 官方主要解析这些信息：
#
# - action_type
#   规范化后的动作类型，例如：
#   - check_server
#   - insufficient_data
#   - add_neighbor
#   - decrease_interfreq
#   - press_down_tilt
#   - lift_tilt
#   - pdcch_2sym
#   - adjust_azimuth
#   - increase_a3
#   - decrease_a3
#   - decrease_power
#   - increase_power
#
# - cells
#   label 里出现的所有 `gNodeBID_CellID`。
#
# - target_cell
#   官方简单取 label 中最后一个 cell。
#   对 `add_neighbor` 来说，这个约定尤其重要，因为 label 里通常有两个 cell。
#
# - degrees
#   从 label 中提取 `3 degrees` / `11 degrees` 这类调整幅度。
#
# - raw
#   原始 label，便于 debug。


# ------------------------------------------------------------------------------
# 3. cell_roles：从选项频率推断角色
# ------------------------------------------------------------------------------
#
# 官方并没有直接告诉你哪个 cell 是 primary / secondary。
#
# 于是 parser 用一个启发式方法：
#
# 1. 统计每个 cell_id 在当前题 option label 中出现了多少次。
# 2. 出现最多的叫 primary。
# 3. 第二多的叫 secondary。
# 4. 第三多的叫 tertiary。
#
# 这不是无线工程里的严格定义，而是一个“选项结构角色”。
#
# 它的作用是帮助 candidate generator 说：
#
#     我想找 decrease_a3 + primary cell 对应的那个 C-code。


# ------------------------------------------------------------------------------
# 4. ParsedScenario 容器
# ------------------------------------------------------------------------------
#
# 官方用 dataclass 存放所有标准化数据。
#
# 重要字段：
#
# - scenario_id
# - tag
# - num_base_stations
# - ground_truth
#
# - user_plane
# - net_config
# - signaling
# - traffic
# - mr
#
# - parsed_options
# - cell_roles
# - role_to_cells
#
# - pci_to_cell_id
# - cell_id_to_pci
#
# - _col_cache
#   缓存列名查找结果，避免重复做模糊匹配。


# ------------------------------------------------------------------------------
# 5. 列名兼容
# ------------------------------------------------------------------------------
#
# 官方写了 `_find_col(df, candidates)`：
#
# - 先精确匹配列名
# - 再做小写包含式的 partial match
#
# 原因是不同表里同一个概念的列名可能略有差异，例如：
#
# - `gNodeB ID`
# - `gNodeB_ID`
# - `Cell ID`
# - `Cell_ID`
#
# 这让后续 skill 不必在每个地方手写一堆列名兜底。


# ------------------------------------------------------------------------------
# 6. 常用 accessor
# ------------------------------------------------------------------------------
#
# ParsedScenario 提供了一批 helper：
#
# - get_timestamps()
# - get_throughput_series()
# - get_rsrp_series()
# - get_sinr_series()
# - get_serving_pci_series()
# - get_row_at_time(timestamp)
# - get_cell_config(pci)
# - get_neighbor_pcis_from_config(pci)
# - is_neighbor_configured(serving_pci, neighbor_pci)
# - get_cell_traffic(gnodeb_id, cell_id_num)
# - get_mr_neighbor_rsrp(neighbor_pci)
# - find_options_by(action_type, role/cell)
# - get_primary_cell_id()
# - get_secondary_cell_id()
#
# 这些函数的意义是：
# 后面的 feature_engine / candidate_generator 不需要直接关心原始表结构。


# ------------------------------------------------------------------------------
# 7. parse_scenario(raw)
# ------------------------------------------------------------------------------
#
# 这是主入口。
#
# 输入：
#     raw scenario dict
#
# 输出：
#     ParsedScenario
#
# 主流程：
#
# 1. 从 raw["data"] 读取五个 CSV 字符串。
# 2. `pd.read_csv(io.StringIO(raw_str), sep="|")` 转 DataFrame。
# 3. 从 context 读 num_base_stations。
# 4. 从 task.options 解析候选动作。
# 5. 根据 option 里 cell 出现频率推断 cell_roles。
# 6. 从 net_config 建立 PCI <-> cell_id 映射。


# ------------------------------------------------------------------------------
# 8. 学习时最该记住的点
# ------------------------------------------------------------------------------
#
# `scenario_parser.py` 是整条 deterministic pipeline 的地基。
#
# 如果 parser 对 option label 或 cell role 理解错了，
# 后面 feature_engine 即使分数正确，也可能映射到错误的 C-code。
