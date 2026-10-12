#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Track A 学习版 server.py

这个文件是官方 `server.py` 的中文注释学习版。

它的核心定位不是“求解器”，而是：
1. 把 `Scenario.data` 里的原始表格字符串解析成 DataFrame
2. 以 FastAPI endpoint 的形式暴露一组诊断工具
3. 再通过 `/tools` 把这些工具描述成 LLM 可调用的 function schema

你可以把整个执行链理解成：
1. `utils.py` 先把 train/test JSON 读成 `Scenario`
2. `server.py` 通过 `X-Scenario-Id` 选定“当前题目”
3. agent 在 `main.py` 中通过 HTTP 调用这里的接口
4. 这些接口从当前 scenario 中抽取某张表、某个时间点、某个 PCI 的信息
5. agent 根据返回结果再决定下一步分析或最终答案

阅读建议：
- 先抓住“server 只是工具层”，不要把它和最终决策逻辑混在一起
- 再重点看 `resolve_scenario()`、`read_*_df()`、`/tools`
- 最后按 endpoint 分组理解：原始数据类、切片查询类、几何仿真类

说明：
- 这里尽量保持官方函数名、路径名、返回结构与官方接近，方便对照阅读
- 增加了大量中文注释和少量导入兜底逻辑，方便在 `learnzhege1` 中独立打开
- 这个文件主要用于学习，不建议直接替换比赛提交代码
"""

import io
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------------------------
# 导入官方的类型与工具函数
# ------------------------------------------------------------------------------
#
# 这个学习版文件放在 `learnzhege1` 目录，而不是官方 Track A 目录里。
# 为了让它在本地也能被 IDE 正确解析，这里做了一个兜底：
# - 优先使用当前 Python 路径中的 `_types` / `utils`
# - 找不到时，再尝试把项目中的 Track A 目录加入 sys.path
#
# 这样做的目的只是让“学习文件更好打开”，不是官方代码的一部分。

try:
    from _types import Scenario
    from utils import (
        _env_bool,
        df_all_or_none,
        df_first_or_none,
        get_fields_at_time,
        get_fields_before_time,
        load_scenarios,
    )
except ModuleNotFoundError:
    candidate_dirs = [
        Path(__file__).resolve().parent.parent / "dataset" / "Track A",
        Path(
            "/home/huangxiaolin/.cache/huggingface/hub/"
            "datasets--netop--Telco-Troubleshooting-Agentic-Challenge/"
            "snapshots/1d6d3533ee03126f67e7f2f6bc23f1c66a192a13/Track A"
        ),
    ]
    for candidate in candidate_dirs:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)

    from _types import Scenario
    from utils import (
        _env_bool,
        df_all_or_none,
        df_first_or_none,
        get_fields_at_time,
        get_fields_before_time,
        load_scenarios,
    )


# ------------------------------------------------------------------------------
# FastAPI 应用对象
# ------------------------------------------------------------------------------

app = FastAPI(title="5G Drive-Test Analysis API", version="1.2.0")


# ------------------------------------------------------------------------------
# 配置项
# ------------------------------------------------------------------------------
#
# 这些参数来自环境变量。
# 其中最关键的是：
# - `DATA_SOURCE`：指向 `train.json` / `test.json` 所在目录
# - `DATA_SPLIT`：决定加载 train 还是 test
#
# `DOWNLOAD_FROM_REMOTE` 在官方代码里保留了下来，但本文件中并没有真正用到。

DOWNLOAD_FROM_REMOTE = _env_bool("DOWNLOAD_FROM_REMOTE", default=False)
FASTAPI_HOST = os.environ.get("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.environ.get("FASTAPI_PORT", "7860"))
DATA_SOURCE = os.environ.get("DATA_SOURCE", "data/Phase_1")
DATA_SPLIT = os.environ.get("DATA_SPLIT", "test")  # 若想本地评估训练集，可切到 train


# ------------------------------------------------------------------------------
# 加载全部 Scenario
# ------------------------------------------------------------------------------
#
# 这里在服务启动时一次性把整个 split 读到内存里：
# - `SCENARIOS`：所有题目的对象列表
# - `DEFAULT_SCENARIO_ID`：默认题号
#
# 后面所有 endpoint 都不是自己再去读文件，而是从这里取“当前 scenario”。

SCENARIOS, DEFAULT_SCENARIO_ID = load_scenarios(DATA_SOURCE, split=DATA_SPLIT)


# ------------------------------------------------------------------------------
# CORS 中间件
# ------------------------------------------------------------------------------
#
# 允许任意来源访问，方便本地 agent、浏览器或 notebook 直接调试。

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Scenario 上下文：通过 X-Scenario-Id 选择“当前题目”
# ------------------------------------------------------------------------------

def get_scenario_by_id(scenario_id: Optional[str]) -> Scenario:
    """
    按 `scenario_id` 从已经加载的 `SCENARIOS` 中找到对应题目。

    设计要点：
    - 如果外部没有显式给 `scenario_id`，就回退到默认题目
    - 如果找不到，就抛 404

    这意味着 server 本身并不维护复杂会话状态；
    它只是“收到一个 scenario id，就在对应题目上执行查询”。
    """

    sid = scenario_id or DEFAULT_SCENARIO_ID
    for s in SCENARIOS:
        if s.scenario_id == sid:
            return s
    raise HTTPException(status_code=404, detail=f"Scenario '{sid}' not found")


def resolve_scenario(
    x_scenario_id: Optional[str] = Header(default=None, alias="X-Scenario-Id"),
) -> Scenario:
    """
    FastAPI 依赖项：从请求头 `X-Scenario-Id` 中解析当前 scenario。

    为什么是 Header 而不是 Query 参数？
    - 因为官方希望“工具参数只描述业务入参”，不要把 `scenario_id` 混进每个工具里
    - 所以 `/tools` 返回的 schema 中也不会出现 `scenario_id`

    这点非常关键：
    agent 调每个工具时，除了参数本身，还必须带对 `X-Scenario-Id` 这个请求头的管理。
    """

    return get_scenario_by_id(x_scenario_id)


# ------------------------------------------------------------------------------
# 数据读取辅助函数：把 Scenario.data 里的字符串转成 DataFrame
# ------------------------------------------------------------------------------
#
# 这是 server.py 最底层、最重要的一层。
# 官方 Track A 的原始观测数据并不是单独的 CSV 文件，而是：
# - 已经嵌在 scenario JSON 中
# - 以 `|` 为分隔符
# - 在需要时才解析成 DataFrame
#
# 你后面自己做的 parser / feature engine，本质上也是沿着这个思路演化出来的。

def read_config_df(scenario: Scenario) -> pd.DataFrame:
    """
    读取网络配置表。

    主要包含：
    - PCI
    - 经度/纬度
    - 天线方位角、下倾角、高度
    - 发射功率
    - 邻区配置
    - A3/A2/A5 等切换参数
    - PDCCH 配置等

    大多数“配置类”和“几何类”工具都依赖这张表。
    """

    data = scenario.data.network_configuration_data
    if not data:
        raise HTTPException(status_code=404, detail="No network configuration data in scenario")

    df = pd.read_csv(io.StringIO(data), sep="|")
    if "PCI" in df.columns:
        df["PCI"] = pd.to_numeric(df["PCI"], errors="coerce")
    return df


def read_user_plane_df(scenario: Scenario) -> pd.DataFrame:
    """
    读取用户面数据表。

    这张表基本是 Track A 最常用的一张表，典型字段包括：
    - Timestamp
    - 用户经纬度
    - Serving PCI
    - Serving RSRP / SINR
    - Throughput
    - 邻区 Top1~Top5 PCI 及对应 RSRP
    - RB 数量
    - GPS Speed

    大部分“按时间点查某个观测值”的工具都来自这里。
    """

    data = scenario.data.user_plane_data
    if not data:
        raise HTTPException(status_code=404, detail="No user-plane data in scenario")

    df = pd.read_csv(io.StringIO(data), sep="|")
    if "5G KPI PCell RF Serving PCI" in df.columns:
        df["5G KPI PCell RF Serving PCI"] = pd.to_numeric(
            df["5G KPI PCell RF Serving PCI"], errors="coerce"
        )
    return df


def read_signaling_plane_df(scenario: Scenario) -> pd.DataFrame:
    """
    读取信令面数据表。

    这张表主要用于：
    - 看切换相关事件
    - 看某个时间点之前发生过哪些 signaling 事件
    """

    data = scenario.data.signaling_plane_data
    if not data:
        raise HTTPException(status_code=404, detail="The signaling-plane data is not available for this scenario")
    df = pd.read_csv(io.StringIO(data), sep="|")
    return df


def read_traffic_df(scenario: Scenario) -> pd.DataFrame:
    """
    读取 KPI / Traffic 数据表。

    主要用于更偏 cell-level KPI 的分析，例如：
    - PRB 利用率
    - 吞吐
    - 干扰
    - CCE 利用率等
    """

    data = scenario.data.traffic_data
    if not data:
        raise HTTPException(status_code=404, detail="The traffic data is not available for this scenario")
    df = pd.read_csv(io.StringIO(data), sep="|")
    return df


def read_mr_df(scenario: Scenario) -> pd.DataFrame:
    """
    读取 MR（Measurement Report）数据表。

    这张表通常用于围绕差体验用户的测量报告分析。

    注意：
    官方这里的报错文案仍然写成了 traffic data，像是一个小笔误；
    我们保留原行为，只在注释里说明。
    """

    data = scenario.data.mr_data
    if not data:
        raise HTTPException(status_code=404, detail="The traffic data is not available for this scenario")
    df = pd.read_csv(io.StringIO(data), sep="|")
    return df


# ------------------------------------------------------------------------------
# 几何 / 天线 / 覆盖辅助函数
# ------------------------------------------------------------------------------
#
# 这一组函数的特点是：
# - 不直接返回业务答案
# - 而是给上层工具提供几何量、覆盖量或简单仿真量
#
# 在官方思路里，这些函数更多是“辅助诊断器”。

def gain_pattern(phi, theta):
    """
    一个简化的天线增益方向图模型。

    输入：
    - `phi`：方位角（弧度）
    - `theta`：俯仰角（弧度）

    输出：
    - 线性域下的相对增益

    这个模型不是严格的工程级商用天线建模，
    而是一个足够支撑比赛诊断逻辑的近似模式：
    - 主瓣：较尖
    - 背瓣/旁瓣：较弱
    - 再加一个极小的各向同性背景值
    """

    phi0 = np.radians(180)
    theta0 = np.radians(90)

    d_phi = np.abs(phi - phi0)
    d_phi = np.minimum(d_phi, 2 * np.pi - d_phi)
    d_theta = theta - theta0

    sigma_phi = np.radians(15)
    sigma_theta = np.radians(15)

    main_lobe = np.exp(-0.5 * ((d_phi / sigma_phi) ** 4) - 0.5 * ((d_theta / sigma_theta) ** 4))
    side_lobe_phi = np.abs(phi - (phi0 + np.pi))
    side_lobe_phi = np.minimum(side_lobe_phi, 2 * np.pi - side_lobe_phi)
    side_lobe_theta = theta - theta0
    side_lobe = 0.1 * np.exp(-0.5 * (side_lobe_phi / sigma_phi) ** 2 - 0.5 * (side_lobe_theta / sigma_theta) ** 2)

    isotropic = 0.001
    return main_lobe + side_lobe + isotropic


def beam_scenario_to_v_bw(name: str) -> float:
    """
    根据 beam scenario 名称映射出垂直波束宽度（degree）。

    官方把 beam scenario 名称粗略映射成三档：
    - DEFAULT / 1~5  -> 6 度
    - 6~11           -> 12 度
    - 12 及以上      -> 25 度

    这个函数更多是一个规则映射器，而不是复杂计算。
    """

    if name == "DEFAULT":
        return 6.0
    try:
        suffix = int(name.split("_")[-1])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid beam scenario name.")
    if 1 <= suffix <= 5:
        return 6.0
    if 6 <= suffix <= 11:
        return 12.0
    if suffix >= 12:
        return 25.0
    raise HTTPException(status_code=400, detail="Invalid beam scenario index.")


def calculate_overlap_coverage(
    lon1,
    lat1,
    azimuth1,
    lon2,
    lat2,
    azimuth2,
    radius: float = 900.0,
) -> float:
    """
    估算两个小区覆盖扇区的重叠比例。

    思路：
    1. 用经纬度近似换算成平面坐标差
    2. 用两个等半径圆的交叠面积近似“覆盖交叠”
    3. 再用方位角差异做一个衰减因子

    这不是严格的无线传播仿真，而是一个可解释、可快速计算的近似指标。
    """

    if lon1 == 0 or lat1 == 0 or lon2 == 0 or lat2 == 0:
        return 0.0

    lat_to_meter = 111320.0
    lon_to_meter = 111320.0 * math.cos(math.radians(lat1))

    x2 = (lon2 - lon1) * lon_to_meter
    y2 = (lat2 - lat1) * lat_to_meter

    distance = math.sqrt(x2 ** 2 + y2 ** 2)

    if distance >= 2 * radius:
        return 0.0

    if distance < 0.1:
        return 1.0

    half_circle_area = 0.5 * math.pi * radius ** 2

    r1 = radius
    r2 = radius
    d = distance

    if d >= r1 + r2:
        overlap_area = 0.0
    elif d <= abs(r1 - r2):
        overlap_area = math.pi * min(r1, r2) ** 2
    else:
        r1_sq = r1 ** 2
        r2_sq = r2 ** 2
        d_sq = d ** 2

        alpha = math.acos((r1_sq + d_sq - r2_sq) / (2 * r1 * d))
        beta = math.acos((r2_sq + d_sq - r1_sq) / (2 * r2 * d))

        overlap_area = r1_sq * alpha + r2_sq * beta - 0.5 * math.sqrt(
            (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
        )

    azimuth_diff = abs(azimuth1 - azimuth2)
    azimuth_diff = min(azimuth_diff, 360 - azimuth_diff)
    if azimuth_diff > 180:
        azimuth_diff = 360 - azimuth_diff

    azimuth_factor = max(0, 1 - azimuth_diff / 180)

    overlap_coverage = (overlap_area / half_circle_area) * azimuth_factor

    return min(1.0, max(0.0, overlap_coverage))


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算从点 1 指向点 2 的方位角（bearing）。

    返回值范围是 [0, 360) 度。
    在本文件中，它主要用于：
    - 先算“基站朝向用户的真实方向”
    - 再与小区机械方位角做差
    """

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    return bearing


def calculate_angle_between_user_and_cell(
    user_lon: float,
    user_lat: float,
    cell_lon: float,
    cell_lat: float,
    cell_azimuth: int,
) -> float:
    """
    计算用户方向相对于小区主方向的水平夹角。

    理解方式：
    - 先求“小区指向用户”的真实方位角
    - 再和小区的机械方位角比较
    - 最终输出最小夹角（0~180）

    该量在判断是否偏离主瓣时非常重要。
    """

    bearing_to_user = calculate_bearing(cell_lat, cell_lon, user_lat, user_lon)
    angle_diff = abs(bearing_to_user - cell_azimuth)
    angle_diff = angle_diff % 360
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    return angle_diff


def haversine_distance(point_lon, point_lat, cell_lon, cell_lat):
    """
    使用 Haversine 公式计算地表两点间距离，返回单位是 km。

    这里主要用于：
    - 估算用户到小区的空间距离
    - 进一步推导俯仰角 / 倾角
    """

    r = 6371.0

    lat1 = math.radians(point_lat)
    lon1 = math.radians(point_lon)
    lat2 = math.radians(cell_lat)
    lon2 = math.radians(cell_lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    distance = r * c

    return distance


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
#
# 下面开始是对外暴露的 HTTP 接口。
# 你可以把它们理解成三层：
# 1. 原始数据导出接口
# 2. 按时间 / PCI 切片后的诊断接口
# 3. 几何 / 覆盖 / 仿真接口


@app.get("/health")
def health():
    """
    健康检查接口。

    一般用于：
    - 确认 server 是否成功启动
    - 让外部 agent 在正式运行前做联通性检查
    """

    return {"status": "ok"}


@app.get("/scenario")
def get_scenario(scenario: Scenario = Depends(resolve_scenario)):
    """
    返回当前 scenario 的完整内容。

    这里用 `model_dump()` 而不是直接返回 Pydantic 对象，
    是为了保证输出是标准 JSON 可序列化 dict。
    """

    return scenario.model_dump()


@app.get("/scenario/random")
def get_random_scenario():
    """
    随机返回一个 scenario。

    更适合 demo / 调试，不太是正式评测主路径的一部分。
    """

    return random.choice(SCENARIOS).model_dump()


@app.get("/scenario/all")
def get_all_scenario():
    """
    返回当前 split 的全部 scenario。

    官方 `main.py` 就是通过这个接口一次性取到所有题目。
    """

    return [s.model_dump() for s in SCENARIOS]


@app.get("/config-data/json")
def get_config_data_json(scenario: Scenario = Depends(resolve_scenario)):
    """
    返回结构化的网络配置表。

    与 `/config-data` 的区别：
    - 这个接口返回 JSON rows
    - 更适合程序直接消费
    """

    df = read_config_df(scenario)
    return {"rows": df.to_dict(orient="records")}


@app.get("/config-data")
def get_config_data(scenario: Scenario = Depends(resolve_scenario)):
    """
    原样返回网络配置表字符串。

    返回的是 pipe-delimited CSV 原文，而不是 DataFrame 转成的记录列表。
    更适合“想拿到整张原始表”的场景。
    """

    data = scenario.data.network_configuration_data
    if not data:
        raise HTTPException(status_code=404, detail="No network configuration data in scenario")
    return {"Network Configuration Data": data}


@app.get("/user-plane-data/json")
def get_user_plane_data_json(scenario: Scenario = Depends(resolve_scenario)):
    """
    返回结构化的用户面表。
    """

    df = read_user_plane_df(scenario)
    return {"rows": df.to_dict(orient="records")}


@app.get("/user-plane-data")
def get_user_plane_data(scenario: Scenario = Depends(resolve_scenario)):
    """
    原样返回用户面数据表字符串。
    """

    data = scenario.data.user_plane_data
    if not data:
        raise HTTPException(status_code=404, detail="No user-plane data in scenario")
    return {"User Plane Data": data}


@app.get("/get_kpi_data")
def get_kpi_data(scenario: Scenario = Depends(resolve_scenario)):
    """
    原样返回 traffic / KPI 数据表。

    这里名字叫 `get_kpi_data`，但底层读的是 `traffic_data`。
    """

    data = scenario.data.traffic_data
    if not data:
        raise HTTPException(status_code=404, detail="No traffic data in scenario")
    return {"Traffic Data": data}


@app.get("/get_mr_data")
def get_mr_data(scenario: Scenario = Depends(resolve_scenario)):
    """
    原样返回 MR 数据表。
    """

    data = scenario.data.mr_data
    if not data:
        raise HTTPException(status_code=404, detail="No traffic data in scenario")
    return {"MR Data": data}


@app.get("/throughput-logs")
def get_throughput_logs(scenario: Scenario = Depends(resolve_scenario)):
    """
    只提取吞吐日志。

    为什么它重要？
    - 官方 baseline 的第一步经常是先看吞吐
    - 找到体验差的时间点后，再围绕那个时间点继续查 Serving / Neighbor / Signal 等

    输出仍然是 `|` 分隔的文本，而不是 JSON 列表。
    """

    df = read_user_plane_df(scenario)
    return {"Logs": df[["Timestamp", "5G KPI PCell Layer2 MAC DL Throughput [Mbps]"]].to_csv(index=False, sep="|")}


@app.get("/cell-info")
def get_cell_info(
    pci: int = Query(..., description="Physical Cell ID"),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    按 PCI 返回完整小区配置。

    这是最重要的配置查询工具之一。
    一旦 agent 锁定某个 serving / neighbor PCI，就可以通过这个接口拿到：
    - 小区位置
    - 方位角 / 倾角 / 高度
    - 频点 / 带宽 / 发射功率
    - 切换参数
    - 邻区关系
    - PDCCH 占用符号数
    等完整配置
    """

    df = read_config_df(scenario)
    info = df.loc[df["PCI"] == int(pci)]
    result = df_first_or_none(info)
    if not result:
        raise HTTPException(status_code=404, detail=f"No cell found for PCI {pci}")

    # 官方这里顺手把 scenario 附带备注也塞进结果里，方便补充解释。
    result["notes"] = scenario.data.notes
    return result


@app.get("/gnodeb-location")
def get_gnodeb_location(
    pci: int = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    按 PCI 返回小区经纬度。
    """

    df = read_config_df(scenario)
    info = df.loc[df["PCI"] == int(pci), ["Longitude", "Latitude"]]
    result = df_first_or_none(info)
    if not result:
        raise HTTPException(status_code=404, detail=f"No location found for PCI {pci}")
    return result


@app.get("/user-location")
def get_user_location(
    time: str = Query(..., description="Timestamp, e.g. '2024-08-25 19:30:57.500'"),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    返回某个时间点 UE 的经纬度。

    注意：
    Track A 很多后续工具都依赖“先有时间点”这个前提，
    所以 `get_throughput_logs` 常常是整条链路的起点。
    """

    df = read_user_plane_df(scenario)
    info = get_fields_at_time(df, time, fields=["Longitude", "Latitude"])
    result = df_first_or_none(info)
    if not result:
        raise HTTPException(status_code=404, detail=f"No user location found for time {time}")
    return result


@app.get("/judge_mainlobe")
def judge_mainlobe_or_not(
    time: str = Query(..., description="Timestamp, e.g. '2024-08-25 19:30:57.500'"),
    pci: int = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    判断在某个时间点，用户是否处于指定小区主瓣之外。

    实现思路：
    1. 从配置表查小区经纬度和机械方位角
    2. 从用户面表查该时刻 UE 位置
    3. 算水平夹角
    4. 再结合小区高度、机械下倾角和用户位置算俯仰方向偏差
    5. 如果水平偏差过大或俯仰偏差过大，就判为主瓣外

    返回字段虽然叫 `Outside_Mainlobe_Flag`，
    但 `/tools` 中工具名叫 `judge_mainlobe_or_not`，要注意这两个层级的区别。
    """

    df = read_config_df(scenario)
    info = df.loc[df["PCI"] == int(pci), ["Longitude", "Latitude"]]
    bs_location = df_first_or_none(info)
    info2 = df.loc[df["PCI"] == int(pci), ["Mechanical Azimuth"]]
    bs_azimuth = df_first_or_none(info2)

    df = read_user_plane_df(scenario)
    info = get_fields_at_time(df, time, fields=["Longitude", "Latitude"])
    user_location = df_first_or_none(info)

    if not bs_location or not user_location:
        raise HTTPException(status_code=404, detail=f"No location found for PCI {pci}")

    horizontal_angle = calculate_angle_between_user_and_cell(
        user_location["Longitude"],
        user_location["Latitude"],
        float(bs_location["Longitude"]),
        float(bs_location["Latitude"]),
        float(bs_azimuth["Mechanical Azimuth"]),
    )

    df = read_config_df(scenario)
    info2 = df.loc[df["PCI"] == int(pci), ["Mechanical Downtilt"]]
    bs_tilt = df_first_or_none(info2)
    info3 = df.loc[df["PCI"] == int(pci), ["Height"]]
    bs_height = df_first_or_none(info3)

    dist_km = haversine_distance(
        user_location["Longitude"],
        user_location["Latitude"],
        float(bs_location["Longitude"]),
        float(bs_location["Latitude"]),
    )

    dist_m = dist_km * 1000
    height_diff = 0 - bs_height["Height"]

    if dist_m == 0:
        tilt_angle = math.copysign(90, height_diff)
    else:
        tilt_angle = math.degrees(math.atan2(height_diff, dist_m))

    final_tilt = tilt_angle - bs_tilt["Mechanical Downtilt"]

    return {"Outside_Mainlobe_Flag": (abs(horizontal_angle) > 50) or (abs(final_tilt) > 6)}


@app.get("/optimize_antenna_gain")
def optimize_antenna_gain(
    time: str = Query(..., description="Timestamp, e.g. '2024-08-25 19:30:57.500'"),
    pci: int = Query(...),
    adjust_horizontal_angle: float = Query(...),
    adjust_tilt_angle: float = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    粗略模拟“如果调整天线方位角和下倾角，用户方向上的增益会变化多少”。

    这是官方提供的一个很有意思的工具：
    - 它不是直接改配置
    - 而是用简化天线方向图，估算调整前后在“用户所在方向”上的增益差

    注意：
    这个接口只是一个近似仿真器，不是完整的无线网络数字孪生。
    """

    df = read_config_df(scenario)
    info = df.loc[df["PCI"] == int(pci), ["Longitude", "Latitude"]]
    bs_location = df_first_or_none(info)
    info2 = df.loc[df["PCI"] == int(pci), ["Mechanical Azimuth"]]
    bs_azimuth = df_first_or_none(info2)

    df = read_user_plane_df(scenario)
    info = get_fields_at_time(df, time, fields=["Longitude", "Latitude"])
    user_location = df_first_or_none(info)

    if not bs_location or not user_location:
        raise HTTPException(status_code=404, detail=f"No location found for PCI {pci}")

    horizontal_angle = calculate_angle_between_user_and_cell(
        user_location["Longitude"],
        user_location["Latitude"],
        float(bs_location["Longitude"]),
        float(bs_location["Latitude"]),
        float(bs_azimuth["Mechanical Azimuth"]),
    )

    df = read_config_df(scenario)
    info2 = df.loc[df["PCI"] == int(pci), ["Mechanical Downtilt"]]
    bs_tilt = df_first_or_none(info2)
    info3 = df.loc[df["PCI"] == int(pci), ["Height"]]
    bs_height = df_first_or_none(info3)

    dist_km = haversine_distance(
        user_location["Longitude"],
        user_location["Latitude"],
        float(bs_location["Longitude"]),
        float(bs_location["Latitude"]),
    )

    dist_m = dist_km * 1000
    height_diff = 0 - bs_height["Height"]

    if dist_m == 0:
        tilt_angle = math.copysign(90, height_diff)
    else:
        tilt_angle = math.degrees(math.atan2(height_diff, dist_m))

    final_tilt = tilt_angle - bs_tilt["Mechanical Downtilt"]

    # 生成一个简单的二维方向图（phi, theta）。
    phi = np.linspace(0, 360, 361)
    theta = np.linspace(0, 180, 181)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    phi_rad = np.radians(phi_grid)
    theta_rad = np.radians(theta_grid)
    g_lin = gain_pattern(phi_rad, theta_rad)
    g_lin = g_lin / np.max(g_lin)
    g_db = 10 * np.log10(g_lin + 1e-12)

    serving_horizontal_location = 180 - abs(round(horizontal_angle))
    serving_tilt_location = 90 - abs(round(final_tilt))
    original_serving_gain = g_db[serving_tilt_location, serving_horizontal_location]
    after_serving_gain = g_db[
        serving_tilt_location + int(adjust_tilt_angle),
        serving_horizontal_location + int(adjust_horizontal_angle),
    ]

    adjust_gain = after_serving_gain - original_serving_gain

    return {"adjust_gain": adjust_gain}


@app.get("/calculate_tilt_angle")
def calculate_tilt_angle(
    time: str = Query(..., description="Timestamp, e.g. '2024-08-25 19:30:57.500'"),
    pci: int = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    计算用户相对于小区的俯仰角偏差。

    结果本质上是：
    - “基站指向用户的几何俯仰角”
    - 再减去当前小区机械下倾角
    """

    df = read_config_df(scenario)
    info = df.loc[df["PCI"] == int(pci), ["Longitude", "Latitude"]]
    bs_location = df_first_or_none(info)
    info2 = df.loc[df["PCI"] == int(pci), ["Mechanical Downtilt"]]
    bs_tilt = df_first_or_none(info2)
    info3 = df.loc[df["PCI"] == int(pci), ["Height"]]
    bs_height = df_first_or_none(info3)

    df = read_user_plane_df(scenario)
    info = get_fields_at_time(df, time, fields=["Longitude", "Latitude"])
    user_location = df_first_or_none(info)

    if not bs_location or not user_location:
        raise HTTPException(status_code=404, detail=f"No location found for PCI {pci}")

    dist_km = haversine_distance(
        user_location["Longitude"],
        user_location["Latitude"],
        float(bs_location["Longitude"]),
        float(bs_location["Latitude"]),
    )

    dist_m = dist_km * 1000
    height_diff = 0 - bs_height["Height"]

    if dist_m == 0:
        tilt_angle = math.copysign(90, height_diff)
    else:
        tilt_angle = math.degrees(math.atan2(height_diff, dist_m))

    final_tilt = tilt_angle - bs_tilt["Mechanical Downtilt"]

    return {"The tilt angle between the user and the cell is ": "{:.2f}".format(final_tilt)}


@app.get("/calculate_horizontal_angle")
def calculate_horizontal_angle(
    time: str = Query(..., description="Timestamp, e.g. '2024-08-25 19:30:57.500'"),
    pci: int = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    计算用户相对于小区方向图主方向的水平夹角。
    """

    df = read_config_df(scenario)
    info = df.loc[df["PCI"] == int(pci), ["Longitude", "Latitude"]]
    bs_location = df_first_or_none(info)
    info2 = df.loc[df["PCI"] == int(pci), ["Mechanical Azimuth"]]
    bs_azimuth = df_first_or_none(info2)

    df = read_user_plane_df(scenario)
    info = get_fields_at_time(df, time, fields=["Longitude", "Latitude"])
    user_location = df_first_or_none(info)

    if not bs_location or not user_location:
        raise HTTPException(status_code=404, detail=f"No location found for PCI {pci}")

    angle = calculate_angle_between_user_and_cell(
        user_location["Longitude"],
        user_location["Latitude"],
        float(bs_location["Longitude"]),
        float(bs_location["Latitude"]),
        float(bs_azimuth["Mechanical Azimuth"]),
    )

    return {"The horizontal angle between the user and the cell is ": "{:.2f}".format(angle)}


@app.get("/calculate_pathloss")
def calculate_pathloss(
    time: str = Query(..., description="Timestamp, e.g. '2024-08-25 19:30:57.500'"),
    pci: int = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    用一个非常简化的方式估算 pathloss：
    - 小区发射功率
    - 减去用户侧观测到的 serving RSRP

    这更像是一个“粗略链路损耗指标”，而不是严格的工程定义 pathloss。
    """

    df = read_config_df(scenario)
    info2 = df.loc[df["PCI"] == int(pci), ["Transmission Power"]]
    bs_power = df_first_or_none(info2)

    df = read_user_plane_df(scenario)
    info = get_fields_at_time(df, time, fields=["5G KPI PCell RF Serving SS-RSRP [dBm]"])
    user_power = df_first_or_none(info)

    if not bs_power or not user_power:
        raise HTTPException(status_code=404, detail=f"No information found for PCI {pci}")

    return {
        "The pathloss between the user and the cell is ": "{:.1f}".format(
            bs_power["Transmission Power"] - user_power["5G KPI PCell RF Serving SS-RSRP [dBm]"]
        )
    }


@app.get("/calculate_overlap_ratio")
def calculate_overlap_ratio(
    pci_serving: int = Query(...),
    pci_neighbor: int = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    估算 serving 小区和 neighbor 小区的覆盖重叠比例。

    用途：
    - 在分析覆盖重叠、越区覆盖、切换问题时提供辅助判断
    """

    df = read_config_df(scenario)
    info = df.loc[df["PCI"] == int(pci_serving), ["Longitude", "Latitude", "Mechanical Azimuth"]]
    serving_information = df_first_or_none(info)

    df = read_config_df(scenario)
    info = df.loc[df["PCI"] == int(pci_neighbor), ["Longitude", "Latitude", "Mechanical Azimuth"]]
    neighbor_information = df_first_or_none(info)

    if not serving_information or not neighbor_information:
        raise HTTPException(status_code=404, detail=f"No information found for PCI {pci_serving} or {pci_neighbor}")

    overlap_ratio = calculate_overlap_coverage(
        serving_information["Longitude"],
        serving_information["Latitude"],
        serving_information["Mechanical Azimuth"],
        neighbor_information["Longitude"],
        neighbor_information["Latitude"],
        neighbor_information["Mechanical Azimuth"],
    )

    return {
        f"The overlap coverage ratio of the cell {pci_serving} and the cell {pci_neighbor} is ": "{:.2f}".format(
            overlap_ratio
        )
    }


@app.get("/user-speed")
def get_user_speed(
    time: str = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    返回某个时间点的 GPS 速度。
    """

    df = read_user_plane_df(scenario)
    info = get_fields_at_time(df, time, fields=["GPS Speed (km/h)"])
    result = df_first_or_none(info)
    if not result:
        raise HTTPException(status_code=404, detail=f"No speed found for time {time}")

    try:
        return {"GPS Speed (km/h)": float(result["GPS Speed (km/h)"])}
    except Exception:
        return {"GPS Speed (km/h)": "Unknown"}


@app.get("/serving-cell-pci")
def get_serving_cell_pci(
    time: str = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    返回某个时间点的 serving PCI。

    这是很多分析链里的第二步或第三步：
    - 先找到问题时间点
    - 再查当时 serving 是谁
    """

    df = read_user_plane_df(scenario)
    info = get_fields_at_time(df, time, fields=["5G KPI PCell RF Serving PCI"])
    result = df_first_or_none(info)
    if not result:
        raise HTTPException(status_code=404, detail=f"No serving cell PCI found for time {time}")
    return {"PCI": int(result["5G KPI PCell RF Serving PCI"])}


@app.get("/serving-cell-rsrp")
def get_serving_cell_rsrp(
    time: str = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    返回某个时间点的 serving SS-RSRP。
    """

    df = read_user_plane_df(scenario)
    info = get_fields_at_time(df, time, fields=["5G KPI PCell RF Serving SS-RSRP [dBm]"])
    result = df_first_or_none(info)
    if not result:
        raise HTTPException(status_code=404, detail=f"Unable to retrieve the serving cell RSRP at {time}")
    return {"SS-RSRP (dBm)": float(result["5G KPI PCell RF Serving SS-RSRP [dBm]"])}


@app.get("/serving-cell-sinr")
def get_serving_cell_sinr(
    time: str = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    返回某个时间点的 serving SS-SINR。
    """

    df = read_user_plane_df(scenario)
    info = get_fields_at_time(df, time, fields=["5G KPI PCell RF Serving SS-SINR [dB]"])
    sinr_dict = df_first_or_none(info)
    if not sinr_dict:
        raise HTTPException(status_code=404, detail=f"Unable to retrieve the serving cell SINR at {time}")
    return {"SS-SINR (dB)": float(sinr_dict["5G KPI PCell RF Serving SS-SINR [dB]"])}


@app.get("/rbs-allocated-to-user")
def get_rbs_allocated_to_user(
    time: str = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    返回某个时间点分配给用户的 RB 数量。

    在拥塞、调度资源不足类问题中，它可以作为一个辅助观察量。
    """

    df = read_user_plane_df(scenario)
    info = get_fields_at_time(df, time, fields=["5G KPI PCell Layer1 DL RB Num (Including 0)"])
    result = df_first_or_none(info)
    if not result:
        raise HTTPException(status_code=404, detail=f"Unable to retrieve the number of RBs allocated at {time}")
    return {"RBs": float(result["5G KPI PCell Layer1 DL RB Num (Including 0)"])}


@app.get("/signaling-plane-event-log")
def get_signaling_plane_event_log(
    time: str = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    返回某个时间点之前的信令事件日志。

    这里用的是 `get_fields_before_time()`，不是“刚好等于某个时间点”。
    原因很自然：
    - 信令事件通常是一个过程
    - 分析切换失败、重选、异常事件时，经常需要看“到这个时刻为止发生了什么”
    """

    df = read_signaling_plane_df(scenario)
    info = get_fields_before_time(df, time, fields=["Timestamp", "Event Name", "Event Content"])
    result = df_all_or_none(info)
    if not result:
        raise HTTPException(status_code=404, detail=f"Unable to retrieve the signaling plane data at {time}")
    return result


@app.get("/neighboring-cells-pci")
def get_neighboring_cells_pci(
    time: str = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    返回某个时间点 serving cell 对应的邻区 PCI 列表。

    数据来源并不是配置表里的静态邻区定义，
    而是用户面测量里的 Top1~Top5 邻区观测结果。
    """

    df = read_user_plane_df(scenario)
    fields = [
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 PCI",
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 PCI",
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 PCI",
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 4 PCI",
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 5 PCI",
    ]
    info = get_fields_at_time(df, time, fields=fields)
    pci_dict = df_first_or_none(info)
    if not pci_dict:
        raise HTTPException(status_code=404, detail=f"Unable to retrieve the neighboring cells at {time}")

    return {"PCIs": [int(r) for r in pci_dict.values() if r != "-"]}


@app.get("/neighboring-cell-rsrp")
def get_neighboring_cell_rsrp(
    time: str = Query(...),
    pci: int = Query(...),
    scenario: Scenario = Depends(resolve_scenario),
):
    """
    返回某个时间点指定邻区 PCI 的 RSRP。

    实现方式有点“列名映射”的味道：
    1. 先在 Top1~Top5 PCI 列里找到目标 PCI 位于哪个 TopN
    2. 再把对应列名里的 `PCI` 替换成 `Filtered Tx BRSRP [dBm]`
    3. 读取相同 TopN 位置的邻区 RSRP

    这是官方表结构驱动的一个典型写法。
    """

    df = read_user_plane_df(scenario)
    fields = [
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 PCI",
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 PCI",
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 PCI",
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 4 PCI",
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 5 PCI",
    ]

    info = get_fields_at_time(df, time, fields=fields)
    pci_dict = df_first_or_none(info)
    if not pci_dict:
        raise HTTPException(status_code=404, detail=f"Unable to retrieve the neighboring cells at {time}")

    for key, cell_pci in pci_dict.items():
        if str(cell_pci) == str(pci):
            label = key.replace("PCI", "Filtered Tx BRSRP [dBm]")
            cell_rsrp = df_first_or_none(get_fields_at_time(df, time, fields=[label]))
            if not cell_rsrp:
                raise HTTPException(status_code=404, detail=f"RSRP not found for PCI {pci} at {time}")
            return {"Filtered Tx BRSRP [dBm]": float(cell_rsrp[label])}

    raise HTTPException(
        status_code=404,
        detail=f"Cell with PCI {pci} was not found in the neighboring cells at {time}",
    )


@app.get("/all-cells-pci")
def get_all_cells_pci(scenario: Scenario = Depends(resolve_scenario)):
    """
    返回配置表中全部小区的 PCI 列表。
    """

    df = read_config_df(scenario)
    return {"PCIs": [int(x) for x in df["PCI"].dropna().astype(int).tolist()]}


@app.get("/beam-scenario-info")
def get_beam_scenario_info(
    name: str = Query(..., description="Beam scenario name, e.g., 'DEFAULT' or 'SCENARIO_9'")
):
    """
    返回 beam scenario 的基本信息。

    当前只暴露了垂直波束宽度这个规则映射结果。
    """

    return {"name": name, "vertical_beamwidth_deg": beam_scenario_to_v_bw(name)}


@app.get("/tools")
def get_available_tools():
    """
    返回给 tool-using agent 的工具描述列表。

    这是整个 server.py 的另一个关键入口。

    它不是执行工具，而是在告诉模型：
    - 有哪些工具可以用
    - 每个工具叫什么
    - 每个工具的用途是什么
    - 参数 schema 是什么

    关键点：
    - 这里返回的是 OpenAI-compatible tool schema
    - 真正的 scenario 选择不在参数里，而在 `X-Scenario-Id` 请求头里
    - 工具名和 endpoint path 不一定完全同名，`main.py` 里会做 endpoint 映射
    """

    return [
        # 1. 先看吞吐日志：定位问题时刻
        {
            "type": "function",
            "function": {
                "name": "get_throughput_logs",
                "description": "Get the timestamped user throughput logs during the drive-test. With the throughtput logs, we can identify the poor-quality point, and use the corresponding timestamp to gather more information for analysis by using tools.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        # 2. 基于 PCI 取完整小区配置
        {
            "type": "function",
            "function": {
                "name": "get_cell_info",
                "description": "Get full cell configuration given its PCI. The configuration contains cell information including the location, Azimuth, Mech Tilt, Elec Tilt, Ant Height, Duplex Mode, PCI, Band, DL ARFCN, Bandwidth, TX/RX Mode, Transmission Power, InterFreqHoEventType, CovInterFreqA2RsrpThld(dBm), InterFreqA2Hyst(0.5dB), CovInterFreqA5RsrpThld1(dBm), CovInterFreqA5RsrpThld2(dBm), IntraFreqHoA3Offset(0.5dB), IntraFreqHoA3Hyst(0.5dB), IntraFreqHoA3TimeToTrig, Neighbor(gNodeB_ID_ARFCN_PCI), PdcchOccupiedSymbolNum",
                "parameters": {
                    "type": "object",
                    "properties": {"pci": {"type": "integer"}},
                    "required": ["pci"],
                },
            },
        },
        # 3. 基础位置类工具
        {
            "type": "function",
            "function": {
                "name": "get_gnodeb_location",
                "description": "Get gNodeB (cell) location (longitude, latitude) by PCI.",
                "parameters": {
                    "type": "object",
                    "properties": {"pci": {"type": "integer"}},
                    "required": ["pci"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_location",
                "description": "Get UE location (longitude, latitude) at a given timestamp, e.g. '2024-08-25 19:30:57.500'.",
                "parameters": {
                    "type": "object",
                    "properties": {"time": {"type": "string"}},
                    "required": ["time"],
                },
            },
        },
        # 4. Serving 小区观测量
        {
            "type": "function",
            "function": {
                "name": "get_serving_cell_pci",
                "description": "Get serving cell PCI at a given timestamp, e.g. '2024-08-25 19:30:57.500'.",
                "parameters": {
                    "type": "object",
                    "properties": {"time": {"type": "string"}},
                    "required": ["time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_serving_cell_rsrp",
                "description": "Get serving cell SS-RSRP at a given timestamp, e.g. '2024-08-25 19:30:57.500'.",
                "parameters": {
                    "type": "object",
                    "properties": {"time": {"type": "string"}},
                    "required": ["time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_serving_cell_sinr",
                "description": "Get serving cell SS-SINR at a given timestamp, e.g. '2024-08-25 19:30:57.500'.",
                "parameters": {
                    "type": "object",
                    "properties": {"time": {"type": "string"}},
                    "required": ["time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_rbs_allocated_to_user",
                "description": "Get the number of RBs allocated to user by the gNobeB at a given timestamp.",
                "parameters": {
                    "type": "object",
                    "properties": {"time": {"type": "string"}},
                    "required": ["time"],
                },
            },
        },
        # 5. 事件与邻区观测
        {
            "type": "function",
            "function": {
                "name": "get_signaling_plane_event_log",
                "description": "Get the signaling plane event log until a given timestamp, e.g. '2024-08-25 19:30:57.500'.",
                "parameters": {
                    "type": "object",
                    "properties": {"time": {"type": "string"}},
                    "required": ["time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_neighboring_cells_pci",
                "description": "Get the neighboring cells PCI of the serving cell at a given timestamp, e.g. '2024-08-25 19:30:57.500'.",
                "parameters": {
                    "type": "object",
                    "properties": {"time": {"type": "string"}},
                    "required": ["time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_neighboring_cell_rsrp",
                "description": "Get the beam-level RSRP of a neighboring cell at a given timestamp.",
                "parameters": {
                    "type": "object",
                    "properties": {"time": {"type": "string"}, "pci": {"type": "integer"}},
                    "required": ["time", "pci"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_all_cells_pci",
                "description": "List the PCIs of all cells in the network.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        # 6. 更原始的 KPI / MR 原表导出
        {
            "type": "function",
            "function": {
                "name": "get_kpi_data",
                "description": "Get the  data with cell-level KPI, including the information of PRB utilization, PRB Interference, Throughput, CCE utilization and CCE Allocation Success Rate",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_mr_data",
                "description": "Get the normal Measurement Report (MR) data surrounding poor-quality users",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        # 7. 几何 / 覆盖 / 仿真工具
        {
            "type": "function",
            "function": {
                "name": "judge_mainlobe_or_not",
                "description": "Judge the user is in the cell's mainlobe or not. True means outside the mainlobe, False means in the mainlobe.",
                "parameters": {
                    "type": "object",
                    "properties": {"time": {"type": "string"}, "pci": {"type": "integer"}},
                    "required": ["time", "pci"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_horizontal_angle",
                "description": "Calculate the horizontal angle between the user and the cell.",
                "parameters": {
                    "type": "object",
                    "properties": {"time": {"type": "string"}, "pci": {"type": "integer"}},
                    "required": ["time", "pci"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_tilt_angle",
                "description": "Calculate the tilt angle between the user and the cell.",
                "parameters": {
                    "type": "object",
                    "properties": {"time": {"type": "string"}, "pci": {"type": "integer"}},
                    "required": ["time", "pci"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_pathloss",
                "description": "Calculate the pathloss between the user and the cell.",
                "parameters": {
                    "type": "object",
                    "properties": {"time": {"type": "string"}, "pci": {"type": "integer"}},
                    "required": ["time", "pci"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_overlap_ratio",
                "description": "Calculate the overlap coverage ratio of the serving cell and the neighbor cell.",
                "parameters": {
                    "type": "object",
                    "properties": {"pci_serving": {"type": "integer"}, "pci_neighbor": {"type": "integer"}},
                    "required": ["pci_serving", "pci_neighbor"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "optimize_antenna_gain",
                "description": "Simulate the RSRP gain if adjust the azimuth angle and tilt angel of the antenna.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time": {"type": "string"},
                        "pci": {"type": "integer"},
                        "adjust_horizontal_angle": {"type": "number"},
                        "adjust_tilt_angle": {"type": "number"},
                    },
                    "required": ["time", "pci", "adjust_horizontal_angle", "adjust_tilt_angle"],
                },
            },
        },
    ]


# ------------------------------------------------------------------------------
# 直接运行方式
# ------------------------------------------------------------------------------
#
# 官方建议命令：
#     uvicorn server:app --reload
#
# 这里保留 `if __name__ == "__main__"` 只是为了方便单文件启动。

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_md:app", host=FASTAPI_HOST, port=FASTAPI_PORT, reload=True)
