#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Track A 学习版 utils.py

这个文件是官方 `utils.py` 的中文注释学习版。

虽然名字叫 utils，但它在官方 starter code 里其实承担了四类很重要的职责：
1. 数据加载：把 JSON 文件读成 `Scenario`
2. DataFrame 小工具：从表里取第一行、取全部行、按时间点切片
3. 日志打印：把模型回复、工具调用、工具结果打印得更清楚
4. 答案后处理：从 `\\boxed{}` 里抽答案，并做一个本地辅助评分

把它放进整体框架里，你可以这样理解：
- `_types.py` 定义数据结构
- `utils.py` 负责把原始 JSON 真正“变成可用对象”
- `server.py` 会继续调用这里的时间切片工具
- `main.py` 会调用这里的答案抽取和本地打分工具

阅读建议：
1. 先看 `load_scenarios()`
2. 再看 `normalize_time()` / `get_fields_at_time()`
3. 再看 `extract_answer()` / `extract_answer_all()`
4. 最后看 `compute_score()`

说明：
- 这里尽量保持官方函数名和主要逻辑，方便对照阅读
- 增加了中文注释和少量导入兜底，方便你在 `learnzhege1` 中单独打开
- 对几个容易误导人的地方，我会直接写在注释里
- 这个文件主要用于学习，不建议直接替换比赛提交脚本
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dateutil import parser

# ------------------------------------------------------------------------------
# 导入官方类型与 logger
# ------------------------------------------------------------------------------
#
# 因为这个学习版文件不在官方 Track A 根目录里，所以这里做一个导入兜底。

try:
    from _types import Scenario
    from logger import logging
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
    from logger import logging


# ------------------------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------------------------


def load_scenarios(config, default_scenario="random", split="test"):
    """
    从 `./{config}/{split}.json` 读取原始样本，并校验成 `Scenario` 对象。

    这是整个 Track A 官方代码真正的数据入口之一。

    流程：
    1. 打开 JSON 文件
    2. 读成原始 dict 列表
    3. 对每条样本做 `Scenario.model_validate()`
    4. 决定默认 scenario id

    为什么这一步重要？
    - 因为 `_types.py` 只有在这里才真正参与“运行时校验”
    - 如果字段缺失、类型不对，这里就会出问题

    额外注意：
    - 这里用的是相对路径 `./{config}/{split}.json`
    - 所以 server / main 往往需要在 Track A 目录下启动，路径才最稳
    """

    with open(f"./{config}/{split}.json", "r", encoding="utf-8") as f:
        raw_scenarios = json.load(f)

    if not raw_scenarios:
        raise RuntimeError(f"No scenarios loaded. Check ./{config}/test.json or DOWNLOAD_FROM_REMOTE.")

    validated_scenarios: List[Scenario] = [Scenario.model_validate(s) for s in raw_scenarios]

    if default_scenario == "random":
        import random

        default_scenario_id = random.choice(validated_scenarios).scenario_id
    else:
        default_scenario_id = validated_scenarios[0].scenario_id

    return validated_scenarios, default_scenario_id


# ------------------------------------------------------------------------------
# DataFrame 转换小工具
# ------------------------------------------------------------------------------


def df_first_or_none(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    如果 DataFrame 非空，就返回第一行 dict；否则返回 None。

    用途非常广：
    - `server.py` 里大量 endpoint 最终都只想返回“一条记录”
    - 例如某个 PCI 的位置、某个时间点的 serving PCI、某个时间点的用户位置
    """

    if df is None or df.empty:
        return None
    return df.iloc[0].to_dict()


def df_all_or_none(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    如果 DataFrame 非空，就把整张表转成 `|` 分隔文本；否则返回 None。

    这个函数名字容易让人误解成“返回全部 dict”；
    但实际上它返回的是 CSV 风格字符串，不是 Python dict/list。

    它主要服务于：
    - `/signaling-plane-event-log`
    - 这类想把多行结果整体返回给模型看的工具
    """

    if df is None or df.empty:
        return None
    return df.to_csv(index=False, sep="|")


# ------------------------------------------------------------------------------
# 时间标准化与按时间切片
# ------------------------------------------------------------------------------


def normalize_time(ts: str) -> datetime:
    """
    把各种时间字符串统一归一化到 UTC 时区的 `datetime`。

    为什么要这么做？
    - 原始表中的时间字段可能有时区，也可能没有
    - 如果不先标准化，字符串直接比较很容易出错

    规则：
    - 若原始时间没有 tzinfo，则默认补成 UTC
    - 若原始时间有时区，则先转换到 UTC
    """

    dt = parser.parse(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def get_fields_at_time(df: pd.DataFrame, user_input_time: str, fields: List[str]) -> pd.DataFrame:
    """
    取“恰好等于某个时间点”的若干字段。

    实现方式：
    1. 先把输入时间和表中的 `Timestamp` 都标准化
    2. 在副本表里新增 `NormalizedTime`
    3. 做等值匹配
    4. 只保留需要的字段列

    这个函数是 `server.py` 中大量按时间点查询工具的基础。
    """

    target_time = normalize_time(user_input_time)
    tmp = df.copy()
    tmp["NormalizedTime"] = tmp["Timestamp"].apply(normalize_time)
    return tmp.loc[tmp["NormalizedTime"] == target_time, fields]


def get_fields_before_time(df: pd.DataFrame, user_input_time: str, fields: List[str]) -> pd.DataFrame:
    """
    取“小于等于某个时间点”的若干字段。

    与 `get_fields_at_time()` 的区别在于：
    - 它更适合事件日志类数据
    - 比如“到某个时刻为止发生过哪些 signaling 事件”
    """

    target_time = normalize_time(user_input_time)
    tmp = df.copy()
    tmp["NormalizedTime"] = tmp["Timestamp"].apply(normalize_time)
    return tmp.loc[tmp["NormalizedTime"] <= target_time, fields]


# ------------------------------------------------------------------------------
# 环境变量布尔解析
# ------------------------------------------------------------------------------


def _env_bool(name: str, default: bool = False) -> bool:
    """
    从环境变量解析布尔值。

    支持的真值字符串包括：
    - `1`
    - `true`
    - `yes`
    - `y`
    - `on`

    `server.py` 会用它来读取 `DOWNLOAD_FROM_REMOTE` 之类的配置。
    """

    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


# ------------------------------------------------------------------------------
# 日志打印辅助函数
# ------------------------------------------------------------------------------
#
# 这些函数不改变逻辑，只是为了让终端日志更容易读。


def print_tool_result(tool_result, logger: logging.Logger):
    """
    用带颜色的形式打印工具返回结果。
    """

    color = "\033[95m"
    reset = "\033[0m"
    logger.info(f"{color}[Tool result] {tool_result} {reset}")


def print_tool_call(tool_call, logger: logging.Logger, step=None):
    """
    打印一次工具调用。

    它会尽量把 `tool_call.function.arguments` 从 JSON 字符串解析成人类可读的
    `k=v` 形式；如果失败，就直接原样打印。
    """

    function_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
        args = ", ".join([f"{k}={v}" for (k, v) in arguments.items()])
    except Exception:
        args = str(tool_call.function.arguments)

    color = "\033[95m"
    reset = "\033[0m"
    step_msg = f"Step {step}: " if step else ""
    logger.info(f"{color}[{step_msg}Executing tool call {tool_call.id}] function {function_name}({args}){reset}")


def print_model_response(response, logger: logging.Logger, minimize=False):
    """
    打印模型回复。

    两种模式：
    - `minimize=True`：只打 content
    - `minimize=False`：打印 content、reasoning_content、tool_calls

    这在调试 tool-using agent 时很有价值，因为你能直接看到：
    - 模型说了什么
    - 模型请求了哪些工具
    - 参数是什么
    """

    color = "\x1b[31;1m"
    reset = "\033[0m"
    if minimize:
        logger.info(f"{color}[Agent response] {getattr(response, 'content', None)}{reset}\n")
    else:
        if getattr(response, "content", None):
            logger.info(f"{color}[Agent response] {response.content}{reset}\n")
        if getattr(response, "reasoning_content", None):
            logger.info(f"[Agent reasoning] {response.reasoning_content}\n")

        if getattr(response, "tool_calls", None):
            for tool_call in response.tool_calls:
                function_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    args = ", ".join([f"{k}={v}" for (k, v) in arguments.items()])
                except Exception:
                    args = str(tool_call.function.arguments)

                logger.info(f"[Request tool call {tool_call.id}] function {function_name}({args})")


# ------------------------------------------------------------------------------
# 答案抽取
# ------------------------------------------------------------------------------
#
# 官方 baseline 约定模型把最终选项写进 `\\boxed{...}`。
# 例如：
# - 单选：`\\boxed{C3}`
# - 多选：`\\boxed{C3|C5}`


def extract_answer(response: str):
    """
    从模型文本中提取最后一个 `\\boxed{...}` 里的内容，并做更激进的清洗。

    清洗包括：
    - 去掉换行
    - 去掉前导冒号
    - 去掉结尾句号或斜杠
    - 去掉多余花括号

    适合在“想尽量拿到一个干净答案字符串”时使用。
    """

    try:
        matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", response)
        if matches:
            pred = matches[-1].strip()
            pred = re.sub(r"\n\s*", "", pred).lstrip(":").rstrip("./")
            pred = re.sub(r"[{}]", "", pred)
            return pred
        return ""
    except Exception:
        return ""


def extract_answer_all(response: str):
    """
    从模型文本中提取最后一个 `\\boxed{...}` 里的内容，但清洗更保守。

    与 `extract_answer()` 相比：
    - 它不去掉换行后的额外格式太多内容
    - 更偏向“原样拿到 boxed 内部文本”

    在 `main.py` 中，本地 benchmark 更常用这个版本。
    """

    try:
        matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", response)
        if matches:
            pred = matches[-1].strip()
            pred = re.sub(r"[{}]", "", pred)
            return pred
        return ""
    except Exception:
        return ""


# ------------------------------------------------------------------------------
# 本地辅助评分
# ------------------------------------------------------------------------------
#
# 这一段特别值得你注意：
# - 它不是 README 中比赛最终 IOU 评分的严格实现
# - 它只是官方 starter code 的一个本地便捷函数
#
# 更关键的是：
# - 函数签名写的是 `compute_score(gt, answer, ...)`
# - 但官方 `main.py` 里调用时传的是 `compute_score(agent_answer, ground_truth)`
# - 也就是参数顺序反了
#
# 所以你看到的本地“accuracy”不要等同于线上 leaderboard 分数。


def compute_score(gt: str, answer: str, decision="hard"):
    """
    计算一个非常简化的本地分数。

    `decision="hard"`：
    - 直接比较字符串是否相等（忽略大小写）

    `decision="soft"`：
    - 只抽其中的数字部分比较
    - 例如 `C3` 和 `3` 会被视为相同

    如果 `gt` 中带 `|`：
    - 它会把 `gt` 视作多个可接受答案之一
    - 只要有一个子项匹配就返回 True

    注意：
    这和比赛 README 中“多答案 IOU + 时间折扣”的真实评分逻辑完全不同。
    """

    if "|" in gt:
        gts = gt.split("|")
        return any([compute_score(g, answer, decision) for g in gts])
    try:
        if decision == "soft":
            match = re.search(r"\d+", gt)
            gt_int = int(match.group()) if match else None
            match = re.search(r"\d+", answer)
            answer_int = int(match.group()) if match else None
            return gt_int == answer_int
        else:
            return gt.lower() == answer.lower()
    except Exception:
        return False
