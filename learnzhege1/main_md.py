#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Track A 学习版 main.py

这个文件是官方 `main.py` 的中文注释学习版。

它的核心作用不是定义赛题数据，也不是实现工具 server，而是：
1. 连接大模型接口
2. 从 `server.py` 的 `/tools` 获取工具描述
3. 在多轮对话中让模型自主决定是否调用工具
4. 收集工具结果并继续喂给模型
5. 最后把答案写成 `result.csv`

把它放在整个官方 starter code 里，你可以这样理解：
- `_types.py`：定义数据长什么样
- `server.py`：把数据包装成可调用工具
- `main.py`：驱动模型 + 工具，一题题求解

建议阅读顺序：
1. 先看 `Environment`
2. 再看 `AgentsRunner._call_model()`
3. 再看 `AgentsRunner.run()`
4. 最后看 `benchmark()` 和 CLI

说明：
- 这里尽量保留官方结构和主要变量名，方便对照原版
- 我加入了中文注释和少量导入兜底，方便在 `learnzhege1` 中独立阅读
- 对官方原文件里几个容易踩坑的地方，我会直接在注释中点出来
- 这个文件主要用于学习，不建议直接替换比赛提交脚本
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import requests
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

# ------------------------------------------------------------------------------
# 导入官方项目中的类型、logger 和工具函数
# ------------------------------------------------------------------------------
#
# 学习版文件位于 `learnzhege1` 目录，所以这里加一个兜底导入逻辑：
# - 如果当前路径下就能导入官方模块，直接使用
# - 否则把 Track A 目录加入 `sys.path`
#
# 这样做是为了让你单独打开这个文件也更容易理解，不是官方原逻辑的一部分。

try:
    from _types import ToolCall
    from logger import init_logger
    from utils import (
        compute_score,
        extract_answer,
        extract_answer_all,
        print_model_response,
        print_tool_call,
        print_tool_result,
    )
except ModuleNotFoundError:
    candidate_dirs = [
        Path(__file__).resolve().parent.parent / "5.4" / "Track A",
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

    from _types import ToolCall
    from logger import init_logger
    from utils import (
        compute_score,
        extract_answer,
        extract_answer_all,
        print_model_response,
        print_tool_call,
        print_tool_result,
    )


# ------------------------------------------------------------------------------
# 环境变量设置
# ------------------------------------------------------------------------------
#
# 官方 main.py 里直接写了：
#   os.environ['AGENT_API_KEY'] = 'sk-XXXXXXXXXXXXX'
#
# 这更像是一个占位示例，而不是正式密钥管理方式。
# 这里保留同样思路，方便你对照理解：
# - `AGENT_API_KEY` 用来给 OpenAI-compatible client 认证
# - 在 5.4 的 Phase 2 云端 server 中，它也对应工具 API 的 Bearer token
# - `NO_PROXY` 让本地访问 localhost 时不要走代理；云端地址不依赖这个设置

os.environ["AGENT_API_KEY"] = "sk-XXXXXXXXXXXXX"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

API_KEY = os.environ.get("AGENT_API_KEY", "dummy")


# ------------------------------------------------------------------------------
# Environment：工具环境层
# ------------------------------------------------------------------------------
#
# 这个类不是 `_types.py` 里的那个数据结构 `Environment`。
# 这里的 `Environment` 是“agent 的工具执行环境”。
#
# 它负责三件事：
# 1. 维护“工具名 -> server endpoint”的映射
# 2. 帮模型执行工具调用
# 3. 通过 `X-Scenario-Id` 把每次调用绑定到当前题目


class Environment:
    """
    管理工具 server 的访问。

    你可以把它理解成：
    - LLM 不直接知道 `/cell-info` 这种 URL
    - LLM 只知道 `get_cell_info` 这样的函数名
    - 这里负责把函数名转成真正的 HTTP 请求
    """

    # 注意：agent 看到的是 tool name，不是 endpoint path。
    # 所以这里必须手动做映射。
    endpoint_mapper = {
        "get_all_scenario": "/scenario/all",
        "get_config_data": "/config-data",
        "get_user_plane_data": "/user-plane-data",
        "get_throughput_logs": "/throughput-logs",
        "get_cell_info": "/cell-info",
        "get_gnodeb_location": "/gnodeb-location",
        "get_user_location": "/user-location",
        "get_serving_cell_pci": "/serving-cell-pci",
        "get_serving_cell_rsrp": "/serving-cell-rsrp",
        "get_serving_cell_sinr": "/serving-cell-sinr",
        "get_rbs_allocated_to_user": "/rbs-allocated-to-user",
        "get_neighboring_cells_pci": "/neighboring-cells-pci",
        "get_neighboring_cell_rsrp": "/neighboring-cell-rsrp",
        "get_signaling_plane_event_log": "/signaling-plane-event-log",
        "get_all_cells_pci": "/all-cells-pci",
        "get_available_tools": "/tools",
        "health": "/health",
        "judge_mainlobe_or_not": "/judge_mainlobe",
        "calculate_horizontal_angle": "/calculate_horizontal_angle",
        "calculate_tilt_angle": "/calculate_tilt_angle",
        "calculate_pathloss": "/calculate_pathloss",
        "calculate_overlap_ratio": "/calculate_overlap_ratio",
        "get_kpi_data": "/get_kpi_data",
        "get_mr_data": "/get_mr_data",
        "optimize_antenna_gain": "/optimize_antenna_gain",
    }

    def __init__(
        self,
        server_url: str,
        verbose: bool = False,
        log_file: Optional[str] = None,
        timeout: float = 15.0,
        logger: logging.Logger = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.verbose = verbose
        self.timeout = timeout
        self.logger = logger if logger is not None else init_logger()

    def _headers(self, scenario_id: Optional[str] = None) -> Dict[str, str]:
        """
        构造请求头。

        最关键的是：
        - 如果有 `scenario_id`，就写入 `X-Scenario-Id`
        - 这样 server 才知道“这次工具调用对应哪一道题”
        """

        headers = {"Content-Type": "application/json"}

        # 5.4 Phase 2 不再只是假设本地 `server.py`。
        # 官方 README 要求访问云端工具服务时带：
        #   Authorization: Bearer <Your-Token>
        #
        # 官方示例代码里这里写的是占位 token。
        # 学习版用 `AGENT_API_KEY` 表达同一件事，避免把 token 分散硬编码。
        headers["Authorization"] = f"Bearer {API_KEY}"
        if scenario_id:
            headers["X-Scenario-Id"] = scenario_id
            # 5.4 官方模板还额外放了一个 `X-API-Token` 占位头。
            # 真实使用时核心鉴权仍应以官方 Phase 2 README 的 Bearer token 为准。
            headers["X-API-Token"] = API_KEY
        return headers

    def _call_api(
        self,
        function_name: str,
        scenario_id: Optional[str] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        """
        真正发出 HTTP GET 请求，调用工具 server。

        流程：
        1. 用 `endpoint_mapper` 找到 endpoint path
        2. 拼成完整 URL
        3. 带上 query params 和 `X-Scenario-Id`
        4. 解析返回 JSON

        错误处理策略：
        - 工具不存在：直接返回 `{"error": ...}`
        - HTTP 失败：尽量提取 FastAPI 返回的 `detail`
        - 其他异常：也统一包装成 `{"error": ...}`

        这样做的好处是：
        - 上层 agent 不必捕获各种 requests 异常
        - 只要把返回文本继续塞回对话即可
        """

        endpoint = self.endpoint_mapper.get(function_name)
        if endpoint is None:
            return {"error": f"Unknown tool '{function_name}'"}

        url = f"{self.server_url}{endpoint}"
        headers = self._headers(scenario_id=scenario_id)

        try:
            # 5.4 官方模板对云端 HTTPS 调用设置了 `verify=False`。
            # 这能绕过某些证书链问题；正式生产代码里则应谨慎关闭 TLS 校验。
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout, verify=False)
            resp.raise_for_status()
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} params={params}")
            data = resp.json()
            return data
        except requests.exceptions.HTTPError:
            try:
                detail = resp.json().get("detail", str(resp.text))
            except Exception:
                detail = str(resp.text)
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} params={params} -> HTTPError: {detail}")
            return {"error": detail}
        except Exception as exc:
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} params={params} -> ERROR: {exc}")
            return {"error": str(exc)}

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        从 `/tools` 获取 OpenAI-compatible tool descriptors。

        这一步很关键：
        - agent 并不是写死每个工具 schema
        - 而是运行时从 server 拉工具描述
        """

        tools = self._call_api("get_available_tools")
        if isinstance(tools, dict) and "error" in tools:
            return []
        if not isinstance(tools, list):
            return []
        return tools

    def get_scenarios(self) -> List[Dict[str, Any]]:
        """
        从 server 拉取全部 scenario。

        官方 baseline 不是逐题向 server 请求“下一道题”，
        而是直接一次性拿 `/scenario/all`。
        """

        scenarios = self._call_api("get_all_scenario")
        if isinstance(scenarios, dict) and "error" in scenarios:
            return []
        if not isinstance(scenarios, list):
            return []
        return scenarios

    def execute(self, tool_call: ToolCall, scenario_id: Optional[str] = None) -> str:
        """
        执行一次模型发出的 tool call，并把结果转回 JSON 字符串。

        为什么返回字符串而不是 dict？
        - 因为 tool message 最终要作为消息内容塞回 `messages`
        - 对模型来说，它看到的是一段文本化的 tool result
        """

        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            result = self._call_api(function_name=function_name, scenario_id=scenario_id, **arguments)
            return json.dumps(result, ensure_ascii=False)

        except json.JSONDecodeError:
            error_msg = f"Tool parameter parsing failed: {tool_call.function.arguments}"
            if self.verbose:
                self.logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

        except Exception as exc:
            error_msg = f"Tool invocation execution failed: {str(exc)}"
            if self.verbose:
                self.logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg}, ensure_ascii=False)


# ------------------------------------------------------------------------------
# AgentsRunner：模型驱动层
# ------------------------------------------------------------------------------
#
# 这个类负责：
# 1. 管理 OpenAI-compatible client
# 2. 管理 solve / run 循环
# 3. 对多个 scenario 做 benchmark


class AgentsRunner:
    """
    封装整个“模型调用 + 工具调用 + 结果记录”的主流程。
    """

    def __init__(
        self,
        environment: Environment,
        model_url: str,
        model_name: str,
        model_provider: Optional[str] = None,
        max_tokens: int = 16000,
        max_retries: int = 3,
        max_iterations: int = 20,
        verbose: bool = False,
        logger: logging.Logger = None,
    ):
        self.environment = environment
        self.model_url = model_url
        self.model_name = model_name
        self.model_provider = model_provider
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.logger = logger if logger is not None else init_logger()
        self.running_metrics = {}

        # 这里用的是 OpenAI-compatible client。
        # 所以只要后端接口兼容 Chat Completions，理论上就能替换 provider。
        self.client = OpenAI(
            base_url=model_url,
            api_key=API_KEY,
            http_client=httpx.Client(verify=False),
        )

    def _call_model(self, messages: List[Dict[str, Any]], functions: List[Dict[str, Any]], **kwargs):
        """
        调用大模型，并带一个简单的重试机制。

        关键点：
        - `messages`：当前对话历史
        - `functions`：如果非空，就作为 tools 传给模型
        - `tool_choice`：固定为 `"auto"`，让模型自行决定是否调用工具

        重试策略：
        - 对限流、网络、超时、服务端错误做指数退避重试
        - 对大部分 4xx 非 429 错误直接不重试
        """

        base_wait_time = 1.0

        call_kwargs = {
            "model": f"{self.model_provider}/{self.model_name}" if self.model_provider else self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

        if functions:
            call_kwargs["tools"] = functions
            call_kwargs["tool_choice"] = "auto"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**call_kwargs)
                return response.choices[0].message

            except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as exc:
                if self.verbose:
                    self.logger.error(traceback.format_exc())

                if hasattr(exc, "status_code") and 400 <= exc.status_code < 500 and exc.status_code != 429:
                    if self.verbose:
                        self.logger.info("Non-retriable exception: %s", exc)
                    return None

                if attempt == self.max_retries:
                    if self.verbose:
                        self.logger.info("Final failure after %s attempts: %s", self.max_retries, exc)
                    return None

                wait = base_wait_time * (2 ** (attempt - 1))
                if self.verbose:
                    self.logger.info("Retry %s/%s after %.1fs due to: %s", attempt, self.max_retries, wait, exc)
                time.sleep(wait)

            except Exception as exc:
                if self.verbose:
                    self.logger.info("Unhandled exception: %s", exc)
                return None

        return None

    def run(self, scenario: Dict[str, Any], free_mode: bool = False) -> Dict[str, Any]:
        """
        对单个 scenario 执行完整求解。

        这是整个官方 baseline 的核心函数。

        它的真实流程是：
        1. 从 scenario 中取题干和候选选项
        2. 从 server 拉工具定义
        3. 构造初始 user prompt
        4. 多轮循环：
           - 调模型
           - 如果模型发工具调用，就执行工具并把结果塞回消息
           - 如果模型直接输出文本，就结束
        5. 如果开启 `free_mode`，且最后答案没有 boxed 格式，再补一次“约束性追问”

        注意一个很重要的设计事实：
        - 官方 baseline 没有 system prompt
        - 只有一个 user prompt
        - 所有推理几乎完全交给模型自己决定
        """

        scenario_id = scenario.get("scenario_id")
        task = scenario.get("task", {})

        root_causes = "".join([f"{item['id']}:{item['label']}\n" for item in task.get("options", [])])

        tool_defs = self.environment.get_tools()
        if not tool_defs:
            return {"scenario_id": scenario_id, "status": "unresolved", "reason": "No tools available"}

        question = task.get("description", "") + f"\nOptions:\n{root_causes}"

        # 官方 baseline 从这里开始对话：
        # 没有 system prompt，只有 user message。
        messages: List[Dict[str, Any]] = [{"role": "user", "content": question}]

        num_tool_calls = 0
        list_tool_calls = []
        status = None
        reason = None
        last_msg = None

        for i in range(self.max_iterations):
            self.logger.info(f"\n[Scenario: {scenario_id}] Round {i + 1} conversation, calling tools:")

            msg = self._call_model(messages, functions=tool_defs)
            if msg is None:
                continue

            last_msg = msg
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

            if self.verbose:
                print_model_response(msg, logger=self.logger, minimize=False)

            # 如果模型发起了工具调用，就逐个执行。
            if msg.tool_calls:
                num_tool_calls += len(msg.tool_calls)

                for j, tool_call in enumerate(msg.tool_calls):
                    if self.verbose:
                        print_tool_call(tool_call, logger=self.logger)

                    tool_result = self.environment.execute(tool_call, scenario_id=scenario_id)

                    messages.append({"role": "tool", "content": tool_result, "tool_call_id": tool_call.id})

                    if self.verbose:
                        print_tool_result(tool_result, logger=self.logger)

                    # 这里的“失败”判定非常朴素：只要返回字符串里含 `"error"` 就算失败。
                    # 这不是严格 JSON 检查，只是一个简单标记。
                    has_failed = "error" in tool_result
                    list_tool_calls.append(
                        {
                            "function_name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                            "turn": i + 1,
                            "has_failed": has_failed,
                            "order": j + 1,
                            "results": tool_result,
                        }
                    )

            # 如果没有 tool_calls，但有文本输出，就认为模型已经给了最终答案。
            elif msg.content:
                status = "solved"
                break

            else:
                status = "unresolved"
                reason = "Unable to answer this question."
                break

        if status is None:
            status = "unresolved"
            reason = "The maximum number of iterations has been reached."

        # ------------------------------------------------------------------
        # Optional final constraint prompt
        # ------------------------------------------------------------------
        #
        # 这段逻辑是官方 baseline 里一个容易忽略的小分支。
        # 作用是：
        # - 如果已经走完主要对话
        # - 但模型没按要求把答案包进 \boxed{}
        # - 那就再追问一次，并明确告诉它这是单选还是多选
        #
        # 这里要注意一个官方原代码中的细节：
        # - `current_answer = ...,` 末尾多了一个逗号
        # - 会把 `current_answer` 变成 tuple
        # - 这会影响 `extract_answer()` 的稳定性
        #
        # 这个学习版为了忠实展示官方行为，保留这个写法，并把问题写在注释里。
        if free_mode:
            current_answer = (
                getattr(last_msg, "content", "") or getattr(last_msg, "reasoning_content", "") if last_msg else "",
            )
            current_traces = getattr(last_msg, "reasoning_content", "") if last_msg else ""
            agent_answer = extract_answer(current_answer) or extract_answer(current_traces)
            if agent_answer == "":
                self.logger.info(f"\n[Scenario: {scenario_id}] Round {i + 2} conversation, answer question:")
                status = "solved"

                if "Select the most appropriate optimization solution" in question:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "This is a single-answer question. Select the most appropriate optimization solution and enclose its number in \\boxed{{}} "
                                f"in the final answer. For example, \\boxed{{C3}} \nPotential root causes:\n{root_causes}\n"
                            ),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "This is a multiple-answer question. Select two to four possible optimization solutions and enclose their numbers in \\boxed{{}} "
                                f"in the final answer. For example,  \\boxed{{C3|C5}} or \\boxed{{C7|C11}}. \nPotential root causes:\n{root_causes}\n"
                            ),
                        }
                    )

                msg2 = self._call_model(messages, functions=[])
                if msg2 is not None:
                    last_msg = msg2

        return {
            "scenario_id": scenario_id,
            "num_iterations": (i + 1),
            "tool_calls": list_tool_calls,
            "num_tool_calls": num_tool_calls,
            "status": status,
            "traces": getattr(last_msg, "reasoning_content", "") if last_msg else "",
            "answer": getattr(last_msg, "content", "") or getattr(last_msg, "reasoning_content", "") if last_msg else "",
            "messages": messages,
            "reason": reason,
        }

    def benchmark(
        self,
        num_attempts: int,
        save_dir: str,
        save_freq: int = 10,
        max_samples: int = None,
        free_mode: bool = False,
    ) -> None:
        """
        对多个 scenario 做批量测试，并周期性保存 `result.csv`。

        流程：
        1. 从 server 拉全部 scenario
        2. 如果设置了 `max_samples`，只跑前 N 道
        3. 对每道题尝试 `num_attempts` 次
        4. 记录平均得分和平均延迟
        5. 每隔 `save_freq` 题写一次结果文件

        需要注意的几个现实问题：
        - 这里的 `compute_score()` 只是本地辅助评分，不等于比赛最终评分逻辑
        - 5.4 官方默认 `max_samples=500`，对应 Phase 1/Phase 2 Track A 测试集规模
        - 5.4 官方除了周期性写 `result.csv`，也会写一份更完整的 `results.json`
        """

        os.makedirs(save_dir, exist_ok=True)

        completions: List[Dict[str, Any]] = []
        save_result: List[Dict[str, Any]] = []

        scenarios = self.environment.get_scenarios()

        if max_samples is not None:
            scenarios = scenarios[: min(max_samples, len(scenarios))]

        for idx, scenario in enumerate(scenarios):
            scenario_id = scenario.get("scenario_id")
            start_time = time.time()

            n_success = 0.0
            agent_answers: List[str] = []
            sample_response: Optional[Dict[str, Any]] = {}

            for attempt in range(num_attempts):
                self.logger.info(f"[Scenario {scenario_id}] attempt {attempt + 1}/{num_attempts}")

                response = self.run(scenario=scenario, free_mode=free_mode)
                sample_response = response

                if response.get("status") == "solved":
                    agent_answer = extract_answer_all(response.get("answer", "")) or extract_answer_all(
                        response.get("traces", "")
                    )

                    # 官方这里直接把本地解析答案和 ground truth 送进 `compute_score()`。
                    # 这只是 starter code 的本地评估逻辑，不是比赛最终 leaderboard 的严格复现。
                    ground_truth = scenario.get("answer")
                    n_success += compute_score(agent_answer, ground_truth)
                    agent_answers.append(agent_answer)
                    pink = "\033[95m"
                    reset = "\033[0m"
                    self.logger.info(
                        f"{pink}\n[Scenario: {scenario_id}] Agent's answer is {agent_answer}, ground truth is {ground_truth}{reset}."
                    )

            acc = n_success / float(num_attempts)
            latency = round((time.time() - start_time) / float(num_attempts), 2)

            completions.append(
                {
                    "scenario_id": scenario_id,
                    "free_mode": free_mode,
                    "response": (sample_response).get("answer", ""),
                    "traces": (sample_response).get("traces", ""),
                    # "messages": (sample_response).get("messages", ""),  # 如需完整消息可取消注释
                    "num_iterations": (sample_response).get("num_iterations", 0),
                    "num_tool_calls": (sample_response or {}).get("num_tool_calls", 0),
                    "tool_calls": (sample_response).get("tool_calls", []),
                    "answers": agent_answers,
                    "ground_truth": scenario.get("answer"),
                    "accuracy": acc,
                    "latency": latency,
                }
            )

            # 官方原版这里直接写 `agent_answers[0]`，
            # 如果这一题一次都没成功，会有越界风险。
            # 这个学习版稍微做了个保护，避免你拿它做实验时直接崩掉。
            save_result.append(
                {
                    "scenario_id": scenario_id,
                    "answers": agent_answers[0] if agent_answers else "",
                }
            )

            if ((idx + 1) % save_freq == 0) or ((idx + 1) == len(scenarios)):
                df = pd.DataFrame(save_result)
                df.to_csv(os.path.join(save_dir, "result.csv"), index=False)

                # 5.4 新增：保存更完整的运行记录，方便复盘每题的工具调用、
                # traces、耗时、答案和模型配置。
                doc = {
                    "running_metrics": self.running_metrics,
                    "model_name": self.model_name,
                    "model_provider": self.model_provider,
                    "completions": completions,
                    "sample_processed": (idx + 1),
                    "status": "completed" if ((idx + 1) == len(scenarios)) else "running",
                }

                out_path = os.path.join(save_dir, "results.json")
                with open(out_path, "w", encoding="utf-8") as fp:
                    json.dump(doc, fp, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------------------
#
# 这部分就是把前面的 runner 真正跑起来。
# 你可以把它理解成：
# - 解析命令行参数
# - 初始化 logger
# - 创建 environment
# - 创建 runner
# - 跑 benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agents benchmarking")
    # 5.4 默认指向 Phase 2 中国区云端工具服务。
    # 如果你仍然本地跑 Phase 1 server.py，可以手动传：
    #   --server_url http://localhost:7860
    parser.add_argument("--server_url", type=str, default="https://120.46.145.77/no")
    parser.add_argument("--model_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--model_name", type=str, default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--model_provider", type=str, default=None)
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--log_file", type=str, default="./log.log")

    # 这个参数很容易读反：
    # - `action="store_false"` 表示默认值其实是 True
    # - 也就是说不传 `--free_mode` 时，得到的是 True
    # 这和很多人直觉会相反。
    parser.add_argument("--free_mode", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = init_logger(log_file=args.log_file)

    # 官方原代码这里写成：
    #   Environment = Environment(...)
    # 这会把类名和实例名撞在一起，虽然能跑，但阅读上很绕。
    # 学习版改成更清楚的实例名 `environment`。
    environment = Environment(server_url=args.server_url, verbose=args.verbose, logger=logger)

    runner = AgentsRunner(
        environment=environment,
        model_url=args.model_url,
        model_name=args.model_name,
        model_provider=args.model_provider,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        logger=logger,
    )

    runner.benchmark(
        max_samples=args.max_samples,
        num_attempts=args.num_attempts,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        free_mode=args.free_mode,
    )
