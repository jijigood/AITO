#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Track A 学习版 _types.py

这个文件不是官方执行入口，只是把官方 `_types.py` 用更容易理解的中文注释重写了一遍。

建议把它当成：
1. 赛题数据结构说明书
2. Tool Calling 协议速查表
3. 理解 `server.py` / `main.py` 的配套参考

注意：
- 这里尽量保持类名、字段名与官方一致，方便对照阅读。
- 这个文件主要用于学习，不建议直接替换官方源码中的 `_types.py`。
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# Tool Calling 相关类型
# ------------------------------------------------------------------------------


class ToolFunction(BaseModel):
    """
    表示一次函数调用里的 `function` 部分。

    这个结构对齐 OpenAI Chat Completions 的 tool call 协议，例如：

    {
      "name": "get_cell_info",
      "arguments": "{\"pci\": 328}"
    }

    关键点：
    - `name` 是工具名
    - `arguments` 不是 dict，而是 JSON 字符串
      这是为了与 OpenAI 的原始返回格式保持一致
    """

    name: str
    arguments: str = Field(
        default="{}",
        description="JSON 编码后的参数字符串，例如 '{\"pci\": 328}'",
    )


class ToolCall(BaseModel):
    """
    表示模型发起的一次完整工具调用请求。

    典型格式：

    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "...",
        "arguments": "{...}"
      }
    }

    字段说明：
    - `id`：这次工具调用的唯一标识
    - `type`：当前协议中固定为 "function"
    - `function`：真正的函数名和参数
    """

    id: str
    type: Literal["function"] = "function"
    function: ToolFunction


class AssistantMessage(BaseModel):
    """
    表示 assistant 的一条消息。

    它兼容几种情况：
    - 只有自然语言内容
    - 只有工具调用
    - 内容 + 工具调用同时存在
    - 某些模型额外返回 reasoning_content

    这在 `main.py` 的工具循环里很常见。
    """

    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    reasoning_content: Optional[str] = None


class ToolCallTrace(BaseModel):
    """
    表示一次工具调用在运行轨迹中的记录。

    这不是 OpenAI 原生协议对象，而是项目为了调试和复盘额外定义的。
    用途：
    - 看第几轮调用了什么工具
    - 看参数是什么
    - 看是否失败
    - 看同一轮里的调用顺序
    """

    function_name: str
    arguments: str
    turn: int
    has_failed: bool
    order: int


class AgentResult(BaseModel):
    """
    表示 agent 解完一个 scenario 后的结构化结果。

    你可以把它看成 “一次题目求解的总记录”。

    包含：
    - 基本状态：是否 solved
    - 工具轨迹：调用了哪些工具
    - 最终答案：answer
    - 原始消息：messages
    - 失败原因：reason
    """

    scenario_id: str
    num_iterations: int = Field(..., ge=1, description="对话轮数，至少为 1")
    tool_calls: List[ToolCallTrace]
    num_tool_calls: int = Field(..., ge=0, description="总工具调用次数，不能为负")
    status: Literal["solved", "unresolved"]
    traces: str
    answer: str
    messages: List[Dict[str, Any]]
    reason: Optional[str] = None


# ------------------------------------------------------------------------------
# 题目本体：Task / 选项
# ------------------------------------------------------------------------------


class RootCause(BaseModel):
    """
    表示一个候选选项。

    虽然类名叫 RootCause，但在 Track A 里它更接近：
    - 一个候选优化动作
    - 或者一个候选结论项

    例如：
    - id = "C1"
    - label = "Decrease transmission power for 3255083_3"
    """

    id: str = Field(..., description="选项编号，例如 C1")
    label: str = Field(..., description="选项的人类可读文本说明")


class Task(BaseModel):
    """
    表示题目层本身。

    主要由三部分组成：
    - `allowed_tools`：允许使用哪些工具
    - `description`：题干
    - `options`：候选答案列表
    """

    allowed_tools: Literal["all"] | str
    description: str
    options: List[RootCause]


# ------------------------------------------------------------------------------
# 上下文信息：不是原始表，而是场景背景
# ------------------------------------------------------------------------------


class Environment(BaseModel):
    """
    表示无线网络场景的背景信息。

    注意：
    这个类名叫 `Environment`，但它不是 `main.py` 里那个负责请求工具 API 的
    Environment 类。这里只是“数据结构中的环境描述”。
    """

    mobility_scenario: str
    network_type: str
    num_base_stations: str  # 官方原始数据里这个字段就是字符串，所以这里保持一致


class Context(BaseModel):
    """
    表示题目的背景上下文。

    它通常包含：
    - 一段对场景的自然语言描述
    - 一组结构化的无线网络背景字段
    """

    description: str
    wireless_network_information: Environment


# ------------------------------------------------------------------------------
# 原始数据部分：真正的诊断素材
# ------------------------------------------------------------------------------


class ScenarioData(BaseModel):
    """
    表示一个 scenario 附带的原始观测数据。

    这是 Track A 最核心的部分之一。

    关键理解：
    - 这里的多个字段不是文件路径
    - 也不是 DataFrame
    - 而是“直接嵌在 JSON 里的 pipe-delimited CSV 字符串”

    例如：
    Timestamp|PCI|RSRP|SINR
    2024-...|328|-83.1|5.2

    后续 `server.py` 会在需要时把这些字符串解析成 DataFrame。
    你自己的本地 parser 也可以直接这么做。
    """

    collection_method: str

    network_configuration_data: str = Field(
        ...,
        description="网络配置表，使用 | 分隔的 CSV 原始字符串",
    )

    user_plane_data: str = Field(
        ...,
        description="用户面数据表，使用 | 分隔的 CSV 原始字符串",
    )

    signaling_plane_data: Optional[str] = Field(
        default=None,
        description="信令面数据表，使用 | 分隔的 CSV 原始字符串，可为空",
    )

    traffic_data: Optional[str] = Field(
        default=None,
        description="话务/KPI 数据表，使用 | 分隔的 CSV 原始字符串，可为空",
    )

    mr_data: Optional[str] = Field(
        default=None,
        description="MR 测量报告数据表，使用 | 分隔的 CSV 原始字符串，可为空",
    )

    notes: Optional[str] = Field(
        default=None,
        description="补充说明，可为空",
    )


# ------------------------------------------------------------------------------
# 工具元信息
# ------------------------------------------------------------------------------


class ToolsMetadata(BaseModel):
    """
    表示题目元数据里附带的“工具能力说明”。

    注意：
    这里不是 `/tools` endpoint 真正返回的函数 schema。
    这里只是 scenario 自带的一段摘要说明。
    """

    capabilities: List[str]
    description: str


# ------------------------------------------------------------------------------
# 期望输出格式（更多是说明性字段）
# ------------------------------------------------------------------------------


class ExpectedOutputContent(BaseModel):
    """
    表示“理想答案报告”里的三个组成部分。

    虽然比赛最终提交通常只需要 `C1~C22`，
    但从数据设计上，官方仍然保留了：
    - evidence
    - findings
    - recommendations

    这说明题目最初是按“诊断报告式输出”来设计过的。
    """

    evidence: str
    findings: str
    recommendations: str


class ExpectedOutput(BaseModel):
    """
    对期望输出整体再包一层。

    - `content`：具体内容
    - `type`：输出类型
    """

    content: ExpectedOutputContent
    type: str


# ------------------------------------------------------------------------------
# 顶层 Scenario：一整道题
# ------------------------------------------------------------------------------


class Scenario(BaseModel):
    """
    表示 Track A 中的一整道题。

    这是最重要的顶层模型。

    一个 scenario 通常包含：
    - `scenario_id`：题目唯一标识
    - `answer`：标准答案
    - `task`：题干与选项
    - `context`：背景说明
    - `data`：原始表格字符串
    - `tools`：工具能力说明

    可选地还可能带：
    - `expected_output`
    - `steps`

    5.4 里还有一个很容易漏掉的事实：
    原始 JSON 顶层通常带有 `tag` 字段，例如：
        "single-answer" / "multiple-answer"

    但官方 `Scenario` 模型并没有声明 `tag`。
    在 Pydantic 默认配置下，`Scenario.model_validate(raw_json)` 会接受这条样本，
    但后续 `scenario.model_dump()` 会把这个额外字段丢掉。

    这就是为什么 5.4 新增的 `phase2_run.py` 没有复用 server 返回的
    `Scenario.model_dump()` 来做题型路由，而是直接读取原始 `test.json`：
    它需要保留 `tag` 来区分 single-answer / multiple-answer。

    实际使用中，`utils.py` 会对原始 JSON 做：
        Scenario.model_validate(raw_json)
    这样就能在加载时完成结构校验。
    """

    scenario_id: str
    answer: str = Field(..., description="标准答案，例如 C6 或 C3|C7")
    task: Task
    context: Context
    data: ScenarioData
    tools: ToolsMetadata
    expected_output: Optional[ExpectedOutput] = None
    steps: Optional[List[str]] = None
