#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Track A 学习版 logger.py

这个文件是官方 `logger.py` 的中文注释学习版。

它的职责很单一，但在整个 starter code 里非常基础：
1. 设置 logger 的级别
2. 把日志输出到终端
3. 可选地把日志追加写到文件

它主要被以下文件使用：
- `main.py`：启动 benchmark 时初始化 logger
- `utils.py`：打印模型回复、工具调用、工具结果时依赖 logger

你可以把它理解成：
- 官方代码里所有“可读的运行痕迹”，最终都要经过这里配置

说明：
- 这里尽量保持官方结构和行为，方便对照阅读
- 我会把几个容易忽略的小细节直接写在注释里
- 这个文件主要用于学习，不建议直接替换正式脚本
"""

import logging
import sys


# ------------------------------------------------------------------------------
# 全局日志格式器
# ------------------------------------------------------------------------------
#
# 这个 formatter 只在“文件日志”里被显式使用。
# 官方代码并没有给 stdout / stderr handler 设置 formatter，
# 所以终端里看到的格式和文件里的格式并不完全一样。

LOG_FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ------------------------------------------------------------------------------
# 设置 root logger 级别
# ------------------------------------------------------------------------------


def set_root_log_level(log_level: str):
    """
    设置 root logger 的级别。

    这个函数支持两种输入：
    - 字符串级别，例如 `"INFO"`、`"WARNING"`
    - 数字级别，例如 `"20"`、`"30"`

    实现思路：
    1. 先默认把传入值转成大写字符串
    2. 如果它本身其实是数字字符串，再转成 int
    3. 调用 `logger.setLevel(...)`
    4. 如果失败，就回退到 `NOTSET`

    这里设置的是 root logger，不是某一个具体 handler 的级别。
    后面每个 handler 还会再单独设置自己的 level。
    """

    logger = logging.getLogger()
    level: int | str = log_level.upper()
    try:
        level = int(log_level)
    except ValueError:
        pass
    try:
        logger.setLevel(level)  # type: ignore[arg-type]
    except Exception:
        logger.warning(f"Failed to set logging level to {log_level}, using default 'NOTSET'")
        logger.setLevel(logging.NOTSET)


# ------------------------------------------------------------------------------
# 初始化 logger
# ------------------------------------------------------------------------------


def init_logger(log_file: str | None = None, name: str | None = None, level: str = "NOTSET"):
    """
    初始化并返回一个 logger。

    参数：
    - `log_file`：如果提供，就额外把日志写进文件
    - `name`：要获取哪个 logger；默认是 root logger
    - `level`：root logger 的级别

    整体结构：
    1. 先设置 root logger 级别
    2. 取到指定名字的 logger
    3. 配两个终端 handler：
       - stdout：级别为 INFO
       - stderr：级别为 WARNING
    4. 如果提供 `log_file`，再加一个文件 handler

    这个设计的直觉是：
    - 普通信息打到 stdout
    - 警告和错误打到 stderr
    - 需要持久化时，再写文件

    但这里有几个很值得注意的实现细节：
    1. `stdout_handler.setLevel(logging.INFO)`
       这意味着 WARNING / ERROR 其实也满足 stdout handler 的阈值。
    2. `stderr_handler.setLevel(logging.WARNING)`
       所以 WARNING 以上日志会同时进入 stdout 和 stderr。
       也就是说，警告/错误有可能在终端里出现“双份”。
    3. `logger.handlers.clear()`
       会清掉当前 logger 之前已有的 handler。
       这能避免重复追加，但如果你在别处手工配过 logger，也会被覆盖。
    4. 文件 handler 是加到 `logging.getLogger()` 返回的 root logger 上，
       而不是前面那个 `logging.getLogger(name)` 的对象上。
       所以当 `name` 不为空时，终端 handler 和文件 handler 可能挂在不同层级的 logger 上。
    5. stdout / stderr handler 没有显式设置 formatter，
       所以终端输出格式取决于 logging 默认行为或外部环境；
       只有文件日志明确使用了 `LOG_FORMATTER`。
    """

    set_root_log_level(level)
    logger = logging.getLogger(name)

    # stdout: 输出 INFO 及以上日志。
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)

    # stderr: 输出 WARNING 及以上日志。
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)

    # 清空旧 handler，避免重复叠加。
    logger.handlers.clear()
    logger.handlers.append(stdout_handler)
    logger.handlers.append(stderr_handler)

    if log_file is not None:
        # 文件日志：追加写入，级别为 INFO 及以上，并使用统一格式。
        file_handler = logging.FileHandler(log_file, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(LOG_FORMATTER)

        # 官方原代码这里重新拿了一次 root logger，再把 file handler 挂上去。
        # 学习时一定要注意：这不一定和上面的 `logger = logging.getLogger(name)` 是同一个对象。
        logger = logging.getLogger()
        logger.addHandler(file_handler)

    return logger
