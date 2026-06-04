"""
日志工具 (基于 loguru).
默认输出到控制台 + 文件, 按天切分.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _logger

from rbvision.utils.config import get_user_data_dir


def get_logger(
    name: Optional[str] = None, level: str = "INFO", log_to_file: bool = True
):
    """
    获取 logger 实例.

    Args:
        name: logger 名称 (用于标签)
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_to_file: 是否输出到文件
    """
    # loguru 是单例, 重新配置会覆盖
    _logger.remove()

    # 控制台
    _logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if log_to_file:
        log_dir = get_user_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _logger.add(
            str(log_dir / "rbvision_{time:YYYY-MM-DD}.log"),
            level=level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            rotation="00:00",      # 每天 0 点切分
            retention="7 days",    # 保留 7 天
            encoding="utf-8",
            enqueue=True,          # 线程安全
        )

    if name:
        return _logger.bind(name=name)
    return _logger
