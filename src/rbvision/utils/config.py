"""
配置加载: 从 YAML 文件读取, 支持默认值和本地覆盖.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """配置容器, 支持点号访问: cfg.modbus.port"""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or {}

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        value = self._data.get(key)
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


def load_config(
    config_path: str | Path, local_override: bool = True
) -> Config:
    """
    加载 YAML 配置.

    Args:
        config_path: 主配置文件路径 (e.g. config/default_robot.yaml)
        local_override: 是否尝试加载同名 local_*.yaml 覆盖

    Returns:
        Config 对象
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if local_override:
        # 尝试加载 local_<filename>.yaml 覆盖
        local_path = config_path.parent / f"local_{config_path.name}"
        if local_path.exists():
            with open(local_path, "r", encoding="utf-8") as f:
                local_data = yaml.safe_load(f) or {}
            data = _deep_merge(data, local_data)

    return Config(data)


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并 dict, override 优先"""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def get_config_dir() -> Path:
    """获取配置目录 (项目根的 config/)"""
    return Path(__file__).parent.parent.parent.parent / "config"


def get_user_data_dir() -> Path:
    """获取用户数据目录 (用于标定文件/日志/录制)"""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", "~")).expanduser()
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    data_dir = base / "RBVISION"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
