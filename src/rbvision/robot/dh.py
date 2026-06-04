"""
DH (Denavit-Hartenberg) 参数模型.

约定 (Modified DH / Craig 格式):
  T_i = Rot_x(α_i) @ Trans_x(a_i) @ Rot_z(θ_i) @ Trans_z(d_i)
  θ_i = q_i + theta_offset_i
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


@dataclass
class DHLink:
    """Modified DH 单连杆参数 (Craig 格式)."""

    a: float              # 连杆长度 (m)
    alpha: float          # 连杆扭角 (rad)
    d: float              # 连杆偏距 (m)
    theta_offset: float = 0.0  # 关节零位偏移 (rad)


@dataclass
class DHParams:
    """DH 参数集合, 任意 DOF 通用."""

    links: List[DHLink]

    @property
    def n_joints(self) -> int:
        """关节数量 (即 link 数量)."""
        return len(self.links)

    # ---------- 工厂方法 ----------

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DHParams":
        """
        从 YAML 文件加载 DHParams.

        期望结构 (兼容 default_robot.yaml 顶层 + 裸 dh_params 列表):
          robot:
            ...
          dh_params:
            - {a: 0.0, alpha: 0.0, d: 0.0, theta_offset: 0.0}
            - ...

        Args:
            path: YAML 文件路径.

        Returns:
            解析后的 DHParams 实例.

        Raises:
            FileNotFoundError: 文件不存在.
            KeyError: 缺少 'dh_params' 字段.
            ValueError: 字段类型错误.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DH YAML not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data: Any = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DHParams":
        """
        从 dict 构造 (dict 顶层必须含 'dh_params' 键).

        容错处理:
          · 顶层 dict 若本身就是 link 列表, 自动按 'dh_params' 处理
          · 单个 link dict 视为长度为 1 的列表
          · 缺失字段抛 ValueError
        """
        if not isinstance(d, dict):
            raise ValueError(f"from_dict expects a dict, got {type(d).__name__}")

        if "dh_params" in d:
            raw_links = d["dh_params"]
        elif isinstance(d, list):
            # 极端情况: 调用者直接传 list (不在任务契约内, 但宽容处理)
            raw_links = d
        else:
            raise KeyError(
                "Missing 'dh_params' key in config dict. "
                "Expected structure: {'dh_params': [{a, alpha, d, theta_offset}, ...]}"
            )

        if not isinstance(raw_links, list) or len(raw_links) == 0:
            raise ValueError("'dh_params' must be a non-empty list of link dicts")

        links: List[DHLink] = []
        required = ("a", "alpha", "d")
        for idx, link in enumerate(raw_links):
            if not isinstance(link, dict):
                raise ValueError(
                    f"dh_params[{idx}] must be a dict, got {type(link).__name__}"
                )
            missing = [k for k in required if k not in link]
            if missing:
                raise ValueError(f"dh_params[{idx}] missing fields: {missing}")
            try:
                links.append(
                    DHLink(
                        a=float(link["a"]),
                        alpha=float(link["alpha"]),
                        d=float(link["d"]),
                        theta_offset=float(link.get("theta_offset", 0.0)),
                    )
                )
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"dh_params[{idx}] field type conversion failed: {e}"
                ) from e

        return cls(links=links)

    @classmethod
    def scara_4dof(cls) -> "DHParams":
        """默认 SCARA 4-DOF (与 config/default_robot.yaml 等效)."""
        return cls(
            links=[
                DHLink(a=0.000, alpha=0.0, d=0.000, theta_offset=0.0),
                DHLink(a=0.225, alpha=0.0, d=0.000, theta_offset=0.0),
                DHLink(a=0.175, alpha=3.14159265, d=0.000, theta_offset=0.0),
                DHLink(a=0.000, alpha=0.0, d=0.050, theta_offset=0.0),
            ]
        )

    @classmethod
    def ur5_6dof(cls) -> "DHParams":
        """UR5 6-DOF (备用)."""
        return cls(
            links=[
                DHLink(a=0.0, alpha=1.5708, d=0.089, theta_offset=0.0),
                DHLink(a=-0.425, alpha=0.0, d=0.0, theta_offset=0.0),
                DHLink(a=-0.392, alpha=0.0, d=0.0, theta_offset=0.0),
                DHLink(a=0.0, alpha=1.5708, d=0.109, theta_offset=0.0),
                DHLink(a=0.0, alpha=-1.5708, d=0.094, theta_offset=0.0),
                DHLink(a=0.0, alpha=0.0, d=0.082, theta_offset=0.0),
            ]
        )

    # ---------- 表达 ----------

    def __repr__(self) -> str:
        return f"DHParams(n_joints={self.n_joints})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DHParams):
            return NotImplemented
        if self.n_joints != other.n_joints:
            return False
        return all(
            (a.a == b.a) and (a.alpha == b.alpha) and (a.d == b.d) and
            (a.theta_offset == b.theta_offset)
            for a, b in zip(self.links, other.links)
        )
