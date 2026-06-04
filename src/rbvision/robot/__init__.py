"""
机械臂模块: DH 参数 + 正运动学 + 通用 N-DOF 机械臂.

公开 API:
  · DHLink: 单连杆 DH 参数 (Modified DH / Craig 格式)
  · DHParams: DH 参数集合, 支持 YAML / dict 加载和工厂方法
  · forward_kinematics(q, dh) -> 末端 4x4 变换
  · link_transforms(q, dh) -> n+1 个累积 4x4 变换 (含基坐标系)
  · jacobian(q, dh) -> 6×n 雅可比矩阵 (M3 预留接口, 当前抛 NotImplementedError)
  · RobotArm: 持有 DH + 限位 + 当前关节角的高层封装

详细文档见 docs/robot_api.md.
"""

from rbvision.robot.arm import RobotArm
from rbvision.robot.dh import DHLink, DHParams
from rbvision.robot.kinematics import (
    forward_kinematics,
    jacobian,
    link_transforms,
)

__all__ = [
    "DHLink",
    "DHParams",
    "forward_kinematics",
    "link_transforms",
    "jacobian",
    "RobotArm",
]
