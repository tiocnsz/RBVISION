"""
4x4 变换矩阵运算
所有函数都是纯 NumPy, 无副作用.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rbvision.core.pose import Pose

from rbvision.core.pose import quat_to_rot


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """从 3x3 旋转矩阵和 3 维平移向量构造 4x4 齐次变换矩阵"""
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R.shape}")
    if t.shape != (3,):
        raise ValueError(f"t must be (3,), got {t.shape}")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """4x4 齐次变换矩阵求逆 (利用刚体变换性质, 更快)"""
    if T.shape != (4, 4):
        raise ValueError(f"T must be 4x4, got {T.shape}")
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compose_T(*Ts: np.ndarray) -> np.ndarray:
    """多个 4x4 矩阵按顺序相乘: T1 @ T2 @ T3 ..."""
    if not Ts:
        return np.eye(4)
    result = Ts[0]
    for T in Ts[1:]:
        result = result @ T
    return result


def pose_to_T(pose: Pose) -> np.ndarray:
    """Pose → 4x4 变换矩阵"""
    R = quat_to_rot(pose.qx, pose.qy, pose.qz, pose.qw)
    t = np.array([pose.x, pose.y, pose.z])
    return make_T(R, t)


def T_to_pose(T: np.ndarray) -> Pose:
    """4x4 变换矩阵 → Pose"""
    return Pose.from_matrix(T)


# ---------- 基础变换 ----------

def rot_x(theta: float) -> np.ndarray:
    """绕 X 轴旋转 theta 弧度"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ]
    )


def rot_y(theta: float) -> np.ndarray:
    """绕 Y 轴旋转 theta 弧度"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ]
    )


def rot_z(theta: float) -> np.ndarray:
    """绕 Z 轴旋转 theta 弧度"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def trans(x: float, y: float, z: float) -> np.ndarray:
    """平移变换矩阵"""
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Modified DH (Craig 格式) 变换矩阵:
        T = Rot_x(alpha) @ Trans_x(a) @ Rot_z(theta) @ Trans_z(d)
    """
    return rot_x(alpha) @ trans(a, 0, 0) @ rot_z(theta) @ trans(0, 0, d)
