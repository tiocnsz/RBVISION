"""
正运动学 (Forward Kinematics) - Modified DH 链式乘积.
"""

from __future__ import annotations

from typing import List

import numpy as np

from rbvision.core.transform import dh_transform
from rbvision.robot.dh import DHParams


def _validate_q(q: np.ndarray, expected_len: int, label: str = "q") -> np.ndarray:
    """
    校验 q 的长度、有限性, 必要时转 float64.

    Args:
        q: 关节角向量.
        expected_len: 期望长度.
        label: 错误信息中的变量名.

    Returns:
        float64 类型的 1D np.ndarray.

    Raises:
        ValueError: 长度不符, 非有限值, 或非 1D.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.ndim != 1:
        raise ValueError(f"{label} must be 1D, got shape {q_arr.shape}")
    if q_arr.shape[0] != expected_len:
        raise ValueError(
            f"{label} length {q_arr.shape[0]} != joints {expected_len}"
        )
    if not np.all(np.isfinite(q_arr)):
        raise ValueError(f"{label} contains non-finite values (nan/inf)")
    return q_arr


def forward_kinematics(q: np.ndarray, dh: DHParams) -> np.ndarray:
    """
    正运动学: 关节角 → 末端 4x4 齐次变换矩阵.

    T = T_base @ T_01 @ T_12 @ ... @ T_n-1,n
    其中 T_i = Rot_x(α_i) @ Trans_x(a_i) @ Rot_z(q_i + θ_offset_i) @ Trans_z(d_i).

    Args:
        q: 关节变量向量 (n,), 转动关节单位 rad, 移动关节单位 m.
        dh: DH 参数 (任意 DOF).

    Returns:
        4x4 末端变换矩阵 (基坐标系 → 末端坐标系).

    Raises:
        ValueError: q 长度不匹配, 或 q 含 nan/inf, 或 DH 为空.
    """
    if dh.n_joints == 0:
        raise ValueError("DHParams has no links; cannot compute FK")
    q_arr = _validate_q(q, dh.n_joints, label="q")

    T = np.eye(4)
    for qi, link in zip(q_arr, dh.links):
        theta = float(qi) + link.theta_offset
        T = T @ dh_transform(a=link.a, alpha=link.alpha, d=link.d, theta=theta)
    return T


def link_transforms(q: np.ndarray, dh: DHParams) -> List[np.ndarray]:
    """
    返回所有 link 的累积 4x4 变换 (用于 3D 渲染).

    Returns:
        长度为 n+1 的列表: [T_base=I, T_01, T_02, ..., T_0n].

    Raises:
        ValueError: q 长度不匹配, q 含 nan/inf, 或 DH 为空.
    """
    if dh.n_joints == 0:
        raise ValueError("DHParams has no links; cannot compute link transforms")
    q_arr = _validate_q(q, dh.n_joints, label="q")

    transforms: List[np.ndarray] = [np.eye(4)]
    T = np.eye(4)
    for qi, link in zip(q_arr, dh.links):
        theta = float(qi) + link.theta_offset
        T = T @ dh_transform(a=link.a, alpha=link.alpha, d=link.d, theta=theta)
        transforms.append(T.copy())
    return transforms
