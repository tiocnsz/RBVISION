"""
Pose: 位姿 (位置 + 姿态)
支持四元数和欧拉角两种表示.

坐标系约定:
  · 世界/基坐标系: X 前, Y 左, Z 上 (右手系)
  · 单位: 位置 m, 角度 rad
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    pass  # 避免循环导入


@dataclass
class Pose:
    """位姿: 位置 (m) + 四元数 (qx, qy, qz, qw)"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    # ---------- 工厂方法 ----------
    @classmethod
    def from_euler(
        cls, x: float, y: float, z: float, rx: float, ry: float, rz: float, order: str = "xyz"
    ) -> "Pose":
        """从欧拉角 (rad) 创建 Pose"""
        qx, qy, qz, qw = euler_to_quat(rx, ry, rz, order)
        return cls(x, y, z, qx, qy, qz, qw)

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> "Pose":
        """从 4x4 变换矩阵创建 Pose"""
        if T.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got {T.shape}")
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        qx, qy, qz, qw = rot_to_quat(T[:3, :3])
        return cls(x, y, z, qx, qy, qz, qw)

    @classmethod
    def identity(cls) -> "Pose":
        return cls()

    # ---------- 转换方法 ----------
    def to_matrix(self) -> np.ndarray:
        """转 4x4 变换矩阵 (内联实现, 避免循环导入)"""
        R = quat_to_rot(self.qx, self.qy, self.qz, self.qw)
        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3] = self.x
        T[1, 3] = self.y
        T[2, 3] = self.z
        return T

    def to_euler(self, order: str = "xyz") -> Tuple[float, float, float]:
        """转欧拉角 (rad)"""
        return quat_to_euler(self.qx, self.qy, self.qz, self.qw, order)

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            "qw": self.qw,
        }

    # ---------- 运算 ----------
    def __repr__(self) -> str:
        return (
            f"Pose(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}, "
            f"qx={self.qx:.4f}, qy={self.qy:.4f}, qz={self.qz:.4f}, qw={self.qw:.4f})"
        )


# ---------- 四元数 ↔ 欧拉角 / 旋转矩阵 ----------

def quat_to_euler(
    qx: float, qy: float, qz: float, qw: float, order: str = "xyz"
) -> Tuple[float, float, float]:
    """四元数 → 欧拉角 (rad)"""
    # roll (x), pitch (y), yaw (z)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = np.pi / 2.0 * np.sign(sinp)  # 万向锁
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    if order == "xyz":
        return roll, pitch, yaw
    elif order == "zyx":
        return yaw, pitch, roll
    else:
        raise ValueError(f"Unsupported euler order: {order}")


def euler_to_quat(
    rx: float, ry: float, rz: float, order: str = "xyz"
) -> Tuple[float, float, float, float]:
    """欧拉角 (rad) → 四元数 (qx, qy, qz, qw)"""
    if order == "xyz":
        # 先绕 X, 再 Y, 再 Z
        cx, sx = np.cos(rx / 2), np.sin(rx / 2)
        cy, sy = np.cos(ry / 2), np.sin(ry / 2)
        cz, sz = np.cos(rz / 2), np.sin(rz / 2)
        qw = cx * cy * cz - sx * sy * sz
        qx = sx * cy * cz + cx * sy * sz
        qy = cx * sy * cz - sx * cy * sz
        qz = cx * cy * sz + sx * sy * cz
        return qx, qy, qz, qw
    else:
        raise ValueError(f"Unsupported euler order: {order}")


def rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """3x3 旋转矩阵 → 四元数 (Shepperd's method)"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = 2.0 * np.sqrt(tr + 1.0)
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return qx, qy, qz, qw


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """四元数 → 3x3 旋转矩阵"""
    # 归一化
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


# ---------- Modbus 编/解码 (float32 <-> 2 个 uint16) ----------

def float_to_registers(value: float) -> Tuple[int, int]:
    """float32 → 2 个 uint16 (大端)"""
    import struct

    packed = struct.pack(">f", value)
    hi, lo = struct.unpack(">HH", packed)
    return hi, lo


def registers_to_float(hi: int, lo: int) -> float:
    """2 个 uint16 (大端) → float32"""
    import struct

    packed = struct.pack(">HH", hi & 0xFFFF, lo & 0xFFFF)
    return struct.unpack(">f", packed)[0]


def int32_to_registers(value: int) -> Tuple[int, int]:
    """int32 → 2 个 uint16 (大端)"""
    if value < 0:
        value = value + (1 << 32)
    hi = (value >> 16) & 0xFFFF
    lo = value & 0xFFFF
    return hi, lo


def registers_to_int32(hi: int, lo: int) -> int:
    """2 个 uint16 (大端) → int32"""
    value = ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)
    if value >= (1 << 31):
        value -= 1 << 32
    return value
