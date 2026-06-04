"""
通用 N-DOF 机械臂: 持有 DH 参数 + 关节限位 + 当前关节角, 提供正运动学.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from rbvision.core.pose import Pose
from rbvision.robot.dh import DHParams
from rbvision.robot.kinematics import forward_kinematics, link_transforms


class RobotArm:
    """通用 N-DOF 机械臂, 基于 Modified DH (Craig 格式) 参数."""

    def __init__(
        self,
        dh: DHParams,
        joint_limits: List[Tuple[float, float]],
        name: str = "Generic Arm",
    ) -> None:
        """
        构造机械臂.

        Args:
            dh: DH 参数.
            joint_limits: 关节限位列表, 长度必须等于 dh.n_joints, 每项 (lo, hi).
            name: 机械臂名称 (用于 __repr__).

        Raises:
            ValueError: 关节限位长度不匹配, DH 为空, 或 lo > hi.
        """
        if dh.n_joints == 0:
            raise ValueError("Cannot build RobotArm from empty DHParams")
        if len(joint_limits) != dh.n_joints:
            raise ValueError(
                f"joint_limits length {len(joint_limits)} != joints {dh.n_joints}"
            )
        for idx, lim in enumerate(joint_limits):
            if not isinstance(lim, (tuple, list)) or len(lim) != 2:
                raise ValueError(
                    f"joint_limits[{idx}] must be a (lo, hi) pair, got {lim!r}"
                )
            lo, hi = float(lim[0]), float(lim[1])
            if lo > hi:
                raise ValueError(
                    f"joint_limits[{idx}] invalid: lo ({lo}) > hi ({hi})"
                )

        self.dh: DHParams = dh
        self.joint_limits: List[Tuple[float, float]] = [
            (float(lo), float(hi)) for lo, hi in joint_limits
        ]
        self.name: str = name
        # 内部状态: 当前关节角, 默认全零
        self._q: np.ndarray = np.zeros(dh.n_joints, dtype=np.float64)

    # ---------- 属性 ----------

    @property
    def n_joints(self) -> int:
        """关节数量."""
        return self.dh.n_joints

    # ---------- 关节角读写 ----------

    def get_joint_angles(self) -> np.ndarray:
        """返回当前关节角 (深拷贝, 防止外部修改)."""
        return self._q.copy()

    def set_joint_angles(
        self, q: np.ndarray, clamp: bool = True
    ) -> np.ndarray:
        """
        设置关节角.

        Args:
            q: 目标关节角向量 (n,).
            clamp: 若 True (默认), 越界值会被 clamp 到 [lo, hi] 区间.

        Returns:
            实际写入的关节角 (深拷贝).

        Raises:
            ValueError: q 长度不匹配, 或 q 含 nan/inf.
        """
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.ndim != 1:
            raise ValueError(f"q must be 1D, got shape {q_arr.shape}")
        if q_arr.shape[0] != self.n_joints:
            raise ValueError(
                f"q length {q_arr.shape[0]} != joints {self.n_joints}"
            )
        if not np.all(np.isfinite(q_arr)):
            raise ValueError("q contains non-finite values (nan/inf)")

        if clamp:
            q_arr = self.clamp_to_limits(q_arr)
        self._q = q_arr.astype(np.float64, copy=True)
        return self._q.copy()

    def clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        """
        将 q 限制到 [lo, hi] 区间 (不修改内部状态).

        Args:
            q: 输入关节角.

        Returns:
            clamp 后的关节角 (新数组).
        """
        q_arr = np.asarray(q, dtype=np.float64).copy()
        if q_arr.ndim != 1 or q_arr.shape[0] != self.n_joints:
            raise ValueError(
                f"q must be 1D length {self.n_joints}, got shape {q_arr.shape}"
            )
        for i, (lo, hi) in enumerate(self.joint_limits):
            if q_arr[i] < lo:
                q_arr[i] = lo
            elif q_arr[i] > hi:
                q_arr[i] = hi
        return q_arr

    def is_within_limits(self, q: np.ndarray) -> bool:
        """
        判断 q 是否完全在限位范围内.

        Args:
            q: 待检测的关节角.

        Returns:
            True 表示所有关节都在 [lo, hi] 内.

        Raises:
            ValueError: q 长度不匹配.
        """
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.ndim != 1 or q_arr.shape[0] != self.n_joints:
            raise ValueError(
                f"q must be 1D length {self.n_joints}, got shape {q_arr.shape}"
            )
        return all(lo <= qi <= hi for qi, (lo, hi) in zip(q_arr, self.joint_limits))

    # ---------- 正运动学 ----------

    def forward_kinematics(self) -> Pose:
        """正运动学: 当前关节角 → 末端 Pose."""
        T = forward_kinematics(self._q, self.dh)
        return Pose.from_matrix(T)

    def get_link_transforms(self) -> List[np.ndarray]:
        """获取所有 link 的累积 4x4 变换 (含基坐标系, 共 n+1 个)."""
        return link_transforms(self._q, self.dh)

    # ---------- 其他 ----------

    def reset(self) -> None:
        """重置关节角到零位."""
        self._q = np.zeros(self.n_joints, dtype=np.float64)

    def teach(self) -> None:
        """
        进入示教模式 (M4 预留接口).

        示教模式允许用户手动拖动机械臂末端, 记录关键位姿用于回放.
        当前 milestone 尚未实现, 调用即抛 NotImplementedError.

        Raises:
            NotImplementedError: M4 尚未发布.

        Notes:
            · M2: 仿真模式下, 通过 VisPy 拖动机械臂末端 (3D 抓取)
            · M4: 真实 Modbus 设备 + 安全联锁 (急停 + 限位检查)
        """
        raise NotImplementedError(
            "teach() is reserved for M4 (teach-pendant + playback). "
            "Use set_joint_angles() for now to manually set joint positions."
        )

    def __repr__(self) -> str:
        q_list = [round(float(qi), 4) for qi in self._q]
        return f"{self.name} ({self.n_joints}-DOF, q={q_list})"
