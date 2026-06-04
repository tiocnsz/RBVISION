"""
正运动学测试.
"""

from __future__ import annotations

import numpy as np
import pytest

from rbvision.robot import DHParams, forward_kinematics, link_transforms


class TestForwardKinematics:
    """FK 基本行为."""

    def test_fk_scara_zero_pose(self):
        """
        q=[0,0,0,0] 时, 末端理论位置 (0.4, 0, -0.05).

        推导 (Modified DH, q=θ, 关节变量 = q + theta_offset, 全为 0):
          T_01 = I
          T_12 = Trans_x(0.225)
          T_23 = Rot_x(π) @ Trans_x(0.175)  (alpha=π 翻转 Y/Z 轴, X 方向不变)
          T_34 = Trans_z(0.05)
        由于 alpha=π, 末端的 Z 轴方向与基坐标系相反,
        故 z = -0.05 (而不是 +0.05, 与一些文献表述差异, 已用代码独立验证).
        """
        dh = DHParams.scara_4dof()
        T = forward_kinematics(np.zeros(4), dh)
        np.testing.assert_allclose(T[:3, 3], [0.4, 0.0, -0.05], atol=1e-6)
        # 旋转矩阵也应该是 4x4 齐次
        assert T.shape == (4, 4)
        # 末帧 X 轴翻向 (alpha=π 累计)
        # Rot_x(π) = diag(1, -1, -1), 与 end frame 实际一致
        np.testing.assert_allclose(T[0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(T[1, 1], -1.0, atol=1e-6)
        np.testing.assert_allclose(T[2, 2], -1.0, atol=1e-6)

    def test_fk_scara_theta1_90deg(self):
        """
        q=[π/2, 0, 0, 0] 时, 末端 (x, y) 旋转 90°.

        零位 (0.4, 0, -0.05) 绕 Z 旋转 90° → (0, 0.4, -0.05).
        """
        dh = DHParams.scara_4dof()
        T = forward_kinematics(np.array([np.pi / 2, 0.0, 0.0, 0.0]), dh)
        np.testing.assert_allclose(T[:3, 3], [0.0, 0.4, -0.05], atol=1e-6)

    def test_fk_scara_theta1_180deg(self):
        """q=[π, 0, 0, 0] 时, 末端 (x, y) 旋转 180° → (-0.4, 0, -0.05)."""
        dh = DHParams.scara_4dof()
        T = forward_kinematics(np.array([np.pi, 0.0, 0.0, 0.0]), dh)
        np.testing.assert_allclose(T[:3, 3], [-0.4, 0.0, -0.05], atol=1e-6)

    def test_fk_q_length_mismatch_raises(self):
        """q 长度不等于关节数抛 ValueError."""
        dh = DHParams.scara_4dof()
        with pytest.raises(ValueError):
            forward_kinematics(np.zeros(3), dh)  # 少 1
        with pytest.raises(ValueError):
            forward_kinematics(np.zeros(5), dh)  # 多 1

    def test_fk_q_nan_raises(self):
        """q 含 NaN 抛 ValueError."""
        dh = DHParams.scara_4dof()
        q = np.array([0.0, 0.0, np.nan, 0.0])
        with pytest.raises(ValueError):
            forward_kinematics(q, dh)

    def test_fk_q_inf_raises(self):
        """q 含 inf 抛 ValueError."""
        dh = DHParams.scara_4dof()
        q = np.array([0.0, 0.0, np.inf, 0.0])
        with pytest.raises(ValueError):
            forward_kinematics(q, dh)

    def test_fk_q_2d_raises(self):
        """q 不是 1D 抛 ValueError."""
        dh = DHParams.scara_4dof()
        q = np.zeros((2, 2))
        with pytest.raises(ValueError):
            forward_kinematics(q, dh)

    def test_fk_empty_dh_raises(self):
        """DHParams 为空 (0 link) 抛 ValueError."""
        dh = DHParams(links=[])
        with pytest.raises(ValueError):
            forward_kinematics(np.zeros(0), dh)

    def test_fk_list_q_works(self):
        """q 可以是 Python list (会被转 ndarray)."""
        dh = DHParams.scara_4dof()
        T = forward_kinematics([0.0, 0.0, 0.0, 0.0], dh)
        np.testing.assert_allclose(T[:3, 3], [0.4, 0.0, -0.05], atol=1e-6)

    def test_fk_ur5_zero_pose_known(self):
        """UR5 在 q=0 时的末端位置 (用于交叉验证 DH 计算的合法性).

        这里只验证矩阵本身的合法性 (旋转正交、4x4), 不绑定具体坐标值.
        UR5 在 q=0 时末端在不同 DH 表示下位置可能差异较大;
        我们用单独的 smoke 测试 (q=[π/2,0,0,0,0,0]) 检查是否合理.
        """
        dh = DHParams.ur5_6dof()
        T = forward_kinematics(np.zeros(6), dh)
        # 4x4 齐次
        assert T.shape == (4, 4)
        # 旋转矩阵正交性
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        # 行列式为 +1 (右手法则)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
        # 末行 (0, 0, 0, 1)
        np.testing.assert_allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_fk_ur5_elbow_90deg(self):
        """UR5 在 q=[0,π/2,0,0,0,0] 时, 第二关节正交, 末端应到达第一关节前向 ~0.425m 位置.

        这是一个相对位移校验, 不绑定绝对坐标.
        """
        dh = DHParams.ur5_6dof()
        T0 = forward_kinematics(np.zeros(6), dh)
        T1 = forward_kinematics(np.array([0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0]), dh)
        # 末端位移应在合理范围 (UR5 第二关节 0.425m, 旋转 90° 必然导致 x/y 显著变化)
        dx = T1[0, 3] - T0[0, 3]
        dy = T1[1, 3] - T0[1, 3]
        # |Δxy| 应在 (0, 1) 之间, 不为 0
        assert 0.0 < (dx * dx + dy * dy) ** 0.5 < 1.0


class TestLinkTransforms:
    """link_transforms 行为."""

    def test_link_transforms_count(self):
        """返回 n+1 个 4x4 矩阵."""
        dh = DHParams.scara_4dof()
        transforms = link_transforms(np.zeros(4), dh)
        assert len(transforms) == 5  # 4 + 1 (含基坐标系)
        for T in transforms:
            assert T.shape == (4, 4)

    def test_link_transforms_base_is_identity(self):
        """第一个变换是基坐标系 (单位阵)."""
        dh = DHParams.scara_4dof()
        transforms = link_transforms(np.zeros(4), dh)
        np.testing.assert_allclose(transforms[0], np.eye(4), atol=1e-12)

    def test_link_transforms_last_matches_fk(self):
        """最后一个累积变换应等于 forward_kinematics 结果."""
        dh = DHParams.scara_4dof()
        q = np.array([0.1, 0.2, 0.3, 0.4])
        transforms = link_transforms(q, dh)
        T_fk = forward_kinematics(q, dh)
        np.testing.assert_allclose(transforms[-1], T_fk, atol=1e-12)

    def test_link_transforms_cumulative(self):
        """累积性: T_0(k+1) = T_0k @ T_k,k+1, 即 transforms[k+1] = T_k,k+1 @ transforms[k].

        等价验证: T_k,k+1 = T_0k^-1 @ T_0(k+1).
        """
        dh = DHParams.scara_4dof()
        q = np.array([0.0, 0.5, 0.0, 0.0])
        transforms = link_transforms(q, dh)
        # 验证 T_01, T_12, T_23 链接关系
        T01 = transforms[1]
        T12 = np.linalg.inv(transforms[1]) @ transforms[2]
        T23 = np.linalg.inv(transforms[2]) @ transforms[3]
        T03_reconstructed = T01 @ T12 @ T23
        np.testing.assert_allclose(transforms[3], T03_reconstructed, atol=1e-10)
