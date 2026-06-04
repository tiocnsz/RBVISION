"""
RobotArm 类测试.
"""

from __future__ import annotations

import numpy as np
import pytest

from rbvision.core.pose import Pose
from rbvision.robot import DHParams, RobotArm, forward_kinematics


class TestRobotArmCreation:
    """构造与基本属性."""

    def test_arm_creation(self):
        """正常构造."""
        dh = DHParams.scara_4dof()
        limits = [(-3.14, 3.14)] * 4
        arm = RobotArm(dh, joint_limits=limits, name="TestArm")
        assert arm.n_joints == 4
        assert arm.name == "TestArm"
        assert arm.dh is dh
        # 默认 q = zeros
        np.testing.assert_array_equal(arm.get_joint_angles(), np.zeros(4))

    def test_arm_creation_default_name(self):
        """name 默认为 'Generic Arm'."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4)
        assert arm.name == "Generic Arm"

    def test_arm_joint_limits_mismatch_raises(self):
        """joint_limits 长度不匹配抛 ValueError."""
        dh = DHParams.scara_4dof()
        with pytest.raises(ValueError):
            RobotArm(dh, [(-1.0, 1.0)] * 3)  # 少 1
        with pytest.raises(ValueError):
            RobotArm(dh, [(-1.0, 1.0)] * 5)  # 多 1

    def test_arm_empty_dh_raises(self):
        """DHParams 为空抛 ValueError."""
        dh = DHParams(links=[])
        with pytest.raises(ValueError):
            RobotArm(dh, [])

    def test_arm_invalid_limit_pair_raises(self):
        """joint_limits 单项不是 (lo, hi) 抛 ValueError."""
        dh = DHParams.scara_4dof()
        with pytest.raises(ValueError):
            RobotArm(dh, [(-1.0, 1.0), (1.0,), (-1.0, 1.0), (-1.0, 1.0)])

    def test_arm_lo_gt_hi_raises(self):
        """joint_limits 单项 lo > hi 抛 ValueError."""
        dh = DHParams.scara_4dof()
        with pytest.raises(ValueError):
            RobotArm(dh, [(1.0, -1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)])


class TestJointAngles:
    """关节角读写."""

    def test_set_get_joint_angles(self):
        """set + get 关节角 (含 clamp 默认行为)."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4)
        q_in = np.array([0.1, 0.2, 0.3, 0.4])
        arm.set_joint_angles(q_in)
        np.testing.assert_array_equal(arm.get_joint_angles(), q_in)

    def test_set_joint_angles_returns_written(self):
        """set_joint_angles 返回实际写入的值 (含 clamp 修正)."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4)
        result = arm.set_joint_angles(np.array([0.5, 0.5, 0.5, 0.5]))
        np.testing.assert_array_equal(result, [0.5, 0.5, 0.5, 0.5])

    def test_set_joint_angles_no_clamp(self):
        """clamp=False 允许越界 (用于离线仿真)."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4)
        arm.set_joint_angles(np.array([5.0, 0.0, 0.0, 0.0]), clamp=False)
        np.testing.assert_array_equal(arm.get_joint_angles(), [5.0, 0.0, 0.0, 0.0])

    def test_set_joint_angles_length_mismatch_raises(self):
        """set 长度不匹配抛 ValueError."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4)
        with pytest.raises(ValueError):
            arm.set_joint_angles(np.zeros(3))

    def test_set_joint_angles_nan_raises(self):
        """set 含 NaN 抛 ValueError."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4)
        with pytest.raises(ValueError):
            arm.set_joint_angles(np.array([0.0, np.nan, 0.0, 0.0]))

    def test_get_returns_copy(self):
        """get 返回拷贝, 外部修改不影响内部状态."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4)
        q_out = arm.get_joint_angles()
        q_out[0] = 99.0
        assert arm.get_joint_angles()[0] == 0.0


class TestJointLimits:
    """关节限位行为."""

    def test_clamp_to_limits(self):
        """clamp_to_limits 修正越界值."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4)
        q_raw = np.array([-2.0, 0.5, 5.0, 0.0])
        q_clamped = arm.clamp_to_limits(q_raw)
        np.testing.assert_array_equal(q_clamped, [-1.0, 0.5, 1.0, 0.0])
        # clamp 不修改原数组
        np.testing.assert_array_equal(q_raw, [-2.0, 0.5, 5.0, 0.0])

    def test_clamp_does_not_modify_internal(self):
        """clamp_to_limits 不修改内部状态 (纯函数)."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4)
        arm.set_joint_angles(np.array([0.5, 0.5, 0.5, 0.5]))
        arm.clamp_to_limits(np.array([5.0, 0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(arm.get_joint_angles(), [0.5, 0.5, 0.5, 0.5])

    def test_is_within_limits(self):
        """is_within_limits 正确判断."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4)
        assert arm.is_within_limits(np.array([0.5, 0.5, 0.5, 0.5])) is True
        assert arm.is_within_limits(np.array([0.0, 0.0, 0.0, 0.0])) is True
        # 越界
        assert arm.is_within_limits(np.array([1.5, 0.0, 0.0, 0.0])) is False
        assert arm.is_within_limits(np.array([0.0, 0.0, 0.0, -1.5])) is False
        # 边界值
        assert arm.is_within_limits(np.array([1.0, 0.0, 0.0, 0.0])) is True
        assert arm.is_within_limits(np.array([-1.0, 0.0, 0.0, 0.0])) is True

    def test_set_auto_clamp(self):
        """set 默认 clamp=True 时, 越界被自动修正."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4)
        arm.set_joint_angles(np.array([2.0, 0.5, -3.0, 0.0]))
        np.testing.assert_array_equal(arm.get_joint_angles(), [1.0, 0.5, -1.0, 0.0])


class TestForwardKinematicsViaArm:
    """RobotArm.forward_kinematics 行为."""

    def test_arm_forward_kinematics_uses_pose(self):
        """arm.fk() 返回 Pose, 与裸 forward_kinematics 结果一致."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-3.14, 3.14)] * 4, name="SCARA")
        arm.set_joint_angles(np.array([0.0, 0.0, 0.0, 0.0]))
        pose = arm.forward_kinematics()
        assert isinstance(pose, Pose)
        # 与函数式 FK 一致
        T = forward_kinematics(np.zeros(4), dh)
        np.testing.assert_allclose(pose.to_matrix(), T, atol=1e-12)
        # 位姿字段
        np.testing.assert_allclose(pose.x, 0.4, atol=1e-6)
        np.testing.assert_allclose(pose.y, 0.0, atol=1e-6)
        np.testing.assert_allclose(pose.z, -0.05, atol=1e-6)

    def test_arm_fk_reflects_joint_change(self):
        """改 q 后 fk 跟着变."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-3.14, 3.14)] * 4)
        arm.set_joint_angles(np.array([np.pi / 2, 0.0, 0.0, 0.0]))
        pose = arm.forward_kinematics()
        np.testing.assert_allclose(pose.x, 0.0, atol=1e-6)
        np.testing.assert_allclose(pose.y, 0.4, atol=1e-6)

    def test_arm_get_link_transforms(self):
        """get_link_transforms 长度 = n+1, 最后一项与 fk 一致."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-3.14, 3.14)] * 4)
        transforms = arm.get_link_transforms()
        assert len(transforms) == 5
        np.testing.assert_allclose(transforms[-1], arm.forward_kinematics().to_matrix(), atol=1e-12)


class TestReset:
    """reset 行为."""

    def test_reset_returns_to_zero(self):
        """reset() 把 q 恢复到全零."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4)
        arm.set_joint_angles(np.array([0.5, 0.5, 0.5, 0.5]))
        arm.reset()
        np.testing.assert_array_equal(arm.get_joint_angles(), np.zeros(4))


class TestRepr:
    """__repr__ 调试输出."""

    def test_repr_includes_name_and_dof(self):
        """__repr__ 包含机械臂名和 DOF 数."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4, name="Demo")
        r = repr(arm)
        assert "Demo" in r
        assert "4-DOF" in r

    def test_repr_includes_current_q(self):
        """__repr__ 包含当前关节角 (调试时方便)."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4, name="X")
        arm.set_joint_angles(np.array([0.1, 0.2, 0.3, 0.4]))
        r = repr(arm)
        # 关节角应该被 round 到 4 位小数
        assert "0.1" in r
        assert "0.4" in r

    def test_repr_uses_default_name(self):
        """__repr__ 在未指定 name 时使用 'Generic Arm'."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-1.0, 1.0)] * 4)
        assert "Generic Arm" in repr(arm)


class TestTeachReserved:
    """teach() 预留接口 (M4 实现)."""

    def test_teach_raises_not_implemented(self):
        """teach() 当前抛 NotImplementedError, 错误信息提示 M4."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-3.14, 3.14)] * 4)
        with pytest.raises(NotImplementedError) as exc_info:
            arm.teach()
        assert "M4" in str(exc_info.value) or "reserved" in str(exc_info.value).lower()

    def test_teach_does_not_modify_state(self):
        """teach() 抛错后, 关节角状态不应被破坏."""
        dh = DHParams.scara_4dof()
        arm = RobotArm(dh, [(-3.14, 3.14)] * 4)
        arm.set_joint_angles(np.array([0.1, 0.2, 0.3, 0.4]))
        q_before = arm.get_joint_angles()
        with pytest.raises(NotImplementedError):
            arm.teach()
        np.testing.assert_array_equal(arm.get_joint_angles(), q_before)


class TestStateIndependence:
    """多个 RobotArm 实例的状态互相独立 (确保没有 class-level 共享状态)."""

    def test_two_arms_independent_q(self):
        """两个 RobotArm 实例的 q 互不影响."""
        dh = DHParams.scara_4dof()
        arm1 = RobotArm(dh, [(-1.0, 1.0)] * 4, name="Arm1")
        arm2 = RobotArm(dh, [(-1.0, 1.0)] * 4, name="Arm2")
        arm1.set_joint_angles(np.array([0.5, 0.5, 0.5, 0.5]))
        # arm2 应该是默认全零
        np.testing.assert_array_equal(arm2.get_joint_angles(), np.zeros(4))
        # 改 arm2 不影响 arm1
        arm2.set_joint_angles(np.array([0.9, 0.9, 0.9, 0.9]))
        np.testing.assert_array_equal(arm1.get_joint_angles(), [0.5, 0.5, 0.5, 0.5])

    def test_reset_isolated(self):
        """一个 arm 的 reset 不影响另一个 arm."""
        dh = DHParams.scara_4dof()
        arm1 = RobotArm(dh, [(-1.0, 1.0)] * 4)
        arm2 = RobotArm(dh, [(-1.0, 1.0)] * 4)
        arm1.set_joint_angles(np.array([0.5, 0.5, 0.5, 0.5]))
        arm2.set_joint_angles(np.array([0.7, 0.7, 0.7, 0.7]))
        arm1.reset()
        np.testing.assert_array_equal(arm1.get_joint_angles(), np.zeros(4))
        np.testing.assert_array_equal(arm2.get_joint_angles(), [0.7, 0.7, 0.7, 0.7])


class TestDHEquality:
    """DHParams 在不同构造路径下的一致性."""

    def test_yaml_matches_factory(self):
        """YAML 加载的 SCARA 应与工厂方法一致."""
        from pathlib import Path

        yaml_path = Path(__file__).resolve().parent.parent / "config" / "default_robot.yaml"
        dh_yaml = DHParams.from_yaml(yaml_path)
        dh_factory = DHParams.scara_4dof()
        assert dh_yaml == dh_factory

    def test_ur5_arm_creation(self):
        """UR5 6-DOF 也能创建 RobotArm."""
        dh = DHParams.ur5_6dof()
        arm = RobotArm(dh, [(-3.14, 3.14)] * 6, name="UR5")
        assert arm.n_joints == 6
        assert arm.name == "UR5"
        # 6-DOF FK 应能正常工作
        pose = arm.forward_kinematics()
        assert pose is not None
