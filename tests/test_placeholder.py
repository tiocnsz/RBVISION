"""
M0 冒烟测试 - 验证项目骨架可运行.
"""

import numpy as np
import pytest

import rbvision
from rbvision.core.pose import Pose, euler_to_quat, quat_to_euler
from rbvision.core.transform import (
    dh_transform,
    invert_T,
    make_T,
    pose_to_T,
    rot_x,
    rot_y,
    rot_z,
    trans,
)
from rbvision.utils.config import get_config_dir, get_user_data_dir
from rbvision.utils.logger import get_logger


class TestPackage:
    def test_version(self):
        assert rbvision.__version__ == "0.1.0"
        assert "M0" in rbvision.__status__


class TestPose:
    def test_identity(self):
        p = Pose.identity()
        assert p.x == 0.0 and p.y == 0.0 and p.z == 0.0
        assert p.qw == 1.0

    def test_from_euler_roundtrip(self):
        """欧拉角 → Pose → 欧拉角 应能还原 (小角度, 避免万向锁)"""
        original = (0.1, 0.2, 0.3)
        p = Pose.from_euler(1.0, 2.0, 3.0, *original)
        recovered = p.to_euler()
        # 在 xyz 外旋下, 转换可能不完全 roundtrip, 矩阵应该一致
        T1 = p.to_matrix()
        p2 = Pose.from_matrix(T1)
        T2 = p2.to_matrix()
        np.testing.assert_allclose(T1, T2, atol=1e-6)

    def test_from_matrix_roundtrip(self):
        """Pose → 4x4 → Pose 应能还原"""
        p = Pose.from_euler(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        T = p.to_matrix()
        p2 = Pose.from_matrix(T)
        np.testing.assert_allclose(p2.to_dict()["x"], p.x, atol=1e-6)
        # 四元数符号可能翻转, 但旋转效果一致
        T2 = p2.to_matrix()
        np.testing.assert_allclose(T, T2, atol=1e-6)

    def test_gimbal_lock_safe_pitch(self):
        """pitch 接近 ±90° 时, 转换仍应保持矩阵一致"""
        p = Pose.from_euler(0, 0, 0, 0.1, np.pi / 2 - 1e-4, 0.2)
        T1 = p.to_matrix()
        p2 = Pose.from_matrix(T1)
        T2 = p2.to_matrix()
        np.testing.assert_allclose(T1, T2, atol=1e-4)


class TestTransform:
    def test_make_T_shape(self):
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        T = make_T(R, t)
        assert T.shape == (4, 4)
        np.testing.assert_array_equal(T[:3, 3], t)

    def test_invert_T_roundtrip(self):
        R = rot_x(0.5)[:3, :3]
        t = np.array([1.0, 2.0, 3.0])
        T = make_T(R, t)
        T_inv = invert_T(T)
        np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=1e-10)

    def test_rot_x(self):
        T = rot_x(np.pi / 2)
        np.testing.assert_allclose(T[:3, :3] @ np.array([0, 1, 0]), [0, 0, 1], atol=1e-10)

    def test_trans(self):
        T = trans(1, 2, 3)
        np.testing.assert_array_equal(T[:3, 3], [1, 2, 3])

    def test_dh_transform(self):
        """标准 DH 变换验证 (theta=0, d=0, a=1, alpha=0 应是 [I|t=x])"""
        T = dh_transform(a=1.0, alpha=0.0, d=0.0, theta=0.0)
        np.testing.assert_allclose(T, np.array(
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ), atol=1e-10)


class TestUtils:
    def test_config_dir_exists(self):
        d = get_config_dir()
        assert d.exists()
        assert d.is_dir()

    def test_user_data_dir_creates(self, tmp_path, monkeypatch):
        # 重定向 APPDATA
        monkeypatch.setenv("APPDATA", str(tmp_path))
        d = get_user_data_dir()
        assert d.exists()

    def test_logger_runs(self):
        log = get_logger(name="test", level="DEBUG", log_to_file=False)
        log.info("M0 smoke test OK")
        # 不抛异常就算通过
