"""
DH 参数模型测试.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbvision.robot import DHLink, DHParams


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_ROBOT_YAML = CONFIG_DIR / "default_robot.yaml"


class TestDHLink:
    """DHLink 数据类基本行为."""

    def test_construction_defaults(self):
        """默认 theta_offset = 0.0."""
        link = DHLink(a=0.1, alpha=0.2, d=0.3)
        assert link.a == 0.1
        assert link.alpha == 0.2
        assert link.d == 0.3
        assert link.theta_offset == 0.0

    def test_construction_full(self):
        link = DHLink(a=0.1, alpha=0.2, d=0.3, theta_offset=1.5)
        assert link.theta_offset == 1.5


class TestDHParamsBasics:
    """DHParams 基本行为."""

    def test_n_joints_property(self):
        """n_joints 返回 link 数量."""
        dh = DHParams.scara_4dof()
        assert dh.n_joints == 4

    def test_n_joints_zero(self):
        """空 link 列表的 n_joints = 0."""
        dh = DHParams(links=[])
        assert dh.n_joints == 0

    def test_from_dict_basic(self):
        """from_dict 能解析标准结构."""
        data = {
            "dh_params": [
                {"a": 0.0, "alpha": 0.0, "d": 0.0, "theta_offset": 0.0},
                {"a": 0.1, "alpha": 0.0, "d": 0.2, "theta_offset": 0.5},
            ]
        }
        dh = DHParams.from_dict(data)
        assert dh.n_joints == 2
        assert dh.links[0].a == 0.0
        assert dh.links[1].a == 0.1
        assert dh.links[1].d == 0.2
        assert dh.links[1].theta_offset == 0.5

    def test_from_dict_missing_key_raises(self):
        """from_dict 缺少 dh_params 键抛 KeyError."""
        with pytest.raises(KeyError):
            DHParams.from_dict({"other_key": []})

    def test_from_dict_empty_list_raises(self):
        """from_dict 收到空列表抛 ValueError."""
        with pytest.raises(ValueError):
            DHParams.from_dict({"dh_params": []})

    def test_from_dict_missing_field_raises(self):
        """from_dict link 缺字段抛 ValueError."""
        data = {"dh_params": [{"a": 0.0, "alpha": 0.0}]}  # 缺 d
        with pytest.raises(ValueError):
            DHParams.from_dict(data)

    def test_from_dict_default_theta_offset(self):
        """link 缺 theta_offset 时默认为 0."""
        data = {"dh_params": [{"a": 0.0, "alpha": 0.0, "d": 0.0}]}
        dh = DHParams.from_dict(data)
        assert dh.links[0].theta_offset == 0.0


class TestFactories:
    """工厂方法."""

    def test_scara_4dof_factory(self):
        """scara_4dof 返回 4-DOF, 与 config/default_robot.yaml 等效."""
        dh = DHParams.scara_4dof()
        assert dh.n_joints == 4
        assert dh.links[0].a == 0.0
        assert dh.links[1].a == pytest.approx(0.225)
        assert dh.links[2].a == pytest.approx(0.175)
        assert dh.links[2].alpha == pytest.approx(3.14159265)
        assert dh.links[3].d == pytest.approx(0.05)

    def test_ur5_6dof_factory(self):
        """ur5_6dof 返回 6-DOF."""
        dh = DHParams.ur5_6dof()
        assert dh.n_joints == 6
        assert dh.links[0].d == pytest.approx(0.089)
        assert dh.links[1].a == pytest.approx(-0.425)


class TestYamlLoading:
    """YAML 加载路径."""

    def test_yaml_load_matches_scara_4dof(self, tmp_path: Path):
        """
        关键测试: 从 config/default_robot.yaml 加载, 必须与 scara_4dof() 等效.

        实际工程中这两个来源必须一致 (前者是 config 文件, 后者是代码默认).
        """
        # 1. 加载 YAML
        assert DEFAULT_ROBOT_YAML.exists(), f"YAML missing: {DEFAULT_ROBOT_YAML}"
        dh_yaml = DHParams.from_yaml(DEFAULT_ROBOT_YAML)
        # 2. 与工厂方法对比
        dh_factory = DHParams.scara_4dof()
        assert dh_yaml == dh_factory
        assert dh_yaml.n_joints == dh_factory.n_joints

    def test_missing_yaml_raises(self, tmp_path: Path):
        """加载不存在的 YAML 抛 FileNotFoundError."""
        nonexistent = tmp_path / "no_such_file.yaml"
        with pytest.raises(FileNotFoundError):
            DHParams.from_yaml(nonexistent)

    def test_yaml_roundtrip(self, tmp_path: Path):
        """写到临时 YAML 再加载回来, 参数应保持不变."""
        import yaml as _yaml

        data = {
            "robot": {"name": "test"},
            "dh_params": [
                {"a": 0.0, "alpha": 0.0, "d": 0.0, "theta_offset": 0.0},
                {"a": 0.5, "alpha": 1.57, "d": 0.1, "theta_offset": 0.0},
            ],
        }
        yaml_path = tmp_path / "roundtrip.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            _yaml.safe_dump(data, f)

        dh = DHParams.from_yaml(yaml_path)
        assert dh.n_joints == 2
        assert dh.links[1].a == 0.5
        assert dh.links[1].alpha == 1.57
        assert dh.links[1].d == 0.1
