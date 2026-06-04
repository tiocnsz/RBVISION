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


class TestEdgeCases:
    """边界条件: 0 link, 1 link, 多 profile 对比."""

    def test_single_link_dh(self):
        """单 link 的 DHParams 也能正确创建 (退化情况, DOF=1)."""
        dh = DHParams(links=[DHLink(a=0.5, alpha=0.0, d=0.1, theta_offset=0.0)])
        assert dh.n_joints == 1
        assert dh.links[0].a == 0.5
        assert dh.links[0].d == 0.1

    def test_single_link_yaml(self, tmp_path: Path):
        """单 link 也能从 YAML 加载."""
        import yaml as _yaml

        data = {"dh_params": [{"a": 0.3, "alpha": 0.0, "d": 0.2, "theta_offset": 0.0}]}
        p = tmp_path / "single.yaml"
        with open(p, "w", encoding="utf-8") as f:
            _yaml.safe_dump(data, f)
        dh = DHParams.from_yaml(p)
        assert dh.n_joints == 1
        assert dh.links[0].a == 0.3

    def test_multi_profile_consistency(self):
        """多 profile 对比: SCARA 和 UR5 必须有不同 DOF, 互不相等.

        确保 factory 不会返回相同的 DHParams.
        """
        dh_scara = DHParams.scara_4dof()
        dh_ur5 = DHParams.ur5_6dof()
        assert dh_scara.n_joints == 4
        assert dh_ur5.n_joints == 6
        assert dh_scara != dh_ur5
        # 第一个 link 的 alpha 也不同 (SCARA=0, UR5=1.5708)
        assert dh_scara.links[0].alpha != dh_ur5.links[0].alpha
        # 第一个 link 的 d 也不同
        assert dh_scara.links[0].d != dh_ur5.links[0].d

    def test_equality_same_content(self):
        """相同内容的 DHParams 应相等."""
        a = DHParams(links=[DHLink(a=0.1, alpha=0.2, d=0.3, theta_offset=0.0)])
        b = DHParams(links=[DHLink(a=0.1, alpha=0.2, d=0.3, theta_offset=0.0)])
        assert a == b

    def test_equality_different_content(self):
        """不同内容的 DHParams 应不相等."""
        a = DHParams(links=[DHLink(a=0.1, alpha=0.2, d=0.3)])
        b = DHParams(links=[DHLink(a=0.1, alpha=0.2, d=0.4)])  # d 不同
        assert a != b

    def test_equality_different_dof(self):
        """不同 DOF 数的 DHParams 不相等."""
        a = DHParams(links=[DHLink(a=0.1, alpha=0.0, d=0.0)])
        b = DHParams(links=[DHLink(a=0.1, alpha=0.0, d=0.0),
                            DHLink(a=0.2, alpha=0.0, d=0.0)])
        assert a != b
        assert a.n_joints == 1
        assert b.n_joints == 2

    def test_repr_includes_dof(self):
        """__repr__ 包含 DOF 数, 便于调试输出."""
        dh = DHParams.scara_4dof()
        assert "4" in repr(dh)
        assert "DHParams" in repr(dh)

    def test_from_dict_field_type_conversion(self):
        """字段支持 int → float 自动转换 (YAML 解析可能返回 int)."""
        data = {"dh_params": [{"a": 1, "alpha": 0, "d": 0}]}  # 整数
        dh = DHParams.from_dict(data)
        assert isinstance(dh.links[0].a, float)
        assert dh.links[0].a == 1.0

    def test_from_dict_extra_fields_ignored(self):
        """from_dict 容许 link 含有额外字段 (如 'name'), 不会被拒绝."""
        data = {
            "dh_params": [
                {"a": 0.0, "alpha": 0.0, "d": 0.0, "theta_offset": 0.0, "name": "joint1"},
            ]
        }
        dh = DHParams.from_dict(data)
        assert dh.n_joints == 1
        assert dh.links[0].a == 0.0
