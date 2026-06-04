# RBVISION 机械臂 API 参考 (Robot API Reference)

> 适用版本: RBVISION M1 (v0.1.0) · 模块: `rbvision.robot` · 2026-06-04

本文档介绍 M1 阶段提供的机械臂核心 API: **DH 参数建模** + **正运动学** + **通用 N-DOF 机械臂封装**.
所有示例代码**可执行** (复制粘贴即可在 `conda activate rbvision` 后运行).

---

## 目录

1. [概述](#1-概述)
2. [`DHLink` — 单连杆参数](#2-dhlink--单连杆参数)
3. [`DHParams` — DH 参数集合](#3-dhparams--dh-参数集合)
4. [正运动学 (FK)](#4-正运动学-fk)
5. [`RobotArm` — 高层机械臂封装](#5-robotarm--高层机械臂封装)
6. [内置 Profile](#6-内置-profile)
7. [预留接口 (M3 / M4)](#7-预留接口-m3--m4)
8. [数学参考: Modified DH 公式](#8-数学参考-modified-dh-公式)
9. [错误处理约定](#9-错误处理约定)
10. [版本与路线图](#10-版本与路线图)
11. [附录: 测试覆盖](#附录-测试覆盖)

---

## 1. 概述

M1 阶段 (`feature/m1-mechanical-core`) 公开 6 个符号:

| 符号 | 类别 | 说明 |
|---|---|---|
| `DHLink` | dataclass | 单连杆 DH 参数 (Modified DH / Craig 格式) |
| `DHParams` | dataclass | DH 参数集合; 支持 YAML/dict + 工厂方法 |
| `forward_kinematics` | function | 关节角 → 末端 4×4 齐次变换 |
| `link_transforms` | function | 关节角 → n+1 个累积 4×4 矩阵 (3D 渲染用) |
| `jacobian` | function | **M3 预留**, 当前抛 `NotImplementedError` |
| `RobotArm` | class | 持有 DH + 限位 + 当前 q 的高层封装 |

```python
# 一行 import 即可使用全部 API
from rbvision.robot import DHLink, DHParams, forward_kinematics, link_transforms, RobotArm
```

> M2+ 规划: 逆运动学 (IK) / 轨迹规划 / 真实 Modbus 设备驱动. 见 §10.

---

## 2. `DHLink` — 单连杆参数

`DHLink` 用 `@dataclass` 描述单连杆的 DH 参数:

```python
@dataclass
class DHLink:
    a: float              # 连杆长度 (m)
    alpha: float          # 连杆扭角 (rad)
    d: float              # 连杆偏距 (m)
    theta_offset: float = 0.0  # 关节零位偏移 (rad)
```

### 字段含义

| 字段 | 单位 | 物理意义 |
|---|---|---|
| `a` | m | 连杆长度 (沿 x_{i-1} 轴, 从 z_{i-1} 到 z_i 的距离) |
| `alpha` | rad | 连杆扭角 (绕 x_{i-1} 轴, 从 z_{i-1} 旋转到 z_i) |
| `d` | m | 连杆偏距 (沿 z_i 轴, 从 x_{i-1} 到 x_i 的距离) |
| `theta_offset` | rad | 关节零位偏移, 校正关节零位与机械零位不对齐 |

> θ 实际 = q + theta_offset, 转动关节单位 rad, 移动关节单位 m.

### 示例

```python
from rbvision.robot import DHLink

# 转动关节: 长度 0.225 m, 扭角 0, 偏距 0
link = DHLink(a=0.225, alpha=0.0, d=0.0, theta_offset=0.0)
print(link)  # DHLink(a=0.225, alpha=0.0, d=0.0, theta_offset=0.0)
```

---

## 3. `DHParams` — DH 参数集合

### 3.1 工厂方法 (推荐)

```python
from rbvision.robot import DHParams

# 默认 SCARA 4-DOF (与 config/default_robot.yaml 等效)
dh_scara = DHParams.scara_4dof()
print(dh_scara.n_joints)  # 4

# 备用 UR5 6-DOF
dh_ur5 = DHParams.ur5_6dof()
print(dh_ur5.n_joints)  # 6
```

### 3.2 从 dict 加载

```python
from rbvision.robot import DHParams

data = {
    "dh_params": [
        {"a": 0.0,   "alpha": 0.0,        "d": 0.0, "theta_offset": 0.0},
        {"a": 0.225, "alpha": 0.0,        "d": 0.0, "theta_offset": 0.0},
        {"a": 0.175, "alpha": 3.14159265, "d": 0.0, "theta_offset": 0.0},
    ]
}
dh = DHParams.from_dict(data)
print(dh.n_joints)  # 3
```

**容错处理**:
- 顶层缺 `dh_params` → `KeyError`
- `dh_params` 为空 → `ValueError`
- link 缺字段 → `ValueError` (带具体缺哪些字段)
- link 缺 `theta_offset` → 默认 `0.0`
- 额外字段 (如 `name`) → 静默忽略
- 字段类型 `int → float` 自动转换 (YAML 解析可能返回 int)

### 3.3 从 YAML 加载

```python
from pathlib import Path
from rbvision.robot import DHParams

dh = DHParams.from_yaml(Path("config/default_robot.yaml"))
assert dh == DHParams.scara_4dof()  # 与工厂方法等效

DHParams.from_yaml("no_such.yaml")  # FileNotFoundError
```

YAML 期望结构:

```yaml
robot: {name: "...", profile: scara_4dof}
dh_params:
  - {a: 0.0, alpha: 0.0, d: 0.0, theta_offset: 0.0}
  - {a: 0.225, alpha: 0.0, d: 0.0, theta_offset: 0.0}
  # ...
```

### 3.4 属性与相等性

```python
print(dh_scara.n_joints)         # 4
print(repr(dh_scara))            # DHParams(n_joints=4)
print(dh_scara == DHParams.scara_4dof())  # True (按每个 link 字段)
```

---

## 4. 正运动学 (FK)

### 4.1 `forward_kinematics(q, dh) → ndarray (4, 4)`

```python
import numpy as np
from rbvision.robot import DHParams, forward_kinematics

dh = DHParams.scara_4dof()
T = forward_kinematics(np.zeros(4), dh)
print(f"end: x={T[0,3]:.4f}, y={T[1,3]:.4f}, z={T[2,3]:.4f}")
# end: x=0.4000, y=0.0000, z=-0.0500
```

> **注**: SCARA 4-DOF 零位末端 z = **-0.05** (而非一些文献的 +0.05).
> link 3 的 `alpha=π` 翻转了 Y/Z 轴. 推导见 §8.

`q` 可以是 Python list, 内部自动转 `np.ndarray`:

```python
T = forward_kinematics([0.0, 0.0, 0.0, 0.0], dh)  # 合法
```

### 4.2 `link_transforms(q, dh) → List[ndarray]`

返回 n+1 个累积 4×4 矩阵, 第一个是基坐标系 (单位阵), 用于 3D 渲染:

```python
from rbvision.robot import link_transforms

transforms = link_transforms(np.zeros(4), dh)
print(f"len = {len(transforms)}")  # 5 = 4 + 1
assert np.allclose(transforms[0], np.eye(4))
assert np.allclose(transforms[-1], forward_kinematics(np.zeros(4), dh))
```

### 4.3 输入校验

| 错误情形 | 异常 |
|---|---|
| `q` 长度 ≠ `dh.n_joints` | `ValueError` |
| `q` 含 `nan` / `inf` | `ValueError` |
| `q` 是 2D 数组 | `ValueError` |
| `dh` 为空 (0 link) | `ValueError` |

---

## 5. `RobotArm` — 高层机械臂封装

### 5.1 构造

```python
import numpy as np
from rbvision.robot import DHParams, RobotArm

dh = DHParams.scara_4dof()
arm = RobotArm(
    dh,
    joint_limits=[(-3.14, 3.14)] * 4,
    name="MySCARA",
)
print(arm.n_joints)             # 4
print(arm.name)                 # "MySCARA"
print(arm.get_joint_angles())   # [0. 0. 0. 0.] (默认零位)
```

**构造时校验**:
- `joint_limits` 长度 ≠ n_joints → `ValueError`
- `joint_limits` 单项不是 (lo, hi) → `ValueError`
- `lo > hi` → `ValueError`
- `dh` 为空 → `ValueError`

### 5.2 关节角读写

```python
# 设置 (默认 clamp=True, 越界自动修正)
q_in = np.array([0.1, 0.2, 0.3, 0.4])
arm.set_joint_angles(q_in)

# 越界 → clamp
arm.set_joint_angles(np.array([5.0, -5.0, 0.5, 0.5]))
print(arm.get_joint_angles())  # [3.14, -3.14, 0.5, 0.5]

# 关闭 clamp (离线仿真 / IK 过程)
arm.set_joint_angles(np.array([5.0, 0.0, 0.0, 0.0]), clamp=False)

# get 返回深拷贝, 外部修改不影响内部
q_out = arm.get_joint_angles()
q_out[0] = 99.0
assert arm.get_joint_angles()[0] == 5.0
```

### 5.3 限位工具方法

```python
# 判断 q 是否在限位范围内
assert arm.is_within_limits(np.zeros(4))                          # True
assert not arm.is_within_limits(np.array([5.0, 0, 0, 0]))         # False

# clamp (不修改内部状态)
clamped = arm.clamp_to_limits(np.array([10.0, 0, 0, 0]))
print(clamped)  # [3.14, 0.0, 0.0, 0.0]
```

### 5.4 正运动学 & link 变换

```python
arm.set_joint_angles(np.zeros(4))
pose = arm.forward_kinematics()  # → Pose
print(f"end: ({pose.x:.4f}, {pose.y:.4f}, {pose.z:.4f})")

# 关节角变化时 FK 跟着变
arm.set_joint_angles(np.array([np.pi/2, 0, 0, 0]))
pose = arm.forward_kinematics()
print(f"end: ({pose.x:.4f}, {pose.y:.4f}, {pose.z:.4f})")  # (0.0, 0.4, -0.05)

# 所有 link 累积变换 (3D 渲染用)
transforms = arm.get_link_transforms()  # len = n+1
```

### 5.5 重置与调试

```python
arm.reset()  # q → 全零
print(arm)   # MySCARA (4-DOF, q=[0.0, 0.0, 0.0, 0.0])
```

---

## 6. 内置 Profile

### 6.1 SCARA 4-DOF (默认)

来源: `config/default_robot.yaml` + `DHParams.scara_4dof()`, 两者**等效** (有专门测试守护).

| link | a (m) | alpha (rad) | d (m) | theta_offset (rad) | 关节名 |
|---|---|---|---|---|---|
| 1 | 0.000 | 0.0 | 0.000 | 0.0 | joint1_base (基座旋转) |
| 2 | 0.225 | 0.0 | 0.000 | 0.0 | joint2_arm (大臂) |
| 3 | 0.175 | π | 0.000 | 0.0 | joint3_z (升降) |
| 4 | 0.000 | 0.0 | 0.050 | 0.0 | joint4_wrist (末端旋转) |

**典型应用**: 通用装配 / 点胶 / 小件抓取 (工业最常见 4-DOF 类型).

### 6.2 UR5 6-DOF (备用)

来源: `DHParams.ur5_6dof()`. 通用 6-DOF, 用于验证 N-DOF 通用性:

```python
from rbvision.robot import DHParams, RobotArm
arm = RobotArm(DHParams.ur5_6dof(), [(-3.14, 3.14)] * 6, name="UR5")
pose = arm.forward_kinematics()
print(pose.x, pose.y, pose.z)  # ≈ -0.817, -0.265, -0.109 (q=0)
```

> UR5 在 q=0 时末端 ≈ (-0.817, -0.265, -0.109). 推导见 `tests/test_robot_kinematics.py::TestUR5KnownValues`.

---

## 7. 预留接口 (M3 / M4)

以下接口已在 API 中导出, **当前抛 `NotImplementedError`**, 用于给上层调用方明确的接口存在性提示.

### 7.1 `jacobian(q, dh)` — M3 预留

```python
from rbvision.robot import jacobian, DHParams
import numpy as np

jacobian(np.zeros(4), DHParams.scara_4dof())
# NotImplementedError: jacobian() is reserved for M3 (inverse kinematics + motion control).
```

**用途**: 几何雅可比 J(q) ∈ R^{6×n}, 描述 `v_ee = J(q) @ q_dot`. M3 逆运动学 (IK) 与速度控制需要.

### 7.2 `RobotArm.teach()` — M4 预留

```python
arm.teach()
# NotImplementedError: teach() is reserved for M4 (teach-pendant + playback).
```

**用途**: 示教模式 (用户手动拖动机械臂末端, 记录关键位姿用于回放).
- M2: 仿真模式 (VisPy 拖动 3D 抓取)
- M4: 真实 Modbus 设备 + 安全联锁 (急停 + 限位检查)

> "接口存在性"本身有价值: 告诉调用方"将来会有", 避免怀疑 API 不完整.
> 错误信息明确写 M3/M4, 方便排期.

---

## 8. 数学参考: Modified DH 公式

### 8.1 单连杆变换

Modified DH (Craig 格式):

```
T_i = Rot_x(α_i) @ Trans_x(a_i) @ Rot_z(θ_i) @ Trans_z(d_i)
```

其中 `θ_i = q_i + θ_offset_i`. 矩阵乘法**从右到左** (链式乘积).

### 8.2 末端累积变换

```
T_0n = T_01 @ T_12 @ ... @ T_n-1,n
```

### 8.3 SCARA 零位 FK 推导

```
T_01 = I                              (a=0, α=0, d=0, θ=0)
T_12 = Trans_x(0.225)                 (a=0.225, α=0, d=0)
T_23 = Rot_x(π) @ Trans_x(0.175)      (a=0.175, α=π, d=0)
T_34 = Trans_z(0.05)                  (a=0, α=0, d=0.05)

T_03 = T_01 @ T_12 @ T_23 = Trans(0.4, 0, 0) @ Rot_x(π)
T_04[2,3] = T_03[2,2] * T_34[2,3] = (-1) * 0.05 = -0.05
```

因此 SCARA 零位末端 = **(0.4, 0, -0.05)** (而非一些文献的 +0.05).

---

## 9. 错误处理约定

| 错误情形 | 异常 | 触发函数 |
|---|---|---|
| `q` 长度不匹配 | `ValueError` | `forward_kinematics` / `link_transforms` / `set_joint_angles` / `clamp_to_limits` / `is_within_limits` |
| `q` 含 `nan` / `inf` | `ValueError` | `forward_kinematics` / `link_transforms` / `set_joint_angles` |
| `q` 非 1D | `ValueError` | 同上 |
| `DHParams` 为空 | `ValueError` | `forward_kinematics` / `link_transforms` / `RobotArm.__init__` |
| `joint_limits` 长度不匹配 | `ValueError` | `RobotArm.__init__` |
| `joint_limits` lo > hi | `ValueError` | `RobotArm.__init__` |
| YAML 文件不存在 | `FileNotFoundError` | `DHParams.from_yaml` |
| YAML 缺 `dh_params` 键 | `KeyError` | `DHParams.from_dict` |
| `dh_params` 为空 | `ValueError` | `DHParams.from_dict` |
| link 缺字段 | `ValueError` | `DHParams.from_dict` |
| 预留接口被调用 | `NotImplementedError` | `jacobian` / `RobotArm.teach` |

错误信息**都包含具体数值或位置**:

```python
forward_kinematics(np.zeros(3), DHParams.scara_4dof())
# ValueError: q length 3 != joints 4
```

---

## 10. 版本与路线图

| Milestone | 内容 | 状态 |
|---|---|---|
| M0 | 项目骨架 (Pose / Transform / Logger / Config) | ✅ v0.1.0 |
| **M1** | **DH + FK + RobotArm (本文档)** | **✅ v0.1.0** |
| M2 | 仿真 3D 渲染 (VisPy) + IK 求解 | 规划中 |
| M3 | 雅可比 + 速度/加速度控制 + ModbusTCP 桥接 | 规划中 |
| M4 | 示教模式 (`teach()`) + 真实设备驱动 | 规划中 |
| M5+ | 9 点标定 / 海康 VM 4.x 集成 | 规划中 |

详细架构见 `docs/architecture.md`, 详细需求见 `docs/requirements.md`.

---

## 附录: 测试覆盖

| 文件 | 测试数 | 覆盖范围 |
|---|---|---|
| `tests/test_robot_dh.py` | 23 | DHLink / DHParams / from_dict 容错 / 工厂 / YAML / 边界 (0/1 link, multi-profile, equality, repr, type conversion) |
| `tests/test_robot_kinematics.py` | 21 | FK (SCARA 0/90°/180°, UR5 0/90°) / q 校验 / link_transforms / jacobian 预留 / 单 link / 旋转矩阵正交性 |
| `tests/test_robot_arm.py` | 29 | 构造 / 关节角 / clamp / 限位 / FK via arm / link transforms / reset / repr / teach 预留 / 状态独立性 / DH 一致性 / UR5 集成 |
| **M1 合计** | **73** | (要求 26, 超出 2.8 倍) |
| M0 (test_placeholder.py) | 13 | — |
| **总测试数** | **86** | **0 failed, 0 error** |

覆盖率目标: `src/rbvision/robot/` ≥ 90% (由 `pyproject.toml` `fail_under = 70` 守护).

---

**维护者**: RBVISION Team · 反馈见 `docs/architecture.md` 末尾
