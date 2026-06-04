# RBVISION 架构设计文档

> 版本: v0.3 · 日期: 2026-06-04 · 配套 `requirements.md` v0.3

---

## 1. 架构总览

### 1.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                       UI Layer (PySide6)                      │
│  MainWindow · 3DView · CalibrationPage · TeachingPage · ...   │
├─────────────────────────────────────────────────────────────┤
│                  Application / Service Layer                 │
│  AppContext (依赖注入) · SimController · CalibController ·   │
│  ModbusBridge · GigeController · TeachingController          │
├─────────────────────────────────────────────────────────────┤
│                      Domain Layer                            │
│  RobotArm · Calibration · Pose · Transform · Trajectory     │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                      │
│  ModbusServer (pymodbus) · GigeGVCP/GVSP · VisPyRenderer ·  │
│  ConfigStore (YAML) · Logger (loguru)                        │
├─────────────────────────────────────────────────────────────┤
│                       Platform / OS                          │
│  Windows 10/11 · Python 3.10+ · Qt 6.5+ · OpenGL · NIC      │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 关键设计原则

| 原则 | 体现 |
|---|---|
| **依赖倒置** | UI 持有 `AppContext`,不直接 import 底层;Service 通过接口解耦 |
| **单数据源** | `RobotState` (机械臂实时状态) 是唯一权威,各模块订阅 |
| **线程隔离** | UI 线程、Modbus 轮询线程、GigE 推流线程、3D 渲染线程 互相通过 Queue/Signal 通信 |
| **配置外置** | 所有可调参数(端口、地址、DH 参数)走 YAML,运行时可重载 |
| **可测性** | Domain 层无 Qt 依赖,纯 Python + NumPy,可独立单测 |

---

## 2. 模块设计

### 2.1 模块清单

```
src/rbvision/
├── core/                  # 基础数据结构 (无依赖,纯 NumPy)
├── robot/                 # 机械臂 (FK + DH)
├── calibration/           # 9 点标定
├── comm/                  # ModbusTCP
├── gige/                  # GigE Vision
├── sim/                   # 3D 场景 + 截图
├── ui/                    # PySide6 界面
├── service/               # ⭐ 业务编排层 (新增)
│   ├── app_context.py     # 全局上下文/依赖注入
│   ├── sim_controller.py  # 仿真主控
│   ├── calib_controller.py # 标定流程编排
│   ├── teaching_controller.py # 示教模式编排
│   └── modbus_bridge.py   # Modbus ↔ Domain 适配
└── utils/                 # 配置/日志/工具
```

### 2.2 关键模块详解

#### 2.2.1 `core/pose.py` - 位姿

```python
@dataclass
class Pose:
    """位姿: 位置 + 姿态(支持四元数和欧拉)"""
    x: float  # m
    y: float  # m
    z: float  # m
    qx: float = 0.0  # 四元数
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    @classmethod
    def from_euler(cls, x, y, z, rx, ry, rz, order='xyz') -> 'Pose': ...

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> 'Pose': ...

    def to_matrix(self) -> np.ndarray: ...  # 4x4

    def to_euler(self, order='xyz') -> Tuple[float, float, float]: ...
```

#### 2.2.2 `core/transform.py` - 4x4 变换矩阵

```python
def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray: ...
def invert_T(T: np.ndarray) -> np.ndarray: ...
def compose_T(*Ts: np.ndarray) -> np.ndarray: ...
def pose_to_T(pose: Pose) -> np.ndarray: ...
def T_to_pose(T: np.ndarray) -> Pose: ...
def rot_x(theta), rot_y(theta), rot_z(theta): ...  # 基础旋转
def trans(x, y, z): ...
```

#### 2.2.3 `robot/arm.py` - 通用 6-DOF 机械臂

```python
class RobotArm:
    """通用 6-DOF 机械臂,基于 DH 参数"""
    def __init__(self, dh_params: DHParams, joint_limits: List[Tuple[float, float]]):
        ...

    def set_joint_angles(self, q: np.ndarray) -> None:
        """设置关节角 (rad)"""
        ...

    def get_joint_angles(self) -> np.ndarray:
        ...

    def forward_kinematics(self) -> Pose:
        """正运动学: 关节角 → 末端位姿"""
        ...

    def get_link_transforms(self) -> List[np.ndarray]:
        """返回所有 link 的 4x4 变换,用于 3D 渲染"""
        ...

    def teach_step(self, axis: str, delta: float) -> None:
        """示教: 沿指定轴(X+/X-/Y+/...)移动末端(简化版)"""
        # 注:严格笛卡尔示教需要 IK,本项目只做 FK,所以示教用:
        # 1. 关节示教 (Joint +/-) - 直接改 q[i]
        # 2. 简化笛卡尔示教 - 用 DH Jacobian 近似
        ...
```

**DH 参数** (默认 UR5 风格, 6-DOF):

```yaml
# config/default_robot.yaml
dh_params:
  links:
    - {a: 0.0,    alpha: 1.5708, d: 0.089,  theta_offset: 0.0}    # base → shoulder
    - {a: -0.425, alpha: 0.0,    d: 0.0,   theta_offset: 0.0}    # shoulder → upper_arm
    - {a: -0.392, alpha: 0.0,    d: 0.0,   theta_offset: 0.0}    # upper_arm → forearm
    - {a: 0.0,    alpha: 1.5708, d: 0.109, theta_offset: 0.0}    # forearm → wrist1
    - {a: 0.0,    alpha: -1.5708,d: 0.094, theta_offset: 0.0}    # wrist1 → wrist2
    - {a: 0.0,    alpha: 0.0,    d: 0.082, theta_offset: 0.0}    # wrist2 → end
joint_limits:
  - [-2*np.pi, 2*np.pi]
  - [-2*np.pi, 2*np.pi]
  - [-2*np.pi, 2*np.pi]
  - [-2*np.pi, 2*np.pi]
  - [-2*np.pi, 2*np.pi]
  - [-2*np.pi, 2*np.pi]
```

#### 2.2.4 `calibration/nine_point.py` - 9 点标定 ⭐

```python
class NinePointCalibration:
    """
    9 点 / N 点标定 (图像坐标 → 机械 XY 坐标,2D 仿射)

    标定流程:
      1. 用户示教: 控制机械臂末端到 N 个 mark 点,记录 (X_i, Y_i)
      2. VM 识别: 通过 Modbus 接收 9 个 (u_i, v_i)
      3. 解算: 最小二乘求 3x3 仿射矩阵 M
      4. 验证: 重投影误差
    """
    def __init__(self, n_points: int = 9):
        self.n = n_points
        self.robot_points: List[Tuple[float, float]] = []  # (X, Y)
        self.image_points: List[Tuple[int, int]] = []     # (u, v)
        self.matrix: Optional[np.ndarray] = None  # 3x3 仿射矩阵

    def add_robot_point(self, x: float, y: float) -> None: ...
    def add_image_point(self, u: int, v: int) -> None: ...

    def solve(self) -> np.ndarray:
        """
        解算 3x3 仿射矩阵
        M @ [u, v, 1]^T = [X, Y, 1]^T
        最小二乘解: A x = b
        """
        assert len(self.robot_points) == self.image_points == self.n
        # 构造方程组: X = a*u + b*v + c, Y = d*u + e*v + f
        A = []
        bx, by = [], []
        for (u, v), (X, Y) in zip(self.image_points, self.robot_points):
            A.append([u, v, 1, 0, 0, 0])
            A.append([0, 0, 0, u, v, 1])
            bx.append(X)
            bx.append(Y)
        params, *_ = np.linalg.lstsq(A, bx, rcond=None)
        a, b, c, d, e, f = params
        M = np.array([
            [a, b, c],
            [d, e, f],
            [0, 0, 1],
        ])
        self.matrix = M
        return M

    def transform(self, u: int, v: int) -> Tuple[float, float]:
        """单点转换: (u, v) → (X, Y)"""
        if self.matrix is None:
            raise ValueError("Matrix not solved yet")
        p = self.matrix @ np.array([u, v, 1])
        return p[0], p[1]

    def evaluate(self) -> Dict[str, float]:
        """评估: 每个点的重投影误差 + RMS"""
        errors = []
        for (u, v), (X_true, Y_true) in zip(self.image_points, self.robot_points):
            X_pred, Y_pred = self.transform(u, v)
            err = np.sqrt((X_pred - X_true)**2 + (Y_pred - Y_true)**2)
            errors.append(err)
        return {
            'per_point': errors,
            'rms': float(np.sqrt(np.mean(np.array(errors)**2))),
            'max': float(max(errors)),
            'mean': float(np.mean(errors)),
        }

    def save(self, path: str) -> None: ...  # YAML
    def load(self, path: str) -> None: ...
    def to_modbus_registers(self) -> List[int]: ...  # 用于下发给 VM
```

**注:** 严格仿射矩阵也支持旋转 + 缩放 + 剪切 + 平移;若需要更严格的"相似变换"(仅旋转+缩放+平移),可改用 `cv2.estimateAffinePartial2D`。

#### 2.2.5 `comm/modbus_server.py` - ModbusTCP Server

```python
class ModbusServer:
    """
    pymodbus 包装,提供 4 段地址:
      0-99   标定数据
      100-199 运动指令
      200-299 位置反馈
      300-399 状态/控制
      400-499 实时坐标转换服务
    """
    def __init__(self, host='0.0.0.0', port=502, unit_id=1):
        self.server = None
        self.holding_registers = [0] * 500  # 500 寄存器
        # 订阅: VM 写入标定图像坐标时,触发 calibration.update()
        # 发布: 机械臂实际位姿变化时,更新 200-299 段

    def start(self): ...  # 启动后台线程
    def stop(self): ...

    def write_registers(self, address: int, values: List[int]) -> None: ...
    def read_registers(self, address: int, count: int) -> List[int]: ...

    def register_callback(self, address_range: Tuple[int, int], callback: Callable): ...
        # 当 VM 写入指定地址段时,调用 callback
```

**关键回调**:
- 监听 `20-37`(VM 写入 9 点图像坐标)→ 触发 `calibration.update_image_points()`
- 监听 `400-401`(VM 写入待转换 u)→ 触发 `calibration.transform()` → 写回 `404-407`

#### 2.2.6 `gige/simulator.py` - GigE Vision 模拟 ⭐

```python
class GigeSimulator:
    """
    GigE Vision 模拟器
      - GVCP Discovery 响应 (UDP 3956)
      - GVSP 数据流推流 (UDP 3957)
    """
    def __init__(self, image_source: ImageSource, config: GigeConfig):
        self.image_source = image_source  # 3D 场景 / 视频 / 图序
        self.config = config
        self.gvcp = GvcpServer(...)
        self.gvsp = GvspSender(...)

    def start(self): ...
    def stop(self): ...

    def set_resolution(self, w: int, h: int): ...
    def set_pixel_format(self, fmt: str): ...  # 'Mono8' / 'RGB8' / 'BayerRG8'
    def set_fps(self, fps: int): ...
    def set_trigger_mode(self, mode: str): ...  # 'continuous' / 'soft' / 'external'

class GvcpServer:
    """GVCP 控制通道 (Discovery + Read/Write Cmd)"""
    def handle_discovery(self, packet: bytes) -> bytes:
        # 构造 DISCOVERY_ACK
        ...
    def handle_read_register(self, packet: bytes) -> bytes: ...
    def handle_write_register(self, packet: bytes) -> bytes: ...

class GvspSender:
    """GVSP 数据通道 (推流)"""
    def push_frame(self, frame: np.ndarray) -> None:
        # 按 GigE Vision 1.2 块结构打包,UDP 发送
        ...
```

**图像源** (3 种, 可插拔):

```python
class ImageSource(ABC):
    @abstractmethod
    def get_frame(self) -> np.ndarray: ...

class SceneRenderSource(ImageSource):
    """从 VisPy 3D 场景截图"""
    def __init__(self, scene: Scene3D):
        self.scene = scene
    def get_frame(self) -> np.ndarray:
        return self.scene.render_to_image(width=1920, height=1080)

class VideoFileSource(ImageSource):
    def __init__(self, path: str, loop: bool = True):
        self.cap = cv2.VideoCapture(path)
    def get_frame(self) -> np.ndarray: ...

class ImageSequenceSource(ImageSource):
    def __init__(self, pattern: str, start: int = 0):
        # pattern: '/path/frame_{:04d}.png'
        ...
    def get_frame(self) -> np.ndarray: ...
```

#### 2.2.7 `sim/scene.py` - 3D 场景

```python
class Scene3D:
    """3D 仿真场景"""
    def __init__(self):
        self.arm: Optional[RobotArm] = None
        self.calibration_board: Optional[CalibrationBoard] = None
        self.target_object: Optional[TargetObject] = None
        self.trajectory: List[Pose] = []
        self.ground = Ground()

    def set_arm(self, arm: RobotArm): ...
    def update_arm_angles(self, q: np.ndarray): ...  # 触发 link transforms 重算
    def add_trajectory_point(self, pose: Pose): ...
    def render_to_image(self, w: int, h: int) -> np.ndarray: ...
    def render_to_widget(self, widget: QWidget): ...  # 嵌入 Qt
```

**对象**:
- `Ground` - 地面网格
- `CalibrationBoard` - 9 点标定板(可在 GUI 中拖动)
- `TargetObject` - 待抓取目标(可选)
- `Trajectory` - 末端历史轨迹(线段渲染)

#### 2.2.8 `service/sim_controller.py` - 仿真主控 ⭐

```python
class SimController:
    """
    仿真主控: 串起机械臂 / 3D 渲染 / Modbus
    """
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.arm: RobotArm = ...
        self.scene: Scene3D = ...
        self.modbus: ModbusServer = ...
        self.running = False

    def start(self):
        """启动仿真循环"""
        # 1. 启动 Modbus
        # 2. 启动 3D 渲染线程
        # 3. 主循环: 读 Modbus → 更新 arm → 触发 3D 渲染 → 写 Modbus 反馈
        ...

    def on_target_pose_update(self, pose: Pose):
        """Modbus 收到新目标位姿"""
        # 简单 FK: 直接更新末端位姿不实际
        # 这里用简化: 给定目标位姿后,用雅可比近似反解(简化 IK)更新关节角
        # 或者: 关节空间直接插值过渡
        ...

    def get_actual_pose(self) -> Pose:
        """返回当前末端位姿 (写回 Modbus 200-299)"""
        return self.arm.forward_kinematics()

    def get_actual_joints(self) -> np.ndarray:
        return self.arm.get_joint_angles()
```

**简化版笛卡尔示教** (无 IK):

```python
# 示教: 用户点 "X+" 按钮
# 方案 A (关节空间): q[0] += delta  (简单)
# 方案 B (雅可比近似):
#   dX = J^+ * dq  →  给定 dX, 反解 dq ≈ J^T * dX
#   其中 J 是当前雅可比矩阵
#   本项目用方案 B,精度不高但够用
```

#### 2.2.9 `service/calib_controller.py` - 标定流程编排 ⭐

```python
class CalibController:
    """
    标定流程编排:
      Step 1: 提示用户放置标定板
      Step 2: 示教模式启动,引导用户依次示教 9 个点
      Step 3: 通知 VM 准备识别 (Modbus 写 CalStatus = 2)
      Step 4: 等待 VM 写入 9 个 (u, v) (Modbus 监听 20-37)
      Step 5: 触发 calibration.solve()
      Step 6: 评估误差,显示
      Step 7: 写 CalStatus = 4 (完成) 或 5 (失败)
    """
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.calib: NinePointCalibration = NinePointCalibration(n_points=9)
        self.state: CalibState = CalibState.IDLE

    def start_teaching(self): ...  # Step 2
    def record_point(self, point_idx: int): ...  # 记录第 N 个示教点
    def on_image_points_received(self, points: List[Tuple[int, int]]): ...  # Step 4
    def solve(self) -> Dict: ...  # Step 5+6
    def abort(self): ...
```

### 2.3 状态机

#### 标定状态机

```
IDLE
  └─ start_teaching() ──▶ TEACHING
                              └─ record_point(0..8) ── (9 次) ──▶ WAITING_IMAGE
                                                                       └─ on_image_points_received() ──▶ SOLVING
                                                                                                            └─ solve() ──▶ DONE
                                                                                                                          └─ start_teaching() ──▶ TEACHING (重新标定)
```

#### 仿真运行状态机

```
IDLE
  └─ start() ──▶ RUNNING
                    └─ stop() ──▶ IDLE
                    └─ error ──▶ ERROR ──▶ reset() ──▶ IDLE
```

---

## 3. 线程模型

```
┌─────────────────────────────────────────────────────────────┐
│ UI 线程 (Main Thread / Qt)                                    │
│   · Qt 事件循环                                               │
│   · 3D 渲染 (VisPy QOpenGLWidget 嵌入)                        │
│   · 处理按钮点击 / GUI 更新                                    │
└────────────────────────┬────────────────────────────────────┘
                         │ Qt Signal/Slot
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 仿真线程 (QThread / SimController)                            │
│   · 主循环: 50Hz                                              │
│   · 读 Modbus 寄存器 → 检查变化 → 更新 RobotArm               │
│   · RobotArm.forward_kinematics()                            │
│   · 写回 Modbus 反馈寄存器                                     │
│   · 通过 Qt Signal 通知 UI 更新 3D 视口                        │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Modbus Thread│  │ GigE Thread  │  │ 标定 Thread  │
│ (pymodbus)   │  │ (GVCP+GVSP)  │  │ (calib ctrl) │
│              │  │              │  │              │
│ · 监听连接   │  │ · UDP 3956   │  │ · 监听 Modbus│
│ · 读写寄存   │  │ · UDP 3957   │  │   20-37 段   │
│ · 触发回调   │  │ · 25 FPS 推流│  │ · 解算矩阵   │
└──────────────┘  └──────────────┘  └──────────────┘
```

**关键设计**:
- UI 线程不直接做 I/O,所有 Modbus / GigE 阻塞操作在子线程
- 子线程通过 Qt Signal 通知 UI 更新 (跨线程安全)
- 共享数据 (RobotState) 走 `QReadWriteLock` 或 `threading.RLock`

---

## 4. 数据流

### 4.1 标定阶段数据流

```
┌──────┐   record_point()   ┌──────────────────┐
│ User │ ─────────────────▶ │ CalibController  │
└──────┘                    │  .calib          │
                            │  .robot_points   │
                            └─────────┬────────┘
                                      │
                                      │ on_image_points_received()
                                      ▼
┌──────┐   write 20-37      ┌──────────────────┐
│  VM  │ ─────────────────▶ │ ModbusServer     │
└──────┘                    │  .on_register_   │
                            │   write(20-37)   │
                            └─────────┬────────┘
                                      │ callback
                                      ▼
                            ┌──────────────────┐
                            │ CalibController  │
                            │  .on_image_      │
                            │   points_        │
                            │   received()     │
                            └─────────┬────────┘
                                      │ solve()
                                      ▼
                            ┌──────────────────┐
                            │ NinePointCalib    │
                            │  .solve()        │
                            │  .evaluate()     │
                            └─────────┬────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │ ModbusServer     │
                            │  .write 38-40    │ (status + error)
                            └──────────────────┘
```

### 4.2 运行阶段数据流

```
┌──────┐  write 100-111     ┌──────────────────┐
│  VM  │ ─────────────────▶ │ ModbusServer     │
└──────┘  (Target_X..RZ)    │  .on_register_   │
                            │   write(100-111) │
                            └─────────┬────────┘
                                      │ callback
                                      ▼
                            ┌──────────────────┐
                            │ SimController    │
                            │  .on_target_pose │
                            │   _update()      │
                            └─────────┬────────┘
                                      │ update arm (interpolated)
                                      ▼
                            ┌──────────────────┐
                            │ RobotArm         │
                            │  .set_joints()   │
                            │  .fk()           │
                            └─────────┬────────┘
                                      │ pose
                                      ▼
                            ┌──────────────────┐
                            │ ModbusServer     │
                            │  .write 200-217  │ (Actual + joints)
                            └──────────────────┘
```

### 4.3 实时坐标转换数据流

```
┌──────┐  write 400-401     ┌──────────────────┐
│  VM  │ ─────────────────▶ │ ModbusServer     │
└──────┘  (Query_U, V)      │  .on_register_   │
                            │   write(400-401) │
                            └─────────┬────────┘
                                      │ callback
                                      ▼
                            ┌──────────────────┐
                            │ CalibController  │
                            │  .on_query()     │
                            │  .calib.transform(u,v) │
                            └─────────┬────────┘
                                      │ X, Y
                                      ▼
                            ┌──────────────────┐
                            │ ModbusServer     │
                            │  .write 404-408  │ (Result + status)
                            └──────────────────┘
```

---

## 5. 关键接口 (跨模块契约)

### 5.1 RobotState (全局状态)

```python
@dataclass
class RobotState:
    joint_angles: np.ndarray           # 6 个关节角 (rad)
    end_pose: Pose                     # 末端位姿
    status: RobotStatus                # IDLE / MOVING / REACHED / ERROR
    last_update: datetime
```

发布: `SimController` 每次更新后调用 `ctx.state_bus.publish(state)`
订阅: UI / Modbus 反馈 / GigE 推流 等模块订阅

### 5.2 CalibState (标定状态)

```python
class CalibState(IntEnum):
    IDLE = 0
    TEACHING = 1
    WAITING_IMAGE = 2
    SOLVING = 3
    DONE = 4
    FAILED = 5
```

### 5.3 GigeFramePacket

```python
@dataclass
class GigeFramePacket:
    width: int
    height: int
    pixel_format: str  # 'Mono8' / 'RGB8' / 'BayerRG8'
    data: np.ndarray   # H x W or H x W x 3
    timestamp: int     # 纳秒
    block_id: int      # GVSP 块 ID
```

---

## 6. 部署结构

### 6.1 开发模式

```
开发者机器 (Windows 10/11)
  ├── Python 3.10 venv
  ├── rbvision/ (源码)
  ├── config/ (YAML)
  └── python scripts/run_app.py
```

### 6.2 生产模式 (PyInstaller 打包)

```
RBVISION-1.0.0/
├── RBVISION.exe          # 主程序
├── _internal/            # 依赖库
│   ├── python310.dll
│   ├── ...
│   └── rbvision/         # 编译后的模块
├── config/               # 用户可改配置
│   ├── default_robot.yaml
│   └── ...
├── logs/                 # 自动生成
└── README.txt
```

### 6.3 文件持久化

```
%APPDATA%/RBVISION/  (Windows) 或 ~/.config/rbvision/ (Linux)
  ├── calibration/        # 标定文件
  │   └── calib_20260604.yaml
  ├── logs/
  │   └── rbvision_20260604.log
  ├── recordings/         # 轨迹回放数据
  │   └── trail_20260604_1130.csv
  └── settings.yaml       # 全局设置
```

---

## 7. 错误处理与日志

### 7.1 错误分类

| 类别 | 处理 |
|---|---|
| Modbus 断线 | 自动重连 (指数退避),UI 显示 "连接中..." |
| GigE 推流异常 | 记录日志,尝试重置 socket,失败则禁用推流 |
| 标定数据不完整 | CalStatus = FAILED,UI 弹窗提示 |
| 关节超限 | 自动 clamp,日志告警 |
| 配置缺失 | 启动失败,提示用户检查 config/ |

### 7.2 日志策略

- 级别: DEBUG / INFO / WARNING / ERROR
- 输出: 控制台 + 文件 (`%APPDATA%/RBVISION/logs/rbvision_YYYYMMDD.log`)
- 轮转: 按天切分,保留 7 天
- 关键事件埋点: 标定完成、Modbus 连接、GigE 推流开始/结束、错误码

---

## 8. 测试策略

### 8.1 测试金字塔

```
              E2E 测试 (少量)
            ┌────────────────┐
           │  端到端: 模拟 VM → 触发标定 → 验证矩阵 │
            └────────────────┘
         集成测试 (中等)
       ┌────────────────────────────┐
       │  Modbus Server/Client 双向  │
       │  GigE 推流 + 接收端验证      │
       │  标定 + 机械臂 + 渲染         │
       └────────────────────────────┘
    单元测试 (大量)
  ┌──────────────────────────────────────┐
  │  Pose / Transform 运算               │
  │  FK (已知 DH 参数对比)                │
  │  9 点标定解算 (合成数据)              │
  │  寄存器编解码                         │
  │  GVCP Discovery 响应                 │
  └──────────────────────────────────────┘
```

### 8.2 关键测试用例

| 编号 | 描述 |
|---|---|
| TC-001 | FK 在 q=[0]*6 时,末端位姿符合 UR5 文档值 |
| TC-002 | 9 点标定在合成数据(已知矩阵)上误差 < 0.001mm |
| TC-003 | Modbus Server 接收 VM 写入 100 段,触发回调 |
| TC-004 | Modbus 断线 → 重连 → 数据恢复 |
| TC-005 | GVCP Discovery 响应包字段正确(厂商名/型号/IP) |
| TC-006 | GVSP 推流 100 帧,无丢包(用本地监听器) |
| TC-007 | 完整标定流程: 示教 → 模拟 VM 写入 → 解算 → 验证 |

---

## 9. 性能预算

| 指标 | 目标 | 备注 |
|---|---|---|
| 3D 渲染帧率 | ≥ 30 FPS | VisPy 在 1920x1080 应轻松达标 |
| Modbus 轮询周期 | ≤ 20ms (50Hz) | pymodbus 异步 |
| GigE 推流帧率 | 25 FPS (1080p Mono8) | UDP 网络需 ≥ 50Mbps |
| FK 单次耗时 | < 0.1ms | 6x4x4 矩阵乘法 |
| 标定解算 (9 点) | < 1ms | 6x6 最小二乘 |
| 启动时间 | < 5s | PySide6 + VisPy 冷启动 |
| 内存占用 | < 500MB | 渲染 + Modbus + 日志 |

---

## 10. 未来扩展 (Out of Scope but Designed For)

- **多相机/多机械臂**: Scene 改为 List[Scene3D];Modbus 单元 ID 区分
- **OPC UA / Profinet**: comm/ 目录下新增 client 实现,Service 层抽象 Bus 接口
- **云端日志**: 增加 REST 上传模块
- **AI 标定辅助**: 用 OpenCV 的 findCirclesGrid / charuco_board 提高识别鲁棒性(VM 端用)
- **3D 仿真升级**: 切换到 Isaac Sim / MuJoCo 做物理仿真(替换 VisPy)

---

## 11. 文档结构

```
docs/
├── requirements.md        # 需求规格 (本规格)
├── architecture.md        # 架构设计 (本文件)
├── modbus_register.md     # Modbus 寄存器详细表 (v0.4 补充)
├── gige_protocol.md       # GigE 协议实现细节 (v0.5 补充)
├── calibration_guide.md   # 9 点标定操作手册 (M2 完成后写)
└── user_manual.md         # 用户手册 (M6 完成后写)
```
