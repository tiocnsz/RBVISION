# RBVISION

> 面向海康 VM 4.x 的工业手眼标定 + 机械臂位置反馈模拟平台
> GigE Vision 模拟相机 + 通用机械臂执行器 + 9 点标定 + ModbusTCP 桥接

## 项目状态

🚧 **M0 - 项目骨架 (进行中)**

详细规划见 [`docs/requirements.md`](docs/requirements.md) 和 [`docs/architecture.md`](docs/architecture.md)。

## 快速开始

### 环境要求
- Python 3.10+
- Anaconda / Miniconda
- Windows 10/11 x64 (主要平台)

### 安装

```bash
# 创建 conda 环境
conda create -n rbvision python=3.11 -y
conda activate rbvision

# 安装依赖
pip install -r requirements.txt
```

### 验证

```bash
python -c "import PySide6, vispy, cv2, pymodbus, loguru, yaml; print('✓ 所有依赖 OK')"
```

### 运行 (M0 完成后)

```bash
python scripts/run_app.py       # 启动 GUI
python scripts/run_headless.py  # 无头模式 (CI 用)
```

### 测试

```bash
pytest tests/ -v                # 跑全部测试
pytest tests/ --cov=rbvision    # 带覆盖率
```

## 项目结构

```
RBVISION/
├── docs/                # 需求、架构、协议文档
├── config/              # YAML 配置 (机械臂、Modbus、标定、GigE)
├── src/rbvision/        # 源代码
│   ├── core/            # 基础数据结构
│   ├── robot/           # 机械臂 (FK + DH)
│   ├── calibration/     # 9 点标定
│   ├── comm/            # ModbusTCP
│   ├── gige/            # GigE Vision 模拟
│   ├── sim/             # 3D 仿真场景
│   ├── service/         # 业务编排层
│   ├── ui/              # PySide6 界面
│   └── utils/           # 工具
├── tests/               # 单元测试
├── scripts/             # 启动脚本
└── requirements.txt     # 依赖清单
```

## 里程碑

- [x] M0 - 项目骨架
- [ ] M1 - 机械臂核心 (FK)
- [ ] M2 - 9 点标定算法
- [ ] M3 - ModbusTCP 双向
- [ ] M4 - 3D 渲染 + 示教 GUI
- [ ] M5 - GigE Vision 模拟 (VM 4.x 兼容)
- [ ] M6 - GUI 整合 + 误差展示
- [ ] M7 - 联调 + 文档 + 打包

## 协议

- **ModbusTCP**: 端口 502 (可配), 字节序 Big Endian
- **GigE Vision**: GVCP Discovery (UDP 3956) + GVSP 数据流 (UDP 3957)
- **3D 仿真**: PySide6 + VisPy, 支持 eye-to-hand / eye-in-hand 多种视角

## 许可

TBD
