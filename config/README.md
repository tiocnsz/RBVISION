# RBVISION 默认配置

## 配置文件说明

| 文件 | 用途 | 加载时机 |
|---|---|---|
| `default_robot.yaml` | 机械臂 DH 参数 + 关节限位 | 启动时 + 切换机器人 profile |
| `default_calibration.yaml` | 9 点标定参数 (点数、模型) | 启动时 + 进入标定页 |
| `default_gige.yaml` | GigE 模拟相机参数 (分辨率/帧率/相机位姿) | 启动时 + 切换相机视角 |
| `default_modbus.yaml` | Modbus 寄存器映射 | 启动时 (可热重载) |

## 用户自定义

复制 `default_*.yaml` 为 `local_*.yaml`（被 .gitignore 忽略），修改后重启应用。
