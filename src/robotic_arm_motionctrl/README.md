---
AIGC:
    ContentProducer: Minimax Agent AI
    ContentPropagator: Minimax Agent AI
    Label: AIGC
    ProduceID: f007e324fef003cf2a7dabfc963d9918
    PropagateID: f007e324fef003cf2a7dabfc963d9918
    ReservedCode1: 3046022100964ac483c27babf46ceb82e487456006ae954d67321706c54b0d6ba885fe01ab022100982013f2364ed746e5058c07aefe8ef79cfa458467741175e97095ff10831f26
    ReservedCode2: 304502200f14d1bf04a8e1b5683384508dcfbb133d40c79396761549e19865c20df6c6be022100cef9f2245711126c537d02a0585ad90c43d328482e603fd94d3b4fc755447465
---

# 机械臂控制系统

## 项目简介

本项目是一款开源的机械臂控制系统，旨在为教育、科研和爱好者提供一套完整的机械臂控制解决方案。系统采用模块化设计，支持多种主流机械臂硬件平台，提供从基础运动控制到高级轨迹规划的完整功能。

## 功能特性

- **运动控制**：支持关节空间和笛卡尔空间双重控制方式
- **轨迹规划**：提供点对点运动、直线插补、圆弧插补算法
- **运动学求解**：集成正向运动学和逆向运动学计算
- **示教功能**：支持手动示教和轨迹录制回放
- **二次开发**：提供Python API接口，便于扩展应用
- **多平台支持**：兼容Windows、Linux、macOS操作系统

## 硬件支持

| 型号 | 自由度 | 通讯接口 |
|-----|-------|---------|
| MG400 | 4轴 | USB/WiFi |
| M1系列 | 4轴 | USB/蓝牙 |
| uArm Swift | 4轴 | USB |
| 自定义舵机组 | 可配置 | 串口 |

## 快速开始

### 安装依赖

```bash
pip install numpy pyserial
```

### 连接机械臂

```python
from robot_arm import RobotArm

arm = RobotArm(port='COM3')
arm.connect()
```

### 基本操作

```python
# 获取关节位置
positions = arm.get_joint_positions()

# 关节运动
arm.joint_move([45, 30, 60, 90])

# 笛卡尔运动
arm.cartesian_move([200, 100, 150, 0, 0, 0])

# 夹爪控制
arm.gripper_open()
arm.gripper_close()

# 断开连接
arm.disconnect()
```

## 目录结构

```
robot_arm/
├── src/                 # 源代码
│   ├── core/           # 核心控制模块
│   ├── kinematics/     # 运动学算法
│   └── utils/          # 工具函数
├── examples/           # 示例代码
├── config/             # 配置文件
├── tests/              # 测试用例
└── docs/               # 文档
```

## 配置说明

配置文件位于 `config/robot.json`，主要参数包括：

```json
{
    "name": "my_robot",
    "dof": 4,
    "baudrate": 115200,
    "joint_limits": [
        [-135, 135],
        [-45, 90],
        [-90, 90],
        [-180, 180]
    ]
}
```

## 常见问题

**无法连接**：检查串口是否正确，确认驱动已安装
**运动抖动**：降低运动速度，检查关节是否松动
**精度不足**：执行校准程序，检查机械结构

## 安全注意

- 维护前务必断电
- 运动范围内不要放置障碍物
- 异常时立即按急停按钮

