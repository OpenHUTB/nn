# 无人机手势控制系统 - 改进版

## 概述

本项目已完成三项主要改进：代码架构优化、配置外部化与UI调节、日志系统添加。

## 项目结构

```
drone_hand_gesture/
├── core/                          # 核心模块
│   ├── __init__.py
│   ├── base_controller.py         # 无人机控制器基类
│   ├── config.py                  # 配置管理
│   └── logger.py                  # 日志系统
├── config.json                    # 配置文件（自动生成）
├── logs/                          # 日志目录（自动生成）
├── airsim_controller.py           # AirSim控制器
├── drone_controller.py            # 仿真控制器
├── config_ui.py                   # 配置编辑器UI
├── launcher.py                    # 程序启动器
└── ...
```

## 功能说明

### 1. 代码架构优化

- 创建了 `BaseDroneController` 抽象基类，定义了统一的无人机控制器接口
- `AirSimController` 和 `SimulationDroneController` 继承自基类
- 提取了公共功能：状态管理、轨迹记录、命令处理等

### 2. 配置文件外部化

- 使用 JSON 格式的配置文件 `config.json`
- 支持配置项：无人机参数、摄像头参数、手势识别参数、仿真参数、AirSim参数
- 提供了图形化配置编辑器

### 3. 日志系统

- 支持分级日志（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- 日志同时输出到控制台和文件
- 日志文件按日期时间命名，保存在 `logs/` 目录

## 使用方法

### 启动程序

1. 使用启动器（推荐）：
```bash
python launcher.py
```

2. 直接运行配置编辑器：
```bash
python config_ui.py
```

### 配置项说明

| 分类 | 配置项 | 说明 |
|------|--------|------|
| 无人机 | max_speed | 最大速度 (m/s) |
|  | max_altitude | 最大高度 (m) |
|  | takeoff_altitude | 起飞高度 (m) |
|  | battery_drain_rate | 电池消耗率 |
| 摄像头 | default_id | 摄像头ID |
|  | width/height | 分辨率 |
| 手势识别 | threshold | 置信度阈值 |
|  | command_cooldown | 命令冷却时间 (s) |
| AirSim | ip_address | AirSim服务器地址 |
|  | port | 端口号 |

### 配置文件示例

```json
{
  "drone": {
    "max_speed": 2.0,
    "max_altitude": 10.0,
    "takeoff_altitude": 2.0
  },
  "camera": {
    "default_id": 1,
    "width": 640,
    "height": 480
  },
  "gesture": {
    "threshold": 0.6,
    "command_cooldown": 1.5
  }
}
```

## 代码示例

### 使用配置管理

```python
from core import ConfigManager

config = ConfigManager()

# 获取配置
speed = config.get("drone.max_speed", 2.0)

# 设置配置
config.set("drone.takeoff_altitude", 3.0)

# 保存配置
config.save_config()
```

### 使用日志系统

```python
from core import Logger

logger = Logger()
logger.info("系统启动")
logger.warning("低电量警告")
logger.error("连接失败")
```

### 使用控制器基类

```python
from core import BaseDroneController, ConfigManager
from drone_controller import SimulationDroneController

config = ConfigManager()
controller = SimulationDroneController(config)

controller.connect()
controller.takeoff()
controller.move_by_velocity(1.0, 0, 0)
controller.hover()
controller.land()
controller.disconnect()
```

## 更新内容

### v2.0 改进版
- ✅ 代码架构优化 - 提取公共基类
- ✅ 配置文件外部化，制作UI界面，可调节参数
- ✅ 添加日志系统
