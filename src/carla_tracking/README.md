# Carla 目标跟踪与自动驾驶系统

这是一个基于CARLA模拟器的目标跟踪与自动驾驶系统，集成了YOLOv5目标检测和优化的SORT跟踪算法，能够实现车辆检测、跟踪、距离估计和基本的自动驾驶控制功能。

## 功能特点

- 实时目标检测（使用YOLOv5模型）
- 多目标跟踪（优化的SORT算法）
- 深度感知与距离估计
- 主车辆自动驾驶控制
- NPC车辆生成与管理
- 可配置的系统参数

## 依赖环境

- Python 3.8+
- CARLA Simulator 0.9.13+
- 所需Python库：
  - carla
  - opencv-python
  - numpy
  - torch
  - ultralytics
  - scipy
  - argparse

## 安装指南

1. 安装CARLA模拟器，请参考官方文档：[CARLA Installation Guide](https://carla.readthedocs.io/en/latest/start_quickstart/)

2. 安装Python依赖：
```bash
pip install carla opencv-python numpy torch ultralytics scipy argparse
```

3. 下载YOLOv5模型文件并放置在指定目录（默认为`D:\yolo\`），支持的模型包括：
   - yolov5s.pt
   - yolov5su.pt
   - yolov5m.pt
   - yolov5mu.pt
   - yolov5x.pt

## 配置说明

系统使用配置管理器进行参数设置，支持默认配置和自定义配置文件（JSON格式）。主要配置项包括：

- CARLA连接参数（主机、端口、超时时间等）
- 检测模型参数（置信度阈值、IOU阈值、模型类型等）
- 跟踪算法参数（最大年龄、最小命中数、IOU阈值等）
- 车辆控制参数（最大速度、目标速度、安全距离等）
- NPC车辆参数（数量、最小距离等）
- 相机参数（宽度、高度、FOV等）

可以通过修改配置文件或在代码中调整`ConfigManager`类的默认配置来定制系统行为。

## 主要组件说明

### 1. 配置管理器 (`ConfigManager`)

负责加载、更新和保存系统配置，支持递归合并默认配置和用户自定义配置。

### 2. 优化的卡尔曼滤波器 (`OptimizedKalmanFilter`)

用于预测目标的运动状态，支持带加速度模型和不带加速度模型两种模式，提高跟踪稳定性。

### 3. 跟踪目标类 (`OptimizedTrack`)

维护单个跟踪目标的状态信息，包括边界框、中心点、速度、距离等，并提供相似度计算方法。

### 4. 优化的SORT跟踪器 (`OptimizedSORT`)

实现多目标跟踪功能，使用匈牙利算法进行检测与跟踪目标的匹配，支持自适应阈值和深度感知。

### 5. YOLOv5检测模型 (`load_detection_model`)

加载YOLOv5目标检测模型，支持多种模型类型，并进行模型预热以提高推理速度。

### 6. 车辆控制器 (`VehicleController`)

实现主车辆的自动驾驶控制，使用PID控制器调节速度，并根据检测到的障碍物进行制动决策。

### 7. NPC管理器 (`NPCManager`)

负责生成和管理NPC车辆，设置自动驾驶行为，并支持动态调整NPC的行驶参数。

### 8. 工具函数

提供深度图像预处理、目标距离计算、边界框绘制等辅助功能。

## 使用方法

1. 启动CARLA模拟器：
```bash
./CarlaUE4.sh  # Linux
CarlaUE4.exe   # Windows
```

2. 运行主程序：
```bash
python main.py [--config config.json]
```

其中`--config`参数可选，用于指定自定义配置文件。

## 扩展与定制

- 可以通过修改`VehicleController`类来实现更复杂的自动驾驶逻辑
- 支持更换不同的YOLOv5模型以平衡检测精度和速度
- 可调整跟踪算法参数以适应不同的场景需求
- 可扩展支持更多类型的传感器数据处理

## 注意事项

- 确保CARLA模拟器与Python API版本匹配
- 首次运行时需要下载YOLOv5模型文件
- 使用GPU可以显著提高检测和跟踪性能
- 调整配置参数时需注意各参数间的兼容性