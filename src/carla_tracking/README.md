# CARLA 目标检测与跟踪系统

## 功能特点

- 基于CARLA仿真环境的实时目标检测与跟踪
- 使用YOLOv5系列模型（yolov5s、yolov5su、yolov5m、yolov5mu、yolov5x）进行目标检测
- 改进版SORT跟踪算法，结合深度信息优化跟踪效果
- 动态调整的卡尔曼滤波器，根据目标距离自适应调整参数
- 基于深度图像的目标距离估算（支持中位数和加权平均两种计算方式）
- 实时计算目标速度信息
- 可视化标注，显示目标边界框、ID、距离和速度信息
- 支持NPC车辆自动生成与交通管理

## 依赖环境

- Python 3.8+
- CARLA Simulator 0.9.10+
- 必要的Python库：
  - carla
  - opencv-python
  - numpy
  - torch
  - ultralytics
  - scipy
  - argparse
  - queue

## 主要组件说明

### 1. 目标检测模块
- `load_detection_model`：加载YOLOv5检测模型，自动选择设备（GPU/CPU），支持半精度计算加速
- 支持模型自动 fallback 机制，当指定模型不存在时自动选择可用模型

### 2. 跟踪模块
- `Sort`类：改进版SORT跟踪器，结合深度信息优化跟踪效果
  - 动态IOU阈值：根据目标距离调整匹配阈值
  - 距离加权匹配：近距离目标权重更高
  - 基于距离的跟踪生命周期管理
- `KalmanFilter`类：卡尔曼滤波器，用于目标运动预测
  - 根据目标距离动态调整过程噪声协方差Q
  - 支持位置和速度状态估计

### 3. 深度信息处理
- `get_target_distance`：从深度图像中计算目标距离
  - 支持中位数和加权平均两种距离计算方式
  - 基于目标区域的有效深度提取

### 4. CARLA相关函数
- `setup_carla_client`：设置CARLA客户端和世界环境，配置同步模式
- `spawn_ego_vehicle`：生成主车辆，优先选择林肯MKZ，失败时自动 fallback 到其他车型
- `spawn_npcs`：生成NPC车辆，确保在主车辆周围合理范围内生成
- 交通管理器配置：设置NPC车辆的自动驾驶行为参数

## 运行说明

### 基本命令
```bash
python main.py --model yolov5mu --tracker sort --npc-count 30
```

### 主要参数
- `--model`：选择YOLOv5模型（yolov5s、yolov5su、yolov5m、yolov5mu、yolov5x）
- `--tracker`：选择跟踪器（当前仅支持sort）
- `--host`：CARLA服务器地址（默认localhost）
- `--port`：CARLA服务器端口（默认2000）
- `--conf-thres`：检测置信度阈值（默认0.15）
- `--iou-thres`：IOU阈值（默认0.4）
- `--use-depth`：使用深度信息（默认True）
- `--show-depth`：显示深度图像（默认False）
- `--npc-count`：NPC车辆数量（默认30）

### 操作说明
- 按 'q' 键：退出程序
- 按 'r' 键：重新生成NPC车辆

## 自定义配置

- 在`load_detection_model`函数中修改模型路径或添加新模型
- 在`Sort`类初始化时调整跟踪参数（max_age, min_hits, iou_threshold）
- 在`KalmanFilter`中调整滤波器参数（Q_base, R, dist_thresholds）
- 在`spawn_npcs`函数中调整NPC生成范围和密度
- 在主循环中调整性能监控参数和显示信息

## 性能优化

- 对于GPU用户，自动使用半精度计算加速
- 深度图像预处理优化，平衡精度和速度
- 跟踪算法中使用距离信息动态调整参数，提高跟踪效率
- 性能监控功能，自动识别瓶颈并提供优化建议
- 通过调整模型类型和参数，可以在检测精度和运行速度之间取得平衡

## 性能监控

程序每50帧会输出一次性能统计，包括：
- CARLA同步时间
- 图像和深度获取时间
- 目标检测和跟踪时间
- 结果绘制和显示时间
- 自动识别性能瓶颈并提供优化建议