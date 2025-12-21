# CARLA 目标检测与跟踪系统

这是一个基于CARLA仿真环境的目标检测与跟踪系统，使用YOLOv5模型进行目标检测，并结合改进的SORT算法实现多目标跟踪，同时利用深度信息增强跟踪稳定性和距离估算。

## 功能特点

- 基于CARLA仿真环境的实时目标检测与跟踪
- 使用YOLOv5系列模型进行高精度目标检测
- 改进版SORT跟踪算法，结合深度信息优化跟踪效果
- 动态调整的卡尔曼滤波器，适应不同距离的目标
- 基于深度信息的距离估算与速度计算
- 可视化标注，显示目标类别、ID、距离和速度信息

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

## 安装步骤

1. 安装CARLA模拟器，请参考[官方文档](https://carla.readthedocs.io/en/latest/start_quickstart/)

2. 安装所需Python库：
```bash
pip install carla opencv-python numpy torch ultralytics scipy
```

3. 下载YOLOv5模型文件，并在`load_detection_model`函数中更新模型路径

## 主要组件说明

### 1. 优化版SORT跟踪器

- `KalmanFilter`：卡尔曼滤波器，用于预测目标位置，根据目标距离动态调整过程噪声
- `Track`：单个目标的跟踪信息，包括边界框、中心点、距离、速度等
- `Sort`：多目标跟踪器，使用匈牙利算法进行数据关联，结合深度信息优化IOU阈值

### 2. YOLOv5检测模型

- `load_detection_model`：加载YOLOv5模型，支持多种模型类型，自动选择可用设备（CPU/GPU）

### 3. 工具函数

- `draw_bounding_boxes`：在图像上绘制边界框及相关信息
- `preprocess_depth_image`：预处理深度图像，提高距离估算精度
- `get_target_distance`：根据深度图像和边界框计算目标距离

### 4. CARLA相关函数

- `setup_carla_client`：设置CARLA客户端和世界环境
- `spawn_ego_vehicle`：生成主车辆
- `spawn_npcs`：生成NPC车辆

## 使用方法

1. 启动CARLA服务器：
```bash
./CarlaUE4.sh  # Linux
CarlaUE4.exe   # Windows
```

2. 运行主程序：
```bash
python main.py
```

## 自定义配置

- 可以在`load_detection_model`函数中切换不同的YOLOv5模型
- 在`Sort`类初始化时调整跟踪参数（max_age, min_hits, iou_threshold）
- 在`KalmanFilter`中调整滤波器参数
- 在`draw_bounding_boxes`中修改可视化样式

## 注意事项

- 确保CARLA服务器在运行程序前已启动
- 首次运行可能需要下载YOLOv5模型权重
- 调整CARLA的画质设置可能会影响性能和检测效果
- 深度信息的准确性对跟踪效果有较大影响

## 性能优化

- 对于GPU用户，程序会自动使用半精度计算加速
- 深度图像预处理使用了轻量级高斯模糊，平衡精度和速度
- 跟踪器中使用了`__slots__`减少内存占用

通过调整模型类型和参数，可以在检测精度和运行速度之间取得平衡，适应不同的硬件配置。