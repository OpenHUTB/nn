# Collision Warning System

## 项目简介

基于 CARLA 仿真平台和 YOLOv8 的自动驾驶碰撞预警系统，实现前方车辆/行人检测、单目测距、目标跟踪和 TTC（Time-To-Collision）碰撞时间计算，提供实时预警功能。

## 功能特性

- CARLA 仿真环境自动连接
- RGB 摄像头图像实时采集
- YOLOv8 目标检测推理
- 单目视觉测距
- 卡尔曼滤波目标跟踪
- TTC 碰撞时间计算
- 实时碰撞预警显示
- 支持车辆、行人检测与跟踪

## 技术栈

- **仿真平台**: CARLA Simulator 0.9.14
- **目标检测**: YOLOv8 (Ultralytics)
- **目标跟踪**: 卡尔曼滤波
- **编程语言**: Python 3.8+
- **主要依赖**: carla, ultralytics, opencv-python, numpy, scipy

## 项目结构

```
collision_warning_system/
├── README.md
├── requirements.txt
├── config/
│   └── settings.py          # 配置文件
├── scripts/
│   ├── connect_carla.py      # CARLA 连接
│   └── camera_sensor.py      # 摄像头配置
├── models/
│   └── detector.py           # YOLO 检测器
├── tracking/
│   └── kalman_tracker.py     # 卡尔曼跟踪器
├── estimation/
│   └── distance_estimator.py # 单目测距
├── warning/
│   └── ttc_calculator.py     # TTC 计算 + 预警
├── utils/
│   └── visualizer.py         # 可视化工具
├── main.py                   # 主程序
└── test.py                   # 测试脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install scipy
pip install F:\hutb\PythonAPI\carla\dist\carla-0.9.14-cp38-cp38-win_amd64.whl
```

### 2. 启动 CARLA 仿真器

```bash
cd F:\hutb
CarlaUE4.exe
```

等待看到 "Listening to TCP port 2000" 后继续。

### 3. 运行系统

```bash
python main.py
```

## 主要模块

### config/settings.py
- CARLA 连接参数配置
- 摄像头参数设置
- YOLO 模型选择
- 检测类别配置
- TTC 预警阈值设置

### scripts/connect_carla.py
- CARLA 客户端连接
- 世界环境初始化
- 车辆生成

### scripts/camera_sensor.py
- RGB 摄像头配置
- 图像数据采集与处理

### models/detector.py
- YOLOv8 模型加载
- 目标检测推理
- 结果过滤与统计

### tracking/kalman_tracker.py
- 卡尔曼滤波器实现
- 多目标关联
- 轨迹管理

### estimation/distance_estimator.py
- 单目测距算法
- 基于车辆高度/宽度的距离估算
- 相机内参配置

### warning/ttc_calculator.py
- TTC 碰撞时间计算
- 预警等级判定
- 预警逻辑

### utils/visualizer.py
- 检测结果可视化
- 跟踪轨迹绘制
- 距离和 TTC 信息显示

## 支持检测的类别

| Class ID | 类别名称 | 用途 |
|----------|----------|------|
| 0 | person (行人) | 行人碰撞预警 |
| 2 | car (汽车) | 车辆碰撞预警 |
| 3 | motorcycle (摩托车) | 摩托车碰撞预警 |
| 5 | bus (公交车) | 车辆碰撞预警 |
| 7 | truck (卡车) | 车辆碰撞预警 |

## TTC 预警等级

| TTC 范围 | 预警等级 | 颜色 |
|----------|----------|------|
| TTC > 3.0s | 安全 (SAFE) | 绿色 |
| 1.5s < TTC <= 3.0s | 警告 (WARNING) | 黄色 |
| TTC <= 1.5s | 危险 (DANGER) | 红色 |

## 核心算法

### 单目测距
基于已知的真实物体尺寸（车辆高度）和图像中的像素高度计算距离：
```
distance = (real_height * focal_length) / pixel_height
```

### TTC 计算
```
TTC = distance / relative_velocity
```

### 卡尔曼滤波
用于平滑检测结果、预测目标轨迹、关联连续帧目标。

## YOLO 模型选择

可在 `config/settings.py` 中修改 `YOLO_MODEL`：

| 模型 | 速度 | 精度 | 显存需求 |
|------|------|------|----------|
| yolov8n.pt | 最快 | 较低 | ~1GB |
| yolov8s.pt | 快 | 中等 | ~2GB |
| yolov8m.pt | 中等 | 高 | ~4GB |

## 实验结果

（待添加）

## 作者

（待填写）

## 许可证

MIT License
