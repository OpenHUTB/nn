# CARLA YOLO Object Detection

## 项目简介

基于 CARLA 仿真平台和 YOLOv8 的实时目标检测系统，用于自动驾驶场景下的车辆、行人、交通灯等目标识别与可视化。

## 功能特性

- CARLA 仿真环境自动连接
- RGB 摄像头图像实时采集
- YOLOv8 目标检测推理
- 多目标可视化显示（bounding box + 标签）
- 支持车辆、行人、交通灯、交通标志检测

## 技术栈

- **仿真平台**: CARLA Simulator 0.9.14
- **目标检测**: YOLOv8 (Ultralytics)
- **编程语言**: Python 3.8+
- **主要依赖**: carla, ultralytics, opencv-python, numpy

## 项目结构

```
carla_yolo_detection/
├── README.md
├── requirements.txt
├── config/
│   └── settings.py          # 配置文件
├── scripts/
│   ├── connect_carla.py     # CARLA 连接
│   └── camera_sensor.py     # 摄像头配置
├── models/
│   └── detector.py          # YOLO 检测器
├── utils/
│   └── visualizer.py        # 可视化工具
├── data/
│   └── collect.py           # 数据采集
├── main.py                  # 主程序
└── test.py                   # 测试脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install carla==0.9.14
pip install ultralytics
pip install opencv-python
pip install numpy
```

### 2. 启动 CARLA 仿真器

```bash
cd <CARLA_PATH>
./CarlaUE4.sh
```

等待看到 "Listening to TCP port 2000" 后继续。

### 3. 运行检测

```bash
python main.py
```

## 主要模块

### config/settings.py
- CARLA 连接参数配置
- 摄像头参数设置
- YOLO 模型选择
- 检测类别配置

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

### utils/visualizer.py
- 检测结果可视化
- Bounding Box 绘制
- 实时显示

## 支持检测的类别

| Class ID | 类别名称 |
|----------|----------|
| 0 | person (行人) |
| 2 | car (汽车) |
| 3 | motorcycle (摩托车) |
| 5 | bus (公交车) |
| 7 | truck (卡车) |
| 9 | traffic light (交通灯) |
| 11 | stop sign (停车标志) |

## YOLO 模型选择

可在 `config/settings.py` 中修改 `YOLO_MODEL`：

| 模型 | 速度 | 精度 | 显存需求 |
|------|------|------|----------|
| yolov8n.pt | 最快 | 较低 | ~1GB |
| yolov8s.pt | 快 | 中等 | ~2GB |
| yolov8m.pt | 中等 | 高 | ~4GB |
| yolov8l.pt | 慢 | 很高 | ~8GB |


