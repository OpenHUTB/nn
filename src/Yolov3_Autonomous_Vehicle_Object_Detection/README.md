# Autonomous Vehicle Object Detection and Trajectory Planning

## 项目简介
本项目是一个基于自动驾驶场景的毕业设计项目。主要目的是利用 **YOLOv3** 目标检测算法与 **CARLA** 自动驾驶模拟器相结合，实现车辆对周围环境的感知（目标检测）以及基于环境反馈的轨迹规划。

通过模拟器获取实时传感器数据，利用深度学习模型识别路面障碍物（行人、车辆、交通标志等），并根据检测结果调整车辆的行驶轨迹，从而实现基础的自动驾驶避障功能。

## 核心目标
* [x] 搭建项目基础框架与环境依赖
* [ ] 实现 CARLA 模拟器环境的自动连接与车辆生成
* [ ] 集成 YOLOv3 算法进行实时目标检测
* [ ] 实现基于检测结果的简单轨迹规划（如紧急制动、自动避障）
* [ ] 使用 TensorBoard 监控系统推理性能与控制参数

## 技术栈
* **开发语言**: Python 3.7+
* **模拟器**: CARLA Simulator (0.9.11)
* **深度学习框架**: PyTorch / OpenCV
* **目标检测**: YOLOv3 (Darknet weights)
* **可视化**: TensorBoard, Pygame

## 项目目录结构
为了确保代码的可维护性与模块化，本项目采用了以下结构：

```text
Yolov3_Autonomous_Vehicle_Object_Detection/
├── assets/             # 存放项目演示图片、视频及文档
├── config/             # 存放系统配置参数 (如端口、模型阈值等)
│   ├── __init__.py
│   └── config.py       # 全局配置文件
├── models/             # 存放 YOLOv3 权重文件与配置文件
│   └── __init__.py
├── utils/              # 存放通用的图像处理、坐标转换等工具函数
│   └── __init__.py
├── main.py             # 项目主程序入口 (待开发)
├── requirements.txt    # 项目依赖列表
└── README.md           # 项目说明文档