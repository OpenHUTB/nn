
---

# 无人驾驶汽车项目

## 项目简介

本项目是一个基于Python和PyCharm Community Edition开发的无人驾驶汽车仿真系统。通过计算机视觉、机器学习和传感器数据融合技术，实现车辆在模拟环境中的自主导航和避障功能。

## 核心功能

**路径规划**: 基于A*算法和RRT算法的智能路径规划

**障碍物检测**: 使用YOLO和OpenCV进行实时目标检测与识别

**车辆控制**: PID控制器实现精确的转向和速度控制

**传感器模拟**: 模拟激光雷达摄像头和超声波传感器数据

**3D可视化**: 基于PyGame的实时场景渲染

## 技术栈
**开发环境**: PyCharm Community Edition 2024+

**核心语言**: Python 3.8+

**计算机视觉**:  OpenCV, PIL, NumPy

**机器学习**: TensorFlow, scikit-learn

**3D图形**: PyGame, Pyglet.

**数据处理**: Pandas, Matplotlib

## 快速开始

1. 克隆项目到PyCharm
2. 安装依赖: `pip install -r requirements.txt`
3. 运行主程序: `python main.py`
