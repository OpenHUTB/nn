# Multisensor-fusion-based-3D-Object-Detecting-Simulation-Under-Rainy-Conditions-for-Autonomous-Vehicl
# 基于CARLA与ROS的雨天场景多传感器融合3D目标检测仿真项目

## 📌 项目简介
本项目基于 **CARLA 0.9.12** 仿真平台与ROS实现，用于自动驾驶车辆雨天场景下的多传感器数据采集与3D目标检测仿真。
项目集成了ROS桥接功能，可同步采集摄像头图像与激光雷达点云数据，支持动态/静态雨天环境配置，并提供数据可视化与初步检测功能。

---

## 🛠️ 环境与依赖
### 核心软件版本
- 仿真平台：CARLA 0.9.12 (WindowsNoEditor版本)
- ROS：Melodic/Noetic（与CARLA ROS桥兼容版本）
- Python：3.7+（与CARLA Python API兼容