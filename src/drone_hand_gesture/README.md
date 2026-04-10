# 手势控制无人机项目

基于计算机视觉的手势识别无人机控制系统，支持本地仿真和 AirSim 真实模拟器。

## 功能特性

- ✅ 实时手势识别（MediaPipe）
- ✅ 本地 3D 仿真模式
- ✅ AirSim 真实模拟器集成
- ✅ 机器学习手势分类
- ✅ 数据收集与训练

## 项目结构

```text
drone_hand_gesture/
├── main.py                    # 主程序（本地仿真模式）
├── main_airsim.py            # AirSim 真实模拟器版本
├── airsim_controller.py      # AirSim 控制器（新增）
├── drone_controller.py        # 无人机控制器
├── simulation_3d.py          # 3D 仿真
├── physics_engine.py         # 物理仿真引擎
├── gesture_detector.py       # 基础手势检测器
├── gesture_detector_enhanced.py  # 增强手势检测器
├── gesture_classifier.py     # 手势识别分类器
├── gesture_data_collector.py # 手势图像数据收集
├── train_gesture_model.py    # 训练识别模型
└── requirements.txt          # 依赖列表
```

## 快速开始

### 方案 1：本地仿真模式（不需要 AirSim）

```bash
# 进入项目目录
cd src/drone_hand_gesture

# 安装依赖
pip install -r requirements.txt

# 运行主程序
python main.py
```

### 方案 2：AirSim 真实模拟器模式

**前提条件**：
1. 安装 AirSim：`pip install airsim`
2. 运行 AirSim 模拟器（如 Blocks.exe）

```bash
# 运行 AirSim 版本
python main_airsim.py
```

**控制方式**：
- **手势控制**：张开手掌（悬停）、食指向上（上升）、握拳（降落）等
- **键盘控制**：空格（起飞/降落）、T（起飞）、L（降落）、H（悬停）、Q/ESC（退出）
   
##参考项目

本项目基于以下开源项目开发：

- [Autonomous Drone Hand Gesture Project](https://github.com/chwee/AutonomusDroneHandGestureProject)
  - 原始手势控制无人机项目
  - 提供了基础架构和实现思路

- [MediaPipe Hands](https://github.com/google/mediapipe)
  - Google开源的手部关键点检测框架
  - 本项目使用其进行实时手势识别