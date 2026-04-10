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

### 运行步骤(在ubuntu或是xubuntu下)
1. 进入项目目录：

   ```bash
   cd /mnt/hgfs/nn/src/Drone_hand_gesture_project

2. 激活虚拟环境：

   ```bash
   source gesture_env/bin/activate

3. 收集手势图像：
   
   ```bash
   python gesture_data_collector.py
   
4. 训练学习模型：

   ```bash
   python train_gesture_model.py --model_type ensemble

5. 运行主程序：

   ```bash
   python main.py
   
##参考项目

本项目基于以下开源项目开发：

- [Autonomous Drone Hand Gesture Project](https://github.com/chwee/AutonomusDroneHandGestureProject)
  - 原始手势控制无人机项目
  - 提供了基础架构和实现思路

- [MediaPipe Hands](https://github.com/google/mediapipe)
  - Google开源的手部关键点检测框架
  - 本项目使用其进行实时手势识别