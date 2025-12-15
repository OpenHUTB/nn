# 通过手势来控制无人机行动

### 运行目录结构
gesture_drone_project/
├── main.py # 主程序入口
├── gesture_detector.py # 手势检测模块
├── drone_controller.py # 无人机控制模块
├── requirements.txt # 依赖包列表
└── README.md # 项目说明

### 运行步骤(在ubuntu或是xubuntu下)
1. 进入项目目录：
   ```bash
   cd /mnt/hgfs/nn/src/Drone_hand_gesture_project

2. 激活虚拟环境：

   ```bash
   source gesture_env/bin/activate

3. 运行主程序：

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