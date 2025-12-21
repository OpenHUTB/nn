# CARLA-DeepSORT-Vehicle-Tracking

## 项目介绍
基于CARLA仿真器和Deep SORT算法的2D车辆实时追踪程序，支持NPC生成、车辆追踪、关键信息显示，已修复文字反向、编码错误等问题。

## 环境依赖
- Python 3.7
- CARLA 0.9.10+
- 依赖库：`pip install numpy opencv-python scipy carla`

## 快速开始
1. 启动CARLA仿真器（建议低画质：`CarlaUE4.exe -quality-level=Low`）
2. 运行程序：`python main.py`
3. 操作：按`q`键退出

## 核心功能
- 自动生成自车和20辆NPC（自动驾驶）
- Deep SORT实时追踪车辆，显示追踪ID和类别
- 实时显示车速、追踪车辆数、地图名称
- 自动清理资源，避免残留进程
