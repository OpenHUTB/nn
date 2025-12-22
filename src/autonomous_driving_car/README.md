CARLA 模拟器和自动驾驶基础算法学习

# 无人驾驶汽车项目（基于CARLA模拟器）
## 项目简介
本项目是基于CARLA开源仿真平台、Python和PyCharm开发的无人驾驶仿真系统，融合计算机视觉、路径规划与控制技术，实现虚拟场景中车辆的自主导航、避障与路径跟踪功能，适用于自动驾驶入门实践。

## 核心功能
- 路径规划：A*算法、RRT/RRT*算法实现起点到终点路径生成
- 障碍物检测：YOLOv8+OpenCV实时识别目标，激光雷达点云聚类定位
- 车辆控制：PID控制器实现转向、速度精准控制
- CARLA交互：加载场景、获取传感器数据（摄像头/激光雷达等）、发送控制指令
- 实时可视化：PyGame显示场景、车辆状态与检测结果

## 技术栈
| 类别         | 具体技术/工具                          |
|--------------|---------------------------------------|
| 开发环境     | PyCharm Community Edition 2024+、Windows/Linux |
| 核心语言     | Python 3.8+                           |
| 仿真平台     | CARLA 0.9.15/0.9.16、CARLA Python API |
| 计算机视觉   | OpenCV、NumPy、YOLOv8（Ultralytics）  |
| 路径规划     | A*、RRT/RRT*算法                      |
| 控制理论     | PID控制器                              |
| 可视化与数据 | PyGame、Matplotlib、Pandas             |

## 快速开始
1. 克隆项目到PyCharm，创建Python 3.8+虚拟环境
2. 安装依赖：`pip install -r requirements.txt`（核心依赖：carla、opencv-python、ultralytics、numpy、pygame）
3. 启动CARLA模拟器（运行`CarlaUE4.exe`/`CarlaUE4.sh`）
4. 运行主程序：`python main.py`，自动连接CARLA并启动自主导航

\## 项目结构

autonomous_driving_car/
├── FPV/             # 第一视角可视化模块：负责车辆摄像头视角、检测结果的实时显示
├── MCP/             # 主控制与模块集成：包含感知（检测）、规划（路径）、控制（PID）的核心逻辑
├── main.py          # 项目入口：启动CARLA连接、调用MCP与FPV模块
└── README.md        # 项目说明文档
```

## 常见问题
- CARLA连接失败：确保模拟器已启动，Python API版本与CARLA一致
- 检测速度慢：使用YOLOv8n轻量模型，或启用GPU加速
- 控制不稳定：调整PID参数或增加路径平滑处理