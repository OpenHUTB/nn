# CARLA AutoVision Navigator

## 项目简介
**CARLA AutoVision Navigator** 是一个全栈式自动驾驶仿真项目。系统基于 CARLA 模拟器，构建了从底层环境感知到高层行为决策，再到物理执行控制的完整闭环。通过集成 YOLOv3 深度学习算法与双 PID 控制策略，主车能够在复杂城市环境中实现自主寻迹与安全避障。

## 项目目录结构
```text
CARLA_AutoVision_Navigator/
├── config.py           # 全局配置中心：管理连接、传感器、PID 增益及安全阈值
├── requirements.txt    # 环境依赖列表
├── README.md           # 项目说明文档
├── src/                # 核心源代码
│   ├── carla_client.py # 系统主入口：负责环境初始化、模块调度与 UI 渲染
│   ├── sensor_manager.py # 感知层：视觉数据采集与预处理
│   ├── object_detector.py # 感知层：基于 YOLOv3 的目标检测
│   ├── decision_maker.py # 决策层：基于视觉反馈的碰撞预警与避障决策 (New!)
│   └── pid_controller.py # 控制层：双 PID 纵横向控制算法实现
├── utils/              
│   ├── geometry.py     # 计算工具：提供车速与航向角转换算法
│   └── model_loader.py # 模型工具：权重文件校验
└── models/             # 模型资产 (YOLOv3 Config, Weights, Names)
```

## 系统架构说明

本项目遵循自动驾驶经典的 **PDA (Perception-Decision-Action)** 架构：

### 1. 感知层 (Perception)
- **视觉处理**：实时捕获 800x600 像素的 RGB 图像流。
- **目标识别**：利用 YOLOv3 对交通环境进行语义分析，识别车辆、行人、卡车等关键障碍物。

### 2. 决策层 (Decision) - **重点更新**
- **碰撞风险评估**：通过分析 YOLO 检测框的几何属性（如 Box 高度占比），在无需 LiDAR 的情况下估算障碍物距离。
- **行为决策**：
    - **常规巡航**：保持设定时速并沿导航点行驶。
    - **紧急制动 (AEB)**：当正前方识别到危险障碍物且距离低于安全阈值时，决策器将强制干预控制层，输出全力刹车信号。

### 3. 控制层 (Control)
- **纵向控制**：PID 控制器实时调节油门/刹车，响应决策层的速度请求。
- **横向控制**：基于 Waypoint 的几何修正逻辑，实现精确的车道线跟随。

## 快速开始

### 运行完整系统
确保 CARLA 服务器已启动且模型权重已下载，运行：
```bash
python src/carla_client.py
```
*运行效果：车辆将自主行驶。若在道路中遇到其他车辆或行人，系统将自动触发红色 "EMERGENCY BRAKE!" 警示并刹停。*

## 开发计划 (Roadmap)
- [x] 初始化项目仓库与环境配置。
- [x] 实现 CARLA 客户端连接与主车管理逻辑。
- [x] 接入视觉传感器并实现画面实时流显示。
- [x] 实现 YOLOv3 实时目标检测逻辑。
- [x] 实现基于导航点的双 PID 纵横向控制。
- [x] **实现基于感知结果的自动避障决策算法 (Decision Layer Logic).**
- [ ] 进行系统多场景仿真测试与性能调优。
- [ ] 最终代码重构、规范化文档与 V1.0 发布。

## 声明
本项目为课程作业/学术研究项目，代码仅供学习与教育参考。