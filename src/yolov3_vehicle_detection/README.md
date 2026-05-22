# Autonomous Vehicle Object Detection and Trajectory Planning

> 基于 YOLOv3 与 CARLA 模拟器的自动驾驶感知与轨迹规划系统 (v1.1.0 Release)

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![CARLA](https://img.shields.io/badge/CARLA-0.9.11-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 项目简介

本项目是一个基于自动驾驶场景的毕业设计/课程作业。核心目标是利用深度学习算法 (**YOLOv3**) 对 **CARLA** 模拟器中的交通环境进行实时感知，并基于视觉反馈实现**自动紧急制动 (AEB)**、**智能绕行避让**及**轨迹跟踪**等决策功能。

系统通过 Python API 与 CARLA 服务器通信，经由 OpenCV DNN 模块进行推理，最终在可视化界面中展示检测结果与车辆控制状态。

## 核心功能

| 功能模块 | 说明 |
|----------|------|
| **实时目标检测** | 识别行人、车辆、交通标志等 80 类目标 |
| **AEB 自动紧急制动** | 检测前方障碍物，自动触发紧急刹车 |
| **障碍物识别与避让** | 识别车辆障碍物，触发自动刹车躲避 |
| **碰撞传感器检测** | 新增碰撞传感器，检测车辆碰撞事件 |
| **智能绕行避让** | 检测到障碍物时自动执行变道绕行 |
| **TTC 安全距离计算** | 基于 Time-To-Collision 计算安全距离，优化变道触发逻辑 |
| **目标跟踪优化** | 使用 scipy 匈牙利算法优化 DeepSORT 目标跟踪的检测与轨迹匹配 |
| **轨迹规划 (Pure Pursuit)** | 实现纯追踪算法，支持动态前视距离调整 |
| **车辆状态显示** | 实时识别并显示直行、加速、刹车、左转、右转、手刹、倒车等状态 |
| **性能监控** | 集成 TensorBoard，实时记录 FPS、TTC 与置信度曲线 |
| **调试可视化系统** | 在 CARLA 仿真器中实时显示 FPS、速度、TTC、轨迹等调试信息 |
| **DeepSORT 跟踪** | 多目标跟踪，维持目标 ID 一致性 |
| **双模态控制** | 支持在"自动驾驶模式"与"紧急接管模式"间自动切换 |
| **灵活部署** | 支持命令行参数配置 IP、端口及无头模式（Headless） |

## 项目结构

```
yolov3_vehicle_detection/
├── config/
│   ├── __init__.py          # 模块导出
│   └── config.py            # 全局配置
├── models/
│   └── yolo_detector.py    # YOLOv3 目标检测
├── utils/
│   ├── carla_client.py     # CARLA 客户端封装 + 调试可视化
│   ├── planner.py          # Pure Pursuit 轨迹规划器
│   ├── deep_sort.py        # DeepSORT 多目标跟踪
│   ├── logger.py           # TensorBoard 日志
│   └── visualization.py    # 可视化绘图
├── main.py                 # 程序主入口
├── download_weights.py     # 模型下载脚本
├── requirements.txt        # 依赖清单
└── README.md               # 项目文档
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 下载 YOLO 模型权重
python -m src.yolov3_vehicle_detection.download_weights
```

### 2. 基础运行

确保 CARLA 模拟器已启动，然后运行：

```bash
python -m src.yolov3_vehicle_detection.main
```

### 3. 高级用法 (CLI)

本项目支持丰富的命令行参数：

```bash
# 连接远程服务器
python -m src.yolov3_vehicle_detection.main --host 192.168.1.X --port 2000

# 后台无界面模式 (Headless)
python -m src.yolov3_vehicle_detection.main --no-render

# 启用 Pure Pursuit 轨迹跟踪
python -m src.yolov3_vehicle_detection.main --use-pure-pursuit

# 组合使用
python -m src.yolov3_vehicle_detection.main --host 192.168.1.X --use-pure-pursuit
```

### 4. 性能监控

```bash
tensorboard --logdir=logs
```

### 5. CARLA 调试可视化

运行程序时，CARLA 仿真器窗口会自动显示以下调试信息：

| 颜色 | 显示内容 | 说明 |
|------|----------|------|
| 🟢 绿色 | FPS | 当前帧率 |
| 🟢 绿色 | Speed | 车辆速度 (km/h) |
| 🟡 黄色 | State | 当前驾驶状态 (正常/减速/紧急刹车) |
| 🔴 红色 | Obs | 前方障碍物距离 (m) |
| 🔴 红色 | TTC | Time-To-Collision 碰撞时间 (s) |
| 🟡 黄色 | 轨迹线 | 规划路径点 (30个采样点) |
| 🟣 紫色 | 前视点 | Pure Pursuit 目标点 |
| 🔵 浅蓝 | 前视连线 | 车辆到前视点的连线 |

## 配置说明

主要配置项位于 `config/config.py`:

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| carla_host | 127.0.0.1 | CARLA 服务器地址 |
| carla_port | 2000 | CARLA 服务器端口 |
| conf_thres | 0.5 | YOLO 置信度阈值 |
| nms_thres | 0.4 | NMS 阈值 |
| camera_width | 800 | 摄像头分辨率 |
| camera_height | 600 | 摄像头分辨率 |
| SAFE_ZONE_RATIO | 0.4 | 安全区域比例 |
| COLLISION_AREA_THRES | 0.05 | 碰撞区域阈值 |

## 技术架构

### TTC (Time-To-Collision) 安全系统

系统采用多级 TTC 阈值实现渐进式安全策略：

| TTC 阈值 | 状态 | 响应动作 |
|----------|------|----------|
| > 4.0s | 正常 | 保持当前速度 |
| 3.0 ~ 4.0s | 注意 | 准备减速 |
| 2.0 ~ 3.0s | 警告 | 减速避让 |
| < 1.5s | 紧急 | 紧急制动 |

### Pure Pursuit 轨迹跟踪

采用纯追踪算法实现路径跟踪：
- **动态前视距离**：根据车速自动调整 (`ld = gain * speed + base`)
- **多轨迹支持**：支持直线、弯道等多种轨迹类型
- **平滑转向**：基于自行车模型的转向角计算

### DeepSORT 目标跟踪

利用匈牙利算法优化检测框与轨迹的匹配：
- 维持目标 ID 一致性
- 减少误检和漏检
- 支持遮挡场景处理

## 开发日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| **v1.1.0** | 2026-05 | 添加性能监控与调试可视化系统；新增车辆状态实时显示功能 |
| **v1.0.0** | 2026-03 | 正式发布；集成 CLI 参数支持、安全走廊可视化与 AEB 完整逻辑 |
| **v0.9.0** | 2026-02 | 完成 TensorBoard 监控与全局配置重构 |
| **v0.8.0** | 2026-02 | 实现 TTC 安全距离计算，优化变道触发逻辑 |
| **v0.7.0** | 2026-02 | 使用 scipy 匈牙利算法优化 DeepSORT 目标跟踪 |
| **v0.6.0** | 2026-01 | 新增碰撞传感器，实现智能绕行避让功能 |
| **v0.5.0** | 2026-01 | 完成车辆障碍物识别与自动刹车躲避功能 |
| **v0.5.0** | 2025-12 | 完成 YOLOv3 与 CARLA 的核心联调 |
