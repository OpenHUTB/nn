# CARLA AutoVision Navigator

## 项目简介
本项目是一个集成 YOLOv3 视觉感知、PID 速度控制与 Waypoint 自动寻迹寻航的自动驾驶仿真系统。

目前项目已成功实现了“感知-控制”闭环，车辆能够在识别环境物体的同时自动维持设定时速。

## 项目目录结构
```text
CARLA_AutoVision_Navigator/
├── config.py           # 全局配置中心：管理服务器连接、传感器参数及 PID 增益
├── requirements.txt    # 环境依赖列表：列出项目运行所需的第三方库
├── README.md           # 项目说明文档
├── .gitignore          # Git 忽略配置：排除权重文件及临时缓存
├── src/                # 核心源代码目录
│   ├── __init__.py     
│   ├── carla_client.py # 系统主入口：负责环境初始化、控制回路调度与 UI 渲染
│   ├── sensor_manager.py # 传感器管理：负责 RGB 摄像头数据的监听与预处理
│   ├── object_detector.py # 感知模块：基于 YOLOv3 的实时目标检测逻辑
│   └── pid_controller.py # 控制模块：通用 PID 控制算法实现
├── models/             # 模型资源目录
│   ├── coco.names      # COCO 数据集类别标签
│   ├── yolov3.cfg      # YOLOv3 网络结构配置文件
│   └── yolov3.weights  # YOLOv3 模型权重文件 (需外部下载)
└── utils/              # 辅助工具目录
    ├── model_loader.py # 模型工具：负责权重文件的完整性校验与下载引导
    └── geometry.py     # 几何计算：负责车速换算及横向偏差航向角计算 (New!)
```

## 核心模块说明

### 1. 感知层 (Perception Layer)
- **目标检测**：集成 YOLOv3 模型，实时识别行人、车辆及障碍物。
- **画面增强**：在实时视频流中同步标注检测框、类别置信度以及车辆当前的实时速度数值。

### 2. 控制层 (Control Layer)
本项目采用了经典的 **PID (Proportional-Integral-Derivative)** 控制算法来实现车辆的纵向速度管理：
- **比例项 (P)**：根据当前速度与目标速度的误差立即做出反应，提供主要的加速动力。
- **积分项 (I)**：用于消除系统静差，确保车辆在长时间行驶中能精准锁定目标时速。
- **微分项 (D)**：预测误差趋势，防止加速过猛导致的过冲，提升行驶的平顺性。

**控制回路流程**：
1. 获取主车实时速度 $v_{current}$。
2. 计算误差 $e = v_{target} - v_{current}$。
3. PID 计算输出信号 $u \in [-1, 1]$。
4. 根据信号正负自动分配给 `throttle` (油门) 或 `brake` (刹车)。

## 快速开始

### 运行完整系统
确保 CARLA 服务器已启动，运行：
```bash
python src/carla_client.py
```
*运行效果：车辆将平滑加速至 20km/h（可在 config.py 修改），同时窗口实时显示 YOLO 检测结果。*

## 开发计划 (Roadmap)
- [x] 初始化项目仓库与环境配置。
- [x] 实现 CARLA 客户端连接与主车管理。
- [x] 接入视觉传感器并实现画面实时流。
- [x] 实现 YOLOv3 实时目标检测逻辑。
- [x] 开发并调优纵向 PID 速度控制器。
- [x] **实现基于地图导航点的自动寻迹转向功能 (Lateral Control).**
- [ ] 开发基础自动避障决策算法。
- [ ] 系统综合性能调优与代码清理。

## 声明
本项目为课程作业/学术研究项目，代码仅供学习与教育参考。