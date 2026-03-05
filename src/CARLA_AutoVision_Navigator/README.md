# CARLA AutoVision Navigator

## 项目简介
**CARLA AutoVision Navigator** 是一个集成实时目标检测与自动驾驶控制的仿真项目。该项目基于 CARLA 模拟器，利用 YOLOv3 深度学习算法进行环境感知，并通过 PID 控制算法实现车辆的自动化导航。

目前项目已成功实现了“感知-控制”闭环，车辆能够在识别环境物体的同时自动维持设定时速。

## 项目目录结构
```text
CARLA_AutoVision_Navigator/
├── config.py           # 全局配置中心：管理连接、传感器及 PID 控制参数
├── requirements.txt    # 环境依赖列表
├── README.md           # 项目说明文档
├── .gitignore          # Git 忽略配置文件
├── src/                # 核心源代码
│   ├── __init__.py     
│   ├── carla_client.py # 主程序：负责环境管理、控制回路与 UI 渲染
│   ├── sensor_manager.py # 传感器模块：负责图像采集与预处理
│   ├── object_detector.py # 检测模块：基于 YOLOv3 的实时物体识别
│   └── pid_controller.py # 控制模块：基于 PID 算法的纵向速度控制器 (New!)
├── models/             # 模型资源 (YOLOv3 权重与配置)
└── utils/              # 工具类 (模型校验、路径管理)
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
- [x] 实现 CARLA 客户端连接与主车管理逻辑。
- [x] 接入视觉传感器并实现画面实时流显示。
- [x] 实现并优化 YOLOv3 目标检测模块。
- [x] **开发并集成纵向 PID 速度控制器 (Control Layer Step 1)。**
- [ ] 开发横向转向控制模块，实现车道保持辅助。
- [ ] 探索基于感知结果的简单自动避障逻辑。
- [ ] 系统综合性能评估与最终代码重构。

## 声明
本项目为课程作业/学术研究项目，代码仅供学习与教育参考。