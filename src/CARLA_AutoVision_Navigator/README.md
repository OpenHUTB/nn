# CARLA AutoVision Navigator (v1.0-Final)

## 项目简介
**CARLA AutoVision Navigator** 是一个集成实时视觉感知、自动避障决策与双 PID 寻迹控制的自动驾驶全栈仿真项目。本项目旨在利用高度真实的 CARLA 模拟器，探索基于深度学习（YOLOv3）的感知结果如何驱动车辆在复杂城市环境中做出安全且平滑的驾驶行为。

## 核心系统架构
本项目完整实现了自动驾驶的 **感知 (Perception) -> 决策 (Decision) -> 控制 (Action)** 闭环逻辑：
- **感知层**：利用 YOLOv3 在 800x600 的视频流中实时检测 80 类交通参与者，并动态监控系统 FPS。
- **决策层**：基于视觉反馈的碰撞风险评估，实现了自动紧急制动 (AEB) 逻辑。
- **控制层**：采用纵向速度 PID 与横向转向 PID 的协同策略，实现自动寻迹与平稳行驶。

## 项目目录结构
```text
CARLA_AutoVision_Navigator/
├── config.py           # 核心配置：集成化的参数管理中心
├── requirements.txt    # 环境依赖：标准化的运行环境定义
├── LICENSE             # 开源协议：MIT License
├── README.md           # 项目文档：V1.0 最终发布版
├── src/                # 核心源码：Perception, Decision, Control 逻辑
├── models/             # 模型资产：YOLOv3 权重与配置文件
└── utils/              # 工具库：几何计算与模型加载校验
```

## 快速开始
1. **启动 CARLA 模拟器**。
2. **下载权重**：运行 `python utils/model_loader.py`。
3. **运行系统**：执行 `python src/carla_client.py`。
*系统将自动连接服务器、生成车辆、并开启全自动驾驶模式。按下 'q' 键安全退出。*

## 开发计划进度 (Final Roadmap)
- [x] 初始化项目仓库与环境配置。
- [x] 实现 CARLA 客户端连接与主车管理逻辑。
- [x] 接入视觉传感器并实现画面实时流显示。
- [x] 实现 YOLOv3 实时目标检测逻辑。
- [x] 实现基于导航点的双 PID 纵横向控制。
- [x] 实现基于感知结果的自动避障决策算法。
- [x] 全代码架构重构与 Google Style 标准化注释。
- [x] 系统综合性能优化与 V1.0 正式发布。
- [x] 全代码架构重构与生产级注释标准化。
- [x] **发布项目总结报告 (SUMMARY.md) 与 V1.0 最终结项 (Current).**

## 项目状态
**Status**: 🎉 Project Concluded - v1.0.0-Stable

## 声明
本项目为课程作业，代码仅供学习参考。
...

## 未来展望
- **感知增强**：后续可引入语义分割模型以提高道路边界识别精度。
- **规划优化**：引入 A* 或 Lattice 算法实现更复杂的局部路径绕障规划。
- **传感器融合**：集成 LiDAR 数据以提升避障逻辑在极端天气下的鲁棒性。

## 声明
本项目为课程作业/学术研究项目，代码仅供学习与教育参考。