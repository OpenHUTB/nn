# CARLA AutoVision Navigator (v1.0-Final)

## 项目简介
**CARLA AutoVision Navigator** 是一个集成实时视觉感知、自动避障决策与双 PID 寻迹控制的自动驾驶全栈仿真项目。本项目旨在利用高度真实的 CARLA 模拟器，构建从底层环境感知到高层行为决策，再到物理执行控制的完整闭环。

## 核心架构 (PDA Architecture)
本项目严格遵循自动驾驶经典的 **感知 (Perception) -> 决策 (Decision) -> 控制 (Action)** 架构：
- **感知层**：集成 YOLOv3 深度学习模型，实现交通参与者的实时检测与 FPS 监控。
- **决策层**：基于视觉反馈的碰撞风险评估，实现了自动紧急制动 (AEB) 逻辑。
- **控制层**：采用纵向速度 PID 与横向转向 PID 的协同策略，实现自动寻迹与平稳行驶。

## 项目目录结构
```text
CARLA_AutoVision_Navigator/
├── LICENSE             # 开源协议 (MIT License)
├── config.py           # 全局配置中心
├── requirements.txt    # 环境依赖列表
├── README.md           # 项目说明文档 (v1.0)
├── src/                # 核心源代码 (Perception, Decision, Control)
├── utils/              # 辅助工具库 (Geometry, Model Loader)
├── tests/              # 自动化测试模块 (Unit Testing Framework)
└── models/             # 模型资源 (YOLOv3 Weights, Config)
```

## 快速开始
1. **环境准备**：启动 CARLA 服务器 (0.9.11+)。
2. **下载权重**：运行 `python utils/model_loader.py`。
3. **运行测试**：执行 `python tests/test_logic.py` 验证算法逻辑。
4. **启动系统**：执行 `python src/carla_client.py` 开启自动驾驶。

## 开发计划进度 (Final Roadmap)
- [x] 初始化项目仓库与环境配置。
- [x] 实现 CARLA 客户端连接与主车管理逻辑。
- [x] 接入视觉传感器并实现画面实时流显示。
- [x] 实现 YOLOv3 实时目标检测逻辑。
- [x] 实现基于导航点的双 PID 纵横向控制。
- [x] 实现基于感知结果的自动避障决策算法。
- [x] 代码架构重构与 Google Style 标准化注释。
- [x] **建立单元测试体系与路径自动化校验 (v1.0 Milestone).**
- [x] **发布结项正式版文档与 LICENSE 声明 (Project Conclusion).**

## 声明
本项目为课程作业/学术研究项目，代码仅供学习与教育参考。