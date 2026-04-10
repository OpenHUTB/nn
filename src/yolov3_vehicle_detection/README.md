# Autonomous Vehicle Object Detection and Trajectory Planning
> 基于 YOLOv3 与 CARLA 模拟器的自动驾驶感知与决策系统 (v1.0.0 Release)

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![CARLA](https://img.shields.io/badge/CARLA-0.9.11-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 项目简介
本项目是一个基于自动驾驶场景的毕业设计/课程作业。核心目标是利用深度学习算法 (**YOLOv3**) 对 **CARLA** 模拟器中的交通环境进行实时感知，并基于视觉反馈实现基础的**自动紧急制动 (AEB)** 决策。

系统通过 Python API 与 CARLA 服务器通信，经由 OpenCV DNN 模块进行推理，最终在可视化界面中展示检测结果与车辆控制状态。

## 🌟 核心功能
*   **实时目标检测**: 识别行人、车辆、交通标志等 80 类目标。
*   **自动避障决策 (AEB)**: 当检测到前方有碰撞风险时，自动触发紧急刹车。
*   **安全走廊可视化**: 实时绘制驾驶辅助线，直观展示算法判定范围。
*   **双模态控制**: 支持在“自动驾驶模式”与“紧急接管模式”间自动切换。
*   **性能监控**: 集成 TensorBoard，实时记录 FPS 与置信度曲线。
*   **灵活部署**: 支持命令行参数配置 IP、端口及无头模式（Headless）。

## 📂 项目结构
```text
Project_Root/
├── assets/              # 演示素材
├── config/              # 全局配置
├── models/              # 模型仓库 (YOLOv3)
├── utils/               # 工具模块
│   ├── carla_client.py  # CARLA 客户端
│   ├── planner.py       # AEB 决策规划
│   ├── logger.py        # 日志记录
│   └── visualization.py # 可视化绘图
├── main.py              # 程序主入口
├── requirements.txt     # 依赖清单
└── LICENSE              # 开源许可证
```

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
```bash
pip install -r requirements.txt
python download_weights.py
```

### 2. 基础运行
确保 CARLA 模拟器已启动，然后运行：
```bash
python main.py
```

### 3. 高级用法 (CLI)
本项目支持命令行参数，适用于不同测试场景：

*   **连接远程服务器**:
    ```bash
    python main.py --host 192.168.1.X --port 2000
    ```

*   **后台无界面模式 (Headless)**:
    用于在服务器上长时间挂机测试，不显示 OpenCV 窗口：
    ```bash
    python main.py --no-render
    ```

### 4. 性能监控
```bash
tensorboard --logdir=runs
```

## 📅 开发日志
*   **v1.0.0 (2026-03)**: 正式发布。集成 CLI 参数支持、安全走廊可视化与 AEB 完整逻辑。
*   **v0.9.0**: 完成 TensorBoard 监控与全局配置重构。
*   **v0.5.0**: 完成 YOLOv3 与 CARLA 的核心联调。