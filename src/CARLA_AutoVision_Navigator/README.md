# CARLA AutoVision Navigator

## 项目简介
欢迎来到 **CARLA AutoVision Navigator** 项目！本项目是一个基于高度真实的 **CARLA 模拟器** 开发的自动驾驶仿真系统。项目旨在将实时的视觉目标检测与动态轨迹规划相结合，实现自动驾驶车辆在复杂交通环境下的智能导航。

## 项目目录结构
```text
CARLA_AutoVision_Navigator/
├── config.py           # 全局配置模块：统一管理服务器连接与传感器参数
├── requirements.txt    # 环境依赖：列出项目运行所需的第三方库
├── README.md           # 项目说明文档
├── .gitignore          # Git 忽略文件：排除冗余文件
├── src/                # 核心源代码文件夹
│   ├── __init__.py     # 模块初始化文件
│   ├── carla_client.py # CARLA 客户端连接与主车管理模块
│   └── sensor_manager.py # 传感器管理模块：处理 RGB 图像采集与转换 (New!)
├── models/             # 模型文件夹：存放 YOLOv3 权重与网络定义
└── utils/              # 辅助工具文件夹：存放通用工具函数
```

## 感知系统 (Perception System)
本项目目前已实现基础视觉感知层，主要特性如下：
- **传感器挂载**：在主车前盖上方（x=1.5, z=2.4）挂载 RGB 摄像头，模拟真实驾驶视角。
- **实时数据流处理**：
    - 采集 CARLA 原始 `sensor.camera.rgb` 数据。
    - 使用 `NumPy` 进行高效的二进制流转换。
    - 使用 `OpenCV` 进行色彩空间转换（RGBA to BGR）及实时窗口渲染。
- **可配置参数**：
    - 分辨率：默认 800x600 (可在 `config.py` 修改)。
    - 视场角 (FOV)：默认 90 度。

## 快速开始 (Quick Start)

### 1. 启动仿真环境
- 确保 CARLA 服务器已开启。

### 2. 运行感知模块测试
运行以下命令，您将看到主车生成的实时视觉画面：
```bash
python src/carla_client.py
```
*提示：在弹出的 OpenCV 窗口中，您可以实时观察主车行进方向的视野。按下键盘上的 **'q'** 键可安全退出测试并销毁车辆。*

## 开发计划 (Roadmap)
- [x] 初始化项目仓库，编写项目 README 文档。
- [x] 提交 `.gitignore` 和环境依赖配置文件。
- [x] 建立全局配置模块 (`config.py`) 并规范项目目录结构。
- [x] 完善项目安装与运行指南。
- [x] 实现 CARLA 客户端连接与主车管理基础代码。
- [x] 接入并调试视觉传感器数据获取模块 (Perception Layer Step 1)。
- [ ] 接入并调试 YOLOv3 视觉目标检测模块。
- [ ] 实现车辆 PID 控制与局部轨迹规划逻辑。
- [ ] 系统模块整合与最终优化。

## 声明
本项目为课程作业/学术研究项目，代码仅供学习与教育参考。
```