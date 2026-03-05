# CARLA AutoVision Navigator

## 项目简介
**CARLA AutoVision Navigator** 是一个集成实时目标检测与自动驾驶控制的仿真项目。该项目基于 CARLA 模拟器，利用 YOLOv3 深度学习算法进行环境感知，并通过 PID 控制算法实现车辆的自动化导航。

目前项目已完成感知层的核心开发与优化，正在进入控制层的算法实现阶段。

## 项目目录结构
```text
CARLA_AutoVision_Navigator/
├── config.py           # 全局配置中心：管理连接、传感器及 PID 控制参数 (Updated!)
├── requirements.txt    # 环境依赖列表
├── README.md           # 项目说明文档
├── .gitignore          # Git 忽略配置文件
├── src/                # 核心源代码
│   ├── __init__.py     # 包初始化
│   ├── carla_client.py # 主程序：负责环境管理与主循环
│   ├── sensor_manager.py # 传感器模块：负责图像采集与预处理
│   └── object_detector.py # 检测模块：基于 YOLOv3 的实时物体识别 (Fixed & Optimized!)
├── models/             # 模型资源
│   ├── coco.names      # 类别标签
│   ├── yolov3.cfg      # 网络配置
│   └── yolov3.weights  # 模型权重 (需手动下载)
└── utils/              # 工具类
    └── model_loader.py  # 权重校验工具
```

## 技术特性

### 1. 感知层 (Perception Layer)
- **实时目标检测**：采用 YOLOv3 模型，能够精准识别周围的行人、车辆、交通灯等 80 类目标。
- **鲁棒性优化**：已修复模型加载时的路径兼容性问题（FileNotFoundError）及 OpenCV 解析配置异常（Parsing Error），支持在多种 IDE 运行环境下稳定启动。
- **高效处理**：通过 NumPy 与 OpenCV 的深度集成，实现了低延迟的视觉数据流转化。

### 2. 控制层 (Control Layer) - 正在开发
- **纵向控制**：引入 PID 速度控制器，旨在实现平滑的加速、减速及恒速巡航。
- **参数化配置**：所有的控制增益（Kp, Ki, Kd）均已在 `config.py` 中模块化，便于针对不同路况进行调优。

## 快速开始

### 1. 环境准备
- 启动 CARLA 服务器。
- 确保 `models/yolov3.weights` 已就绪（可通过运行 `python utils/model_loader.py` 查看指引）。

### 2. 运行系统
```bash
python src/carla_client.py
```
*提示：在预览窗口中观察目标检测结果，按下 'q' 键可退出并清理环境。*

## 常见问题排查 (Troubleshooting)
- **OpenCV Parsing Error**: 若遇到 `Failed to parse NetParameter file`，请检查 `models/yolov3.cfg` 是否下载完整，项目已增加异常捕获机制。
- **路径加载失败**: 项目现采用动态绝对路径寻找模型文件，建议在项目根目录下运行脚本以获得最佳兼容性。

## 开发计划 (Roadmap)
- [x] 初始化项目仓库与环境配置。
- [x] 实现 CARLA 客户端连接与主车管理逻辑。
- [x] 接入视觉传感器并实现画面实时流显示。
- [x] **实现并修复 YOLOv3 目标检测模块的集成问题 (感知层完成)。**
- [x] **预配置控制层 PID 参数并优化系统目录结构。**
- [ ] 开发基于 PID 算法的车辆纵向速度控制器。
- [ ] 实现车辆基础转向控制逻辑。
- [ ] 系统模块集成测试与性能优化。

## 声明
本项目为课程作业/学术研究项目，代码仅供学习与教育参考。
