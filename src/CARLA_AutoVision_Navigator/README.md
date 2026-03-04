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
├── models/             
│   ├── coco.names      # COCO数据集类别标签 (New!)
│   ├── yolov3.cfg      # YOLOv3 网络结构配置文件
│   └── yolov3.weights  # YOLOv3 权重文件 (需手动下载)
└── utils/              
    ├── model_loader.py # 模型校验与加载工具类 (New!)
    └── ...
```

## 感知系统 (Perception System)
本项目感知层由“图像采集”与“目标检测”两个核心环节组成：

### 1. 图像采集 (Image Acquisition)
- 在主车前方挂载 RGB 摄像头。
- 利用 NumPy 处理每秒 20+ 帧的实时画面流。

### 2. 目标检测 (Object Detection)
- **核心算法**：采用 **YOLOv3 (You Only Look Once v3)**，在保持较高检测精度的同时具备卓越的推理速度。
- **类别支持**：基于 COCO 数据集，能够识别交通环境中的行人、车辆（小汽车、卡车、公交车）、交通灯、自行车等 80 类物体。
- **模型管理**：通过 `utils/model_loader.py` 自动化检查本地环境，确保模型资产就绪。

## 快速开始 (Quick Start)

### 1. 权重文件下载
由于 `.weights` 权重文件体积较大，不包含在代码仓库中。请运行以下命令检查或根据提示下载：
```bash
python utils/model_loader.py
```
*提示：您也可以直接从 [YOLO 官网](https://pjreddie.com/media/files/yolov3.weights) 下载并放入 `models/` 目录。*

### 2. 运行基础连接测试
```bash
python src/carla_client.py
```

## 开发计划 (Roadmap)
- [x] 初始化项目仓库，编写项目 README 文档。
- [x] 提交 `.gitignore` 和环境依赖配置文件。
- [x] 建立全局配置模块 (`config.py`) 并规范项目目录结构。
- [x] 完善项目安装与运行指南。
- [x] 实现 CARLA 客户端连接与主车管理基础代码。
- [x] 接入并调试视觉传感器数据获取模块。
- [x] 搭建目标检测模块基础架构与模型配置 (Object Detection Phase 1)。
- [ ] 实现基于 YOLOv3 的实时物体识别逻辑。
- [ ] 实现车辆 PID 控制与局部轨迹规划逻辑。
- [ ] 系统模块整合与最终优化。

## 声明
本项目为课程作业/学术研究项目，代码仅供学习与教育参考。