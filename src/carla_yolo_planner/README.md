# Autonomous Vehicle Object Detection and Trajectory Planning
> 基于 YOLOv3 与 CARLA 模拟器的自动驾驶感知与决策系统 (v1.0.0 Release)

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![CARLA](https://img.shields.io/badge/CARLA-0.9.11-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 项目简介
本项目是一个基于自动驾驶场景的毕业设计/课程作业。核心目标是利用深度学习算法 (**YOLOv3**) 对 **CARLA** 模拟器中的交通环境进行实时感知，并基于视觉反馈实现基础的**自动紧急制动 (AEB)** 决策。

系统通过 Python API 与 CARLA 服务器通信，经由 OpenCV DNN 模块进行推理，最终在可视化界面中展示检测结果与车辆控制状态。

## 核心功能
* **实时目标检测**: 识别行人、车辆、交通标志等 80 类目标。
* **自动避障决策 (AEB)**: 当检测到前方有碰撞风险时，自动触发紧急刹车。
* **安全走廊可视化**: 实时绘制驾驶辅助线，直观展示算法判定范围。
* **双模态控制**: 支持在"自动驾驶模式"与"紧急接管模式"间自动切换。
* **性能监控**: 集成 TensorBoard，实时记录 FPS 与置信度曲线。
* **灵活部署**: 支持命令行参数配置 IP、端口及无头模式（Headless）。

## 项目结构
```
yolov3_vehicle_detection/
├── config/
│   ├── __init__.py      # 模块导出
│   └── config.py         # 全局配置
├── models/
│   └── yolo_detector.py  # YOLOv3 目标检测
├── utils/
│   ├── carla_client.py   # CARLA 客户端封装
│   ├── planner.py        # AEB 决策规划
│   ├── logger.py          # TensorBoard 日志
│   └── visualization.py   # 可视化绘图
├── main.py               # 程序主入口
├── download_weights.py   # 模型下载脚本
├── requirements.txt      # 依赖清单
└── README.md             # 项目文档
```

## 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt


```

### 2. 模型权重获取
由于模型权重文件较大，未包含在 Git 仓库中，您可以通过以下方式获取：
自动下载：运行上述 `download_weights.py` 脚本，会自动从官方源下载所需文件

下载 YOLO 模型权重
python download_weights.py

下载完成后，请确保将权重文件放置在 `models/` 目录下，文件结构如下：
```
models/
├── yolov3-tiny.weights    # YOLO 模型权重文件
├── yolov3-tiny.cfg         # YOLO 模型配置文件
└── coco.names              # 类别名称文件
```

### 3. 基础运行
确保 CARLA 模拟器已启动，然后运行：
```bash
python main.py
```

### 4. 高级用法 (CLI)
本项目支持命令行参数，适用于不同测试场景：

* **连接远程服务器**:
  ```bash
  python main.py --host 192.168.1.X --port 2000
  ```

* **后台无界面模式 (Headless)**:
  用于在服务器上长时间挂机测试，不显示 OpenCV 窗口：
  ```bash
  python main.py --no-render
  ```

### 5. 性能监控
```bash
tensorboard --logdir=logs
```

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

## 开发日志
* **v1.0.0 (2026-03)**: 正式发布。集成 CLI 参数支持、安全走廊可视化与 AEB 完整逻辑。
* **v0.9.0**: 完成 TensorBoard 监控与全局配置重构。
* **v0.5.0**: 完成 YOLOv3 与 CARLA 的核心联调。
