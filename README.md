# AutoPilot System: Lane & Object Detection

这是一个结合了**传统计算机视觉**与**深度学习**的自动驾驶辅助系统原型。它利用 OpenCV 进行高精度的车道线检测，并集成 YOLOv8 模型实现对车辆、行人等障碍物的实时识别。

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

## ✨ 核心功能

### 1. 🛣️ 车道线检测 (Lane Detection)
* **视觉管线**: 灰度化 -> 高斯模糊 -> Canny 边缘检测 -> 动态 ROI -> 霍夫变换。
* **算法优化**:
    * **智能平滑**: 使用 `deque` 历史队列消除车道线抖动。
    * **斜率过滤**: 自动剔除路面阴影和垂直路标干扰。
    * **可视化调参**: 提供 `tuner.py` 工具，实时寻找针对当前天气的最佳参数。

### 2. 🚗 目标检测 (Object Detection)
* **深度学习**: 集成 **YOLOv8 (Nano)** 模型。
* **识别对象**: 实时框出前方车辆 (Car)、卡车 (Truck)、巴士 (Bus) 和行人 (Person)。
* **性能**: 针对 CPU 优化，轻量级推理。

## 🛠️ 环境准备

### 1. 运行环境
* Python 3.8+ (推荐 Python 3.13)
* 建议使用独立显卡 (GPU) 以获得更高帧率，但在 CPU 上也能运行 (FPS 5-10)。

### 2. 安装依赖
请确保安装了最新版本的依赖库（包含 ultralytics）：
```bash
pip install -r requirements.txt

```

## 🚀 使用指南

### 第一步：视觉参数调优 (可选但推荐)

如果发现车道线检测不准（乱飞或消失），请先运行调参工具：

```bash
python tuner.py sample.hevc

```

* 按 `空格` 暂停，拖动滑动条调整 Canny 阈值和 ROI 区域。
* 记下最佳参数，并填入 `main.py` 顶部的配置区。

### 第二步：启动自动驾驶系统


直接运行主程序，系统将同时加载车道检测器和 YOLO 模型：

```bash
python main.py sample.hevc

```

* **首次运行提示**: 程序会自动下载 `yolov8n.pt` 模型权重文件 (约 6MB)，请保持网络连接。

## 📂 项目结构

```text
Project_Root/
├── main.py            # [入口] 主程序，协调车道与YOLO检测
├── yolo_det.py        # [模块] 封装 YOLOv8 目标检测逻辑
├── tuner.py           # [工具] 视觉参数调试器
├── requirements.txt   # 依赖清单
├── README.md          # 项目文档
└── sample.hevc        # 测试数据

```

## ⚠️ 常见问题

**Q1: 画面非常卡顿 (Low FPS)？**

* **原因**: 深度学习模型在没有 GPU 加速的电脑上运行较慢。
* **优化**: 可以在 `main.py` 中实现“跳帧机制”（例如每 3 帧跑一次 YOLO，中间帧沿用结果）。

**Q2: 报错 `ModuleNotFoundError: No module named 'ultralytics'`?**

* **解决**: 请运行 `pip install -r requirements.txt` 确保库已安装。

## 🔮 路线图 (Roadmap)

* [x] 基础 OpenCV 车道线检测
* [x] 可视化参数调试工具 (`tuner.py`)
* [x] 帧间平滑与防抖算法
* [x] 集成 YOLOv8 目标检测 (`yolo_det.py`)
* [ ] 性能优化：加入多线程或跳帧处理
* [ ] 车辆距离估算 (基于检测框大小)
* [ ] 偏离预警系统 (LDW)

## 📚 参考资料

* [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
* [OpenCV Computer Vision](https://opencv.org/)

```

```

