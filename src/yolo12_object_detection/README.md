# YOLO12 目标检测项目

基于 Ultralytics YOLO12 框架，使用 CARLA 自动驾驶数据集进行目标检测训练。

## 项目简介

本项目使用 [CARLA Object Detection Dataset](https://github.com/DanielHfnr/Carla-Object-Detection-Dataset) 数据集，结合 YOLO12 模型进行自动驾驶场景下的目标检测任务。

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA (推荐)
- Ultralytics 框架 (scripts/ 目录下)

## 目录结构

```
yolo12_object_detection/
├── scripts/ultralytics/      # Ultralytics 框架代码
├── datasets/                 # 数据集目录
└── runs/                     # 训练结果
```

## 使用说明

### 1. 数据集准备

从 [CARLA Object Detection Dataset](https://github.com/DanielHfnr/Carla-Object-Detection-Dataset) 下载数据集，放置到 `datasets/carla/` 目录下。

### 2. 配置数据集

在 `scripts/ultralytics/cfg/datasets/` 下创建 `carla.yaml` 文件，配置数据集路径和类别信息。

### 3. 训练模型

```bash
cd scripts/ultralytics
python train.py model=yolo12n.yaml data=carla.yaml epochs=100 imgsz=640
```

### 4. 验证和推理

```bash
# 验证
python val.py model=runs/train/exp/weights/best.pt data=carla.yaml

# 推理
python predict.py model=runs/train/exp/weights/best.pt source=test_images/
```

## 模型选择

提供多种 YOLO12 模型版本：
- yolo12n.yaml (Nano - 最快)
- yolo12s.yaml (Small)
- yolo12m.yaml (Medium)
- yolo12l.yaml (Large)
- yolo12x.yaml (XLarge - 最准)

## 待完善

- [ ] 训练结果和性能指标
- [ ] 详细的配置说明
- [ ] 部署和使用示例
- [ ] 实验对比分析