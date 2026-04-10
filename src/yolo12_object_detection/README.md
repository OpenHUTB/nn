# YOLO12 目标检测项目

基于 Ultralytics YOLO12 框架，使用 CARLA 自动驾驶数据集进行目标检测训练。

## 项目简介

本项目使用整理好的 [CARLA 数据集](https://drive.google.com/drive/folders/1lApgN0pp_OcZ4L1fXWY4Vabs8F3vTZcM?usp=sharing)，结合 YOLO12 模型进行自动驾驶场景下的目标检测任务。

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA (推荐)
- Ultralytics 框架 (scripts/ 目录下)

## 目录结构

```
yolo12_object_detection/
├─dataset                     # 数据集目录
│  ├─annotations              # 标注文件
│  ├─images                   # 图像文件
│  │  ├─test                  # 测试集图像
│  │  └─train                 # 训练集图像
│  ├─image_sets               # 图像集列表文件
│  └─labels                   # 标签文件
│      ├─test                 # 测试集标签
│      └─train                # 训练集标签
└─scripts                     # 脚本和框架代码
    ├─runs                    # 运行结果目录
    │  ├─train                # 训练结果
    │  │  └─baseline          # 基线实验
    │  │      └─weights       # 模型权重
    │  └─val                  # 验证结果
    │      └─baseline         # 基线验证
    └─ultralytics             # Ultralytics 框架
        ├─assets              # 资源文件
        ├─cfg                 # 配置文件
        │  ├─datasets         # 数据集配置
        │  ├─models           # 模型配置
        │  ├─solutions        # 解决方案配置
        │  └─trackers         # 跟踪器配置
        ├─data                # 数据处理
        ├─engine              # 训练推理引擎
        ├─hub                 # Hub 功能
        ├─models              # 模型定义
        ├─nn                  # 神经网络模块
        ├─solutions           # 解决方案
        ├─trackers            # 跟踪器
        └─utils               # 工具函数
```

## 使用说明

### 1. 数据集准备

从 [Google Drive 数据集链接](https://drive.google.com/drive/folders/1lApgN0pp_OcZ4L1fXWY4Vabs8F3vTZcM?usp=sharing) 下载整理好的数据集，放置到 `datasets/carla/` 目录下，保持上述目录结构。

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