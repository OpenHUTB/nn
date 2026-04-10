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

从 [Google Drive 数据集链接](https://drive.google.com/drive/folders/1lApgN0pp_OcZ4L1fXWY4Vabs8F3vTZcM?usp=sharing) 下载整理好的数据集，放置到 `dataset/` 目录下，保持上述目录结构。

### 2. 配置数据集

在 `scripts/ultralytics/cfg/datasets/` 下创建 `data.yaml` 文件，配置数据集路径和类别信息。

参考配置：
```yaml
path: ../dataset  # 数据集根目录
train: images/train  # 训练集图像
val: images/test  # 验证集图像（根据实际数据集调整）
test: images/test  # 测试集图像

names:
  0: class0
  1: class1
  # 根据实际数据集类别配置
```

### 3. 训练模型

```bash
# 方法1：使用 main.py 入口
python main.py train

# 方法2：直接进入 scripts 目录运行
cd scripts
python train.py
```

### 4. 验证模型

```bash
# 方法1：使用 main.py 入口
python main.py val

# 方法2：直接进入 scripts 目录运行
cd scripts
python val.py
```

### 5. 推理检测

```bash
# 使用 main.py 入口
# 对图片/视频/摄像头进行推理
python main.py predict --source "dataset/images/test/image.jpg"
python main.py predict --source 0  # 使用摄像头
python main.py predict --source "video.mp4"  # 使用视频

# 指定自定义模型路径
python main.py predict --source "test.jpg" --model "runs/train/baseline/weights/best.pt"
```

