# YOLO12 目标检测项目

基于 Ultralytics YOLO12 框架，使用 CARLA 自动驾驶数据集进行目标检测训练。

## 项目简介

本项目使用 [CARLA Object Detection Dataset](https://github.com/DanielHfnr/Carla-Object-Detection-Dataset) 数据集，通过 Google Drive 整理版本进行下载，结合 YOLO12 模型进行自动驾驶场景下的目标检测任务。

**新增自定义模型**：项目中包含多个改进的 YOLO12 模型：
- `yolo12-A2C2f-CGLU.yaml`：在标准 YOLO12 基础上使用 A2C2f_CGLU 模块替换 A2C2f 模块
- `yolo12-A2C2f-DFFN.yaml`：在标准 YOLO12 基础上使用 A2C2f_DFFN 模块替换 A2C2f 模块
以上模型均预设类别数为 5，适配 CARLA 数据集（vehicle, bike, motobike, traffic_light, traffic_sign）。

数据集来源于：[https://github.com/DanielHfnr/Carla-Object-Detection-Dataset](https://github.com/DanielHfnr/Carla-Object-Detection-Dataset)

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA (推荐)
- Ultralytics 框架 (scripts/ 目录下)

## 目录结构

```
yolo12_object_detection/
├─dataset                     # 数据集目录 (从Google Drive下载)
│  ├─annotations              # VOC格式标注文件
│  ├─images                   # 图像文件
│  │  ├─test                  # 测试集图像
│  │  └─train                 # 训练集图像
│  ├─image_sets               # 图像集列表文件 (train.txt, val.txt, test.txt)
│  └─labels                   # YOLO格式标签文件
│      ├─test                 # 测试集标签
│      └─train                # 训练集标签
├─main.py                     # 项目统一入口文件
└─scripts                     # 脚本和框架代码
    ├─train.py               # 训练脚本
    ├─val.py                 # 验证脚本
    ├─runs                    # 运行结果目录
    │  ├─train                # 训练结果
    │  │  └─baseline          # 基线实验
    │  │      └─weights       # 模型权重
    │  └─val                  # 验证结果
    │      └─baseline         # 基线验证
    └─ultralytics             # Ultralytics 框架
        ├─cfg                 # 配置文件
        │  ├─datasets         # 数据集配置
        │  └─models           # 模型配置
        └─...                 # 其他框架文件
```

## 使用说明

### 1. 数据集准备

从 [Google Drive 数据集链接](https://drive.google.com/drive/folders/1lApgN0pp_OcZ4L1fXWY4Vabs8F3vTZcM?usp=sharing) 下载整理好的数据集，解压到项目根目录的 `dataset/` 文件夹下。

**注意**：下载后的数据集已经包含了预配置好的 `data.yaml` 文件，无需额外创建。

### 2. 配置数据集

数据集已包含预配置的 `data.yaml` 文件，位于 `dataset/` 目录下。实际配置如下：

```yaml
# dataset/data.yaml 实际配置
train: train_yolo.txt  # 训练集图像列表
val: val_yolo.txt      # 验证集图像列表
test: test_yolo.txt    # 测试集图像列表

# number of classes
nc: 5

# class names
names: [ 'vehicle', 'bike', 'motobike', 'traffic_light', 'traffic_sign' ]
```

**注意**：数据集的 `image_sets/` 目录下包含了 `train_yolo.txt`、`val_yolo.txt`、`test_yolo.txt` 文件，这些文件列出了对应数据集的图像路径。`labels/` 目录下包含 YOLO 格式的标签文件。

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

