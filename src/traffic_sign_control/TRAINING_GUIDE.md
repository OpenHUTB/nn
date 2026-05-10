# CARLA交通标志检测系统 - 训练指南

## 目录结构

```
traffic_sign_control/
├── main.py                 # 主程序（数据采集+推理）
├── train.py               # 模型训练脚本
├── dataset.yaml           # 数据集配置文件
├── yolov8n.pt            # 预训练模型
├── final_dataset/        # 数据集（499张图片）
│   ├── images/          # RGB图像
│   ├── labels/          # YOLO格式标注
│   └── semantic/        # 语义分割图像
└── runs/detect/
    └── final_training/  # 训练结果
        └── train/weights/
            ├── best.pt  # 最佳模型
            └── last.pt  # 最后模型
```

## 快速开始

### 1. 训练模型

```bash
python train.py --epochs 50
```

### 2. 运行推理（自动使用最新训练模型）

```bash
python main.py
```

### 3. 查看训练结果

```bash
python start_tensorboard.py
# 访问 http://localhost:6006
```

## 训练参数

```bash
# 基本用法
python train.py

# 自定义参数
python train.py --epochs 100 --batch 16 --img-size 640

# 使用不同基础模型
python train.py --model yolov8s.pt --epochs 50
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch` | 16 | 批次大小 |
| `--img-size` | 640 | 图像尺寸 |
| `--model` | yolov8n.pt | 基础模型 |
| `--project` | runs/detect | 输出目录 |
| `--name` | traffic_sign | 实验名称 |

## 训练输出

训练完成后会生成：

- `runs/detect/traffic_sign/weights/best.pt` - 最佳模型
- `runs/detect/traffic_sign/weights/last.pt` - 最后模型
- `runs/detect/traffic_sign/results.csv` - 训练指标

## 手动指定模型

```bash
# 使用特定模型
python main.py --model runs/detect/final_training/train/weights/best.pt

# 使用预训练模型
python main.py --model yolov8n.pt
```

## 数据采集

```bash
# 采集数据
python main.py --save-interval 10 --output-dir new_dataset

# 按 S 键手动保存，按 ESC 退出
```

## 类别说明

1. **stop_sign** (类别ID: 0) - 停止标志
2. **traffic_light** (类别ID: 1) - 红绿灯
