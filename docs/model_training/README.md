# Model Training

模型训练子模块，基于 STEGO 框架实现无监督语义分割模型训练。

## 功能特性

- STEGO 模型配置与训练
- 训练参数设置
- 训练过程监控
- 模型保存与恢复

## STEGO 模型配置

### 配置文件

```yaml
model:
  backbone: resnet50
  num_classes: 19
  dropout_rate: 0.1
  pretrained: true

training:
  batch_size: 4
  epochs: 100
  lr: 0.001
  weight_decay: 0.0001
```

## 训练参数设置

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| batch_size | 批次大小 | 4 |
| epochs | 训练轮数 | 100 |
| lr | 学习率 | 0.001 |
| weight_decay | 权重衰减 | 0.0001 |

## 训练命令

### 启动训练

```bash
cd STEGO/src
python train.py \
    --config config/stego_config.yaml \
    --dataset ./data/processed \
    --output ./output/model_training
```

### 训练监控

```bash
tensorboard --logdir ./output/model_training/logs
```

## 训练过程

### 训练循环

1. 加载预训练模型
2. 初始化优化器和损失函数
3. 迭代训练数据
4. 计算损失并反向传播
5. 保存模型权重

### 损失函数

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

## 模型保存

### 保存路径

```
output/model_training/
├── model_best.pth
├── model_last.pth
├── logs/
│   └── events.out.tfevents.*
└── config.yaml
```

## 运行效果

### 训练日志示例

```
Epoch 1/100:
Train Loss: 2.345 | Train Acc: 0.654
Val Loss: 2.123 | Val Acc: 0.687

Epoch 50/100:
Train Loss: 0.456 | Train Acc: 0.923
Val Loss: 0.567 | Val Acc: 0.891

Epoch 100/100:
Train Loss: 0.234 | Train Acc: 0.967
Val Loss: 0.456 | Val Acc: 0.912
```

## 注意事项

- 确保 GPU 内存充足
- 训练时间较长，建议使用多 GPU
- 定期保存检查点防止意外中断