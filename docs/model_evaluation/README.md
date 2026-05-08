# Model Evaluation

模型评估子模块，负责对训练好的 STEGO 模型进行评估和测试。

## 功能特性

- 评估指标计算
- 实时分割测试
- 视频/图像分割
- 结果可视化

## 评估指标

### 主要指标

| 指标 | 说明 | 计算公式 |
|------|------|----------|
| mIoU | 平均交并比 | (TP)/(TP+FP+FN) |
| Pixel Acc | 像素准确率 | (TP+TN)/(TP+TN+FP+FN) |
| F1 Score | F1 分数 | 2*(Precision*Recall)/(Precision+Recall) |

## 评估命令

### 运行评估

```bash
cd STEGO/src
python evaluate.py \
    --model ./output/model_training/model_best.pth \
    --dataset ./data/processed/val \
    --output ./output/model_evaluation
```

### 指标计算

```bash
python compute_metrics.py \
    --pred ./output/model_evaluation/predictions \
    --gt ./data/processed/val/labels
```

## 实时分割测试

### 单图像测试

```bash
python test_single_image.py \
    --model ./output/model_training/model_best.pth \
    --image ./test/image.png \
    --output ./output/model_evaluation/result.png
```

### 视频分割

```bash
python test_video.py \
    --model ./output/model_training/model_best.pth \
    --input ./test/video.mp4 \
    --output ./output/model_evaluation/result.mp4
```

## 结果可视化

### 可视化命令

```bash
python visualize_results.py \
    --input ./output/model_evaluation/predictions \
    --output ./output/model_evaluation/visualization
```

## 运行效果

### 评估结果示例

```
Evaluating on validation set...
--------------------------------
mIoU: 0.785
Pixel Accuracy: 0.892
F1 Score: 0.834
--------------------------------
Class-wise IoU:
Road: 0.891
Building: 0.723
Vegetation: 0.815
Vehicle: 0.756
...
```

## 注意事项

- 确保测试数据与训练数据格式一致
- 评估前检查模型文件路径
- 可视化需要足够的显存