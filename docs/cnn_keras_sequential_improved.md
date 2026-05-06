# Keras Sequential CNN 改进报告

## 一、概述

本次改进基于原 `tutorial_mnist_conv-keras-sequential.py` 代码，针对其存在的多个问题进行了系统性优化。

核心改进涵盖以下六个方面：

1. 数据利用效率（使用全部训练样本）
2. 模型架构（添加 BatchNormalization 和 Dropout）
3. 激活函数选择（tanh -> relu）
4. 训练策略（增加 epoch、添加回调函数）
5. 学习率调度（ReduceLROnPlateau）
6. 正则化（Dropout 防止过拟合）

---

## 二、原代码问题分析

### 2.1 数据利用不足

**原代码：**
```python
ds = ds.take(20000).shuffle(20000).batch(100)
```

**问题：**
- MNIST 训练集有 60000 个样本，但只使用了 20000 个（33%）
- 浪费了 40000 个标注数据，模型无法充分学习

**改进：**
```python
ds = ds.shuffle(60000).batch(100)
```

### 2.2 无 BatchNormalization

**问题：**
- 卷积层后没有归一化处理，每层输入分布随训练不断变化（内部协变量偏移）
- 导致训练不稳定，收敛速度慢

**改进：**
在每个 Conv2D 后添加 `BatchNormalization` 层，稳定每层输入分布。

### 2.3 无 Dropout 正则化

**问题：**
- 模型没有任何正则化手段，容易过拟合训练数据
- 尤其在全连接层（参数量大）更明显

**改进：**
- 展平后添加 `Dropout(0.25)`
- 全连接层后添加 `Dropout(0.5)`

### 2.4 Dense 层使用 tanh 激活

**原代码：**
```python
layers.Dense(128, activation='tanh'),
```

**问题：**
- tanh 在深层网络中容易出现梯度消失问题
- ReLU 计算更简单，收敛更快

**改进：**
```python
layers.Dense(128, activation='relu'),
```

### 2.5 学习率过小且无调度

**原代码：**
```python
optimizer = optimizers.Adam(0.0001)
```

**问题：**
- 学习率 0.0001 过小，训练 5 个 epoch 远远不够收敛
- 固定学习率无法适应训练后期的精细调整需求

**改进：**
- 初始学习率提升到 0.001
- 添加 `ReduceLROnPlateau` 回调，当验证集 loss 不再下降时自动降低学习率

### 2.6 训练轮数不足且无早停

**原代码：**
```python
model.fit(train_ds, epochs=5)
```

**问题：**
- 5 个 epoch 对于 MNIST 严重不足
- 没有验证集监控，无法判断是否过拟合
- 没有早停机制，训练时间固定

**改进：**
- 最大 epoch 增加到 15
- 添加 `validation_data=test_ds` 监控测试集表现
- 添加 `EarlyStopping(patience=3)` 回调，验证集 loss 连续 3 个 epoch 不下降时自动停止

---

## 三、改进后模型架构

```
Conv2D(32, 5x5, relu, same) -> BatchNorm -> MaxPool(2x2)
Conv2D(64, 5x5, relu, same) -> BatchNorm -> MaxPool(2x2)
Flatten -> Dropout(0.25)
Dense(128, relu) -> Dropout(0.5)
Dense(10, softmax)
```

---

## 四、改进效果预期

| 指标 | 原版 | 改进版 |
|------|------|--------|
| 训练样本数 | 20000 | 60000 |
| BatchNormalization | 无 | 每个卷积层后 |
| Dropout | 无 | 0.25 + 0.5 |
| Dense 激活函数 | tanh | relu |
| 初始学习率 | 0.0001 | 0.001 |
| 学习率调度 | 无 | ReduceLROnPlateau |
| 训练轮数 | 5 | 最多15（早停） |
| 验证集监控 | 无 | 每个 epoch |
| 预期测试准确率 | ~98% | ~99.2%+ |
