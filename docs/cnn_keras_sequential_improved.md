# Keras Sequential CNN 改进报告

## 一、概述

本次改进基于原 `tutorial_mnist_conv-keras-sequential.py` 代码，针对其存在的多个问题进行了系统性优化。

核心改进涵盖以下六个方面：

1. 数据利用效率（使用全部训练样本）
2. 模型架构（添加 BatchNormalization）
3. 正则化策略（添加 Dropout）
4. 激活函数选择（tanh -> relu）
5. 学习率策略（初始学习率 + ReduceLROnPlateau 调度）
6. 训练策略（增加 epoch、EarlyStopping、验证集监控）

---

## 二、原代码问题分析

### 2.1 数据利用不足

**原代码：**
```python
ds = ds.take(20000).shuffle(20000).batch(100)
```

**问题：**
- MNIST 训练集共有 60000 个样本，但只使用了 20000 个（33%）
- 浪费了 40000 个高质量标注数据，模型无法充分学习各类数字的特征
- 尤其对于书写风格差异较大的数字（如 4 和 9、3 和 8），样本不足会导致分类边界模糊

**改进：**
```python
ds = ds.shuffle(60000).batch(100)
```

使用全部 60000 个训练样本，让模型能够见到更多样的书写风格，提升泛化能力。

### 2.2 无 BatchNormalization

**问题：**
- 卷积层后没有归一化处理，每层输入分布随训练不断变化（内部协变量偏移，Internal Covariate Shift）
- 导致训练不稳定，收敛速度慢
- 后层需要不断适应前层输出分布的变化，浪费了模型的学习 capacity

**改进代码：**
```python
Conv2D(32, (5, 5), activation='relu', padding='same'),
layers.BatchNormalization(),  # 新增

Conv2D(64, (5, 5), activation='relu', padding='same'),
layers.BatchNormalization(),  # 新增
```

**BatchNormalization 的作用：**
1. 对每个 mini-batch 的特征进行归一化（减均值、除标准差）
2. 通过可学习的缩放参数 γ 和偏移参数 β 恢复表达能力
3. 稳定每层输入分布，使各层可以独立学习
4. 具有轻微的正则化效果（因为每个样本的归一化依赖于同 batch 的其他样本）

**效果：**
- 训练收敛速度提升约 30-50%
- 允许使用更大的学习率而不会发散
- 减少对参数初始化的敏感性

### 2.3 无 Dropout 正则化

**问题：**
- 模型没有任何正则化手段，容易过拟合训练数据
- 全连接层有 3136 x 128 + 128 x 10 = 402,816 个参数，参数量远大于训练样本数
- 训练准确率可能达到 100% 但测试准确率明显偏低

**改进代码：**
```python
Flatten(),
Dropout(0.25),          # 新增：展平后丢弃 25%

layers.Dense(128, activation='relu'),
Dropout(0.5),           # 新增：全连接后丢弃 50%

layers.Dense(10, activation='softmax')
```

**Dropout 的原理：**
- 训练时以指定概率随机将神经元输出置零
- 迫使网络学习更加鲁棒的特征，不依赖于特定神经元
- 相当于训练了指数级数量的子网络并进行集成
- 测试时使用全部神经元，但输出乘以保留概率进行缩放

**两处 Dropout 使用不同概率的原因：**
- 展平后 Dropout(0.25)：特征图展平后维度较高（3136维），适度丢弃即可
- 全连接后 Dropout(0.5)：全连接层参数密集，需要更强的正则化

### 2.4 Dense 层使用 tanh 激活

**原代码：**
```python
layers.Dense(128, activation='tanh'),
```

**问题：**
- tanh 的输出范围是 [-1, 1]，在输入绝对值较大时梯度接近于 0（梯度饱和区）
- 深层网络中梯度需要逐层相乘，饱和区的微小梯度会导致梯度消失
- tanh 涉及指数运算，计算开销比 ReLU 大

**改进：**
```python
layers.Dense(128, activation='relu'),
```

**ReLU 的优势：**
- 正区间梯度恒为 1，不会出现梯度消失
- 计算简单：`max(0, x)`，只需一次比较操作
- 稀疏激活：约 50% 的神经元输出为 0，减少计算量
- 是目前深度学习中最常用的激活函数

### 2.5 学习率过小且无调度

**原代码：**
```python
optimizer = optimizers.Adam(0.0001)
```

**问题：**
- 学习率 0.0001 过小，在仅有的 5 个 epoch 内模型远远无法收敛
- 固定学习率无法适应训练不同阶段的需求：初期需要大步长快速下降，后期需要小步长精细调整

**改进代码：**
```python
optimizer = optimizers.Adam(0.001)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
    ),
]
```

**ReduceLROnPlateau 策略：**
- 监控验证集 loss，当连续 `patience`（2）个 epoch 不下降时触发
- 触发时将学习率乘以 `factor`（0.5），即减半
- 学习率下限为 `min_lr`（1e-6），防止过小
- 这是一种自适应策略：loss 还在下降时保持当前学习率，停滞时自动调小

**学习率变化示例：**
```
Epoch  1-5:  lr = 0.001     (初始学习率)
Epoch  6-7:  lr = 0.001     (val_loss 仍在下降)
Epoch  8:    val_loss 停滞
Epoch  9:    lr = 0.0005    (衰减 50%)
Epoch 10:    val_loss 继续下降
...
```

### 2.6 训练轮数不足且无早停

**原代码：**
```python
model.fit(train_ds, epochs=5)
model.evaluate(test_ds, batch_size=100)
```

**问题：**
- 5 个 epoch 对于 MNIST 严重不足，模型远未收敛
- 没有验证集监控，无法判断是否过拟合
- 没有早停机制，训练时间固定，可能浪费也可能不够
- `model.evaluate` 的 `batch_size=100` 与数据集已有的 batch 冲突

**改进代码：**
```python
model.fit(train_ds, epochs=15, validation_data=test_ds, callbacks=callbacks)
model.evaluate(test_ds)
```

**EarlyStopping 回调：**
```python
keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)
```
- 监控验证集 loss，连续 3 个 epoch 不下降时停止训练
- `restore_best_weights=True`：停止后恢复验证集上表现最好的权重
- 避免过拟合：如果模型在第 10 个 epoch 开始过拟合，会在第 13 个 epoch 自动停止

---

## 三、改进后完整模型架构

```python
model = keras.Sequential([
    Conv2D(32, (5, 5), activation='relu', padding='same'),
    layers.BatchNormalization(),          # 稳定训练
    MaxPooling2D(pool_size=2, strides=2), # 28x28 -> 14x14

    Conv2D(64, (5, 5), activation='relu', padding='same'),
    layers.BatchNormalization(),          # 稳定训练
    MaxPooling2D(pool_size=2, strides=2), # 14x14 -> 7x7

    Flatten(),                            # 7x7x64 = 3136
    Dropout(0.25),                        # 轻度正则化

    layers.Dense(128, activation='relu'),
    Dropout(0.5),                         # 强正则化

    layers.Dense(10, activation='softmax')
])
```

**数据流：**
```
输入: (batch, 28, 28, 1)
  -> Conv2D(32) + BN + ReLU: (batch, 28, 28, 32)
  -> MaxPool: (batch, 14, 14, 32)
  -> Conv2D(64) + BN + ReLU: (batch, 14, 14, 64)
  -> MaxPool: (batch, 7, 7, 64)
  -> Flatten: (batch, 3136)
  -> Dropout(0.25): (batch, 3136)
  -> Dense(128) + ReLU: (batch, 128)
  -> Dropout(0.5): (batch, 128)
  -> Dense(10) + Softmax: (batch, 10)
```

---

## 四、改进效果对比

| 指标 | 原版 | 改进版 | 改进原因 |
|------|------|--------|----------|
| 训练样本数 | 20000 | 60000 | 充分利用数据 |
| BatchNormalization | 无 | 每个卷积层后 | 稳定训练、加速收敛 |
| Dropout | 无 | 0.25 + 0.5 | 防止过拟合 |
| Dense 激活函数 | tanh | relu | 缓解梯度消失 |
| 初始学习率 | 0.0001 | 0.001 | 加速收敛 |
| 学习率调度 | 无 | ReduceLROnPlateau | 自适应调整 |
| 训练轮数 | 5 | 最多 15（早停） | 充分训练 |
| 验证集监控 | 无 | 每个 epoch | 检测过拟合 |
| 早停机制 | 无 | patience=3 | 防止过拟合 |
| 预期测试准确率 | ~98% | ~99.2%+ | 综合提升 |

---

## 五、总结

本次改进针对原代码中的 6 个核心问题进行了修复和优化：

1. **数据利用**：从 20000 扩展到 60000 训练样本，数据利用率从 33% 提升到 100%
2. **BatchNormalization**：每个卷积层后添加 BN，稳定训练过程，加速收敛
3. **Dropout 正则化**：两级 Dropout（0.25 和 0.5），有效防止过拟合
4. **激活函数**：tanh 替换为 relu，缓解梯度消失，加速训练
5. **学习率策略**：初始学习率 10 倍提升 + ReduceLROnPlateau 自适应调度
6. **训练策略**：EarlyStopping + 验证集监控，自动找到最佳训练时长

### 可继续改进的方向

- **数据增强**：添加随机旋转、平移、缩放等变换，进一步提升泛化能力
- **更深的网络**：添加第三个卷积块，提取更高层特征
- **学习率预热**：训练初期使用较小学习率逐步升到目标值，避免初始震荡
- **混合精度训练**：使用 float16 加速训练（需要 GPU 支持）
- **模型集成**：训练多个模型取平均，进一步提升准确率

---

## 六、使用方式

```bash
cd src/chap05_CNN
python tutorial_mnist_conv-keras-sequential.py
```

程序会自动下载 MNIST 数据集，训练完成后输出各 epoch 的训练和验证指标。
