# SVM 支持向量机代码改进报告

## 一、概述

本次改进针对 `src/chap03_SVM/svm.py` 中的线性 SVM 实现，修复了一个关键 Bug 并增加了四项训练优化功能。改进保持了原有算法（hinge loss + L2 正则化 + 梯度下降）不变，仅在工程实现层面进行优化。

核心改进：

1. **修复 predict() 崩溃 Bug** — 原代码调用 `train()` 后再调用 `predict()` 会因 `y_train_unique` 未定义而报错
2. **学习率指数衰减** — 训练初期快速下降，后期精细调优
3. **训练损失监控** — 每隔固定轮数打印 loss、accuracy、lr，便于观察收敛过程
4. **早停机制** — 损失不再下降时自动停止，节省计算资源
5. **默认开启正则化** — `reg_lambda` 从 0.0 改为 0.001，防止过拟合

---

## 二、原代码问题分析

### 2.1 predict() 方法存在崩溃 Bug

**原代码：**
```python
def predict(self, x_raw):
    x = self.scaler.transform(x_raw)
    score = np.dot(x, self.w) + self.b
    return np.where(score >= 0, 1, -1) if -1 in self.y_train_unique else np.where(score >= 0, 1, 0)

def train_with_label_tracking(self, data_train):
    self.y_train_unique = np.unique(data_train[:, 2])  # 只在这里赋值
    self.train(data_train)
```

**问题：**
- `predict()` 依赖 `self.y_train_unique` 属性
- 该属性只在 `train_with_label_tracking()` 中赋值
- 如果直接调用 `train()` 后再调用 `predict()`，会抛出 `AttributeError: 'SVM' object has no attribute 'y_train_unique'`
- 需要一个专门的 wrapper 方法来避免这个 Bug，说明封装存在缺陷

### 2.2 训练过程无反馈

**原代码：**
```python
for epoch in range(self.max_iter):
    # ... 梯度计算和更新 ...
    # 无任何输出
```

**问题：**
- 默认 20000 次迭代，全程无输出
- 无法判断模型是否收敛、收敛速度如何
- 调试和调参非常困难

### 2.3 固定学习率

**原代码：**
```python
self.learning_rate = learning_rate  # 始终不变
```

**问题：**
- 训练初期需要大学习率快速收敛
- 训练后期需要小学习率精细调优
- 固定学习率无法同时满足两者，容易在最优点附近震荡

### 2.4 无早停机制

**问题：**
- 无论是否已收敛，都跑满 `max_iter` 轮
- 浪费计算资源
- 过多的迭代可能导致过拟合

### 2.5 正则化默认关闭

**原代码：**
```python
def __init__(self, learning_rate=0.1, reg_lambda=0.0, max_iter=20000):
```

**问题：**
- `reg_lambda=0.0` 意味着没有正则化
- SVM 的核心优势之一就是通过正则化控制模型复杂度
- 默认关闭正则化容易导致过拟合

---

## 三、改进内容详解

### 3.1 修复 predict() Bug

**改进代码：**
```python
def train(self, data_train):
    X_raw = data_train[:, :2]
    y_raw = data_train[:, 2]
    # 在 train() 中直接记录标签空间
    self.label_set = np.unique(y_raw)
    # ...

def predict(self, x_raw):
    x = self.scaler.transform(x_raw)
    score = np.dot(x, self.w) + self.b
    # 使用 train() 中记录的 self.label_set
    if -1 in self.label_set:
        return np.where(score >= 0, 1, -1)
    else:
        return np.where(score >= 0, 1, 0)
```

**改进点：**
- 将标签记录逻辑从 `train_with_label_tracking()` 移入 `train()` 本身
- 删除了多余的 `train_with_label_tracking()` 方法
- `train()` + `predict()` 可以直接配合使用，无需 wrapper

### 3.2 学习率指数衰减

**改进代码：**
```python
def __init__(self, ..., lr_decay=0.9995, ...):
    self.lr_decay = lr_decay

def train(self, data_train):
    lr = self.learning_rate
    for epoch in range(self.max_iter):
        # ... 使用 lr 进行梯度更新 ...
        # 学习率衰减
        lr = self.learning_rate * (self.lr_decay ** epoch)
```

**原理：**
- 学习率按 `lr = lr_0 × decay^epoch` 指数衰减
- 默认 `decay=0.9995`，即每轮衰减 0.05%
- 20000 轮后学习率衰减至初始值的约 0.004 倍

**效果：**
- 线性数据集：lr 从 0.1 衰减至 0.000033
- 非线性数据集：lr 从 0.1 衰减至 0.004977（6000 轮时早停）

### 3.3 训练损失监控

**改进代码：**
```python
def _compute_hinge_loss(self, X, y):
    """计算 hinge loss + L2 正则化损失"""
    score = np.dot(X, self.w) + self.b
    hinge = np.maximum(0, 1 - y * score)
    return np.mean(hinge) + self.reg_lambda * np.dot(self.w, self.w)

def train(self, data_train):
    for epoch in range(self.max_iter):
        # ... 训练逻辑 ...
        if (epoch + 1) % self.print_interval == 0 or epoch == 0:
            loss = self._compute_hinge_loss(X, y)
            pred = np.where(np.dot(X, self.w) + self.b >= 0, 1, -1)
            acc = np.mean(pred == y)
            print(f"  epoch {epoch+1:>6d} | loss: {loss:.4f} | acc: {acc*100:.1f}% | lr: {lr:.6f}")
```

**效果：**
- 每 2000 轮打印一次训练状态
- 显示当前 epoch、hinge loss、训练准确率、学习率
- 直观观察收敛过程

### 3.4 早停机制

**改进代码：**
```python
best_loss = float('inf')
no_improve_count = 0

for epoch in range(self.max_iter):
    # ... 训练逻辑 ...

    if (epoch + 1) % self.print_interval == 0:
        loss = self._compute_hinge_loss(X, y)
        if loss < best_loss - 1e-6:
            best_loss = loss
            no_improve_count = 0
        else:
            no_improve_count += self.print_interval
            if no_improve_count >= self.patience:
                print(f"  早停触发于 epoch {epoch+1}，最佳损失: {best_loss:.4f}")
                break
```

**原理：**
- 每 `print_interval` 轮检查一次损失
- 如果损失连续 `patience` 轮没有显著下降（< 1e-6），则停止训练
- 默认 `patience=2000`，即连续 2000 轮无改善则早停

**效果（实测）：**

| 数据集 | 原始轮数 | 改进后轮数 | 节省 |
|--------|----------|------------|------|
| 线性数据 | 20000 | 16000 | 20% |
| 非线性数据 | 20000 | 6000 | **70%** |

### 3.5 默认开启正则化

**改进代码：**
```python
def __init__(self, learning_rate=0.1, reg_lambda=0.001, ...):
```

**效果：**
- `reg_lambda` 从 0.0 改为 0.001
- 提供适度的 L2 正则化，限制权重大小
- 防止过拟合，提高泛化能力

---

## 四、结果对比

### 线性数据集

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 训练准确率 | 95.5% | 95.5% |
| 测试准确率 | 97.5% | 97.5% |
| 训练轮数 | 20000 | 16000（早停） |
| 训练过程 | 无输出 | 每 2000 轮打印 loss/acc/lr |

### 非线性数据集

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 训练准确率 | 81.5% | 81.5% |
| 测试准确率 | 81.0% | 81.0% |
| 训练轮数 | 20000 | 6000（早停） |
| 训练过程 | 无输出 | 每 2000 轮打印 loss/acc/lr |

### 训练过程日志（线性数据集）

```
  epoch      1 | loss: 0.8834 | acc: 96.0% | lr: 0.100000
  epoch   2000 | loss: 0.1116 | acc: 95.5% | lr: 0.036797
  epoch   4000 | loss: 0.1109 | acc: 95.5% | lr: 0.013534
  epoch   6000 | loss: 0.1107 | acc: 95.5% | lr: 0.004977
  epoch   8000 | loss: 0.1106 | acc: 95.5% | lr: 0.001831
  epoch  10000 | loss: 0.1106 | acc: 95.5% | lr: 0.000673
  epoch  12000 | loss: 0.1106 | acc: 95.5% | lr: 0.000248
  epoch  14000 | loss: 0.1106 | acc: 95.5% | lr: 0.000091
  epoch  16000 | loss: 0.1106 | acc: 95.5% | lr: 0.000033
  早停触发于 epoch 16000，最佳损失: 0.1106
train accuracy: 95.5%
test accuracy: 97.5%
```

### 训练过程日志（非线性数据集）

```
  epoch      1 | loss: 0.9460 | acc: 81.5% | lr: 0.100000
  epoch   2000 | loss: 0.3947 | acc: 81.0% | lr: 0.036797
  epoch   4000 | loss: 0.3947 | acc: 81.5% | lr: 0.013534
  epoch   6000 | loss: 0.3947 | acc: 81.5% | lr: 0.004977
  早停触发于 epoch 6000，最佳损失: 0.3947
train accuracy: 81.5%
test accuracy: 81.0%
```

---

## 五、新增命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--learning-rate` | 0.1 | 初始学习率 |
| `--reg-lambda` | 0.001 | L2 正则化系数（原默认 0.0） |
| `--max-iter` | 20000 | 最大迭代次数 |
| `--lr-decay` | 0.9995 | 学习率衰减系数（新增） |
| `--patience` | 2000 | 早停耐心值（新增） |

---

## 六、总结

### 改进总结

本次对 `svm.py` 的改进聚焦于工程实践层面，不改变核心算法：

1. **修复了 predict() 的属性未定义 Bug** — 消除了对 `train_with_label_tracking()` 的依赖
2. **添加学习率指数衰减** — 兼顾初期快速收敛和后期精细调优
3. **添加训练过程监控** — 实时显示 loss、accuracy、lr，便于调试
4. **添加早停机制** — 线性数据节省 20% 迭代，非线性数据节省 70%
5. **默认开启 L2 正则化** — 提高泛化能力

### 可继续改进的方向

- **Mini-batch SGD**：用小批量梯度下降替代全批量，加速大数据集训练
- **核函数扩展**：引入 RBF 核处理非线性数据
- **交叉验证**：自动选择最优超参数组合
- **可视化**：绘制决策边界和训练 loss 曲线

---

## 七、使用方式

```bash
cd src/chap03_SVM

# 使用默认参数运行（线性数据集）
python svm.py

# 自定义参数运行
python svm.py --learning-rate 0.05 --reg-lambda 0.01 --lr-decay 0.999 --patience 3000

# 使用非线性数据集
python svm.py --train-file data/train_kernel.txt --test-file data/test_kernel.txt
```
