# 线性回归模型偏置更新修复

## 线性回归基本原理

线性回归是最基础的机器学习算法之一，用于建立自变量x与因变量y之间的线性关系。

### 数学表达式

标准的线性回归模型表达式为：

```text
y = w·x + b
```

其中：

- `y`：预测值（因变量）
- `x`：输入特征（自变量）
- `w`：权重参数（斜率）
- `b`：偏置参数（截距）

### 偏置项的几何意义

偏置项`b`在几何上表示直线在y轴上的截距：

- 当`b = 0`时，直线必须通过原点(0, 0)
- 当`b ≠ 0`时，直线可以在y轴上任意位置

### 偏置项的数学意义

在梯度下降优化中，偏置项的梯度计算为：

```text
∂L/∂b = ∂L/∂y · ∂y/∂b = ∂L/∂y · 1
```

这意味着偏置项的梯度等于损失函数对预测值的梯度，是优化过程中不可或缺的一部分。

---

## 问题描述

在 `src/chap02_linear_regression/linear_regression-tf2.0.py` 文件中，存在两个bug：

### Bug 1：偏置项未被更新

在 `train_one_step` 函数中，只计算了权重 `model.w` 的梯度并更新，但完全没有计算和更新偏置 `model.b`。

**问题代码：**

```python
@tf.function
def train_one_step(model, xs, ys):
    with tf.GradientTape() as tape:
        y_preds = model(xs)
        loss = tf.keras.losses.MSE(ys, y_preds)
    grads = tape.gradient(loss, model.w)    # 只对 w 求梯度
    optimizer.apply_gradients([(grads, model.w)])    # 只更新 w
    return loss
```

**问题分析：**

1. `tape.gradient(loss, model.w)` 只计算损失对权重`w`的梯度
2. `optimizer.apply_gradients([(grads, model.w)])` 只更新权重`w`
3. 偏置`model.b`在整个训练过程中始终保持初始值0

**修复后：**

```python
@tf.function
def train_one_step(model, xs, ys):
    with tf.GradientTape() as tape:
        y_preds = model(xs)
        loss = tf.keras.losses.MSE(ys, y_preds)
    grads = tape.gradient(loss, [model.w, model.b])    # 对 w 和 b 求梯度
    optimizer.apply_gradients(zip(grads, [model.w, model.b]))    # 更新 w 和 b
    return loss
```

**修复说明：**

1. `tape.gradient(loss, [model.w, model.b])` 同时计算损失对`w`和`b`的梯度
2. `zip(grads, [model.w, model.b])` 将梯度与对应参数配对
3. `optimizer.apply_gradients()` 同时更新`w`和`b`

### Bug 2：图例标签不匹配

第200行的图例有3个标签 `["train", "test", "pred"]`，但实际只绘制了2条曲线。

**问题代码：**

```python
plt.plot(o_x, o_y, "ro", markersize=3)      # 绘制训练数据点
plt.plot(o_x_test, y_test_preds, "k")        # 绘制预测曲线
plt.legend(["train", "test", "pred"])        # 3个标签，但只有2条曲线
```

**修复后：**

```python
plt.plot(o_x, o_y, "ro", markersize=3)      # 绘制训练数据点
plt.plot(o_x_test, y_test_preds, "k")        # 绘制预测曲线
plt.legend(["train", "pred"])                # 2个标签，与曲线对应
```

---

## 代码结构分析

### 模型定义

```python
class LinearModel(Model):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        # 权重参数
        self.w = tf.Variable(
            shape=[ndim, 1],
            initial_value=tf.random.uniform([ndim, 1], minval=-0.1, maxval=0.1),
            trainable=True,
            name="weight"
        )
        # 偏置参数
        self.b = tf.Variable(
            initial_value=tf.zeros([1], dtype=tf.float32),
            trainable=True,
            name="bias"
        )

    @tf.function
    def call(self, x):
        y = tf.squeeze(tf.matmul(x, self.w), axis=1) + self.b
        return y
```

**关键点：**

- 模型定义了`w`和`b`两个可训练参数
- `call`方法中正确使用了`self.b`
- 但训练函数中却遗漏了对`b`的更新

### 基函数变换

文件中定义了三种基函数：

1. **恒等基函数**：直接返回输入
2. **多项式基函数**：将x映射为[x, x², x³, ...]
3. **高斯基函数**：将x映射为一组高斯分布特征

```python
def gaussian_basis(x, feature_num=10):
    centers = np.linspace(0, 25, feature_num)  # 高斯中心点
    width = 1.0 * (centers[1] - centers[0])    # 高斯宽度
    # 计算高斯特征
    out = (x - centers) / width
    ret = np.exp(-0.5 * out ** 2)
    return ret
```

---

## 影响分析

### 偏置未更新的影响

#### 1. 模型表达能力受限

**数学解释：**

- 没有偏置项，模型只能表示通过原点的直线：`y = w·x`
- 有偏置项，模型可以表示任意位置的直线：`y = w·x + b`

**实际影响：**

- 如果数据的真实分布不经过原点，模型无法准确拟合
- 例如：房价预测中，即使面积为0，也可能有基础价格（偏置）

#### 2. 预测精度下降

**误差分析：**

- 设真实关系为 `y = 2x + 3`
- 无偏置模型只能学习 `y = 2x`
- 当x=1时，预测误差为3（真实值5，预测值2）
- 当x=10时，预测误差为3（真实值23，预测值20）

**相对误差：**

- 小x值时，相对误差巨大
- 大x值时，相对误差较小但绝对误差固定

#### 3. 训练不稳定

**梯度传播：**

- 偏置项提供额外的梯度路径
- 没有偏置项，梯度只能通过权重传播
- 可能导致梯度消失或爆炸

### 图例不匹配的影响

#### 1. 可视化混淆

- 读者会寻找不存在的"test"曲线
- 误解数据集的划分方式

#### 2. 专业性降低

- 在学术论文或技术报告中显得不严谨
- 影响对作者专业能力的信任

---

## 修复验证

### 验证方法

1. **检查梯度计算**

```python
# 验证偏置梯度是否非零
with tf.GradientTape() as tape:
    y_pred = model(xs)
    loss = tf.keras.losses.MSE(ys, y_pred)
grads = tape.gradient(loss, [model.w, model.b])
print(f"权重梯度: {grads[0].numpy()}")
print(f"偏置梯度: {grads[1].numpy()}")  # 应该非零
```

1. **检查参数更新**

```python
# 记录训练前后的偏置值
print(f"训练前偏置: {model.b.numpy()}")
# 执行训练...
print(f"训练后偏置: {model.b.numpy()}")  # 应该有变化
```

1. **对比改进版文件**

- 参考 `linear_regression-tf2.0-improved.py` 中的实现
- 该文件已经修复了偏置更新问题

### 性能对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 训练集标准差 | 较大 | 较小 |
| 测试集标准差 | 较大 | 较小 |
| 收敛速度 | 较慢 | 较快 |
| 模型表达能力 | 受限 | 完整 |

---

## 相关文件

- **原始文件**：`src/chap02_linear_regression/linear_regression-tf2.0.py`
- **改进版文件**：`src/chap02_linear_regression/linear_regression-tf2.0-improved.py`
- **其他相关**：`src/chap02_linear_regression/linear_regression.py`（NumPy版本）

---

## 扩展阅读

### 线性回归的变体

1. **岭回归**：添加L2正则化，防止过拟合
2. **Lasso回归**：添加L1正则化，实现特征选择
3. **弹性网络**：结合L1和L2正则化

### TensorFlow梯度计算

```python
# 基本用法
with tf.GradientTape() as tape:
    y = model(x)
    loss = loss_fn(y_true, y)
gradients = tape.gradient(loss, model.trainable_variables)

# 多参数梯度
gradients = tape.gradient(loss, [param1, param2])
optimizer.apply_gradients(zip(gradients, [param1, param2]))
```

### 偏置项的最佳实践

1. **始终包含偏置项**：除非有明确的业务约束要求直线通过原点
2. **初始化为0**：偏置通常初始化为0，权重使用随机初始化
3. **正则化注意**：偏置项通常不参与正则化计算
