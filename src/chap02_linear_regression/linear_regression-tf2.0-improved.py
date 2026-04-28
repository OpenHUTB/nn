#!/usr/bin/env python
# coding: utf-8
"""
改进版线性回归 —— 针对原有代码的多个问题进行优化
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, layers, Model


def identity_basis(x):
    """恒等基函数"""
    return np.expand_dims(x, axis=1)


def multinomial_basis(x, feature_num=10):
    """多项式基函数"""
    x = np.expand_dims(x, axis=1)
    feat = [x]
    for i in range(2, feature_num + 1):
        feat.append(x**i)
    return np.concatenate(feat, axis=1)


def gaussian_basis(x, feature_num=10):
    """高斯基函数 — 中心点根据数据范围自适应"""
    x_min, x_max = x.min(), x.max()
    # 在数据范围内均匀放置中心点，两端各扩展一点避免边缘效应
    centers = np.linspace(x_min - 0.5, x_max + 0.5, feature_num)
    width = 1.0 * (centers[1] - centers[0])
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x] * feature_num, axis=1)
    out = (x - centers) / width
    return np.exp(-0.5 * out ** 2)


def load_data(filename, basis_func=gaussian_basis):
    """
    载入数据并进行基函数变换
    数据格式: 每行三个值 (序号, x, y) —— 跳过序号列
    """
    x_list, y_list = [], []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            x_list.append(float(parts[0]))
            y_list.append(float(parts[1]))
    xs = np.asarray(x_list, dtype=np.float32)
    ys = np.asarray(y_list, dtype=np.float32)
    o_x, o_y = xs.copy(), ys.copy()

    phi0 = np.expand_dims(np.ones_like(xs), axis=1)
    phi1 = basis_func(xs)
    xs = np.concatenate([phi0, phi1], axis=1)
    return (xs, ys), (o_x, o_y)


class LinearModel(Model):
    """线性回归模型"""

    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        # Xavier 初始化 — 根据输入维度自适应缩放
        limit = np.sqrt(6.0 / ndim)
        self.w = tf.Variable(
            shape=[ndim, 1],
            initial_value=tf.random.uniform(
                [ndim, 1], minval=-limit, maxval=limit, dtype=tf.float32
            ),
            trainable=True,
            name="weight",
        )
        self.b = tf.Variable(
            initial_value=tf.zeros([1], dtype=tf.float32),
            trainable=True,
            name="bias",
        )

    @tf.function
    def call(self, x):
        y = tf.squeeze(tf.matmul(x, self.w), axis=1) + self.b
        return y


# ========== 训练流程 ==========

(xs, ys), (o_x, o_y) = load_data("train.txt")
ndim = xs.shape[1]

model = LinearModel(ndim=ndim)

# 使用指数衰减学习率：初始 0.05，每 2000 步衰减至 96%
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.05,
    decay_steps=2000,
    decay_rate=0.96,
    staircase=True,
)
optimizer = optimizers.Adam(learning_rate=lr_schedule)


@tf.function
def train_one_step(model, xs, ys):
    with tf.GradientTape() as tape:
        y_preds = model(xs)
        # 正确计算标量 loss
        loss = tf.reduce_mean(tf.keras.losses.MSE(ys, y_preds))
    # 同时对 w 和 b 求梯度并更新
    grads = tape.gradient(loss, [model.w, model.b])
    optimizer.apply_gradients(zip(grads, [model.w, model.b]))
    return loss


@tf.function
def predict(model, xs):
    return model(xs)


def evaluate(ys, ys_pred):
    """计算标准差和 R² 决定系数"""
    ys = ys.numpy() if hasattr(ys, 'numpy') else ys
    ys_pred = ys_pred.numpy() if hasattr(ys_pred, 'numpy') else ys_pred
    std = np.std(ys - ys_pred)
    ss_res = np.sum((ys - ys_pred) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    return std, r2


# ---------- 训练 ----------
EPOCHS = 10000
PRINT_INTERVAL = 500

for i in range(1, EPOCHS + 1):
    loss = train_one_step(model, xs, ys)
    if i % PRINT_INTERVAL == 0:
        print(f"Step {i:5d} | loss = {loss:.4f} | lr = {optimizer.learning_rate.numpy():.5f}")

print("\n" + "=" * 50)
print("Train evaluation:")
y_preds = predict(model, xs)
std_train, r2_train = evaluate(ys, y_preds)
print(f"  Std: {std_train:.3f}")
print(f"  R^2: {r2_train:.4f}")

# ---------- 测试 ----------
(xs_test, ys_test), (o_x_test, o_y_test) = load_data("test.txt")
y_test_preds = predict(model, xs_test)
std_test, r2_test = evaluate(ys_test, y_test_preds)
print("\nTest evaluation:")
print(f"  Std: {std_test:.3f}")
print(f"  R^2: {r2_test:.4f}")
print("=" * 50)

# ---------- 可视化 ----------
plt.figure(figsize=(10, 6))
plt.plot(o_x, o_y, "ro", markersize=3, label="train")
plt.plot(o_x_test, o_y_test, "bo", markersize=3, label="test", alpha=0.6)
# 绘制预测曲线（在完整 x 范围上平滑绘制）
x_smooth = np.linspace(
    min(o_x.min(), o_x_test.min()),
    max(o_x.max(), o_x_test.max()),
    500,
)
# 对平滑点做同样的基函数变换
phi0_s = np.expand_dims(np.ones_like(x_smooth), axis=1)
phi1_s = gaussian_basis(x_smooth)
xs_smooth = np.concatenate([phi0_s, phi1_s], axis=1).astype(np.float32)
y_smooth = predict(model, xs_smooth)

plt.plot(x_smooth, y_smooth, "k-", linewidth=2, label="prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Improved Linear Regression (Gaussian Basis)")
plt.grid(True, linestyle="--", alpha=0.5, color="gray")
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("regression_result_improved.png", dpi=150)
plt.show()
