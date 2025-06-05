#!/usr/bin/env python
# coding: utf-8

# ## 设计基函数(basis function) 以及数据读取

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, layers, Model


def identity_basis(x):
    """恒等基函数"""
    return np.expand_dims(x, axis=1)

# 生成多项式基函数
def multinomial_basis(x, feature_num=10):
    """多项式基函数"""
    x = np.expand_dims(x, axis=1)  # shape(N, 1)
# 初始化特征列表
    feat = [x]
    for i in range(2, feature_num + 1):
        feat.append(x**i)
    ret = np.concatenate(feat, axis=1)
    return ret


def gaussian_basis(x, feature_num=10):
    """高斯基函数"""
    # 使用np.linspace在区间[0, 25]上均匀生成feature_num个中心点
    centers = np.linspace(0, 25, feature_num)
    # 计算高斯函数的宽度(标准差)
    width = 1.0 * (centers[1] - centers[0])
    # 使用np.expand_dims在x的第1维度(axis=1)上增加一个维度
    x = np.expand_dims(x, axis=1)
    # 将x沿着第1维度(axis=1)复制feature_num次并连接
    x = np.concatenate([x] * feature_num, axis=1)
    
    out = (x - centers) / width  # 计算每个样本点到每个中心点的标准化距离
    ret = np.exp(-0.5 * out ** 2)  # 对标准化距离应用高斯函数
    return ret


def load_data(filename, basis_func=gaussian_basis):
    """载入数据"""
    xys = []
    with open(filename, "r") as f:
        for line in f:
            # 改进: 转换为list
            xys.append(list(map(float, line.strip().split())))
        xs, ys = zip(*xys)
        xs, ys = np.asarray(xs), np.asarray(ys)
        
        o_x, o_y = xs, ys
        phi0 = np.expand_dims(np.ones_like(xs), axis=1)
        phi1 = basis_func(xs)
        xs = np.concatenate([phi0, phi1], axis=1)
        return (np.float32(xs), np.float32(ys)), (o_x, o_y)


# ## 定义模型
class linearModel(Model):
    def __init__(self, ndim):
        super(linearModel, self).__init__()
        self.w = tf.Variable(
            shape=[ndim, 1],
            initial_value=tf.random.uniform(
                [ndim, 1], minval=-0.1, maxval=0.1, dtype=tf.float32
            )
        )
        
    @tf.function
    def call(self, x):
        """模型前向传播
        
        参数:
            x: 输入特征，形状为(batch_size, ndim)
            
        返回:
            预测值，形状为(batch_size,)
        """
        y = tf.squeeze(tf.matmul(x, self.w), axis=1)
        return y


(xs, ys), (o_x, o_y) = load_data("train.txt")        
ndim = xs.shape[1]

model = linearModel(ndim=ndim)


# ## 训练以及评估
optimizer = optimizers.Adam(0.1)


@tf.function
def train_one_step(model, xs, ys):
    # 在梯度带(GradientTape)上下文中记录前向计算过程
    with tf.GradientTape() as tape:
        y_preds = model(xs)    # 模型前向传播计算预测值
        loss = tf.reduce_mean(tf.sqrt(1e-12 + (ys - y_preds) ** 2))    #计算损失函数
    grads = tape.gradient(loss, model.w)    # 计算损失函数对模型参数w的梯度
    optimizer.apply_gradients([(grads, model.w)])    # 更新模型参数
    return loss


@tf.function
def predict(model, xs):
    y_preds = model(xs)     # 模型前向传播
    return y_preds


def evaluate(ys, ys_pred):
    """评估模型。"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std


# 评估指标的计算
for i in range(1000):
    loss = train_one_step(model, xs, ys)
    if i % 100 == 1:
        print(f"loss is {loss:.4}")
                
y_preds = predict(model, xs)
std = evaluate(ys, y_preds)
print("训练集预测值与真实值的标准差：{:.1f}".format(std))

(xs_test, ys_test), (o_x_test, o_y_test) = load_data("test.txt")

y_test_preds = predict(model, xs_test)
std = evaluate(ys_test, y_test_preds)
print("训练集预测值与真实值的标准差：{:.1f}".format(std))

plt.plot(o_x, o_y, "ro", markersize=3)
plt.plot(o_x_test, y_test_preds, "k")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression")
# 虚线网格，半透明灰色
plt.grid(True, linestyle="--", alpha=0.7, color="gray")
plt.legend(["train", "test", "pred"])
plt.tight_layout()  # 自动调整布局
plt.show()
