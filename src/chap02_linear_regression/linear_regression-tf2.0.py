#!/usr/bin/env python
# coding: utf-8

# ## 设计基函数(basis function) 以及数据读取

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, layers, Model

# 基函数实现优化
def identity_basis(x):
    return np.expand_dims(x, axis=1)

def multinomial_basis(x, feature_num=10):
    x = np.expand_dims(x, axis=1)
    exponents = np.arange(1, feature_num+1)  # 从x^1到x^10
    return x ** exponents

def gaussian_basis(x, centers, width):
    x = np.expand_dims(x, axis=1)
    return np.exp(-0.5 * ((x - centers)/width)**2)

def load_data(filename, basis_func, **basis_params):
    """载入数据。"""
    with open(filename, 'r') as f:
        xys = [list(map(float, line.strip().split())) for line in f]
        xs, ys = zip(*xys)
        xs, ys = np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
        
        # 应用基函数转换
        phi = basis_func(xs, **basis_params) if basis_params else basis_func(xs)
        phi = np.concatenate([np.ones((len(xs), 1)), phi], axis=1)  # 添加偏置项
        
        return (phi, ys), (xs, ys)

# 预先计算高斯基函数的参数
feature_num = 10
centers = np.linspace(0, 25, feature_num)
width = 2.0 * (centers[1] - centers[0])  # 调整宽度参数

# 定义模型
class LinearModel(Model):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.random.normal([ndim, 1], stddev=0.1),
            trainable=True
        )
    
    @tf.function
    def call(self, x):
        return tf.squeeze(tf.linalg.matvec(x, self.w), axis=1)

# 训练与评估
def train_and_evaluate():
    # 加载训练数据
    (train_features, train_labels), (o_x_train, o_y_train) = load_data(
        'train.txt', 
        basis_func=gaussian_basis, 
        centers=centers, 
        width=width
    )
    
    # 加载测试数据
    (test_features, test_labels), (o_x_test, o_y_test) = load_data(
        'test.txt', 
        basis_func=gaussian_basis, 
        centers=centers, 
        width=width
    )
    
    # 初始化模型
    model = LinearModel(train_features.shape[1])
    optimizer = optimizers.Adam(learning_rate=0.01)  # 调整学习率
    
    # 定义训练步骤
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = tf.reduce_mean(tf.square(y - predictions))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    # 训练循环
    for epoch in range(5000):
        loss = train_step(train_features, train_labels)
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss.numpy():.4f}")
    
    # 评估
    def evaluate_rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred.numpy())**2))
    
    # 训练集评估
    train_pred = model(train_features)
    train_rmse = evaluate_rmse(train_labels, train_pred)
    print(f"训练集RMSE: {train_rmse:.2f}")
    
    # 测试集评估
    test_pred = model(test_features)
    test_rmse = evaluate_rmse(test_labels, test_pred)
    print(f"测试集RMSE: {test_rmse:.2f}")
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.scatter(o_x_train, o_y_train, c='r', label='训练集', s=15)
    plt.scatter(o_x_test, o_y_test, c='g', label='测试集真实值', s=15)
    plt.plot(o_x_test, test_pred, 'b-', label='预测曲线', linewidth=2)
    plt.title('基函数回归结果')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    train_and_evaluate()