#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, layers, Model

# 基函数实现优化
def identity_basis(x):
    return np.expand_dims(x, axis=1)  # 转为列向量（形状从(n,)变为(n,1)）

def multinomial_basis(x, feature_num=10):
    x = np.expand_dims(x, axis=1)  # 将x转为列向量
    exponents = np.arange(1, feature_num+1)  # 生成指数数组1~feature_num（例如1-10）
    return x ** exponents  # 计算多项式特征，结果形状为(n, feature_num)

def gaussian_basis(x, centers, width):
    x = np.expand_dims(x, axis=1)  # 将x转为列向量
    # 计算高斯核函数：exp(-0.5*((x - centers)/width)^2)
    return np.exp(-0.5 * ((x - centers)/width)**2)

def load_data(filename, basis_func, **basis_params):
    """载入数据并转换特征
    Args:
        filename: 数据文件路径
        basis_func: 基函数转换方法
        basis_params: 基函数参数（如centers, width）
    Returns:
        (特征矩阵, 标签), (原始x, 原始y)
    """
    with open(filename, 'r') as f:
        xys = [list(map(float, line.strip().split())) for line in f]
        xs, ys = zip(*xys)
        xs, ys = np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
        
        # 应用基函数转换
        if basis_params:  # 如果有参数则传递参数
            phi = basis_func(xs, **basis_params)
        else:
            phi = basis_func(xs)
        phi = np.concatenate([np.ones((len(xs), 1)), phi], axis=1)  # 添加偏置项（第一列全1）
        
        return (phi, ys), (xs, ys)

# 高斯基参数预计算
feature_num = 10
centers = np.linspace(0, 25, feature_num)  # 在0-25区间均匀生成10个中心点
width = 2.0 * (centers[1] - centers[0])  # 宽度设为相邻中心点距离的两倍（控制高斯核覆盖范围）

class LinearModel(Model):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        self.w = tf.Variable(  # 权重矩阵初始化
            initial_value=tf.random.normal([ndim, 1], stddev=0.1),  # 正态分布初始化
            trainable=True
        )
    
    @tf.function
    def call(self, x):
        return tf.squeeze(  # 压缩维度（从(n,1)变为(n,)）
            tf.linalg.matvec(x, self.w),  # 矩阵向量乘法：x * w
            axis=1
        )

def train_and_evaluate():
    # 加载训练数据（使用高斯基函数）
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
    
    model = LinearModel(train_features.shape[1])  # 初始化线性模型
    optimizer = optimizers.Adam(learning_rate=0.01)  # 使用Adam优化器，学习率0.01
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)  # 前向传播
            loss = tf.reduce_mean(tf.square(y - predictions))  # 计算均方误差损失
        gradients = tape.gradient(loss, model.trainable_variables)  # 计算梯度
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 更新权重
        return loss
    
    # 训练循环（5000次迭代）
    for epoch in range(5000):
        loss = train_step(train_features, train_labels)
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss.numpy():.4f}")
    
    # 评估函数（计算RMSE）
    def evaluate_rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred.numpy())**2))
    
    # 训练集评估
    train_pred = model(train_features)  # 预测训练集
    train_rmse = evaluate_rmse(train_labels, train_pred)
    print(f"训练集RMSE: {train_rmse:.2f}")
    
    # 测试集评估
    test_pred = model(test_features)  # 预测测试集
    test_rmse = evaluate_rmse(test_labels, test_pred)
    print(f"测试集RMSE: {test_rmse:.2f}")
    
    # 结果可视化
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