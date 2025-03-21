#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_data(filename):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(list(map(float, line.strip().split())))
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)


def identity_basis(x):
    ret = np.expand_dims(x, axis=1)
    return ret



def multinomial_basis(x, feature_num=10):
    '''多项式基函数'''
    x = np.expand_dims(x, axis = 1)  # shape(N, 1)
    ret = np.hstack([x ** i for i in range(feature_num)])
    return ret


def gaussian_basis(x, feature_num=10):
    '''高斯基函数'''
    centers = np.linspace(0, 25, feature_num)
    width = 1.0
    x = np.expand_dims(x, axis = 1)
    ret = np.hstack([np.exp(-(x - c) ** 2 / (2 * width ** 2)) for c in centers])
    return ret


def main(x_train, y_train):
    """
    训练模型，并返回从x到y的映射。
    """
    basis_func = gaussian_basis
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)
    phi1 = basis_func(x_train)
    phi = np.concatenate([phi0, phi1], axis = 1)

    # 最小二乘法
    w_ls = np.dot(np.linalg.pinv(phi), y_train)
    # 梯度下降
    learning_rate = 0.01
    epochs = 1000
    w_gd = np.zeros(phi.shape[1])

    for epoch in range(epochs):
        y_pred = np.dot(phi, w_gd)
        error = y_pred - y_train
        gradient = np.dot(phi.T, error) / len(y_train)
        w_gd -= learning_rate * gradient
    w = w_gd


    def f(x):
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_func(x)
        phi = np.concatenate([phi0, phi1], axis=1)
        y = np.dot(phi, w)
        return y

    return f


def evaluate(ys, ys_pred):
    """评估模型。"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std


if __name__ == '__main__':
    train_file = 'train.txt'
    test_file = 'test.txt'
    # 载入数据
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    print(x_train.shape)
    print(x_test.shape)

    # 使用线性回归训练模型，返回一个函数f()使得y = f(x)
    f = main(x_train, y_train)

    y_train_pred = f(x_train)
    std = evaluate(y_train, y_train_pred)
    print('训练集预测值与真实值的标准差：{:.1f}'.format(std))

    # 计算预测的输出值
    y_test_pred = f(x_test)
    # 使用测试集评估模型
    std = evaluate(y_test, y_test_pred)
    print('预测值与真实值的标准差：{:.1f}'.format(std))


    # 显示结果
    plt.plot(x_train, y_train, 'ro', markersize=3, label='训练数据')  # 添加 label
    plt.plot(x_test, y_test_pred, 'k', label='预测结果')  # 添加 label

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('高斯基函数')
    plt.legend()  # 显示图例
    plt.show()

