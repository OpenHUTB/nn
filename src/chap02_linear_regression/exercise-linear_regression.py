#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """
    从文件中读取数据，然后把数据拆分成特征和标签，最后以 NumPy 数组的形式返回。

    Args:
        filename: 数据文件的路径。

    Returns:
        tuple: 包含特征和标签的 numpy 数组 (xs, ys)。
    """
    xys = []
    with open(filename, "r") as f:
        for line in f:
            line_data = list(map(float, line.strip().split()))
            xys.append(line_data)
    xs, ys = zip(*xys)
    return np.asarray(xs), np.asarray(ys)


def identity_basis(x):
    """
    恒等基函数：在 x 的最后一个维度上增加一个维度，将其转换为二维数组，
    用于适配线性回归的矩阵运算格式。

    Args:
        x: 输入数据。

    Returns:
        ret: 转换后的二维数组。
    """
    ret = np.expand_dims(x, axis=1)
    return ret


def multinomial_basis(x, feature_num=10):
    """
    多项式基函数：将输入映射为多项式特征。

    Args:
        x: 输入数据。
        feature_num: 多项式特征的数量，默认为 10。

    Returns:
        ret: 多项式基函数转换后的数组。
    """
    x = np.expand_dims(x, axis=1)
    ret = [x**i for i in range(1, feature_num + 1)]
    ret = np.concatenate(ret, axis=1)
    return ret


def gaussian_basis(x, feature_num=10):
    """
    高斯基函数：将输入映射为一组高斯函数响应。

    Args:
        x: 输入数据。
        feature_num: 高斯基函数的数量，默认为 10。

    Returns:
        ret: 高斯基函数转换后的数组。
    """
    centers = np.linspace(0, 25, feature_num)
    sigma = 25 / feature_num
    return np.exp(-0.5 * ((x[:, np.newaxis] - centers) / sigma) ** 2)


def least_squares(phi, y, alpha=0.0, solver="pinv"):
    """
    带正则化的最小二乘法优化，支持多种求解器。

    Args:
        phi: 设计矩阵，形状为 (n_samples, n_features)。
        y: 目标值，形状为 (n_samples,) 或 (n_samples, n_targets)。
        alpha: 正则化参数，默认值为 0.0（无正则化）。
        solver: 求解器类型，支持 'pinv'、'cholesky' 和 'svd'。

    Returns:
        w: 优化后的权重向量，形状为 (n_features,) 或 (n_features, n_targets)。

    Raises:
        ValueError: 当 solver 参数不是支持的类型时抛出。
    """
    if phi.size == 0 or y.size == 0:
        raise ValueError("输入矩阵 phi 和目标值 y 不能为零矩阵")

    if phi.shape[0] != y.shape[0]:
        raise ValueError(
            f"设计矩阵 phi 的样本数 ({phi.shape[0]}) 与目标值 y 的样本数 ({y.shape[0]}) 不匹配"
        )

    n_samples, n_features = phi.shape

    if solver == "pinv":
        A = phi.T @ phi + alpha * np.eye(n_features)
        w = np.linalg.pinv(A) @ phi.T @ y

    elif solver == "cholesky":
        if alpha < 0:
            raise ValueError("使用 Cholesky 求解器时，正则化参数 alpha 必须为非负数")
        A = phi.T @ phi + alpha * np.eye(n_features)
        try:
            L = np.linalg.cholesky(A)
            z = np.linalg.solve(L, phi.T @ y)
            w = np.linalg.solve(L.T, z)
        except np.linalg.LinAlgError:
            print("警告: Cholesky 分解失败，矩阵可能非正定，回退到伪逆求解")
            w = np.linalg.pinv(A) @ phi.T @ y

    elif solver == "svd":
        U, s, Vt = np.linalg.svd(phi, full_matrices=False)
        s_reg = s / (s**2 + alpha)
        S_reg = np.zeros((n_features, n_samples))
        np.fill_diagonal(S_reg, s_reg)
        w = Vt.T @ S_reg @ U.T @ y

    else:
        raise ValueError(
            f"不支持的求解器: {solver}，支持的选项有 'pinv', 'cholesky', 'svd'"
        )

    return w


def gradient_descent(phi, y, lr=0.01, epochs=1000):
    """
    梯度下降优化。

    Args:
        phi: 特征矩阵。
        y: 标签向量。
        lr: 学习率（默认为 0.01）。
        epochs: 迭代次数（默认为 1000）。

    Returns:
        w: 优化后的权重向量。
    """
    w = np.zeros(phi.shape[1])
    for epoch in range(epochs):
        y_pred = phi @ w
        gradient = -2 * phi.T @ (y - y_pred) / len(y)
        w -= lr * gradient
    return w


def main(x_train, y_train, use_gradient_descent=False):
    """
    训练模型，并返回从 x 到 y 的映射。

    Args:
        x_train: 训练集特征。
        y_train: 训练集标签。
        use_gradient_descent: 是否使用梯度下降，默认为 False。

    Returns:
        f: 预测函数。
        w_lsq: 最小二乘法得到的权重向量。
        w_gd: 梯度下降法得到的权重向量。
    """
    basis_func = identity_basis

    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)
    phi1 = basis_func(x_train)
    phi = np.concatenate([phi0, phi1], axis=1)

    w_lsq = np.dot(np.linalg.pinv(phi), y_train)

    w_gd = None
    if use_gradient_descent:
        w_gd = gradient_descent(phi, y_train, lr=0.001, epochs=5000)

    def f(x):
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_func(x)
        phi = np.concatenate([phi0, phi1], axis=1)
        if use_gradient_descent and w_gd is not None:
            return np.dot(phi, w_gd)
        else:
            return np.dot(phi, w_lsq)

    return f, w_lsq, w_gd


def evaluate(ys, ys_pred):
    """
    评估模型。

    Args:
        ys: 真实值。
        ys_pred: 预测值。

    Returns:
        std: 预测值与真实值的标准差。
    """
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std


if __name__ == "__main__":
    train_file = "train.txt"
    test_file = "test.txt"
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    print(x_train.shape)
    print(x_test.shape)

    f, w_lsq, w_gd = main(x_train, y_train)

    y_train_pred = f(x_train)
    std = evaluate(y_train, y_train_pred)
    print("训练集预测值与真实值的标准差：{:.1f}".format(std))

    y_test_pred = f(x_test)
    std = evaluate(y_test, y_test_pred)
    print("预测值与真实值的标准差：{:.1f}".format(std))

    plt.plot(x_train, y_train, "ro", markersize=3)
    plt.plot(x_test, y_test, "k")
    plt.plot(x_test, y_test_pred, "k")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.legend(["train", "test", "pred"])
    plt.show()
