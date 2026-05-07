#!/usr/bin/env python
# coding: utf-8
"""Linear regression exercise with basis functions and two optimizers."""

from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def load_data(filename):
    """Load two-column data from a text file.

    Blank lines are ignored. Each non-blank line must contain exactly two
    numeric values: x and y.
    """
    data = []
    with open(filename, "r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            values = stripped.split()
            if len(values) != 2:
                raise ValueError(
                    f"{filename} 第 {line_number} 行应包含 2 个数值，实际为 {len(values)} 个"
                )
            data.append([float(values[0]), float(values[1])])

    if not data:
        raise ValueError(f"{filename} 没有可用的数据")

    xs, ys = np.asarray(data, dtype=float).T
    return xs, ys


def identity_basis(x):
    """Return x as a single-column design matrix."""
    x = np.asarray(x, dtype=float)
    return np.expand_dims(x, axis=1)


def multinomial_basis(x, feature_num=10):
    """Map x to polynomial features x^1, x^2, ..., x^feature_num."""
    if feature_num <= 0:
        raise ValueError("feature_num 必须为正整数")

    x = np.asarray(x, dtype=float)
    x = np.expand_dims(x, axis=1)
    return np.concatenate([x**i for i in range(1, feature_num + 1)], axis=1)


def gaussian_basis(x, feature_num=10, min_value=0.0, max_value=25.0):
    """Map x to radial basis function features."""
    if feature_num <= 0:
        raise ValueError("feature_num 必须为正整数")
    if max_value <= min_value:
        raise ValueError("max_value 必须大于 min_value")

    x = np.asarray(x, dtype=float)
    centers = np.linspace(min_value, max_value, feature_num)
    sigma = (max_value - min_value) / feature_num
    return np.exp(-0.5 * ((x[:, np.newaxis] - centers) / sigma) ** 2)


def least_squares(phi, y, alpha=0.0, solver="pinv"):
    """Solve linear regression weights with optional L2 regularization."""
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y, dtype=float)

    if phi.size == 0 or y.size == 0:
        raise ValueError("输入矩阵 phi 和目标值 y 不能为空")
    if phi.ndim != 2:
        raise ValueError("设计矩阵 phi 必须是二维数组")
    if phi.shape[0] != y.shape[0]:
        raise ValueError(
            f"设计矩阵 phi 的样本数 ({phi.shape[0]}) 与目标值 y 的样本数 ({y.shape[0]}) 不匹配"
        )
    if alpha < 0:
        raise ValueError("正则化参数 alpha 必须为非负数")

    n_samples, n_features = phi.shape
    gram = phi.T @ phi + alpha * np.eye(n_features)
    target = phi.T @ y

    if solver == "pinv":
        return np.linalg.pinv(gram) @ target

    if solver == "cholesky":
        try:
            factor = np.linalg.cholesky(gram)
            intermediate = np.linalg.solve(factor, target)
            return np.linalg.solve(factor.T, intermediate)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(gram) @ target

    if solver == "svd":
        u, singular_values, vt = np.linalg.svd(phi, full_matrices=False)
        regularized = singular_values / (singular_values**2 + alpha)
        return vt.T @ (regularized * (u.T @ y))

    raise ValueError("solver 仅支持 'pinv', 'cholesky', 'svd'")


def gradient_descent(phi, y, lr=0.01, epochs=1000):
    """Optimize linear regression weights with batch gradient descent."""
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y, dtype=float)

    if phi.size == 0 or y.size == 0:
        raise ValueError("输入矩阵 phi 和目标值 y 不能为空")
    if phi.ndim != 2:
        raise ValueError("设计矩阵 phi 必须是二维数组")
    if phi.shape[0] != y.shape[0]:
        raise ValueError("phi 和 y 的样本数必须一致")
    if lr <= 0:
        raise ValueError("学习率 lr 必须为正数")
    if epochs <= 0:
        raise ValueError("epochs 必须为正整数")

    w = np.zeros(phi.shape[1])
    for _ in range(epochs):
        y_pred = phi @ w
        gradient = 2 * phi.T @ (y_pred - y) / len(y)
        w -= lr * gradient
    return w


def build_design_matrix(x, basis_func):
    """Build a design matrix with a bias column and basis features."""
    x = np.asarray(x, dtype=float)
    bias = np.expand_dims(np.ones_like(x), axis=1)
    features = basis_func(x)
    return np.concatenate([bias, features], axis=1)


def main(x_train, y_train, use_gradient_descent=False, basis_func=None):
    """Train the model and return a prediction function plus learned weights."""
    if basis_func is None:
        basis_func = identity_basis

    phi = build_design_matrix(x_train, basis_func)
    w_lsq = least_squares(phi, y_train)

    w_gd = None
    if use_gradient_descent:
        w_gd = gradient_descent(phi, y_train, lr=0.01, epochs=1000)

    def predict(x):
        test_phi = build_design_matrix(x, basis_func)
        weights = w_gd if use_gradient_descent and w_gd is not None else w_lsq
        return test_phi @ weights

    return predict, w_lsq, w_gd


def evaluate(ys, ys_pred):
    """Return root mean squared error between true and predicted values."""
    ys = np.asarray(ys, dtype=float)
    ys_pred = np.asarray(ys_pred, dtype=float)
    if ys.shape != ys_pred.shape:
        raise ValueError("真实值和预测值的形状必须一致")
    return float(np.sqrt(np.mean((ys - ys_pred) ** 2)))


def plot_results(x_train, y_train, x_test, y_test, y_test_pred):
    """Plot train data, test data, and predictions."""
    if plt is None:
        raise RuntimeError("matplotlib 未安装，无法绘图")

    plt.plot(x_train, y_train, "ro", markersize=3)
    plt.plot(x_test, y_test, "k", label="test")
    plt.plot(x_test, y_test_pred, "b--", label="pred")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.legend(["train", "test", "pred"])
    plt.show()


def run_demo():
    """Run the example using train.txt and test.txt next to this file."""
    base_dir = Path(__file__).resolve().parent
    x_train, y_train = load_data(base_dir / "train.txt")
    x_test, y_test = load_data(base_dir / "test.txt")

    predict, w_lsq, _ = main(x_train, y_train)
    y_train_pred = predict(x_train)
    y_test_pred = predict(x_test)

    print(f"训练集 RMSE：{evaluate(y_train, y_train_pred):.4f}")
    print(f"测试集 RMSE：{evaluate(y_test, y_test_pred):.4f}")
    print("最小二乘法权重：")
    print(w_lsq)

    plot_results(x_train, y_train, x_test, y_test, y_test_pred)


if __name__ == "__main__":
    run_demo()
