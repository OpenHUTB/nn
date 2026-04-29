#!/usr/bin/env python
# coding: utf-8
import numpy as np # 导入NumPy库。NumPy（Numerical Python）是 Python 中最基础、最强大的科学计算库之一

# 条件导入 matplotlib，增强兼容性
try:
    import matplotlib.pyplot as plt
except ImportError:
    # 当 matplotlib 不可用时，创建模拟对象以支持测试
    class DummyPlt:
        def plot(self, *args, **kwargs): pass
        def show(self, *args, **kwargs): pass
        def xlabel(self, *args, **kwargs): pass
        def ylabel(self, *args, **kwargs): pass
        def title(self, *args, **kwargs): pass
        def legend(self, *args, **kwargs): pass
    plt = DummyPlt()

# 用于创建各种静态、交互式和动画可视化图表


def load_data(filename):
    """载入数据。
    Args:
        filename: 数据文件的路径
    Returns:
        tuple: 包含特征和标签的numpy数组 (xs, ys)
    """
    xys = []
    with open(filename, "r") as f:
        for line in f:
            line_data = list(map(float, line.strip().split()))
            xys.append(line_data)
    xs, ys = zip(*xys)
    return np.asarray(xs), np.asarray(ys)


def identity_basis(x):
    """恒等基函数"""
    return np.expand_dims(x, axis=1)


def multinomial_basis(x, feature_num=10):
    """多项式基函数：将输入x映射为多项式特征
    feature_num: 多项式的最高次数
    返回 shape (N, feature_num)"""
    x = np.expand_dims(x, axis=1)
    ret = [x ** i for i in range(1, feature_num + 1)]
    ret = np.concatenate(ret, axis=1)
    return ret


def gaussian_basis(x, feature_num=10):
    """高斯基函数：将输入x映射为一组高斯分布特征
    用于提升模型对非线性关系的拟合能力"""
    centers = np.linspace(0, 25, feature_num)
    sigma = 25 / feature_num
    return np.exp(-0.5 * ((x[:, np.newaxis] - centers) / sigma) ** 2)


def least_squares(phi, y, alpha=0.0, solver="pinv"):
    """带正则化的最小二乘法优化，支持多种求解器

    参数:
    phi (np.ndarray): 设计矩阵，形状为 (n_samples, n_features)
    y (np.ndarray): 目标值，形状为 (n_samples,) 或 (n_samples, n_targets)
    alpha (float, 可选): 正则化参数，默认值为 0.0（无正则化）
    solver (str, 可选): 求解器类型，支持 'pinv'（默认）、'cholesky' 和 'svd'

    返回:
    np.ndarray: 优化后的权重向量，形状为 (n_features,) 或 (n_features, n_targets)
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
        s_reg = s / (s ** 2 + alpha)
        S_reg = np.zeros((n_features, n_samples))
        np.fill_diagonal(S_reg, s_reg)
        w = Vt.T @ S_reg @ U.T @ y
    else:
        raise ValueError(
            f"不支持的求解器: {solver}，支持的选项有 'pinv', 'cholesky', 'svd'"
        )
    return w


def gradient_descent(phi, y, lr=0.01, epochs=1000):
    """实现批量梯度下降算法优化线性回归权重
    参数:
        phi: 设计矩阵（特征矩阵），形状为 (n_samples, n_features)
        y: 目标值向量，形状为 (n_samples,)
        lr: 学习率（步长），控制参数更新幅度，默认0.01
        epochs: 训练轮数，默认1000
    返回:
        w: 优化后的权重向量，形状为 (n_features,)
    """
    w = np.zeros(phi.shape[1])
    for epoch in range(epochs):
        y_pred = phi @ w
        error = y - y_pred
        gradient = -2 * phi.T @ error / len(y)
        w -= lr * gradient
    return w


# ========== 新增：特征标准化 ==========
def standardize_features(X):
    """
    将特征矩阵标准化为均值为0，标准差为1。

    参数:
        X : numpy.ndarray, shape (n_samples, n_features)
            输入特征矩阵。

    返回:
        X_scaled : numpy.ndarray, shape same as X
            标准化后的特征矩阵。
        mean : numpy.ndarray, shape (n_features,)
            每个特征的均值。
        std : numpy.ndarray, shape (n_features,)
            每个特征的标准差。
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # 避免除以零：如果标准差为0，则标准化后仍为0
    std = np.where(std == 0, 1, std)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


# ========== 新增：回归评估指标 ==========
def regression_metrics(y_true, y_pred):
    """
    计算回归评估指标：MSE, MAE, R²

    参数:
        y_true : numpy.ndarray, shape (n_samples,)
            真实值。
        y_pred : numpy.ndarray, shape (n_samples,)
            预测值。

    返回:
        metrics : dict
            包含 'MSE', 'MAE', 'R2' 的字典。
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))   # 加小量避免除零
    return {'MSE': mse, 'MAE': mae, 'R2': r2}


# ========== 修改：main 函数增加标准化选项 ==========
def main(x_train, y_train, use_gradient_descent=False, basis_func=None, use_standardization=False):
    """训练模型，并返回从x到y的映射。
    basis_func: 可选，基函数（如identity_basis, multinomial_basis, gaussian_basis），默认恒等基
    use_standardization: 是否对基函数特征进行标准化（不包括偏置列）
    """
    if basis_func is None:
        basis_func = identity_basis

    # 生成偏置项和特征矩阵
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)
    phi1 = basis_func(x_train)
    
    # 如果需要标准化，对 phi1 进行标准化（保留 phi0 不变）
    std_mean = (None, None)  # 用于返回标准化参数
    if use_standardization:
        phi1_scaled, mean1, std1 = standardize_features(phi1)
        phi1 = phi1_scaled
        std_mean = (mean1, std1)
    
    phi = np.concatenate([phi0, phi1], axis=1)

    # 最小二乘法求解权重
    w_lsq = least_squares(phi, y_train)

    w_gd = None
    if use_gradient_descent:
        w_gd = gradient_descent(phi, y_train, lr=0.01, epochs=1000)

    def f(x, std_mean_params=None):
        # 预测时的特征处理
        phi0_pred = np.expand_dims(np.ones_like(x), axis=1)
        phi1_pred = basis_func(x)
        if std_mean_params is not None and use_standardization:
            mean1, std1 = std_mean_params
            # 使用训练集上的均值和标准差进行标准化
            phi1_pred = (phi1_pred - mean1) / std1
        phi_pred = np.concatenate([phi0_pred, phi1_pred], axis=1)
        if use_gradient_descent and w_gd is not None:
            return np.dot(phi_pred, w_gd)
        else:
            return np.dot(phi_pred, w_lsq)

    # 为了让预测函数能够访问标准化参数，将参数绑定为闭包变量
    # 这里返回一个包装函数，使得调用 f(x) 时可自动使用训练时的参数
    if use_standardization:
        mean1, std1 = std_mean
        def predict_func(x):
            return f(x, (mean1, std1))
        return predict_func, w_lsq, w_gd
    else:
        def predict_func(x):
            return f(x, None)
        return predict_func, w_lsq, w_gd


def evaluate(ys, ys_pred):
    """评估模型：计算标准差（保留原有函数）"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std


def plot_results(x_train, y_train, x_test, y_test, y_test_pred):
    """绘制训练集、测试集和预测结果"""
    plt.plot(x_train, y_train, "ro", markersize=3)
    plt.plot(x_test, y_test, "k")
    plt.plot(x_test, y_test_pred, "k")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.legend(["train", "test", "pred"])
    plt.show()


# 程序主入口
if __name__ == "__main__":
    # 定义训练和测试数据文件路径
    train_file = "train.txt"
    test_file = "test.txt"
    
    # 载入数据
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    print("训练集形状:", x_train.shape)
    print("测试集形状:", x_test.shape)

    # 可以选择是否使用标准化（此处设置为 True 演示效果，可改为 False 对比）
    # use_std = True   # 启用标准化
    use_std = False  # 关闭标准化

    f, w_lsq, w_gd = main(x_train, y_train, use_gradient_descent=False,
                          basis_func=gaussian_basis, use_standardization=use_std)

    # 训练集预测
    y_train_pred = f(x_train)
    train_std = evaluate(y_train, y_train_pred)
    print(f"训练集标准差 (原有指标): {train_std:.4f}")
    
    # 测试集预测
    y_test_pred = f(x_test)
    test_std = evaluate(y_test, y_test_pred)
    print(f"测试集标准差 (原有指标): {test_std:.4f}")
    
    # 新增：计算更多评估指标
    train_metrics = regression_metrics(y_train, y_train_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)
    
    print("\n=== 训练集评估指标 ===")
    print(f"MSE: {train_metrics['MSE']:.4f}")
    print(f"MAE: {train_metrics['MAE']:.4f}")
    print(f"R² : {train_metrics['R2']:.4f}")
    
    print("\n=== 测试集评估指标 ===")
    print(f"MSE: {test_metrics['MSE']:.4f}")
    print(f"MAE: {test_metrics['MAE']:.4f}")
    print(f"R² : {test_metrics['R2']:.4f}")
    
    print("\n最小二乘法权重 (前10个):", w_lsq[:10])   # 只打印前10个避免过多
    if w_gd is not None:
        print("梯度下降法权重 (前10个):", w_gd[:10])
    
    # 绘图
    plot_results(x_train, y_train, x_test, y_test, y_test_pred)