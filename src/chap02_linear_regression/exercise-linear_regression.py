#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


#  数据加载函数 
def load_data(filename):
    """从文件加载训练/测试数据
    
    Args:
        filename: 数据文件路径
        
    Returns:
        tuple: (特征数组, 标签数组)
    """
    xys = []
    with open(filename, "r") as f:
        for line in f:
            # 分割每行数据并转换为浮点数
            line_data = list(map(float, line.strip().split()))
            xys.append(line_data)
    
    # 解压为特征和标签数组
    xs, ys = zip(*xys)
    return np.asarray(xs), np.asarray(ys)


#  基函数定义 
def identity_basis(x):
    """恒等基函数（不做特征变换）
    
    Args:
        x: 输入特征向量
        
    Returns:
        ndarray: 形状为(n_samples, 1)的数组
    """
    return np.expand_dims(x, axis=1)


def multinomial_basis(x, feature_num=10):
    """多项式基函数（生成多项式特征）
    
    Args:
        x: 输入特征向量
        feature_num: 生成的特征数量（最高次项次数）
        
    Returns:
        ndarray: 形状为(n_samples, feature_num)的数组
    """
    x = np.expand_dims(x, axis=1)  # 转换为列向量
    # 生成x^1到x^feature_num的特征
    features = [x**i for i in range(1, feature_num+1)]
    return np.concatenate(features, axis=1)


def gaussian_basis(x, feature_num=10):
    """高斯基函数（径向基函数）
    
    Args:
        x: 输入特征向量
        feature_num: 高斯函数的数量
        
    Returns:
        ndarray: 形状为(n_samples, feature_num)的数组
    """
    # 在输入范围内均匀分布中心点
    centers = np.linspace(0, 25, feature_num)
    # 计算带宽（标准差）
    sigma = 25 / feature_num  
    # 计算高斯响应
    return np.exp(-0.5 * ((x[:, np.newaxis] - centers) / sigma) ** 2)


#  优化方法 
def least_squares(phi, y, alpha=0.0, solver="pinv"):
    """带正则化的最小二乘法
    
    Args:
        phi: 设计矩阵 (n_samples, n_features)
        y: 目标值 (n_samples,)
        alpha: L2正则化系数
        solver: 求解器类型 ('pinv'|'cholesky'|'svd')
        
    Returns:
        ndarray: 优化后的权重向量
        
    Raises:
        ValueError: 当输入无效或求解器不支持时
    """
    # 输入验证
    if phi.size == 0 or y.size == 0:
        raise ValueError("输入矩阵不能为空")
    if phi.shape[0] != y.shape[0]:
        raise ValueError("样本数量不匹配")

    n_samples, n_features = phi.shape

    # 选择求解器
    if solver == "pinv":
        # 伪逆求解（数值稳定）
        A = phi.T @ phi + alpha * np.eye(n_features)
        w = np.linalg.pinv(A) @ phi.T @ y
        
    elif solver == "cholesky":
        # Cholesky分解（高效但要求矩阵正定）
        if alpha < 0:
            raise ValueError("正则化系数必须非负")
        A = phi.T @ phi + alpha * np.eye(n_features)
        try:
            L = np.linalg.cholesky(A)
            z = np.linalg.solve(L, phi.T @ y)
            w = np.linalg.solve(L.T, z)
        except np.linalg.LinAlgError:
            print("警告: Cholesky分解失败，回退到伪逆")
            w = np.linalg.pinv(A) @ phi.T @ y
            
    elif solver == "svd":
        # SVD分解（最稳定但计算量大）
        U, s, Vt = np.linalg.svd(phi, full_matrices=False)
        s_reg = s / (s**2 + alpha)
        S_reg = np.zeros((n_features, n_samples))
        np.fill_diagonal(S_reg, s_reg)
        w = Vt.T @ S_reg @ U.T @ y
        
    else:
        raise ValueError(f"不支持的求解器: {solver}")

    return w


def gradient_descent(phi, y, lr=0.01, epochs=1000):
    """批量梯度下降优化
    
    Args:
        phi: 设计矩阵
        y: 目标值
        lr: 学习率
        epochs: 迭代次数
        
    Returns:
        ndarray: 优化后的权重向量
    """
    w = np.zeros(phi.shape[1])
    for _ in range(epochs):
        y_pred = phi @ w
        gradient = phi.T @ (y_pred - y) / len(y)
        w -= lr * gradient
    return w


#  主函数 
def main(x_train, y_train, use_gradient_descent=False):
    """训练线性回归模型
    
    Args:
        x_train: 训练特征
        y_train: 训练标签
        use_gradient_descent: 是否使用梯度下降
        
    Returns:
        tuple: (预测函数, 最小二乘权重, 梯度下降权重)
    """
    # 使用恒等基函数
    basis_func = identity_basis  

    # 构造设计矩阵 [1, phi(x)]
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)  # 偏置项
    phi1 = basis_func(x_train)  # 基函数变换
    phi = np.concatenate([phi0, phi1], axis=1)

    # 最小二乘法求解
    w_lsq = least_squares(phi, y_train)
    
    # 梯度下降求解
    w_gd = None
    if use_gradient_descent:
        w_gd = gradient_descent(phi, y_train, lr=0.001, epochs=5000)

    # 定义预测函数
    def predict(x):
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_func(x)
        phi = np.concatenate([phi0, phi1], axis=1)
        return phi @ (w_gd if use_gradient_descent and w_gd is not None else w_lsq)

    return predict, w_lsq, w_gd


# 评估函数 
def evaluate(ys, ys_pred):
    """评估模型性能
    
    Args:
        ys: 真实值
        ys_pred: 预测值
        
    Returns:
        float: RMSE评估指标
    """
    return np.sqrt(np.mean((ys - ys_pred) ** 2))


#  主程序 
if __name__ == "__main__":
    # 数据加载
    train_file = "train.txt"
    test_file = "test.txt"
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    
    print(f"训练集形状: {x_train.shape}")
    print(f"测试集形状: {x_test.shape}")

    # 模型训练
    model, w_lsq, w_gd = main(x_train, y_train, use_gradient_descent=True)
    
    # 训练集评估
    y_train_pred = model(x_train)
    train_rmse = evaluate(y_train, y_train_pred)
    print(f"训练集RMSE: {train_rmse:.1f}")
    
    # 测试集评估
    y_test_pred = model(x_test)
    test_rmse = evaluate(y_test, y_test_pred)
    print(f"测试集RMSE: {test_rmse:.1f}")

    # 结果可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, c='red', s=5, label="训练数据")
    plt.plot(x_test, y_test, 'k-', label="真实值")
    plt.plot(x_test, y_test_pred, 'b--', label="预测值")
    
    plt.xlabel("特征 x")
    plt.ylabel("目标值 y")
    plt.title("线性回归结果")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
