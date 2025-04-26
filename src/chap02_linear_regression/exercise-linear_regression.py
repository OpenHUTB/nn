#!/usr/bin/env python
# coding: utf-8

# ## 说明
# 
# 本代码实现基于基函数的回归模型训练与评估，包含以下功能：
# 1. 数据加载
# 2. 三种基函数实现（恒等基、多项式基、高斯基）
# 3. 模型参数优化（最小二乘法与梯度下降）
# 4. 模型评估与可视化
# 
# 请按以下顺序完成填空：
# (1) 完成最小二乘法优化
# (2) 实现多项式基函数和高斯基函数
# (3) 实现梯度下降优化

# In[1]:
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """载入数据文件，返回特征x和标签y的numpy数组。
    
    Args:
        filename (str): 数据文件路径
    
    Returns:
        tuple: (x, y) 其中x是形状为(n_samples,)的特征数组，y是标签数组
    """
    with open(filename, 'r') as f:
        xys = [list(map(float, line.strip().split())) for line in f]
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)

# ## 不同的基函数 (basis function)的实现 填空顺序 2
# 
# 请实现以下基函数：
# 1. **恒等基函数**：直接返回x的列向量（用于线性回归）
# 2. **多项式基函数**：生成多项式特征（x^0, x^1, ..., x^9）
# 3. **高斯基函数**：生成高斯核函数（中心均匀分布于0-25区间）

# In[6]:
def identity_basis(x):
    """恒等基函数：将输入x转换为一维列向量。
    
    Args:
        x (array): 形状为(n_samples,)的输入数组
    
    Returns:
        array: 形状为(n_samples,1)的列向量
    """
    return np.expand_dims(x, 1)

def multinomial_basis(x, feature_num=10):
    """多项式基函数：生成多项式特征矩阵。
    
    Args:
        x (array): 形状为(n_samples,)的输入数组
        feature_num (int): 特征维度（默认10）
    
    Returns:
        array: 形状为(n_samples, feature_num)的多项式特征矩阵
    """
    x = np.expand_dims(x, 1)          # 转换为列向量
    exponents = np.arange(feature_num) # 指数范围0-9
    return x[:, None] ** exponents     # 广播计算x^0到x^9

def gaussian_basis(x, feature_num=10):
    """高斯基函数：生成高斯核特征矩阵。
    
    Args:
        x (array): 形状为(n_samples,)的输入数组
        feature_num (int): 高斯中心数量（默认10）
    
    Returns:
        array: 形状为(n_samples, feature_num)的高斯特征矩阵
    """
    centers = np.linspace(0, 25, feature_num)  # 在0-25区间均匀分布中心点
    widths = (25 / feature_num) * np.ones(feature_num)  # 宽度设置为区间长度/中心数
    # 计算高斯核：exp(-((x-c)/w)^2/2)
    return np.exp(-((x[:, None] - centers)/widths)**2 / 2)

# ## 返回一个训练好的模型 填空顺序 1（最小二乘法）和 3（梯度下降）
# 
# 需要实现两种参数优化方法：
# 1. **最小二乘法**：直接求解线性回归的解析解（参考公式：w = (Φ^TΦ)^{-1}Φ^T y）
# 2. **梯度下降**：通过迭代优化求解权重（参考梯度公式：dw = 2Φ^T(Φw - y)）

# In[7]:
def least_squares(phi, y, alpha=0.0):
    """最小二乘法求解权重参数。
    
    Args:
        phi (array): 形状(n_samples, n_features)的特征矩阵
        y (array): 标签数组
        alpha (float): L2正则化系数（默认0）
    
    Returns:
        array: 优化后的权重向量w
    """
    # 正则化项：alpha*I
    reg_matrix = alpha * np.eye(phi.shape[1])
    return np.linalg.inv(phi.T @ phi + reg_matrix) @ phi.T @ y

def gradient_descent(phi, y, lr=0.1, epochs=1000):
    """梯度下降法求解权重参数。
    
    Args:
        phi (array): 特征矩阵
        y (array): 标签数组
        lr (float): 学习率（默认0.1）
        epochs (int): 迭代次数（默认1000）
    
    Returns:
        array: 优化后的权重向量w
    """
    w = np.zeros(phi.shape[1])  # 初始化权重
    for _ in range(epochs):
        residuals = phi @ w - y  # 预测残差
        grad = 2 * phi.T @ residuals / len(y)  # 梯度计算
        w -= lr * grad  # 权重更新
    return w

def main(x_train, y_train, use_gradient_descent=False):
    """训练回归模型并返回预测函数。
    
    Args:
        x_train (array): 训练集特征
        y_train (array): 训练集标签
        use_gradient_descent (bool): 是否使用梯度下降（默认False使用最小二乘）
    
    Returns:
        tuple: (predictor函数, 权重w)
    """
    # 当前使用恒等基函数（需要根据需求替换其他基函数）
    basis_func = identity_basis
    # 构建特征矩阵（添加偏置项）
    phi = np.concatenate([np.ones((len(x_train),1)), basis_func(x_train)], axis=1)
    
    # 选择优化方法
    if not use_gradient_descent:
        w = least_squares(phi, y_train)
    else:
        w = gradient_descent(phi, y_train, lr=0.01, epochs=10000)
    
    # 定义预测函数
    def predictor(x):
        phi_pred = np.concatenate([np.ones((len(x),1)), basis_func(x)], axis=1)
        return phi_pred @ w  # 线性组合预测
    return predictor, w

# ## 评估结果 
def evaluate(ys, ys_pred):
    """计算均方根误差（RMSE）。
    
    Args:
        ys (array): 真实标签
        ys_pred (array): 预测值
    
    Returns:
        float: RMSE值
    """
    return np.sqrt(np.mean((ys - ys_pred)**2))

if __name__ == '__main__':
    # 数据加载
    x_train, y_train = load_data('train.txt')
    x_test, y_test = load_data('test.txt')
    
    # 使用最小二乘法训练
    f_lsq, w_lsq = main(x_train, y_train, use_gradient_descent=False)
    y_train_pred = f_lsq(x_train)
    print('LSQ训练集RMSE:', evaluate(y_train, y_train_pred))
    
    # 使用梯度下降训练
    f_gd, w_gd = main(x_train, y_train, use_gradient_descent=True)
    y_train_pred = f_gd(x_train)
    print('GD训练集RMSE:', evaluate(y_train, y_train_pred))
    
    # 测试集评估
    for f, name in zip([f_lsq, f_gd], ['LSQ', 'GD']):
        y_test_pred = f(x_test)
        print(f'{name}测试集RMSE:', evaluate(y_test, y_test_pred))
        
    # 结果可视化
    plt.figure(figsize=(12,6))
    plt.scatter(x_train, y_train, c='r', label='训练集', s=10)
    plt.scatter(x_test, y_test, c='k', label='测试集', s=10)
    xs = np.linspace(0,25,200)
    for f, style, label in zip([f_lsq, f_gd], ['b--', 'g-'], ['最小二乘法', '梯度下降']):
        plt.plot(xs, f(xs), style, label=label)
    plt.title('回归模型对比')
    plt.xlabel('特征x')
    plt.ylabel('预测值y')
    plt.legend()
    plt.show()