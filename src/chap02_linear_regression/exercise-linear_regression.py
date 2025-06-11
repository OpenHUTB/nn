#!/usr/bin/env python
# coding: utf-8
import numpy as np # 导入NumPy库
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt # 导入Matplotlib的pyplot模块并命名为plt
# 用于创建各种静态、交互式和动画可视化图表

# 下面这段代码从文件中读取数据，然后把数据拆分成特征和标签，最后以 NumPy 数组的形式返回
def load_data(filename):
    """载入数据。
    Args:
        filename: 数据文件的路径
    Returns:
        tuple: 包含特征和标签的numpy数组 (xs, ys)
    """
    xys = []# 用于存储每行的数据，每行数据是一个列表
    with open(filename, "r") as f:  # 打开文件进行读取
        for line in f: # 遍历文件的每一行
            # 将每行内容按空格分割并转换为浮点数
            # strip() 去除行首尾的空白字符，split() 按空格分割字符串
            # map(float, ...) 将分割后的字符串转换为浮点数
            line_data = list(map(float, line.strip().split()))
            xys.append(line_data)
    # 将数据拆分为特征和标签
    # 假设每行数据的最后一个元素是标签，其余是特征
    # zip(*xys) 将 xys 列表的行和列进行转置
    xs, ys = zip(*xys)# xs 是特征列表，ys 是标签列表
    # 将特征和标签列表转换为 NumPy 数组
    # NumPy 数组便于后续的数学运算和数据处理
    return np.asarray(xs), np.asarray(ys) # 返回特征和标签的 NumPy 数组


# ## 恒等基函数（Identity Basis Function）的实现 填空顺序 2
def identity_basis(x):
    # 在 x 的最后一个维度上增加一个维度，将其转换为二维数组
    # 用于适配线性回归的矩阵运算格式
    # 通过 np.expand_dims，将 x 转换为列向量的形式，形状变为 (len(x), 1)
    return np.expand_dims(x, axis=1)


# 请分别在这里实现"多项式基函数"（Multinomial Basis Function）以及"高斯基函数"（Gaussian Basis Function）

# 其中以及训练集的x的范围在0-25之间
def multinomial_basis(x, feature_num=10):
    """多项式基函数：将输入x映射为多项式特征
    feature_num: 多项式的最高次数
    返回 shape (N, feature_num)"""
    # 在 x 的最后一个维度上增加一个维度，将其转换为二维数组
    x = np.expand_dims(x, axis=1)  # shape(N, 1)
    # 可以替换成 x = identity_basis(x)
    # ==========
    # todo '''请实现多项式基函数'''
    # 在 x 的最后一个维度上增加一个维度，将其转换为三维数组
    # 通过列表推导式创建各次项，最后在列方向拼接合并
    x = np.expand_dims(x, axis=1)  # shape(N, 1)
    # 生成 x, x^2, ..., x^(feature_num)
    ret = [x**i for i in range(1, feature_num + 1)]
    # 将生成的列表合并成 shape(N, feature_num) 的二维数组
    # ==========
    return np.concatenate(ret, axis=1)


def gaussian_basis(x, feature_num=10):
    """
    高斯基函数：将输入x映射为一组高斯分布特征
    用于提升模型对非线性关系的拟合能力
    """
    # 定义中心在区间 [0, 25] 内均匀分布
    centers = np.linspace(0, 25, feature_num)
    # 每个高斯函数的标准差（带宽）
    sigma = 25 / feature_num
    # 计算每个输入 x 对所有中心的响应，输出 shape (N, feature_num)
    return np.exp(-0.5 * ((x[:, np.newaxis] - centers) / sigma) ** 2)


# 返回一个训练好的模型 填空顺序 1 用最小二乘法进行模型优化
# ## 填空顺序 3 用梯度下降进行模型优化
# > 先完成最小二乘法的优化 (参考书中第二章 2.3中的公式)
#
# > 再完成梯度下降的优化   (参考书中第二章 2.3中的公式)
#
# 在main中利用训练集训练好模型的参数，并且返回一个训练好的模型。
#
# 计算出一个优化后的w，请分别使用最小二乘法以及梯度下降两种办法优化w


def least_squares(phi, y, alpha=0.0, solver="pinv"):
    """
    带正则化的最小二乘法优化，支持多种求解器

    参数:
    phi (np.ndarray): 设计矩阵，形状为 (n_samples, n_features)
    y (np.ndarray): 目标值，形状为 (n_samples,) 或 (n_samples, n_targets)
    alpha (float, 可选): 正则化参数，默认值为 0.0（无正则化）
    solver (str, 可选): 求解器类型，支持 'pinv'（默认）、'cholesky' 和 'svd'

    返回:
    np.ndarray: 优化后的权重向量，形状为 (n_features,) 或 (n_features, n_targets)

    异常:
    ValueError: 当 solver 参数不是支持的类型时抛出
    """
    # 检查输入矩阵是否为空
    if phi.size == 0 or y.size == 0: # 如果矩阵 phi 或 y 是空的，抛出 ValueError 异常
        raise ValueError("输入矩阵 phi 和目标值 y 不能为零矩阵")

    # 检查维度是否兼容
    if phi.shape[0] != y.shape[0]:
        raise ValueError(
            f"设计矩阵 phi 的样本数 ({phi.shape[0]}) 与目标值 y 的样本数 ({y.shape[0]}) 不匹配"
        )

    n_samples, n_features = phi.shape

    # 根据选择的求解器执行计算
    if solver == "pinv":
        # 使用 numpy 的伪逆函数，基于 SVD 分解
        # 对病态矩阵具有良好的数值稳定性
        A = phi.T @ phi + alpha * np.eye(n_features)
        w = np.linalg.pinv(A) @ phi.T @ y

    elif solver == "cholesky":
        # 使用 Cholesky 分解求解正规方程
        # 计算效率高，但要求矩阵正定（alpha > 0 时保证）
        if alpha < 0:
            raise ValueError("使用 Cholesky 求解器时，正则化参数 alpha 必须为非负数")
        A = phi.T @ phi + alpha * np.eye(n_features)
        try:
            L = np.linalg.cholesky(A)
            # 解 L*z = phi.T @ y
            z = np.linalg.solve(L, phi.T @ y)
            # 解 L^T*w = z
            w = np.linalg.solve(L.T, z)
        except np.linalg.LinAlgError:
            # 处理非正定矩阵的情况，回退到 pinv
            print("警告: Cholesky 分解失败，矩阵可能非正定，回退到伪逆求解")
            w = np.linalg.pinv(A) @ phi.T @ y

    elif solver == "svd":
        # 直接使用 SVD 分解求解
        # 对病态矩阵最稳定，但计算成本较高
        U, s, Vt = np.linalg.svd(phi, full_matrices=False)
        # 计算正则化的 SVD 解
        s_reg = s / (s**2 + alpha)
        # 构建对角矩阵
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
    数学原理:
        最小化损失函数 J(w) = 1/m * ||φw - y||²
        梯度计算: ∇J(w) = 2/m * φ.T @ (φw - y)
        参数更新: w := w - α * ∇J(w)
    """
    # 初始化权重向量（全零开始）
    # 形状与特征数量相同，即每个特征对应一个权重
    w = np.zeros(phi.shape[1])
    
    # 迭代优化循环
    for epoch in range(epochs):
        # 1. 前向传播：计算当前权重下的预测值
        # 矩阵乘法 φw，结果形状 (n_samples,)
        y_pred = phi @ w
        
        # 2. 计算误差：预测值与真实值的差
        # 形状 (n_samples,)
        error = y - y_pred
        
        # 3. 计算梯度（损失函数对权重的导数）
        # 梯度公式推导：
        #   J(w) = 1/m * ∑(φw - y)²
        #   ∇J(w) = 2/m * φ.T @ (φw - y)
        # 其中：
        # φ.T @ error 计算每个特征上的误差总和
        # -2/len(y) 是损失函数导数的系数
        # 最终形状 (n_features,)
        gradient = -2 * phi.T @ error / len(y)
        
        # 4. 参数更新：沿负梯度方向调整权重
        # 学习率控制更新步长
        # 公式: w_new = w_old - lr * ∇J(w)
        w -= lr * gradient
    
    return w


def main(x_train, y_train, use_gradient_descent=False):
    """训练模型，并返回从x到y的映射。
    流程:
    1. 选择基函数（默认恒等基）
    2. 构建设计矩阵 φ = [1, basis_func(x)]
    3. 使用最小二乘或梯度下降求解权重 w
    4. 返回预测函数 f(x) = φ(x) · w
    """
    # 默认使用恒等基函数
    basis_func = identity_basis  

    # 生成偏置项和特征矩阵
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)
    phi1 = basis_func(x_train)
    phi = np.concatenate([phi0, phi1], axis=1)

    # 最小二乘法求解权重
    w_lsq = np.dot(np.linalg.pinv(phi), y_train)

    w_gd = None
    if use_gradient_descent:
        # 梯度下降求解权重（缩进修正）
        # 设置学习率为0.01
        learning_rate = 0.01 
        # 设置训练轮数(epochs)为1000，表示整个训练数据集将被遍历1000次。
        epochs = 1000  
        # 初始化梯度下降法使用的权重向量
        w_gd = np.zeros(phi.shape[1])
        w_gd = gradient_descent(phi, y_train, lr=0.001, epochs=5000)
        # 开始梯度下降的迭代循环，将进行epochs次参数更新。
        for epoch in range(epochs):     # 梯度下降循环
            y_pred = np.dot(phi, w_gd)  # 计算预测值
            error = y_pred - y_train    # 计算误差
            gradient = np.dot(phi.T, error) / len(y_train) # 计算梯度
            w_gd -= learning_rate * gradient # 更新权重

   # 定义预测函数
    def f(x):
    """
    使用线性模型进行预测，可选择使用梯度下降或最小二乘法的权重
    
    参数:
        x: 输入特征向量/矩阵
        
    返回:
        模型的预测值
    """
    # 创建偏置项(截距项)的特征列
    # np.ones_like(x)生成与x形状相同的全1向量
    # np.expand_dims增加一个维度，将形状从(n,)变为(n,1)
    phi0 = np.expand_dims(np.ones_like(x), axis=1)  # 形状变为(n,1)
    
    # 调用基函数转换输入特征
    # basis_func对原始特征进行非线性变换(如多项式扩展)
    phi1 = basis_func(x)  # 假设返回形状为(n,m)
    
    # 构建设计矩阵(特征矩阵)
    # 将偏置项列phi0和转换后的特征phi1按列拼接
    phi = np.concatenate([phi0, phi1], axis=1)  # 最终形状(n,m+1)
    
    # 根据训练方法选择对应的权重进行预测
    if use_gradient_descent and w_gd is not None:
        # 使用梯度下降法训练得到的权重
        return np.dot(phi, w_gd)  # 矩阵乘法计算预测值
    else:
        # 使用最小二乘法训练得到的权重(默认)
        return np.dot(phi, w_lsq)  # 矩阵乘法计算预测值

    # 确保返回值为可迭代对象
    return f, w_lsq, w_gd


# ## 评估结果
# ## 评估结果
# > 没有需要填写的代码，但是建议读懂


def evaluate(ys, ys_pred):
    """评估模型。"""
    # 计算预测值与真实值的标准差
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std


# 程序主入口（建议不要改动以下函数的接口）
if __name__ == "__main__":
    # 定义训练和测试数据文件路径
    train_file = "train.txt"  # 训练集文件
    test_file = "test.txt"  # 测试集文件
    # 载入数据
    x_train, y_train = load_data(
        train_file
    )  # 从文件加载训练数据，返回特征矩阵x_train和标签向量y_train
    x_test, y_test = load_data(
        test_file
    )  # 从文件加载测试数据，返回特征矩阵x_test和标签向量y_test
    print(x_train.shape)  # x_train.shape 返回训练集特征矩阵的维度信息
    print(x_test.shape)  # x_test.shape 返回测试集特征矩阵的维度信息

    # 使用线性回归训练模型，返回一个函数 f() 使得 y = f(x)
    # f: 预测函数 y = f(x)
    # w_lsq: 通过最小二乘法得到的权重向量
    # w_gd: 通过梯度下降法得到的权重向量
    f, w_lsq, w_gd = main(x_train, y_train)

    y_train_pred = f(x_train)  # 对训练数据应用预测函数
    std = evaluate(y_train, y_train_pred)  # 计算预测值与真实值的标准差作为评估指标
    print("训练集预测值与真实值的标准差：{:.1f}".format(std))

    # 计算预测的输出值
    y_test_pred = f(x_test)
    # 使用测试集评估模型
    std = evaluate(y_test, y_test_pred)
    print("预测值与真实值的标准差：{:.1f}".format(std))

    # 显示结果

    plt.plot(x_train, y_train, "ro", markersize=3)  # 红色点为训练集数据
    plt.plot(x_test, y_test, "k")                   # 红色点为训练集数据
    plt.plot(x_test, y_test_pred, "k")              # 黑线为预测值（可以用其他颜色区分）
    plt.xlabel("x")                                 # 设置x轴的标签
    plt.ylabel("y")                                 # 设置y轴的标签
    plt.title("Linear Regression")                  # 设置图表标题
    plt.legend(["train", "test", "pred"])           # 添加图例，表示每条线的含义
    plt.show()                                      # 显示图表
