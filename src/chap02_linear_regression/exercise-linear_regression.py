#!/usr/bin/env python
# coding: utf-8

# ## 说明
# 
# 请按照填空顺序编号分别完成 参数优化，不同基函数的实现

# In[1]:
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(map(float, line.strip().split()))
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)


# ## 不同的基函数 (basis function)的实现 填空顺序 2
# 
# 请分别在这里实现“多项式基函数”以及“高斯基函数”
# 
# 其中以及训练集的x的范围在0-25之间

# In[6]:
def identity_basis(x):
    ret = np.expand_dims(x, axis=1)
    return ret

def multinomial_basis(x, feature_num=10):
    '''多项式基函数'''
    x = np.expand_dims(x, axis=1) # shape(N, 1)
    #==========
    #todo '''请实现多项式基函数'''
    x = np.expand_dims(x, axis=1) # shape(N, 1)
    ret = [x**i for i in range(1, feature_num+1)]  # 生成 1, x, x^2, ..., x^(feature_num-1)
    ret = np.concatenate(ret, axis=1)  # 合并成 shape(N, feature_num)
    #==========
    return ret

def gaussian_basis(x, feature_num=10):
    '''高斯基函数'''
    #==========
    #todo '''请实现高斯基函数'''
    centers = np.linspace(0, 25, feature_num)  # 在 0 到 25 之间均匀分布中心
    widths = np.ones(feature_num) * (25 / feature_num)  # 每个基函数的宽度
    ret = np.exp(-0.5 * ((x[:, np.newaxis] - centers) / widths)**2)  # shape(N, feature_num)
    #==========
    return ret


# ## 返回一个训练好的模型 填空顺序 1 用最小二乘法进行模型优化 
# ## 填空顺序 3 用梯度下降进行模型优化
# > 先完成最小二乘法的优化 (参考书中第二章 2.3中的公式)
# 
# > 再完成梯度下降的优化   (参考书中第二章 2.3中的公式)
# 
# 在main中利用训练集训练好模型的参数，并且返回一个训练好的模型。
# 
# 计算出一个优化后的w，请分别使用最小二乘法以及梯度下降两种办法优化w

# In[7]:
def least_squares(phi, y, alpha=0.0):
    """最小二乘法优化"""
    w = np.linalg.pinv(phi.T @ phi + alpha * np.eye(phi.shape[1])) @ phi.T @ y  # 使用伪逆更稳定
    return w
def gradient_descent(phi, y, lr=0.01, epochs=1000):
    """梯度下降优化"""
    w = np.zeros(phi.shape[1])  # 初始化 w
    for epoch in range(epochs):
        y_pred = phi @ w
        gradient = -2 * phi.T @ (y - y_pred) / len(y)  # 计算梯度
        w -= lr * gradient  # 更新 w
    return w

def main(x_train, y_train, use_gradient_descent=False):
    """
    训练模型，并返回从x到y的映射。
    
    """
    basis_func = identity_basis
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)
    phi1 = basis_func(x_train)
    phi = np.concatenate([phi0, phi1], axis=1)
    
    
    #==========
    #todo '''计算出一个优化后的w，请分别使用最小二乘法以及梯度下降两种办法优化w'''
    #最小二乘法
    #通过 np.linalg.pinv(phi) 计算伪逆矩阵来求解 w
    w_lsq = np.dot(np.linalg.pinv(phi), y_train)

    #梯度下降（使用时取消注释）
    learning_rate=0.01,
    epochs=1000
    w_gd = np.zeros(phi.shape[1])
        
    for epoch in range(epochs):
     y_pred = np.dot(phi, w)
    error = y_pred - y_train
    gradient = np.dot(phi.T, error) / len(y_train)
    w_gd -= learning_rate * gradient
    #==========
    
    def f(x):
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_func(x)
        phi = np.concatenate([phi0, phi1], axis=1)
        y = np.dot(phi, w)
        if 'w_lsq' in locals() or 'w_lsq' in globals():  # 使用最小二乘法得到的w进行预测
            return np.dot(phi, w_lsq)
        else:  # 使用梯度下降得到的w进行预测（需要确保w_gd已在当前作用域内定义）
            return np.dot(phi, w_gd)
# 这里我们默认返回使用最小二乘法训练的模型，如果您想使用梯度下降训练的模型，
    # 请将下面的'w_lsq'替换为'w_gd'，并确保在调用此函数之前已经通过梯度下降训练了模型。
    return f, w_lsq, w_gd  # 返回预测函数以及两种方法得到的权重w（可选）


# ## 评估结果 
# > 没有需要填写的代码，但是建议读懂

# In[ ]:


def evaluate(ys, ys_pred):
    """评估模型。"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std

# 程序主入口（建议不要改动以下函数的接口）
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

    #显示结果
    plt.plot(x_train, y_train, 'ro', markersize=3)
#     plt.plot(x_test, y_test, 'k')
    plt.plot(x_test, y_test_pred, 'k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend(['train', 'test', 'pred'])
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




