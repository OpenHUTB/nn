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
    with open(filename, 'r') as f:
        xys = [list(map(float, line.strip().split())) for line in f]
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)

# ## 不同的基函数 (basis function)的实现 填空顺序 2
# 
# 请分别在这里实现“多项式基函数”以及“高斯基函数”
# 
# 其中以及训练集的x的范围在0-25之间

# In[6]:
def identity_basis(x):
    return np.expand_dims(x, 1)

def multinomial_basis(x, feature_num=10):
    x = np.expand_dims(x, 1)
    exponents = np.arange(feature_num)  # 从x^0到x^9
    return x[:, None] ** exponents       # 广播计算多项式

def gaussian_basis(x, feature_num=10):
    centers = np.linspace(0, 25, feature_num)
    widths = (25 / feature_num) * np.ones(feature_num)
    return np.exp(-((x[:,None]-centers)/widths)**2/2)

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
    return np.linalg.inv(phi.T @ phi + alpha*np.eye(phi.shape[1])) @ phi.T @ y

def gradient_descent(phi, y, lr=0.1, epochs=1000):
    w = np.zeros(phi.shape[1])
    for _ in range(epochs):
        grad = 2 * phi.T @ (phi @ w - y)
        w -= lr * grad / len(y)
    return w

def main(x_train, y_train, use_gradient_descent=False):
    basis_func = identity_basis
    phi = np.concatenate([np.ones((len(x_train),1)), basis_func(x_train)], axis=1)
    
    if not use_gradient_descent:
        w = least_squares(phi, y_train)
    else:
        w = gradient_descent(phi, y_train, lr=0.01, epochs=10000)
    
    def predictor(x):
        phi_pred = np.concatenate([np.ones((len(x),1)), basis_func(x)], axis=1)
        return phi_pred @ w
    return predictor, w

# ## 评估结果 
# > 没有需要填写的代码，但是建议读懂

# In[ ]:
def evaluate(ys, ys_pred):
    return np.sqrt(np.mean((ys - ys_pred)**2))

if __name__ == '__main__':
    x_train, y_train = load_data('train.txt')
    x_test, y_test = load_data('test.txt')
    
    # 使用最小二乘法
    f_lsq, w_lsq = main(x_train, y_train, use_gradient_descent=False)
    y_train_pred = f_lsq(x_train)
    print('LSQ训练集RMSE:', evaluate(y_train, y_train_pred))
    
    # 使用梯度下降
    f_gd, w_gd = main(x_train, y_train, use_gradient_descent=True)
    y_train_pred = f_gd(x_train)
    print('GD训练集RMSE:', evaluate(y_train, y_train_pred))
    
    # 测试集评估
    for f, name in zip([f_lsq, f_gd], ['LSQ', 'GD']):
        y_test_pred = f(x_test)
        print(f'{name}测试集RMSE:', evaluate(y_test, y_test_pred))
        
    # 绘图
    plt.figure(figsize=(12,6))
    plt.scatter(x_train, y_train, c='r', label='Train', s=10)
    plt.scatter(x_test, y_test, c='k', label='Test', s=10)
    xs = np.linspace(0,25,200)
    for f, style, label in zip([f_lsq, f_gd], ['b--', 'g-'], ['LSQ', 'GD']):
        plt.plot(xs, f(xs), style, label=label)
    plt.legend()
    plt.show()