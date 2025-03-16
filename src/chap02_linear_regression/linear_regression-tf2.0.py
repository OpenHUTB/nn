#!/usr/bin/env python
# coding: utf-8

# ## 设计基函数(basis function) 以及数据读取

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def identity_basis(x):
    ret = np.expand_dims(x, axis=1)
    return ret


def multinomial_basis(x, feature_num=10):
    x = np.expand_dims(x, axis=1)  # shape(N, 1)
    feat = [x]
    for i in range(2, feature_num + 1):
        feat.append(x ** i)
    ret = np.concatenate(feat, axis=1)
    return ret


def gaussian_basis(x, feature_num=10):
    centers = np.linspace(0, 25, feature_num)
    width = 1.0 * (centers[1] - centers[0])
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x] * feature_num, axis=1)

    out = (x - centers) / width
    ret = np.exp(-0.5 * out ** 2)
    return ret


def load_data(filename, basis_func=gaussian_basis):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(list(map(float, line.strip().split())))
        xs, ys = zip(*xys)
        xs, ys = np.asarray(xs), np.asarray(ys)

        o_x, o_y = xs, ys
        phi0 = np.expand_dims(np.ones_like(xs), axis=1)
        phi1 = basis_func(xs)
        xs = np.concatenate([phi0, phi1], axis=1)
        return (torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)), (o_x, o_y)


# ## 定义模型
class LinearModel(nn.Module):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        self.w = nn.Parameter(torch.randn(ndim, 1) * 0.1)

    def forward(self, x):
        y = torch.squeeze(torch.matmul(x, self.w), dim=1)
        return y


(xs, ys), (o_x, o_y) = load_data('train.txt')
ndim = xs.shape[1]

model = LinearModel(ndim=ndim)

# ## 训练以及评估
optimizer = optim.Adam(model.parameters(), lr=0.1)


def train_one_step(model, xs, ys):
    optimizer.zero_grad()
    y_preds = model(xs)
    loss = torch.mean(torch.sqrt(1e-12 + (ys - y_preds) ** 2))
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, xs):
    model.eval()
    with torch.no_grad():
        y_preds = model(xs)
    return y_preds.numpy()



def evaluate(ys, ys_pred):
    """评估模型。"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std



for i in range(1000):
    loss = train_one_step(model, xs, ys)
    if i % 100 == 1:
        print(f'loss is {loss:.4}')

y_preds = predict(model, xs)
std = evaluate(ys.numpy(), y_preds)
print('训练集预测值与真实值的标准差：{:.1f}'.format(std))

(xs_test, ys_test), (o_x_test, o_y_test) = load_data('test.txt')


y_test_preds = predict(model, xs_test)
std = evaluate(ys_test.numpy(), y_test_preds)
print('测试集预测值与真实值的标准差：{:.1f}'.format(std))

plt.plot(o_x, o_y, 'ro', markersize=3)
plt.plot(o_x_test, y_test_preds, 'k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend(['train data', 'test prediction'])
plt.show()