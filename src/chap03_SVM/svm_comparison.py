#!/usr/bin/env python
# coding: utf-8
"""比较线性分类器 (平方误差)、逻辑回归 (交叉熵) 和 SVM (合页损失)"""

import numpy as np
import os
# import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# ============================================================
# 数据加载与预处理
# ============================================================
def load_data(fname):
    """载入数据。"""
    if not os.path.exists(fname):
        raise FileNotFoundError(f"数据文件未找到: {fname}")
    with open(fname, 'r') as f:
        data = []
        line = f.readline()  # 跳过表头行
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)

def eval_acc(label, pred):
    """计算准确率。"""
    return np.sum(label == pred) / len(pred)

# ============================================================
# 模型实现
# ============================================================

class LinearClassifierComparison:
    """比较不同损失函数的线性分类器。"""
    
    def __init__(self, method='svm', learning_rate=0.01, reg_lambda=0.01, max_iter=1000):
        self.method = method
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.w = None
        self.b = None

    def train(self, X, y):
        """根据指定的损失函数进行训练。"""
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0
        
        # 标签预处理：统一转换为 {-1, 1}
        y_proc = np.where(y <= 0, -1, 1)

        for epoch in range(self.max_iter):
            # 预测分数: score = w*x + b
            score = np.dot(X, self.w) + self.b
            
            if self.method == 'linear':
                # 1. 线性分类器 (平方误差): E = sum(score - y)^2 + lambda*||w||^2
                diff = score - y_proc
                dw = (2 * np.dot(X.T, diff) + 2 * self.reg_lambda * self.w) / m
                db = 2 * np.mean(diff)
                
            elif self.method == 'logistic':
                # 2. 逻辑回归 (交叉熵): E = sum(log(1 + exp(-y*score))) + lambda*||w||^2
                # 损失函数的导数 (w.r.t score): -y / (1 + exp(y*score))
                grad_factor = -y_proc / (1 + np.exp(y_proc * score))
                dw = (np.dot(X.T, grad_factor) + 2 * self.reg_lambda * self.w) / m
                db = np.mean(grad_factor)
                
            elif self.method == 'svm':
                # 3. SVM (合页损失): E = sum(max(0, 1 - y*score)) + lambda*||w||^2
                # 这里的 y 是 -1/1 标签
                margin = y_proc * score
                idx = np.where(margin < 1)[0]
                if len(idx) > 0:
                    dw = (2 * self.reg_lambda * self.w - np.sum(y_proc[idx, None] * X[idx], axis=0)) / m
                    db = -np.mean(y_proc[idx])
                else:
                    dw = (2 * self.reg_lambda * self.w) / m
                    db = 0
            
            # 更新参数
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        """预测类别。"""
        score = np.dot(X, self.w) + self.b
        # 所有模型均以 0 为界限进行分类 (基于 -1/1 训练)
        return (score >= 0).astype(np.int32) * 2 - 1  # 返回 -1 或 1

# ============================================================
# 主程序
# ============================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, 'data', 'train_linear.txt')
    test_file = os.path.join(base_dir, 'data', 'test_linear.txt')
    
    data_train = load_data(train_file)
    data_test = load_data(test_file)
    
    X_train = data_train[:, :2]
    y_train = data_train[:, 2]
    X_test = data_test[:, :2]
    y_test = data_test[:, 2]
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    methods = [
        ('linear', '线性分类器 (平方误差)'),
        ('logistic', '逻辑回归 (交叉熵)'),
        ('svm', '支持向量机 (合页损失)')
    ]
    
    print("-" * 60)
    print(f"{'模型方法':<25} {'训练准确率':<15} {'测试准确率':<15}")
    print("-" * 60)
    
    for method_id, method_name in methods:
        model = LinearClassifierComparison(method=method_id, max_iter=5000, learning_rate=0.1)
        model.train(X_train_scaled, y_train)
        
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        acc_train = eval_acc(y_train, y_train_pred)
        acc_test = eval_acc(y_test, y_test_pred)
        
        print(f"{method_name:<25} {acc_train*100:>10.2f}% {acc_test*100:>10.2f}%")

    print("-" * 60)

if __name__ == '__main__':
    main()
