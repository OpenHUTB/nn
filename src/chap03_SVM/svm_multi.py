#!/usr/bin/env python
# coding: utf-8
"""多分类 SVM 实现 (One-vs-Rest 策略)"""

import numpy as np
import os

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
# 多分类 SVM
# ============================================================

class MultiClassSVM:
    """使用 One-vs-Rest (OvR) 策略的多分类 SVM。"""
    
    def __init__(self, n_classes=3, learning_rate=0.01, reg_lambda=0.01, max_iter=2000):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.models = []  # 存储每个类别的二分类模型 (w, b)

    def _train_binary_svm(self, X, y_binary):
        """训练一个简单的线性二分类 SVM (基于合页损失)。"""
        m, n = X.shape
        w = np.zeros(n)
        b = 0
        
        for epoch in range(self.max_iter):
            # 决策得分: score = wx + b
            score = np.dot(X, w) + b
            margin = y_binary * score
            
            # 找到违反间隔条件的样本 (margin < 1)
            idx = np.where(margin < 1)[0]
            
            # 计算梯度
            if len(idx) > 0:
                dw = (2 * self.reg_lambda * w - np.sum(y_binary[idx, None] * X[idx], axis=0)) / m
                db = -np.mean(y_binary[idx])
            else:
                dw = (2 * self.reg_lambda * w) / m
                db = 0
            
            # 梯度下降更新
            w -= self.learning_rate * dw
            b -= self.learning_rate * db
            
        return w, b

    def train(self, X, y):
        """对每一类训练一个二分类模型 (One-vs-Rest)。"""
        self.models = []
        unique_classes = np.unique(y)
        
        for c in unique_classes:
            # 将当前类标为 1，其他类标为 -1
            y_binary = np.where(y == c, 1, -1)
            w, b = self._train_binary_svm(X, y_binary)
            self.models.append((c, w, b))

    def predict(self, X):
        """预测类别：选择决策得分最高的类别。"""
        scores = []
        for class_label, w, b in self.models:
            score = np.dot(X, w) + b
            scores.append(score)
        
        # 将分数堆叠并找到最大得分的索引
        scores = np.vstack(scores)  # (n_classes, n_samples)
        class_indices = np.argmax(scores, axis=0)
        
        # 映射回原始类标
        return np.array([self.models[i][0] for i in class_indices])

# ============================================================
# 主程序
# ============================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, 'data', 'train_multi.txt')
    test_file = os.path.join(base_dir, 'data', 'test_multi.txt')
    
    data_train = load_data(train_file)
    data_test = load_data(test_file)
    
    X_train = data_train[:, :2]
    y_train = data_train[:, 2]
    X_test = data_test[:, :2]
    y_test = data_test[:, 2]
    
    print("-" * 60)
    print("正在训练三分类 SVM (One-vs-Rest)...")
    
    model = MultiClassSVM(n_classes=3, max_iter=2000)
    model.train(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    acc_train = eval_acc(y_train, y_train_pred)
    acc_test = eval_acc(y_test, y_test_pred)
    
    print("-" * 60)
    print(f"训练准确率 (3-Class): {acc_train * 100:.2f}%")
    print(f"测试准确率 (3-Class): {acc_test * 100:.2f}%")
    print("-" * 60)

if __name__ == '__main__':
    main()
