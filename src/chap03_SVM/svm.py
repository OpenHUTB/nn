#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM分类器实现
基于hinge loss和L2正则化的支持向量机分类器
"""

import numpy as np
import os


def load_data(fname):
    """载入数据集。

    参数:
        fname (str): 数据文件路径

    返回:
        np.array: 包含特征和标签的numpy数组，形状为(n_samples, 3)

    异常:
        FileNotFoundError: 当数据文件不存在时抛出
    """
    # 检查文件是否存在，确保数据加载的可靠性
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"数据文件未找到: {fname}\n"
            f"请确认文件路径是否正确，当前工作目录为: {os.getcwd()}"
        )

    data = []
    with open(fname, 'r', encoding='utf-8') as f:
        # 跳过表头行
        f.readline()
        for line in f:
            # 去除空白字符并按空格分割
            line = line.strip().split()
            if len(line) < 3:
                continue  # 跳过不完整的行
            # 解析特征和标签
            x1 = float(line[0])  # 特征1
            x2 = float(line[1])  # 特征2
            t = int(line[2])  # 标签：0或1
            data.append([x1, x2, t])

    return np.array(data)


def eval_acc(label, pred):
    """计算分类准确率。

    参数:
        label (array-like): 真实标签数组
        pred (array-like): 预测标签数组

    返回:
        float: 准确率，范围[0, 1]
    """
    return np.sum(label == pred) / len(pred)


class SVM:
    """支持向量机分类器。

    基于最大间隔原则的分类模型，使用hinge loss和L2正则化。

    属性:
        learning_rate (float): 学习率，控制梯度下降步长
        reg_lambda (float): L2正则化系数
        max_iter (int): 最大训练迭代次数
        w (np.array): 权重向量
        b (float): 偏置项
    """

    def __init__(self, learning_rate=0.01, reg_lambda=0.01, max_iter=1000):
        """初始化SVM模型。

        参数:
            learning_rate (float): 学习率，默认0.01
            reg_lambda (float): 正则化系数，默认0.01
            max_iter (int): 最大迭代次数，默认1000
        """
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.w = None  # 权重向量
        self.b = None  # 偏置项

    def _initialize_parameters(self, n_features):
        """初始化模型参数。

        参数:
            n_features (int): 特征数量
        """
        self.w = np.zeros(n_features)  # 权重初始化为0
        self.b = 0.0  # 偏置初始化为0

    def train(self, data_train, verbose=False):
        """训练SVM模型。

        基于hinge loss + L2正则化，使用梯度下降法优化参数。

        参数:
            data_train (np.array): 训练数据，形状为(n_samples, 3)
            verbose (bool): 是否显示训练进度，默认False

        算法步骤:
            1. 数据预处理和参数初始化
            2. 迭代优化：
               - 计算函数间隔
               - 识别违反间隔条件的样本
               - 计算梯度
               - 更新参数
        """
        # 提取特征和标签
        X = data_train[:, :2]  # 特征矩阵
        y = data_train[:, 2]  # 原始标签
        y = np.where(y == 0, -1, 1)  # 转换为{-1, 1}格式

        m, n = X.shape  # m: 样本数, n: 特征数

        # 初始化参数
        self._initialize_parameters(n)

        # 训练循环
        for epoch in range(self.max_iter):
            # 计算函数间隔：y(wx + b)
            margin = y * (np.dot(X, self.w) + self.b)

            # 找出违反间隔条件的样本（margin < 1）
            # 包括误分类样本(margin < 0)和间隔内样本(0 ≤ margin < 1)
            violating_indices = np.where(margin < 1)[0]

            # 计算梯度
            if len(violating_indices) > 0:
                # 正则化项梯度 + hinge loss梯度
                dw = (2 * self.reg_lambda * self.w) - np.mean(
                    y[violating_indices].reshape(-1, 1) * X[violating_indices],
                    axis=0
                )
                db = -np.mean(y[violating_indices])
            else:
                # 如果没有违反间隔的样本，只更新正则化项
                dw = 2 * self.reg_lambda * self.w
                db = 0

            # 梯度下降更新参数
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # 可选：打印训练进度
            if verbose and (epoch + 1) % 100 == 0:
                acc = self._compute_accuracy(X, y)
                print(f"Epoch {epoch + 1}/{self.max_iter}, "
                      f"Training Accuracy: {acc:.3f}")

    def _compute_accuracy(self, X, y):
        """计算当前模型在给定数据上的准确率。

        参数:
            X (np.array): 特征矩阵
            y (np.array): 标签数组（{-1, 1}格式）

        返回:
            float: 准确率
        """
        predictions = np.sign(np.dot(X, self.w) + self.b)
        return np.mean(predictions == y)

    def predict(self, X):
        """预测样本标签。

        参数:
            X (np.array): 特征矩阵，形状为(n_samples, n_features)

        返回:
            np.array: 预测标签，0或1
        """
        # 计算决策函数值：w·x + b
        decision_scores = np.dot(X, self.w) + self.b
        # 根据符号预测标签：正数->1，负数->0
        return np.where(decision_scores >= 0, 1, 0)

    def get_parameters(self):
        """获取模型参数。

        返回:
            tuple: (权重向量w, 偏置项b)
        """
        return self.w.copy(), self.b


def main():
    """主函数：SVM分类器的完整训练和评估流程。"""
    try:
        # 配置数据文件路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        train_file = os.path.join(base_dir, 'data', 'train_linear.txt')
        test_file = os.path.join(base_dir, 'data', 'test_linear.txt')

        # 加载数据
        print("正在加载数据...")
        data_train = load_data(train_file)
        data_test = load_data(test_file)
        print(f"训练集大小: {len(data_train)} 个样本")
        print(f"测试集大小: {len(data_test)} 个样本")

        # 训练模型
        print("开始训练SVM模型...")
        svm = SVM(learning_rate=0.01, reg_lambda=0.01, max_iter=1000)
        svm.train(data_train, verbose=True)

        # 评估模型
        print("\n模型评估:")
        # 训练集评估
        X_train = data_train[:, :2]
        y_train = data_train[:, 2]
        y_train_pred = svm.predict(X_train)
        acc_train = eval_acc(y_train, y_train_pred)

        # 测试集评估
        X_test = data_test[:, :2]
        y_test = data_test[:, 2]
        y_test_pred = svm.predict(X_test)
        acc_test = eval_acc(y_test, y_test_pred)

        # 输出结果
        print(f"训练集准确率: {acc_train * 100:.1f}%")
        print(f"测试集准确率: {acc_test * 100:.1f}%")

        # 输出模型参数
        w, b = svm.get_parameters()
        print(f"\n模型参数:")
        print(f"权重向量 w: {w}")
        print(f"偏置项 b: {b:.4f}")

    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")


if __name__ == '__main__':
    main()