# python: 3.5.2
# encoding: utf-8

import numpy as np


def load_data(fname):
    """
    载入数据。

    Args:
        fname: 数据文件的路径。

    Returns:
        data: 包含特征和标签的 numpy 数组。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。

    Args:
        label: 真实标签。
        pred: 预测标签。

    Returns:
        准确率。
    """
    return np.sum(label == pred) / len(pred)


class SVM:
    """SVM 模型。"""

    def __init__(self):
        """
        初始化 SVM 模型。

        Attributes:
            learning_rate: 学习率，默认为 0.01。
            reg_lambda: 正则化参数，默认为 0.01。
            max_iter: 最大迭代次数，默认为 1000。
            w: 权重向量，初始化为 None。
            b: 偏置项，初始化为 None。
        """
        self.learning_rate = 0.01
        self.reg_lambda = 0.01
        self.max_iter = 1000
        self.w = None  # 权重向量
        self.b = None  # 偏置项

    def train(self, data_train):
        """
        训练模型。

        Args:
            data_train: 训练数据，包含特征和标签。
        """
        # 请补全此处代码

    def predict(self, x):
        """
        预测标签。

        Args:
            x: 输入特征。

        Returns:
            预测标签。
        """
        # 请补全此处代码


if __name__ == '__main__':
    # 载入数据，实际使用时将 x 替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式 [x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练 SVM 模型
    svm = SVM()  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用 SVM 模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
