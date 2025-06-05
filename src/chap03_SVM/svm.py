# python: 3.5.2
# encoding: utf-8

import numpy as np


def load_data(fname):

    """
    载入数据。
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
    """
    return np.sum(label == pred) / len(pred)


class SVM():
    """
    SVM模型。
    """

    def __init__(self):
        self.learning_rate = 0.01
        self.reg_lambda = 0.01
        self.max_iter = 1000
        self.w = None  # 权重向量
        self.b = None  # 偏置项

    def train(self, data_train):
        """
        训练模型。
        """
        # 请补全此处代码

        x = data_train[:, :2]  # 提取特征 (N x 2)
        y = data_train[:, 2]   # 提取标签 (N,)

        # 将标签转换为 +1 和 -1
        y = np.where(y == 0, -1, 1)

        num_samples, num_features = x.shape
        self.w = np.zeros(num_features)  # 初始化权重
        self.b = 0.0                      # 初始化偏置

        for _ in range(self.max_iter):
            for i in range(num_samples):
                xi = x[i]
                yi = y[i]
                condition = yi * (np.dot(self.w, xi) + self.b)

                if condition >= 1:
                    # 不违反间隔约束，只考虑正则项
                    self.w -= self.learning_rate * self.reg_lambda * self.w
                    # self.b 不变
                else:
                    # 违反间隔约束，考虑损失项和正则项
                    self.w -= self.learning_rate * (self.reg_lambda * self.w - yi * xi)
                    self.b += self.learning_rate * yi

    
    def predict(self, x):
        """
        预测标签。
        """
        # 请补全此处代码
        linear_output = np.dot(x, self.w) + self.b
        predictions = np.where(linear_output >= 0, 1, 0)  # 注意返回0或1标签
        return predictions


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM()  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    # 评估结果，计算准确率
    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
