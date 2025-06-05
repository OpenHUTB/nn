# python: 3.5.2
# encoding: utf-8

import numpy as np


def load_data(fname):
    """载入数据。"""
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
    """计算准确率。"""
    return np.sum(label == pred) / len(pred)


class SVM():
    """SVM模型。"""

    def __init__(self):
        self.learning_rate = 0.01
        self.reg_lambda = 0.01
        self.max_iter = 1000
        self.w = None  # 权重向量
        self.b = None  # 偏置项

    def train(self, data_train):
        """训练模型。"""
        # 请补全此处代码
        X = data_train[:, :2]
        y = data_train[:, 2]
        y = np.where(y == 0, -1, 1)  # 将标签转换为{-1, 1}
        m, n = X.shape
        
        # 初始化参数
        self.w = np.zeros(n)
        self.b = 0
        
        for epoch in range(self.max_iter):
            # 计算函数间隔
            margin = y * (np.dot(X, self.w) + self.b)
            # 找出违反间隔条件的样本（margin < 1）
            idx = np.where(margin < 1)[0]
            
            # 计算梯度
            dw = (2 * self.reg_lambda * self.w) - np.mean(y[idx].reshape(-1, 1) * X[idx], axis=0)
            db = -np.mean(y[idx])
            
            # 参数更新
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, x):
        """预测标签。"""
        # 请补全此处代码
        score = np.dot(x, self.w) + self.b
        return np.where(score >= 0, 1, 0)  # 转换回{0, 1}标签


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
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
