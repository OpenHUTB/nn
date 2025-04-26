# python: 3.5.2
# encoding: utf-8

import numpy as np


def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
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
    def __init__(self, learning_rate=0.01, lambda_=0.01, epochs=1000):
        # 请补全此处代码
        self.w = None  # 权重向量
        self.b = 0.0   # 偏置项
        self.learning_rate = learning_rate  # 学习率
        self.lambda_ = lambda_             # 正则化系数
        self.epochs = epochs               # 迭代次数
        pass

    def train(self, data_train):
        """
        训练模型。
        """
        # 提取特征和标签
        x = data_train[:, :2]
        t = data_train[:, 2]
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)  # 初始化权重

        # 训练过程优化：随机梯度下降
        for _ in range(self.epochs):
            # 打乱样本顺序提升训练效果
            indices = np.random.permutation(n_samples)  
            for idx in indices:
                x_i = x[idx]
                t_i = t[idx]

                # 计算预测值（判断是否满足条件）
                condition = t_i * (np.dot(x_i, self.w) + self.b)

                if condition >= 1:
                    # 更新权重（仅正则化项）
                    grad_w = self.lambda_ * self.w
                    self.w -= self.learning_rate * grad_w
                else:
                    # 更新权重和偏置项（梯度下降方向）
                    grad_w = self.lambda_ * self.w - t_i * x_i
                    grad_b = -t_i
                    self.w -= self.learning_rate * grad_w
                    self.b -= self.learning_rate * grad_b

    def predict(self, x):
        """
        预测标签。
        """
        # 请补全此处代码
        # 计算决策函数值：w·x + b
        decision_values = np.dot(x, self.w) + self.b
        # 返回预测标签（0或1）
        return np.where(decision_values >= 0, 1, 0)  # 映射到0/1标签


if __name__ == '__main__':
    # 载入数据，实际使用时将x替换为具体名称
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