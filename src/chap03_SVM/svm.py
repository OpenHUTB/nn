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

    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        """
        初始化SVM模型
        参数:
            learning_rate: 学习率
            lambda_param: 正则化参数
            n_iters: 迭代次数
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def train(self, data_train):
        """
        训练模型。
        """
        # 提取特征和标签
        X = data_train[:, :2]
        y = data_train[:, 2]
        
        # 转换为-1和1的标签
        y = np.where(y == 0, -1, 1)
        
        # 添加偏置项
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        
        n_samples, n_features = X.shape
        
        # 初始化权重
        self.w = np.zeros(n_features)
        
        # 梯度下降
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * np.dot(x_i, self.w) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y[idx])
                    )

    def predict(self, x):
        """
        预测标签。
        """
        # 添加偏置项
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        approx = np.dot(x, self.w)
        return np.sign(approx)


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)  # 初始化模型
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
    
    print(f"Training accuracy: {acc_train:.2f}")
    print(f"Test accuracy: {acc_test:.2f}")
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
