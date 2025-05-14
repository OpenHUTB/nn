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
        # 请补全此处代码
        self.w = None  # 权重向量
        self.b = 0     # 偏置项
        pass

    def train(self, data_train):
        """
        训练模型。
        """
# 提取特征和标签
        x = data_train[:, :2]
        t = data_train[:, 2]
        
        # 初始化参数
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)
        learning_rate = 0.01
        lambda_ = 0.01
        epochs = 1000
 
        # 梯度下降优化
        for _ in range(epochs):
            for idx, x_i in enumerate(x):
                # 计算预测值（使用符号函数判断分类）
                condition = t[idx] * (np.dot(x_i, self.w) + self.b)
                if condition >= 1:
                    self.w -= learning_rate * (lambda_ * self.w)  # 仅正则化项
                else:
                    self.w -= learning_rate * (lambda_ * self.w - t[idx] * x_i)
                    self.b -= learning_rate * (-t[idx])
        # 请补全此处代码

        def predict(self, x):
            if self.weights is None:
                raise RuntimeError("Model not trained yet")
            logits = np.dot(x, self.weights) + self.bias
            return (logits >= 0).astype(int)  
        # 二分类示例
    #def predict(self, x):
        """
        预测标签。
        """

        # 请补全此处代码
# 计算决策函数值
        decision_values = np.dot(x, self.w) + self.b
        # 返回预测标签（0或1）
        return np.where(decision_values >= 0, 1, 0)


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
