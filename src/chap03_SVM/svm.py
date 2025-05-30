# python: 3.5.2
# encoding: utf-8

import numpy as np


def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline() # 首行是标题行，自动跳过
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
    return np.sum(label == pred) / len(pred)#准确率 = 正确预测的样本数 / 总样本数

#SVM模型 实现了线性SVM分类器
class SVM():
    """
    SVM模型。
    """
    #目标函数：(1/2)||w||² + C * Σmax(0, 1 - y_i(w·x_i + b))
    def __init__(self):
        # 请补全此处代码
        self.w = None  # 权重向量
        self.b = 0     # 偏置项
        pass
    

    def train(self, data_train):
        """
        训练 SVM 模型。
        :param data_train: 包含特征和标签的 NumPy 数组，形状为 (n_samples, n_features + 1)
        """
        X = data_train[:, :-1]
        y = np.where(data_train[:, -1] == 0, -1, 1)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            # 计算 margins
            margin = y * (X @ self.w + self.b)
            misclassified = margin < 1

            # 梯度计算
            dw = self.lambda_ * self.w - np.mean((misclassified * y)[:, np.newaxis] * X, axis=0)
            db = -np.mean(misclassified * y)

            # 参数更新
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # 提前停止：权重更新变化太小
            if np.linalg.norm(self.lr * dw) < self.tolerance:
                break
        # 请补全此处代码

    def predict(self, x):
        """
        预测标签。
        """
        # 请补全此处代码
        # 计算决策函数值
        if x.ndim == 1:
           x = np.expand_dims(x, axis=0)  # 处理单样本输入
        decision_values = np.dot(x, self.w) + self.b  # logits = x·w + b
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
