# python: 3.5.2
# encoding: utf-8

import numpy as np

def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()  # 首行是标题行，自动跳过
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
    return np.sum(label == pred) / len(pred)  # 准确率 = 正确预测的样本数 / 总样本数

class SVM():
    """
    SVM模型。
    目标函数：(1/2)||w||² + C * Σmax(0, 1 - y_i(w·x_i + b))
    """
    def __init__(self):
        self.w = None  # 权重向量
        self.b = 0.0   # 偏置项

    def train(self, data_train):
        """
        训练模型。
        """
        x = data_train[:, :2]
        t = data_train[:, 2]
        # 将标签转为-1和1（二分类 SVM 的要求）
        t = np.where(t == 0, -1, 1)

        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)
        learning_rate = 0.01
        lambda_ = 0.01  # 正则化系数
        epochs = 1000

        for _ in range(epochs):
            for idx, x_i in enumerate(x):
                condition = t[idx] * (np.dot(x_i, self.w) + self.b)
                if condition >= 1:
                    self.w -= learning_rate * lambda_ * self.w
                else:
                    self.w -= learning_rate * (lambda_ * self.w - t[idx] * x_i)
                    self.b -= learning_rate * (-t[idx])

    def predict(self, x):
        """
        预测标签。
        """
        if self.w is None:
            raise RuntimeError("模型尚未训练")

        decision_values = np.dot(x, self.w) + self.b
        return np.where(decision_values >= 0, 1, 0)

if __name__ == '__main__':
    # 载入数据
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)
    data_test = load_data(test_file)

    # 训练模型
    svm = SVM()
    svm.train(data_train)

    # 预测与评估
    x_train = data_train[:, :2]
    t_train = data_train[:, 2]
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]

    t_train_pred = svm.predict(x_train)
    t_test_pred = svm.predict(x_test)

    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)

    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
