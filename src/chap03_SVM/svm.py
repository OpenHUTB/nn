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
        self.b = None  # 偏置项
        self.C = 1.0   # 惩罚参数
        self.tol = 1e-3  # 容错率
        self.max_iter = 100  # 最大迭代次数
        self.alpha = None  # 拉格朗日乘子
        self.support_vectors = None  # 支持向量

    def train(self, data_train):
        """
        训练模型。
        使用SMO算法求解SVM
        """
        X = data_train[:, :2]  # 特征
        y = data_train[:, 2]   # 标签
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.w = np.zeros(n_features)
        
        # SMO算法
        iter_num = 0
        while iter_num < self.max_iter:
            alpha_changed = 0
            for i in range(n_samples):
                # 计算预测值和误差
                Ei = self.predict(X[i]) - y[i]
                
                # 判断是否满足KKT条件
                if (y[i] * Ei < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * Ei > self.tol and self.alpha[i] > 0):
                    
                    # 随机选择另一个alpha
                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    # 计算另一个样本的误差
                    Ej = self.predict(X[j]) - y[j]
                    
                    # 保存旧的alpha值
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # 计算L和H
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # 计算eta
                    eta = 2 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
                    if eta >= 0:
                        continue
                    
                    # 更新alpha_j
                    self.alpha[j] = self.alpha[j] - y[j] * (Ei - Ej) / eta
                    self.alpha[j] = min(H, max(L, self.alpha[j]))
                    
                    # 如果变化太小，则跳过
                    if abs(self.alpha[j] - alpha_j_old) < 1e-4:
                        continue
                    
                    # 更新alpha_i
                    self.alpha[i] = self.alpha[i] + y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # 更新b
                    b1 = self.b - Ei - y[i] * (self.alpha[i] - alpha_i_old) * np.dot(X[i], X[i]) - \
                         y[j] * (self.alpha[j] - alpha_j_old) * np.dot(X[i], X[j])
                    b2 = self.b - Ej - y[i] * (self.alpha[i] - alpha_i_old) * np.dot(X[i], X[j]) - \
                         y[j] * (self.alpha[j] - alpha_j_old) * np.dot(X[j], X[j])
                    self.b = (b1 + b2) / 2
                    
                    alpha_changed += 1
            
            if alpha_changed == 0:
                iter_num += 1
            else:
                iter_num = 0
        
        # 计算权重向量
        self.w = np.sum(self.alpha[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)
        
        # 保存支持向量
        self.support_vectors = X[self.alpha > 0]

    def predict(self, x):
        """
        预测标签。
        """
        if self.w is None or self.b is None:
            raise ValueError("Model not trained yet!")
        
        # 计算决策函数值
        f = np.dot(x, self.w) + self.b
        # 返回预测标签
        return np.sign(f)


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
