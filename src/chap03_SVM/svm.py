import numpy as np
import os


def load_data(fname):
    """载入数据。"""
    if not os.path.exists(fname):
        raise FileNotFoundError(f"数据文件未找到: {fname}\n请确认文件路径是否正确，当前工作目录为: {os.getcwd()}")
    with open(fname, 'r') as f:
        data = []
        line = f.readline()  # 跳过表头行
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


class KernelSVM:
    """核函数SVM（支持线性、多项式、高斯核）"""

    def __init__(self, kernel='linear', degree=3, gamma=0.1, C=1.0, max_iter=1000):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.C = C
        self.max_iter = max_iter
        self.alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None
        self.support_vectors = None
        self.support_vector_labels = None

    def kernel_function(self, x1, x2):
        """核函数"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("不支持的核函数类型")

    def train(self, data_train):
        """训练核函数SVM"""
        X = data_train[:, :2]
        y = data_train[:, 2]
        y = np.where(y == 0, -1, 1)  # 转换为{-1, 1}
        m, n = X.shape

        self.X_train = X
        self.y_train = y

        # 初始化参数
        self.alpha = np.zeros(m)
        self.b = 0

        # 简化的SMO算法
        for _ in range(self.max_iter):
            for i in range(m):
                # 计算预测误差
                Ei = self.decision_function(X[i]) - y[i]

                # 检查KKT条件
                if (y[i] * Ei < -0.001 and self.alpha[i] < self.C) or \
                        (y[i] * Ei > 0.001 and self.alpha[i] > 0):

                    # 随机选择另一个样本
                    j = np.random.randint(0, m)
                    while j == i:
                        j = np.random.randint(0, m)

                    Ej = self.decision_function(X[j]) - y[j]

                    # 保存旧的alpha值
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    # 计算边界
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    # 计算eta
                    eta = 2 * self.kernel_function(X[i], X[j]) - \
                          self.kernel_function(X[i], X[i]) - \
                          self.kernel_function(X[j], X[j])

                    if eta >= 0:
                        continue

                    # 更新alpha[j]
                    self.alpha[j] -= y[j] * (Ei - Ej) / eta

                    # 裁剪alpha[j]
                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    if self.alpha[j] < L:
                        self.alpha[j] = L

                    # 检查alpha[j]变化是否显著
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # 更新alpha[i]
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    # 更新偏置b
                    b1 = self.b - Ei - y[i] * (self.alpha[i] - alpha_i_old) * \
                         self.kernel_function(X[i], X[i]) - \
                         y[j] * (self.alpha[j] - alpha_j_old) * \
                         self.kernel_function(X[i], X[j])

                    b2 = self.b - Ej - y[i] * (self.alpha[i] - alpha_i_old) * \
                         self.kernel_function(X[i], X[j]) - \
                         y[j] * (self.alpha[j] - alpha_j_old) * \
                         self.kernel_function(X[j], X[j])

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

        # 提取支持向量
        sv_indices = self.alpha > 1e-5
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.alpha = self.alpha[sv_indices]

    def decision_function(self, x):
        """决策函数"""
        result = 0
        for i in range(len(self.alpha)):
            result += self.alpha[i] * self.support_vector_labels[i] * \
                      self.kernel_function(self.support_vectors[i], x)
        return result + self.b

    def predict(self, x):
        """预测标签"""
        if len(x.shape) == 1:
            # 单样本预测
            score = self.decision_function(x)
            return 1 if score >= 0 else 0
        else:
            # 多样本预测
            scores = np.array([self.decision_function(xi) for xi in x])
            return np.where(scores >= 0, 1, 0)


def test_kernel_svm():
    """测试核函数SVM"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, 'data', 'train_kernel.txt')
    test_file = os.path.join(base_dir, 'data', 'test_kernel.txt')

    # 加载数据
    data_train = load_data(train_file)
    data_test = load_data(test_file)

    # 测试不同核函数
    kernels = ['linear', 'poly', 'rbf']

    print("=== 核函数SVM比较 ===")
    for kernel in kernels:
        if kernel == 'linear':
            svm = KernelSVM(kernel='linear', C=1.0)
        elif kernel == 'poly':
            svm = KernelSVM(kernel='poly', degree=3, C=1.0)
        else:  # rbf
            svm = KernelSVM(kernel='rbf', gamma=0.1, C=1.0)

        # 训练模型
        svm.train(data_train)

        # 预测
        x_train = data_train[:, :2]
        t_train = data_train[:, 2]
        x_test = data_test[:, :2]
        t_test = data_test[:, 2]

        train_pred = svm.predict(x_train)
        test_pred = svm.predict(x_test)

        # 计算准确率
        train_acc = eval_acc(t_train, train_pred)
        test_acc = eval_acc(t_test, test_pred)

        print(f"{kernel.upper()}核SVM - 训练集准确率: {train_acc * 100:.1f}%, 测试集准确率: {test_acc * 100:.1f}%")
        print(f"  支持向量数量: {len(svm.support_vectors)}")


if __name__ == '__main__':
    # 运行核函数SVM测试
    test_kernel_svm()
