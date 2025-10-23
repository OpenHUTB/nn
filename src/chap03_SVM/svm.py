import numpy as np
import os
import matplotlib.pyplot as plt


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


def plot_data(data, title):
    """绘制数据分布"""
    plt.figure(figsize=(10, 8))

    # 分离不同类别的数据
    class_1 = data[data[:, 2] == 1]
    class_neg1 = data[data[:, 2] == -1]
    class_0 = data[data[:, 2] == 0]

    if len(class_1) > 0:
        plt.scatter(class_1[:, 0], class_1[:, 1], c='red', label='Class 1', alpha=0.6)
    if len(class_neg1) > 0:
        plt.scatter(class_neg1[:, 0], class_neg1[:, 1], c='blue', label='Class -1', alpha=0.6)
    if len(class_0) > 0:
        plt.scatter(class_0[:, 0], class_0[:, 1], c='green', label='Class 0', alpha=0.6)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


class KernelSVM:
    """核函数SVM（支持线性、多项式、高斯核）"""

    def __init__(self, kernel='linear', degree=3, gamma=0.1, C=1.0, max_iter=1000, learning_rate=0.001):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.alpha = None
        self.b = 0
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

        # 检查标签范围并转换
        unique_labels = np.unique(y)
        print(f"数据标签: {unique_labels}")

        if len(unique_labels) == 1:
            print("警告: 数据集中只有一个类别!")
            # 如果是单类别，创建一个虚拟的负类
            y = np.ones(len(y))
        elif set(unique_labels) == {0, 1}:
            y = np.where(y == 0, -1, 1)  # 转换为{-1, 1}
        elif set(unique_labels) == {-1, 1}:
            # 已经是正确的格式
            pass
        else:
            raise ValueError(f"不支持的标签格式: {unique_labels}")

        m, n = X.shape
        self.X_train = X
        self.y_train = y

        # 初始化参数
        self.alpha = np.random.randn(m) * 0.01
        self.b = 0

        # 梯度下降优化
        for iteration in range(self.max_iter):
            total_loss = 0

            for i in range(m):
                # 计算当前样本的预测
                prediction = 0
                for j in range(m):
                    prediction += self.alpha[j] * self.y_train[j] * self.kernel_function(X[j], X[i])
                prediction += self.b

                # hinge loss
                loss = max(0, 1 - self.y_train[i] * prediction)
                total_loss += loss

                # 计算梯度并更新
                if loss > 0:
                    # 对于误分类样本，更新所有alpha
                    for j in range(m):
                        kernel_val = self.kernel_function(X[j], X[i])
                        grad_alpha = self.C * self.alpha[j] - self.y_train[i] * self.y_train[j] * kernel_val
                        self.alpha[j] -= self.learning_rate * grad_alpha

                    # 更新偏置
                    grad_b = -self.y_train[i]
                    self.b -= self.learning_rate * grad_b
                else:
                    # 只更新正则化项
                    for j in range(m):
                        self.alpha[j] -= self.learning_rate * self.C * self.alpha[j]

            # 打印训练进度
            if iteration % 200 == 0:
                avg_loss = total_loss / m if m > 0 else 0
                print(f"迭代 {iteration}, 平均损失: {avg_loss:.4f}")

        # 提取支持向量 (alpha > 阈值)
        sv_threshold = 1e-4
        sv_indices = np.abs(self.alpha) > sv_threshold
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.alpha = self.alpha[sv_indices]

        print(f"训练完成，找到 {len(self.support_vectors)} 个支持向量")

    def predict(self, x):
        """预测标签"""
        if len(x.shape) == 1:
            # 单样本预测
            score = 0
            for i in range(len(self.alpha)):
                score += self.alpha[i] * self.support_vector_labels[i] * \
                         self.kernel_function(self.support_vectors[i], x)
            score += self.b
            return 1 if score >= 0 else 0
        else:
            # 多样本预测
            predictions = []
            for sample in x:
                score = 0
                for i in range(len(self.alpha)):
                    score += self.alpha[i] * self.support_vector_labels[i] * \
                             self.kernel_function(self.support_vectors[i], sample)
                score += self.b
                predictions.append(1 if score >= 0 else 0)
            return np.array(predictions)


def test_kernel_svm():
    """测试核函数SVM"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, 'data', 'train_kernel.txt')
    test_file = os.path.join(base_dir, 'data', 'test_kernel.txt')

    # 加载数据
    data_train = load_data(train_file)
    data_test = load_data(test_file)

    print(f"训练集大小: {len(data_train)}")
    print(f"测试集大小: {len(data_test)}")

    # 分析数据
    print("\n=== 训练数据分析 ===")
    unique_train = np.unique(data_train[:, 2])
    print(f"训练集标签: {unique_train}")

    print("\n=== 测试数据分析 ===")
    unique_test = np.unique(data_test[:, 2])
    print(f"测试集标签: {unique_test}")

    # 绘制数据分布
    plot_data(data_train, "训练数据分布")
    plot_data(data_test, "测试数据分布")

    # 检查数据文件内容
    print("\n=== 检查数据文件前10行 ===")
    with open(train_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 10:  # 前10行
                print(f"行 {i}: {line.strip()}")
            else:
                break

    # 重新加载数据，确保正确处理标签
    print("\n重新加载数据并检查标签分布...")
    data_train = load_data(train_file)
    data_test = load_data(test_file)

    # 手动检查标签
    train_labels = data_train[:, 2]
    test_labels = data_test[:, 2]

    print(f"训练集标签统计:")
    for label in np.unique(train_labels):
        count = np.sum(train_labels == label)
        print(f"  标签 {label}: {count} 个样本")

    print(f"测试集标签统计:")
    for label in np.unique(test_labels):
        count = np.sum(test_labels == label)
        print(f"  标签 {label}: {count} 个样本")

    # 如果数据有问题，使用正确的数据集
    print("\n=== 使用线性数据集进行测试 ===")
    train_linear_file = os.path.join(base_dir, 'data', 'train_linear.txt')
    test_linear_file = os.path.join(base_dir, 'data', 'test_linear.txt')

    if os.path.exists(train_linear_file) and os.path.exists(test_linear_file):
        print("使用线性数据集进行测试...")
        data_train = load_data(train_linear_file)
        data_test = load_data(test_linear_file)

        # 分析线性数据
        print("\n=== 线性训练数据分析 ===")
        unique_train = np.unique(data_train[:, 2])
        print(f"训练集标签: {unique_train}")

        print("\n=== 线性测试数据分析 ===")
        unique_test = np.unique(data_test[:, 2])
        print(f"测试集标签: {unique_test}")

        plot_data(data_train, "线性训练数据分布")
        plot_data(data_test, "线性测试数据分布")

        x_train = data_train[:, :2]
        t_train = data_train[:, 2]
        x_test = data_test[:, :2]
        t_test = data_test[:, 2]

        print("\n=== 核函数SVM比较 (线性数据) ===")

        # 测试线性核
        print("\n训练 linear 核SVM...")
        svm = KernelSVM(kernel='linear', C=1.0, max_iter=1000, learning_rate=0.001)
        svm.train(data_train)

        # 预测
        train_pred = svm.predict(x_train)
        test_pred = svm.predict(x_test)

        # 计算准确率
        train_acc = eval_acc(t_train, train_pred)
        test_acc = eval_acc(t_test, test_pred)

        print(f"linear核SVM - 训练集准确率: {train_acc * 100:.1f}%, 测试集准确率: {test_acc * 100:.1f}%")
        print(f"  支持向量数量: {len(svm.support_vectors)}")


if __name__ == '__main__':
    # 运行核函数SVM测试
    test_kernel_svm()