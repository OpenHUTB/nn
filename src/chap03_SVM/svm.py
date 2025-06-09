# python: 3.5.2
# encoding: utf-8

# 导入NumPy科学计算库，并使用缩写np简化调用
import numpy as np
import os  # 新增：用于处理路径

def load_data(fname):
    """载入数据。"""
    # 检查文件是否存在，若不存在则给出详细报错
    if not os.path.exists(fname):
        raise FileNotFoundError(f"数据文件未找到: {fname}\n请确认文件路径是否正确，当前工作目录为: {os.getcwd()}")
    with open(fname, 'r') as f:        # 'r' 表示以只读模式打开文件
        data = []                      # 初始化一个空列表用于存储处理后的数据
        line = f.readline()            # 读取并丢弃第一行（通常可能是标题行或不需要的数据）
        for line in f:
            line = line.strip().split()  # 去除行首尾的空白字符（如换行符、空格等），然后按空白字符分割字符串，得到元素列表
            x1 = float(line[0])        # 将第一列转换为浮点数，作为第一个特征 x1
            x2 = float(line[1])        # 将第二列转换为浮点数，作为第二个特征 x2
            t = int(line[2])           # 将第三列转换为整数，作为目标值/标签 t
            data.append([x1, x2, t])   # 将处理后的数据组合成列表并添加到 data 中
        return np.array(data)          # 返回numpy数组，方便后续数值计算

def eval_acc(label, pred):
    """计算准确率。"""
    return np.sum(label == pred) / len(pred)  # 统计预测正确的样本比例

class SVM:
    """SVM模型。"""

    def __init__(self):
        # 初始化模型参数
        self.learning_rate = 0.01 # 学习率：控制梯度下降中参数更新的步长
        self.reg_lambda = 0.01    # 正则化系数：防止过拟合
        self.max_iter = 1000      # 最大迭代次数
        self.w = None             # 权重向量
        self.b = None             # 偏置项

    def train(self, data_train):
        """训练 SVM 模型（基于 hinge loss + L2 正则化）

        参数:
            data_train: 训练数据集，shape=(m, 3)，前两列为特征，第三列为标签（0/1）
        """

        X = data_train[:, :2]         # 提取特征部分
        y = data_train[:, 2]          # 提取标签部分
        y = np.where(y == 0, -1, 1)   # 将标签转换为{-1, 1}
        m, n = X.shape                # m为样本数，n为特征数


        # 初始化参数
        self.w = np.zeros(n)      # 权重初始化为0
        self.b = 0                # 偏置初始化为0

        for epoch in range(self.max_iter):
            # 计算函数间隔
            margin = y * (np.dot(X, self.w) + self.b)
            # 找出违反间隔条件的样本（margin < 1）
            idx = np.where(margin < 1)[0]

            # 计算梯度
            # L2正则化项 + 错误分类样本的平均梯度
            dw = (2 * self.reg_lambda * self.w) - np.mean(y[idx].reshape(-1, 1) * X[idx], axis = 0)
            db = -np.mean(y[idx])

            # 参数更新
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, x):
        """预测标签。"""
        score = np.dot(x, self.w) + self.b     # 计算得分
        return np.where(score >= 0, 1, 0)      # 转换回{0, 1}标签

if __name__ == '__main__':
    # 数据加载部分

    # 获取当前脚本所在目录，确保路径正确
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建训练集和测试集的绝对路径，指向src/chap03_SVM/data目录
    train_file = os.path.join(base_dir, 'data', 'train_linear.txt')  # 训练集文件路径
    test_file = os.path.join(base_dir, 'data', 'test_linear.txt')    # 测试集文件路径

    # 加载训练数据
    data_train = load_data(train_file)

    # 加载测试数据
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM()            # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    # 从训练数据中提取前两列作为特征(x1, x2)
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]   # 训练集标签
    # 使用训练好的SVM模型对训练数据进行标签预测
    t_train_pred = svm.predict(x_train)     # 预测标签

    x_test = data_test[:, :2]    # 测试集特征
    t_test = data_test[:, 2]     # 测试集标签
    # 对测试数据进行预测，获取预测标签
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
