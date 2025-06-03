# python: 3.5.2
# encoding: utf-8

import numpy as np

def load_data(fname):
    """载入数据文件，并将其转换为数值数组。
    
    参数:
        fname (str): 文件路径，指向包含数据的文本文件。
    
    返回:
        numpy.ndarray: 一个二维数组，每行对应一个数据样本，格式为 [x1, x2, t]。
    """

    # 打开文件（只读模式），自动处理文件关闭
    with open(fname, 'r') as f:
        data = []  # 创建一个空列表，用于存储每一行数据

        line = f.readline()  # 读取首行（通常是标题），这里自动跳过标题行，不作处理

        # 遍历剩下的每一行
        for line in f:
            # 去除每行首尾的空白字符，并按空格或制表符进行分割
            line = line.strip().split()

            # 将前两个字段转换为浮点数（特征值）
            x1 = float(line[0])
            x2 = float(line[1])

            # 将第三个字段转换为整数（标签/类别）
            t = int(line[2])

            # 将这三个值组成一个列表，并添加到 data 列表中
            data.append([x1, x2, t])

        # 将整个列表转换为 NumPy 数组，方便后续数值计算和处理
        return np.array(data)

def eval_acc(label, pred):
    """计算准确率"""
    return np.sum(label == pred) / len(pred)#准确率 = 正确预测的样本数 / 总样本数
  
#SVM模型 实现了线性SVM分类
class SVM():
    """SVM模型"""
    #目标函数：(1/2)||w||² + C * Σmax(0, 1 - y_i(w·x_i + b))
    def __init__(self, lr=0.001, epochs=1000, lambda_=0.001, tolerance=1e-3):
        # 补全此处代码
        self.w = None  # w: 权重向量(决定分类超平面的方向)
        self.b = 0     # b: 偏置项(决定分类超平面的位置)
        self.lr = lr   # 学习率
        self.epochs = epochs  # 训练轮数
        self.lambda_ = lambda_  # 正则化参数
        self.tolerance = tolerance  # 提前停止的阈值
        pass
    

    def train(self, data_train):
        """
        训练 SVM 模型。
        :param data_train: 包含特征和标签的 NumPy 数组，形状为 (n_samples, n_features + 1)
        """
        X = data_train[:, :-1] #从data_train中提取特征矩阵X
        y = np.where(data_train[:, -1] == 0, -1, 1) #处理标签列，将0类标签转换为-1，非0类标签转换为1，data_train[:, -1]选择最后一列(标签列)，np.where(condition, x, y)：如果condition为True则选x，否则选y
        y = data_train[:, -1] #处理标签列，将0类标签转换为-1，非0类标签转换为1，data_train[:, -1]选择最后一列(标签列)，np.where(condition, x, y)：如果condition为True则选x，否则选y

        n_samples, n_features = X.shape #获取样本数量和特征数量
        self.w = np.zeros(n_features) #初始化权重向量w为零向量
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
        预测标签
        """
        # 请补全此处代码
        # 计算决策函数值
        if x.ndim == 1:
           x = np.expand_dims(x, axis=0)  # 处理单样本输入
        decision_values = np.dot(x, self.w) + self.b  # logits = x·w + b
        # 返回预测标签（0或1）
        return np.where(decision_values >= 0, 1, -1)


if __name__ == '__main__':
    # 载入数据，实际使用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    #print(data_train[:1000])  # 查看前5行数据

    # 使用训练集训练SVM模型
    svm = SVM()  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]  # 提取测试集特征（x1, x2）
    t_test = data_test[:, 2] # 提取测试集真实标签
    t_test_pred = svm.predict(x_test) # 对测试集进行预测，得到预测标签

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
