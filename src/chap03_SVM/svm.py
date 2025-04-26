# python: 3.5.2
# encoding: utf-8

import numpy as np


def load_data(fname):
    """
    载入数据文件，返回特征和标签的numpy数组。
    
    参数:
    fname (str): 数据文件路径
    
    返回:
    np.ndarray: 形状为(N,3)的数组，每行包含[x1, x2, 标签]
    """
    with open(fname, 'r') as f:
        data = []
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])  # 提取第一个特征值
            x2 = float(line[1])  # 提取第二个特征值
            t = int(line[2])     # 提取标签（0或1）
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算分类准确率。
    
    参数:
    label (np.ndarray): 真实标签数组
    pred (np.ndarray): 预测标签数组
    
    返回:
    float: 准确率（0到1之间的浮点数）
    """
    correct = np.sum(label == pred)  # 统计预测正确的样本数
    total = len(pred)                # 总样本数
    return correct / total          # 返回准确率


class SVM():
    """
    支持向量机（SVM）分类模型。
    使用Hinge Loss和随机梯度下降（SGD）进行训练。
    """
    def __init__(self, learning_rate=0.01, lambda_=0.01, epochs=1000):
        """
        初始化模型参数。
        
        参数:
        learning_rate (float): 学习率（默认0.01）
        lambda_ (float): L2正则化系数（默认0.01）
        epochs (int): 训练迭代次数（默认1000）
        """
        self.w = None  # 权重向量（特征维度）
        self.b = 0.0   # 偏置项初始值设为0
        self.learning_rate = learning_rate
        self.lambda_ = lambda_     # 控制正则化强度
        self.epochs = epochs       # 总迭代次数
        
    def train(self, data_train):
        """
        使用训练数据训练SVM模型。
        
        参数:
        data_train (np.ndarray): 训练数据，形状(N,3)
        
        训练过程:
        1. 提取特征和标签
        2. 初始化权重向量
        3. 使用随机梯度下降进行迭代优化
        4. 根据Hinge Loss计算梯度并更新参数
        """
        # 提取特征和标签
        x = data_train[:, :2]  # 特征矩阵(N,2)
        t = data_train[:, 2]   # 标签向量(N,)
        
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)  # 初始化权重为0向量
        
        # 随机梯度下降优化
        for _ in range(self.epochs):
            # 打乱样本顺序提升训练效果
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x_i = x[idx]  # 当前样本的特征向量
                t_i = t[idx]  # 当前样本的真实标签
                
                # 计算预测值的条件：t_i*(w·x + b) ≥ 1
                condition = t_i * (np.dot(x_i, self.w) + self.b)
                
                if condition >= 1:
                    # 当前样本满足分类条件（正确分类且足够远）
                    # 只需更新正则化项对应的梯度
                    grad_w = self.lambda_ * self.w  # 正则化梯度分量
                    self.w -= self.learning_rate * grad_w
                else:
                    # 当前样本未满足分类条件（误分类或距离不足）
                    # 需要同时更新分类损失和正则化梯度
                    grad_w = self.lambda_ * self.w - t_i * x_i
                    grad_b = -t_i
                    # 更新权重和偏置
                    self.w -= self.learning_rate * grad_w
                    self.b -= self.learning_rate * grad_b

    def predict(self, x):
        """
        使用训练好的模型进行预测。
        
        参数:
        x (np.ndarray): 测试特征数据，形状(N,2)
        
        返回:
        np.ndarray: 预测标签数组，元素为0或1
        """
        decision_values = np.dot(x, self.w) + self.b  # 计算决策函数值
        # 根据决策函数值符号判断类别
        return np.where(decision_values >= 0, 1, 0)  # 转换为二分类标签(0/1)


if __name__ == '__main__':
    # 数据加载与预处理
    train_file = 'data/train_linear.txt'  # 训练数据文件路径
    test_file = 'data/test_linear.txt'    # 测试数据文件路径
    data_train = load_data(train_file)    # 载入训练数据
    data_test = load_data(test_file)      # 载入测试数据
    
    # 模型训练阶段
    svm = SVM(learning_rate=0.01, lambda_=0.01, epochs=1000)  # 初始化SVM模型
    svm.train(data_train)  # 启动模型训练
    
    # 预测与评估
    # 训练集评估
    x_train = data_train[:, :2]
    t_train = data_train[:, 2]
    t_train_pred = svm.predict(x_train)  # 预测训练集标签
    acc_train = eval_acc(t_train, t_train_pred)  # 计算训练集准确率
    
    # 测试集评估
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)
    acc_test = eval_acc(t_test, t_test_pred)  # 计算测试集准确率
    
    # 输出结果
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))