import numpy as np
import os

def load_data(fname):
    """载入数据。"""
    # 检查文件是否存在，确保数据加载的可靠性
    if not os.path.exists(fname):
        raise FileNotFoundError(f"数据文件未找到: {fname}\n请确认文件路径是否正确，当前工作目录为: {os.getcwd()}")
    with open(fname, 'r') as f:
        data = []
        line = f.readline()  # 跳过表头行
        for line in f:
            line = line.strip().split()  # 去除空白并按空格分割
            x1 = float(line[0])  # 特征1：例如坐标x
            x2 = float(line[1])  # 特征2：例如坐标y
            t = int(line[2])     # 标签：0或1
            data.append([x1, x2, t])
        return np.array(data)  # 返回numpy数组，便于矩阵运算

def eval_acc(label, pred):
    """计算准确率。
    
    参数:
        label: 真实标签数组
        pred: 预测标签数组
        
    返回:
        准确率 (0到1之间的浮点数)
    """
    return np.sum(label == pred) / len(pred)  # 正确预测的样本比例

class SVM:
    """SVM模型：基于最大间隔分类的监督学习算法。"""

    def __init__(self):
        # 超参数设置
        self.learning_rate = 0.01  # 控制梯度下降步长
        self.reg_lambda = 0.01     # L2正则化系数，平衡间隔最大化与分类误差
        self.max_iter = 1000       # 最大训练迭代次数
        self.w = None              # 权重向量，决定分类超平面的方向
        self.b = None              # 偏置项，决定分类超平面的位置

    def train(self, data_train):
        """训练SVM模型（基于hinge loss + L2正则化）
        
        算法核心：
        1. 寻找能最大化间隔的超平面 wx + b = 0
        2. 间隔定义为：样本到超平面的最小距离
        3. 使用hinge loss处理分类错误和边界样本
        4. 添加L2正则化防止过拟合
        """
        X = data_train[:, :2]         # 提取特征矩阵
        y = data_train[:, 2]          # 提取标签
        y = np.where(y == 0, -1, 1)   # 将标签转换为{-1, 1}，符合SVM理论要求
        m, n = X.shape                # m:样本数，n:特征数

        # 初始化模型参数
        self.w = np.zeros(n)  # 权重向量初始化为0
        self.b = 0            # 偏置项初始化为0

        for epoch in range(self.max_iter):
            # 计算函数间隔：y(wx+b)，衡量样本到超平面的距离和方向
            margin = y * (np.dot(X, self.w) + self.b)
            
            # 找出违反间隔条件的样本（margin < 1）
            # 这些样本包括：误分类样本(margin<0)和间隔内样本(0<=margin<1)
            idx = np.where(margin < 1)[0]
            
            # 如果所有样本都满足margin>=1，说明已找到完美超平面
            # 移除continue语句，确保即使所有样本都满足间隔条件
            # 也会更新权重以优化正则化项

            # 计算梯度：正则化项梯度 + 误分类样本梯度
            # L2正则化：减小权重，防止过拟合
            # hinge loss梯度：只对误分类和边界样本计算梯度
            dw = (2 * self.reg_lambda * self.w) - np.mean(y[idx].reshape(-1, 1) * X[idx], axis=0)
            db = -np.mean(y[idx])

            # 梯度下降更新参数
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # 训练逻辑总结：
            # - 对误分类样本，向正确方向调整超平面
            # - 对间隔内样本，微调超平面使其远离
            # - 正则化项约束权重大小，使间隔更平滑

    def predict(self, x):
        """预测标签。
        
        预测逻辑：
        1. 计算样本到超平面的有符号距离 wx + b
        2. 距离为正 -> 预测为正类(1)
        3. 距离为负 -> 预测为负类(0)
        """
        score = np.dot(x, self.w) + self.b     # 计算决策函数值
        return np.where(score >= 0, 1, 0)      # 转换回{0, 1}标签格式

if __name__ == '__main__':
    # 数据加载部分
    base_dir = os.path.dirname(os.path.abspath(__file__))             # 获取当前脚本的绝对路径
    train_file = os.path.join(base_dir, 'data', 'train_linear.txt')   # 拼接训练数据文件路径
    test_file = os.path.join(base_dir, 'data', 'test_linear.txt')     # 拼接测试数据文件路径

    # 加载训练数据
    data_train = load_data(train_file)
    # 加载测试数据
    data_test = load_data(test_file)

    # 模型训练
    svm = SVM()            # 初始化SVM模型
    svm.train(data_train)  # 训练模型寻找最优超平面

    # 训练集评估
    x_train = data_train[:, :2]  # 训练特征
    t_train = data_train[:, 2]   # 训练标签
    t_train_pred = svm.predict(x_train)  # 预测训练集标签

    # 测试集评估
    x_test = data_test[:, :2]    # 测试特征
    t_test = data_test[:, 2]     # 测试标签
    t_test_pred = svm.predict(x_test)  # 预测测试集标签

    # 计算并打印准确率
    acc_train = eval_acc(t_train, t_train_pred)  # 训练集准确率
    acc_test = eval_acc(t_test, t_test_pred)     # 测试集准确率
    
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
