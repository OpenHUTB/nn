# python: 3.5.2
# encoding: utf-8

# 导入NumPy科学计算库，并使用缩写np简化调用
import numpy as np
import os  # 用于处理文件路径和系统操作

def load_data(fname):
    """载入数据并进行预处理
    参数:
        fname: 数据文件路径
    返回:
        处理后的numpy数组，shape=(m,3)，前两列为特征，第三列为标签
    """
    # 检查文件是否存在，若不存在则抛出包含当前工作目录的详细异常
    if not os.path.exists(fname):
        raise FileNotFoundError(f"数据文件未找到: {fname}\n当前工作目录: {os.getcwd()}")
    
    with open(fname, 'r') as f:
        data = []
        next(f)  # 读取并丢弃第一行（通常为标题行）
        for line in f:
            line = line.strip().split()  # 去除首尾空白并按空格分割
            x1 = float(line[0])  # 第一个特征
            x2 = float(line[1])  # 第二个特征
            t = int(line[2])    # 标签（0或1）
            data.append([x1, x2, t])
        return np.array(data)  # 转换为numpy数组便于矩阵运算

def eval_acc(label, pred):
    """计算分类准确率
    参数:
        label: 真实标签数组
        pred: 预测标签数组
    返回:
        正确预测的样本比例（0-1之间）
    """
    return np.sum(label == pred) / len(pred)  # 统计相等元素比例

class SVM:
    """线性支持向量机模型（基于hinge loss和L2正则化）"""

    def __init__(self):
        """初始化模型超参数"""
        self.learning_rate = 0.01  # 梯度下降步长
        self.reg_lambda = 0.01     # L2正则化强度（防止过拟合）
        self.max_iter = 1000       # 最大训练迭代次数
        self.w = None              # 权重向量（特征权重）
        self.b = None              # 偏置项（超平面截距）

    def train(self, data_train):
        """训练SVM模型
        参数:
            data_train: 训练数据，shape=(m,3)，前两列特征，第三列标签(0/1)
        """
        X = data_train[:, :2]       # 提取特征矩阵(m,2)
        y = data_train[:, 2]        # 提取标签向量(m,)
        y = np.where(y == 0, -1, 1) # 转换标签为{-1,1}（SVM标准标签格式）
        m, n = X.shape              # m=样本数，n=特征数（此处n=2）

        # 初始化模型参数
        self.w = np.zeros(n)        # 权重向量初始化为0向量
        self.b = 0                  # 偏置初始化为0

        for epoch in range(self.max_iter):
            # 计算每个样本的函数间隔：y*(w·x + b)
            margin = y * (np.dot(X, self.w) + self.b)
            # 找出违反间隔条件的样本（margin < 1）
            idx = np.where(margin < 1)[0]

            # 若所有样本都满足间隔条件，提前结束训练
            if len(idx) == 0:
                print(f"Epoch {epoch}: 所有样本满足间隔条件，提前终止训练")
                break

            # 计算梯度（包含L2正则化项）
            # 正则化项梯度 + 错误样本平均梯度
            dw = 2 * self.reg_lambda * self.w - np.mean(y[idx].reshape(-1, 1) * X[idx], axis=0)
            db = -np.mean(y[idx])

            # 梯度下降更新参数
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, x):
        """预测样本标签
        参数:
            x: 输入特征矩阵，shape=(m,2)
        返回:
            预测标签数组，值为0或1
        """
        score = np.dot(x, self.w) + self.b  # 计算决策函数值
        return np.where(score >= 0, 1, 0)   # 转换为{0,1}标签格式

if __name__ == '__main__':
    """主程序：训练并评估SVM模型"""
    # 获取当前脚本所在目录，确保数据文件路径正确
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, 'data', 'train_linear.txt')
    test_file = os.path.join(base_dir, 'data', 'test_linear.txt')

    try:
        # 加载训练数据和测试数据
        data_train = load_data(train_file)
        data_test = load_data(test_file)

        # 训练SVM模型
        svm = SVM()
        svm.train(data_train)

        # 模型预测
        x_train, t_train = data_train[:, :2], data_train[:, 2]
        x_test, t_test = data_test[:, :2], data_test[:, 2]
        
        t_train_pred = svm.predict(x_train)
        t_test_pred = svm.predict(x_test)

        # 计算准确率
        acc_train = eval_acc(t_train, t_train_pred)
        acc_test = eval_acc(t_test, t_test_pred)

        # 输出结果
        print(f"训练集准确率: {acc_train*100:.1f}%")
        print(f"测试集准确率: {acc_test*100:.1f}%")

    except FileNotFoundError as e:
        print(f"数据加载错误: {e}")
        print("请确认data目录下包含train_linear.txt和test_linear.txt")
