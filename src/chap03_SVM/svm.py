import numpy as np
import os
import argparse
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def load_data(fname):
    """载入数据。"""
    # 检查文件是否存在，确保数据加载的可靠性
    if not os.path.exists(fname): 
        raise FileNotFoundError(f"数据文件未找到: {fname}\n请确认文件路径是否正确，当前工作目录为: {os.getcwd()}") # 如果文件不存在，抛出异常
    with open(fname, 'r') as f: # 打开文件
        data = [] # 初始化一个空列表，用于存储数据
        line = f.readline()  # 跳过表头行
        for line in f:
            line = line.strip().split()  # 去除空白并按空格分割
            x1 = float(line[0])  # 特征1：例如坐标x
            x2 = float(line[1])  # 特征2：例如坐标y
            t = int(float(line[2]))     # 标签：处理可能存在的浮点数字符串
            data.append([x1, x2, t])
        return np.array(data)  # 返回numpy数组，便于矩阵运算

def eval_acc(label, pred):
    """计算准确率。
    
    参数:
        label: 真实标签的数组
        pred: 预测标签的数组
        
    返回:
        准确率 (0到1之间的浮点数)
    """
    return np.sum(label == pred) / len(pred)  # 正确预测的样本比例

class SVM:
    """SVM模型：基于最大间隔分类的监督学习算法。"""
#支持向量机（Support Vector Machine, SVM） 是一种经典的监督学习算法，主要用于分类（也可用于回归和异常检测）。
    def __init__(self, learning_rate=0.1, reg_lambda=0.0, max_iter=20000):
        # 超参数设置
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.w = None              # 权重向量，决定分类超平面的方向
        self.b = None              # 偏置项，决定分类超平面的位置
        self.scaler = StandardScaler() # 添加标准化器

    def train(self, data_train):
        """训练SVM模型（基于hinge loss + L2正则化）
        
        算法核心：
        1. 寻找能最大化间隔的超平面 wx + b = 0
        2. 间隔定义为：样本到超平面的最小距离
        3. 使用hinge loss处理分类错误和边界样本
        4. 添加L2正则化防止过拟合
        """
        X_raw = data_train[:, :2]     # 提取原始特征矩阵
        y_raw = data_train[:, 2]      # 提取原始标签
        
        # 标准化特征 (SVM 对特征缩放非常敏感)
        X = self.scaler.fit_transform(X_raw)
        
        # 修正标签转换逻辑：确保映射到 {-1, 1}
        # 如果原始标签是 {-1, 1}，y <= 0 保持 -1 为 -1
        # 如果原始标签是 {0, 1}，y <= 0 将 0 映射为 -1
        y = np.where(y_raw <= 0, -1, 1)
        
        m, n = X.shape                # m:样本数，n:特征数

        # 初始化模型参数 (使用随机小值可能有助于打破对称性，但这里全零也可以)
        self.w = np.zeros(n)
        self.b = 0

        for epoch in range(self.max_iter):
            # 计算决策得分: score = wx + b
            score = np.dot(X, self.w) + self.b
            
            # 计算函数间隔：y * score
            margin = y * score
            
            # 找出违反间隔条件的样本 (hinge loss 区域: y*f(x) < 1)
            idx = np.where(margin < 1)[0]
            
            # 计算梯度
            # 损失函数: L = (1/m) * sum(max(0, 1 - y*f(x))) + lambda * ||w||^2
            if len(idx) > 0:
                # dw = d/dw [lambda * ||w||^2 + (1/m) * sum(1 - y*(wx+b))]
                # dw = 2 * lambda * w - (1/m) * sum(y*x)
                dw = (2 * self.reg_lambda * self.w) - np.sum(y[idx, None] * X[idx], axis=0) / m
                db = -np.mean(y[idx])
            else:
                # 只有正则化项的梯度
                dw = 2 * self.reg_lambda * self.w
                db = 0

            # 梯度下降更新参数
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, x_raw):
        """预测标签。"""
        # 使用训练时的标准化参数处理新数据
        x = self.scaler.transform(x_raw)
        score = np.dot(x, self.w) + self.b     # 计算决策函数值
        
        # 返回原始标签空间 (这里假设原始正类是 1, 负类是 0 或 -1)
        # 如果 load_data 将标签读为 0/1，则应返回 0/1
        # 如果是 -1/1，则应返回 -1/1
        # 这里为了兼容 eval_acc，我们返回与输入数据一致的标签
        return np.where(score >= 0, 1, -1) if -1 in self.y_train_unique else np.where(score >= 0, 1, 0)

    def train_with_label_tracking(self, data_train):
        """带有标签信息记录的训练方法"""
        self.y_train_unique = np.unique(data_train[:, 2])
        self.train(data_train)


if __name__ == '__main__':
    # 数据加载部分以及数据路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_train = os.path.join(base_dir, 'data', 'train_linear.txt')
    default_test = os.path.join(base_dir, 'data', 'test_linear.txt')

    parser = argparse.ArgumentParser(description='Linear SVM training script')
    parser.add_argument('--train-file', type=str, default=default_train, help='训练集文件路径')
    parser.add_argument('--test-file', type=str, default=default_test, help='测试集文件路径')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='学习率')
    parser.add_argument('--reg-lambda', type=float, default=0.0, help='L2正则化系数')
    parser.add_argument('--max-iter', type=int, default=20000, help='最大迭代次数')
    parser.add_argument('--out-dir', type=str, default='outputs', help='结果输出目录')
    args = parser.parse_args()

    train_file = args.train_file if os.path.isabs(args.train_file) else os.path.join(base_dir, args.train_file)
    test_file = args.test_file if os.path.isabs(args.test_file) else os.path.join(base_dir, args.test_file)

    # 加载训练数据
    data_train = load_data(train_file)
    # 加载测试数据
    data_test = load_data(test_file)

    # 模型训练
    svm = SVM(
        learning_rate=args.learning_rate,
        reg_lambda=args.reg_lambda,
        max_iter=args.max_iter,
    )
    svm.train_with_label_tracking(data_train)  # 训练模型寻找最优超平面

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
    
    print("train accuracy: {:.1f}%".format(acc_train * 100))  # 输出训练集准确率
    print("test accuracy: {:.1f}%".format(acc_test * 100))  # 输出测试集准确率


    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / 'svm_metrics.json'
    metrics = {
        'train_accuracy': float(acc_train),
        'test_accuracy': float(acc_test),
        'learning_rate': float(args.learning_rate),
        'reg_lambda': float(args.reg_lambda),
        'max_iter': int(args.max_iter),
        'train_file': str(train_file),
        'test_file': str(test_file),
    }
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"metrics saved: {metrics_path.resolve()}")

