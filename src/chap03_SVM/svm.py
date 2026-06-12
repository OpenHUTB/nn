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
    """SVM模型：基于最大间隔分类的监督学习算法。

    改进点：
    1. 修复 predict() 中 y_train_unique 未初始化的 bug
    2. 添加学习率指数衰减，兼顾快速收敛和精细调优
    3. 添加训练损失计算与进度打印，便于监控收敛
    4. 添加早停机制，收敛后自动停止避免浪费计算
    5. 默认开启 L2 正则化，防止过拟合
    """

    def __init__(self, learning_rate=0.1, reg_lambda=0.001, max_iter=20000,
                 lr_decay=0.9995, print_interval=2000, patience=2000):
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.lr_decay = lr_decay          # 学习率衰减系数
        self.print_interval = print_interval  # 打印间隔
        self.patience = patience          # 早停耐心值（连续无改善的轮数）
        self.w = None
        self.b = None
        self.scaler = StandardScaler()

    def _compute_hinge_loss(self, X, y):
        """计算 hinge loss + L2 正则化损失"""
        score = np.dot(X, self.w) + self.b
        hinge = np.maximum(0, 1 - y * score)
        return np.mean(hinge) + self.reg_lambda * np.dot(self.w, self.w)

    def train(self, data_train):
        """训练SVM模型（基于hinge loss + L2正则化）

        改进：
        - 学习率指数衰减：lr = lr_0 * decay^epoch
        - 每个 epoch 计算损失并打印进度
        - 早停：连续 patience 轮损失不再下降则停止
        """
        X_raw = data_train[:, :2]
        y_raw = data_train[:, 2]

        # 记录原始标签空间，供 predict() 使用（修复原 bug）
        self.label_set = np.unique(y_raw)

        X = self.scaler.fit_transform(X_raw)
        y = np.where(y_raw <= 0, -1, 1)

        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        lr = self.learning_rate
        best_loss = float('inf')
        no_improve_count = 0

        for epoch in range(self.max_iter):
            score = np.dot(X, self.w) + self.b
            margin = y * score
            idx = np.where(margin < 1)[0]

            if len(idx) > 0:
                dw = (2 * self.reg_lambda * self.w) - np.sum(y[idx, None] * X[idx], axis=0) / m
                db = -np.mean(y[idx])
            else:
                dw = 2 * self.reg_lambda * self.w
                db = 0

            self.w -= lr * dw
            self.b -= lr * db

            # 学习率衰减
            lr = self.learning_rate * (self.lr_decay ** epoch)

            # 定期打印训练进度
            if (epoch + 1) % self.print_interval == 0 or epoch == 0:
                loss = self._compute_hinge_loss(X, y)
                pred = np.where(np.dot(X, self.w) + self.b >= 0, 1, -1)
                acc = np.mean(pred == y)
                print(f"  epoch {epoch+1:>6d} | loss: {loss:.4f} | acc: {acc*100:.1f}% | lr: {lr:.6f}")

            # 早停检查：每 print_interval 轮检查一次损失
            if (epoch + 1) % self.print_interval == 0:
                loss = self._compute_hinge_loss(X, y)
                if loss < best_loss - 1e-6:
                    best_loss = loss
                    no_improve_count = 0
                else:
                    no_improve_count += self.print_interval
                    if no_improve_count >= self.patience:
                        print(f"  早停触发于 epoch {epoch+1}，最佳损失: {best_loss:.4f}")
                        break

    def predict(self, x_raw):
        """预测标签。

        修复：使用 train() 中记录的 self.label_set 自动判断返回格式，
        不再依赖 train_with_label_tracking()。
        """
        x = self.scaler.transform(x_raw)
        score = np.dot(x, self.w) + self.b
        # 根据训练数据的标签空间决定输出格式
        if -1 in self.label_set:
            return np.where(score >= 0, 1, -1)
        else:
            return np.where(score >= 0, 1, 0)


if __name__ == '__main__':
    # 数据加载部分以及数据路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_train = os.path.join(base_dir, 'data', 'train_linear.txt')
    default_test = os.path.join(base_dir, 'data', 'test_linear.txt')

    parser = argparse.ArgumentParser(description='Linear SVM training script')
    parser.add_argument('--train-file', type=str, default=default_train, help='训练集文件路径')
    parser.add_argument('--test-file', type=str, default=default_test, help='测试集文件路径')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='学习率')
    parser.add_argument('--reg-lambda', type=float, default=0.001, help='L2正则化系数')
    parser.add_argument('--max-iter', type=int, default=20000, help='最大迭代次数')
    parser.add_argument('--lr-decay', type=float, default=0.9995, help='学习率衰减系数')
    parser.add_argument('--patience', type=int, default=2000, help='早停耐心值')
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
        lr_decay=args.lr_decay,
        patience=args.patience,
    )
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
        'lr_decay': float(args.lr_decay),
        'patience': int(args.patience),
        'train_file': str(train_file),
        'test_file': str(test_file),
    }
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"metrics saved: {metrics_path.resolve()}")

