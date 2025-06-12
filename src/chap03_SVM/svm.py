import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple

def load_data(fname: str) -> np.ndarray:
    """
    加载数据文件，自动处理路径和数据格式
    
    参数:
        fname: 数据文件路径
    
    返回:
        包含特征和标签的二维数组，形状为(n_samples, n_features+1)
    """
    file_path = Path(fname).resolve()
    if not file_path.exists():
        raise FileNotFoundError(
            f"数据文件未找到: {file_path}\n"
            f"当前工作目录为: {Path.cwd()}"
        )
    
    with file_path.open('r') as f:
        # 自动跳过表头（假设首行以#开头或包含标题）
        first_line = f.readline()
        while first_line.startswith('#') or 'label' in first_line.lower():
            first_line = f.readline()
        
        data = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            data.append(parts)
    
    return np.array(data, dtype=np.float32)

def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    特征标准化处理（均值为0，方差为1）
    
    参数:
        X: 特征矩阵，形状为(n_samples, n_features)
    
    返回:
        标准化后的特征矩阵，均值向量，标准差向量
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)  # 使用无偏标准差
    std = np.where(std < 1e-8, 1.0, std)  # 防止除零
    X_standardized = (X - mean) / std
    return X_standardized, mean, std

def eval_acc(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    计算准确率（包含边界检查）
    
    参数:
        true_labels: 真实标签数组
        pred_labels: 预测标签数组
    
    返回:
        准确率（0到1之间的浮点数）
    
    抛出:
        ValueError: 标签数组长度不一致
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("真实标签和预测标签长度必须一致")
    return np.mean(true_labels == pred_labels)

class SVM:
    """
    线性SVM分类器（使用随机梯度下降优化Hinge Loss + L2正则化）
    
    参数:
        learning_rate: 学习率（默认0.01）
        reg_lambda: L2正则化系数（默认0.01）
        max_iter: 最大迭代次数（默认1000）
        tol: 收敛阈值（默认1e-3）
        random_state: 随机种子（默认None）
    """
    def __init__(
        self,
        learning_rate: float = 0.01,
        reg_lambda: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: Optional[int] = None
    ):
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.w: Optional[np.ndarray] = None  # 权重向量
        self.b: float = 0.0                # 偏置项
        self.loss_history: list = []       # 损失历史记录
        self.rng = np.random.RandomState(random_state)
        
    def _hinge_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算Hinge Loss（带L2正则化）"""
        margins = y * (np.dot(X, self.w) + self.b)
        hinge_loss = np.mean(np.maximum(0, 1 - margins))
        l2_loss = self.reg_lambda * np.sum(self.w ** 2)
        return hinge_loss + l2_loss
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        standardize: bool = True
    ) -> None:
        """
        训练SVM模型
        
        参数:
            X: 特征矩阵，形状为(n_samples, n_features)
            y: 标签数组（0/1格式，将自动转为-1/1）
            standardize: 是否进行特征标准化（默认True）
        """
        # 数据预处理
        y = np.where(y == 0, -1, 1).astype(np.float32)  # 转换为-1/1标签
        if standardize:
            X, self.mean, self.std = standardize_features(X)
        else:
            self.mean = np.zeros(X.shape[1])
            self.std = np.ones(X.shape[1])
        
        m, n = X.shape
        self.w = self.rng.randn(n).astype(np.float32)  # 随机初始化权重
        self.b = 0.0
        
        for epoch in range(self.max_iter):
            # 随机选择单个样本进行SGD
            idx = self.rng.randint(m)
            xi = X[idx]
            yi = y[idx]
            
            # 计算函数间隔
            margin = yi * (np.dot(xi, self.w) + self.b)
            
            # 计算梯度
            if margin < 1:
                # 对违反间隔的样本进行梯度更新
                dw = 2 * self.reg_lambda * self.w - yi * xi
                db = -yi
            else:
                # 仅正则化项梯度
                dw = 2 * self.reg_lambda * self.w
                db = 0.0
            
            # 梯度下降更新
            self.w -= self.learning_rate * dw / m
            self.b -= self.learning_rate * db / m
            
            # 记录损失并检查收敛
            if epoch % 100 == 0:
                loss = self._hinge_loss(X, y)
                self.loss_history.append(loss)
                if len(self.loss_history) > 1:
                    if np.abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                        print(f"提前停止，迭代次数: {epoch}")
                        break
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本标签
        
        参数:
            X: 特征矩阵，形状为(n_samples, n_features)
        
        返回:
            预测标签数组（0/1格式）
        """
        # 特征标准化（使用训练时的均值和标准差）
        X_standardized = (X - self.mean) / self.std
        scores = np.dot(X_standardized, self.w) + self.b
        return np.where(scores >= 0, 1, 0).astype(np.int32)

if __name__ == '__main__':
    # 路径处理优化
    base_path = Path(__file__).parent.resolve()
    data_dir = base_path / "data"
    train_file = data_dir / "train_linear.txt"
    test_file = data_dir / "test_linear.txt"
    
    # 加载数据
    try:
        data_train = load_data(train_file)
        data_test = load_data(test_file)
    except FileNotFoundError as e:
        print(f"数据加载失败: {e}")
        exit(1)
    
    # 拆分特征和标签
    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_test, y_test = data_test[:, :-1], data_test[:, -1]
    
    # 模型训练
    svm = SVM(
        learning_rate=0.001,
        reg_lambda=0.1,
        max_iter=2000,
        tol=1e-4,
        random_state=42
    )
    svm.train(X_train, y_train)
    
    # 模型评估
    def evaluate_model(model, X, y, name: str):
        pred = model.predict(X)
        acc = eval_acc(y, pred)
        print(f"{name}准确率: {acc * 100:.1f}%")
    
    evaluate_model(svm, X_train, y_train, "训练集")
    evaluate_model(svm, X_test, y_test, "测试集")