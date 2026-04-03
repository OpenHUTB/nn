import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# ===================== 1. 纯NumPy实现孤立森林 =====================
class IsolationForestPure:
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.05, random_state=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        np.random.seed(random_state)
        self.trees = []  # 存储所有孤立树
        self.max_depth = None  # 树的最大深度
        self.sample_size = None  # 每棵树的采样数量

    def _init_params(self, X):
        """初始化参数：采样数量、最大树深度"""
        n_samples = X.shape[0]
        # 确定采样数量
        if self.max_samples == 'auto':
            self.sample_size = min(256, n_samples)
        elif isinstance(self.max_samples, int):
            self.sample_size = min(self.max_samples, n_samples)
        else:  # 浮点数（比例）
            self.sample_size = int(self.max_samples * n_samples)
        # 最大深度：log2(采样数量)
        self.max_depth = int(np.ceil(np.log2(max(2, self.sample_size))))

    def _build_tree(self, X):
        """构建单棵孤立树"""
        # 随机采样
        sample_idx = np.random.choice(X.shape[0], self.sample_size, replace=False)
        X_sample = X[sample_idx]

        # 递归构建树节点
        def build_node(X_node, depth):
            if depth >= self.max_depth or X_node.shape[0] <= 1:
                return {'is_leaf': True, 'size': X_node.shape[0]}

            # 随机选择特征和分割点
            n_features = X_node.shape[1]
            feature_idx = np.random.choice(n_features)
            feature_vals = X_node[:, feature_idx]
            min_val, max_val = np.min(feature_vals), np.max(feature_vals)

            if min_val == max_val:  # 特征值无差异，无法分割
                return {'is_leaf': True, 'size': X_node.shape[0]}

            # 随机选择分割点
            split_val = np.random.uniform(min_val, max_val)

            # 分割数据
            left_idx = X_node[:, feature_idx] < split_val
            X_left = X_node[left_idx]
            X_right = X_node[~left_idx]

            return {
                'is_leaf': False,
                'feature_idx': feature_idx,
                'split_val': split_val,
                'left': build_node(X_left, depth + 1),
                'right': build_node(X_right, depth + 1)
            }

        return build_node(X_sample, depth=0)

    def fit(self, X):
        """训练孤立森林"""
        X = np.array(X, dtype=np.float64)
        self._init_params(X)
        # 构建多棵树
        for _ in range(self.n_estimators):
            tree = self._build_tree(X)
            self.trees.append(tree)

    def _path_length(self, x, tree):
        """计算单个样本在单棵树中的路径长度"""

        def traverse(node, depth):
            if node['is_leaf']:
                # 叶子节点：路径长度 += 调整项（针对小样本）
                n = node['size']
                if n <= 1:
                    return depth
                # 调和数近似：H(n-1) + 0.5772156649（欧拉常数）
                h = np.log(n - 1) + 0.5772156649
                return depth + h
            # 非叶子节点：继续遍历
            feature_val = x[node['feature_idx']]
            if feature_val < node['split_val']:
                return traverse(node['left'], depth + 1)
            else:
                return traverse(node['right'], depth + 1)

        return traverse(tree, depth=0)

    def decision_function(self, X):
        """计算异常评分（越低越异常）"""
        X = np.array(X, dtype=np.float64)
        # 计算所有样本在所有树中的平均路径长度
        path_lengths = []
        for x in X:
            lengths = [self._path_length(x, tree) for tree in self.trees]
            avg_length = np.mean(lengths)
            path_lengths.append(avg_length)

        # 归一化路径长度（转换为异常评分，范围[-1,1]）
        c = self._average_path_length(self.sample_size)  # 平均路径长度基准
        scores = 2 ** (-np.array(path_lengths) / c)
        # 调整评分范围为[-1,1]（与sklearn对齐）
        scores = - (scores - np.percentile(scores, 100 * (1 - self.contamination)))
        return scores

    def predict(self, X):
        """预测异常标签（-1=异常，1=正常）"""
        scores = self.decision_function(X)
        # 根据异常比例确定阈值
        threshold = np.percentile(scores, 100 * self.contamination)
        return np.where(scores < threshold, -1, 1)

    def _average_path_length(self, n):
        """计算n个样本的平均路径长度（基准值）"""
        if n <= 1:
            return 0
        h = np.log(n - 1) + 0.5772156649  # 调和数+欧拉常数
        return 2 * h - 2 * (n - 1) / n


# ===================== 2. 数据生成模块 =====================
def generate_abnormal_data(n_normal=1000, n_abnormal=50, random_state=42):
    np.random.seed(random_state)
    # 正常数据（正态分布）
    normal_features = np.random.normal(loc=0, scale=1, size=(n_normal, 3))
    normal_labels = np.zeros(n_normal, dtype=int)
    # 异常数据（偏离正态分布）
    abnormal_features = np.random.normal(loc=5, scale=2, size=(n_abnormal, 3))
    abnormal_labels = np.ones(n_abnormal, dtype=int)
    # 合并并打乱
    all_features = np.vstack([normal_features, abnormal_features])
    all_labels = np.hstack([normal_labels, abnormal_labels])
    shuffle_idx = np.random.permutation(len(all_features))
    all_features = all_features[shuffle_idx]
    all_labels = all_labels[shuffle_idx]
    # 生成时间戳
    start_time = datetime(2025, 1, 1)
    timestamps = [start_time + timedelta(minutes=i) for i in range(len(all_features))]
    return all_features, all_labels, timestamps


# ===================== 3. 数据预处理模块 =====================
def preprocess_data(features):
    """标准化（无sklearn的StandardScaler）"""
    # 均值填充缺失值
    mask = np.random.choice([0, 1], size=features.shape, p=[0.01, 0.99])
    features_with_nan = np.where(mask == 0, np.nan, features)
    mean_vals = np.nanmean(features_with_nan, axis=0)
    for i in range(features_with_nan.shape[1]):
        nan_idx = np.isnan(features_with_nan[:, i])
        features_with_nan[nan_idx, i] = mean_vals[i]
    # 手动标准化（均值0，方差1）
    mean = np.mean(features_with_nan, axis=0)
    std = np.std(features_with_nan, axis=0)
    std[std == 0] = 1  # 避免除以0
    scaled_features = (features_with_nan - mean) / std
    return scaled_features, (mean, std)


# ===================== 4. 模型封装 =====================
class AnomalyDetector:
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForestPure(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state
        )

    def train(self, features):
        self.model.fit(features)

    def detect(self, features):
        anomaly_labels = self.model.predict(features)
        anomaly_scores = self.model.decision_function(features)
        return anomaly_labels, anomaly_scores


# ===================== 5. 结果分析（无sklearn metrics） =====================
def analyze_results(true_labels, pred_labels, anomaly_scores):
    """手动计算评估指标"""
    pred_labels_converted = np.where(pred_labels == -1, 1, 0)

    # 手动计算TP/TN/FP/FN
    TP = np.sum((true_labels == 1) & (pred_labels_converted == 1))
    TN = np.sum((true_labels == 0) & (pred_labels_converted == 0))
    FP = np.sum((true_labels == 0) & (pred_labels_converted == 1))
    FN = np.sum((true_labels == 1) & (pred_labels_converted == 0))

    # 计算指标
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("===== 检测结果评估 =====")
    print(f"准确率（Accuracy）: {accuracy:.4f}")
    print(f"召回率（Recall）: {recall:.4f}")
    print(f"精确率（Precision）: {precision:.4f}")
    print(f"F1分数（F1-Score）: {f1:.4f}")

    n_true_anomalies = np.sum(true_labels == 1)
    n_pred_anomalies = np.sum(pred_labels_converted == 1)
    print(f"\n真实异常数: {n_true_anomalies} | 检测出的异常数: {n_pred_anomalies}")

    return pred_labels_converted, anomaly_scores


# ===================== 6. 可视化模块 =====================
def plot_results(timestamps, features, true_labels, pred_labels, anomaly_scores):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1：特征时序+异常标注
    ax1 = axes[0]
    feature1 = features[:, 0]
    timestamps_plt = [ts for ts in timestamps]
    ax1.plot(timestamps_plt, feature1, color='blue', alpha=0.6, label='Feature 1')

    # 真实异常
    true_anomaly_idx = np.where(true_labels == 1)[0]
    ax1.scatter(
        [timestamps_plt[i] for i in true_anomaly_idx],
        [feature1[i] for i in true_anomaly_idx],
        color='red', s=50, label='True Anomaly', zorder=5
    )

    # 检测异常
    pred_anomaly_idx = np.where(pred_labels == 1)[0]
    ax1.scatter(
        [timestamps_plt[i] for i in pred_anomaly_idx],
        [feature1[i] for i in pred_anomaly_idx],
        color='orange', s=30, marker='x', label='Detected Anomaly', zorder=4
    )

    ax1.set_title('Feature Time Series with Anomaly Labels', fontsize=12)
    ax1.set_ylabel('Feature 1 Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：异常评分分布
    ax2 = axes[1]
    normal_scores = anomaly_scores[true_labels == 0]
    anomaly_score_values = anomaly_scores[true_labels == 1]
    ax2.hist(normal_scores, bins=30, alpha=0.6, label='Normal Samples', color='blue')
    ax2.hist(anomaly_score_values, bins=30, alpha=0.6, label='Anomaly Samples', color='red')
    ax2.axvline(x=0, color='black', linestyle='--', label='Threshold (0)')
    ax2.set_title('Anomaly Score Distribution', fontsize=12)
    ax2.set_xlabel('Anomaly Score (Lower = More Anomalous)')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ===================== 7. 主函数 =====================
def main():
    # 1. 生成数据11
    raw_features, true_labels, timestamps = generate_abnormal_data(n_normal=1000, n_abnormal=50)
    # 2. 预处理
    scaled_features, scaler_params = preprocess_data(raw_features)
    # 3. 划分训练/测试集（手动实现，无sklearn）
    np.random.seed(42)
    test_size = 0.2
    n_samples = len(scaled_features)
    test_idx = np.random.choice(n_samples, int(n_samples * test_size), replace=False)
    train_idx = np.setdiff1d(np.arange(n_samples), test_idx)

    X_train = scaled_features[train_idx]
    X_test = scaled_features[test_idx]
    y_train = true_labels[train_idx]
    y_test = true_labels[test_idx]
    ts_train = [timestamps[i] for i in train_idx]
    ts_test = [timestamps[i] for i in test_idx]

    # 4. 训练模型
    detector = AnomalyDetector(contamination=0.05)
    detector.train(X_train)
    # 5. 检测
    pred_labels, anomaly_scores = detector.detect(X_test)
    # 6. 分析
    pred_labels_converted, anomaly_scores = analyze_results(y_test, pred_labels, anomaly_scores)
    # 7. 11可视化
    plot_results(ts_test, X_test, y_test, pred_labels_converted, anomaly_scores)


if __name__ == "__main__":
    main()