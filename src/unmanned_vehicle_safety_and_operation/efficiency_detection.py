import random
import math
import sys
import time
import os
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, Canvas
import threading

# ===================== 全局配置（PyCharm适配）=====================
# 创建图片保存目录（模拟保存，实际为tkinter绘图窗口）
if not os.path.exists('efficiency_plots'):
    os.makedirs('efficiency_plots')
random.seed(42)


# ===================== 1. 数据生成与预处理（纯原生）=====================
class DataPreprocessor:
    def __init__(self):
        self.feature_means = {}  # 特征均值（填充缺失值）
        self.feature_stds = {}  # 特征标准差（标准化）

    def generate_simulation_data(self, n_samples=1000):
        """生成无人车模拟数据（纯原生）"""
        data = []
        self.feature_names = [
            '规划路径长度(km)', '实际路径长度(km)', '路径偏差(km)',
            '航点完成率', '能耗(kWh/100km)', '平均速度(km/h)',
            '加减速频率(次/分)', '安全距离违规次数',
            '紧急制动次数', '车道保持准确率',
            '红绿灯等待时间(秒)', '平均车流量(辆/公里)', '行程耗时(分钟)'
        ]

        for _ in range(n_samples):
            # 单条样本数据
            sample = {
                '规划路径长度(km)': random.uniform(5, 50),
                '实际路径长度(km)': random.uniform(5, 55),
                '路径偏差(km)': abs(random.uniform(0, 5)),
                '航点完成率': random.uniform(0.8, 1.0),
                '能耗(kWh/100km)': random.uniform(15, 35),
                '平均速度(km/h)': random.uniform(20, 80),
                '加减速频率(次/分)': random.uniform(0.1, 1.0),
                '安全距离违规次数': random.randint(0, 5),
                '紧急制动次数': random.randint(0, 3),
                '车道保持准确率': random.uniform(0.85, 1.0),
                '红绿灯等待时间(秒)': random.uniform(0, 120),
                '平均车流量(辆/公里)': random.uniform(20, 100),
                '行程耗时(分钟)': random.uniform(10, 120)
            }

            # 计算效率得分（0-100）
            路径效率 = (1 - sample['路径偏差(km)'] / sample['规划路径长度(km)']) * 100
            能耗效率 = (1 - (sample['能耗(kWh/100km)'] - 15) / 20) * 100
            安全效率 = (1 - (sample['安全距离违规次数'] + sample['紧急制动次数']) / 8) * 100
            通行效率 = (1 - sample['行程耗时(分钟)'] / 120) * 100
            效率得分 = (路径效率 + 能耗效率 + 安全效率 + 通行效率) / 4
            sample['效率得分'] = max(0, min(100, 效率得分))
            data.append(sample)

        return data

    def preprocess_data(self, data):
        """数据预处理：缺失值填充+标准化（纯原生）"""
        # 1. 统计特征均值和标准差
        feature_sums = defaultdict(float)
        feature_sq_sums = defaultdict(float)  # 平方和（计算标准差）
        feature_counts = defaultdict(int)

        # 模拟5%缺失值 + 统计基础值
        for sample in data:
            for feat in self.feature_names:
                # 模拟缺失值
                if random.random() < 0.05:
                    sample[feat] = None
                # 统计非缺失值
                if sample[feat] is not None:
                    feature_sums[feat] += sample[feat]
                    feature_sq_sums[feat] += sample[feat] ** 2
                    feature_counts[feat] += 1

        # 计算均值和标准差
        for feat in self.feature_names:
            self.feature_means[feat] = feature_sums[feat] / feature_counts[feat] if feature_counts[feat] > 0 else 0
            # 标准差 = sqrt( (Σx²/N) - (均值)² )
            variance = (feature_sq_sums[feat] / feature_counts[feat]) - (self.feature_means[feat] ** 2) if \
            feature_counts[feat] > 0 else 0
            self.feature_stds[feat] = math.sqrt(variance) if variance > 0 else 1

        # 2. 填充缺失值+标准化
        X = []
        y = []
        for sample in data:
            features = []
            for feat in self.feature_names:
                # 填充缺失值
                val = sample[feat] if sample[feat] is not None else self.feature_means[feat]
                # 标准化 (x-mean)/std
                scaled_val = (val - self.feature_means[feat]) / self.feature_stds[feat]
                features.append(scaled_val)
            X.append(features)
            y.append(sample['效率得分'])

        return X, y


# ===================== 2. 简化版随机森林（纯原生）=====================
class SimpleRandomForest:
    def __init__(self, n_estimators=10, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.feature_importance = defaultdict(int)

    def _bootstrap_sample(self, X, y):
        """自助采样"""
        n_samples = len(X)
        indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        return [X[i] for i in indices], [y[i] for i in indices]

    def _split_data(self, X, y, feature_idx, threshold):
        """划分数据集"""
        left_X, left_y = [], []
        right_X, right_y = [], []
        for i in range(len(X)):
            if X[i][feature_idx] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        return left_X, left_y, right_X, right_y

    def _mse(self, y):
        """计算均方误差"""
        if not y:
            return 0
        mean = sum(y) / len(y)
        return sum([(val - mean) ** 2 for val in y]) / len(y)

    def _find_best_split(self, X, y):
        """寻找最优划分"""
        best_mse = float('inf')
        best_feature = -1
        best_threshold = None
        n_features = len(X[0]) if X else 0

        for feature_idx in range(n_features):
            # 提取特征值并去重
            feature_vals = list(set([X[i][feature_idx] for i in range(len(X))]))
            feature_vals.sort()

            for threshold in feature_vals:
                left_X, left_y, right_X, right_y = self._split_data(X, y, feature_idx, threshold)
                if not left_y or not right_y:
                    continue
                # 计算划分后MSE
                mse = (len(left_y) * self._mse(left_y) + len(right_y) * self._mse(right_y)) / len(y)
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        if depth >= self.max_depth or len(y) < 5:
            return {'value': sum(y) / len(y) if y else 0}

        feature_idx, threshold = self._find_best_split(X, y)
        if feature_idx == -1:
            return {'value': sum(y) / len(y) if y else 0}

        # 统计特征重要性
        self.feature_importance[feature_idx] += 1

        left_X, left_y, right_X, right_y = self._split_data(X, y, feature_idx, threshold)
        left_tree = self._build_tree(left_X, left_y, depth + 1)
        right_tree = self._build_tree(right_X, right_y, depth + 1)

        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }

    def train(self, X, y):
        """训练随机森林"""
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = self._build_tree(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """预测效率得分"""
        predictions = []
        for sample in X:
            tree_preds = [self._predict_tree(sample, tree) for tree in self.trees]
            predictions.append(sum(tree_preds) / len(tree_preds))
        return predictions

    def _predict_tree(self, sample, tree):
        """单棵树预测"""
        if 'value' in tree:
            return tree['value']
        if sample[tree['feature_idx']] <= tree['threshold']:
            return self._predict_tree(sample, tree['left'])
        else:
            return self._predict_tree(sample, tree['right'])


# ===================== 3. 异常检测（纯原生）=====================
class SimpleIsolationForest:
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.trees = []

    def _build_tree(self, X, depth=0, max_depth=10):
        """构建孤立树"""
        if depth >= max_depth or len(X) <= 1:
            return {'depth': depth}

        # 随机选特征和阈值
        n_features = len(X[0]) if X else 0
        feature_idx = random.randint(0, n_features - 1)
        feature_vals = [x[feature_idx] for x in X]
        min_val, max_val = min(feature_vals), max(feature_vals)
        threshold = random.uniform(min_val, max_val)

        # 划分数据
        left_X = [x for x in X if x[feature_idx] <= threshold]
        right_X = [x for x in X if x[feature_idx] > threshold]

        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': self._build_tree(left_X, depth + 1, max_depth),
            'right': self._build_tree(right_X, depth + 1, max_depth)
        }

    def _path_length(self, sample, tree, depth=0):
        """计算路径长度"""
        if 'depth' in tree:
            return depth + self._avg_path(len(self.X_train))
        if sample[tree['feature_idx']] <= tree['threshold']:
            return self._path_length(sample, tree['left'], depth + 1)
        else:
            return self._path_length(sample, tree['right'], depth + 1)

    def _avg_path(self, n):
        """平均路径长度"""
        if n <= 1:
            return 0
        return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def train(self, X):
        """训练孤立森林"""
        self.X_train = X
        for _ in range(10):  # 10棵树
            sample_size = min(256, len(X))
            sample_indices = [random.randint(0, len(X) - 1) for _ in range(sample_size)]
            X_sample = [X[i] for i in sample_indices]
            self.trees.append(self._build_tree(X_sample))

    def detect_anomalies(self, X):
        """检测异常（-1=异常，1=正常）"""
        avg_lengths = []
        for sample in X:
            lengths = [self._path_length(sample, tree) for tree in self.trees]
            avg_lengths.append(sum(lengths) / len(lengths))

        # 按污染系数划分异常
        avg_lengths_sorted = sorted(avg_lengths)
        threshold = avg_lengths_sorted[int(len(avg_lengths) * self.contamination)]
        return [-1 if l < threshold else 1 for l in avg_lengths]


# ===================== 4. PyCharm绘图（基于tkinter纯原生）=====================
class NativePlotter:
    def __init__(self, title):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("800x600")
        self.canvas = Canvas(self.root, width=800, height=600, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def plot_efficiency_dist(self, scores):
        """绘制效率得分分布直方图"""
        # 分桶统计
        bins = [0, 20, 40, 60, 80, 100]
        counts = [0] * 4  # 0-20,20-40,40-60,60-80,80-100
        for score in scores:
            if 0 <= score < 20:
                counts[0] += 1
            elif 20 <= score < 40:
                counts[1] += 1
            elif 40 <= score < 60:
                counts[2] += 1
            elif 60 <= score < 80:
                counts[3] += 1
            else:
                counts[4] += 1 if 80 <= score <= 100 else 0

        # 归一化高度
        max_count = max(counts) if counts else 1
        bar_width = 100
        x_start = 100

        # 绘制坐标轴
        self.canvas.create_line(80, 500, 720, 500, width=2)  # X轴
        self.canvas.create_line(80, 100, 80, 500, width=2)  # Y轴

        # 绘制柱子
        colors = ["lightblue", "skyblue", "dodgerblue", "royalblue", "navy"]
        for i in range(5):
            height = (counts[i] / max_count) * 350
            x1 = x_start + i * (bar_width + 20)
            y1 = 500 - height
            x2 = x1 + bar_width
            y2 = 500
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=colors[i], outline="black")
            # 标注
            self.canvas.create_text(x1 + bar_width / 2, 520, text=f"{bins[i]}-{bins[i + 1]}")
            self.canvas.create_text(x1 + bar_width / 2, y1 - 10, text=str(counts[i]))

        # 标题
        self.canvas.create_text(400, 50, text="无人车效率得分分布", font=("Arial", 16, "bold"))
        self.root.mainloop()

    def plot_feature_importance(self, importance, feature_names):
        """绘制特征重要性条形图"""
        # 排序特征
        imp_list = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_features = imp_list[:8]  # 取前8个特征

        # 绘制坐标轴
        self.canvas.create_line(150, 500, 150, 100, width=2)  # Y轴
        self.canvas.create_line(150, 500, 750, 500, width=2)  # X轴

        # 绘制条形
        max_imp = max([x[1] for x in top_features]) if top_features else 1
        y_start = 120
        bar_height = 40
        for i, (feat_idx, imp) in enumerate(top_features):
            width = (imp / max_imp) * 550
            x1 = 150
            y1 = y_start + i * (bar_height + 10)
            x2 = 150 + width
            y2 = y1 + bar_height
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="orange", outline="black")
            # 标注特征名和重要性
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"特征{feat_idx}"
            self.canvas.create_text(120, y1 + 20, text=feat_name, anchor="e", font=("Arial", 10))
            self.canvas.create_text(x2 + 10, y1 + 20, text=f"{imp}", font=("Arial", 10))

        self.canvas.create_text(400, 50, text="特征重要性排名", font=("Arial", 16, "bold"))
        self.root.mainloop()

    def plot_real_time(self, time_series, scores, anomalies):
        """绘制实时效率监测曲线"""
        # 绘制坐标轴
        self.canvas.create_line(80, 500, 720, 500, width=2)  # X轴
        self.canvas.create_line(80, 500, 80, 100, width=2)  # Y轴

        # 绘制网格
        for x in range(180, 720, 100):
            self.canvas.create_line(x, 100, x, 500, dash=(2, 2), fill="lightgray")
        for y in range(200, 500, 100):
            self.canvas.create_line(80, y, 720, y, dash=(2, 2), fill="lightgray")

        # 归一化坐标
        max_score = 100
        min_score = 0
        x_step = (720 - 80) / len(time_series) if time_series else 1

        # 绘制曲线
        points = []
        for i, (t, s) in enumerate(zip(time_series, scores)):
            x = 80 + i * x_step
            y = 500 - ((s - min_score) / (max_score - min_score)) * 400
            points.append((x, y))
            # 标记异常点
            if anomalies[i] == -1:
                self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red")

        # 连接曲线
        for i in range(len(points) - 1):
            self.canvas.create_line(points[i][0], points[i][1],
                                    points[i + 1][0], points[i + 1][1],
                                    fill="blue", width=2)

        self.canvas.create_text(400, 50, text="实时效率得分监测", font=("Arial", 16, "bold"))
        self.root.mainloop()


# ===================== 5. 主程序 =====================
def main():
    # 初始化模块
    preprocessor = DataPreprocessor()
    rf_model = SimpleRandomForest(n_estimators=10, max_depth=5)
    anomaly_detector = SimpleIsolationForest(contamination=0.05)

    # 1. 生成数据
    print("=" * 50)
    print("生成无人车模拟数据...")
    data = preprocessor.generate_simulation_data(n_samples=1000)
    X, y = preprocessor.preprocess_data(data)

    # 2. 划分训练/测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 3. 训练模型
    print("训练效率检测模型...")
    start_time = time.time()
    rf_model.train(X_train, y_train)
    anomaly_detector.train(X_train)
    print(f"训练耗时: {time.time() - start_time:.2f}秒")

    # 4. 模型预测与评估
    y_pred = rf_model.predict(X_test)
    # 计算MSE和R²（纯原生）
    mse = sum([(y_test[i] - y_pred[i]) ** 2 for i in range(len(y_test))]) / len(y_test)
    y_mean = sum(y_test) / len(y_test)
    ss_total = sum([(y - y_mean) ** 2 for y in y_test])
    ss_res = sum([(y_test[i] - y_pred[i]) ** 2 for i in range(len(y_test))])
    r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0

    print("\n=== 模型性能 ===")
    print(f"均方误差(MSE): {mse:.4f}")
    print(f"决定系数(R²): {r2:.4f}")
    print(f"平均效率得分: {sum(y_pred) / len(y_pred):.2f}")

    # 5. 异常检测
    anomalies = anomaly_detector.detect_anomalies(X_test)
    anomaly_rate = sum([1 for a in anomalies if a == -1]) / len(anomalies)
    print(f"\n异常样本比例: {anomaly_rate:.2%}")

    # 6. 特征重要性
    print("\n=== 特征重要性 ===")
    for feat_idx, imp in sorted(rf_model.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        if feat_idx < len(preprocessor.feature_names):
            print(f"{preprocessor.feature_names[feat_idx]}: {imp}")

    # 7. 生成可视化图片（PyCharm中弹出tkinter窗口）
    print("\n生成可视化图片（请查看弹出窗口）...")

    # 效率分布图表（子线程运行，避免阻塞）
    def plot_dist():
        plotter = NativePlotter("效率得分分布")
        plotter.plot_efficiency_dist(y_pred)

    threading.Thread(target=plot_dist).start()

    # 特征重要性图表
    def plot_feat():
        plotter = NativePlotter("特征重要性")
        plotter.plot_feature_importance(rf_model.feature_importance, preprocessor.feature_names)

    threading.Thread(target=plot_feat).start()

    # 实时监测图表11
    def plot_realtime():
        time_series = list(range(50))
        plotter = NativePlotter("实时效率监测")
        plotter.plot_real_time(time_series, y_pred[:50], anomalies[:50])

    threading.Thread(target=plot_realtime).start()

    # 8. 实时检测示例11111
    print("\n=== 实时检测示例 ===")
    test_data = preprocessor.generate_simulation_data(n_samples=10)
    rt_X, _ = preprocessor.preprocess_data(test_data)
    rt_scores = rf_model.predict(rt_X)
    rt_anomalies = anomaly_detector.detect_anomalies(rt_X)

    print("样本ID | 效率得分 | 状态")
    print("-" * 30)
    for i in range(10):
        status = "异常" if rt_anomalies[i] == -1 else "正常"
        print(f"{i + 1:<6} | {rt_scores[i]:<8.2f} | {status}")

    print("=" * 50)
    print("运行完成！可视化窗口已弹出，图片已模拟保存到 efficiency_plots 目录")


if __name__ == "__main__":
    main()
    # 保持tkinter窗口运行
    tk.mainloop()