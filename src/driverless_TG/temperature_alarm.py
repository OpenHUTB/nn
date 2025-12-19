import numpy as np
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# 解决PyCharm中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ===================== 1. 数据生成模块 =====================
def generate_temperature_data(n_samples=10000):
    """生成模拟的无人车温度数据集"""
    np.random.seed(42)
    # 基础特征
    motor_temp = np.random.normal(60, 10, n_samples)
    battery_temp = np.random.normal(45, 8, n_samples)
    controller_temp = np.random.normal(50, 9, n_samples)
    runtime = np.random.uniform(0, 10, n_samples)
    speed = np.random.normal(40, 15, n_samples)

    # 构造异常数据
    anomaly_idx = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
    mild_idx = anomaly_idx[:int(len(anomaly_idx) * 0.5)]
    moderate_idx = anomaly_idx[int(len(anomaly_idx) * 0.5):int(len(anomaly_idx) * 0.8)]
    severe_idx = anomaly_idx[int(len(anomaly_idx) * 0.8):]

    motor_temp[mild_idx] += np.random.uniform(5, 10, len(mild_idx))
    battery_temp[mild_idx] += np.random.uniform(3, 7, len(mild_idx))

    motor_temp[moderate_idx] += np.random.uniform(10, 20, len(moderate_idx))
    battery_temp[moderate_idx] += np.random.uniform(7, 15, len(moderate_idx))
    controller_temp[moderate_idx] += np.random.uniform(8, 12, len(moderate_idx))

    motor_temp[severe_idx] += np.random.uniform(20, 35, len(severe_idx))
    battery_temp[severe_idx] += np.random.uniform(15, 25, len(severe_idx))
    controller_temp[severe_idx] += np.random.uniform(12, 20, len(severe_idx))
    runtime[severe_idx] = np.random.uniform(8, 12, len(severe_idx))

    # 生成标签
    labels = np.zeros(n_samples)
    labels[mild_idx] = 1
    labels[moderate_idx] = 2
    labels[severe_idx] = 3

    # 限制数值合理性
    motor_temp = np.clip(motor_temp, 0, 120)
    battery_temp = np.clip(battery_temp, 0, 80)
    controller_temp = np.clip(controller_temp, 0, 100)
    runtime = np.clip(runtime, 0, 12)
    speed = np.clip(speed, 0, 120)

    features = np.column_stack([motor_temp, battery_temp, controller_temp, runtime, speed])
    return features, labels


# ===================== 2. 数据预处理 =====================
def standard_scaler(X):
    """手动实现标准化"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1e-8
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def train_test_split(X, y, test_size=0.2, random_state=42):
    """手动实现数据集划分"""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    test_samples = int(n_samples * test_size)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


# ===================== 3. 决策树分类器 =====================
class DecisionTreeClassifier:
    """手动实现决策树分类器"""

    def __init__(self, max_depth=10, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = {}

    def _gini_impurity(self, y):
        """计算基尼不纯度"""
        classes, counts = np.unique(y, return_counts=True)
        impurity = 1.0
        for count in counts:
            p = count / len(y)
            impurity -= p * p
        return impurity

    def _best_split(self, X, y):
        """寻找最优划分特征和阈值"""
        n_features = X.shape[1]
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            values = X[:, feature]
            unique_values = np.unique(values)

            for threshold in unique_values:
                left_mask = values <= threshold
                right_mask = values > threshold

                if len(y[left_mask]) < 1 or len(y[right_mask]) < 1:
                    continue

                gini_left = self._gini_impurity(y[left_mask])
                gini_right = self._gini_impurity(y[right_mask])
                gini = (len(y[left_mask]) * gini_left + len(y[right_mask]) * gini_right) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                n_classes == 1):
            classes, counts = np.unique(y, return_counts=True)
            return classes[np.argmax(counts)]

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            classes, counts = np.unique(y, return_counts=True)
            return classes[np.argmax(counts)]

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def fit(self, X, y):
        """训练决策树"""
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, x, tree):
        """预测单个样本"""
        if not isinstance(tree, dict):
            return tree

        feature = tree['feature']
        threshold = tree['threshold']

        if x[feature] <= threshold:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])

    def predict(self, X):
        """预测多个样本"""
        predictions = [self._predict_sample(x, self.tree) for x in X]
        return np.array(predictions)


# ===================== 4. 模型评估与可视化 =====================
def evaluate_model(y_true, y_pred):
    """评估模型并生成可视化图表"""
    classes = np.unique(y_true)
    n_classes = len(classes)

    # 混淆矩阵
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[int(t), int(p)] += 1

    # 计算评估指标
    overall_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    precision = []
    recall = []
    f1 = []

    for i in range(n_classes):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision.append(p)
        recall.append(r)
        f1.append(f)

    # 1. 绘制混淆矩阵热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.title('模型混淆矩阵', fontsize=14)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(range(n_classes), ['正常', '轻度异常', '中度异常', '重度异常'])
    plt.yticks(range(n_classes), ['正常', '轻度异常', '中度异常', '重度异常'])

    # 添加数值标注
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black', fontsize=10)

    plt.colorbar(label='样本数量')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)  # 保存图片
    plt.show()  # PyCharm中显示图片

    # 2. 绘制精确率/召回率/F1对比图
    plt.figure(figsize=(10, 6))
    x = np.arange(n_classes)
    width = 0.25

    plt.bar(x - width, precision, width, label='精确率', color='#1f77b4')
    plt.bar(x, recall, width, label='召回率', color='#ff7f0e')
    plt.bar(x + width, f1, width, label='F1分数', color='#2ca02c')

    plt.title('模型分类性能指标', fontsize=14)
    plt.xlabel('异常等级', fontsize=12)
    plt.ylabel('指标值', fontsize=12)
    plt.xticks(x, ['正常', '轻度异常', '中度异常', '重度异常'])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('classification_metrics.png', dpi=150)
    plt.show()

    # 打印文本报告
    print("=== 模型性能报告 ===")
    print(f"{'类别':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8}")
    print("-" * 32)
    for i, cls in enumerate(classes):
        print(f"{int(cls):<8} {precision[i]:.4f}    {recall[i]:.4f}    {f1[i]:.4f}")
    print(f"\n整体准确率：{overall_accuracy:.4f}")

    return conf_matrix


def plot_temperature_distribution(X, y):
    """绘制温度特征分布直方图"""
    feature_names = ['电机温度(℃)', '电池温度(℃)', '控制器温度(℃)']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, ax in enumerate(axes):
        # 正常数据
        normal_data = X[y == 0, i]
        # 异常数据
        anomaly_data = X[y != 0, i]

        ax.hist(normal_data, bins=30, alpha=0.7, label='正常', color='green', density=True)
        ax.hist(anomaly_data, bins=30, alpha=0.7, label='异常', color='red', density=True)

        ax.set_title(feature_names[i], fontsize=12)
        ax.set_xlabel('温度值')
        ax.set_ylabel('概率密度')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('关键部件温度分布对比', fontsize=14)
    plt.tight_layout()
    plt.savefig('temperature_distribution.png', dpi=150)
    plt.show()


# ===================== 5. 实时监测与报警系统 =====================
class TemperatureAlarmSystem:
    """无人车温度报警系统（带可视化）"""

    def __init__(self, model, mean, std):
        self.model = model
        self.mean = mean
        self.std = std
        self.alarm_config = {
            0: {"level": "正常", "msg": "各部件温度正常", "action": "持续监测", "color": 'green'},
            1: {"level": "轻度异常", "msg": "温度略高于正常阈值", "action": "提醒驾驶员关注", "color": 'yellow'},
            2: {"level": "中度异常", "msg": "温度明显升高", "action": "降低车速", "color": 'orange'},
            3: {"level": "重度异常", "msg": "温度严重超标", "action": "紧急停车", "color": 'red'}
        }
        self.history = []
        self.hard_thresholds = {"motor": 100, "battery": 70, "controller": 90}

    def _scale_data(self, data):
        """标准化数据"""
        scaled = (data - self.mean) / self.std
        return scaled

    def _check_hard_threshold(self, sensor_data):
        """检查硬件阈值"""
        motor_temp = sensor_data[0][0]
        battery_temp = sensor_data[0][1]
        controller_temp = sensor_data[0][2]

        if motor_temp >= self.hard_thresholds["motor"] or \
                battery_temp >= self.hard_thresholds["battery"] or \
                controller_temp >= self.hard_thresholds["controller"]:
            return 3
        return None

    def simulate_sensor(self):
        """模拟传感器数据"""
        motor = random.uniform(50, 70)
        battery = random.uniform(40, 50)
        controller = random.uniform(45, 55)
        runtime = random.uniform(0, 10)
        speed = random.uniform(30, 50)

        anomaly_prob = random.random()
        if anomaly_prob < 0.05:
            motor += 30
            battery += 20
            controller += 18
        elif anomaly_prob < 0.15:
            motor += 15
            battery += 10
            controller += 10
        elif anomaly_prob < 0.3:
            motor += 8
            battery += 5
            controller += 5

        motor = max(0, min(motor, 120))
        battery = max(0, min(battery, 80))
        controller = max(0, min(controller, 100))
        runtime = max(0, min(runtime, 12))
        speed = max(0, min(speed, 120))

        return np.array([[motor, battery, controller, runtime, speed]])

    def predict_risk(self, sensor_data):
        """预测风险等级"""
        hard_risk = self._check_hard_threshold(sensor_data)
        if hard_risk is not None:
            return hard_risk

        scaled_data = self._scale_data(sensor_data)
        risk_level = self.model.predict(scaled_data)[0]
        return int(risk_level)

    def trigger_alarm(self, risk_level, sensor_data):
        """触发报警"""
        alarm_info = self.alarm_config[risk_level]
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        log = {
            "时间": timestamp,
            "时间戳": time.time(),
            "电机温度": round(sensor_data[0][0], 2),
            "电池温度": round(sensor_data[0][1], 2),
            "控制器温度": round(sensor_data[0][2], 2),
            "运行时长": round(sensor_data[0][3], 2),
            "车速": round(sensor_data[0][4], 2),
            "风险等级": risk_level,
            "报警级别": alarm_info["level"],
            "颜色": alarm_info["color"]
        }

        self.history.append(log)

        print(f"\n【{timestamp}】【{alarm_info['level']}】")
        print(f"传感器数据：电机{log['电机温度']}℃ | 电池{log['电池温度']}℃ | 控制器{log['控制器温度']}℃")
        print(f"报警信息：{alarm_info['msg']} | 建议操作：{alarm_info['action']}")

        return log

    def plot_monitoring_results(self):
        """绘制实时监测结果可视化图表"""
        if not self.history:
            print("无监测数据可绘制")
            return

        # 提取数据
        timestamps = [log["时间戳"] for log in self.history]
        motor_temps = [log["电机温度"] for log in self.history]
        battery_temps = [log["电池温度"] for log in self.history]
        controller_temps = [log["控制器温度"] for log in self.history]
        risk_levels = [log["风险等级"] for log in self.history]
        colors = [log["颜色"] for log in self.history]

        # 归一化时间戳（便于显示）
        start_ts = timestamps[0]
        timestamps = [ts - start_ts for ts in timestamps]

        # 1. 温度趋势图
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, motor_temps, 'o-', label='电机温度', linewidth=2, markersize=6)
        plt.plot(timestamps, battery_temps, 's-', label='电池温度', linewidth=2, markersize=6)
        plt.plot(timestamps, controller_temps, '^-', label='控制器温度', linewidth=2, markersize=6)

        # 标注异常点
        for i, (t, r) in enumerate(zip(timestamps, risk_levels)):
            if r > 0:
                plt.scatter(t, motor_temps[i], color=colors[i], s=100, edgecolor='black', zorder=5)
                plt.annotate(f'等级{r}', (t, motor_temps[i]), xytext=(5, 5), textcoords='offset points')

        plt.title('无人车温度实时监测趋势', fontsize=14)
        plt.xlabel('监测时长(秒)', fontsize=12)
        plt.ylabel('温度(℃)', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('temperature_trend.png', dpi=150)
        plt.show()

        # 2. 报警等级统计饼图
        alarm_counts = defaultdict(int)
        for log in self.history:
            alarm_counts[log["报警级别"]] += 1

        labels = list(alarm_counts.keys())
        sizes = list(alarm_counts.values())
        colors = [self.alarm_config[0]["color"] if l == "正常" else
                  self.alarm_config[1]["color"] if l == "轻度异常" else
                  self.alarm_config[2]["color"] if l == "中度异常" else
                  self.alarm_config[3]["color"] for l in labels]

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
        plt.title('报警等级分布统计', fontsize=14)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('alarm_distribution.png', dpi=150)
        plt.show()

    def run_monitor(self, duration=10, interval=2):
        """运行实时监测系统"""
        print("\n========== 无人车温度报警系统启动 ==========")
        print(f"监测时长：{duration}秒 | 采样间隔：{interval}秒")
        print("===========================================\n")

        start_time = time.time()
        while time.time() - start_time < duration:
            sensor_data = self.simulate_sensor()
            risk_level = self.predict_risk(sensor_data)
            self.trigger_alarm(risk_level, sensor_data)
            time.sleep(interval)

        print("\n========== 监测结束 ==========")
        # 生成监测结果可视化
        self.plot_monitoring_results()

        # 输出统计信息
        anomaly_stats = defaultdict(int)
        for log in self.history:
            anomaly_stats[log["报警级别"]] += 1
        print("异常统计：")
        for level, count in anomaly_stats.items():
            print(f"{level}：{count}次")


# ===================== 6. 主程序入口 =====================
if __name__ == "__main__":
    # 步骤1：生成数据并可视化分布
    print("=== 1. 生成温度数据集 ===")
    X, y = generate_temperature_data(n_samples=10000)
    print(f"数据集规模：特征{X.shape} | 标签{y.shape}")
    print(f"标签分布：{np.bincount(y.astype(int))}")

    # 绘制温度分布
    plot_temperature_distribution(X, y)

    # 步骤2：数据预处理
    print("\n=== 2. 数据预处理 ===")
    X_scaled, mean, std = standard_scaler(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    print(f"训练集：{X_train.shape} | 测试集：{X_test.shape}")

    # 步骤3：训练模型
    print("\n=== 3. 训练决策树模型 ===")
    model = DecisionTreeClassifier(max_depth=8, min_samples_split=5)
    model.fit(X_train, y_train)
    print("模型训练完成！")

    # 步骤4：评估模型并可视化
    print("\n=== 4. 模型评估 ===")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)

    # 步骤5：启动报警系统
    print("\n=== 5. 启动温度报警系统 ===")
    alarm_system = TemperatureAlarmSystem(model, mean, std)
    alarm_system.run_monitor(duration=10, interval=2)

    print("\n=== 所有可视化图片已保存到当前目录 ===")
    print("生成的图片：")
    print("1. temperature_distribution.png - 温度分布对比图")
    print("2. confusion_matrix.png - 混淆矩阵热力图")
    print("3. classification_metrics.png - 分类性能指标图")
    print("4. temperature_trend.png - 实时温度趋势图")
    print("5. alarm_distribution.png - 报警等级统计饼图")