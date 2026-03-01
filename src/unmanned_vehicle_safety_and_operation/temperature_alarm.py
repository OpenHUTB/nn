import numpy as np
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# 解决PyCharm中文显示问题（全局字体配置）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


# ===================== 1. 数据生成模块 =====================
def generate_temperature_data(n_samples=10000):
    """
    生成模拟的无人车关键部件温度数据集（含正常/异常样本）
    模拟场景：无人车电机、电池、控制器温度随运行时长/车速变化，同时注入不同等级的异常

    参数说明：
    --------
    n_samples : int
        生成样本总数，默认10000条

    返回值：
    --------
    Tuple[np.ndarray, np.ndarray]
        features: 特征数组 (n_samples, 5)，列顺序：电机温度、电池温度、控制器温度、运行时长、车速
        labels: 标签数组 (n_samples,)，取值：0=正常，1=轻度异常，2=中度异常，3=重度异常
    """
    # 设置随机种子（保证结果可复现）
    np.random.seed(42)

    # ========== 生成基础正常数据（符合正态分布） ==========
    motor_temp = np.random.normal(60, 10, n_samples)  # 电机温度：均值60℃，标准差10℃
    battery_temp = np.random.normal(45, 8, n_samples)  # 电池温度：均值45℃，标准差8℃
    controller_temp = np.random.normal(50, 9, n_samples)  # 控制器温度：均值50℃，标准差9℃
    runtime = np.random.uniform(0, 10, n_samples)  # 运行时长：0-10小时均匀分布
    speed = np.random.normal(40, 15, n_samples)  # 车速：均值40km/h，标准差15km/h

    # ========== 构造不同等级的异常数据（模拟故障） ==========
    # 随机选择30%的样本作为异常样本
    anomaly_idx = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
    # 异常样本细分：50%轻度、30%中度、20%重度
    mild_idx = anomaly_idx[:int(len(anomaly_idx) * 0.5)]  # 轻度异常索引
    moderate_idx = anomaly_idx[int(len(anomaly_idx) * 0.5):int(len(anomaly_idx) * 0.8)]  # 中度异常索引
    severe_idx = anomaly_idx[int(len(anomaly_idx) * 0.8):]  # 重度异常索引

    # 轻度异常：温度小幅升高
    motor_temp[mild_idx] += np.random.uniform(5, 10, len(mild_idx))
    battery_temp[mild_idx] += np.random.uniform(3, 7, len(mild_idx))

    # 中度异常：温度明显升高（扩展到控制器）
    motor_temp[moderate_idx] += np.random.uniform(10, 20, len(moderate_idx))
    battery_temp[moderate_idx] += np.random.uniform(7, 15, len(moderate_idx))
    controller_temp[moderate_idx] += np.random.uniform(8, 12, len(moderate_idx))

    # 重度异常：温度大幅升高 + 运行时长异常
    motor_temp[severe_idx] += np.random.uniform(20, 35, len(severe_idx))
    battery_temp[severe_idx] += np.random.uniform(15, 25, len(severe_idx))
    controller_temp[severe_idx] += np.random.uniform(12, 20, len(severe_idx))
    runtime[severe_idx] = np.random.uniform(8, 12, len(severe_idx))  # 运行时长接近上限

    # ========== 生成标签并限制数值合理性（物理约束） ==========
    labels = np.zeros(n_samples)  # 初始化标签：0=正常
    labels[mild_idx] = 1  # 轻度异常
    labels[moderate_idx] = 2  # 中度异常
    labels[severe_idx] = 3  # 重度异常

    # 数值裁剪：确保温度/时长/车速在合理物理范围内
    motor_temp = np.clip(motor_temp, 0, 120)  # 电机温度：0-120℃
    battery_temp = np.clip(battery_temp, 0, 80)  # 电池温度：0-80℃
    controller_temp = np.clip(controller_temp, 0, 100)  # 控制器温度：0-100℃
    runtime = np.clip(runtime, 0, 12)  # 运行时长：0-12小时
    speed = np.clip(speed, 0, 120)  # 车速：0-120km/h

    # 组合特征矩阵
    features = np.column_stack([motor_temp, battery_temp, controller_temp, runtime, speed])
    return features, labels


# ===================== 2. 数据预处理模块 =====================
def standard_scaler(X):
    """
    手动实现标准化（Z-Score）：将特征缩放到均值为0，标准差为1
    公式：X_scaled = (X - mean) / std

    参数说明：
    --------
    X : np.ndarray
        原始特征矩阵 (n_samples, n_features)

    返回值：
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        X_scaled: 标准化后的特征矩阵
        mean: 各特征的均值
        std: 各特征的标准差（避免除0，替换0为1e-8）
    """
    mean = np.mean(X, axis=0)  # 按列计算均值
    std = np.std(X, axis=0)  # 按列计算标准差
    std[std == 0] = 1e-8  # 防止标准差为0导致除0错误
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    手动实现数据集划分：随机将数据分为训练集和测试集
    替代sklearn的train_test_split，保证无外部依赖

    参数说明：
    --------
    X : np.ndarray
        特征矩阵
    y : np.ndarray
        标签数组
    test_size : float
        测试集比例（0-1），默认0.2
    random_state : int
        随机种子（保证划分结果可复现）

    返回值：
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train: 训练集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_test: 测试集标签
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    test_samples = int(n_samples * test_size)  # 测试集样本数

    # 随机打乱索引
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # 划分训练/测试集索引
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    # 按索引划分数据
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


# ===================== 3. 决策树分类器（手动实现） =====================
class DecisionTreeClassifier:
    """
    手动实现决策树分类器（基于基尼不纯度）
    核心原理：递归划分特征空间，使每个子节点的样本尽可能纯（基尼不纯度最小）

    参数说明：
    --------
    max_depth : int
        决策树最大深度（防止过拟合），默认10
    min_samples_split : int
        节点分裂的最小样本数（防止过拟合），默认5
    """

    def __init__(self, max_depth=10, min_samples_split=5):
        self.max_depth = max_depth  # 树的最大深度
        self.min_samples_split = min_samples_split  # 节点分裂最小样本数
        self.tree = {}  # 存储决策树结构（字典形式）

    def _gini_impurity(self, y):
        """
        计算基尼不纯度（衡量样本纯度）
        公式：Gini = 1 - Σ(p_i²)，p_i为类别i在样本中的占比
        基尼不纯度越小，样本越纯

        参数说明：
        --------
        y : np.ndarray
            节点的标签数组

        返回值：
        --------
        float
            基尼不纯度值（0-1）
        """
        classes, counts = np.unique(y, return_counts=True)
        impurity = 1.0
        for count in counts:
            p = count / len(y)  # 类别占比
            impurity -= p * p
        return impurity

    def _best_split(self, X, y):
        """
        寻找最优划分特征和阈值（遍历所有特征和阈值，选择基尼不纯度最小的划分）

        参数说明：
        --------
        X : np.ndarray
            节点的特征矩阵
        y : np.ndarray
            节点的标签数组

        返回值：
        --------
        Tuple[int, float]
            best_feature: 最优划分特征索引
            best_threshold: 最优划分阈值
        """
        n_features = X.shape[1]
        best_gini = float('inf')  # 初始化最优基尼不纯度为无穷大
        best_feature = None  # 最优特征
        best_threshold = None  # 最优阈值

        # 遍历所有特征
        for feature in range(n_features):
            values = X[:, feature]
            unique_values = np.unique(values)  # 去重，减少阈值遍历次数

            # 遍历该特征的所有唯一值作为候选阈值
            for threshold in unique_values:
                # 按阈值划分左右子节点
                left_mask = values <= threshold
                right_mask = values > threshold

                # 跳过样本数不足的划分
                if len(y[left_mask]) < 1 or len(y[right_mask]) < 1:
                    continue

                # 计算划分后的基尼不纯度（加权平均）
                gini_left = self._gini_impurity(y[left_mask])
                gini_right = self._gini_impurity(y[right_mask])
                gini = (len(y[left_mask]) * gini_left + len(y[right_mask]) * gini_right) / len(y)

                # 更新最优划分
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """
        递归构建决策树

        参数说明：
        --------
        X : np.ndarray
            当前节点的特征矩阵
        y : np.ndarray
            当前节点的标签数组
        depth : int
            当前树的深度

        返回值：
        --------
        dict/int
            叶子节点：返回类别标签；内部节点：返回划分字典（feature/threshold/left/right）
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # ========== 递归终止条件（叶子节点） ==========
        # 1. 树深度达到上限 2. 节点样本数不足 3. 节点样本已纯（单类别）
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                n_classes == 1):
            # 返回节点中数量最多的类别
            classes, counts = np.unique(y, return_counts=True)
            return classes[np.argmax(counts)]

        # ========== 寻找最优划分并递归构建子树 ==========
        best_feature, best_threshold = self._best_split(X, y)
        # 无有效划分时，返回数量最多的类别
        if best_feature is None:
            classes, counts = np.unique(y, return_counts=True)
            return classes[np.argmax(counts)]

        # 按最优划分拆分数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        # 递归构建左右子树
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # 返回当前节点的划分信息
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def fit(self, X, y):
        """
        训练决策树（对外接口）

        参数说明：
        --------
        X : np.ndarray
            训练集特征矩阵
        y : np.ndarray
            训练集标签数组
        """
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, x, tree):
        """
        递归预测单个样本的类别

        参数说明：
        --------
        x : np.ndarray
            单个样本的特征向量
        tree : dict/int
            决策树结构（递归遍历）

        返回值：
        --------
        int
            样本的预测类别
        """
        # 叶子节点：直接返回类别
        if not isinstance(tree, dict):
            return tree

        # 内部节点：按特征和阈值递归
        feature = tree['feature']
        threshold = tree['threshold']

        if x[feature] <= threshold:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])

    def predict(self, X):
        """
        预测多个样本的类别（对外接口）

        参数说明：
        --------
        X : np.ndarray
            待预测的特征矩阵

        返回值：
        --------
        np.ndarray
            预测类别数组
        """
        predictions = [self._predict_sample(x, self.tree) for x in X]
        return np.array(predictions)


# ===================== 4. 模型评估与可视化模块 =====================
def evaluate_model(y_true, y_pred):
    """
    评估分类模型性能：计算混淆矩阵、精确率、召回率、F1分数，并生成可视化图表

    参数说明：
    --------
    y_true : np.ndarray
        真实标签数组
    y_pred : np.ndarray
        预测标签数组

    返回值：
    --------
    np.ndarray
        混淆矩阵 (n_classes, n_classes)
    """
    classes = np.unique(y_true)
    n_classes = len(classes)

    # ========== 1. 计算混淆矩阵 ==========
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[int(t), int(p)] += 1  # 行=真实标签，列=预测标签

    # ========== 2. 计算分类指标 ==========
    overall_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)  # 整体准确率（对角线和/总数）
    precision = []  # 精确率：TP/(TP+FP)
    recall = []  # 召回率：TP/(TP+FN)
    f1 = []  # F1分数：2*P*R/(P+R)

    for i in range(n_classes):
        tp = conf_matrix[i, i]  # 真正例
        fp = np.sum(conf_matrix[:, i]) - tp  # 假正例
        fn = np.sum(conf_matrix[i, :]) - tp  # 假负例

        # 计算指标（避免除0）
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision.append(p)
        recall.append(r)
        f1.append(f)

    # ========== 3. 可视化：混淆矩阵热力图 ==========
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap='Blues')  # 蓝色系热力图
    plt.title('模型混淆矩阵', fontsize=14)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(range(n_classes), ['正常', '轻度异常', '中度异常', '重度异常'])
    plt.yticks(range(n_classes), ['正常', '轻度异常', '中度异常', '重度异常'])

    # 添加数值标注（显示每个单元格的样本数）
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black', fontsize=10)

    plt.colorbar(label='样本数量')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)  # 保存图片（150DPI）
    plt.show()  # 显示图片

    # ========== 4. 可视化：精确率/召回率/F1对比图 ==========
    plt.figure(figsize=(10, 6))
    x = np.arange(n_classes)
    width = 0.25  # 柱状图宽度

    # 绘制三组柱状图
    plt.bar(x - width, precision, width, label='精确率', color='#1f77b4')
    plt.bar(x, recall, width, label='召回率', color='#ff7f0e')
    plt.bar(x + width, f1, width, label='F1分数', color='#2ca02c')

    # 图表样式设置
    plt.title('模型分类性能指标', fontsize=14)
    plt.xlabel('异常等级', fontsize=12)
    plt.ylabel('指标值', fontsize=12)
    plt.xticks(x, ['正常', '轻度异常', '中度异常', '重度异常'])
    plt.ylim(0, 1.1)  # Y轴范围0-1.1（便于观察）
    plt.legend()
    plt.grid(axis='y', alpha=0.3)  # 添加水平网格线
    plt.tight_layout()
    plt.savefig('classification_metrics.png', dpi=150)
    plt.show()

    # ========== 5. 打印文本报告 ==========
    print("=== 模型性能报告 ===")
    print(f"{'类别':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8}")
    print("-" * 32)
    for i, cls in enumerate(classes):
        print(f"{int(cls):<8} {precision[i]:.4f}    {recall[i]:.4f}    {f1[i]:.4f}")
    print(f"\n整体准确率：{overall_accuracy:.4f}")

    return conf_matrix


def plot_temperature_distribution(X, y):
    """
    绘制关键部件温度分布直方图（正常vs异常样本对比）

    参数说明：
    --------
    X : np.ndarray
        特征矩阵
    y : np.ndarray
        标签数组
    """
    feature_names = ['电机温度(℃)', '电池温度(℃)', '控制器温度(℃)']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1行3列子图

    # 遍历三个温度特征绘制直方图
    for i, ax in enumerate(axes):
        # 分离正常/异常数据
        normal_data = X[y == 0, i]  # 正常样本
        anomaly_data = X[y != 0, i]  # 异常样本

        # 绘制密度直方图（density=True：归一化为概率密度）
        ax.hist(normal_data, bins=30, alpha=0.7, label='正常', color='green', density=True)
        ax.hist(anomaly_data, bins=30, alpha=0.7, label='异常', color='red', density=True)

        # 子图样式设置
        ax.set_title(feature_names[i], fontsize=12)
        ax.set_xlabel('温度值')
        ax.set_ylabel('概率密度')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('关键部件温度分布对比', fontsize=14)  # 总标题
    plt.tight_layout()
    plt.savefig('temperature_distribution.png', dpi=150)
    plt.show()


# ===================== 5. 实时监测与报警系统 =====================
class TemperatureAlarmSystem:
    """
    无人车温度实时报警系统（集成模型预测+硬件阈值+可视化）
    核心功能：模拟传感器数据 → 风险等级预测 → 报警触发 → 数据可视化

    参数说明：
    --------
    model : DecisionTreeClassifier
        训练好的决策树分类模型
    mean : np.ndarray
        特征均值（标准化用）
    std : np.ndarray
        特征标准差（标准化用）
    """

    def __init__(self, model, mean, std):
        self.model = model  # 分类模型
        self.mean = mean  # 标准化均值
        self.std = std  # 标准化标准差
        # 报警配置：风险等级→报警信息映射
        self.alarm_config = {
            0: {"level": "正常", "msg": "各部件温度正常", "action": "持续监测", "color": 'green'},
            1: {"level": "轻度异常", "msg": "温度略高于正常阈值", "action": "提醒驾驶员关注", "color": 'yellow'},
            2: {"level": "中度异常", "msg": "温度明显升高", "action": "降低车速", "color": 'orange'},
            3: {"level": "重度异常", "msg": "温度严重超标", "action": "紧急停车", "color": 'red'}
        }
        self.history = []  # 存储监测历史数据
        # 硬件安全阈值（直接触发重度报警）
        self.hard_thresholds = {"motor": 100, "battery": 70, "controller": 90}

    def _scale_data(self, data):
        """
        标准化传感器数据（与训练数据保持一致）

        参数说明：
        --------
        data : np.ndarray
            原始传感器数据 (1, 5)

        返回值：
        --------
        np.ndarray
            标准化后的数据
        """
        scaled = (data - self.mean) / self.std
        return scaled

    def _check_hard_threshold(self, sensor_data):
        """
        检查硬件安全阈值（优先级高于模型预测）
        若任一部件温度超过硬件阈值，直接返回重度异常（3）

        参数说明：
        --------
        sensor_data : np.ndarray
            原始传感器数据 (1, 5)

        返回值：
        --------
        int/None
            3=重度异常，None=未触发硬件阈值
        """
        motor_temp = sensor_data[0][0]
        battery_temp = sensor_data[0][1]
        controller_temp = sensor_data[0][2]

        # 检查是否超过硬件阈值
        if motor_temp >= self.hard_thresholds["motor"] or \
                battery_temp >= self.hard_thresholds["battery"] or \
                controller_temp >= self.hard_thresholds["controller"]:
            return 3
        return None

    def simulate_sensor(self):
        """
        模拟传感器实时数据生成（含随机异常）
        模拟逻辑：基础正常数据 + 随机概率注入异常

        返回值：
        --------
        np.ndarray
            传感器数据 (1, 5)：电机、电池、控制器、运行时长、车速
        """
        # 生成基础正常数据
        motor = random.uniform(50, 70)
        battery = random.uniform(40, 50)
        controller = random.uniform(45, 55)
        runtime = random.uniform(0, 10)
        speed = random.uniform(30, 50)

        # 随机注入异常
        anomaly_prob = random.random()
        if anomaly_prob < 0.05:  # 5%概率重度异常
            motor += 30
            battery += 20
            controller += 18
        elif anomaly_prob < 0.15:  # 10%概率中度异常
            motor += 15
            battery += 10
            controller += 10
        elif anomaly_prob < 0.3:  # 15%概率轻度异常
            motor += 8
            battery += 5
            controller += 5

        # 数值裁剪（物理约束）
        motor = max(0, min(motor, 120))
        battery = max(0, min(battery, 80))
        controller = max(0, min(controller, 100))
        runtime = max(0, min(runtime, 12))
        speed = max(0, min(speed, 120))

        return np.array([[motor, battery, controller, runtime, speed]])

    def predict_risk(self, sensor_data):
        """
        预测风险等级（硬件阈值优先 → 模型预测）

        参数说明：
        --------
        sensor_data : np.ndarray
            原始传感器数据 (1, 5)

        返回值：
        --------
        int
            风险等级：0=正常，1=轻度，2=中度，3=重度
        """
        # 第一步：检查硬件阈值（优先级最高）
        hard_risk = self._check_hard_threshold(sensor_data)
        if hard_risk is not None:
            return hard_risk

        # 第二步：模型预测（标准化后）
        scaled_data = self._scale_data(sensor_data)
        risk_level = self.model.predict(scaled_data)[0]
        return int(risk_level)

    def trigger_alarm(self, risk_level, sensor_data):
        """
        触发报警并记录日志

        参数说明：
        --------
        risk_level : int
            风险等级
        sensor_data : np.ndarray
            原始传感器数据

        返回值：
        --------
        dict
            报警日志
        """
        alarm_info = self.alarm_config[risk_level]
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # 格式化时间戳

        # 构造日志
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

        self.history.append(log)  # 保存到历史记录

        # 打印报警信息
        print(f"\n【{timestamp}】【{alarm_info['level']}】")
        print(f"传感器数据：电机{log['电机温度']}℃ | 电池{log['电池温度']}℃ | 控制器{log['控制器温度']}℃")
        print(f"报警信息：{alarm_info['msg']} | 建议操作：{alarm_info['action']}")

        return log

    def plot_monitoring_results(self):
        """
        可视化监测结果：温度趋势图 + 报警等级饼图
        """
        if not self.history:
            print("无监测数据可绘制")
            return

        # ========== 提取历史数据 ==========
        timestamps = [log["时间戳"] for log in self.history]
        motor_temps = [log["电机温度"] for log in self.history]
        battery_temps = [log["电池温度"] for log in self.history]
        controller_temps = [log["控制器温度"] for log in self.history]
        risk_levels = [log["风险等级"] for log in self.history]
        colors = [log["颜色"] for log in self.history]

        # 归一化时间戳（从0开始）
        start_ts = timestamps[0]
        timestamps = [ts - start_ts for ts in timestamps]

        # ========== 1. 温度趋势图 ==========
        plt.figure(figsize=(12, 6))
        # 绘制温度趋势线
        plt.plot(timestamps, motor_temps, 'o-', label='电机温度', linewidth=2, markersize=6)
        plt.plot(timestamps, battery_temps, 's-', label='电池温度', linewidth=2, markersize=6)
        plt.plot(timestamps, controller_temps, '^-', label='控制器温度', linewidth=2, markersize=6)

        # 标注异常点
        for i, (t, r) in enumerate(zip(timestamps, risk_levels)):
            if r > 0:
                plt.scatter(t, motor_temps[i], color=colors[i], s=100, edgecolor='black', zorder=5)
                plt.annotate(f'等级{r}', (t, motor_temps[i]), xytext=(5, 5), textcoords='offset points')

        # 图表样式
        plt.title('无人车温度实时监测趋势', fontsize=14)
        plt.xlabel('监测时长(秒)', fontsize=12)
        plt.ylabel('温度(℃)', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('temperature_trend.png', dpi=150)
        plt.show()

        # ========== 2. 报警等级统计饼图 ==========
        alarm_counts = defaultdict(int)
        for log in self.history:
            alarm_counts[log["报警级别"]] += 1

        labels = list(alarm_counts.keys())
        sizes = list(alarm_counts.values())
        # 匹配报警颜色
        colors = [self.alarm_config[0]["color"] if l == "正常" else
                  self.alarm_config[1]["color"] if l == "轻度异常" else
                  self.alarm_config[2]["color"] if l == "中度异常" else
                  self.alarm_config[3]["color"] for l in labels]

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
        plt.title('报警等级分布统计', fontsize=14)
        plt.axis('equal')  # 正圆形饼图
        plt.tight_layout()
        plt.savefig('alarm_distribution.png', dpi=150)
        plt.show()

    def run_monitor(self, duration=10, interval=2):
        """
        运行实时监测系统

        参数说明：
        --------
        duration : int
            总监测时长（秒），默认10秒
        interval : int
            采样间隔（秒），默认2秒
        """
        print("\n========== 无人车温度报警系统启动 ==========")
        print(f"监测时长：{duration}秒 | 采样间隔：{interval}秒")
        print("===========================================\n")

        start_time = time.time()
        # 循环采样直到达到监测时长
        while time.time() - start_time < duration:
            sensor_data = self.simulate_sensor()  # 生成模拟传感器数据
            risk_level = self.predict_risk(sensor_data)  # 预测风险等级
            self.trigger_alarm(risk_level, sensor_data)  # 触发报警
            time.sleep(interval)  # 等待采样间隔

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
    print(f"标签分布：{np.bincount(y.astype(int))}")  # 统计各标签数量

    # 绘制温度分布对比图
    plot_temperature_distribution(X, y)

    # 步骤2：数据预处理（标准化+划分训练/测试集）
    print("\n=== 2. 数据预处理 ===")
    X_scaled, mean, std = standard_scaler(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    print(f"训练集：{X_train.shape} | 测试集：{X_test.shape}")

    # 步骤3：训练决策树模型
    print("\n=== 3. 训练决策树模型 ===")
    model = DecisionTreeClassifier(max_depth=8, min_samples_split=5)  # 调整参数防止过拟合
    model.fit(X_train, y_train)
    print("模型训练完成！")

    # 步骤4：评估模型并可视化
    print("\n=== 4. 模型评估 ===")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)

    # 步骤5：启动实时报警系统
    print("\n=== 5. 启动温度报警系统 ===")
    alarm_system = TemperatureAlarmSystem(model, mean, std)
    alarm_system.run_monitor(duration=10, interval=2)  # 监测10秒，每2秒采样一次

    # 输出生成的图片列表
    print("\n=== 所有可视化图片已保存到当前目录 ===")
    print("生成的图片：")
    print("1. temperature_distribution.png - 温度分布对比图")
    print("2. confusion_matrix.png - 混淆矩阵热力图")
    print("3. classification_metrics.png - 分类性能指标图")
    print("4. temperature_trend.png - 实时温度趋势图")
    print("5. alarm_distribution.png - 报警等级统计饼图")