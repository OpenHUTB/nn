import math
import random
import sys

# 强制matplotlib使用TkAgg后端（PyCharm中显示图片必备）
import matplotlib

matplotlib.use('TkAgg')  # 关键：解决PyCharm中图片不显示的问题
import matplotlib.pyplot as plt

# 配置中文显示（PyCharm中兼容）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
VISUALIZATION_ENABLE = True


# -------------------------- 1. 数据生成（纯Python实现） --------------------------
def generate_driving_data(n_samples=10000):
    """生成人机共驾数据集（纯Python字典+列表实现）"""
    random.seed(42)
    data = {
        'road_type': [],  # 0:highway, 1:urban, 2:rural
        'weather': [],  # 0:clear, 1:rainy, 2:foggy
        'traffic_density': [],
        'driver_fatigue': [],
        'driver_attention': [],
        'driver_reaction_time': [],
        'vehicle_speed': [],
        'autopilot_confidence': [],
        'vehicle_stability': [],
        'takeover_time': []  # 目标变量：接管时间（秒）
    }

    for _ in range(n_samples):
        # 随机生成特征
        road_type = random.choice([0, 1, 2])
        weather = random.choice([0, 1, 2])
        traffic_density = random.uniform(0, 10)
        driver_fatigue = random.uniform(0, 1)
        driver_attention = random.uniform(0, 1)
        driver_reaction_time = random.uniform(0.2, 1.5)
        vehicle_speed = random.uniform(20, 120)
        autopilot_confidence = random.uniform(0, 1)
        vehicle_stability = random.uniform(0.8, 1)

        # 计算接管时间（加入特征关联）
        takeover_time = random.uniform(0, 10)
        takeover_time += driver_fatigue * 5  # 疲劳度影响
        takeover_time += (1 - autopilot_confidence) * 3  # 自动驾驶置信度影响
        takeover_time += weather * 1.0  # 天气影响（雨天+1，雾天+2）
        takeover_time += random.normalvariate(0, 0.5)  # 噪声

        # 填充数据
        data['road_type'].append(road_type)
        data['weather'].append(weather)
        data['traffic_density'].append(traffic_density)
        data['driver_fatigue'].append(driver_fatigue)
        data['driver_attention'].append(driver_attention)
        data['driver_reaction_time'].append(driver_reaction_time)
        data['vehicle_speed'].append(vehicle_speed)
        data['autopilot_confidence'].append(autopilot_confidence)
        data['vehicle_stability'].append(vehicle_stability)
        data['takeover_time'].append(takeover_time)

    # 模拟5%缺失值并填充（纯Python实现）
    for i in random.sample(range(n_samples), int(0.05 * n_samples)):
        data['driver_attention'][i] = None
    # 填充缺失值（均值填充）
    attention_vals = [v for v in data['driver_attention'] if v is not None]
    attention_mean = sum(attention_vals) / len(attention_vals)
    data['driver_attention'] = [v if v is not None else attention_mean for v in data['driver_attention']]

    return data


# -------------------------- 2. 数据预处理（纯Python实现） --------------------------
def normalize_feature(feature_list):
    """特征标准化（Z-score）：(x-mean)/std"""
    mean = sum(feature_list) / len(feature_list)
    std = math.sqrt(sum([(x - mean) ** 2 for x in feature_list]) / len(feature_list))
    if std == 0:
        return [0.0 for _ in feature_list]
    return [(x - mean) / std for x in feature_list]


def process_data(data):
    """数据预处理：标准化+特征合并+数据集划分"""
    # 1. 提取特征列表并标准化
    features = [
        'road_type', 'weather', 'traffic_density', 'driver_fatigue',
        'driver_attention', 'driver_reaction_time', 'vehicle_speed',
        'autopilot_confidence', 'vehicle_stability'
    ]
    normalized_data = {}
    for feat in features:
        normalized_data[feat] = normalize_feature(data[feat])
    normalized_data['takeover_time'] = data['takeover_time']  # 目标变量不标准化

    # 2. 合并特征为二维列表（每行是一个样本的所有特征）
    X = []
    y = []
    n_samples = len(data['takeover_time'])
    for i in range(n_samples):
        sample = [normalized_data[feat][i] for feat in features]
        sample.insert(0, 1.0)  # 加入偏置项（截距）
        X.append(sample)
        y.append(normalized_data['takeover_time'][i])

    # 3. 划分训练集/测试集（8:2）
    train_size = int(0.8 * n_samples)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    return X_train, y_train, X_test, y_test, features


# -------------------------- 3. 线性回归模型（纯Python实现） --------------------------
class LinearRegression:
    def __init__(self):
        self.weights = None  # 模型权重（包含偏置项）

    def _mean_squared_error(self, y_true, y_pred):
        """计算均方误差（MSE）"""
        return sum([(y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)]) / len(y_true)

    def fit(self, X, y, learning_rate=0.01, epochs=1000, verbose=False):
        """梯度下降训练模型"""
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_features)]  # 初始化权重

        for epoch in range(epochs):
            # 计算预测值
            y_pred = [sum([w * x for w, x in zip(self.weights, sample)]) for sample in X]
            # 计算梯度
            gradients = [0.0] * n_features
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                for j in range(n_features):
                    gradients[j] += error * X[i][j]
            gradients = [g / n_samples for g in gradients]

            # 更新权重
            self.weights = [w - learning_rate * g for w, g in zip(self.weights, gradients)]

            # 打印训练进度
            if verbose and (epoch % 100 == 0):
                mse = self._mean_squared_error(y, y_pred)
                print(f"Epoch {epoch}, MSE: {mse:.4f}")

    def predict(self, X):
        """预测"""
        if self.weights is None:
            raise ValueError("模型未训练，请先调用fit()")
        return [sum([w * x for w, x in zip(self.weights, sample)]) for sample in X]


# -------------------------- 4. 模型评估（纯Python实现） --------------------------
def evaluate_model(y_true, y_pred, set_name):
    """计算回归评估指标（MAE、RMSE、R²）"""
    # MAE
    mae = sum([abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred)]) / len(y_true)
    # MSE & RMSE
    mse = sum([(y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)]) / len(y_true)
    rmse = math.sqrt(mse)
    # R²
    y_mean = sum(y_true) / len(y_true)
    ss_total = sum([(y_t - y_mean) ** 2 for y_t in y_true])
    ss_res = sum([(y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)])
    r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0.0

    print(f"\n{set_name} 评估结果:")
    print(f"平均绝对误差(MAE): {mae:.4f} 秒")
    print(f"均方误差(MSE): {mse:.4f}")
    print(f"均方根误差(RMSE): {rmse:.4f} 秒")
    print(f"决定系数(R²): {r2:.4f}")
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}


# -------------------------- 5. PyCharm适配的可视化 --------------------------
def plot_results(y_test, y_test_pred, train_metrics, test_metrics, features, weights):
    """预测效果可视化（适配PyCharm显示）"""
    if not VISUALIZATION_ENABLE:
        return

    # 创建画布（加大分辨率，避免图片模糊）
    fig = plt.figure(figsize=(16, 12), dpi=100)

    # 5.1 真实值vs预测值（仅展示前500个点，提升绘制速度）
    ax1 = plt.subplot(2, 2, 1)
    plt.scatter(y_test[:500], y_test_pred[:500], alpha=0.6, color='steelblue', s=20)
    plt.plot([min(y_test[:500]), max(y_test[:500])],
             [min(y_test[:500]), max(y_test[:500])], 'r--', lw=2)
    plt.xlabel('真实接管时间(秒)', fontsize=10)
    plt.ylabel('预测接管时间(秒)', fontsize=10)
    plt.title('测试集：真实值 vs 预测值', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')

    # 5.2 残差分布
    ax2 = plt.subplot(2, 2, 2)
    residuals = [y_t - y_p for y_t, y_p in zip(y_test[:500], y_test_pred[:500])]
    plt.hist(residuals, bins=30, color='forestgreen', alpha=0.7, edgecolor='black')
    plt.xlabel('残差(真实值-预测值)', fontsize=10)
    plt.ylabel('频次', fontsize=10)
    plt.title('残差分布', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')

    # 5.3 特征重要性（权重绝对值）
    ax3 = plt.subplot(2, 2, 3)
    feat_importance = [abs(w) for w in weights[1:]]  # 排除偏置项
    plt.barh(features, feat_importance, color='darkviolet', alpha=0.8)
    plt.xlabel('权重绝对值（特征重要性）', fontsize=10)
    plt.ylabel('特征', fontsize=10)
    plt.title('特征重要性排名', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')

    # 5.4 指标对比
    ax4 = plt.subplot(2, 2, 4)
    metrics = ['MAE', 'RMSE', 'R²']
    train_vals = [train_metrics['mae'], train_metrics['rmse'], train_metrics['r2']]
    test_vals = [test_metrics['mae'], test_metrics['rmse'], test_metrics['r2']]

    x = [0, 1, 2]
    width = 0.35
    plt.bar([i - width / 2 for i in x], train_vals, width, label='训练集', color='skyblue', alpha=0.8)
    plt.bar([i + width / 2 for i in x], test_vals, width, label='测试集', color='orange', alpha=0.8)
    plt.xlabel('评估指标', fontsize=10)
    plt.ylabel('数值', fontsize=10)
    plt.title('训练集 vs 测试集 指标对比', fontsize=12, fontweight='bold')
    plt.xticks(x, metrics)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3, linestyle='--')

    # 调整布局，避免标签重叠
    plt.tight_layout(pad=2.0)

    # 关键：PyCharm中显示图片 + 保存图片到本地（可选）
    plt.show(block=True)  # block=True确保图片窗口阻塞，不会一闪而过
    # 可选：保存图片到项目目录（方便查看）
    fig.savefig('human_machine_co_driving_prediction.png', dpi=150, bbox_inches='tight')
    print("\n✅ 可视化图片已保存到项目根目录：human_machine_co_driving_prediction.png")


# -------------------------- 6. 人机共驾预测应用 --------------------------
def predict_takeover_time(features, scaler_stats, model):
    """
    预测接管时间
    :param features: 原始特征字典
    :param scaler_stats: 特征标准化的均值和标准差
    :param model: 训练好的线性回归模型
    :return: 预测的接管时间（秒）
    """
    # 特征映射（类别特征转数字）
    road_type_map = {'highway': 0, 'urban': 1, 'rural': 2}
    weather_map = {'clear': 0, 'rainy': 1, 'foggy': 2}
    features['road_type'] = road_type_map[features['road_type']]
    features['weather'] = weather_map[features['weather']]

    # 标准化（使用训练集的均值和标准差）
    feat_list = [
        features['road_type'], features['weather'],
        features['traffic_density'], features['driver_fatigue'],
        features['driver_attention'], features['driver_reaction_time'],
        features['vehicle_speed'], features['autopilot_confidence'],
        features['vehicle_stability']
    ]
    normalized_feat = []
    for i, feat in enumerate(feat_list):
        mean, std = scaler_stats[i]
        if std == 0:
            normalized_feat.append(0.0)
        else:
            normalized_feat.append((feat - mean) / std)
    normalized_feat.insert(0, 1.0)  # 加入偏置项

    # 预测
    takeover_time = model.predict([normalized_feat])[0]
    return round(takeover_time, 2)


# -------------------------- 主流程执行 --------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("开始运行人机共驾预测模型（PyCharm适配版）")
    print("=" * 50)

    # 1. 生成数据
    print("\n🔹 步骤1：生成人机共驾数据集...")
    data = generate_driving_data(n_samples=10000)

    # 2. 预处理数据
    print("\n🔹 步骤2：数据预处理（标准化+划分数据集）...")
    X_train, y_train, X_test, y_test, features = process_data(data)

    # 保存标准化的均值和标准差（用于预测）
    scaler_stats = []
    for feat in features:
        mean = sum(data[feat]) / len(data[feat])
        std = math.sqrt(sum([(x - mean) ** 2 for x in data[feat]]) / len(data[feat]))
        scaler_stats.append((mean, std))

    # 3. 训练模型
    print("\n🔹 步骤3：训练线性回归模型（梯度下降）...")
    model = LinearRegression()
    model.fit(X_train, y_train, learning_rate=0.01, epochs=1000, verbose=True)

    # 4. 预测与评估
    print("\n🔹 步骤4：模型评估...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_metrics = evaluate_model(y_train, y_train_pred, "训练集")
    test_metrics = evaluate_model(y_test, y_test_pred, "测试集")

    # 5. 可视化（PyCharm中显示+保存）
    print("\n🔹 步骤5：生成预测效果可视化...")
    plot_results(y_test, y_test_pred, train_metrics, test_metrics, features, model.weights)

    # 6. 应用示例
    print("\n🔹 步骤6：人机共驾接管时间预测示例...")
    # 理想场景
    ideal_scenario = {
        'road_type': 'highway',
        'weather': 'clear',
        'traffic_density': 2.0,
        'driver_fatigue': 0.1,
        'driver_attention': 0.9,
        'driver_reaction_time': 0.3,
        'vehicle_speed': 100,
        'autopilot_confidence': 0.95,
        'vehicle_stability': 0.98
    }
    # 危险场景
    dangerous_scenario = {
        'road_type': 'urban',
        'weather': 'rainy',
        'traffic_density': 8.0,
        'driver_fatigue': 0.8,
        'driver_attention': 0.2,
        'driver_reaction_time': 1.2,
        'vehicle_speed': 60,
        'autopilot_confidence': 0.4,
        'vehicle_stability': 0.9
    }

    # 11预测
    ideal_time = predict_takeover_time(ideal_scenario, scaler_stats, model)
    dangerous_time = predict_takeover_time(dangerous_scenario, scaler_stats, model)

    print("\n=== 人机共驾接管时间预测结果 ===")
    print(f"🌞 理想场景（高速/晴天/低疲劳）：{ideal_time} 秒")
    print(f"⚠️  危险场景（城市/雨天/高疲劳）：{dangerous_time} 秒")

    # 决策建议
    if dangerous_time > 5:
        print("\n🚨 预警：危险场景下驾驶员接管时间过长，建议立即降低车速/提醒驾驶员！")
    else:
        print("\n✅ 安全：接管时间在合理范围内")

    print("\n=" * 50)
    print("模型运行完成！")
    print("=" * 50)