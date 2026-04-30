import numpy as np
import random
import math
import matplotlib.pyplot as plt

# ===================== 全局配置（解决PyCharm中文显示/图片样式） =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 兼容不同系统中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.rcParams['figure.figsize'] = (14, 9)      # 图片默认尺寸
plt.rcParams['savefig.dpi'] = 300             # 保存图片的分辨率
plt.rcParams['figure.dpi'] = 100              # 显示图片的分辨率

# ===================== 1.11 生成无人车刹车场景数据 111=====================
def generate_vehicle_data(n_samples=8000):
    """生成模拟数据集（纯Python实现，含物理模型+噪声）"""
    random.seed(42)  # 固定随机种子，保证结果可复现
    features = []
    labels = []

    for _ in range(n_samples):
        # 核心特征（无人车感知数据）
        vehicle_speed = random.uniform(0, 120)       # 车速(km/h)
        obstacle_distance = random.uniform(0, 200)   # 障碍物距离(m)
        obstacle_speed = random.uniform(-50, 50)     # 障碍物相对速度(km/h)
        road_friction = random.uniform(0.1, 1.0)     # 路面摩擦系数（雨雪天低/干燥高）
        weather_visibility = random.uniform(50, 1000)# 能见度(m)
        brake_delay = random.uniform(0.05, 0.2)      # 刹车系统延迟(s)

        # 刹车时间核心公式（物理模型）
        base_time = (obstacle_distance / (vehicle_speed/3.6)) * (1/road_friction) - brake_delay
        # 环境修正项
        vis_correction = 0.5 if weather_visibility < 200 else 0.1  # 低能见度需更早刹车
        obs_correction = obstacle_speed / 100                      # 障碍物速度修正
        noise = random.gauss(0, 0.15)                             # 高斯噪声模拟真实误差

        # 限制刹车时间合理范围（0.1~5秒）
        brake_time = max(0.1, min(base_time + vis_correction + obs_correction + noise, 5.0))

        features.append([vehicle_speed, obstacle_distance, obstacle_speed, road_friction, weather_visibility, brake_delay])
        labels.append(brake_time)

    return np.array(features), np.array(labels)

# ===================== 2. 特征归一化（纯Python实现） =====================
class StandardScaler:
    """均值-标准差归一化，适配PyCharm运行"""
    def __init__(self):
        self.feature_mean = None
        self.feature_std = None

    def fit(self, X):
        """计算每个特征的均值和标准差"""
        n_features = X.shape[1]
        self.feature_mean = np.zeros(n_features)
        self.feature_std = np.zeros(n_features)

        for i in range(n_features):
            col_data = X[:, i]
            self.feature_mean[i] = np.mean(col_data)
            self.feature_std[i] = np.std(col_data) + 1e-8  # 避免除以0

    def transform(self, X):
        """应用归一化"""
        return (X - self.feature_mean) / self.feature_std

# ===================== 3. 简化版决策树回归（核心预测模型） =====================
class SimpleDecisionTree:
    """轻量级决策树，适配PyCharm低资源运行"""
    def __init__(self, max_depth=3, min_samples=5):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = {}
        self.feat_importance = np.zeros(6)  # 特征重要性（6个特征）

    def _mse(self, y):
        """计算均方误差（回归损失）"""
        if len(y) == 0:
            return 0.0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)

    def _best_split(self, X, y):
        """寻找最优分割特征和阈值"""
        best_feat = -1
        best_thresh = None
        best_mse = self._mse(y)
        n_features = X.shape[1]

        for feat_idx in range(n_features):
            # 去重阈值，减少计算量
            thresholds = np.unique(X[:, feat_idx])[:20]  # 限制阈值数量，加速PyCharm运行
            for thresh in thresholds:
                # 分割数据集
                left_mask = X[:, feat_idx] <= thresh
                right_mask = ~left_mask

                # 跳过样本数不足的分割
                if len(y[left_mask]) < self.min_samples or len(y[right_mask]) < self.min_samples:
                    continue

                # 计算分割后的总MSE
                mse_left = self._mse(y[left_mask])
                mse_right = self._mse(y[right_mask])
                total_mse = (len(y[left_mask]) * mse_left + len(y[right_mask]) * mse_right) / len(y)

                # 更新最优分割
                if total_mse < best_mse:
                    best_mse = total_mse
                    best_feat = feat_idx
                    best_thresh = thresh

        # 累计特征重要性
        if best_feat != -1:
            self.feat_importance[best_feat] += (self._mse(y) - best_mse) * len(y)
        return best_feat, best_thresh

    def _build_tree(self, X, y, depth):
        """递归构建决策树"""
        # 终止条件：深度达标/损失足够小
        if depth >= self.max_depth or self._mse(y) < 1e-5:
            return {"value": np.mean(y)}

        # 寻找最优分割
        feat_idx, thresh = self._best_split(X, y)
        if feat_idx == -1:
            return {"value": np.mean(y)}

        # 分割数据集并递归构建子树
        left_mask = X[:, feat_idx] <= thresh
        right_mask = ~left_mask
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "feature": feat_idx,
            "threshold": thresh,
            "left": left_tree,
            "right": right_tree
        }

    def fit(self, X, y):
        """训练决策树"""
        self.tree = self._build_tree(X, y, depth=0)
        # 归一化特征重要性
        self.feat_importance = self.feat_importance / np.sum(self.feat_importance) if np.sum(self.feat_importance) > 0 else self.feat_importance

    def _predict_single(self, x):
        """预测单个样本"""
        tree_node = self.tree
        while "feature" in tree_node:
            if x[tree_node["feature"]] <= tree_node["threshold"]:
                tree_node = tree_node["left"]
            else:
                tree_node = tree_node["right"]
        return tree_node["value"]

    def predict(self, X):
        """批量预测"""
        return np.array([self._predict_single(x) for x in X])

# ===================== 4. 梯度提升回归（集成模型） =====================
class GradientBoostRegressor:
    """简化版梯度提升，适配PyCharm快速运行"""
    def __init__(self, n_trees=80, lr=0.1, max_depth=3):
        self.n_trees = n_trees       # 树的数量（减少数量加速运行）
        self.lr = lr                 # 学习率
        self.max_depth = max_depth   # 树深度
        self.trees = []              # 保存所有树
        self.base_pred = None        # 初始预测值
        self.total_feat_importance = np.zeros(6)  # 总特征重要性

    def fit(self, X, y):
        """训练梯度提升模型"""
        # 初始预测：所有样本的均值
        self.base_pred = np.mean(y)
        y_pred = np.full(len(y), self.base_pred)

        print("模型训练进度：")
        for i in range(self.n_trees):
            # 计算残差（负梯度）
            residual = y - y_pred

            # 训练一棵决策树拟合残差
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X, residual)

            # 累加特征重要性
            self.total_feat_importance += tree.feat_importance

            # 更新预测值
            tree_pred = tree.predict(X)
            y_pred += self.lr * tree_pred
            self.trees.append(tree)

            # 打印训练进度（PyCharm控制台可见）
            if (i + 1) % 20 == 0:
                mse = np.mean((y - y_pred) ** 2)
                print(f"  完成 {i+1}/{self.n_trees} 棵树，当前MSE：{mse:.4f}")

        # 归一化总特征重要性
        self.total_feat_importance = self.total_feat_importance / np.sum(self.total_feat_importance) if np.sum(self.total_feat_importance) > 0 else self.total_feat_importance

    def predict(self, X):
        """预测刹车时间"""
        y_pred = np.full(len(X), self.base_pred)
        for tree in self.trees:
            y_pred += self.lr * tree.predict(X)
        return y_pred

# ===================== 5. 可视化绘图函数（PyCharm专用） =====================
def plot_results(y_true, y_pred, feat_importance, X_test):
    """生成4张子图，在PyCharm中显示并保存"""
    # 创建2x2子图布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))

    # 子图1：真实值vs预测值散点图
    ax1.scatter(y_true, y_pred, color="#2E86AB", alpha=0.6, s=8)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", linewidth=2, label="完美预测线")
    ax1.set_xlabel("真实刹车时间 (秒)", fontsize=10)
    ax1.set_ylabel("预测刹车时间 (秒)", fontsize=10)
    ax1.set_title("刹车时间：真实值 vs 预测值", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 子图2：特征重要性柱状图
    feat_names = ["车速", "障碍物距离", "障碍物速度", "路面摩擦", "能见度", "刹车延迟"]
    ax2.bar(feat_names, feat_importance, color="#A23B72", alpha=0.8)
    ax2.set_xlabel("特征名称", fontsize=10)
    ax2.set_ylabel("重要性权重", fontsize=10)
    ax2.set_title("特征重要性排名", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)  # 旋转x轴标签，避免重叠
    ax2.grid(alpha=0.3, axis="y")

    # 子图3：预测误差分布直方图
    error = y_true - y_pred
    ax3.hist(error, bins=40, color="#F18F01", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax3.axvline(x=0, color="red", linestyle="--", linewidth=2, label="无误差线")
    ax3.set_xlabel("预测误差 (秒)", fontsize=10)
    ax3.set_ylabel("样本数量", fontsize=10)
    ax3.set_title("预测误差分布", fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 子图4：不同路面摩擦系数的刹车时间箱线图
    friction_bins = [0.1, 0.4, 0.7, 1.0]
    friction_labels = ["低摩擦(雨雪)", "中摩擦", "高摩擦(干燥)"]
    friction_groups = []
    for i in range(3):
        mask = (X_test[:, 3] >= friction_bins[i]) & (X_test[:, 3] < friction_bins[i+1])
        friction_groups.append(y_true[mask])

    box_plot = ax4.boxplot(friction_groups, labels=friction_labels, patch_artist=True)
    for patch in box_plot["boxes"]:
        patch.set_facecolor("#C73E1D")
        patch.set_alpha(0.7)
    ax4.set_xlabel("路面摩擦系数区间", fontsize=10)
    ax4.set_ylabel("刹车时间 (秒)", fontsize=10)
    ax4.set_title("不同路面摩擦的刹车时间分布", fontsize=12, fontweight="bold")
    ax4.grid(alpha=0.3, axis="y")

    # 调整子图间距（避免标签重叠）
    plt.tight_layout(pad=2.0)

    # 关键：在PyCharm中显示图片（内置Plot窗口）
    plt.show()

    # 保存图片到本地（代码同级目录）
    fig.savefig("无人车刹车时间分析图.png", dpi=300, bbox_inches="tight")
    print("\n✅ 分析图片已保存为：无人车刹车时间分析图.png")

# ===================== 6. 主函数（PyCharm运行入口） =====================
if __name__ == "__main__":
    print("="*50)
    print("        无人车紧急刹车时间预测模型（PyCharm专用）        ")
    print("="*50)

    # 1. 生成数据集
    print("\n📊 正在生成模拟数据集...")
    X, y = generate_vehicle_data(n_samples=8000)  # 减少样本数加速运行

    # 2. 划分训练集/测试集（8:2）
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"✅ 数据集划分完成：训练集{len(X_train)}条，测试集{len(X_test)}条")

    # 3. 特征归一化
    print("\n🔧 正在进行特征归一化...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 训练模型
    print("\n🚀 正在训练梯度提升模型...")
    model = GradientBoostRegressor(n_trees=80, lr=0.1, max_depth=3)
    model.fit(X_train_scaled, y_train)

    # 5. 模型评估
    print("\n📈 模型评估结果：")
    y_pred = model.predict(X_test_scaled)
    mae = np.mean(np.abs(y_test - y_pred))  # 平均绝对误差
    r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))  # 决定系数
    print(f"  平均绝对误差（MAE）：{mae:.4f} 秒（越小越好）")
    print(f"  决定系数（R²）：{r2:.4f}（越接近1越好）")

    # 6. 生成可视化图片（PyCharm显示+本地保存）
    print("\n🎨 正在生成分析图片...")
    plot_results(y_test, y_pred, model.total_feat_importance, X_test)

    # 7. 实时预测示例（模拟无人车实时感知数据）
    print("\n🔍 实时刹车时间预测示例：")
    def predict_brake_time(vehicle_state):
        """输入车辆状态，预测刹车时间"""
        # 转换为模型输入格式
        state_arr = np.array([list(vehicle_state.values())])
        state_scaled = scaler.transform(state_arr)
        # 预测并限制范围a
        brake_time = model.predict(state_scaled)[0]
        return max(0.1, min(brake_time, 5.0))

    # 模拟真实场景：高速+干燥路面+近距离障碍物
    test_state = {
        "vehicle_speed": 90.0,       # 车速90km/h
        "obstacle_distance": 70.0,   # 障碍物距离70m
        "obstacle_speed": 8.0,       # 障碍物同向8km/h
        "road_friction": 0.8,        # 干燥路面摩擦系数
        "weather_visibility": 600.0, # 能见度600m
        "brake_delay": 0.12          # 刹车系统延迟0.12s
    }

    # 预测并输出结果
    pred_time = predict_brake_time(test_state)
    print("📌 测试场景：高速干燥路面+近距离障碍物")
    for k, v in test_state.items():
        print(f"  {k}：{v}")
    print(f"✅ 预测紧急刹车时间：{pred_time:.2f} 秒")
    print("\n🎉 所有任务完成！")