import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")


# ===================== 1. 数据生成（纯numpy+内置库） =====================
def generate_vehicle_motion_data(
        sample_num: int = 1000,  # 样本数量
        noise_level: float = 0.1  # 传感器噪声
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成无人车运动数据（纯numpy实现）
    特征：速度(km/h)、加速度(km/h/s)、方向盘角度(°)、偏航角(°)、角速度(°/s)
    标签：运动方向（0=直行，1=左转，2=右转）
    """
    # 初始化特征矩阵（sample_num行 × 5列）和标签数组
    features = np.zeros((sample_num, 5), dtype=np.float32)
    labels = np.zeros(sample_num, dtype=np.int32)

    # 随机生成数据
    for i in range(sample_num):
        # 基础特征（速度、加速度）
        speed = np.random.uniform(0, 40)  # 0-40km/h
        acceleration = np.random.uniform(-2, 2)  # -2~2 km/h/s

        # 随机选择运动方向（直行60%、左转20%、右转20%）
        direction = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])

        # 根据方向生成核心特征
        if direction == 0:  # 直行
            steering_angle = np.random.normal(0, 1)  # 方向盘角度接近0°
            yaw_angle = np.random.normal(0, 0.5)  # 偏航角接近0°
            angular_velocity = np.random.normal(0, 0.3)  # 角速度接近0°/s
        elif direction == 1:  # 左转
            steering_angle = np.random.uniform(10, 30)  # 方向盘左偏10-30°
            yaw_angle = np.random.uniform(5, 15)  # 偏航角左偏5-15°
            angular_velocity = np.random.uniform(1, 3)  # 左转角速度1-3°/s
        else:  # 右转
            steering_angle = np.random.uniform(-30, -10)  # 方向盘右偏10-30°
            yaw_angle = np.random.uniform(-15, -5)  # 偏航角右偏5-15°
            angular_velocity = np.random.uniform(-3, -1)  # 右转角速度1-3°/s

        # 加入传感器噪声
        steering_angle += np.random.normal(0, noise_level)
        yaw_angle += np.random.normal(0, noise_level)
        angular_velocity += np.random.normal(0, noise_level)

        # 赋值到特征矩阵（保留2位小数）
        features[i] = [
            round(speed, 2),
            round(acceleration, 2),
            round(steering_angle, 2),
            round(yaw_angle, 2),
            round(angular_velocity, 2)
        ]
        labels[i] = direction

    return features, labels


# ===================== 2. 数据预处理 =====================
def preprocess_data(features: np.ndarray, labels: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    数据预处理：标准化 + 划分训练/测试集
    """
    # 划分训练集（80%）和测试集（20%），保证标签分布均匀
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 特征标准化（提升模型性能）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ===================== 3. 模型训练 =====================
def train_direction_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = "random_forest"  # 可选：random_forest / logistic_regression
) -> object:
    """
    训练运动方向预测模型
    """
    if model_type == "random_forest":
        # 随机森林（核心模型，鲁棒性强）
        model = RandomForestClassifier(
            n_estimators=100,  # 决策树数量
            max_depth=8,  # 防止过拟合
            random_state=42
        )
    elif model_type == "logistic_regression":
        # 逻辑回归（基准模型，易解释）
        model = LogisticRegression(max_iter=200, random_state=42)
    else:
        raise ValueError("仅支持 random_forest / logistic_regression 模型")

    # 训练模型
    model.fit(X_train, y_train)
    return model


# ===================== 4. 模型评估（无pandas可视化） =====================
def evaluate_model(
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str]
) -> Dict[str, float]:
    """
    评估模型性能，输出指标并绘制混淆矩阵（纯matplotlib实现）
    """
    # 预测测试集
    y_pred = model.predict(X_test)

    # 计算核心指标
    accuracy = accuracy_score(y_test, y_pred)

    # 输出分类报告
    print("\n===== 分类详细报告 =====")
    print(classification_report(
        y_test, y_pred,
        target_names=["直行", "左转", "右转"]
    ))

    # 输出特征重要性（仅随机森林支持）
    if hasattr(model, "feature_importances_"):
        print("\n===== 特征重要性 =====")
        importance = model.feature_importances_
        for name, imp in zip(feature_names, importance):
            print(f"{name}: {imp:.4f}")

    # 绘制混淆矩阵（纯matplotlib，无pandas依赖）
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # 设置坐标轴标签
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=["直行", "左转", "右转"],
           yticklabels=["直行", "左转", "右转"],
           title="混淆矩阵（运动方向预测）",
           ylabel="真实标签",
           xlabel="预测标签")

    # 标注数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()

    return {"accuracy": accuracy}


# ===================== 5. 特征关系可视化（纯matplotlib） =====================
def visualize_feature_relationship(features: np.ndarray, labels: np.ndarray):
    """
    可视化核心特征与运动方向的关系（无pandas依赖）
    """
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 特征名称映射
    feature_idx = {
        "speed": 0,
        "steering_angle": 2,
        "yaw_angle": 3,
        "angular_velocity": 4
    }

    # 创建2x1子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 子图1：方向盘角度 vs 速度（散点图）
    # 直行（绿）、左转（红）、右转（蓝）
    ax1.scatter(
        features[labels == 0, feature_idx["steering_angle"]],
        features[labels == 0, feature_idx["speed"]],
        label="直行", alpha=0.6, s=30, color="green"
    )
    ax1.scatter(
        features[labels == 1, feature_idx["steering_angle"]],
        features[labels == 1, feature_idx["speed"]],
        label="左转", alpha=0.6, s=30, color="red"
    )
    ax1.scatter(
        features[labels == 2, feature_idx["steering_angle"]],
        features[labels == 2, feature_idx["speed"]],
        label="右转", alpha=0.6, s=30, color="blue"
    )
    ax1.set_xlabel("方向盘角度 (°)")
    ax1.set_ylabel("速度 (km/h)")
    ax1.set_title("方向盘角度 vs 速度 vs 运动方向")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 子图2：角速度分布（直方图）
    ax2.hist(
        [features[labels == 0, feature_idx["angular_velocity"]],
         features[labels == 1, feature_idx["angular_velocity"]],
         features[labels == 2, feature_idx["angular_velocity"]]],
        bins=20, label=["直行", "左转", "右转"],
        color=["green", "red", "blue"], alpha=0.6
    )
    ax2.set_xlabel("角速度 (°/s)")
    ax2.set_ylabel("样本数量")
    ax2.set_title("角速度分布 vs 运动方向")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ===================== 6. 单样本实时预测 =====================
def predict_single_sample(
        model: object,
        scaler: object,
        sample: List[float]  # [speed, acceleration, steering_angle, yaw_angle, angular_velocity]
) -> Tuple[int, str]:
    """
    预测单个样本的运动方向（模拟无人车在线推理）
    """
    # 特征标准化
    sample_scaled = scaler.transform([sample])

    # 预测方向
    pred_label = model.predict(sample_scaled)[0]

    # 标签转名称
    direction_map = {0: "直行", 1: "左转", 2: "右转"}
    pred_name = direction_map[pred_label]

    return pred_label, pred_name


# ===================== 主函数（整合全流程） =====================
def main():
    # 1. 生成模拟数据
    print("===== 1. 生成无人车运动数据 =====")
    features, labels = generate_vehicle_motion_data(sample_num=1000, noise_level=0.1)
    print(f"生成样本数量：{len(features)} 条")
    print("前5条特征数据：")
    print(features[:5])
    print("前5条标签（0=直行，1=左转，2=右转）：", labels[:5])

    # 2. 可视化特征关系
    print("\n===== 2. 可视化特征与运动方向的关系 =====")
    visualize_feature_relationship(features, labels)

    # 3. 数据预处理
    print("\n===== 3. 数据预处理 =====")
    feature_names = ["speed", "acceleration", "steering_angle", "yaw_angle", "angular_velocity"]
    X_train, X_test, y_train, y_test, scaler = preprocess_data(features, labels)
    print(f"训练集数量：{len(X_train)} 条，测试集数量：{len(X_test)} 条")

    # 4. 训练模型
    print("\n===== 4. 训练运动方向预测模型 =====")
    model = train_direction_model(X_train, y_train, model_type="random_forest")

    # 5. 评估模型
    print("\n===== 5. 模型评估 =====")
    eval_metrics = evaluate_model(model, X_test, y_test, feature_names)
    print(f"模型整体准确率：{eval_metrics['accuracy']:.2%}")

    # 6. 单样本实时预测
    print("\n===== 6. 单样本实时预测 =====")
    # 示例1：直行样本
    sample1 = [20.0, 0.5, 0.2, 0.1, 0.05]
    pred_label1, pred_name1 = predict_single_sample(model, scaler, sample1)
    print(f"样本1特征：{sample1} → 预测结果：{pred_name1}（标签：{pred_label1}）")

    # 示例2：左转样本
    sample2 = [15.0, 0.3, 20.0, 10.0, 2.0]
    pred_label2, pred_name2 = predict_single_sample(model, scaler, sample2)
    print(f"样本2特征：{sample2} → 预测结果：{pred_name2}（标签：{pred_label2}）")

    # 示例3：右转样本
    sample3 = [18.0, 0.2, -18.0, -8.0, -1.5]
    pred_label3, pred_name3 = predict_single_sample(model, scaler, sample3)
    print(f"样本3特征：{sample3} → 预测结果：{pred_name3}（标签：{pred_label3}）")


if __name__ == "__main__":
    main()