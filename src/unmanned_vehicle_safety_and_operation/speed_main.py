import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, Dict, List
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")  # 忽略无关警告


# ===================== 基础数据生成 =====================
def generate_velocity_data(
        test_duration: int = 60,
        sample_freq: int = 1,
        max_velocity: float = 30.0,
        noise_level: float = 0.5,
        add_abnormal: bool = True  # 是否加入异常值（模拟故障）
) -> Tuple[np.ndarray, np.ndarray]:
    """生成模拟无人车速度数据（含可选异常值）"""
    time_steps = np.arange(0, test_duration, 1 / sample_freq)
    num_samples = len(time_steps)
    velocity = np.zeros(num_samples)

    # 模拟加速→匀速→减速
    for i, t in enumerate(time_steps):
        if t < 10:
            v = (max_velocity / 10) * t
        elif 10 <= t < 40:
            v = max_velocity
        else:
            v = max_velocity - (max_velocity / 20) * (t - 40)

        # 加入噪声
        v += random.uniform(-noise_level, noise_level)
        velocity[i] = max(v, 0.0)

    # 随机加入2-3个异常值（模拟传感器故障/急刹）
    if add_abnormal:
        abnormal_idx = random.sample(range(num_samples), random.randint(2, 3))
        for idx in abnormal_idx:
            velocity[idx] = velocity[idx] * random.uniform(2, 3)  # 异常飙升
            if random.random() > 0.5:
                velocity[idx] = 0  # 异常归零

    return time_steps, velocity


# ===================== 机器学习：速度预测 =====================
def velocity_prediction(
        time_steps: np.ndarray,
        velocity: np.ndarray,
        predict_ratio: float = 0.2  # 预测未来20%的数据
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    基于线性回归预测未来速度
    :param time_steps: 时间戳数组
    :param velocity: 速度数组
    :param predict_ratio: 预测未来数据占比
    :return: 预测时间戳、预测速度、MAE误差
    """
    # 数据预处理：划分训练/预测时间范围
    total_len = len(time_steps)
    train_len = int(total_len * (1 - predict_ratio))
    train_time = time_steps[:train_len].reshape(-1, 1)  # 回归要求特征为2D
    train_vel = velocity[:train_len]
    predict_time = time_steps[train_len:].reshape(-1, 1)

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(train_time, train_vel)

    # 预测未来速度
    predict_vel = model.predict(predict_time)

    # 计算预测误差（仅当有真实值时）
    if len(predict_vel) == len(velocity[train_len:]):
        mae = mean_absolute_error(velocity[train_len:], predict_vel)
    else:
        mae = 0.0

    return predict_time.flatten(), predict_vel, mae


# ===================== 机器学习：异常检测 =====================
def detect_abnormal_velocity(
        velocity: np.ndarray,
        contamination: float = 0.05  # 异常值占比（默认5%）
) -> np.ndarray:
    """
    基于孤立森林检测速度异常值
    :param velocity: 速度数组
    :param contamination: 异常值比例
    :return: 异常标记数组（1=正常，-1=异常）
    """
    # 数据预处理：重塑为2D数组
    vel_2d = velocity.reshape(-1, 1)

    # 训练孤立森林模型
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    abnormal_label = model.fit_predict(vel_2d)  # 1=正常，-1=异常

    return abnormal_label


# ===================== 机器学习：驾驶模式分类 =====================
def classify_driving_mode(
        time_steps: np.ndarray,
        velocity: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    基于随机森林分类驾驶模式（0=减速，1=匀速，2=加速）
    :param time_steps: 时间戳数组
    :param velocity: 速度数组
    :return: 模式标签数组、分类准确率
    """
    # 特征工程：计算加速度（速度差分/时间差分）作为核心特征
    vel_diff = np.diff(velocity)
    time_diff = np.diff(time_steps)
    acceleration = vel_diff / time_diff

    # 构建标签：根据加速度分类
    # 加速度>0.5 → 加速(2)；-0.5<加速度<0.5 → 匀速(1)；加速度<-0.5 → 减速(0)
    labels = []
    for acc in acceleration:
        if acc > 0.5:
            labels.append(2)
        elif acc > -0.5:
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels)

    # 构建特征集：加速度+当前速度（需对齐长度）
    features = np.column_stack([
        acceleration,
        velocity[:-1]  # 差分后长度减1
    ])

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42
    )

    # 训练随机森林分类器
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 预测并计算准确率
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 预测全量数据的模式标签（补全最后一个点的标签）
    full_pred = model.predict(features)
    full_pred = np.append(full_pred, full_pred[-1])  # 补全长度

    return full_pred, accuracy


# ===================== 可视化整合 =====================
def plot_ml_results(
        time_steps: np.ndarray,
        velocity: np.ndarray,
        predict_time: np.ndarray,
        predict_vel: np.ndarray,
        abnormal_label: np.ndarray,
        mode_labels: np.ndarray
):
    """可视化机器学习分析结果"""
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建2行1列的子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # -------- 子图1：速度预测+异常检测 --------
    # 绘制原始速度
    ax1.plot(time_steps, velocity, color="#2E86AB", linewidth=2, label="原始速度")
    # 绘制预测速度
    ax1.plot(predict_time, predict_vel, color="#F18F01", linestyle="--", linewidth=2, label="预测速度")
    # 标注异常点
    abnormal_idx = np.where(abnormal_label == -1)[0]
    ax1.scatter(time_steps[abnormal_idx], velocity[abnormal_idx],
                color="#E63946", s=100, label="异常速度点", zorder=5)

    ax1.set_xlabel("时间 (秒)", fontsize=12)
    ax1.set_ylabel("速度 (km/h)", fontsize=12)
    ax1.set_title("无人车速度预测 + 异常检测", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # -------- 子图2：驾驶模式分类 --------
    # 定义模式颜色和标签
    mode_colors = {0: "#A23B72", 1: "#2E86AB", 2: "#F18F01"}
    mode_names = {0: "减速", 1: "匀速", 2: "加速"}

    # 绘制分类结果
    for mode in [0, 1, 2]:
        mode_idx = np.where(mode_labels == mode)[0]
        ax2.scatter(time_steps[mode_idx], velocity[mode_idx],
                    color=mode_colors[mode], label=mode_names[mode], s=50, alpha=0.7)

    ax2.set_xlabel("时间 (秒)", fontsize=12)
    ax2.set_ylabel("速度 (km/h)", fontsize=12)
    ax2.set_title("无人车驾驶模式分类", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ===================== 主函数 =====================
def main():
    # 1. 配置参数
    TEST_DURATION = 60
    SAMPLE_FREQ = 2  # 提高采样频率（每秒2次）
    MAX_VELOCITY = 30
    NOISE_LEVEL = 0.5

    # 2. 生成测试数据
    print("===== 生成无人车速度测试数据 =====")
    time_steps, velocity = generate_velocity_data(
        test_duration=TEST_DURATION,
        sample_freq=SAMPLE_FREQ,
        max_velocity=MAX_VELOCITY,
        noise_level=NOISE_LEVEL,
        add_abnormal=True
    )

    # 3. 速度预测
    print("\n===== 速度预测（线性回归） =====")
    predict_time, predict_vel, mae = velocity_prediction(time_steps, velocity)
    print(f"预测未来{len(predict_time)}个时间点的速度")
    print(f"预测平均绝对误差（MAE）：{mae:.2f} km/h")

    # 4. 异常检测
    print("\n===== 异常检测（孤立森林） =====")
    abnormal_label = detect_abnormal_velocity(velocity)
    abnormal_num = len(np.where(abnormal_label == -1)[0])
    print(f"检测到异常速度点数量：{abnormal_num} 个")
    abnormal_time = time_steps[abnormal_label == -1]
    print(f"异常点时间戳：{abnormal_time.round(2)}")

    # 5. 驾驶模式分类
    print("\n===== 驾驶模式分类（随机森林） =====")
    mode_labels, accuracy = classify_driving_mode(time_steps, velocity)
    print(f"分类准确率：{accuracy:.2%}")
    mode_count = {0: 0, 1: 0, 2: 0}
    for label in mode_labels:
        mode_count[label] += 1
    print(f"减速模式次数：{mode_count[0]} | 匀速模式次数：{mode_count[1]} | 加速模式次数：{mode_count[2]}")

    # 6. 可视化结果
    print("\n===== 可视化分析结果 =====")
    plot_ml_results(time_steps, velocity, predict_time, predict_vel, abnormal_label, mode_labels)


if __name__ == "__main__":
    main()