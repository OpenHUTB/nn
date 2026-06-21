import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# ===================== 1. 烟雾传感器数据生成 =====================
def generate_smoke_sensor_data(
    sample_num: int = 1000,
    sample_freq: int = 1,
    noise_level: float = 0.05,
    add_abnormal: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成无人车烟雾传感器数据
    标签：0=无报警, 1=低风险, 2=高风险
    """
    time_steps = np.arange(0, sample_num / sample_freq, 1 / sample_freq)
    smoke_conc = np.zeros(sample_num, dtype=np.float32)
    base_conc = np.random.uniform(0, 5, sample_num)

    # 低风险 15%
    low_risk_idx = np.random.choice(sample_num, int(sample_num * 0.15), replace=False)
    smoke_conc[low_risk_idx] = np.random.uniform(5, 20, len(low_risk_idx))

    # 高风险 5%
    high_risk_idx = np.random.choice(sample_num, int(sample_num * 0.05), replace=False)
    smoke_conc[high_risk_idx] = np.random.uniform(20, 50, len(high_risk_idx))

    # 正常段
    normal_idx = np.setdiff1d(np.arange(sample_num), np.union1d(low_risk_idx, high_risk_idx))
    smoke_conc[normal_idx] = base_conc[normal_idx]

    # 加噪声
    smoke_conc += np.random.normal(0, noise_level, sample_num)
    smoke_conc = np.maximum(smoke_conc, 0)

    # 突发异常
    if add_abnormal:
        abnormal_idx = np.random.choice(sample_num, np.random.randint(3, 8), replace=False)
        smoke_conc[abnormal_idx] = np.random.uniform(60, 100, len(abnormal_idx))

    # 标签生成
    labels = np.zeros(sample_num, dtype=np.int32)
    labels[(smoke_conc >= 5) & (smoke_conc < 20)] = 1
    labels[smoke_conc >= 20] = 2

    smoke_conc = np.round(smoke_conc, 2)
    return time_steps, smoke_conc, labels

# ===================== 2. 数据预处理 =====================
def preprocess_smoke_data(
    smoke_conc: np.ndarray,
    labels: np.ndarray,
    sample_freq: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, np.ndarray]:
    """
    构建时序特征 + 标准化 + 训练/测试集划分
    """
    prev_conc = np.concatenate([[0], smoke_conc[:-1]])
    conc_diff = np.diff(smoke_conc, prepend=0) * sample_freq
    features = np.column_stack([smoke_conc, prev_conc, conc_diff])

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

# ===================== 3. 烟雾异常检测 =====================
def detect_smoke_abnormal(smoke_conc: np.ndarray, contamination: float = 0.08) -> np.ndarray:
    """孤立森林异常检测"""
    conc_2d = smoke_conc.reshape(-1, 1)
    iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    return iso_forest.fit_predict(conc_2d)

# ===================== 4. 报警等级模型训练 =====================
def train_smoke_alarm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest"
) -> object:
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    else:
        raise ValueError("仅支持 random_forest 模型")
    model.fit(X_train, y_train)
    return model

# ===================== 5. 模型评估 =====================
def evaluate_alarm_model(model: object, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n===== 报警等级分类报告 =====")
    print(classification_report(y_test, y_pred, target_names=["无报警", "低风险", "高风险"]))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["无报警", "低风险", "高风险"],
        yticklabels=["无报警", "低风险", "高风险"],
        title="烟雾报警等级混淆矩阵",
        ylabel="真实等级",
        xlabel="预测等级"
    )
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return {"accuracy": accuracy}

# ===================== 6. 可视化 =====================
def visualize_smoke_alarm(time_steps: np.ndarray, smoke_conc: np.ndarray, abnormal_label: np.ndarray, alarm_level: np.ndarray):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 烟雾浓度 + 异常
    ax1.plot(time_steps, smoke_conc, color="#1f77b4", linewidth=1.5, label="烟雾浓度")
    alarm_idx = np.where(abnormal_label == -1)[0]
    ax1.scatter(time_steps[alarm_idx], smoke_conc[alarm_idx], color="#d62728", s=80, label="报警触发点", zorder=5)
    ax1.axhline(y=5, color="#ff7f0e", linestyle="--", label="低风险阈值")
    ax1.axhline(y=20, color="#d62728", linestyle="--", label="高风险阈值")
    ax1.set_xlabel("时间 (秒)")
    ax1.set_ylabel("烟雾浓度 (ppm)")
    ax1.set_title("无人车烟雾浓度监测曲线 + 报警标记")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # 报警等级
    colors = np.array(["#2ca02c", "#ff7f0e", "#d62728"])[alarm_level]
    ax2.scatter(time_steps, smoke_conc, c=colors, s=20, alpha=0.7, label="报警等级")
    for color, label in zip(["#2ca02c", "#ff7f0e", "#d62728"], ["无报警","低风险","高风险"]):
        ax2.scatter([], [], color=color, s=50, label=label)
    ax2.set_xlabel("时间 (秒)")
    ax2.set_ylabel("烟雾浓度 (ppm)")
    ax2.set_title("无人车烟雾报警等级分类")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ===================== 7. 实时单样本监测 =====================
def realtime_smoke_monitor(model: object, scaler: object, current_conc: float, prev_conc: float, sample_freq: int = 1) -> Tuple[int, str]:
    conc_diff = (current_conc - prev_conc) * sample_freq
    features = np.array([[current_conc, prev_conc, conc_diff]])
    features_scaled = scaler.transform(features)
    level_label = model.predict(features_scaled)[0]
    level_map = {0: "无报警", 1: "低风险", 2: "高风险"}
    return level_label, level_map[level_label]

# ===================== 8. 主函数 =====================
def main():
    SAMPLE_NUM = 1000
    SAMPLE_FREQ = 1
    NOISE_LEVEL = 0.05

    print("===== 1. 生成无人车烟雾传感器数据 =====")
    time_steps, smoke_conc, labels = generate_smoke_sensor_data(SAMPLE_NUM, SAMPLE_FREQ, NOISE_LEVEL)
    print(f"生成采样点数：{len(smoke_conc)}，时间范围：0 ~ {time_steps[-1]:.1f} 秒，浓度范围：{smoke_conc.min():.2f} ~ {smoke_conc.max():.2f} ppm")

    print("\n===== 2. 烟雾异常检测 =====")
    abnormal_label = detect_smoke_abnormal(smoke_conc)
    print(f"检测到报警触发点数量：{len(np.where(abnormal_label==-1)[0])} 个")

    print("\n===== 3. 数据预处理 =====")
    X_train, X_test, y_train, y_test, scaler, features = preprocess_smoke_data(smoke_conc, labels, SAMPLE_FREQ)
    print(f"训练集数量：{len(X_train)}，测试集数量：{len(X_test)}")

    print("\n===== 4. 训练报警等级模型 =====")
    alarm_model = train_smoke_alarm_model(X_train, y_train)

    print("\n===== 5. 模型评估 =====")
    eval_metrics = evaluate_alarm_model(alarm_model, X_test, y_test)
    print(f"报警等级分类准确率：{eval_metrics['accuracy']:.2%}")

    print("\n===== 6. 全量数据等级预测 =====")
    features_scaled = scaler.transform(features)
    alarm_level = alarm_model.predict(features_scaled)
    print({level: list(alarm_level).count(level) for level in [0,1,2]})

    print("\n===== 7. 可视化烟雾监测结果 =====")
    visualize_smoke_alarm(time_steps, smoke_conc, abnormal_label, alarm_level)

    print("\n===== 8. 实时监测示例 =====")
    for current, prev, desc in [(3.0, 2.8, "正常"), (10.0, 8.5, "低风险"), (25.0, 20.0, "高风险")]:
        level, name = realtime_smoke_monitor(alarm_model, scaler, current, prev, SAMPLE_FREQ)
        print(f"实时样本 {desc}：当前浓度 {current}ppm → 报警等级：{name}")


if __name__ == "__main__":
    main()