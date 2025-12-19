import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import warnings

warnings.filterwarnings("ignore")


# ===================== 1. 烟雾传感器数据生成（模拟无人车场景） =====================
def generate_smoke_sensor_data(
        sample_num: int = 1000,  # 采样点数（对应时间序列）
        sample_freq: int = 1,  # 采样频率（Hz）
        noise_level: float = 0.05,  # 传感器噪声（ppm）
        add_abnormal: bool = True  # 是否加入烟雾异常值
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成无人车烟雾传感器数据（时间戳+烟雾浓度+报警标签）
    烟雾浓度单位：ppm（百万分之一）
    标签定义：0=无报警（0-5ppm），1=低风险（5-20ppm），2=高风险（>20ppm）
    """
    # 生成时间戳（模拟连续采样）
    time_steps = np.arange(0, sample_num / sample_freq, 1 / sample_freq)

    # 初始化烟雾浓度数组（默认正常浓度）
    smoke_conc = np.zeros(sample_num, dtype=np.float32)

    # 生成基础浓度（正常场景：0-5ppm随机波动）
    base_conc = np.random.uniform(0, 5, sample_num)

    # 随机插入低风险/高风险烟雾段（模拟局部烟雾）
    # 低风险段（5-20ppm）：占比15%
    low_risk_idx = np.random.choice(sample_num, int(sample_num * 0.15), replace=False)
    smoke_conc[low_risk_idx] = np.random.uniform(5, 20, len(low_risk_idx))

    # 高风险段（>20ppm）：占比5%
    high_risk_idx = np.random.choice(sample_num, int(sample_num * 0.05), replace=False)
    smoke_conc[high_risk_idx] = np.random.uniform(20, 50, len(high_risk_idx))

    # 正常段赋值
    normal_idx = np.setdiff1d(np.arange(sample_num), np.union1d(low_risk_idx, high_risk_idx))
    smoke_conc[normal_idx] = base_conc[normal_idx]

    # 加入传感器噪声（模拟真实采样误差）
    smoke_conc += np.random.normal(0, noise_level, sample_num)
    smoke_conc = np.maximum(smoke_conc, 0)  # 浓度非负

    # 加入突发异常值（模拟传感器故障/瞬时浓烟）
    if add_abnormal:
        abnormal_idx = np.random.choice(sample_num, np.random.randint(3, 8), replace=False)
        smoke_conc[abnormal_idx] = np.random.uniform(60, 100, len(abnormal_idx))

    # 生成报警标签（基于浓度阈值）
    labels = np.zeros(sample_num, dtype=np.int32)
    labels[(smoke_conc >= 5) & (smoke_conc < 20)] = 1  # 低风险
    labels[smoke_conc >= 20] = 2  # 高风险

    # 保留2位小数
    smoke_conc = np.round(smoke_conc, 2)

    return time_steps, smoke_conc, labels


# ===================== 2. 数据预处理（修复sample_freq未定义问题） =====================
def preprocess_smoke_data(smoke_conc: np.ndarray, labels: np.ndarray, sample_freq: int = 1) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, np.ndarray]:
    """
    数据预处理：构建时序特征 + 标准化 + 划分训练/测试集
    特征：当前浓度 + 前1帧浓度 + 浓度变化率（差分）
    :param sample_freq: 采样频率（Hz），用于计算浓度变化率
    """
    # 构建时序特征（适配时间序列监测）
    # 前1帧浓度（首帧补0）
    prev_conc = np.concatenate([[0], smoke_conc[:-1]])
    # 浓度变化率（ppm/s）：修复sample_freq参数传递
    conc_diff = np.diff(smoke_conc, prepend=0) * sample_freq  # 乘以采样频率转速率

    # 特征矩阵（3维特征）
    features = np.column_stack([smoke_conc, prev_conc, conc_diff])

    # 划分训练/测试集（8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features


# ===================== 3. 烟雾异常检测（报警触发） =====================
def detect_smoke_abnormal(smoke_conc: np.ndarray, contamination: float = 0.08) -> np.ndarray:
    """
    基于孤立森林检测烟雾异常值（触发报警）
    :return: 异常标记数组（1=正常，-1=异常/报警）
    """
    # 重塑为2D数组（适配sklearn输入）
    conc_2d = smoke_conc.reshape(-1, 1)

    # 训练孤立森林模型（无监督异常检测）
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,  # 异常值占比（8%）
        random_state=42
    )
    abnormal_label = iso_forest.fit_predict(conc_2d)

    return abnormal_label


# ===================== 4. 报警等级分类模型训练 =====================
def train_smoke_alarm_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = "random_forest"
) -> object:
    """
    训练烟雾报警等级分类模型
    """
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
    else:
        raise ValueError("仅支持 random_forest 模型")

    model.fit(X_train, y_train)
    return model


# ===================== 5. 模型评估 =====================
def evaluate_alarm_model(
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray
) -> Dict[str, float]:
    """
    评估报警等级分类lll模型性能
    """
    # 预测测试集
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)

    # 输出分类报告
    print("\n===== 报警等级分类报告 =====")
    print(classification_report(
        y_test, y_pred,
        target_names=["无报警", "低风险", "高风险"]
    ))

    # 绘制混淆矩阵
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    ax.figure.colorbar(im, ax=ax)

    # 设置标签111111
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["无报警", "低风险", "高风险"],
        yticklabels=["无报警", "低风险", "高风险"],
        title="烟雾报警等级混淆矩阵",
        ylabel="真实等级",
        xlabel="预测等级"
    )

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


# ===================== 6. 实时监测可视化 =====================
def visualize_smoke_alarm(
        time_steps: np.ndarray,
        smoke_conc: np.ndarray,
        abnormal_label: np.ndarray,
        alarm_level: np.ndarray
):
    """
    可视化烟雾监测结果：浓度曲线 + 异常报警标记 + 等级分类
    """
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建2行1列子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1：烟雾浓度曲线 + 异常报警标记
    ax1.plot(time_steps, smoke_conc, color="#1f77b4", linewidth=1.5, label="烟雾浓度")
    # 标注异常报警点（红色散点）
    alarm_idx = np.where(abnormal_label == -1)[0]
    ax1.scatter(
        time_steps[alarm_idx], smoke_conc[alarm_idx],
        color="#d62728", s=80, label="报警触发点", zorder=5
    )
    # 绘制阈值线
    ax1.axhline(y=5, color="#ff7f0e", linestyle="--", label="低风险阈值（5ppm）")
    ax1.axhline(y=20, color="#d62728", linestyle="--", label="高风险阈值（20ppm）")

    ax1.set_xlabel("时间 (秒)")
    ax1.set_ylabel("烟雾浓度 (ppm)")
    ax1.set_title("无人车烟雾浓度监测曲线 + 报警标记")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # 子图2：报警等级分类结果
    # 等级颜色映射：0=绿色（无报警），1=黄色（低风险），2=红色（高风险）
    colors = np.array(["#2ca02c", "#ff7f0e", "#d62728"])[alarm_level]
    ax2.scatter(
        time_steps, smoke_conc,
        c=colors, s=20, alpha=0.7, label="报警等级"
    )
    # 手动添加图例
    ax2.scatter([], [], color="#2ca02c", s=50, label="无报警")
    ax2.scatter([], [], color="#ff7f0e", s=50, label="低风险")
    ax2.scatter([], [], color="#d62728", s=50, label="高风险")

    ax2.set_xlabel("时间 (秒)")
    ax2.set_ylabel("烟雾浓度 (ppm)")
    ax2.set_title("无人车烟雾报警等级分类")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ===================== 7. 单样本实时监测（车载场景） =====================
def realtime_smoke_monitor(
        model: object,
        scaler: object,
        current_conc: float,
        prev_conc: float,
        sample_freq: int = 1
) -> Tuple[int, str]:
    """
    实时监测单样本烟雾浓度，返回报警等级
    :param current_conc: 当前浓度（ppm）
    :param prev_conc: 上一帧浓度（ppm）
    :return: 等级标签、等级名称
    """
    # 计算浓度变化率
    conc_diff = (current_conc - prev_conc) * sample_freq

    # 构建特征
    features = np.array([[current_conc, prev_conc, conc_diff]])
    # 标准化
    features_scaled = scaler.transform(features)
    # 预测等级
    level_label = model.predict(features_scaled)[0]

    # 等级映射
    level_map = {0: "无报警", 1: "低风险", 2: "高风险"}
    level_name = level_map[level_label]

    return level_label, level_name


# ===================== 主函数（修复参数传递） =====================
def main():
    # 全局参数
    SAMPLE_NUM = 1000  # 采样点数
    SAMPLE_FREQ = 1  # 采样频率（1Hz）
    NOISE_LEVEL = 0.05  # 传感器噪声

    # 1. 生成烟雾传感器数据
    print("===== 1. 生成无人车烟雾传感器数据 =====")
    time_steps, smoke_conc, labels = generate_smoke_sensor_data(
        sample_num=SAMPLE_NUM,
        sample_freq=SAMPLE_FREQ,
        noise_level=NOISE_LEVEL
    )
    print(f"生成采样点数：{len(smoke_conc)} 个")
    print(f"时间范围：0 ~ {time_steps[-1]:.1f} 秒")
    print(f"浓度范围：{smoke_conc.min():.2f} ~ {smoke_conc.max():.2f} ppm")

    # 2. 烟雾异常检测（触发报警）
    print("\n===== 2. 烟雾异常检测（报警触发） =====")
    abnormal_label = detect_smoke_abnormal(smoke_conc)
    alarm_count = len(np.where(abnormal_label == -1)[0])
    print(f"检测到报警触发点数量：{alarm_count} 个")
    alarm_time = time_steps[abnormal_label == -1][:5]  # 打印前5个报警时间
    print(f"前5个报警时间戳：{alarm_time.round(2)} 秒")

    # 3. 数据预处理（修复：传递sample_freq参数）
    print("\n===== 3. 数据预处理 =====")
    X_train, X_test, y_train, y_test, scaler, features = preprocess_smoke_data(
        smoke_conc, labels, sample_freq=SAMPLE_FREQ  # 补充sample_freq参数
    )
    print(f"训练集数量：{len(X_train)} 条，测试集数量：{len(X_test)} 条")

    # 4. 训练报警等级分类模型
    print("\n===== 4. 训练烟雾报警等级模型 =====")
    alarm_model = train_smoke_alarm_model(X_train, y_train)

    # 5. 评估模型
    print("\n===== 5. 模型评估 =====")
    eval_metrics = evaluate_alarm_model(alarm_model, X_test, y_test)
    print(f"报警等级分类准确率：{eval_metrics['accuracy']:.2%}")

    # 6. 全量数据等级预测
    print("\n===== 6. 全量数据报警等级预测 =====")
    features_scaled = scaler.transform(features)
    alarm_level = alarm_model.predict(features_scaled)
    level_count = {0: 0, 1: 0, 2: 0}
    for level in alarm_level:
        level_count[level] += 1
    print(f"无报警样本数：{level_count[0]} | 低风险样本数：{level_count[1]} | 高风险样本数：{level_count[2]}")

    # 7. 可视化监测结果  222
    print("\n===== 7. 可视化烟雾监测结果 =====")
    visualize_smoke_alarm(time_steps, smoke_conc, abnormal_label, alarm_level)

    # 8. 实时监测示例（模拟车载传感器）
    print("\n===== 8. 实时烟雾监测示例 =====")
    # 示例1：正常浓度（3ppm）
    level1, name1 = realtime_smoke_monitor(alarm_model, scaler, 3.0, 2.8, SAMPLE_FREQ)
    print(f"实时样本1：当前浓度3.0ppm → 报警等级：{name1}")

    # 示例2：低风险（10ppm）
    level2, name2 = realtime_smoke_monitor(alarm_model, scaler, 10.0, 8.5, SAMPLE_FREQ)
    print(f"实时样本2：当前浓度10.0ppm → 报警等级：{name2}")

    # 示例3：高风险（25ppm）
    level3, name3 = realtime_smoke_monitor(alarm_model, scaler, 25.0, 20.0, SAMPLE_FREQ)
    print(f"实时样本3：当前浓度25.0ppm → 报警等级：{name3}")


if __name__ == "__main__":
    main()