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

# 全局设置：忽略无关警告（提升代码运行整洁性）
warnings.filterwarnings("ignore")


# ===================== 1. 烟雾传感器数据生成（模拟无人车场景） =====================
def generate_smoke_sensor_data(
        sample_num: int = 1000,  # 采样点数（对应时间序列）
        sample_freq: int = 1,  # 采样频率（Hz），每秒采样次数
        noise_level: float = 0.05,  # 传感器噪声幅度（ppm）
        add_abnormal: bool = True  # 是否加入烟雾异常值（模拟故障/瞬时浓烟）
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成无人车烟雾传感器模拟数据（时间序列），模拟真实车载场景的烟雾浓度变化

    核心业务规则：
    --------
    - 烟雾浓度单位：ppm（百万分之一）
    - 标签定义：0=无报警（0-5ppm），1=低风险（5-20ppm），2=高风险（>20ppm）
    - 数据分布：正常样本80%、低风险15%、高风险5%，额外注入突发异常值

    参数说明：
    --------
    sample_num : int
        总采样点数，默认1000个
    sample_freq : int
        采样频率（Hz），默认1Hz（每秒1个样本）
    noise_level : float
        传感器随机噪声幅度（ppm），默认±0.05ppm
    add_abnormal : bool
        是否添加突发异常值（模拟传感器故障/瞬时浓烟），默认True

    返回值：
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        time_steps: 时间戳数组（秒），形状=(sample_num,)
        smoke_conc: 烟雾浓度数组（ppm），形状=(sample_num,)
        labels: 报警标签数组（0/1/2），形状=(sample_num,)
    """
    # 生成时间戳序列：从0开始，步长=1/采样频率，总长度=采样点数/采样频率
    time_steps = np.arange(0, sample_num / sample_freq, 1 / sample_freq)

    # 初始化烟雾浓度数组（浮点型，保证精度）
    smoke_conc = np.zeros(sample_num, dtype=np.float32)

    # ========== 1. 生成基础浓度分布 ==========
    # 正常场景：0-5ppm随机波动（无报警）
    base_conc = np.random.uniform(0, 5, sample_num)

    # ========== 2. 注入风险样本 ==========
    # 低风险段（5-20ppm）：占总样本15%，模拟轻微烟雾泄漏
    low_risk_idx = np.random.choice(sample_num, int(sample_num * 0.15), replace=False)
    smoke_conc[low_risk_idx] = np.random.uniform(5, 20, len(low_risk_idx))

    # 高风险段（>20ppm）：占总样本5%，模拟严重烟雾泄漏
    high_risk_idx = np.random.choice(sample_num, int(sample_num * 0.05), replace=False)
    smoke_conc[high_risk_idx] = np.random.uniform(20, 50, len(high_risk_idx))

    # 正常段赋值（剩余80%样本）
    normal_idx = np.setdiff1d(np.arange(sample_num), np.union1d(low_risk_idx, high_risk_idx))
    smoke_conc[normal_idx] = base_conc[normal_idx]

    # ========== 3. 加入传感器噪声与物理约束 ==========
    # 加入高斯噪声（模拟真实传感器采样误差）
    smoke_conc += np.random.normal(0, noise_level, sample_num)
    # 浓度非负约束（物理意义：浓度不能为负）
    smoke_conc = np.maximum(smoke_conc, 0)

    # ========== 4. 注入突发异常值（模拟故障） ==========
    if add_abnormal:
        # 随机生成3-8个异常点，浓度60-100ppm（远超正常范围）
        abnormal_idx = np.random.choice(sample_num, np.random.randint(3, 8), replace=False)
        smoke_conc[abnormal_idx] = np.random.uniform(60, 100, len(abnormal_idx))

    # ========== 5. 生成报警标签（基于浓度阈值） ==========
    labels = np.zeros(sample_num, dtype=np.int32)  # 初始化标签为无报警
    labels[(smoke_conc >= 5) & (smoke_conc < 20)] = 1  # 低风险标签
    labels[smoke_conc >= 20] = 2  # 高风险标签

    # 浓度值保留2位小数（符合传感器实际输出精度）
    smoke_conc = np.round(smoke_conc, 2)

    return time_steps, smoke_conc, labels


# ===================== 2. 数据预处理（时序特征工程） =====================
def preprocess_smoke_data(smoke_conc: np.ndarray, labels: np.ndarray, sample_freq: int = 1) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, np.ndarray]:
    """
    烟雾数据预处理：构建时序特征 + 标准化 + 划分训练/测试集
    核心特征工程：结合时间序列特性，提升模型对浓度变化的感知能力

    特征说明：
    --------
    - 特征1：当前浓度（ppm）
    - 特征2：前1帧浓度（ppm）（首帧补0）
    - 特征3：浓度变化率（ppm/s）= 浓度差分 × 采样频率

    参数说明：
    --------
    smoke_conc : np.ndarray
        烟雾浓度数组（ppm），形状=(n_samples,)
    labels : np.ndarray
        报警标签数组，形状=(n_samples,)
    sample_freq : int
        采样频率（Hz），用于计算浓度变化率，默认1Hz

    返回值：
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, np.ndarray]
        X_train_scaled: 标准化训练特征，形状=(n_train, 3)
        X_test_scaled: 标准化测试特征，形状=(n_test, 3)
        y_train: 训练标签，形状=(n_train,)
        y_test: 测试标签，形状=(n_test,)
        scaler: 拟合后的标准化器（用于实时监测）
        features: 全量特征矩阵（未标准化），形状=(n_samples, 3)
    """
    # ========== 1. 构建时序特征 ==========
    # 前1帧浓度（首帧补0，保证特征长度一致）
    prev_conc = np.concatenate([[0], smoke_conc[:-1]])
    # 浓度变化率（ppm/s）：差分 × 采样频率（转换为每秒变化量）
    conc_diff = np.diff(smoke_conc, prepend=0) * sample_freq

    # 组合特征矩阵（3维特征）
    features = np.column_stack([smoke_conc, prev_conc, conc_diff])

    # ========== 2. 划分训练/测试集 ==========
    # 分层抽样（stratify=labels）：保证训练/测试集标签分布一致
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # ========== 3. 特征标准化 ==========
    # 初始化标准化器（均值0，标准差1）
    scaler = StandardScaler()
    # 训练集拟合+转换，测试集仅转换（避免数据泄露）
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features


# ===================== 3. 烟雾异常检测（报警触发） =====================
def detect_smoke_abnormal(smoke_conc: np.ndarray, contamination: float = 0.08) -> np.ndarray:
    """
    基于孤立森林（Isolation Forest）检测烟雾浓度异常值（无监督异常检测）
    核心原理：异常值在特征空间中更容易被孤立，适用于传感器故障/瞬时浓烟检测

    参数说明：
    --------
    smoke_conc : np.ndarray
        烟雾浓度数组（ppm），形状=(n_samples,)
    contamination : float
        异常值占比（0-1），默认0.08（8%）

    返回值：
    --------
    np.ndarray
        abnormal_label: 异常标记数组，1=正常，-1=异常/报警，形状=(n_samples,)
    """
    # 重塑为2D数组（适配sklearn模型输入要求：[n_samples, n_features]）
    conc_2d = smoke_conc.reshape(-1, 1)

    # 初始化孤立森林模型
    iso_forest = IsolationForest(
        n_estimators=100,  # 决策树数量，默认100
        contamination=contamination,  # 异常值比例
        random_state=42  # 随机种子（保证结果可复现）
    )
    # 训练模型并预测异常值
    abnormal_label = iso_forest.fit_predict(conc_2d)

    return abnormal_label


# ===================== 4. 报警等级分类模型训练 =====================
def train_smoke_alarm_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = "random_forest"
) -> object:
    """
    训练烟雾报警等级分类模型（监督学习），用于预测烟雾风险等级

    参数说明：
    --------
    X_train : np.ndarray
        标准化训练特征，形状=(n_train, 3)
    y_train : np.ndarray
        训练标签，形状=(n_train,)
    model_type : str
        模型类型，仅支持"random_forest"，默认随机森林

    返回值：
    --------
    object
        训练好的分类模型
    """
    # 模型选择（当前仅支持随机森林）
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,  # 决策树数量
            max_depth=6,  # 树最大深度（防止过拟合）
            random_state=42  # 随机种子
        )
    else:
        raise ValueError("仅支持 random_forest 模型")

    # 训练模型
    model.fit(X_train, y_train)
    return model


# ===================== 5. 模型评估 =====================
def evaluate_alarm_model(
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray
) -> Dict[str, float]:
    """
    评估报警等级分类模型性能，输出分类报告+混淆矩阵可视化

    参数说明：
    --------
    model : object
        训练好的分类模型
    X_test : np.ndarray
        标准化测试特征，形状=(n_test, 3)
    y_test : np.ndarray
        测试标签，形状=(n_test,)

    返回值：
    --------
    Dict[str, float]
        评估指标字典，包含准确率（accuracy）
    """
    # 预测测试集
    y_pred = model.predict(X_test)

    # 计算整体准确率
    accuracy = accuracy_score(y_test, y_pred)

    # 输出详细分类报告（精确率/召回率/F1分数）
    print("\n===== 报警等级分类报告 =====")
    print(classification_report(
        y_test, y_pred,
        target_names=["无报警", "低风险", "高风险"]  # 标签名称映射
    ))

    # ========== 混淆矩阵可视化 ==========
    # 设置中文字体（避免乱码）
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 6))
    # 绘制混淆矩阵热力图
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    # 添加颜色条
    ax.figure.colorbar(im, ax=ax)

    # 设置坐标轴标签
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["无报警", "低风险", "高风险"],
        yticklabels=["无报警", "低风险", "高风险"],
        title="烟雾报警等级混淆矩阵",
        ylabel="真实等级",
        xlabel="预测等级"
    )

    # 在单元格中标注数值
    thresh = cm.max() / 2.  # 颜色阈值（区分文字颜色）
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # 调整布局并显示
    fig.tight_layout()
    plt.show()

    # 返回评估指标
    return {"accuracy": accuracy}


# ===================== 6. 实时监测可视化 =====================
def visualize_smoke_alarm(
        time_steps: np.ndarray,
        smoke_conc: np.ndarray,
        abnormal_label: np.ndarray,
        alarm_level: np.ndarray
):
    """
    可视化烟雾监测全流程结果：
    1. 浓度时序曲线 + 异常报警标记 + 风险阈值线
    2. 报警等级分类散点图（按等级着色）

    参数说明：
    --------
    time_steps : np.ndarray
        时间戳数组（秒），形状=(n_samples,)
    smoke_conc : np.ndarray
        烟雾浓度数组（ppm），形状=(n_samples,)
    abnormal_label : np.ndarray
        异常检测标记数组（1/-1），形状=(n_samples,)
    alarm_level : np.ndarray
        报警等级预测数组（0/1/2），形状=(n_samples,)
    """
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建2行1列子图（总画布大小12×10）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # ========== 子图1：烟雾浓度曲线 + 异常报警标记 ==========
    # 绘制浓度时序曲线
    ax1.plot(time_steps, smoke_conc, color="#1f77b4", linewidth=1.5, label="烟雾浓度")
    # 标注异常报警点（红色散点，zorder=5确保在顶层）
    alarm_idx = np.where(abnormal_label == -1)[0]
    ax1.scatter(
        time_steps[alarm_idx], smoke_conc[alarm_idx],
        color="#d62728", s=80, label="报警触发点", zorder=5
    )
    # 绘制风险阈值线（虚线）
    ax1.axhline(y=5, color="#ff7f0e", linestyle="--", label="低风险阈值（5ppm）")
    ax1.axhline(y=20, color="#d62728", linestyle="--", label="高风险阈值（20ppm）")

    # 子图1样式设置
    ax1.set_xlabel("时间 (秒)")
    ax1.set_ylabel("烟雾浓度 (ppm)")
    ax1.set_title("无人车烟雾浓度监测曲线 + 报警标记")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)  # 网格透明度0.3

    # ========== 子图2：报警等级分类结果 ==========
    # 等级颜色映射：0=绿色（无报警），1=黄色（低风险），2=红色（高风险）
    colors = np.array(["#2ca02c", "#ff7f0e", "#d62728"])[alarm_level]
    # 绘制等级散点图
    ax2.scatter(
        time_steps, smoke_conc,
        c=colors, s=20, alpha=0.7, label="报警等级"
    )
    # 手动添加图例（解决动态着色无法自动生成图例的问题）
    ax2.scatter([], [], color="#2ca02c", s=50, label="无报警")
    ax2.scatter([], [], color="#ff7f0e", s=50, label="低风险")
    ax2.scatter([], [], color="#d62728", s=50, label="高风险")

    # 子图2样式设置
    ax2.set_xlabel("时间 (秒)")
    ax2.set_ylabel("烟雾浓度 (ppm)")
    ax2.set_title("无人车烟雾报警等级分类")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    # 调整子图间距并显示
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
    车载实时烟雾监测：单样本预测报警等级（适配嵌入式/实时系统）
    核心流程：构建时序特征 → 标准化 → 模型预测 → 等级映射

    参数说明：
    --------
    model : object
        训练好的分类模型
    scaler : object
        拟合后的标准化器
    current_conc : float
        当前帧烟雾浓度（ppm）
    prev_conc : float
        上一帧烟雾浓度（ppm）
    sample_freq : int
        采样频率（Hz），用于计算浓度变化率，默认1Hz

    返回值：
    --------
    Tuple[int, str]
        level_label: 报警等级标签（0/1/2）
        level_name: 报警等级名称（无报警/低风险/高风险）
    """
    # 计算浓度变化率（ppm/s）
    conc_diff = (current_conc - prev_conc) * sample_freq

    # 构建3维特征（适配模型输入）
    features = np.array([[current_conc, prev_conc, conc_diff]])
    # 标准化（使用训练好的scaler，避免数据泄露）
    features_scaled = scaler.transform(features)
    # 预测等级标签
    level_label = model.predict(features_scaled)[0]

    # 等级名称映射（便于业务理解）
    level_map = {0: "无报警", 1: "低风险", 2: "高风险"}
    level_name = level_map[level_label]

    return level_label, level_name


# ===================== 主函数（流程整合） =====================
def main():
    """
    程序主入口：整合数据生成→异常检测→模型训练→评估→可视化→实时监测全流程
    模拟无人车烟雾监测系统的完整运行逻辑
    """
    # ========== 全局参数配置 ==========
    SAMPLE_NUM = 1000  # 总采样点数
    SAMPLE_FREQ = 1  # 采样频率（1Hz）
    NOISE_LEVEL = 0.05  # 传感器噪声幅度

    # ========== 1. 生成烟雾传感器数据 ==========
    print("===== 1. 生成无人车烟雾传感器数据 =====")
    time_steps, smoke_conc, labels = generate_smoke_sensor_data(
        sample_num=SAMPLE_NUM,
        sample_freq=SAMPLE_FREQ,
        noise_level=NOISE_LEVEL
    )
    # 输出数据基本信息
    print(f"生成采样点数：{len(smoke_conc)} 个")
    print(f"时间范围：0 ~ {time_steps[-1]:.1f} 秒")
    print(f"浓度范围：{smoke_conc.min():.2f} ~ {smoke_conc.max():.2f} ppm")

    # ========== 2. 烟雾异常检测（报警触发） ==========
    print("\n===== 2. 烟雾异常检测（报警触发） =====")
    abnormal_label = detect_smoke_abnormal(smoke_conc)
    alarm_count = len(np.where(abnormal_label == -1)[0])
    print(f"检测到报警触发点数量：{alarm_count} 个")
    # 输出前5个报警时间戳（示例）
    alarm_time = time_steps[abnormal_label == -1][:5]
    print(f"前5个报警时间戳：{alarm_time.round(2)} 秒")

    # ========== 3. 数据预处理 ==========
    print("\n===== 3. 数据预处理 =====")
    X_train, X_test, y_train, y_test, scaler, features = preprocess_smoke_data(
        smoke_conc, labels, sample_freq=SAMPLE_FREQ
    )
    print(f"训练集数量：{len(X_train)} 条，测试集数量：{len(X_test)} 条")

    # ========== 4. 训练报警等级分类模型 ==========
    print("\n===== 4. 训练烟雾报警等级模型 =====")
    alarm_model = train_smoke_alarm_model(X_train, y_train)

    # ========== 5. 模型评估 ==========
    print("\n===== 5. 模型评估 =====")
    eval_metrics = evaluate_alarm_model(alarm_model, X_test, y_test)
    print(f"报警等级分类准确率：{eval_metrics['accuracy']:.2%}")

    # ========== 6. 全量数据等级预测 ==========
    print("\n===== 6. 全量数据报警等级预测 =====")
    # 全量特征标准化
    features_scaled = scaler.transform(features)
    # 预测全量数据等级
    alarm_level = alarm_model.predict(features_scaled)
    # 统计各等级样本数
    level_count = {0: 0, 1: 0, 2: 0}
    for level in alarm_level:
        level_count[level] += 1
    print(f"无报警样本数：{level_count[0]} | 低风险样本数：{level_count[1]} | 高风险样本数：{level_count[2]}")

    # ========== 7. 可视化监测结果 ==========
    print("\n===== 7. 可视化烟雾监测结果 =====")
    visualize_smoke_alarm(time_steps, smoke_conc, abnormal_label, alarm_level)

    # ========== 8. 实时监测示例（模拟车载场景） ==========
    print("\n===== 8. 实时烟雾监测示例 =====")
    # 示例1：正常浓度（3ppm）
    level1, name1 = realtime_smoke_monitor(alarm_model, scaler, 3.0, 2.8, SAMPLE_FREQ)
    print(f"实时样本1：当前浓度3.0ppm → 报警等级：{name1}")

    # 示例2：低风险（10ppm）11111
    level2, name2 = realtime_smoke_monitor(alarm_model, scaler, 10.0, 8.5, SAMPLE_FREQ)
    print(f"实时样本2：当前浓度10.0ppm → 报警等级：{name2}")

    # 示例3：高风险（25ppm）
    level3, name3 = realtime_smoke_monitor(alarm_model, scaler, 25.0, 20.0, SAMPLE_FREQ)
    print(f"实时样本3：当前浓度25.0ppm → 报警等级：{name3}")


# 程序执行入口
if __name__ == "__main__":
    main()