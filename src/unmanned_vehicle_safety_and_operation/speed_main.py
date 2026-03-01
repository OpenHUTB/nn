# 导入必要的库
import numpy as np  # 数值计算基础库
import matplotlib.pyplot as plt  # 数据可视化库
import random  # 随机数生成库
from typing import Tuple, Dict, List  # 类型注解库
# 机器学习相关库
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.ensemble import IsolationForest, RandomForestClassifier  # 孤立森林（异常检测）、随机森林（分类）
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report  # 评估指标
import warnings  # 警告处理库

# 全局设置：忽略无关警告（提升代码运行整洁性）
warnings.filterwarnings("ignore")


# ===================== 基础数据生成模块 =====================
def generate_velocity_data(
        test_duration: int = 60,
        sample_freq: int = 1,
        max_velocity: float = 30.0,
        noise_level: float = 0.5,
        add_abnormal: bool = True  # 是否加入异常值（模拟故障）
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成模拟无人车速度时序数据，模拟真实驾驶的加速-匀速-减速过程，并可添加异常值

    参数说明：
    --------
    test_duration : int
        测试总时长（秒），默认60秒
    sample_freq : int
        采样频率（次/秒），默认1次/秒
    max_velocity : float
        最大行驶速度（km/h），默认30km/h
    noise_level : float
        速度噪声幅度（模拟传感器误差），默认±0.5km/h
    add_abnormal : bool
        是否添加异常值（模拟传感器故障/急刹等异常），默认True

    返回值：
    --------
    Tuple[np.ndarray, np.ndarray]
        time_steps: 时间戳数组（秒）
        velocity: 对应时间的速度数组（km/h）
    """
    # 生成时间戳序列：从0到test_duration，步长为1/sample_freq
    time_steps = np.arange(0, test_duration, 1 / sample_freq)
    num_samples = len(time_steps)  # 总采样点数
    velocity = np.zeros(num_samples)  # 初始化速度数组

    # 模拟典型驾驶过程：0-10秒加速 | 10-40秒匀速 | 40-60秒减速
    for i, t in enumerate(time_steps):
        if t < 10:
            # 加速阶段：线性增长到最大速度
            v = (max_velocity / 10) * t
        elif 10 <= t < 40:
            # 匀速阶段：保持最大速度
            v = max_velocity
        else:
            # 减速阶段：线性减速至0
            v = max_velocity - (max_velocity / 20) * (t - 40)

        # 添加随机噪声（模拟传感器测量误差）
        v += random.uniform(-noise_level, noise_level)
        # 确保速度非负（物理意义：速度不能为负）
        velocity[i] = max(v, 0.0)

    # 随机添加2-3个异常值（模拟传感器故障/急刹/数据跳变）
    if add_abnormal:
        # 随机选择2-3个异常点位置
        abnormal_idx = random.sample(range(num_samples), random.randint(2, 3))
        for idx in abnormal_idx:
            # 异常值类型1：速度飙升（2-3倍当前值）
            velocity[idx] = velocity[idx] * random.uniform(2, 3)
            # 50%概率触发异常值类型2：速度归零（模拟急刹/传感器掉线）
            if random.random() > 0.5:
                velocity[idx] = 0

    return time_steps, velocity


# ===================== 机器学习：速度预测模块 =====================
def velocity_prediction(
        time_steps: np.ndarray,
        velocity: np.ndarray,
        predict_ratio: float = 0.2  # 预测未来20%的数据
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    基于线性回归模型对无人车未来速度进行预测

    参数说明：
    --------
    time_steps : np.ndarray
        历史时间戳数组
    velocity : np.ndarray
        历史速度数组
    predict_ratio : float
        预测未来数据占总数据的比例，默认0.2（预测未来20%）

    返回值：
    --------
    Tuple[np.ndarray, np.ndarray, float]
        predict_time: 预测时间段的时间戳数组
        predict_vel: 预测的速度值数组
        mae: 平均绝对误差（衡量预测精度）
    """
    # 数据预处理：划分训练集和预测集
    total_len = len(time_steps)
    train_len = int(total_len * (1 - predict_ratio))  # 训练集长度（80%数据）
    # 训练集特征：时间戳（需reshape为2D数组，满足sklearn输入要求）
    train_time = time_steps[:train_len].reshape(-1, 1)
    train_vel = velocity[:train_len]  # 训练集标签：对应时间的速度
    predict_time = time_steps[train_len:].reshape(-1, 1)  # 预测时间段

    # 初始化并训练线性回归模型
    model = LinearRegression()
    model.fit(train_time, train_vel)  # 拟合时间-速度关系

    # 预测未来速度
    predict_vel = model.predict(predict_time)

    # 计算预测误差（仅当有真实值可对比时）
    if len(predict_vel) == len(velocity[train_len:]):
        mae = mean_absolute_error(velocity[train_len:], predict_vel)
    else:
        mae = 0.0  # 无真实值时误差为0

    # 展平时间数组（便于后续可视化）
    return predict_time.flatten(), predict_vel, mae


# ===================== 机器学习：异常检测模块 =====================
def detect_abnormal_velocity(
        velocity: np.ndarray,
        contamination: float = 0.05  # 异常值占比（默认5%）
) -> np.ndarray:
    """
    基于孤立森林（Isolation Forest）算法检测速度序列中的异常值
    孤立森林原理：通过随机划分特征空间，孤立异常点（异常点更容易被孤立）

    参数说明：
    --------
    velocity : np.ndarray
        速度序列数组
    contamination : float
        预设异常值比例（0-1），默认0.05（5%）

    返回值：
    --------
    np.ndarray
        abnormal_label: 异常标记数组（1=正常样本，-1=异常样本）
    """
    # 数据预处理：重塑为2D数组（sklearn模型输入要求）
    vel_2d = velocity.reshape(-1, 1)

    # 初始化孤立森林模型
    model = IsolationForest(
        n_estimators=100,  # 决策树数量，默认100
        contamination=contamination,  # 异常值比例
        random_state=42  # 随机种子（保证结果可复现）
    )
    # 训练模型并预测异常值
    abnormal_label = model.fit_predict(vel_2d)  # 1=正常，-1=异常

    return abnormal_label


# ===================== 机器学习：驾驶模式分类模块 =====================
def classify_driving_mode(
        time_steps: np.ndarray,
        velocity: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    基于随机森林分类器识别驾驶模式（加速/匀速/减速）
    核心特征：加速度（速度变化率）+ 当前速度

    参数说明：
    --------
    time_steps : np.ndarray
        时间戳数组
    velocity : np.ndarray
        速度数组

    返回值：
    --------
    Tuple[np.ndarray, float]
        full_pred: 全量数据的驾驶模式标签（0=减速，1=匀速，2=加速）
        accuracy: 分类准确率（测试集）
    """
    # 特征工程：计算加速度（核心特征）
    vel_diff = np.diff(velocity)  # 速度差分（后-前）
    time_diff = np.diff(time_steps)  # 时间差分
    acceleration = vel_diff / time_diff  # 加速度 = 速度变化/时间变化

    # 构建标签：根据加速度阈值分类驾驶模式
    labels = []
    for acc in acceleration:
        if acc > 0.5:
            labels.append(2)  # 加速模式：加速度>0.5 km/h/s
        elif acc > -0.5:
            labels.append(1)  # 匀速模式：-0.5<加速度<0.5 km/h/s
        else:
            labels.append(0)  # 减速模式：加速度<-0.5 km/h/s
    labels = np.array(labels)

    # 构建特征集：加速度 + 当前速度（需对齐长度，差分后长度减1）
    features = np.column_stack([
        acceleration,
        velocity[:-1]  # 速度数组去掉最后一个元素，与加速度长度匹配
    ])

    # 划分训练集（70%）和测试集（30%）
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42
    )

    # 初始化并训练随机森林分类器
    model = RandomForestClassifier(
        n_estimators=100,  # 决策树数量
        random_state=42  # 随机种子
    )
    model.fit(X_train, y_train)

    # 预测测试集并计算准确率
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 预测全量数据的驾驶模式（补全最后一个点的标签）
    full_pred = model.predict(features)
    full_pred = np.append(full_pred, full_pred[-1])  # 补全长度，保持与原始数据一致

    return full_pred, accuracy


# ===================== 可视化整合模块 =====================
def plot_ml_results(
        time_steps: np.ndarray,
        velocity: np.ndarray,
        predict_time: np.ndarray,
        predict_vel: np.ndarray,
        abnormal_label: np.ndarray,
        mode_labels: np.ndarray
):
    """
    可视化所有机器学习分析结果：速度预测+异常检测+驾驶模式分类

    参数说明：
    --------
    time_steps : np.ndarray
        原始时间戳数组
    velocity : np.ndarray
        原始速度数组
    predict_time : np.ndarray
        预测时间段的时间戳
    predict_vel : np.ndarray
        预测的速度值
    abnormal_label : np.ndarray
        异常检测标记数组
    mode_labels : np.ndarray
        驾驶模式分类标签数组
    """
    # 设置中文字体（避免中文乱码）
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建2行1列的子图布局，设置画布大小
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # -------- 子图1：速度预测 + 异常检测可视化 --------
    # 绘制原始速度曲线11
    ax1.plot(time_steps, velocity, color="#2E86AB", linewidth=2, label="原始速度")
    # 绘制预测速度曲线（虚线）
    ax1.plot(predict_time, predict_vel, color="#F18F01", linestyle="--", linewidth=2, label="预测速度")
    # 标注异常点（红色高亮）
    abnormal_idx = np.where(abnormal_label == -1)[0]
    ax1.scatter(time_steps[abnormal_idx], velocity[abnormal_idx],
                color="#E63946", s=100, label="异常速度点", zorder=5)  # zorder确保异常点在顶层

    # 子图1样式设置
    ax1.set_xlabel("时间 (秒)", fontsize=12)
    ax1.set_ylabel("速度 (km/h)", fontsize=12)
    ax1.set_title("无人车速度预测 + 异常检测", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)  # 添加网格（透明度0.3）

    # -------- 子图2：驾驶模式分类可视化 1--------
    # 定义模式颜色和标签映射
    mode_colors = {0: "#A23B72", 1: "#2E86AB", 2: "#F18F01"}  # 减速-紫 | 匀速-蓝 | 加速-橙
    mode_names = {0: "减速", 1: "匀速", 2: "加速"}

    # 按驾驶模式绘制散点图
    for mode in [0, 1, 2]:
        mode_idx = np.where(mode_labels == mode)[0]
        ax2.scatter(time_steps[mode_idx], velocity[mode_idx],
                    color=mode_colors[mode], label=mode_names[mode], s=50, alpha=0.7)

    # 子图2样式设置
    ax2.set_xlabel("时间 (秒)", fontsize=12)
    ax2.set_ylabel("速度 (km/h)", fontsize=12)
    ax2.set_title("无人车驾驶模式分类", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # 调整子图间距，避免重叠
    plt.tight_layout()
    # 显示图形
    plt.show()


# ===================== 主函数（程序入口） =====================
def main():
    """
    程序主入口：整合所有模块，执行数据生成→模型训练→结果分析→可视化
    """
    # 1. 配置核心参数
    TEST_DURATION = 60  # 测试时长60秒
    SAMPLE_FREQ = 2  # 采样频率2次/秒（提高数据密度）
    MAX_VELOCITY = 30  # 最大速度30km/h
    NOISE_LEVEL = 0.5  # 速度噪声±0.5km/h

    # 2. 生成模拟无人车速度数据
    print("===== 生成无人车速度测试数据 =====")
    time_steps, velocity = generate_velocity_data(
        test_duration=TEST_DURATION,
        sample_freq=SAMPLE_FREQ,
        max_velocity=MAX_VELOCITY,
        noise_level=NOISE_LEVEL,
        add_abnormal=True
    )

    # 3. 执行速度预测（线性回归）
    print("\n===== 速度预测（线性回归） =====")
    predict_time, predict_vel, mae = velocity_prediction(time_steps, velocity)
    print(f"预测未来{len(predict_time)}个时间点的速度")
    print(f"预测平均绝对误差（MAE）：{mae:.2f} km/h")  # 保留2位小数

    # 4. 执行异常检测（孤立森林）
    print("\n===== 异常检测（孤立森林） =====")
    abnormal_label = detect_abnormal_velocity(velocity)
    abnormal_num = len(np.where(abnormal_label == -1)[0])  # 统计异常点数量
    print(f"检测到异常速度点数量：{abnormal_num} 个")
    abnormal_time = time_steps[abnormal_label == -1]  # 异常点时间戳
    print(f"异常点时间戳：{abnormal_time.round(2)}")  # 保留2位小数

    # 5. 执行驾驶模式分类（随机森林）
    print("\n===== 驾驶模式分类（随机森林） =====")
    mode_labels, accuracy = classify_driving_mode(time_steps, velocity)
    print(f"分类准确率：{accuracy:.2%}")  # 百分比格式显示准确率
    # 统计各模式数量
    mode_count = {0: 0, 1: 0, 2: 0}
    for label in mode_labels:
        mode_count[label] += 1
    print(f"减速模式次数：{mode_count[0]} | 匀速模式次数：{mode_count[1]} | 加速模式次数：{mode_count[2]}")

    # 6. 可视化所有分析结果
    print("\n===== 可视化分析结果 =====")
    plot_ml_results(time_steps, velocity, predict_time, predict_vel, abnormal_label, mode_labels)


# 程序执行入口
if __name__ == "__main__":
    main()