"""
无人车烟雾传感器监测系统
功能：数据生成 → 异常检测 → 等级分类 → 实时监测
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 中文字体全局设置 ────────────────────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ===================== 0. 全局配置 =============================================

@dataclass
class SensorConfig:
    """烟雾传感器系统参数（集中管理，避免魔法数字）"""

    # 数据生成
    sample_num: int = 1000          # 采样点数
    sample_freq: int = 1            # 采样频率 (Hz)
    noise_level: float = 0.05       # 传感器高斯噪声 (ppm)
    add_abnormal: bool = True       # 是否插入突发异常值

    # 风险阈值 (ppm)
    low_risk_threshold: float = 5.0
    high_risk_threshold: float = 20.0

    # 数据分布比例
    low_risk_ratio: float = 0.15    # 低风险段占比
    high_risk_ratio: float = 0.05   # 高风险段占比
    abnormal_count_range: Tuple[int, int] = (3, 8)  # 突发异常点数范围

    # 模型
    iso_contamination: float = 0.08  # 孤立森林异常比例估计
    rf_n_estimators: int = 100
    rf_max_depth: int = 6
    test_size: float = 0.2
    random_state: int = 42

    # 标签映射
    LEVEL_NAMES: Dict[int, str] = None

    def __post_init__(self):
        self.LEVEL_NAMES = {0: "无报警", 1: "低风险", 2: "高风险"}


CFG = SensorConfig()


# ===================== 1. 数据生成 ============================================

def generate_smoke_sensor_data(
        cfg: SensorConfig = CFG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成无人车烟雾传感器时序数据。

    Returns
    -------
    time_steps : (N,) float  时间戳 (秒)
    smoke_conc : (N,) float  烟雾浓度 (ppm)
    labels     : (N,) int    报警标签 {0: 无, 1: 低风险, 2: 高风险}
    """
    rng = np.random.default_rng(cfg.random_state)
    n = cfg.sample_num

    # 时间轴
    time_steps = np.arange(n) / cfg.sample_freq

    # 基础正常浓度 (0 ~ low_risk_threshold)
    smoke_conc = rng.uniform(0, cfg.low_risk_threshold, n).astype(np.float32)

    # 确保三类索引互不重叠
    all_idx = np.arange(n)
    n_low = int(n * cfg.low_risk_ratio)
    n_high = int(n * cfg.high_risk_ratio)

    low_risk_idx = rng.choice(all_idx, n_low, replace=False)
    remaining_idx = np.setdiff1d(all_idx, low_risk_idx)
    high_risk_idx = rng.choice(remaining_idx, n_high, replace=False)

    smoke_conc[low_risk_idx] = rng.uniform(
        cfg.low_risk_threshold, cfg.high_risk_threshold, n_low
    )
    smoke_conc[high_risk_idx] = rng.uniform(cfg.high_risk_threshold, 50.0, n_high)

    # 传感器高斯噪声
    smoke_conc += rng.normal(0, cfg.noise_level, n).astype(np.float32)
    smoke_conc = np.maximum(smoke_conc, 0)

    # 突发异常值（传感器故障 / 瞬时浓烟）
    if cfg.add_abnormal:
        n_abnormal = rng.integers(*cfg.abnormal_count_range, endpoint=True)
        abnormal_idx = rng.choice(n, n_abnormal, replace=False)
        smoke_conc[abnormal_idx] = rng.uniform(60.0, 100.0, n_abnormal)

    smoke_conc = np.round(smoke_conc, 2)

    # 基于阈值生成标签
    labels = np.zeros(n, dtype=np.int32)
    labels[smoke_conc >= cfg.low_risk_threshold] = 1
    labels[smoke_conc >= cfg.high_risk_threshold] = 2

    return time_steps, smoke_conc, labels


# ===================== 2. 特征工程 + 预处理 ===================================

def build_features(
        smoke_conc: np.ndarray,
        cfg: SensorConfig = CFG,
        window: int = 5,
) -> np.ndarray:
    """
    从原始浓度序列构建 7 维时序特征。

    特征列
    ------
    0: 当前浓度
    1: 前 1 帧浓度
    2: 浓度变化率 (ppm/s)
    3: 滑动均值 (window 帧)
    4: 滑动标准差 (window 帧)
    5: 与滑动均值的偏差
    6: 二阶差分（加速度）
    """
    prev_conc = np.concatenate([[0.0], smoke_conc[:-1]])
    diff1 = np.diff(smoke_conc, prepend=smoke_conc[0]) * cfg.sample_freq
    diff2 = np.diff(diff1, prepend=diff1[0])

    # 用 stride_tricks 高效计算滑动统计量
    def rolling_stat(arr: np.ndarray, w: int):
        pad = np.pad(arr, (w - 1, 0), mode="edge")
        shape = (len(arr), w)
        strides = (pad.strides[0], pad.strides[0])
        windows = np.lib.stride_tricks.as_strided(pad, shape=shape, strides=strides)
        return windows.mean(axis=1), windows.std(axis=1)

    roll_mean, roll_std = rolling_stat(smoke_conc, window)
    deviation = smoke_conc - roll_mean

    return np.column_stack([
        smoke_conc,
        prev_conc,
        diff1,
        roll_mean,
        roll_std,
        deviation,
        diff2,
    ])


def preprocess_smoke_data(
        smoke_conc: np.ndarray,
        labels: np.ndarray,
        cfg: SensorConfig = CFG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, np.ndarray]:
    """
    特征构建 → 标准化 → 训练/测试集划分。

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, features_raw
    """
    features = build_features(smoke_conc, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features


# ===================== 3. 异常检测（孤立森林） ================================

def detect_smoke_abnormal(
        smoke_conc: np.ndarray,
        cfg: SensorConfig = CFG,
) -> np.ndarray:
    """
    基于孤立森林检测异常浓度点。

    Returns
    -------
    labels : (N,) int  1 = 正常，-1 = 异常
    """
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=cfg.iso_contamination,
        random_state=cfg.random_state,
    )
    return iso_forest.fit_predict(smoke_conc.reshape(-1, 1))


# ===================== 4. 分类模型训练 =======================================

def train_smoke_alarm_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        cfg: SensorConfig = CFG,
) -> RandomForestClassifier:
    """训练随机森林报警等级分类模型。"""
    model = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        class_weight="balanced",   # 应对类别不平衡
        random_state=cfg.random_state,
    )
    model.fit(X_train, y_train)
    return model


# ===================== 5. 模型评估 ============================================

def evaluate_alarm_model(
        model: RandomForestClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cfg: SensorConfig = CFG,
) -> Dict[str, float]:
    """
    评估分类模型并绘制混淆矩阵。

    Returns
    -------
    dict 包含 accuracy 及各类 precision / recall / f1
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n===== 报警等级分类报告 =====")
    report = classification_report(
        y_test, y_pred,
        target_names=list(cfg.LEVEL_NAMES.values()),
        output_dict=True,
    )
    print(classification_report(y_test, y_pred, target_names=list(cfg.LEVEL_NAMES.values())))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm, list(cfg.LEVEL_NAMES.values()))

    metrics = {"accuracy": accuracy}
    for cls_name in cfg.LEVEL_NAMES.values():
        for metric in ("precision", "recall", "f1-score"):
            metrics[f"{cls_name}_{metric}"] = report[cls_name][metric]
    return metrics


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
    """绘制混淆矩阵热力图（内部辅助函数）。"""
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="烟雾报警等级 — 混淆矩阵",
        ylabel="真实等级",
        xlabel="预测等级",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    fig.tight_layout()
    plt.show()


# ===================== 6. 可视化监测结果 ======================================

_LEVEL_COLORS = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}


def visualize_smoke_alarm(
        time_steps: np.ndarray,
        smoke_conc: np.ndarray,
        abnormal_label: np.ndarray,
        alarm_level: np.ndarray,
        cfg: SensorConfig = CFG,
) -> None:
    """
    绘制双子图：
      - 上图：浓度曲线 + 孤立森林触发点 + 阈值线
      - 下图：按分类等级着色的散点图
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle("无人车烟雾传感器实时监测", fontsize=14, fontweight="bold")

    # ── 子图 1：浓度曲线 ──────────────────────────────────────
    ax1.plot(time_steps, smoke_conc, color="#1f77b4", linewidth=1.2,
             label="烟雾浓度", zorder=2)

    alarm_idx = np.where(abnormal_label == -1)[0]
    if len(alarm_idx):
        ax1.scatter(time_steps[alarm_idx], smoke_conc[alarm_idx],
                    color="#d62728", s=60, zorder=5, label=f"异常触发（{len(alarm_idx)} 点）")

    ax1.axhline(cfg.low_risk_threshold,  color="#ff7f0e", ls="--", lw=1,
                label=f"低风险阈值（{cfg.low_risk_threshold} ppm）")
    ax1.axhline(cfg.high_risk_threshold, color="#d62728",  ls="--", lw=1,
                label=f"高风险阈值（{cfg.high_risk_threshold} ppm）")

    ax1.set_ylabel("烟雾浓度 (ppm)")
    ax1.set_title("浓度曲线 + 异常报警标记")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    # ── 子图 2：等级分类散点 ─────────────────────────────────
    point_colors = np.array([_LEVEL_COLORS[lv] for lv in alarm_level])
    ax2.scatter(time_steps, smoke_conc, c=point_colors, s=12, alpha=0.7)

    for level, name in cfg.LEVEL_NAMES.items():
        ax2.scatter([], [], color=_LEVEL_COLORS[level], s=40, label=name)

    ax2.set_xlabel("时间 (秒)")
    ax2.set_ylabel("烟雾浓度 (ppm)")
    ax2.set_title("报警等级分类")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ===================== 7. 实时单点预测 ========================================

def realtime_smoke_monitor(
        model: RandomForestClassifier,
        scaler: StandardScaler,
        current_conc: float,
        prev_conc: float,
        cfg: SensorConfig = CFG,
        roll_mean: Optional[float] = None,
        roll_std: Optional[float] = None,
) -> Tuple[int, str]:
    """
    对单帧传感器读数预测报警等级。

    Parameters
    ----------
    current_conc : 当前浓度 (ppm)
    prev_conc    : 上一帧浓度 (ppm)
    roll_mean    : 近期滑动均值（可选，缺省用当前值替代）
    roll_std     : 近期滑动标准差（可选，缺省用 0）

    Returns
    -------
    (level_label, level_name)
    """
    diff1 = (current_conc - prev_conc) * cfg.sample_freq
    diff2 = 0.0  # 单帧无法计算二阶差分，填 0

    _roll_mean = roll_mean if roll_mean is not None else current_conc
    _roll_std  = roll_std  if roll_std  is not None else 0.0
    deviation  = current_conc - _roll_mean

    features = np.array([[current_conc, prev_conc, diff1,
                          _roll_mean, _roll_std, deviation, diff2]])
    features_scaled = scaler.transform(features)
    level_label = int(model.predict(features_scaled)[0])

    return level_label, cfg.LEVEL_NAMES[level_label]


# ===================== 主流程 =================================================

def main() -> None:
    cfg = SensorConfig()

    # 1. 生成数据
    print("===== 1. 生成无人车烟雾传感器数据 =====")
    time_steps, smoke_conc, labels = generate_smoke_sensor_data(cfg)
    print(f"采样点数 : {len(smoke_conc)}")
    print(f"时间范围 : 0 ~ {time_steps[-1]:.1f} 秒")
    print(f"浓度范围 : {smoke_conc.min():.2f} ~ {smoke_conc.max():.2f} ppm")
    label_dist = {cfg.LEVEL_NAMES[k]: int(np.sum(labels == k)) for k in cfg.LEVEL_NAMES}
    print(f"标签分布 : {label_dist}")

    # 2. 异常检测
    print("\n===== 2. 孤立森林异常检测 =====")
    abnormal_label = detect_smoke_abnormal(smoke_conc, cfg)
    alarm_count = int(np.sum(abnormal_label == -1))
    print(f"检测到异常点 : {alarm_count} 个")
    print(f"前 5 个异常时间戳 : {time_steps[abnormal_label == -1][:5].round(2)} 秒")

    # 3. 预处理
    print("\n===== 3. 特征工程 + 数据预处理 =====")
    X_train, X_test, y_train, y_test, scaler, features = preprocess_smoke_data(
        smoke_conc, labels, cfg
    )
    print(f"特征维度 : {features.shape[1]} 维")
    print(f"训练集 / 测试集 : {len(X_train)} / {len(X_test)}")

    # 4. 训练
    print("\n===== 4. 训练随机森林分类模型 =====")
    alarm_model = train_smoke_alarm_model(X_train, y_train, cfg)

    # 5. 评估
    print("\n===== 5. 模型评估 =====")
    metrics = evaluate_alarm_model(alarm_model, X_test, y_test, cfg)
    print(f"整体准确率 : {metrics['accuracy']:.2%}")

    # 6. 全量预测
    print("\n===== 6. 全量数据等级预测 =====")
    alarm_level = alarm_model.predict(scaler.transform(features))
    level_dist = {cfg.LEVEL_NAMES[k]: int(np.sum(alarm_level == k)) for k in cfg.LEVEL_NAMES}
    print(f"预测分布 : {level_dist}")

    # 7. 可视化
    print("\n===== 7. 可视化监测结果 =====")
    visualize_smoke_alarm(time_steps, smoke_conc, abnormal_label, alarm_level, cfg)

    # 8. 实时监测演示
    print("\n===== 8. 实时监测演示 =====")
    demos = [
        (3.0,  2.8,  "正常浓度"),
        (10.0, 8.5,  "低风险"),
        (25.0, 20.0, "高风险"),
        (80.0, 10.0, "突发异常"),
    ]
    for cur, prev, desc in demos:
        lv, name = realtime_smoke_monitor(alarm_model, scaler, cur, prev, cfg)
        print(f"  [{desc}] 当前 {cur:.1f} ppm → {name}")


if __name__ == "__main__":
    main()