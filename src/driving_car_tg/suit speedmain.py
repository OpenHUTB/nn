import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ===================== 配置参数（可根据测试需求调整） =====================
SAMPLING_FREQ = 10  # 采样频率（Hz），即每秒采集10次速度数据
TEST_DURATION = 60  # 测试时长（秒）
BASE_SPEED = 20  # 基础速度（km/h），模拟无人车正常行驶速度
SPEED_NOISE = 0.8  # 速度噪声强度（模拟传感器误差）
ACCELERATION = 0.1  # 平均加速度（km/h/s），可设为0模拟匀速
ABNORMAL_PROB = 0.01  # 异常值出现概率（模拟传感器故障/突发状况）
ABNORMAL_SCALE = 3  # 异常值放大倍给数就


# ===================== 核心功能函数 =====================
def generate_speed_data() -> tuple:
    """
    生成模拟的无人车速度数据（替换此函数可接入真实传感器数据）
    返回：时间序列（s）、速度序列（km/h）
    """
    # 生成时间序列
    time_seq = np.linspace(0, TEST_DURATION, int(SAMPLING_FREQ * TEST_DURATION))

    # 基础速度（含线性加速/减速）
    base_speed_seq = BASE_SPEED + ACCELERATION * time_seq

    # 加入随机噪声
    noise = np.random.normal(0, SPEED_NOISE, size=len(time_seq))
    speed_seq = base_speed_seq + noise

    # 插入异常值（模拟突发故障）
    abnormal_indices = np.random.choice(
        len(time_seq),
        size=int(ABNORMAL_PROB * len(time_seq)),
        replace=False
    )
    speed_seq[abnormal_indices] *= ABNORMAL_SCALE

    return time_seq, speed_seq


def calculate_speed_metrics(time_seq: np.ndarray, speed_seq: np.ndarray) -> dict:
    """
    计算速度测试关键指标
    返回：指标字典
    """
    # 基础统计指标
    avg_speed = np.mean(speed_seq)
    max_speed = np.max(speed_seq)
    min_speed = np.min(speed_seq)
    speed_std = np.std(speed_seq)  # 速度波动率（标准差）

    # 计算加速度（差分法，km/h/s）
    speed_diff = np.diff(speed_seq)
    time_diff = np.diff(time_seq)
    acceleration_seq = speed_diff / time_diff
    avg_acceleration = np.mean(acceleration_seq)
    max_acceleration = np.max(acceleration_seq)

    # 异常值检测（3σ原则）
    abnormal_threshold = 3 * speed_std
    abnormal_mask = np.abs(speed_seq - avg_speed) > abnormal_threshold
    abnormal_count = np.sum(abnormal_mask)
    abnormal_ratio = abnormal_count / len(speed_seq)

    return {
        "测试时长(s)": TEST_DURATION,
        "采样频率(Hz)": SAMPLING_FREQ,
        "平均速度(km/h)": round(avg_speed, 2),
        "最大速度(km/h)": round(max_speed, 2),
        "最小速度(km/h)": round(min_speed, 2),
        "速度波动率(km/h)": round(speed_std, 2),
        "平均加速度(km/h/s)": round(avg_acceleration, 2),
        "最大加速度(km/h/s)": round(max_acceleration, 2),
        "异常值数量": int(abnormal_count),
        "异常值占比(%)": round(abnormal_ratio * 100, 2)
    }


def plot_speed_curve(time_seq: np.ndarray, speed_seq: np.ndarray, metrics: dict):
    """
    可视化速度-时间曲线，标注关键指标
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制速度曲线
    ax.plot(time_seq, speed_seq, color='#1f77b4', linewidth=1.5, label='实时速度')

    # 标注关键指标
    avg_speed = metrics["平均速度(km/h)"]
    max_speed = metrics["最大速度(km/h)"]
    min_speed = metrics["最小速度(km/h)"]

    ax.axhline(y=avg_speed, color='#ff7f0e', linestyle='--', linewidth=2, label=f'平均速度 {avg_speed} km/h')
    ax.scatter(
        time_seq[np.argmax(speed_seq)], max_speed,
        color='#d62728', s=100, label=f'最大速度 {max_speed} km/h'
    )
    ax.scatter(
        time_seq[np.argmin(speed_seq)], min_speed,
        color='#2ca02c', s=100, label=f'最小速度 {min_speed} km/h'
    )

    # 格式设置
    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_ylabel('速度 (km/h)', fontsize=12)
    ax.set_title(f'无人车速度测试曲线（测试时长：{TEST_DURATION}s）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    # 保存图片
    plt.tight_layout()
    plt.savefig(f'无人车速度测试曲线_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300)
    plt.show()


def generate_test_report(metrics: dict, save_to_file: bool = True):
    """
    生成测试报告（控制台输出+可选文件保存）
    """
    report_title = f"===== 无人车速度测试报告 =====\n测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_content = report_title + "-" * 50 + "\n"

    for key, value in metrics.items():
        report_content += f"{key}: {value}\n"
    report_content += "-" * 50 + "\n"

    # 控制台输出
    print(report_content)

    # 保存到文件
    if save_to_file:
        report_filename = f'无人车速度测试报告_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"测试报告已保存至：{report_filename}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("===== 开始无人车速度测试 =====")

    # 1. 生成速度数据（模拟/真实）
    print("步骤1：生成/采集速度数据...")
    time_seq, speed_seq = generate_speed_data()

    # 2. 计算测试指标
    print("步骤2：计算速度测试指标...")
    speed_metrics = calculate_speed_metrics(time_seq, speed_seq)

    # 3. 可视化速度曲线
    print("步骤3：绘制速度测试曲线...")
    plot_speed_curve(time_seq, speed_seq, speed_metrics)

    # 4. 生成测试报告
    print("步骤4：生成测试报告...")
    generate_test_report(speed_metrics, save_to_file=True)

    print("===== 无人车速度测试完成 =====")