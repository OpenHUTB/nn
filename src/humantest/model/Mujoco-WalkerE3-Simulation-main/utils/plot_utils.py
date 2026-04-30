"""绘图和数据记录模块"""
import matplotlib.pyplot as plt
import numpy as np

# 绘图数据采集
recording = False
record_start_time = None
record_duration = 5.0  # 5秒钟
tau_history = []       # 存储每次采样的tau（12维）
dqj_history = []       # 存储每次采样的dqj（12维）
time_history = []      # 存储时间戳


def plot_data():
    """绘制采集的 tau 和 dqj 数据"""
    # 将数据转换为 numpy 数组，便于处理
    tau_data = np.array(tau_history)  # shape: (N, 12)
    dqj_data = np.array(dqj_history)  # shape: (N, 12)
    t_data = np.array(time_history)

    # 绘制 tau 数据，每个关节一个子图
    fig1, axs1 = plt.subplots(3, 4, figsize=(15, 8))
    fig1.suptitle("Tau Data (12 joints)")
    for i in range(12):
        ax = axs1[i // 4, i % 4]
        ax.plot(t_data, tau_data[:, i])
        ax.set_title(f"Joint {i}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tau")
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 绘制 dqj 数据，每个关节一个子图
    fig2, axs2 = plt.subplots(3, 4, figsize=(15, 8))
    fig2.suptitle("dqj Data (12 joints)")
    for i in range(12):
        ax = axs2[i // 4, i % 4]
        ax.plot(t_data, dqj_data[:, i])
        ax.set_title(f"Joint {i}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("dqj")
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def reset_recording():
    """重置数据记录"""
    global recording, record_start_time
    recording = False
    record_start_time = None
    tau_history.clear()
    dqj_history.clear()
    time_history.clear()

