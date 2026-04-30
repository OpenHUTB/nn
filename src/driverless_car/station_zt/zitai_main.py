import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ========== 环境配置：解决中文显示和实时绘图问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑（Windows）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.switch_backend('TkAgg')  # 兼容PyCharm的后端

# 1. 生成模拟数据：传感器输入 → 姿态输出
def generate_sensor_data(n_samples=1000):
    acc = np.random.normal(0, 0.5, (n_samples, 3))  # 加速度 x/y/z
    gyro = np.random.normal(0, 0.3, (n_samples, 3)) # 陀螺仪 x/y/z
    X = np.hstack([acc, gyro])
    # 模拟姿态：俯仰角、横滚角、偏航角（加入噪声）
    pitch = 0.2 * acc[:, 0] + np.random.normal(0, 0.1, n_samples)
    roll = 0.3 * acc[:, 1] + np.random.normal(0, 0.1, n_samples)
    yaw = 0.1 * gyro[:, 2] + np.random.normal(0, 0.1, n_samples)
    y = np.vstack([pitch, roll, yaw]).T
    return X, y

# 2. 训练模型
X_train, y_train = generate_sensor_data(800)
X_test, y_test = generate_sensor_data(200)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 3. 模型预测（获取所有预测值，用于实时显示）
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"测试集均方误差（MSE）：{mse:.4f}")

# 4. 实时动态显示数据变化过程
def real_time_display(y_test, y_pred, delay=0.01):
    # 创建画布和3个子图
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(3, 1, 1)  # 俯仰角
    ax2 = plt.subplot(3, 1, 2)  # 横滚角
    ax3 = plt.subplot(3, 1, 3)  # 偏航角

    # 初始化空的线条对象（用于后续更新数据）
    line1_real, = ax1.plot([], [], label='真实俯仰角', color='blue', linewidth=1.5)
    line1_pred, = ax1.plot([], [], label='预测俯仰角', color='red', linestyle='--', linewidth=1.5)
    line2_real, = ax2.plot([], [], label='真实横滚角', color='green', linewidth=1.5)
    line2_pred, = ax2.plot([], [], label='预测横滚角', color='orange', linestyle='--', linewidth=1.5)
    line3_real, = ax3.plot([], [], label='真实偏航角', color='purple', linewidth=1.5)
    line3_pred, = ax3.plot([], [], label='预测偏航角', color='brown', linestyle='--', linewidth=1.5)

    # 设置子图的标题、标签、图例和网格
    ax1.set_title('无人机俯仰角：真实值 vs 预测值', fontsize=12)
    ax1.set_ylabel('俯仰角（°）', fontsize=10)
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)

    ax2.set_title('无人机横滚角：真实值 vs 预测值', fontsize=12)
    ax2.set_ylabel('横滚角（°）', fontsize=10)
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)

    ax3.set_title('无人机偏航角：真实值 vs 预测值', fontsize=12)
    ax3.set_xlabel('样本序号', fontsize=10)
    ax3.set_ylabel('偏航角（°）', fontsize=10)
    ax3.legend(loc='best')
    ax3.grid(alpha=0.3)

    # 设置x轴和y轴的范围（根据数据动态调整，也可以固定）
    x_max = len(y_test)
    y_limits = {
        'pitch': (np.min(y_test[:, 0]) - 0.5, np.max(y_test[:, 0]) + 0.5),
        'roll': (np.min(y_test[:, 1]) - 0.5, np.max(y_test[:, 1]) + 0.5),
        'yaw': (np.min(y_test[:, 2]) - 0.5, np.max(y_test[:, 2]) + 0.5)
    }
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(y_limits['pitch'])
    ax2.set_xlim(0, x_max)
    ax2.set_ylim(y_limits['roll'])
    ax3.set_xlim(0, x_max)
    ax3.set_ylim(y_limits['yaw'])

    # 开启交互式模式（关键：让图表可以实时更新）
    plt.ion()
    plt.tight_layout()
    plt.show(block=False)  # 非阻塞显示图表

    # 逐样本更新数据，实现实时显示
    for i in range(1, x_max + 1):
        # 更新俯仰角数据
        line1_real.set_data(range(i), y_test[:i, 0])
        line1_pred.set_data(range(i), y_pred[:i, 0])
        # 更新横滚角数据
        line2_real.set_data(range(i), y_test[:i, 1])
        line2_pred.set_data(range(i), y_pred[:i, 1])
        # 更新偏航角数据
        line3_real.set_data(range(i), y_test[:i, 2])
        line3_pred.set_data(range(i), y_pred[:i, 2])

        # 刷新图表
        plt.draw()
        plt.pause(delay)  # 暂停一小段时间，控制显示速度

    # 关闭交互式模式，保持最终图表
    plt.ioff()
    plt.show()

# 调用实时显示函数（delay=0.01表示每更新一个样本暂停0.01秒，可调整）
real_time_display(y_test, y_pred, delay=0.01)