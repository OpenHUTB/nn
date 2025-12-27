"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 传感器数据生成器
# ============================================================================

class SensorSimulator:
    """传感器数据模拟器"""

    def __init__(self, duration=5.0, fs=500):
        self.duration = duration
        self.fs = fs
        self.dt = 1.0 / fs
        self.time = np.arange(0, duration, self.dt)
        self.n_samples = len(self.time)

        # 噪声参数
        self.noise_levels = {
            'joint_pos': 0.02,
            'joint_vel': 0.05,
            'force': 0.3,
            'accel': 0.1,
            'gyro': 0.02,
        }

    def generate_true_trajectory(self):
        """生成真实轨迹"""
        t = self.time

        # 6个关节的真实轨迹
        true_joints = np.zeros((self.n_samples, 6))

        # 不同频率的正弦运动
        freqs = [0.5, 0.8, 1.2, 1.5, 2.0, 2.5]
        amplitudes = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1]

        for i in range(6):
            true_joints[:, i] = amplitudes[i] * np.sin(2 * np.pi * freqs[i] * t + i * 0.5)

        return true_joints

    def add_sensor_noise(self, clean_signal, sensor_type):
        """添加传感器噪声"""
        noise_std = self.noise_levels.get(sensor_type, 0.1)
        noise = noise_std * np.random.randn(*clean_signal.shape)
        bias = 0.01 * np.random.randn()  # 小偏置
        return clean_signal + noise + bias

    def simulate_sensors(self, true_joints):
        """模拟多个传感器"""
        n_samples = self.n_samples

        # 计算真实速度（通过差分）
        true_vel = np.gradient(true_joints, axis=0) / self.dt

        # 模拟传感器数据
        sensors = {}

        # 1. 关节位置传感器
        sensors['joint_pos'] = self.add_sensor_noise(true_joints, 'joint_pos')

        # 2. 关节速度传感器
        sensors['joint_vel'] = self.add_sensor_noise(true_vel, 'joint_vel')

        # 3. 力传感器（模拟抓取力）
        force = np.zeros((n_samples, 3))
        for i in range(3):
            base_force = 2.0 + 0.5 * np.sin(2 * np.pi * 0.3 * self.time)
            force[:, i] = base_force + 0.1 * np.random.randn(n_samples)
        sensors['force'] = force

        # 4. IMU传感器
        # 加速度计
        accel = np.zeros((n_samples, 3))
        accel[:, 2] = -9.81  # 重力
        accel += 0.1 * np.random.randn(n_samples, 3)
        sensors['accel'] = self.add_sensor_noise(accel, 'accel')

        # 陀螺仪
        gyro = np.zeros((n_samples, 3))
        gyro[:, 0] = 0.5 * np.sin(2 * np.pi * 0.2 * self.time)
        sensors['gyro'] = self.add_sensor_noise(gyro, 'gyro')

        return sensors, true_joints, true_vel

# ============================================================================
# 2. 滤波与融合算法
# ============================================================================

class KalmanFilter:
    """简化版卡尔曼滤波器"""

    def __init__(self, n_states, n_measurements):
        self.n_states = n_states
        self.n_measurements = n_measurements

        # 初始化
        self.x = np.zeros(n_states)  # 状态
        self.P = np.eye(n_states)    # 协方差
        self.F = np.eye(n_states)    # 状态转移矩阵
        self.H = np.eye(n_measurements, n_states)  # 测量矩阵
        self.Q = np.eye(n_states) * 0.01  # 过程噪声
        self.R = np.eye(n_measurements) * 0.1  # 测量噪声

        # 历史记录
        self.x_history = []
        self.P_history = []

    def predict(self):
        """预测步骤"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        """更新步骤"""
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 更新状态
        y = z - self.H @ self.x
        self.x = self.x + K @ y

        # 更新协方差
        I = np.eye(self.n_states)
        self.P = (I - K @ self.H) @ self.P

        # 保存历史
        self.x_history.append(self.x.copy())
        self.P_history.append(self.P.copy())

        return self.x.copy()

    def run(self, measurements):
        """运行滤波器"""
        estimates = []
        for z in measurements:
            self.predict()
            x_est = self.update(z)
            estimates.append(x_est)
        return np.array(estimates)

def butter_lowpass_filter(data, cutoff_freq, fs, order=4):
    """巴特沃斯低通滤波器"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data, axis=0)
    return y

# ============================================================================
# 3. 可视化系统
# ============================================================================

class SensorFusionVisualizer:
    """传感器融合可视化器"""

    def __init__(self):
        plt.style.use('default')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6A8D73', '#FF6B6B', '#4ECDC4']

    def plot_sensor_comparison(self, time, sensors, true_data, filtered_data, sensor_name):
        """绘制传感器数据对比"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 1. 原始传感器数据
        ax1 = axes[0, 0]
        if sensor_name in sensors:
            sensor_data = sensors[sensor_name]
            if sensor_data.ndim > 1:
                for i in range(min(3, sensor_data.shape[1])):
                    ax1.plot(time, sensor_data[:, i],
                            label=f'{sensor_name}[{i}]', alpha=0.7)
            else:
                ax1.plot(time, sensor_data, label=sensor_name, alpha=0.7)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Sensor Value')
        ax1.set_title(f'Raw {sensor_name} Data')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. 滤波前后对比
        ax2 = axes[0, 1]
        if sensor_name in sensors and sensor_name in filtered_data:
            raw = sensors[sensor_name]
            filtered = filtered_data[sensor_name]

            if raw.ndim == 1:
                ax2.plot(time, raw, 'b-', alpha=0.5, label='Raw', linewidth=1)
                ax2.plot(time, filtered, 'r-', label='Filtered', linewidth=1.5)
            else:
                ax2.plot(time, raw[:, 0], 'b-', alpha=0.5, label='Raw[0]', linewidth=1)
                ax2.plot(time, filtered[:, 0], 'r-', label='Filtered[0]', linewidth=1.5)

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Value')
        ax2.set_title('Filter Comparison')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. 频谱分析
        ax3 = axes[1, 0]
        if sensor_name in sensors:
            data = sensors[sensor_name]
            if data.ndim == 1:
                signal_data = data
            else:
                signal_data = data[:, 0] if data.shape[1] > 0 else np.mean(data, axis=1)

            # 计算功率谱
            if len(time) > 1:
                fs = 1.0 / (time[1] - time[0])
            else:
                fs = 1000

            n_segments = min(256, len(signal_data))
            if n_segments > 10:
                f, Pxx = signal.welch(signal_data, fs, nperseg=n_segments)
                ax3.semilogy(f, Pxx, 'g-', linewidth=1.5)
                ax3.set_xlabel('Frequency (Hz)')
                ax3.set_ylabel('Power Spectral Density')
                ax3.set_title('Frequency Spectrum')
                ax3.grid(True, alpha=0.3)

        # 4. 残差分析
        ax4 = axes[1, 1]
        if sensor_name in sensors and sensor_name in filtered_data:
            raw = sensors[sensor_name]
            filtered = filtered_data[sensor_name]

            if raw.ndim == 1:
                residuals = raw - filtered
            else:
                residuals = raw[:, 0] - filtered[:, 0]

            ax4.plot(time, residuals, 'r-', alpha=0.7, linewidth=1)
            ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)

            # 添加3σ边界
            if len(residuals) > 10:
                std = np.std(residuals)
                ax4.axhline(y=3*std, color='b', linestyle=':', alpha=0.5)
                ax4.axhline(y=-3*std, color='b', linestyle=':', alpha=0.5)
                ax4.fill_between(time, -3*std, 3*std, alpha=0.1, color='blue')

                ax4.text(0.02, 0.98, f'Std: {std:.3f}',
                        transform=ax4.transAxes, fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Residual')
        ax4.set_title('Filter Residuals')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'{sensor_name} Sensor Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_state_estimation(self, time, true_states, estimates, state_names=None):
        """绘制状态估计结果"""
        if true_states is None or estimates is None:
            print("警告: 没有状态估计数据")
            return None

        if state_names is None:
            state_names = [f'State {i+1}' for i in range(true_states.shape[1])]

        n_states = min(4, true_states.shape[1])  # 最多显示4个状态

        fig, axes = plt.subplots(n_states, 2, figsize=(14, 3*n_states))
        if n_states == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_states):
            # 左侧：真实值 vs 估计值
            ax1 = axes[i, 0]

            min_len = min(len(time), len(true_states), len(estimates))
            t_plot = time[:min_len]
            true_plot = true_states[:min_len, i]
            est_plot = estimates[:min_len, i]

            ax1.plot(t_plot, true_plot, 'g-', linewidth=2, label='True')
            ax1.plot(t_plot, est_plot, 'r--', linewidth=2, label='Estimated')

            # 计算并显示误差
            error = true_plot - est_plot
            rmse = np.sqrt(np.mean(error**2))
            mae = np.mean(np.abs(error))

            ax1.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}',
                    transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Value')
            ax1.set_title(f'{state_names[i]} Estimation')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            # 右侧：误差分布
            ax2 = axes[i, 1]

            ax2.hist(error, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)

            # 添加正态分布拟合
            if len(error) > 10:
                mu, sigma = norm.fit(error)
                x = np.linspace(min(error), max(error), 100)
                y = norm.pdf(x, mu, sigma) * len(error) * (max(error)-min(error))/30
                ax2.plot(x, y, 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.3f}, σ={sigma:.3f}')

            ax2.set_xlabel('Error')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'{state_names[i]} Error Distribution')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

        plt.suptitle('State Estimation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_sensor_fusion_summary(self, time, sensors, filtered_data, true_states, estimates):
        """绘制传感器融合总结"""
        fig = plt.figure(figsize=(16, 10))

        # 1. 所有传感器原始数据
        ax1 = plt.subplot(3, 3, 1)
        sensor_keys = list(sensors.keys())[:3]  # 只显示前3个传感器

        for i, key in enumerate(sensor_keys):
            data = sensors[key]
            if data.ndim == 1:
                ax1.plot(time, data, alpha=0.5, linewidth=1, label=key)
            else:
                ax1.plot(time, data[:, 0], alpha=0.5, linewidth=1, label=f'{key}[0]')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Value')
        ax1.set_title('Raw Sensor Data')
        ax1.legend(loc='best', fontsize=7)
        ax1.grid(True, alpha=0.3)

        # 2. 所有传感器滤波后数据
        ax2 = plt.subplot(3, 3, 2)
        for i, key in enumerate(sensor_keys):
            if key in filtered_data:
                data = filtered_data[key]
                if data.ndim == 1:
                    ax2.plot(time, data, alpha=0.7, linewidth=1.5, label=key)
                else:
                    ax2.plot(time, data[:, 0], alpha=0.7, linewidth=1.5, label=f'{key}[0]')

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Value')
        ax2.set_title('Filtered Sensor Data')
        ax2.legend(loc='best', fontsize=7)
        ax2.grid(True, alpha=0.3)

        # 3. 状态估计对比（第一个状态）
        ax3 = plt.subplot(3, 3, 3)
        if true_states is not None and estimates is not None:
            min_len = min(len(time), len(true_states), len(estimates))
            if min_len > 0:
                ax3.plot(time[:min_len], true_states[:min_len, 0], 'g-', linewidth=2, label='True')
                ax3.plot(time[:min_len], estimates[:min_len, 0], 'r--', linewidth=2, label='Estimated')

                # 计算误差
                error = true_states[:min_len, 0] - estimates[:min_len, 0]
                rmse = np.sqrt(np.mean(error**2))

                ax3.text(0.02, 0.98, f'RMSE: {rmse:.4f}',
                        transform=ax3.transAxes, fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('State Value')
        ax3.set_title('State 1: True vs Estimated')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # 4. 传感器噪声分析
        ax4 = plt.subplot(3, 3, 4)
        noise_levels = []
        sensor_labels = []

        for key in sensor_keys[:5]:
            if key in sensors:
                data = sensors[key]
                if data.ndim == 1:
                    noise = np.std(data)
                else:
                    noise = np.std(data, axis=0).mean()
                noise_levels.append(noise)
                sensor_labels.append(key)

        if noise_levels:
            bars = ax4.bar(range(len(noise_levels)), noise_levels, alpha=0.7, color='lightcoral')
            ax4.set_xticks(range(len(noise_levels)))
            ax4.set_xticklabels(sensor_labels, rotation=45, fontsize=8)
            ax4.set_ylabel('Noise Std')
            ax4.set_title('Sensor Noise Levels')
            ax4.grid(True, alpha=0.3, axis='y')

        # 5. 滤波效果指标
        ax5 = plt.subplot(3, 3, 5)
        improvement_ratios = []

        for key in sensor_keys[:3]:
            if key in sensors and key in filtered_data:
                raw = sensors[key]
                filtered = filtered_data[key]

                if raw.ndim == 1:
                    raw_std = np.std(raw)
                    filt_std = np.std(filtered)
                else:
                    raw_std = np.std(raw, axis=0).mean()
                    filt_std = np.std(filtered, axis=0).mean()

                if raw_std > 0:
                    improvement = (raw_std - filt_std) / raw_std * 100
                    improvement_ratios.append(improvement)

        if improvement_ratios:
            bars = ax5.bar(range(len(improvement_ratios)), improvement_ratios, alpha=0.7, color='lightgreen')
            ax5.set_xticks(range(len(improvement_ratios)))
            ax5.set_xticklabels(sensor_labels[:len(improvement_ratios)], rotation=45, fontsize=8)
            ax5.set_ylabel('Noise Reduction (%)')
            ax5.set_title('Filter Performance')
            ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax5.grid(True, alpha=0.3, axis='y')

            for bar, imp in zip(bars, improvement_ratios):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{imp:.1f}%', ha='center', va='bottom', fontsize=8)

        # 6. 频谱对比
        ax6 = plt.subplot(3, 3, 6)
        if sensor_keys and sensor_keys[0] in sensors and sensor_keys[0] in filtered_data:
            key = sensor_keys[0]
            raw = sensors[key]
            filtered = filtered_data[key]

            if raw.ndim == 1:
                signal_raw = raw
                signal_filt = filtered
            else:
                signal_raw = raw[:, 0] if raw.shape[1] > 0 else np.mean(raw, axis=1)
                signal_filt = filtered[:, 0] if filtered.shape[1] > 0 else np.mean(filtered, axis=1)

            if len(time) > 1:
                fs = 1.0 / (time[1] - time[0])
            else:
                fs = 1000

            n_segments = min(256, len(signal_raw))
            if n_segments > 10:
                f_raw, Pxx_raw = signal.welch(signal_raw, fs, nperseg=n_segments)
                f_filt, Pxx_filt = signal.welch(signal_filt, fs, nperseg=n_segments)

                ax6.semilogy(f_raw, Pxx_raw, 'b-', alpha=0.5, label='Raw', linewidth=1)
                ax6.semilogy(f_filt, Pxx_filt, 'r-', label='Filtered', linewidth=1.5)

                ax6.set_xlabel('Frequency (Hz)')
                ax6.set_ylabel('Power')
                ax6.set_title('Frequency Spectrum Comparison')
                ax6.legend(loc='best')
                ax6.grid(True, alpha=0.3)

        # 7. 误差相关性
        ax7 = plt.subplot(3, 3, 7)
        if true_states is not None and estimates is not None and true_states.shape[1] >= 2:
            min_len = min(len(true_states), len(estimates))
            if min_len > 0:
                errors = true_states[:min_len] - estimates[:min_len]

                if errors.shape[1] >= 2:
                    ax7.scatter(errors[:, 0], errors[:, 1], alpha=0.5, s=10)
                    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                    ax7.axvline(x=0, color='k', linestyle='--', alpha=0.3)

                    # 计算相关系数
                    corr = np.corrcoef(errors[:, 0], errors[:, 1])[0, 1]
                    ax7.text(0.02, 0.98, f'Correlation: {corr:.3f}',
                            transform=ax7.transAxes, fontsize=9,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                    ax7.set_xlabel('Error State 1')
                    ax7.set_ylabel('Error State 2')
                    ax7.set_title('Error Correlation')
                    ax7.grid(True, alpha=0.3)

        # 8. 估计误差随时间变化
        ax8 = plt.subplot(3, 3, 8)
        if true_states is not None and estimates is not None:
            min_len = min(len(time), len(true_states), len(estimates))
            if min_len > 0:
                errors = true_states[:min_len, 0] - estimates[:min_len, 0]

                ax8.plot(time[:min_len], errors, 'r-', alpha=0.7, linewidth=1)
                ax8.axhline(y=0, color='k', linestyle='--', alpha=0.5)

                # 添加滑动平均
                if len(errors) > 50:
                    window = min(50, len(errors))
                    smooth_errors = np.convolve(errors, np.ones(window)/window, mode='valid')
                    ax8.plot(time[window-1:min_len], smooth_errors, 'b-', linewidth=1.5, label=f'{window}-point avg')

                ax8.set_xlabel('Time (s)')
                ax8.set_ylabel('Error')
                ax8.set_title('Estimation Error Over Time')
                ax8.legend(loc='best', fontsize=8)
                ax8.grid(True, alpha=0.3)

        # 9. 协方差演化
        ax9 = plt.subplot(3, 3, 9)
        # 模拟协方差演化
        if len(time) > 0:
            t_short = time[:100] if len(time) > 100 else time
            cov_evolution = 0.1 * np.exp(-0.1 * t_short) + 0.01  # 指数衰减

            # 修复：正确的颜色和线型指定方式
            ax9.plot(t_short, cov_evolution, color='purple', linestyle='-', linewidth=2)
            ax9.set_xlabel('Time (s)')
            ax9.set_ylabel('Covariance')
            ax9.set_title('State Covariance Evolution')
            ax9.grid(True, alpha=0.3)

        plt.suptitle('Sensor Fusion System Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

# ============================================================================
# 4. 主程序
# ============================================================================

def main():
    """主函数"""
    print("=" * 60)
    print("传感器融合与状态估计可视化系统")
    print("=" * 60)

    # 1. 生成数据
    print("\n1. 生成传感器数据...")
    simulator = SensorSimulator(duration=5.0, fs=500)
    sensors, true_states, true_vel = simulator.simulate_sensors(simulator.generate_true_trajectory())

    print(f"   时间点数: {len(simulator.time)}")
    print(f"   传感器: {list(sensors.keys())}")
    print(f"   真实状态形状: {true_states.shape}")

    # 2. 滤波处理
    print("\n2. 应用滤波算法...")
    filtered_data = {}

    for sensor_name, data in sensors.items():
        if data.ndim == 1:
            filtered_data[sensor_name] = butter_lowpass_filter(data, cutoff_freq=50, fs=simulator.fs)
        else:
            # 对每个维度单独滤波
            filtered_dims = []
            for dim in range(data.shape[1]):
                filtered_dim = butter_lowpass_filter(data[:, dim], cutoff_freq=50, fs=simulator.fs)
                filtered_dims.append(filtered_dim)
            filtered_data[sensor_name] = np.column_stack(filtered_dims)

    print("   滤波完成!")

    # 3. 状态估计（卡尔曼滤波）
    print("\n3. 运行卡尔曼滤波器...")

    # 使用关节位置进行状态估计
    if 'joint_pos' in filtered_data:
        measurements = filtered_data['joint_pos']

        # 初始化卡尔曼滤波器
        n_states = 6
        n_measurements = 6
        kf = KalmanFilter(n_states, n_measurements)

        # 设置状态转移矩阵（简单模型）
        dt = simulator.dt
        for i in range(n_states):
            kf.F[i, i] = 1.0  # 保持位置不变

        # 运行滤波器
        estimates = kf.run(measurements)
        print(f"   状态估计完成: {estimates.shape}")
    else:
        estimates = None
        print("   警告: 没有关节位置数据")

    # 4. 创建可视化
    print("\n4. 创建可视化图表...")
    visualizer = SensorFusionVisualizer()

    # 4.1 传感器对比图
    print("   创建传感器对比图...")
    sensor_names_to_plot = ['joint_pos', 'joint_vel', 'accel']
    for sensor_name in sensor_names_to_plot[:2]:  # 只创建前2个
        if sensor_name in sensors:
            fig = visualizer.plot_sensor_comparison(
                simulator.time, sensors, true_states,
                filtered_data, sensor_name
            )
            fig.savefig(f'sensor_{sensor_name}_analysis.png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"     ✓ 保存: sensor_{sensor_name}_analysis.png")

    # 4.2 状态估计图
    if estimates is not None:
        print("   创建状态估计图...")
        fig = visualizer.plot_state_estimation(
            simulator.time, true_states, estimates,
            state_names=[f'Joint {i+1}' for i in range(6)]
        )
        if fig is not None:
            fig.savefig('state_estimation_results.png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            print("     ✓ 保存: state_estimation_results.png")

    # 4.3 融合总结图
    print("   创建融合总结图...")
    fig = visualizer.plot_sensor_fusion_summary(
        simulator.time, sensors, filtered_data,
        true_states, estimates
    )
    fig.savefig('sensor_fusion_summary.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print("     ✓ 保存: sensor_fusion_summary.png")

    # 5. 性能评估
    print("\n5. 系统性能评估...")

    if estimates is not None:
        # 计算误差指标
        min_len = min(len(true_states), len(estimates))
        errors = true_states[:min_len] - estimates[:min_len]

        rmse_per_state = np.sqrt(np.mean(errors**2, axis=0))
        mae_per_state = np.mean(np.abs(errors), axis=0)

        print("   状态估计误差:")
        print("   " + "-" * 40)
        print("   State |    RMSE    |    MAE    ")
        print("   " + "-" * 40)

        for i in range(min(6, len(rmse_per_state))):
            print(f"     {i+1}   |  {rmse_per_state[i]:.6f}  |  {mae_per_state[i]:.6f}")

        print("   " + "-" * 40)
        print(f"   平均  |  {np.mean(rmse_per_state):.6f}  |  {np.mean(mae_per_state):.6f}")

        # 计算滤波效果
        if 'joint_pos' in sensors and 'joint_pos' in filtered_data:
            raw_std = np.std(sensors['joint_pos'], axis=0).mean()
            filt_std = np.std(filtered_data['joint_pos'], axis=0).mean()
            noise_reduction = (raw_std - filt_std) / raw_std * 100

            print(f"\n   滤波效果:")
            print(f"     - 原始数据标准差: {raw_std:.4f}")
            print(f"     - 滤波后标准差: {filt_std:.4f}")
            print(f"     - 噪声降低: {noise_reduction:.1f}%")

    print("\n" + "=" * 60)
    print("所有任务完成!")
    print("=" * 60)

    # 6. 显示一个示例图
    print("\n6. 显示示例图表...")
    if estimates is not None:
        # 创建简单的性能总结图
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 左侧：RMSE对比
        ax1 = axes[0]
        states = range(1, min(7, len(rmse_per_state)+1))
        bars = ax1.bar(states, rmse_per_state[:len(states)], alpha=0.7, color='lightcoral')
        ax1.set_xlabel('Joint Number')
        ax1.set_ylabel('RMSE (rad)')
        ax1.set_title('Estimation Error per Joint')
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, rmse_val in zip(bars, rmse_per_state[:len(states)]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rmse_val:.4f}', ha='center', va='bottom', fontsize=9)

        # 右侧：时间序列示例
        ax2 = axes[1]
        if len(simulator.time) > 0 and len(true_states) > 0 and len(estimates) > 0:
            min_len = min(len(simulator.time), len(true_states), len(estimates))
            ax2.plot(simulator.time[:min_len], true_states[:min_len, 0],
                    'g-', linewidth=2, label='True (Joint 1)')
            ax2.plot(simulator.time[:min_len], estimates[:min_len, 0],
                    'r--', linewidth=2, label='Estimated')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Joint Angle (rad)')
            ax2.set_title('Joint 1: True vs Estimated')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

        plt.suptitle('Sensor Fusion System Performance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('performance_summary.png', dpi=120, bbox_inches='tight')
        plt.show()

        print("\n生成的文件:")
        print("  1. sensor_joint_pos_analysis.png")
        print("  2. sensor_joint_vel_analysis.png")
        print("  3. state_estimation_results.png")
        print("  4. sensor_fusion_summary.png")
        print("  5. performance_summary.png")
    else:
        print("警告: 无法生成性能总结图，估计数据为空")

if __name__ == "__main__":
    main()