# 导入必要的库
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from matplotlib.animation import FuncAnimation
import random

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class DroneHeightVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("无人机高度保持可视化系统")
        self.root.geometry("1000x700")

        # 数据存储
        self.time_data = []  # 时间轴
        self.current_height = []  # 当前高度
        self.target_height = []  # 目标高度
        self.error_data = []  # 高度误差
        self.throttle_data = []  # 油门控制量
        self.max_points = 100  # 最多显示的数据点

        # 控制参数
        self.target_h = 10.0  # 初始目标高度（米）
        self.running = False  # 是否运行
        self.simulate_noise = True  # 是否模拟气流噪声
        self.dt = 0.1  # 采样时间间隔（秒）

        # 创建UI组件
        self._create_widgets()

        # 初始化绘图
        self._init_plots()

        # 模拟深度学习控制器（实际应用中替换为真实模型）
        self.controller = self._init_controller()

    def _create_widgets(self):
        """创建UI控件"""
        # 控制面板
        control_frame = ttk.LabelFrame(self.root, text="控制参数")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # 目标高度设置
        ttk.Label(control_frame, text="目标高度 (米):").grid(row=0, column=0, padx=5, pady=5)
        self.target_entry = ttk.Entry(control_frame, width=10)
        self.target_entry.insert(0, str(self.target_h))
        self.target_entry.grid(row=0, column=1, padx=5, pady=5)

        # 控制按钮
        self.start_btn = ttk.Button(control_frame, text="开始", command=self.start)
        self.start_btn.grid(row=0, column=2, padx=5, pady=5)

        self.pause_btn = ttk.Button(control_frame, text="暂停", command=self.pause, state=tk.DISABLED)
        self.pause_btn.grid(row=0, column=3, padx=5, pady=5)

        self.reset_btn = ttk.Button(control_frame, text="重置", command=self.reset)
        self.reset_btn.grid(row=0, column=4, padx=5, pady=5)

        # 噪声开关
        self.noise_var = tk.BooleanVar(value=True)
        self.noise_check = ttk.Checkbutton(
            control_frame, text="模拟气流噪声", variable=self.noise_var,
            command=lambda: setattr(self, 'simulate_noise', self.noise_var.get())
        )
        self.noise_check.grid(row=0, column=5, padx=5, pady=5)

        # 状态显示
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=0, column=6, padx=20, pady=5)

    def _init_plots(self):
        """初始化绘图区域"""
        # 创建画布
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 高度曲线（上半部分）
        self.line_current, = self.ax1.plot([], [], label="当前高度", color='blue', linewidth=2)
        self.line_target, = self.ax1.plot([], [], label="目标高度", color='red', linestyle='--', linewidth=1.5)
        self.ax1.set_ylabel("高度 (米)")
        self.ax1.set_title("无人机高度变化曲线")
        self.ax1.grid(True)
        self.ax1.legend()

        # 误差和油门曲线（下半部分）
        self.line_error, = self.ax2.plot([], [], label="高度误差", color='green', linewidth=2)
        self.ax2_twin = self.ax2.twinx()  # 双Y轴
        self.line_throttle, = self.ax2_twin.plot([], [], label="油门控制量", color='orange', linewidth=2, alpha=0.7)
        self.ax2.set_xlabel("时间 (秒)")
        self.ax2.set_ylabel("误差 (米)")
        self.ax2_twin.set_ylabel("油门 (0~1)")
        self.ax2.set_title("高度误差与油门控制量")
        self.ax2.grid(True)
        # 合并图例
        lines = [self.line_error, self.line_throttle]
        self.ax2.legend(lines, [l.get_label() for l in lines], loc='upper left')

        self.fig.tight_layout()

    def _init_controller(self):
        """模拟深度学习控制器（实际应用中替换为训练好的模型）"""

        class SimpleController:
            def __init__(self):
                # 模拟一个经过训练的控制器（实际使用时加载PyTorch/TensorFlow模型）
                # 无人机高度保持：控制器的核心参数，模拟模型学到的比例和微分系数
                self.kp = 0.05  # 比例系数（对应高度误差的响应强度）
                self.kd = 0.02  # 微分系数（对应误差变化率的响应强度，抑制震荡）

            def predict(self, error, error_rate, current_h):
                """
                无人机高度保持：控制器的核心预测逻辑，输入状态输出油门控制量
                输入：误差e（目标高度-当前高度）、误差变化率de、当前高度h
                输出：油门控制量（0~1，0为最小油门，1为最大油门）
                """
                # 模拟深度学习模型的输出（实际中是model(inputs)的前向传播）
                # 基础油门0.5为悬停基准，叠加比例和微分控制项实现高度保持
                throttle = 0.5 + self.kp * error + self.kd * error_rate
                # 限制油门范围在0~1之间，避免超出无人机的物理控制范围
                return np.clip(throttle, 0.0, 1.0)

        return SimpleController()

    def _update_data(self):
        """更新无人机数据（模拟无人机的高度动力学和控制逻辑）"""
        # 获取当前时间
        t = self.time_data[-1] + self.dt if self.time_data else 0.0
        self.time_data.append(t)

        # 更新目标高度（从输入框读取，支持实时修改目标高度，测试高度保持的动态响应）
        # 无人机高度保持：实时更新目标高度，模拟用户手动调整无人机的期望高度
        try:
            new_target = float(self.target_entry.get())
            self.target_h = new_target
        except ValueError:
            pass  # 忽略无效输入，保持原有目标高度
        self.target_height.append(self.target_h)

        # 计算当前高度（基于简化的物理模型模拟无人机的高度变化）
        # 无人机高度保持：核心的高度动力学模拟，反映油门对高度的影响
        if not self.current_height:
            # 初始高度：从地面（0米）开始，模拟无人机起飞阶段
            current_h = 0.0
        else:
            # 基于上一时刻高度和油门计算当前高度（简化的无人机高度物理模型）
            last_h = self.current_height[-1]
            last_throttle = self.throttle_data[-1] if self.throttle_data else 0.5

            # 高度变化 = 油门贡献 - 重力衰减（简化模型：油门>0.45时上升，<0.45时下降）
            # 无人机高度保持：油门是控制高度的核心变量，0.45为悬停临界油门
            h_change = (last_throttle - 0.45) * 0.8  # 0.8为高度变化的增益系数

            # 加入噪声（模拟气流扰动，贴近真实的无人机飞行环境）
            # 无人机高度保持：气流噪声是影响高度稳定的主要干扰因素
            if self.simulate_noise:
                h_change += random.gauss(0, 0.15)  # 加入高斯噪声，均值0，标准差0.15

            # 计算当前高度：上一高度 + 高度变化 * 采样时间间隔
            current_h = last_h + h_change * self.dt

        self.current_height.append(current_h)

        # 计算高度误差（目标高度 - 当前高度，是高度保持控制的核心输入）
        # 无人机高度保持：误差是控制器的关键输入，误差为正表示需要上升，为负表示需要下降
        error = self.target_h - current_h
        self.error_data.append(error)

        # 计算误差变化率（反映误差的变化趋势，抑制高度震荡）
        # 无人机高度保持：误差变化率用于阻尼控制，避免无人机高度频繁震荡
        error_rate = (error - self.error_data[-2]) / self.dt if len(self.error_data) > 1 else 0.0

        # 深度学习控制器计算油门（核心的高度保持控制逻辑）
        # 无人机高度保持：控制器根据误差和误差变化率输出油门，实现高度的闭环控制
        throttle = self.controller.predict(error, error_rate, current_h)
        self.throttle_data.append(throttle)

        # 限制数据长度，只保留最近的max_points个数据点，保证绘图的流畅性
        if len(self.time_data) > self.max_points:
            self.time_data.pop(0)
            self.current_height.pop(0)
            self.target_height.pop(0)
            self.error_data.pop(0)
            self.throttle_data.pop(0)

    def _update_plot(self, frame):
        """更新绘图，可视化无人机高度保持的各项数据"""
        if self.running:
            self._update_data()
            # 实时显示无人机的高度状态，便于监控高度保持的效果
            self.status_var.set(
                f"运行中 | 当前高度: {self.current_height[-1]:.2f}米 | 误差: {self.error_data[-1]:.2f}米")

        # 更新高度曲线（可视化当前高度和目标高度的偏差，直观反映高度保持效果）
        # 无人机高度保持：核心可视化项，观察当前高度是否跟踪目标高度
        self.line_current.set_data(self.time_data, self.current_height)
        self.line_target.set_data(self.time_data, self.target_height)

        # 更新误差和油门曲线（分析高度误差的变化和油门的控制响应）
        # 无人机高度保持：误差曲线反映控制精度，油门曲线反映控制器的输出特性
        self.line_error.set_data(self.time_data, self.error_data)
        self.line_throttle.set_data(self.time_data, self.throttle_data)

        # 自动调整坐标轴范围，保证数据始终在视图内
        for ax in [self.ax1, self.ax2, self.ax2_twin]:
            ax.relim()
            ax.autoscale_view()

        return self.line_current, self.line_target, self.line_error, self.line_throttle

    def start(self):
        """开始运行，启动无人机高度保持的模拟过程"""
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.status_var.set("启动中...")

    def pause(self):
        """暂停运行，暂停无人机高度保持的模拟过程"""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.status_var.set("已暂停")

    def reset(self):
        """重置数据，清空所有历史数据，恢复初始状态"""
        self.pause()
        self.time_data = []
        self.current_height = []
        self.target_height = []
        self.error_data = []
        self.throttle_data = []
        self.status_var.set("已重置")
        # 刷新绘图
        self._update_plot(0)
        self.canvas.draw()

    def run(self):
        """启动动画，开始无人机高度保持的可视化循环"""
        self.ani = FuncAnimation(
            self.fig, self._update_plot, interval=int(self.dt * 1000),
            blit=True, cache_frame_data=False
        )
        self.root.mainloop()


if __name__ == "__main__":
    # 无人机高度保持：程序入口，创建主窗口和可视化应用实例
    root = tk.Tk()
    app = DroneHeightVisualizer(root)
    app.run()