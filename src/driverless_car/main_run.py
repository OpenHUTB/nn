import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import random

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


#
# 1. 数据生成与预处理（模拟无人机飞行数据）
class DroneSpeedDataset(Dataset):
    """无人机速度控制数据集（模拟数据）"""

    def __init__(self, num_samples=10000, dt=0.1):
        self.dt = dt  # 采样时间间隔（秒）
        self.data = self._generate_data(num_samples)

    def _generate_data(self, num_samples):
        """生成模拟数据：输入为速度误差、误差变化率，输出为最优控制量"""
        data = []
        # 模拟不同目标速度（-5~5 m/s，无人机水平速度范围）
        target_speeds = np.random.uniform(-5, 5, num_samples)
        # 初始速度
        current_speeds = np.random.uniform(-3, 3, num_samples)

        for v_target, v_current in zip(target_speeds, current_speeds):
            # 计算误差
            error = v_target - v_current
            # 误差变化率（简化：假设上一时刻误差为当前误差的0.8倍）
            error_dot = (error - 0.8 * error) / self.dt
            # 用PID公式生成"最优"控制量（作为标签）
            # 控制量范围：0~1（归一化的电机输出）
            kp, ki, kd = 0.1, 0.01, 0.05
            control = 0.5 + kp * error + kd * error_dot
            control = np.clip(control, 0.0, 1.0)  # 限制范围

            # 加入噪声（模拟真实环境）
            error += np.random.normal(0, 0.1)
            error_dot += np.random.normal(0, 0.05)

            data.append((error, error_dot, control))
        return np.array(data, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, :2]  # 输入：误差、误差变化率
        y = self.data[idx, 2:]  # 输出：控制量
        return torch.tensor(x), torch.tensor(y)


# 2. 深度学习控制器模型
class SpeedController(nn.Module):
    """基于MLP的速度控制器"""

    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
        super(SpeedController, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # 输出控制量（0~1）
        )

    def forward(self, x):
        return self.network(x)


# 3. 模型训练函数
def train_controller(num_epochs=50, batch_size=64):
    # 生成数据集
    dataset = DroneSpeedDataset(num_samples=20000)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = SpeedController()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练过程
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], 平均损失: {avg_loss:.6f}")

    # 保存模型
    torch.save(model.state_dict(), "speed_controller.pth")
    return model, loss_history


# 4. 无人机速度控制仿真
class DroneSimulator:
    """无人机速度控制仿真器（简化物理模型）"""

    def __init__(self, dt=0.1):
        self.dt = dt  # 采样时间
        self.current_speed = 0.0  # 当前速度（m/s）
        self.mass = 1.5  # 无人机质量（kg）
        self.drag_coef = 0.2  # 阻力系数
        self.max_thrust = 5.0  # 最大推力（N）

    def update(self, control):
        """根据控制量更新速度"""
        # 控制量（0~1）映射到推力
        thrust = control * self.max_thrust
        # 合力 = 推力 - 阻力（阻力与速度平方成正比）
        drag = self.drag_coef * self.current_speed ** 2 * np.sign(self.current_speed)
        force = thrust - drag
        # 加速度 = 合力 / 质量
        acceleration = force / self.mass
        # 更新速度（加入噪声模拟风干扰）
        self.current_speed += acceleration * self.dt + np.random.normal(0, 0.05)
        # 限制最大速度（-8~8 m/s）
        self.current_speed = np.clip(self.current_speed, -8, 8)
        return self.current_speed


# 5. 可视化控制界面
class SpeedControlVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("无人机速度控制可视化系统")
        self.root.geometry("1000x700")

        # 控制参数
        self.dt = 0.1  # 控制频率（10Hz）
        self.target_speed = 2.0  # 初始目标速度（m/s）
        self.running = False
        self.data = {
            "time": [],
            "current_speed": [],
            "target_speed": [],
            "error": [],
            "control": []
        }
        self.max_points = 200  # 最大显示数据点

        # 初始化组件
        self._init_ui()
        self._init_plots()

        # 初始化模型和仿真器
        self.model = None
        self.simulator = DroneSimulator(dt=self.dt)

    def _init_ui(self):
        """创建UI控件"""
        # 控制面板
        control_frame = ttk.LabelFrame(self.root, text="控制参数")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # 目标速度设置
        ttk.Label(control_frame, text="目标速度 (m/s):").grid(row=0, column=0, padx=5, pady=5)
        self.target_entry = ttk.Entry(control_frame, width=10)
        self.target_entry.insert(0, str(self.target_speed))
        self.target_entry.grid(row=0, column=1, padx=5, pady=5)

        # 模型控制按钮
        self.train_btn = ttk.Button(control_frame, text="训练模型", command=self.train_model)
        self.train_btn.grid(row=0, column=2, padx=5, pady=5)

        self.start_btn = ttk.Button(control_frame, text="开始控制", command=self.start_control, state=tk.DISABLED)
        self.start_btn.grid(row=0, column=3, padx=5, pady=5)

        self.pause_btn = ttk.Button(control_frame, text="暂停", command=self.pause_control, state=tk.DISABLED)
        self.pause_btn.grid(row=0, column=4, padx=5, pady=5)

        # 状态显示
        self.status_var = tk.StringVar(value="就绪 | 模型状态: 未训练")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=0, column=5, padx=20, pady=5)

    def _init_plots(self):
        """初始化绘图区域"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 速度曲线（上半部分）
        self.line_current, = self.ax1.plot([], [], label="当前速度", color='blue', linewidth=2)
        self.line_target, = self.ax1.plot([], [], label="目标速度", color='red', linestyle='--', linewidth=1.5)
        self.ax1.set_ylabel("速度 (m/s)")
        self.ax1.set_title("无人机速度跟踪曲线")
        self.ax1.grid(True)
        self.ax1.legend()
        self.ax1.set_ylim(-6, 6)  # 速度范围

        # 误差和控制量曲线（下半部分）
        self.line_error, = self.ax2.plot([], [], label="速度误差", color='green', linewidth=2)
        self.ax2_twin = self.ax2.twinx()  # 双Y轴
        self.line_control, = self.ax2_twin.plot([], [], label="控制量", color='orange', linewidth=2, alpha=0.7)
        self.ax2.set_xlabel("时间 (秒)")
        self.ax2.set_ylabel("误差 (m/s)")
        self.ax2_twin.set_ylabel("控制量 (0~1)")
        self.ax2.set_title("速度误差与控制量变化")
        self.ax2.grid(True)
        self.ax2.set_ylim(-4, 4)  # 误差范围
        self.ax2_twin.set_ylim(0, 1)  # 控制量范围
        # 合并图例
        lines = [self.line_error, self.line_control]
        self.ax2.legend(lines, [l.get_label() for l in lines], loc='upper left')

        self.fig.tight_layout()

    def train_model(self):
        """训练深度学习控制器"""
        self.status_var.set("开始训练模型...")
        self.root.update()

        try:
            # 训练模型
            self.model, loss_history = train_controller(num_epochs=50)
            self.status_var.set("模型训练完成")
            self.start_btn.config(state=tk.NORMAL)

            # 显示训练损失曲线
            self._plot_loss_curve(loss_history)
        except Exception as e:
            messagebox.showerror("训练失败", str(e))
            self.status_var.set(f"训练失败: {str(e)}")

    def _plot_loss_curve(self, loss_history):
        """显示训练损失曲线"""
        # 创建新窗口显示损失曲线
        loss_window = tk.Toplevel(self.root)
        loss_window.title("训练损失曲线")
        loss_window.geometry("600x400")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(loss_history) + 1), loss_history, color='purple')
        ax.set_title("模型训练损失")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE损失")
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=loss_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def start_control(self):
        """开始速度控制"""
        # 更新目标速度
        try:
            self.target_speed = float(self.target_entry.get())
        except ValueError:
            messagebox.showerror("输入错误", "目标速度必须是数字")
            return

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.status_var.set(f"控制中 | 目标速度: {self.target_speed} m/s")

        # 重置仿真器
        self.simulator = DroneSimulator(dt=self.dt)
        # 启动控制循环
        self._control_loop()

    def pause_control(self):
        """暂停控制"""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.status_var.set(f"已暂停 | 当前速度: {self.data['current_speed'][-1]:.2f} m/s")

    def _control_loop(self):
        """控制循环（实时更新速度和控制量）"""
        if not self.running:
            return

        # 获取当前时间
        t = self.data["time"][-1] + self.dt if self.data["time"] else 0.0

        # 1. 计算当前状态
        current_speed = self.simulator.current_speed
        error = self.target_speed - current_speed
        # 计算误差变化率
        error_dot = (error - self.data["error"][-1]) / self.dt if self.data["error"] else 0.0

        # 2. 深度学习模型计算控制量
        model_input = torch.tensor([[error, error_dot]], dtype=torch.float32)
        with torch.no_grad():
            control = self.model(model_input).item()  # 输出0~1

        # 3. 更新无人机速度
        new_speed = self.simulator.update(control)

        # 4. 保存数据
        self.data["time"].append(t)
        self.data["current_speed"].append(new_speed)
        self.data["target_speed"].append(self.target_speed)
        self.data["error"].append(error)
        self.data["control"].append(control)

        # 限制数据长度
        if len(self.data["time"]) > self.max_points:
            for key in self.data:
                self.data[key].pop(0)

        # 5. 更新绘图
        self._update_plots()

        # 6. 循环调用（模拟实时控制）
        self.root.after(int(self.dt * 1000), self._control_loop)

    def _update_plots(self):
        """更新图表"""
        # 更新速度曲线
        self.line_current.set_data(self.data["time"], self.data["current_speed"])
        self.line_target.set_data(self.data["time"], self.data["target_speed"])

        # 更新误差和控制量曲线
        self.line_error.set_data(self.data["time"], self.data["error"])
        self.line_control.set_data(self.data["time"], self.data["control"])

        # 自动调整X轴范围
        for ax in [self.ax1, self.ax2, self.ax2_twin]:
            ax.set_xlim(max(0, self.data["time"][-1] - 20), self.data["time"][-1] + 1)  # 显示最近20秒数据

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedControlVisualizer(root)
    root.mainloop()