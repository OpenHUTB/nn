import tkinter as tk
from tkinter import ttk
import random
import time

class UAVTemperatureSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("无人车温度调节系统仿真")
        self.root.geometry("500x400")

        # 温度参数初始化
        self.current_temp = 25.0  # 初始温度
        self.target_temp = 25.0   # 目标温度
        self.max_temp = 35.0      # 温度上限
        self.min_temp = 15.0      # 温度下限

        # 创建UI组件
        self.create_widgets()

        # 启动温度监测循环
        self.update_temp()

    def create_widgets(self):
        # 标题标签
        ttk.Label(self.root, text="无人车温度调节系统", font=("Arial", 16, "bold")).pack(pady=10)

        # 当前温度显示
        self.temp_label = ttk.Label(self.root, text=f"当前温度: {self.current_temp:.1f} °C", font=("Arial", 14))
        self.temp_label.pack(pady=5)

        # 目标温度设置
        ttk.Label(self.root, text="设置目标温度:").pack(pady=2)
        self.target_entry = ttk.Entry(self.root, width=10)
        self.target_entry.insert(0, "25")
        self.target_entry.pack(pady=2)
        ttk.Button(self.root, text="确认设置", command=self.set_target_temp).pack(pady=5)

        # 系统状态显示
        self.status_label = ttk.Label(self.root, text="系统状态: 待机", font=("Arial", 12), foreground="blue")
        self.status_label.pack(pady=10)

        # 温度曲线画布（简易模拟）
        self.canvas = tk.Canvas(self.root, width=400, height=150, bg="white")
        self.canvas.pack(pady=10)
        self.canvas.create_line(10, 75, 390, 75, fill="black")  # 基准线
        self.canvas.create_text(200, 10, text="温度变化趋势", font=("Arial", 10))

        self.x_pos = 10  # 曲线绘制横坐标

    def set_target_temp(self):
        try:
            self.target_temp = float(self.target_entry.get())
            self.status_label.config(text=f"目标温度已设为: {self.target_temp:.1f} °C", foreground="green")
        except ValueError:
            self.status_label.config(text="输入无效！请输入数字", foreground="red")

    def update_temp(self):
        # 模拟温度波动（无人车运行时的温度变化）
        self.current_temp += random.uniform(-0.5, 0.8)
        self.current_temp = round(self.current_temp, 1)

        # 更新温度显示
        self.temp_label.config(text=f"当前温度: {self.current_temp:.1f} °C")

        # 判断温度状态并执行调节逻辑
        if self.current_temp > self.max_temp:
            self.status_label.config(text="状态: 温度过高 → 启动制冷系统", foreground="red")
            self.current_temp -= 1.2  # 制冷降温
        elif self.current_temp < self.min_temp:
            self.status_label.config(text="状态: 温度过低 → 启动加热系统", foreground="orange")
            self.current_temp += 1.2  # 加热升温
        elif abs(self.current_temp - self.target_temp) > 1.0:
            if self.current_temp > self.target_temp:
                self.status_label.config(text="状态: 高于目标 → 轻度制冷", foreground="blue")
                self.current_temp -= 0.5
            else:
                self.status_label.config(text="状态: 低于目标 → 轻度加热", foreground="blue")
        else:
            self.status_label.config(text="状态: 温度正常 → 系统待机", foreground="green")

        # 绘制温度变化曲线
        y_pos = 75 - (self.current_temp - 25) * 5  # 映射温度到画布坐标
        if self.x_pos < 390:
            self.canvas.create_line(self.x_pos, 75, self.x_pos + 1, y_pos, fill="red", width=2)
            self.x_pos += 1
        else:
            # 曲线满屏后重置
            self.canvas.delete("all")
            self.canvas.create_line(10, 75, 390, 75, fill="black")
            self.x_pos = 10

        # 每隔1秒更新一次
        self.root.after(1000, self.update_temp)

if __name__ == "__main__":
    root = tk.Tk()
    app = UAVTemperatureSystem(root)
    root.mainloop()