# 注释掉carla导入
# import carla
import tkinter as tk
from tkinter import ttk
import random
import threading
import time

class CarlaBatteryMeter:
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Vehicle Battery Gauge")
        self.root.geometry("400x300")
        self.root.resizable(False, False)

        self.battery_capacity = 100
        self.current_battery = 85
        self.lock = threading.Lock()

        # 注释掉Carla初始化
        # self.carla_client = None
        # self.carla_vehicle = None
        # self.init_carla()

        self.create_battery_ui()

        self.running = True
        self.battery_thread = threading.Thread(target=self.update_battery_from_carla, daemon=True)
        self.battery_thread.start()

        self.update_battery_display()

    # 注释掉Carla初始化函数
    # def init_carla(self):
    #     ...

    def create_battery_ui(self):
        title_label = ttk.Label(self.root, text="Autonomous Vehicle Battery Gauge", font=("Microsoft YaHei", 16))
        title_label.pack(pady=20)

        self.battery_label = tk.Label(
            self.root,
            text=str(self.current_battery),
            font=("Microsoft YaHei", 48, "bold"),
            bg="white"
        )
        self.battery_label.pack(pady=30, fill=tk.BOTH, expand=True)

        desc_label = ttk.Label(
            self.root,
            text="Battery Range: 70~100(Green) | 20~69(Yellow) | 3~19(Red) | 0~2(Low Power)",
            font=("Microsoft YaHei", 10)
        )
        desc_label.pack(pady=10)

    def update_battery_from_carla(self):
        while self.running:
            with self.lock:
                # 直接使用模拟数据，注释掉Carla相关逻辑
                # if self.carla_vehicle:
                #     ...
                # else:
                if random.random() > 0.4:
                    self.current_battery = max(0, self.current_battery - 1)
                else:
                    self.current_battery = min(self.battery_capacity, self.current_battery + 1)
            time.sleep(1)

    def update_battery_display(self):
        with self.lock:
            current = self.current_battery

        if 70 <= current <= 100:
            text = str(current)
            color = "#00cc00"
        elif 20 <= current < 70:
            text = str(current)
            color = "#ffcc00"
        elif 3 <= current < 20:
            text = str(current)
            color = "#ff3300"
        else:
            text = "Low Power"
            color = "#ff0000"

        self.battery_label.config(text=text, fg=color)
        self.root.after(200, self.update_battery_display)

    def stop(self):
        self.running = False
        # 注释掉Carla车辆销毁逻辑
        # if self.carla_vehicle:
        #     self.carla_vehicle.destroy()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = CarlaBatteryMeter(root)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()