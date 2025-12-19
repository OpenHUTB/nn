import tkinter as tk
import random
import threading
import time

# ------------------- 系统配置常量 -------------------
# 颜色定义
COLOR_NORMAL = "#00ff00"    # 正常-绿色
COLOR_OVERLOAD = "#ff0000"  # 超重-红色
COLOR_BACKGROUND = "#000000"# 背景-黑色
TEXT_SIZE = 40              # 文字大小

# 重量阈值配置
MIN_NORMAL_WEIGHT = 1      # 最小正常重量(kg)
MAX_NORMAL_WEIGHT = 1000   # 最大正常重量(kg)

# 窗口配置
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500

class VehicleOverloadSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Unmanned Vehicle Overload Detection System")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=COLOR_BACKGROUND)

        # 创建显示文字的标签
        self.status_label = tk.Label(
            root,
            text="",
            font=("Arial", TEXT_SIZE, "bold"),
            bg=COLOR_BACKGROUND
        )
        # 居中显示
        self.status_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # 控制循环的标志
        self.running = True

        # 启动重量检测线程
        self.detection_thread = threading.Thread(target=self.check_weight)
        self.detection_thread.daemon = True  # 随主窗口关闭而退出
        self.detection_thread.start()

        # 窗口关闭事件绑定
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def get_vehicle_weight(self):
        """模拟无人车重量传感器数据（可替换为真实硬件读取逻辑）"""
        # 随机生成0~1500kg的重量
        return random.uniform(0, 1500)

    def check_weight(self):
        """循环检测重量并更新界面"""
        while self.running:
            current_weight = self.get_vehicle_weight()
            # 判断重量状态
            if MIN_NORMAL_WEIGHT <= current_weight <= MAX_NORMAL_WEIGHT:
                status_text = "正常"
                text_color = COLOR_NORMAL
            else:
                status_text = "超重"
                text_color = COLOR_OVERLOAD

            # 更新界面（需在主线程中执行）
            self.root.after(0, self.update_label, status_text, text_color)
            # 控制检测频率（每秒1次，可调整）
            time.sleep(1)

    def update_label(self, text, color):
        """更新标签的文字和颜色"""
        self.status_label.config(text=text, fg=color)

    def on_close(self):
        """窗口关闭时的清理操作"""
        self.running = False
        self.root.destroy()
        print("无人车超重检测系统已关闭")

if __name__ == '__main__':
    root = tk.Tk()
    app = VehicleOverloadSystem(root)
    root.mainloop()