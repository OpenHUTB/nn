# 注释掉carla导入（因当前版本使用模拟数据，暂不依赖Carla引擎）
# import carla
# 导入tkinter库用于创建图形用户界面（GUI）
import tkinter as tk
from tkinter import ttk
# 导入random库用于生成模拟电池数据的随机数
import random
# 导入threading库用于创建独立线程更新电池数据，避免阻塞GUI主线程
import threading
# 导入time库用于控制电池数据更新的时间间隔
import time

class CarlaBatteryMeter:
    """
    自动驾驶车辆电池电量显示仪表类
    功能：模拟获取电池电量数据，并在GUI界面实时显示电量及对应状态颜色
    """
    def __init__(self, root):
        """
        类的初始化方法，完成界面初始化、数据初始化和线程启动
        :param root: tkinter的主窗口对象
        """
        # 保存主窗口对象
        self.root = root
        # 设置窗口标题
        self.root.title("Autonomous Vehicle Battery Gauge")
        # 设置窗口大小（宽度400px，高度300px）
        self.root.geometry("400x300")
        # 禁止窗口大小调整
        self.root.resizable(False, False)

        # 电池总容量（百分比）
        self.battery_capacity = 100
        # 当前电池电量（初始值设为85%）
        self.current_battery = 85
        # 创建线程锁，用于保护多线程下的电量数据安全（防止读写冲突）
        self.lock = threading.Lock()

        # 注释掉Carla初始化（当前使用模拟数据，无需连接Carla引擎）
        # self.carla_client = None
        # self.carla_vehicle = None
        # self.init_carla()

        # 创建电池显示的GUI界面
        self.create_battery_ui()

        # 线程运行状态标识（用于控制后台线程的启停）
        self.running = True
        # 创建后台线程，用于模拟从Carla获取电池数据并更新（守护线程：主程序退出时自动结束）
        self.battery_thread = threading.Thread(target=self.update_battery_from_carla, daemon=True)
        # 启动后台线程
        self.battery_thread.start()

        # 启动电量显示的更新循环（GUI界面实时刷新）
        self.update_battery_display()

    # 注释掉Carla初始化函数（当前使用模拟数据，无需连接Carla引擎）
    # def init_carla(self):
    #     ...

    def create_battery_ui(self):
        """创建电池电量显示的GUI界面元素，包括标题、电量数字、状态说明"""
        # 创建标题标签（字体：微软雅黑，大小16）
        title_label = ttk.Label(self.root, text="Autonomous Vehicle Battery Gauge", font=("Microsoft YaHei", 16))
        # 布局：垂直方向间距20px
        title_label.pack(pady=20)

        # 创建电量显示的主标签（初始显示当前电量，字体：微软雅黑，大小48，加粗，背景白色）
        self.battery_label = tk.Label(
            self.root,
            text=str(self.current_battery),
            font=("Microsoft YaHei", 48, "bold"),
            bg="white"
        )
        # 布局：垂直方向间距30px，填充水平和垂直方向，允许扩展
        self.battery_label.pack(pady=30, fill=tk.BOTH, expand=True)

        # 创建状态说明标签（说明不同电量区间的颜色标识）
        desc_label = ttk.Label(
            self.root,
            text="Battery Range: 70~100(Green) | 20~69(Yellow) | 3~19(Red) | 0~2(Low Power)",
            font=("Microsoft YaHei", 10)
        )
        # 布局：垂直方向间距10px
        desc_label.pack(pady=10)

    def update_battery_from_carla(self):
        """
        后台线程的执行函数：模拟从Carla引擎获取电池电量数据并更新
        逻辑：随机生成电量变化（60%概率减1%，40%概率加1%），保证电量在0~100之间
        """
        # 循环执行，直到running标识为False
        while self.running:
            # 加锁：保护current_battery数据，避免多线程读写冲突
            with self.lock:
                # 直接使用模拟数据，注释掉Carla相关逻辑（原逻辑为从Carla车辆获取电量）
                # if self.carla_vehicle:
                #     ...
                # else:
                # 生成0~1的随机数，控制电量增减
                if random.random() > 0.4:
                    # 电量减1，最低为0
                    self.current_battery = max(0, self.current_battery - 1)
                else:
                    # 电量加1，最高为电池总容量（100）
                    self.current_battery = min(self.battery_capacity, self.current_battery + 1)
            # 暂停1秒，控制电量更新频率（每秒更新一次）
            time.sleep(1)

    def update_battery_display(self):
        """
        实时更新GUI界面的电量显示：根据当前电量设置不同的文字和颜色
        使用tkinter的after方法实现循环刷新（每200ms刷新一次）
        """
        # 加锁：安全读取当前电量数据
        with self.lock:
            current = self.current_battery

        # 根据电量区间设置显示文本和颜色
        if 70 <= current <= 100:
            # 高电量：绿色（#00cc00）
            text = str(current)
            color = "#00cc00"
        elif 20 <= current < 70:
            # 中等电量：黄色（#ffcc00）
            text = str(current)
            color = "#ffcc00"
        elif 3 <= current < 20:
            # 低电量：红色（#ff3300）
            text = str(current)
            color = "#ff3300"
        else:
            # 极低电量：深红色（#ff0000），显示"Low Power"
            text = "Low Power"
            color = "#ff0000"

        # 更新电量标签的文本和文字颜色
        self.battery_label.config(text=text, fg=color)
        # 200ms后再次调用自身，实现GUI的循环刷新
        self.root.after(200, self.update_battery_display)

    def stop(self):
        """
        程序停止时的清理操作：停止后台线程，退出GUI主循环
        绑定到窗口关闭事件（WM_DELETE_WINDOW）
        """
        # 设置线程运行状态为False，终止后台线程的循环
        self.running = False
        # 注释掉Carla车辆销毁逻辑（当前未使用Carla，无需销毁车辆）
        # if self.carla_vehicle:
        #     self.carla_vehicle.destroy()
        # 退出tkinter的主循环，关闭窗口
        self.root.quit()

# 程序入口
if __name__ == "__main__":
    # 创建tkinter的主窗口对象
    root = tk.Tk()
    # 实例化电池仪表类
    app = CarlaBatteryMeter(root)
    # 绑定窗口关闭事件：点击右上角关闭按钮时执行stop方法
    root.protocol("WM_DELETE_WINDOW", app.stop)
    # 启动tkinter的主循环（阻塞，直到窗口关闭）
    root.mainloop()