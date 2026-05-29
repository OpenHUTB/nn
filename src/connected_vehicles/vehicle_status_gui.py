import tkinter as tk
from tkinter import ttk
import threading

# 全局车况数据
vehicle_status = {
    "speed": 0.0,
    "weather": "clear",
    "visibility": 100.0,
    "collision_speed": 0.0,
    "red_light_violation": False,
    "collision_occurred": False
}

# GUI控制标记
gui_update_flag = True
root_window = None


def create_status_window():
    """创建车况监控窗口（置顶+非阻塞+防卡死）"""
    global root_window
    # 1. 新建GUI线程（彻底隔离CARLA主线程）
    gui_thread = threading.Thread(target=_create_gui, daemon=True)
    gui_thread.start()
    # 等待窗口初始化完成
    while root_window is None:
        pass
    return root_window


def _create_gui():
    """在独立线程中创建GUI（核心修复）"""
    global root_window
    root_window = tk.Tk()

    # 关键设置：窗口置顶+不被CARLA遮挡+可拖动
    root_window.title("车辆实时状态监控")
    root_window.geometry("400x300+20+20")  # 固定位置（左上角20,20），避免被CARLA覆盖
    root_window.resizable(False, False)
    root_window.attributes("-topmost", True)  # 窗口置顶，始终显示在最上层
    root_window.attributes("-toolwindow", True)  # 简化窗口样式，减少卡顿
    root_window.update_idletasks()

    # 设置样式
    style = ttk.Style()
    style.configure("Title.TLabel", font=("Arial", 14, "bold"))
    style.configure("Status.TLabel", font=("Arial", 12))
    style.configure("Warning.TLabel", font=("Arial", 12), foreground="red")

    # 标题
    ttk.Label(root_window, text="🚗 车辆实时状态", style="Title.TLabel").pack(pady=10)

    # 车速
    speed_frame = ttk.Frame(root_window)
    speed_frame.pack(fill="x", padx=20, pady=5)
    ttk.Label(speed_frame, text="当前车速：", style="Status.TLabel").pack(side="left")
    speed_label = ttk.Label(speed_frame, text="0.0 km/h", style="Status.TLabel")
    speed_label.pack(side="left", padx=5)

    # 天气
    weather_frame = ttk.Frame(root_window)
    weather_frame.pack(fill="x", padx=20, pady=5)
    ttk.Label(weather_frame, text="当前天气：", style="Status.TLabel").pack(side="left")
    weather_label = ttk.Label(weather_frame, text="clear", style="Status.TLabel")
    weather_label.pack(side="left", padx=5)

    # 能见度
    visibility_frame = ttk.Frame(root_window)
    visibility_frame.pack(fill="x", padx=20, pady=5)
    ttk.Label(visibility_frame, text="能见度：", style="Status.TLabel").pack(side="left")
    visibility_label = ttk.Label(visibility_frame, text="100%", style="Status.TLabel")
    visibility_label.pack(side="left", padx=5)

    # 碰撞车速
    collision_speed_frame = ttk.Frame(root_window)
    collision_speed_frame.pack(fill="x", padx=20, pady=5)
    ttk.Label(collision_speed_frame, text="碰撞车速：", style="Status.TLabel").pack(side="left")
    collision_speed_label = ttk.Label(collision_speed_frame, text="0.0 km/h", style="Status.TLabel")
    collision_speed_label.pack(side="left", padx=5)

    # 碰撞状态
    collision_frame = ttk.Frame(root_window)
    collision_frame.pack(fill="x", padx=20, pady=5)
    ttk.Label(collision_frame, text="碰撞状态：", style="Status.TLabel").pack(side="left")
    collision_label = ttk.Label(collision_frame, text="未碰撞", style="Status.TLabel")
    collision_label.pack(side="left", padx=5)

    # 闯红灯状态
    red_light_frame = ttk.Frame(root_window)
    red_light_frame.pack(fill="x", padx=20, pady=5)
    ttk.Label(red_light_frame, text="闯红灯：", style="Status.TLabel").pack(side="left")
    red_light_label = ttk.Label(red_light_frame, text="否", style="Status.TLabel")
    red_light_label.pack(side="left", padx=5)

    def update_gui():
        """GUI线程内的循环更新"""
        if gui_update_flag:
            # 更新数据
            speed_label.config(text=f"{vehicle_status['speed']:.1f} km/h")
            weather_label.config(text=vehicle_status['weather'])
            visibility_label.config(text=f"{vehicle_status['visibility']:.0f}%")
            collision_speed_label.config(text=f"{vehicle_status['collision_speed']:.1f} km/h")

            # 碰撞状态
            if vehicle_status['collision_occurred']:
                collision_label.config(text="⚠️ 已碰撞", style="Warning.TLabel")
            else:
                collision_label.config(text="未碰撞", style="Status.TLabel")

            # 闯红灯状态
            if vehicle_status['red_light_violation']:
                red_light_label.config(text="🚨 是", style="Warning.TLabel")
            else:
                red_light_label.config(text="否", style="Status.TLabel")

            # 延迟50ms后再次更新（避免占用过多CPU）
            root_window.after(50, update_gui)

    # 启动GUI更新
    update_gui()

    # 窗口关闭回调
    def on_close():
        stop_gui()

    root_window.protocol("WM_DELETE_WINDOW", on_close)

    # 启动GUI主循环（在独立线程中）
    root_window.mainloop()


def update_vehicle_status(key, value):
    """更新车况数据（线程安全）"""
    if key in vehicle_status:
        vehicle_status[key] = value


def stop_gui():
    """停止GUI并关闭窗口"""
    global gui_update_flag, root_window
    gui_update_flag = False
    if root_window:
        try:
            root_window.quit()  # 退出主循环
            root_window.destroy()  # 销毁窗口
        except:
            pass
    root_window = None