import tkinter as tk
from tkinter import ttk
import threading
import logging
from utils import safe_update_dict
from config import GUI_WINDOW_SIZE, GUI_TITLE, GUI_UPDATE_INTERVAL_MS

logger = logging.getLogger(__name__)

class VehicleStatusGUI:
    def __init__(self):
        # 车况数据
        self.vehicle_status = {
            "speed": 0.0,
            "weather": "clear",
            "visibility": 100.0,
            "collision_speed": 0.0,
            "red_light_violation": False,
            "collision_occurred": False
        }
        self.gui_update_flag = True
        self.root_window = None
        self.gui_thread = None
        self.status_lock = threading.Lock()

    def create_status_window(self) -> None:
        """创建车况监控窗口（独立线程+置顶+非阻塞）"""
        if self.gui_thread and self.gui_thread.is_alive():
            logger.warning("GUI窗口已启动，无需重复创建")
            return

        self.gui_thread = threading.Thread(target=self._create_gui, daemon=True)
        self.gui_thread.start()
        # 等待窗口初始化完成
        while self.root_window is None:
            threading.Event().wait(0.01)
        logger.info("GUI窗口已启动")

    def _create_gui(self) -> None:
        """在独立线程中创建GUI"""
        self.root_window = tk.Tk()
        self.root_window.title(GUI_TITLE)
        self.root_window.geometry(GUI_WINDOW_SIZE)
        self.root_window.resizable(False, False)
        self.root_window.attributes("-topmost", True)
        self.root_window.attributes("-toolwindow", True)
        self.root_window.update_idletasks()

        # 设置样式
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Arial", 14, "bold"))
        style.configure("Status.TLabel", font=("Arial", 12))
        style.configure("Warning.TLabel", font=("Arial", 12), foreground="red")

        # 构建UI组件
        self._build_ui(style)

        # 启动GUI更新循环
        self._update_gui()

        # 窗口关闭回调
        self.root_window.protocol("WM_DELETE_WINDOW", self.stop)

        # 启动GUI主循环
        self.root_window.mainloop()

    def _build_ui(self, style: ttk.Style) -> None:
        """构建GUI界面组件"""
        # 标题
        ttk.Label(self.root_window, text="🚗 车辆实时状态", style="Title.TLabel").pack(pady=10)

        # 车速
        speed_frame = ttk.Frame(self.root_window)
        speed_frame.pack(fill="x", padx=20, pady=5)
        ttk.Label(speed_frame, text="当前车速：", style="Status.TLabel").pack(side="left")
        self.speed_label = ttk.Label(speed_frame, text="0.0 km/h", style="Status.TLabel")
        self.speed_label.pack(side="left", padx=5)

        # 天气
        weather_frame = ttk.Frame(self.root_window)
        weather_frame.pack(fill="x", padx=20, pady=5)
        ttk.Label(weather_frame, text="当前天气：", style="Status.TLabel").pack(side="left")
        self.weather_label = ttk.Label(weather_frame, text="clear", style="Status.TLabel")
        self.weather_label.pack(side="left", padx=5)

        # 能见度
        visibility_frame = ttk.Frame(self.root_window)
        visibility_frame.pack(fill="x", padx=20, pady=5)
        ttk.Label(visibility_frame, text="能见度：", style="Status.TLabel").pack(side="left")
        self.visibility_label = ttk.Label(visibility_frame, text="100%", style="Status.TLabel")
        self.visibility_label.pack(side="left", padx=5)

        # 碰撞车速
        collision_speed_frame = ttk.Frame(self.root_window)
        collision_speed_frame.pack(fill="x", padx=20, pady=5)
        ttk.Label(collision_speed_frame, text="碰撞车速：", style="Status.TLabel").pack(side="left")
        self.collision_speed_label = ttk.Label(collision_speed_frame, text="0.0 km/h", style="Status.TLabel")
        self.collision_speed_label.pack(side="left", padx=5)

        # 碰撞状态
        collision_frame = ttk.Frame(self.root_window)
        collision_frame.pack(fill="x", padx=20, pady=5)
        ttk.Label(collision_frame, text="碰撞状态：", style="Status.TLabel").pack(side="left")
        self.collision_label = ttk.Label(collision_frame, text="未碰撞", style="Status.TLabel")
        self.collision_label.pack(side="left", padx=5)

        # 闯红灯状态
        red_light_frame = ttk.Frame(self.root_window)
        red_light_frame.pack(fill="x", padx=20, pady=5)
        ttk.Label(red_light_frame, text="闯红灯：", style="Status.TLabel").pack(side="left")
        self.red_light_label = ttk.Label(red_light_frame, text="否", style="Status.TLabel")
        self.red_light_label.pack(side="left", padx=5)

    def _update_gui(self) -> None:
        """GUI循环更新（线程安全）"""
        if not self.gui_update_flag:
            return

        with self.status_lock:
            # 更新数据
            self.speed_label.config(text=f"{self.vehicle_status['speed']:.1f} km/h")
            self.weather_label.config(text=self.vehicle_status['weather'])
            self.visibility_label.config(text=f"{self.vehicle_status['visibility']:.0f}%")
            self.collision_speed_label.config(text=f"{self.vehicle_status['collision_speed']:.1f} km/h")

            # 碰撞状态
            if self.vehicle_status['collision_occurred']:
                self.collision_label.config(text="⚠️ 已碰撞", style="Warning.TLabel")
            else:
                self.collision_label.config(text="未碰撞", style="Status.TLabel")

            # 闯红灯状态
            if self.vehicle_status['red_light_violation']:
                self.red_light_label.config(text="🚨 是", style="Warning.TLabel")
            else:
                self.red_light_label.config(text="否", style="Status.TLabel")

        # 延迟后再次更新
        self.root_window.after(GUI_UPDATE_INTERVAL_MS, self._update_gui)

    def update_vehicle_status(self, key: str, value) -> None:
        """线程安全更新车况数据"""
        safe_update_dict(self.vehicle_status, key, value)

    def stop(self) -> None:
        """停止GUI并销毁窗口"""
        self.gui_update_flag = False
        if self.root_window:
            try:
                self.root_window.quit()
                self.root_window.destroy()
            except Exception as e:
                logger.error(f"关闭GUI窗口失败：{e}")
        self.root_window = None
        logger.info("GUI窗口已关闭")

# 全局实例
gui_instance = VehicleStatusGUI()
create_status_window = gui_instance.create_status_window
update_vehicle_status = gui_instance.update_vehicle_status
stop_gui = gui_instance.stop
vehicle_status = gui_instance.vehicle_status