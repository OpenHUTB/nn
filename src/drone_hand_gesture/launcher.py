# -*- coding: utf-8 -*-
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

from core import ConfigManager, Logger


class Launcher:
    def __init__(self):
        self.config = ConfigManager()
        self.logger = Logger()
        self.root: Optional[tk.Tk] = None

    def show(self):
        self.root = tk.Tk()
        self.root.title("无人机手势控制系统 - 启动器")
        self.root.geometry("550x480")

        title_frame = ttk.Frame(self.root)
        title_frame.pack(pady=20)

        ttk.Label(title_frame, text="🚁", font=("Arial", 48)).pack()
        ttk.Label(title_frame, text="无人机手势控制系统", font=("Arial", 16, "bold")).pack()

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=30, padx=20, fill=tk.BOTH, expand=True)

        style = ttk.Style()
        style.configure("Launch.TButton", font=("Arial", 12))

        ttk.Button(
            button_frame, text="🔧 配置编辑器", style="Launch.TButton", command=self._open_config, width=35).pack(pady=8)
        ttk.Button(
            button_frame, text="🎮 本地仿真模式", style="Launch.TButton", command=self._launch_simulation, width=35).pack(pady=8)
        ttk.Button(
            button_frame, text="🛩️ AirSim仿真模式", style="Launch.TButton", command=self._launch_airsim, width=35).pack(pady=8)
        ttk.Button(
            button_frame, text="📷 AirSim仿真模式 (带无人机摄像头)", style="Launch.TButton", 
            command=self._launch_airsim_camera, width=35).pack(pady=8)

        info_frame = ttk.LabelFrame(self.root, text="信息")
        info_frame.pack(pady=20, padx=20, fill=tk.X)

        ttk.Label(info_frame, text=f"配置文件: {self.config.config_path.absolute()}",
                   ).pack(anchor=tk.W, padx=10, pady=5)
        ttk.Label(info_frame, text=f"日志目录: logs/",
                   ).pack(anchor=tk.W, padx=10, pady=5)

        self.root.mainloop()

    def _open_config(self):
        try:
            from config_ui import ConfigEditor
            editor = ConfigEditor(self.config)
            editor.show()
        except Exception as e:
            self.logger.error(f"打开配置编辑器失败: {e}")
            messagebox.showerror("错误", f"打开配置编辑器失败: {e}")

    def _launch_simulation(self):
        try:
            self.logger.info("启动本地仿真模式 (新架构)...")
            if os.path.exists("main_v2.py"):
                os.system(f"{sys.executable} main_v2.py")
            elif os.path.exists("main.py"):
                self.logger.info("使用旧版 main.py")
                os.system(f"{sys.executable} main.py")
            else:
                messagebox.showinfo("提示", "正在启动仿真模式")
        except Exception as e:
            self.logger.error(f"启动仿真模式失败: {e}")
            messagebox.showerror("错误", f"启动仿真模式失败: {e}")

    def _launch_airsim(self):
        try:
            self.logger.info("启动 AirSim 模式...")
            if os.path.exists("main_airsim.py"):
                os.system(f"{sys.executable} main_airsim.py")
            else:
                messagebox.showinfo("提示", "正在启动 AirSim 模式，请运行 main_airsim.py 文件")
        except Exception as e:
            self.logger.error(f"启动 AirSim 模式失败: {e}")
            messagebox.showerror("错误", f"启动 AirSim 模式失败: {e}")

    def _launch_airsim_camera(self):
        try:
            self.logger.info("启动 AirSim 模式 (带无人机摄像头)...")
            if os.path.exists("main_airsim_camera.py"):
                os.system(f"{sys.executable} main_airsim_camera.py")
            else:
                messagebox.showinfo("提示", "正在启动 AirSim 模式 (带无人机摄像头)，请运行 main_airsim_camera.py 文件")
        except Exception as e:
            self.logger.error(f"启动 AirSim 模式 (带无人机摄像头) 失败: {e}")
            messagebox.showerror("错误", f"启动 AirSim 模式 (带无人机摄像头) 失败: {e}")


def main():
    launcher = Launcher()
    launcher.show()


if __name__ == "__main__":
    main()
