# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional

# 获取当前脚本所在目录
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from core import ConfigManager, Logger


class Launcher:
    def __init__(self):
        self.config = ConfigManager()
        self.logger = Logger()
        self.root: Optional[tk.Tk] = None

    def show(self):
        self.root = tk.Tk()
        self.root.title("无人机手势控制系统 - 启动器")
        self.root.geometry("550x500")

        title_frame = ttk.Frame(self.root)
        title_frame.pack(pady=20)

        ttk.Label(title_frame, text="🚁", font=("Arial", 48)).pack()
        ttk.Label(title_frame, text="无人机手势控制系统", font=("Arial", 16, "bold")).pack()

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        style = ttk.Style()
        style.configure("Launch.TButton", font=("Arial", 11))
        style.configure("LaunchSmall.TButton", font=("Arial", 10))

        ttk.Button(
            button_frame, text="🔧 配置编辑器", style="Launch.TButton", 
            command=self._open_config, width=35).pack(pady=8)
        
        ttk.Separator(button_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(button_frame, text="本地仿真", font=("Arial", 10, "bold")).pack(pady=5)
        ttk.Button(
            button_frame, text="🎮 新版仿真 (main_v2.py)", style="LaunchSmall.TButton", 
            command=self._launch_simulation, width=35).pack(pady=5)
        ttk.Button(
            button_frame, text="📺 旧版仿真 (main.py)", style="LaunchSmall.TButton", 
            command=self._launch_old_simulation, width=35).pack(pady=5)
        
        ttk.Separator(button_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(button_frame, text="AirSim", font=("Arial", 10, "bold")).pack(pady=5)
        ttk.Button(
            button_frame, text="🛩️ AirSim 模式", style="LaunchSmall.TButton", 
            command=self._launch_airsim, width=35).pack(pady=5)

        info_frame = ttk.LabelFrame(self.root, text="信息")
        info_frame.pack(pady=15, padx=20, fill=tk.X)

        ttk.Label(info_frame, text=f"配置文件: {self.config.config_path.absolute()}",
                   ).pack(anchor=tk.W, padx=10, pady=3)
        ttk.Label(info_frame, text=f"日志目录: {SCRIPT_DIR / 'logs'}",
                   ).pack(anchor=tk.W, padx=10, pady=3)

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
            # 使用相对于脚本目录的路径
            main_v2_path = SCRIPT_DIR / "main_v2.py"
            
            if main_v2_path.exists():
                os.chdir(SCRIPT_DIR)
                subprocess.run([sys.executable, "main_v2.py"])
            else:
                messagebox.showinfo("提示", "未找到 main_v2.py")
        except Exception as e:
            self.logger.error(f"启动仿真模式失败: {e}")
            messagebox.showerror("错误", f"启动仿真模式失败: {e}")

    def _launch_old_simulation(self):
        try:
            self.logger.info("启动本地仿真模式 (旧版)...")
            main_path = SCRIPT_DIR / "main.py"
            
            if main_path.exists():
                os.chdir(SCRIPT_DIR)
                subprocess.run([sys.executable, "main.py"])
            else:
                messagebox.showinfo("提示", "未找到 main.py")
        except Exception as e:
            self.logger.error(f"启动旧版仿真失败: {e}")
            messagebox.showerror("错误", f"启动旧版仿真失败: {e}")

    def _launch_airsim(self):
        try:
            self.logger.info("启动 AirSim 模式...")
            # 使用相对于脚本目录的路径
            main_airsim_path = SCRIPT_DIR / "main_airsim.py"
            
            if main_airsim_path.exists():
                os.chdir(SCRIPT_DIR)
                subprocess.run([sys.executable, "main_airsim.py"])
            else:
                messagebox.showinfo("提示", "未找到 main_airsim.py")
        except Exception as e:
            self.logger.error(f"启动 AirSim 模式失败: {e}")
            messagebox.showerror("错误", f"启动 AirSim 模式失败: {e}")


def main():
    launcher = Launcher()
    launcher.show()


if __name__ == "__main__":
    main()
