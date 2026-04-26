# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional, Callable

# 获取当前脚本所在目录
BASE_DIR = Path(__file__).resolve().parent

from core import ConfigManager, Logger


class ConfigEditor:
    def __init__(self, config: Optional[ConfigManager] = None, on_save: Optional[Callable] = None):
        self.config = config or ConfigManager()
        self.logger = Logger()
        self.on_save = on_save
        self.root: Optional[tk.Tk] = None
        self.entries: dict = {}

    def show(self):
        self.root = tk.Tk()
        self.root.title("无人机参数配置")
        self.root.geometry("600x700")

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        drone_frame = ttk.Frame(notebook)
        notebook.add(drone_frame, text="无人机")
        self._create_drone_tab(drone_frame)

        camera_frame = ttk.Frame(notebook)
        notebook.add(camera_frame, text="摄像头")
        self._create_camera_tab(camera_frame)

        gesture_frame = ttk.Frame(notebook)
        notebook.add(gesture_frame, text="手势识别")
        self._create_gesture_tab(gesture_frame)

        simulation_frame = ttk.Frame(notebook)
        notebook.add(simulation_frame, text="仿真")
        self._create_simulation_tab(simulation_frame)

        airsim_frame = ttk.Frame(notebook)
        notebook.add(airsim_frame, text="AirSim")
        self._create_airsim_tab(airsim_frame)

        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        save_btn = ttk.Button(button_frame, text="保存配置", command=self._save_config)
        save_btn.pack(side=tk.RIGHT, padx=5)

        reset_btn = ttk.Button(button_frame, text="重置默认", command=self._reset_config)
        reset_btn.pack(side=tk.RIGHT, padx=5)

        self._load_current_values()

        self.root.mainloop()

    def _create_drone_tab(self, parent):
        ttk.Label(parent, text="最大速度 (m/s):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('drone', 'max_speed')] = entry

        ttk.Label(parent, text="最大高度 (m):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('drone', 'max_altitude')] = entry

        ttk.Label(parent, text="起飞高度 (m):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('drone', 'takeoff_altitude')] = entry

        ttk.Label(parent, text="悬停阈值:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('drone', 'hover_threshold')] = entry

        ttk.Label(parent, text="电池消耗率:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('drone', 'battery_drain_rate')] = entry

        ttk.Label(parent, text="质量 (kg):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=5, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('drone', 'mass')] = entry

        ttk.Label(parent, text="重力加速度 (m/s²):").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=6, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('drone', 'gravity')] = entry

        parent.columnconfigure(1, weight=1)

    def _create_camera_tab(self, parent):
        ttk.Label(parent, text="默认摄像头ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('camera', 'default_id')] = entry

        ttk.Label(parent, text="宽度:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('camera', 'width')] = entry

        ttk.Label(parent, text="高度:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('camera', 'height')] = entry

        ttk.Label(parent, text="FPS:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('camera', 'fps')] = entry

        parent.columnconfigure(1, weight=1)

    def _create_gesture_tab(self, parent):
        ttk.Label(parent, text="置信度阈值:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('gesture', 'threshold')] = entry

        ttk.Label(parent, text="命令冷却时间 (s):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('gesture', 'command_cooldown')] = entry

        parent.columnconfigure(1, weight=1)

    def _create_simulation_tab(self, parent):
        ttk.Label(parent, text="窗口宽度:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('simulation', 'window_width')] = entry

        ttk.Label(parent, text="窗口高度:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('simulation', 'window_height')] = entry

        ttk.Label(parent, text="目标FPS:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('simulation', 'target_fps')] = entry

        ttk.Label(parent, text="最大轨迹点数:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('simulation', 'max_trajectory_points')] = entry

        parent.columnconfigure(1, weight=1)

    def _create_airsim_tab(self, parent):
        ttk.Label(parent, text="IP地址:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('airsim', 'ip_address')] = entry

        ttk.Label(parent, text="端口:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('airsim', 'port')] = entry

        ttk.Label(parent, text="飞行器名称:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent)
        entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self.entries[('airsim', 'vehicle_name')] = entry

        parent.columnconfigure(1, weight=1)

    def _load_current_values(self):
        for (section, key), entry in self.entries.items():
            value = self.config.get(f"{section}.{key}")
            if value is not None:
                entry.delete(0, tk.END)
                entry.insert(0, str(value))

    def _save_config(self):
        try:
            for (section, key), entry in self.entries.items():
                value_str = entry.get().strip()
                if not value_str:
                    continue

                default_value = self.config.get(f"{section}.{key}")

                if isinstance(default_value, int):
                    value = int(value_str)
                elif isinstance(default_value, float):
                    value = float(value_str)
                else:
                    value = value_str

                self.config.set(f"{section}.{key}", value)

            self.config.save_config()
            self.logger.info("配置保存成功！")

            if self.on_save:
                self.on_save()

            messagebox.showinfo("成功", "配置已保存！")

        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            messagebox.showerror("错误", f"保存配置失败: {e}")

    def _reset_config(self):
        if messagebox.askyesno("确认", "确定要重置为默认配置吗？"):
            self.config.reset()
            self._load_current_values()
            self.logger.info("配置已重置为默认")


if __name__ == "__main__":
    editor = ConfigEditor()
    editor.show()
