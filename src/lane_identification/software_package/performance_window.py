"""
性能监控窗口 - 独立的性能数据显示和管理
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox


class PerformanceWindow:
    """性能监控窗口类"""

    def __init__(self, parent, performance_monitor, video_processor=None):
        self.parent = parent
        self.performance_monitor = performance_monitor
        self.video_processor = video_processor
        self.window = None
        self.update_timer = None

    def show(self):
        """显示性能监控窗口"""
        if self.window is not None and self.window.winfo_exists():
            self.window.deiconify()
            self.window.lift()
            return

        self.window = tk.Toplevel(self.parent)
        self.window.title("性能监控")
        self.window.geometry("600x500")
        self.window.transient(self.parent)

        # 创建界面
        self._create_ui()

        # 开始定时更新
        self._start_update()

    def _create_ui(self):
        """创建窗口界面"""
        # 创建选项卡
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # 实时监控标签页
        realtime_frame = ttk.Frame(notebook)
        notebook.add(realtime_frame, text="实时监控")

        self.perf_text = scrolledtext.ScrolledText(
            realtime_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            height=20
        )
        self.perf_text.pack(fill="both", expand=True, padx=10, pady=10)

        # 配置标签页
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="配置查看")

        self.config_text = scrolledtext.ScrolledText(
            config_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            height=20
        )
        self.config_text.pack(fill="both", expand=True, padx=10, pady=10)

        # 按钮框架
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(
            button_frame,
            text="刷新",
            command=self.update_display
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            button_frame,
            text="重置统计",
            command=self.reset_stats
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            button_frame,
            text="保存配置",
            command=self.save_config
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            button_frame,
            text="关闭",
            command=self.window.destroy
        ).pack(side="right")

    def update_display(self, config_dict=None):
        """更新显示内容"""
        if self.window is None or not self.window.winfo_exists():
            return

        # 更新性能数据
        self._update_performance_data()

        # 更新配置数据
        if config_dict:
            self._update_config_data(config_dict)

    def _update_performance_data(self):
        """更新性能数据"""
        summary = self.performance_monitor.get_summary()

        self.perf_text.delete(1.0, tk.END)
        self.perf_text.insert(tk.END, "=== 性能监控数据 ===\n\n")

        for metric, stats in summary.items():
            self.perf_text.insert(tk.END, f"{metric}:\n")
            self.perf_text.insert(tk.END, f"  平均值: {stats['avg']:.4f}s\n")
            self.perf_text.insert(tk.END, f"  最小值: {stats['min']:.4f}s\n")
            self.perf_text.insert(tk.END, f"  最大值: {stats['max']:.4f}s\n")
            self.perf_text.insert(tk.END, f"  标准差: {stats['std']:.4f}s\n")
            self.perf_text.insert(tk.END, f"  95%分位: {stats['p95']:.4f}s\n")
            self.perf_text.insert(tk.END, f"  样本数: {stats['count']}\n\n")

        # 添加视频处理信息
        if self.video_processor and hasattr(self.video_processor, 'is_playing'):
            if self.video_processor.is_playing:
                video_info = self.video_processor.get_video_info()
                self.perf_text.insert(tk.END, "=== 视频处理信息 ===\n\n")
                for key, value in video_info.items():
                    if isinstance(value, float):
                        self.perf_text.insert(tk.END, f"{key}: {value:.3f}\n")
                    else:
                        self.perf_text.insert(tk.END, f"{key}: {value}\n")

    def _update_config_data(self, config_dict):
        """更新配置数据"""
        self.config_text.delete(1.0, tk.END)
        self.config_text.insert(tk.END, "=== 当前配置 ===\n\n")
        for key, value in config_dict.items():
            self.config_text.insert(tk.END, f"{key}: {value}\n")

    def _start_update(self):
        """开始定时更新"""
        if self.window is not None and self.window.winfo_exists():
            self.update_display()
            self.update_timer = self.parent.after(2000, self._start_update)

    def reset_stats(self):
        """重置性能统计"""
        self.performance_monitor.reset()
        self.update_display()

    def save_config(self, config_manager=None):
        """保存当前配置"""
        if config_manager:
            if config_manager.save_current_config():
                messagebox.showinfo("成功", "配置已保存")
                # 重新加载配置显示
                config_dict = config_manager.config.to_dict()
                self._update_config_data(config_dict)
        else:
            messagebox.showwarning("警告", "配置管理器未提供")

    def destroy(self):
        """销毁窗口"""
        if self.update_timer:
            self.parent.after_cancel(self.update_timer)
        if self.window and self.window.winfo_exists():
            self.window.destroy()
