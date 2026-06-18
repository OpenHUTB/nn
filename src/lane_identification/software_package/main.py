"""
主应用程序模块 - 支持图像、视频和摄像头实时识别
添加异常恢复、性能监控和配置热重载
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import threading
import time
import traceback
from collections import deque
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys
<<<<<<< HEAD

# 导入各个模块
from config import AppConfig, SceneConfig, config_manager
from image_processor import SmartImageProcessor, RoadDetector
from lane_detector import LaneDetector
from direction_analyzer import DirectionAnalyzer
from visualizer import Visualizer
from video_processor import VideoProcessor
from utils import PerformanceMonitor, Timer, safe_resize, calculate_fps
=======
import json
import csv
from datetime import datetime

# 导入各个模块
from config import AppConfig, SceneConfig, config_manager
from detection_service import DetectionService
from video_processor import VideoProcessor
from utils import PerformanceMonitor, Timer, safe_resize, calculate_fps
from confidence_calibrator import ConfidenceCalibrator
from quality_evaluator import QualityEvaluator
from batch_processor import BatchProcessor
from export_manager import ExportManager
from performance_window import PerformanceWindow
>>>>>>> upstream/main


class LaneDetectionApp:
    """道路方向识别系统主应用程序 - 优化版本"""
<<<<<<< HEAD
    
    def __init__(self, root):
        self.root = root
        self._setup_window()
        
        # 初始化配置管理器
        self.config_manager = config_manager
        self.config = self.config_manager.config
        
        # 注册配置变更回调
        self.config_manager.add_change_callback(self._on_config_changed, "main_app")
        
        # 启动配置监控
        self.config_manager.start_watching(interval=2.0)
        
        # 性能监控器
        self.performance_monitor = PerformanceMonitor(max_samples=100)
        
=======

    def __init__(self, root):
        self.root = root
        self._setup_window()

        # 初始化配置管理器
        self.config_manager = config_manager
        self.config = self.config_manager.config

        # 注册配置变更回调
        self.config_manager.add_change_callback(self._on_config_changed, "main_app")

        # 启动配置监控
        self.config_manager.start_watching(interval=2.0)

        # 性能监控器
        self.performance_monitor = PerformanceMonitor(max_samples=100)

>>>>>>> upstream/main
        # 错误处理
        self.error_count = 0
        self.last_error_time = 0
        self.recovery_mode = False
        self.max_errors = self.config.max_error_count
        self.recovery_timeout = self.config.recovery_timeout
<<<<<<< HEAD
        
        # 初始化各个模块
        self._initialize_modules()
        
=======

        # 初始化检测服务（替换原来的多个模块初始化）
        self.detection_service = DetectionService(self.config)
        self.video_processor = VideoProcessor(self.config)
        self.confidence_calibrator = ConfidenceCalibrator()
        self.quality_evaluator = QualityEvaluator()
        self.batch_processor = BatchProcessor(self.detection_service)
        self.export_manager = ExportManager()

>>>>>>> upstream/main
        # 状态变量
        self.current_image = None
        self.current_image_path = None
        self.is_processing = False
        self.is_video_mode = False
        self.processing_history = deque(maxlen=10)
<<<<<<< HEAD
        
=======

        # 导出相关状态
        self.last_detection_result = None
        self.result_image = None

>>>>>>> upstream/main
        # 视频相关变量
        self.video_file_path = None
        self.camera_mode = False
        self.camera_index = 0
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 性能统计
        self.processing_times = []
        self.frame_counter = 0
        self.last_fps_update = time.time()
        self.current_fps = 0
<<<<<<< HEAD
        
        # 创建界面
        self._create_ui()
        
        # 创建性能监控窗口（隐藏）
        self._create_performance_window()
        
        print("道路方向识别系统已启动（优化版本）")
        print(f"配置: {self.config.to_dict()}")
    
    def _initialize_modules(self):
        """初始化各个模块"""
        try:
            self.image_processor = SmartImageProcessor(self.config)
            self.road_detector = RoadDetector(self.config)
            self.lane_detector = LaneDetector(self.config)
            self.direction_analyzer = DirectionAnalyzer(self.config)
            self.visualizer = Visualizer(self.config)
            self.video_processor = VideoProcessor(self.config)
            print("所有模块初始化完成")
        except Exception as e:
            print(f"模块初始化失败: {e}")
            self._recover_from_error()
    
=======

        # 创建界面
        self._create_ui()

        # 创建性能监控窗口管理器
        self.perf_window_manager = PerformanceWindow(
            self.root,
            self.performance_monitor,
            self.video_processor
        )

        print("道路方向识别系统已启动（优化版本）")
        print(f"配置: {self.config.to_dict()}")

>>>>>>> upstream/main
    def _recover_from_error(self):
        """从错误中恢复"""
        if self.recovery_mode:
            return
<<<<<<< HEAD
            
        self.recovery_mode = True
        print("进入恢复模式...")
        
=======

        self.recovery_mode = True
        print("进入恢复模式...")

>>>>>>> upstream/main
        try:
            # 延迟后重新初始化
            self.root.after(1000, self._perform_recovery)
        except Exception as e:
            print(f"恢复失败: {e}")
<<<<<<< HEAD
    
    def _perform_recovery(self):
        """执行恢复操作"""
        try:
            # 重新初始化模块
            self._initialize_modules()
            
            # 重置状态
            self.error_count = 0
            self.recovery_mode = False
            
            # 更新状态显示
            self.status_var.set("系统已恢复")
            print("系统恢复完成")
            
            # 如果是视频模式，尝试重新开始
            if self.is_video_mode and self.video_file_path:
                self._open_video(self.video_file_path)
                
=======

    def _perform_recovery(self):
        """执行恢复操作"""
        try:
            # 重新初始化检测服务
            self.detection_service = DetectionService(self.config)

            # 重置状态
            self.error_count = 0
            self.recovery_mode = False

            # 更新状态显示
            self.status_var.set("系统已恢复")
            print("系统恢复完成")

            # 如果是视频模式，尝试重新开始
            if self.is_video_mode and self.video_file_path:
                self._open_video(self.video_file_path)

>>>>>>> upstream/main
        except Exception as e:
            print(f"恢复操作失败: {e}")
            # 如果恢复失败，等待更长时间后重试
            self.root.after(5000, self._perform_recovery)
<<<<<<< HEAD
    
=======

>>>>>>> upstream/main
    def _on_config_changed(self, new_config):
        """配置变更回调"""
        print("检测到配置变更，重新初始化模块...")
        self.config = new_config
<<<<<<< HEAD
        
        # 重新初始化模块
        self._initialize_modules()
        
        # 更新UI
        self._update_ui_from_config()
        
        # 重新检测当前图像
        if self.current_image_path and not self.is_processing and not self.is_video_mode:
            self.root.after(500, self._redetect)
    
=======

        # 更新检测服务配置
        self.detection_service.update_config(new_config)

        # 更新UI
        self._update_ui_from_config()

        # 重新检测当前图像
        if self.current_image_path and not self.is_processing and not self.is_video_mode:
            self.root.after(500, self._redetect)

>>>>>>> upstream/main
    def _update_ui_from_config(self):
        """根据配置更新UI"""
        # 更新敏感度滑块
        if hasattr(self, 'sensitivity_var'):
            # 从配置计算敏感度值（0-1范围）
            sensitivity = (self.config.canny_threshold1 - 30) / 40.0
            sensitivity = max(0.0, min(1.0, sensitivity))
            self.sensitivity_var.set(sensitivity)
<<<<<<< HEAD
    
=======

>>>>>>> upstream/main
    def _setup_window(self):
        """设置窗口"""
        self.root.title("道路方向识别系统 - 优化版本")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 设置窗口图标（如果有）
        try:
            if os.path.exists("icon.ico"):
                self.root.iconbitmap("icon.ico")
        except:
            pass
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 设置窗口居中
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
<<<<<<< HEAD
        
        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
=======

        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

>>>>>>> upstream/main
    def _create_ui(self):
        """创建用户界面"""
        # 主容器
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
<<<<<<< HEAD
        
        # 标题栏
        self._create_title_bar(main_container)
        
        # 内容区域
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        # 左侧控制面板
        control_frame = self._create_control_panel(content_frame)
        control_frame.pack(side="left", fill="y", padx=(0, 10))
        
        # 右侧图像显示区域
        display_frame = self._create_display_panel(content_frame)
        display_frame.pack(side="right", fill="both", expand=True)
        
        # 状态栏
        self._create_status_bar(main_container)
        
        # 性能监控按钮
        self._create_performance_button(main_container)
    
=======

        # 标题栏
        self._create_title_bar(main_container)

        # 内容区域
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill="both", expand=True, pady=(10, 0))

        # 左侧控制面板
        control_frame = self._create_control_panel(content_frame)
        control_frame.pack(side="left", fill="y", padx=(0, 10))

        # 右侧图像显示区域
        display_frame = self._create_display_panel(content_frame)
        display_frame.pack(side="right", fill="both", expand=True)

        # 状态栏
        self._create_status_bar(main_container)

        # 性能监控按钮
        self._create_performance_button(main_container)

>>>>>>> upstream/main
    def _create_title_bar(self, parent):
        """创建标题栏"""
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill="x", pady=(0, 10))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 标题
        title_label = ttk.Label(
            title_frame,
            text="道路方向识别系统 - 优化版本",
            font=("微软雅黑", 16, "bold"),
            foreground="#2c3e50"
        )
        title_label.pack(side="left")
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 模式指示器
        self.mode_label = ttk.Label(
            title_frame,
            text="[图像模式]",
            font=("微软雅黑", 10),
            foreground="#3498db"
        )
        self.mode_label.pack(side="right", padx=(0, 10))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 配置状态指示器
        self.config_status_label = ttk.Label(
            title_frame,
            text="✓",
            font=("微软雅黑", 10),
            foreground="#27ae60"
        )
        self.config_status_label.pack(side="right", padx=(0, 10))
<<<<<<< HEAD
    
    def _create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(
            parent,
=======

    def _create_control_panel(self, parent):
        """创建控制面板"""
        # 创建外部容器
        control_container = ttk.Frame(parent)
        control_container.pack_propagate(False)
        control_container.config(width=350)

        # 创建Canvas用于滚动
        canvas = tk.Canvas(control_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=canvas.yview)

        # 创建内部框架（控制面板）
        control_frame = ttk.LabelFrame(
            canvas,
>>>>>>> upstream/main
            text="控制面板",
            padding="15",
            relief="groove"
        )
<<<<<<< HEAD
        control_frame.pack_propagate(False)
        control_frame.config(width=350)
        
        # 输入模式选择
        mode_frame = ttk.LabelFrame(control_frame, text="输入模式", padding="10")
        mode_frame.pack(fill="x", pady=(0, 15))
        
        # 模式选择按钮
        mode_buttons_frame = ttk.Frame(mode_frame)
        mode_buttons_frame.pack()
        
=======

        # 配置滚动
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas_window = canvas.create_window((0, 0), window=control_frame, anchor="nw")

        # 绑定事件以更新滚动区域
        def _configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        control_frame.bind("<Configure>", _configure_scroll_region)

        # 布局Canvas和滚动条
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 鼠标滚轮支持
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # 输入模式选择
        mode_frame = ttk.LabelFrame(control_frame, text="输入模式", padding="10")
        mode_frame.pack(fill="x", pady=(0, 15))

        # 模式选择按钮
        mode_buttons_frame = ttk.Frame(mode_frame)
        mode_buttons_frame.pack()

>>>>>>> upstream/main
        self.image_mode_btn = ttk.Button(
            mode_buttons_frame,
            text="图像模式",
            command=self._switch_to_image_mode,
            width=12
        )
        self.image_mode_btn.pack(side="left", padx=(0, 5))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        self.video_mode_btn = ttk.Button(
            mode_buttons_frame,
            text="视频模式",
            command=self._switch_to_video_mode,
            width=12
        )
        self.video_mode_btn.pack(side="left", padx=(0, 5))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        self.camera_mode_btn = ttk.Button(
            mode_buttons_frame,
            text="摄像头模式",
            command=self._switch_to_camera_mode,
            width=12
        )
        self.camera_mode_btn.pack(side="left")
<<<<<<< HEAD
        
        # 文件操作区域
        self.file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding="10")
        self.file_frame.pack(fill="x", pady=(0, 15))
        
=======

        # 文件操作区域
        self.file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding="10")
        self.file_frame.pack(fill="x", pady=(0, 15))

>>>>>>> upstream/main
        # 选择图片按钮
        self.select_image_btn = ttk.Button(
            self.file_frame,
            text="选择图片",
            command=self._select_image,
            width=20
        )
        self.select_image_btn.pack(pady=(0, 10))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 重新检测按钮
        self.redetect_btn = ttk.Button(
            self.file_frame,
            text="重新检测",
            command=self._redetect,
            width=20,
            state="disabled"
        )
        self.redetect_btn.pack(pady=(0, 10))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 文件信息显示
        self.file_info_label = ttk.Label(
            self.file_frame,
            text="未选择图片",
            wraplength=300,
            foreground="#3498db"
        )
        self.file_info_label.pack()
<<<<<<< HEAD
        
        # 视频控制区域（初始隐藏）
        self.video_frame = ttk.LabelFrame(control_frame, text="视频控制", padding="10")
        
=======

        # 视频控制区域（初始隐藏）
        self.video_frame = ttk.LabelFrame(control_frame, text="视频控制", padding="10")

>>>>>>> upstream/main
        # 选择视频按钮
        self.select_video_btn = ttk.Button(
            self.video_frame,
            text="选择视频文件",
            command=self._select_video,
            width=20
        )
        self.select_video_btn.pack(pady=(0, 10))
<<<<<<< HEAD
        
        # 摄像头索引选择
        camera_frame = ttk.Frame(self.video_frame)
        camera_frame.pack(fill="x", pady=(0, 10))
        
=======

        # 摄像头索引选择
        camera_frame = ttk.Frame(self.video_frame)
        camera_frame.pack(fill="x", pady=(0, 10))

>>>>>>> upstream/main
        ttk.Label(camera_frame, text="摄像头索引:").pack(side="left")
        self.camera_index_var = tk.StringVar(value=str(self.config.camera_id))
        self.camera_index_combo = ttk.Combobox(
            camera_frame,
            textvariable=self.camera_index_var,
            values=["0", "1", "2", "3"],
            state="readonly",
            width=8
        )
        self.camera_index_combo.pack(side="left", padx=(5, 0))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 摄像头测试按钮
        self.camera_test_btn = ttk.Button(
            self.video_frame,
            text="测试摄像头",
            command=self._test_camera,
            width=20
        )
        self.camera_test_btn.pack(pady=(0, 10))
<<<<<<< HEAD
        
        # 视频控制按钮
        self.video_control_frame = ttk.Frame(self.video_frame)
        self.video_control_frame.pack()
        
=======

        # 视频控制按钮
        self.video_control_frame = ttk.Frame(self.video_frame)
        self.video_control_frame.pack()

>>>>>>> upstream/main
        self.play_btn = ttk.Button(
            self.video_control_frame,
            text="开始",
            command=self._play_video,
            width=8,
            state="disabled"
        )
        self.play_btn.pack(side="left", padx=(0, 5))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        self.pause_btn = ttk.Button(
            self.video_control_frame,
            text="暂停",
            command=self._pause_video,
            width=8,
            state="disabled"
        )
        self.pause_btn.pack(side="left", padx=(0, 5))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        self.stop_btn = ttk.Button(
            self.video_control_frame,
            text="停止",
            command=self._stop_video,
            width=8,
            state="disabled"
        )
        self.stop_btn.pack(side="left")
<<<<<<< HEAD
        
        # 高级设置区域
        advanced_frame = ttk.LabelFrame(control_frame, text="高级设置", padding="10")
        advanced_frame.pack(fill="x", pady=(0, 15))
        
=======

        # 高级设置区域
        advanced_frame = ttk.LabelFrame(control_frame, text="高级设置", padding="10")
        advanced_frame.pack(fill="x", pady=(0, 15))

>>>>>>> upstream/main
        # 性能优化选项
        self.enable_buffer_var = tk.BooleanVar(value=self.config.enable_frame_buffer)
        buffer_check = ttk.Checkbutton(
            advanced_frame,
            text="启用帧缓冲",
            variable=self.enable_buffer_var,
            command=self._on_advanced_option_change
        )
        buffer_check.pack(anchor="w", pady=(0, 5))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        self.adaptive_skip_var = tk.BooleanVar(value=self.config.adaptive_skip_frames)
        skip_check = ttk.Checkbutton(
            advanced_frame,
            text="自适应跳帧",
            variable=self.adaptive_skip_var,
            command=self._on_advanced_option_change
        )
        skip_check.pack(anchor="w", pady=(0, 10))
<<<<<<< HEAD
        
        # 参数调节区域
        param_frame = ttk.LabelFrame(control_frame, text="参数调节", padding="10")
        param_frame.pack(fill="x", pady=(0, 15))
        
=======

        # 参数调节区域
        param_frame = ttk.LabelFrame(control_frame, text="参数调节", padding="10")
        param_frame.pack(fill="x", pady=(0, 15))

>>>>>>> upstream/main
        # 敏感度调节
        ttk.Label(param_frame, text="检测敏感度:").pack(anchor="w", pady=(0, 5))
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        sensitivity_scale = ttk.Scale(
            param_frame,
            from_=0.1,
            to=1.0,
            variable=self.sensitivity_var,
            orient="horizontal",
            command=self._on_parameter_change,
            length=300
        )
        sensitivity_scale.pack(fill="x", pady=(0, 10))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 场景选择
        ttk.Label(param_frame, text="场景模式:").pack(anchor="w", pady=(0, 5))
        self.scene_var = tk.StringVar(value="auto")
        scene_combo = ttk.Combobox(
            param_frame,
            textvariable=self.scene_var,
            values=["自动", "高速公路", "城市道路", "乡村道路"],
            state="readonly",
            width=20
        )
        scene_combo.pack(fill="x", pady=(0, 10))
        scene_combo.bind("<<ComboboxSelected>>", self._on_scene_change)
<<<<<<< HEAD
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(control_frame, text="检测结果", padding="10")
        result_frame.pack(fill="x")
        
=======

        # 导出操作区域（移到参数调节下面）
        export_frame = ttk.LabelFrame(control_frame, text="导出选项", padding="10")
        export_frame.pack(fill="x", pady=(0, 15))

        # 导出按钮行
        export_buttons_frame = ttk.Frame(export_frame)
        export_buttons_frame.pack(fill="x")

        self.export_image_btn = ttk.Button(
            export_buttons_frame,
            text="导出图片",
            command=self._export_result_image,
            width=10,
            state="disabled"
        )
        self.export_image_btn.pack(side="left", padx=(0, 5), pady=(0, 5))

        self.export_json_btn = ttk.Button(
            export_buttons_frame,
            text="导出JSON",
            command=self._export_json_report,
            width=10,
            state="disabled"
        )
        self.export_json_btn.pack(side="left", padx=(0, 5), pady=(0, 5))

        self.export_csv_btn = ttk.Button(
            export_buttons_frame,
            text="导出CSV",
            command=self._export_csv_report,
            width=10,
            state="disabled"
        )
        self.export_csv_btn.pack(side="left", pady=(0, 5))

        # 批量导出按钮
        self.batch_export_btn = ttk.Button(
            export_frame,
            text="批量导出文件夹",
            command=self._batch_export_folder,
            width=20
        )
        self.batch_export_btn.pack(pady=(5, 0))

        # 结果显示区域
        result_frame = ttk.LabelFrame(control_frame, text="检测结果", padding="10")
        result_frame.pack(fill="x")

>>>>>>> upstream/main
        # 方向显示
        self.direction_label = ttk.Label(
            result_frame,
            text="等待检测...",
            font=("微软雅黑", 14, "bold"),
            foreground="#2c3e50"
        )
        self.direction_label.pack(anchor="w", pady=(0, 5))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 置信度显示
        self.confidence_label = ttk.Label(
            result_frame,
            text="",
            font=("微软雅黑", 11)
        )
        self.confidence_label.pack(anchor="w", pady=(0, 5))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 检测质量显示
        self.quality_label = ttk.Label(
            result_frame,
            text="",
            font=("微软雅黑", 10),
            foreground="#7f8c8d"
        )
        self.quality_label.pack(anchor="w", pady=(0, 5))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 处理时间/FPS显示
        self.time_label = ttk.Label(
            result_frame,
            text="",
            font=("微软雅黑", 9),
            foreground="#95a5a6"
        )
        self.time_label.pack(anchor="w")
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 性能信息显示
        self.performance_label = ttk.Label(
            result_frame,
            text="",
            font=("微软雅黑", 8),
            foreground="#bdc3c7"
        )
        self.performance_label.pack(anchor="w", pady=(5, 0))
<<<<<<< HEAD
        
        return control_frame
    
    def _create_display_panel(self, parent):
        """创建显示面板"""
        display_frame = ttk.Frame(parent)
        
        # 图像显示区域
        images_frame = ttk.Frame(display_frame)
        images_frame.pack(fill="both", expand=True)
        
=======

        return control_container

    def _create_display_panel(self, parent):
        """创建显示面板"""
        display_frame = ttk.Frame(parent)

        # 图像显示区域
        images_frame = ttk.Frame(display_frame)
        images_frame.pack(fill="both", expand=True)

>>>>>>> upstream/main
        # 原图显示
        original_frame = ttk.LabelFrame(
            images_frame,
            text="原始图像",
            padding="5",
            relief="groove"
        )
        original_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        self.original_canvas = tk.Canvas(
            original_frame,
            bg="#ecf0f1",
            highlightthickness=1,
            highlightbackground="#bdc3c7"
        )
        self.original_canvas.pack(fill="both", expand=True)
        self.original_canvas.create_text(
            300, 200,
            text="请选择输入源",
            font=("微软雅黑", 12),
            fill="#7f8c8d"
        )
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 结果图显示
        result_frame = ttk.LabelFrame(
            images_frame,
            text="检测结果",
            padding="5",
            relief="groove"
        )
        result_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        self.result_canvas = tk.Canvas(
            result_frame,
            bg="#ecf0f1",
            highlightthickness=1,
            highlightbackground="#bdc3c7"
        )
        self.result_canvas.pack(fill="both", expand=True)
        self.result_canvas.create_text(
            300, 200,
            text="检测结果将显示在这里",
            font=("微软雅黑", 12),
            fill="#7f8c8d"
        )
<<<<<<< HEAD
        
        return display_frame
    
=======

        return display_frame

>>>>>>> upstream/main
    def _create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent, relief="sunken", borderwidth=1)
        status_frame.pack(fill="x", pady=(10, 0))
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 进度条
        self.progress_bar = ttk.Progressbar(
            status_frame,
            mode='indeterminate',
            length=200
        )
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(5, 10), pady=5)
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 状态文本
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("微软雅黑", 9)
        )
        status_label.pack(side="right", padx=(0, 10), pady=5)
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 错误计数显示
        self.error_label = ttk.Label(
            status_frame,
            text="错误: 0",
            font=("微软雅黑", 9),
            foreground="#e74c3c"
        )
        self.error_label.pack(side="right", padx=(0, 20), pady=5)
<<<<<<< HEAD
    
=======

>>>>>>> upstream/main
    def _create_performance_button(self, parent):
        """创建性能监控按钮"""
        perf_button = ttk.Button(
            parent,
            text="性能监控",
            command=self._show_performance_window,
            width=10
        )
        perf_button.pack(side="right", padx=(0, 10), pady=5)
<<<<<<< HEAD
    
    def _create_performance_window(self):
        """创建性能监控窗口"""
        self.perf_window = None
        
    def _show_performance_window(self):
        """显示性能监控窗口"""
        if self.perf_window is not None and self.perf_window.winfo_exists():
            self.perf_window.deiconify()
            self.perf_window.lift()
            return
            
        self.perf_window = tk.Toplevel(self.root)
        self.perf_window.title("性能监控")
        self.perf_window.geometry("600x500")
        self.perf_window.transient(self.root)
        
        # 创建选项卡
        notebook = ttk.Notebook(self.perf_window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 实时监控标签页
        realtime_frame = ttk.Frame(notebook)
        notebook.add(realtime_frame, text="实时监控")
        
        # 性能指标显示
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
        
        # 更新配置显示
        self._update_config_display()
        
        # 按钮框架
        button_frame = ttk.Frame(self.perf_window)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ttk.Button(
            button_frame,
            text="刷新",
            command=self._update_performance_display
        ).pack(side="left", padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="重置统计",
            command=self._reset_performance_stats
        ).pack(side="left", padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="保存配置",
            command=self._save_current_config
        ).pack(side="left", padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="关闭",
            command=self.perf_window.destroy
        ).pack(side="right")
        
        # 开始定时更新
        self._start_performance_update()
    
    def _update_performance_display(self):
        """更新性能显示"""
        if self.perf_window is None or not self.perf_window.winfo_exists():
            return
            
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
        if self.is_video_mode:
            video_info = self.video_processor.get_video_info()
            self.perf_text.insert(tk.END, "=== 视频处理信息 ===\n\n")
            for key, value in video_info.items():
                if isinstance(value, float):
                    self.perf_text.insert(tk.END, f"{key}: {value:.3f}\n")
                else:
                    self.perf_text.insert(tk.END, f"{key}: {value}\n")
    
    def _update_config_display(self):
        """更新配置显示"""
        if self.perf_window is None or not self.perf_window.winfo_exists():
            return
            
        self.config_text.delete(1.0, tk.END)
        config_dict = self.config.to_dict()
        
        self.config_text.insert(tk.END, "=== 当前配置 ===\n\n")
        for key, value in config_dict.items():
            self.config_text.insert(tk.END, f"{key}: {value}\n")
    
    def _start_performance_update(self):
        """开始性能数据更新"""
        if self.perf_window is not None and self.perf_window.winfo_exists():
            self._update_performance_display()
            self.root.after(2000, self._start_performance_update)
    
    def _reset_performance_stats(self):
        """重置性能统计"""
        self.performance_monitor.reset()
        self._update_performance_display()
    
    def _save_current_config(self):
        """保存当前配置"""
        if self.config_manager.save_current_config():
            messagebox.showinfo("成功", "配置已保存")
            self._update_config_display()
    
=======

    def _show_performance_window(self):
        """显示性能监控窗口"""
        config_dict = self.config.to_dict()
        self.perf_window_manager.show()
        self.perf_window_manager.update_display(config_dict)

>>>>>>> upstream/main
    def _on_advanced_option_change(self):
        """高级选项变更"""
        updates = {
            'enable_frame_buffer': self.enable_buffer_var.get(),
            'adaptive_skip_frames': self.adaptive_skip_var.get()
        }
<<<<<<< HEAD
        
        self.config_manager.update_config(updates, save_to_file=True)
    
=======

        self.config_manager.update_config(updates, save_to_file=True)

>>>>>>> upstream/main
    def _switch_to_image_mode(self):
        """切换到图像模式"""
        if self.is_video_mode:
            self._stop_video()
<<<<<<< HEAD
        
        self.is_video_mode = False
        self.camera_mode = False
        self.mode_label.config(text="[图像模式]", foreground="#3498db")
        
        # 显示图像控制，隐藏视频控制
        self.file_frame.pack(fill="x", pady=(0, 15))
        self.video_frame.pack_forget()
        
        # 更新按钮状态
        self.select_image_btn.config(state="normal")
        self.redetect_btn.config(state="normal" if self.current_image_path else "disabled")
        
        self.status_var.set("已切换到图像模式")
    
=======

        self.is_video_mode = False
        self.camera_mode = False
        self.mode_label.config(text="[图像模式]", foreground="#3498db")

        # 显示图像控制，隐藏视频控制
        self.file_frame.pack(fill="x", pady=(0, 15))
        self.video_frame.pack_forget()

        # 更新按钮状态
        self.select_image_btn.config(state="normal")
        self.redetect_btn.config(state="normal" if self.current_image_path else "disabled")

        self.status_var.set("已切换到图像模式")

>>>>>>> upstream/main
    def _switch_to_video_mode(self):
        """切换到视频模式"""
        if self.is_video_mode and self.camera_mode:
            self._stop_video()
<<<<<<< HEAD
        
        self.is_video_mode = True
        self.camera_mode = False
        self.mode_label.config(text="[视频模式]", foreground="#e74c3c")
        
        # 隐藏图像控制，显示视频控制
        self.file_frame.pack_forget()
        self.video_frame.pack(fill="x", pady=(0, 15))
        
=======

        self.is_video_mode = True
        self.camera_mode = False
        self.mode_label.config(text="[视频模式]", foreground="#e74c3c")

        # 隐藏图像控制，显示视频控制
        self.file_frame.pack_forget()
        self.video_frame.pack(fill="x", pady=(0, 15))

>>>>>>> upstream/main
        # 更新按钮状态
        self.select_video_btn.config(state="normal")
        self.play_btn.config(state="disabled")
        self.pause_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        self.camera_test_btn.config(state="normal")
<<<<<<< HEAD
        
        self.status_var.set("已切换到视频模式")
    
=======

        self.status_var.set("已切换到视频模式")

>>>>>>> upstream/main
    def _switch_to_camera_mode(self):
        """切换到摄像头模式"""
        try:
            if self.is_video_mode:
                self._stop_video()
<<<<<<< HEAD
            
            self.is_video_mode = True
            self.camera_mode = True
            self.mode_label.config(text="[摄像头模式]", foreground="#9b59b6")
            
            # 隐藏图像控制，显示视频控制
            self.file_frame.pack_forget()
            self.video_frame.pack(fill="x", pady=(0, 15))
            
=======

            self.is_video_mode = True
            self.camera_mode = True
            self.mode_label.config(text="[摄像头模式]", foreground="#9b59b6")

            # 隐藏图像控制，显示视频控制
            self.file_frame.pack_forget()
            self.video_frame.pack(fill="x", pady=(0, 15))

>>>>>>> upstream/main
            # 更新按钮状态
            self.select_video_btn.config(state="normal")
            self.play_btn.config(state="disabled")
            self.pause_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.camera_test_btn.config(state="normal")
<<<<<<< HEAD
            
            # 显示摄像头预览
            self._show_camera_preview()
            
            self.status_var.set("已切换到摄像头模式")
            
        except Exception as e:
            print(f"切换摄像头模式失败: {e}")
            self.status_var.set("切换摄像头模式失败")
    
=======

            # 显示摄像头预览
            self._show_camera_preview()

            self.status_var.set("已切换到摄像头模式")

        except Exception as e:
            print(f"切换摄像头模式失败: {e}")
            self.status_var.set("切换摄像头模式失败")

>>>>>>> upstream/main
    def _show_camera_preview(self):
        """显示摄像头预览"""
        try:
            # 获取摄像头索引
            try:
                self.camera_index = int(self.camera_index_var.get())
            except:
                self.camera_index = self.config.camera_id
<<<<<<< HEAD
            
            # 尝试打开摄像头
            self.status_var.set(f"正在打开摄像头 {self.camera_index}...")
            self.root.update()
            
=======

            # 尝试打开摄像头
            self.status_var.set(f"正在打开摄像头 {self.camera_index}...")
            self.root.update()

>>>>>>> upstream/main
            if self.video_processor.open_camera(self.camera_index):
                # 获取预览帧
                ret, frame = self.video_processor.get_frame()
                if ret:
                    self._display_image(frame, self.original_canvas, "摄像头预览")
<<<<<<< HEAD
                    
=======

>>>>>>> upstream/main
                    # 在结果区域显示提示
                    self.result_canvas.delete("all")
                    self.result_canvas.create_text(
                        300, 200,
                        text="摄像头已就绪，点击'开始'按钮进行实时检测",
                        font=("微软雅黑", 12),
                        fill="#3498db"
                    )
<<<<<<< HEAD
                    
=======

>>>>>>> upstream/main
                    self.play_btn.config(state="normal")
                    self.status_var.set(f"摄像头 {self.camera_index} 已就绪")
                else:
                    self.status_var.set("无法获取摄像头画面")
            else:
                self.status_var.set("无法打开摄像头")
<<<<<<< HEAD
                
        except Exception as e:
            print(f"显示摄像头预览失败: {e}")
            self.status_var.set("摄像头预览失败")
    
=======

        except Exception as e:
            print(f"显示摄像头预览失败: {e}")
            self.status_var.set("摄像头预览失败")

>>>>>>> upstream/main
    def _test_camera(self):
        """测试摄像头功能"""
        try:
            # 创建测试窗口
            test_window = tk.Toplevel(self.root)
            test_window.title("摄像头测试")
            test_window.geometry("500x400")
            test_window.transient(self.root)
<<<<<<< HEAD
            
            # 创建标签
            label = ttk.Label(test_window, text="正在测试摄像头...", font=("微软雅黑", 10))
            label.pack(pady=20)
            
            # 创建文本显示区域
            result_text = tk.Text(test_window, height=15, width=50, font=("微软雅黑", 9))
            result_text.pack(pady=10, padx=20)
            
=======

            # 创建标签
            label = ttk.Label(test_window, text="正在测试摄像头...", font=("微软雅黑", 10))
            label.pack(pady=20)

            # 创建文本显示区域
            result_text = tk.Text(test_window, height=15, width=50, font=("微软雅黑", 9))
            result_text.pack(pady=10, padx=20)

>>>>>>> upstream/main
            # 开始测试按钮
            def start_test():
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, "=== 摄像头测试结果 ===\n\n")
<<<<<<< HEAD
                
                # 测试不同的摄像头索引
                for i in range(4):  # 测试0-3号摄像头
                    result_text.insert(tk.END, f"测试摄像头 {i}:\n")
                    
                    try:
                        # 尝试打开摄像头
                        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                        
=======

                # 测试不同的摄像头索引
                for i in range(4):  # 测试0-3号摄像头
                    result_text.insert(tk.END, f"测试摄像头 {i}:\n")

                    try:
                        # 尝试打开摄像头
                        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

>>>>>>> upstream/main
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret:
                                # 获取摄像头信息
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                fps = cap.get(cv2.CAP_PROP_FPS)
<<<<<<< HEAD
                                
=======

>>>>>>> upstream/main
                                result_text.insert(tk.END, f"  ✓ 正常\n")
                                result_text.insert(tk.END, f"     分辨率: {width}x{height}\n")
                                result_text.insert(tk.END, f"     FPS: {fps:.1f}\n")
                            else:
                                result_text.insert(tk.END, "  ✗ 打开但无法读取帧\n")
                            cap.release()
                        else:
                            result_text.insert(tk.END, "  ✗ 无法打开\n")
                    except Exception as e:
                        result_text.insert(tk.END, f"  ✗ 错误: {str(e)}\n")
<<<<<<< HEAD
                    
                    result_text.insert(tk.END, "\n")
                
                result_text.insert(tk.END, "=== 测试完成 ===\n")
                result_text.see(1.0)  # 滚动到顶部
            
            test_btn = ttk.Button(test_window, text="开始测试", command=start_test)
            test_btn.pack(pady=(0, 10))
            
            # 关闭按钮
            close_btn = ttk.Button(test_window, text="关闭", command=test_window.destroy)
            close_btn.pack(pady=(0, 10))
            
        except Exception as e:
            messagebox.showerror("测试错误", f"摄像头测试失败: {str(e)}")
    
=======

                    result_text.insert(tk.END, "\n")

                result_text.insert(tk.END, "=== 测试完成 ===\n")
                result_text.see(1.0)  # 滚动到顶部

            test_btn = ttk.Button(test_window, text="开始测试", command=start_test)
            test_btn.pack(pady=(0, 10))

            # 关闭按钮
            close_btn = ttk.Button(test_window, text="关闭", command=test_window.destroy)
            close_btn.pack(pady=(0, 10))

        except Exception as e:
            messagebox.showerror("测试错误", f"摄像头测试失败: {str(e)}")

>>>>>>> upstream/main
    def _select_image(self):
        """选择图片"""
        if self.is_processing:
            messagebox.showwarning("提示", "正在处理中，请稍候...")
            return
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        file_types = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("所有文件", "*.*")
        ]
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        file_path = filedialog.askopenfilename(
            title="选择道路图片",
            filetypes=file_types
        )
<<<<<<< HEAD
        
        if file_path:
            self.current_image_path = file_path
            self._load_image(file_path)
    
=======

        if file_path:
            self.current_image_path = file_path
            self._load_image(file_path)

>>>>>>> upstream/main
    def _select_video(self):
        """选择视频文件"""
        if self.is_processing and self.is_video_mode:
            messagebox.showwarning("提示", "正在处理视频，请先停止当前处理")
            return
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        file_types = [
            ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv"),
            ("所有文件", "*.*")
        ]
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        file_path = filedialog.askopenfilename(
            title="选择道路视频",
            filetypes=file_types
        )
<<<<<<< HEAD
        
        if file_path:
            self.video_file_path = file_path
            self._open_video(file_path)
    
=======

        if file_path:
            self.video_file_path = file_path
            self._open_video(file_path)

>>>>>>> upstream/main
    def _open_video(self, file_path):
        """打开视频文件"""
        if self.video_processor.open_video_file(file_path):
            self.file_info_label.config(text=os.path.basename(file_path))
            self.play_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.status_var.set(f"视频已加载: {os.path.basename(file_path)}")
<<<<<<< HEAD
            
=======

>>>>>>> upstream/main
            # 显示第一帧作为预览
            ret, frame = self.video_processor.get_frame()
            if ret:
                self._display_image(frame, self.original_canvas, "视频预览")
<<<<<<< HEAD
                
=======

>>>>>>> upstream/main
                # 在结果区域显示提示
                self.result_canvas.delete("all")
                self.result_canvas.create_text(
                    300, 200,
                    text="视频已加载，点击'开始'按钮进行检测",
                    font=("微软雅黑", 12),
                    fill="#3498db"
                )
        else:
            messagebox.showerror("错误", "无法打开视频文件")
<<<<<<< HEAD
    
=======

>>>>>>> upstream/main
    def _play_video(self):
        """播放视频/摄像头"""
        if self.is_processing and not self.is_video_mode:
            return
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        # 如果是摄像头模式但未打开，先尝试打开
        if self.camera_mode and (self.video_processor.video_capture is None or not self.video_processor.video_capture.isOpened()):
            self._show_camera_preview()
            if self.video_processor.video_capture is None or not self.video_processor.video_capture.isOpened():
                messagebox.showwarning("警告", "请先打开摄像头")
                return
<<<<<<< HEAD
        
        # 更新界面状态
        self.status_var.set("正在启动处理...")
        self.root.update()
        
=======

        # 更新界面状态
        self.status_var.set("正在启动处理...")
        self.root.update()

>>>>>>> upstream/main
        # 启动处理
        if self.video_processor.start_processing(self._process_video_frame):
            self.is_processing = True
            self.play_btn.config(state="disabled")
            self.pause_btn.config(state="normal")
            self.stop_btn.config(state="normal")
<<<<<<< HEAD
            
=======

>>>>>>> upstream/main
            if self.camera_mode:
                self.status_var.set("摄像头实时检测中...")
            else:
                self.status_var.set("视频处理中...")
        else:
            messagebox.showerror("错误", "无法开始处理")
            self.status_var.set("启动失败")
<<<<<<< HEAD
    
=======

>>>>>>> upstream/main
    def _pause_video(self):
        """暂停视频"""
        if self.is_video_mode and self.video_processor.is_playing:
            self.video_processor.pause()
            self.play_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.status_var.set("已暂停")
<<<<<<< HEAD
    
=======

>>>>>>> upstream/main
    def _stop_video(self):
        """停止视频"""
        if self.is_video_mode:
            self.video_processor.stop()
            self.is_processing = False
            self.play_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.status_var.set("已停止")
<<<<<<< HEAD
            
            # 清空显示
            self._clear_canvas_display()
    
    def _process_video_frame(self, frame, frame_info):
        """处理视频帧"""
        try:
            with self.performance_monitor.start_timer("total_processing") as timer:
                # 预处理帧
                processed_frame, roi_info = self.image_processor.preprocess_frame(frame)
                
                # 道路检测
                with self.performance_monitor.start_timer("road_detection"):
                    road_info = self.road_detector.detect_road(
                        processed_frame, 
                        roi_info.get('mask', np.ones(processed_frame.shape[:2], dtype=np.uint8))
                    )
                
                # 车道线检测
                with self.performance_monitor.start_timer("lane_detection"):
                    lane_info = self.lane_detector.detect(
                        processed_frame, 
                        roi_info.get('mask', np.ones(processed_frame.shape[:2], dtype=np.uint8))
                    )
                
                # 方向分析
                with self.performance_monitor.start_timer("direction_analysis"):
                    direction_info = self.direction_analyzer.analyze(road_info, lane_info)
                
                # 创建可视化
                with self.performance_monitor.start_timer("visualization"):
                    visualization = self.visualizer.create_visualization(
                        processed_frame, road_info, lane_info, direction_info, 
                        True, frame_info
                    )
            
            processing_time = timer.stop()
            
            # 在主线程中更新UI
            self.root.after(0, self._update_video_results, 
                          processed_frame, visualization, direction_info, 
                          lane_info, processing_time, frame_info)
            
            # 更新错误计数
            self.error_count = 0
            
=======

            # 清空显示
            self._clear_canvas_display()

    def _process_video_frame(self, frame, frame_info):
        """处理视频帧"""
        try:
            start_time = time.time()

            # 使用检测服务处理帧
            result = self.detection_service.detect_frame(frame)

            if result is None:
                raise ValueError("帧处理失败")

            processed_frame, detection_result = result

            processing_time = time.time() - start_time
            self.performance_monitor.record_metric("total_processing", processing_time)

            # 在主线程中更新UI
            self.root.after(0, self._update_video_results,
                            processed_frame,
                            detection_result.get('visualization', processed_frame),
                            detection_result['direction_info'],
                            detection_result['lane_info'],
                            processing_time,
                            frame_info)

            # 更新错误计数
            self.error_count = 0

>>>>>>> upstream/main
        except Exception as e:
            print(f"视频帧处理失败: {e}")
            self.error_count += 1
            self._handle_error(e)
<<<<<<< HEAD
    
    def _handle_error(self, error):
        """处理错误"""
        current_time = time.time()
        
        # 更新错误显示
        self.root.after(0, lambda: self.error_label.config(text=f"错误: {self.error_count}"))
        
        # 检查是否需要进入恢复模式
        if (self.error_count >= self.max_errors and 
            current_time - self.last_error_time < self.recovery_timeout):
            
            self.last_error_time = current_time
            
=======

    def _handle_error(self, error):
        """处理错误"""
        current_time = time.time()

        # 更新错误显示
        self.root.after(0, lambda: self.error_label.config(text=f"错误: {self.error_count}"))

        # 检查是否需要进入恢复模式
        if (self.error_count >= self.max_errors and
            current_time - self.last_error_time < self.recovery_timeout):

            self.last_error_time = current_time

>>>>>>> upstream/main
            # 自动恢复
            if self.config.auto_recovery:
                print("错误次数过多，触发自动恢复...")
                self._recover_from_error()
            else:
                messagebox.showwarning("警告", "错误次数过多，建议重启应用程序")
<<<<<<< HEAD
        
        self.last_error_time = current_time
    
=======

        self.last_error_time = current_time

>>>>>>> upstream/main
    def _load_image(self, file_path):
        """加载图像"""
        try:
            # 更新界面状态
            self.status_var.set("正在加载图片...")
            self.file_info_label.config(text=os.path.basename(file_path))
            self.redetect_btn.config(state="normal")
<<<<<<< HEAD
            
=======

>>>>>>> upstream/main
            # 在后台线程中处理
            thread = threading.Thread(target=self._process_image_with_recovery, args=(file_path,))
            thread.daemon = True
            thread.start()
<<<<<<< HEAD
            
=======

>>>>>>> upstream/main
        except Exception as e:
            self._handle_error(e)
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")
            self.status_var.set("加载失败")
<<<<<<< HEAD
    
=======

>>>>>>> upstream/main
    def _process_image_with_recovery(self, file_path):
        """带恢复机制的图像处理"""
        try:
            self._process_image(file_path)
            self.error_count = 0
            self.recovery_mode = False
<<<<<<< HEAD
            
        except Exception as e:
            self._handle_error(e)
            traceback.print_exc()
            
=======

        except Exception as e:
            self._handle_error(e)
            traceback.print_exc()

>>>>>>> upstream/main
    def _process_image(self, file_path):
        """处理图像"""
        # 标记为处理中
        self.is_processing = True
        self.root.after(0, self._update_processing_state, True)
<<<<<<< HEAD
        
        try:
            with self.performance_monitor.start_timer("total_processing") as timer:
                # 1. 图像预处理
                result = self.image_processor.load_and_preprocess(file_path)
                if result is None:
                    raise ValueError("无法处理图像")
                
                self.current_image, roi_info = result
                
                # 2. 道路检测
                with self.performance_monitor.start_timer("road_detection"):
                    road_info = self.road_detector.detect_road(self.current_image, roi_info['mask'])
                
                # 3. 车道线检测
                with self.performance_monitor.start_timer("lane_detection"):
                    lane_info = self.lane_detector.detect(self.current_image, roi_info['mask'])
                
                # 4. 方向分析
                with self.performance_monitor.start_timer("direction_analysis"):
                    direction_info = self.direction_analyzer.analyze(road_info, lane_info)
                
                # 5. 创建可视化
                with self.performance_monitor.start_timer("visualization"):
                    visualization = self.visualizer.create_visualization(
                        self.current_image, road_info, lane_info, direction_info
                    )
            
            processing_time = timer.stop()
            
            # 在主线程中更新UI
            self.root.after(0, self._update_results, 
                          direction_info, lane_info, visualization, processing_time)
            
=======

        try:
            # 使用检测服务进行处理
            result = self.detection_service.detect_image(file_path)

            if result is None:
                raise ValueError("无法处理图像")

            # 提取结果
            self.current_image = result['image']
            visualization = result['visualization']
            direction_info = result['direction_info']
            lane_info = result['lane_info']
            processing_time = result['processing_time']

            # 记录性能
            self.performance_monitor.record_metric("total_processing", processing_time)

            # 在主线程中更新UI
            self.root.after(0, self._update_results,
                            direction_info, lane_info, visualization, processing_time)

>>>>>>> upstream/main
            # 更新性能统计
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 10:
                self.processing_times.pop(0)
<<<<<<< HEAD
            
=======

>>>>>>> upstream/main
            # 记录处理历史
            self.processing_history.append({
                'file': file_path,
                'time': processing_time,
                'direction': direction_info['direction'],
                'confidence': direction_info['confidence']
            })
<<<<<<< HEAD
            
        except Exception as e:
            print(f"处理失败: {e}")
            self.root.after(0, self._show_error, str(e))
            
        finally:
            self.is_processing = False
            self.root.after(0, self._update_processing_state, False)
    
=======

        except Exception as e:
            print(f"处理失败: {e}")
            self.root.after(0, self._show_error, str(e))

        finally:
            self.is_processing = False
            self.root.after(0, self._update_processing_state, False)

>>>>>>> upstream/main
    def _update_processing_state(self, is_processing):
        """更新处理状态"""
        if is_processing:
            self.progress_bar.start()
            self.status_var.set("正在分析...")
            self.redetect_btn.config(state="disabled")
        else:
            self.progress_bar.stop()
            self.status_var.set("分析完成")
            self.redetect_btn.config(state="normal")
<<<<<<< HEAD
    
=======

>>>>>>> upstream/main
    def _update_results(self, direction_info, lane_info, visualization, processing_time):
        """更新结果（图像模式）"""
        try:
            # 显示图像
            self._display_image(self.current_image, self.original_canvas, "原始图像")
            self._display_image(visualization, self.result_canvas, "检测结果")
<<<<<<< HEAD
            
=======

            # 保存结果供导出使用
            self.result_image = visualization
            self.last_detection_result = direction_info

>>>>>>> upstream/main
            # 获取信息
            direction = direction_info['direction']
            confidence = direction_info['confidence']
            quality = lane_info.get('detection_quality', 0.0)
<<<<<<< HEAD
            
            # 更新方向信息
            self.direction_label.config(text=f"方向: {direction}")
            
=======

            # 更新方向信息
            self.direction_label.config(text=f"方向: {direction}")

>>>>>>> upstream/main
            # 设置置信度文本和颜色
            if confidence > 0.7:
                color = "#27ae60"  # 绿色
                confidence_text = f"置信度: {confidence:.1%} (高)"
            elif confidence > 0.4:
                color = "#f39c12"  # 橙色
                confidence_text = f"置信度: {confidence:.1%} (中)"
            else:
                color = "#e74c3c"  # 红色
                confidence_text = f"置信度: {confidence:.1%} (低)"
<<<<<<< HEAD
            
            self.confidence_label.config(text=confidence_text, foreground=color)
            
            # 设置检测质量
            self.quality_label.config(text=f"检测质量: {quality:.1%}")
            
            # 设置处理时间
            avg_time = self.performance_monitor.get_statistics("total_processing_time")['avg']
            self.time_label.config(text=f"处理时间: {processing_time:.3f}s (平均: {avg_time:.3f}s)")
            
=======

            self.confidence_label.config(text=confidence_text, foreground=color)

            # 设置检测质量
            self.quality_label.config(text=f"检测质量: {quality:.1%}")

            # 设置处理时间
            avg_time = self.performance_monitor.get_statistics("total_processing_time")['avg']
            self.time_label.config(text=f"处理时间: {processing_time:.3f}s (平均: {avg_time:.3f}s)")

>>>>>>> upstream/main
            # 更新性能信息
            perf_stats = self.performance_monitor.get_summary()
            if perf_stats:
                recent_avg = np.mean(list(self.processing_times[-5:])) if len(self.processing_times) >= 5 else 0
                self.performance_label.config(
                    text=f"模块耗时: 道路{perf_stats.get('road_detection_time', {}).get('avg', 0):.3f}s, "
                         f"车道{perf_stats.get('lane_detection_time', {}).get('avg', 0):.3f}s, "
                         f"方向{perf_stats.get('direction_analysis_time', {}).get('avg', 0):.3f}s"
                )
<<<<<<< HEAD
            
            # 更新状态
            self.status_var.set(f"分析完成 - {direction}")
            
            print(f"处理完成: 方向={direction}, 置信度={confidence:.1%}, 耗时={processing_time:.3f}s")
            
        except Exception as e:
            print(f"更新结果失败: {e}")
            self.status_var.set("更新结果失败")
    
=======

            # 更新导出按钮状态
            self._update_export_buttons_state()

            # 更新状态
            self.status_var.set(f"分析完成 - {direction}")

            print(f"处理完成: 方向={direction}, 置信度={confidence:.1%}, 耗时={processing_time:.3f}s")

        except Exception as e:
            print(f"更新结果失败: {e}")
            self.status_var.set("更新结果失败")

>>>>>>> upstream/main
    def _update_video_results(self, original_frame, visualization, direction_info, lane_info, processing_time, frame_info):
        """更新视频结果"""
        try:
            # 显示图像
            self._display_image(original_frame, self.original_canvas, "原始视频")
            self._display_image(visualization, self.result_canvas, "实时检测")
<<<<<<< HEAD
            
=======

            # 保存结果供导出使用
            self.result_image = visualization
            self.last_detection_result = direction_info

>>>>>>> upstream/main
            # 获取信息
            direction = direction_info['direction']
            confidence = direction_info['confidence']
            quality = lane_info.get('detection_quality', 0.0)
<<<<<<< HEAD
            
            # 更新方向信息
            self.direction_label.config(text=f"方向: {direction}")
            
=======

            # 更新方向信息
            self.direction_label.config(text=f"方向: {direction}")

>>>>>>> upstream/main
            # 设置置信度文本和颜色
            if confidence > 0.7:
                color = "#27ae60"
                confidence_text = f"置信度: {confidence:.1%} (高)"
            elif confidence > 0.4:
                color = "#f39c12"
                confidence_text = f"置信度: {confidence:.1%} (中)"
            else:
                color = "#e74c3c"
                confidence_text = f"置信度: {confidence:.1%} (低)"
<<<<<<< HEAD
            
            self.confidence_label.config(text=confidence_text, foreground=color)
            
            # 设置检测质量
            self.quality_label.config(text=f"检测质量: {quality:.1%}")
            
=======

            self.confidence_label.config(text=confidence_text, foreground=color)

            # 设置检测质量
            self.quality_label.config(text=f"检测质量: {quality:.1%}")

>>>>>>> upstream/main
            # 计算FPS
            self.frame_counter += 1
            current_time = time.time()
            if current_time - self.last_fps_update >= 1.0:
                self.current_fps = self.frame_counter / (current_time - self.last_fps_update)
                self.last_fps_update = current_time
                self.frame_counter = 0
<<<<<<< HEAD
            
=======

>>>>>>> upstream/main
            # 获取视频信息
            video_info = self.video_processor.get_video_info()
            adaptive_skip = video_info.get('adaptive_frame_skip', 1)
            buffer_usage = video_info.get('buffer_usage', '0/5')
<<<<<<< HEAD
            
            # 设置处理时间和FPS
            time_text = f"处理: {processing_time:.3f}s | FPS: {self.current_fps:.1f} | 跳帧: {adaptive_skip} | 缓冲: {buffer_usage}"
            self.time_label.config(text=time_text)
            
=======

            # 设置处理时间和FPS
            time_text = f"处理: {processing_time:.3f}s | FPS: {self.current_fps:.1f} | 跳帧: {adaptive_skip} | 缓冲: {buffer_usage}"
            self.time_label.config(text=time_text)

>>>>>>> upstream/main
            # 更新性能信息
            perf_stats = self.performance_monitor.get_summary()
            if perf_stats:
                module_times = []
                for module in ['road_detection_time', 'lane_detection_time', 'direction_analysis_time', 'visualization_time']:
                    if module in perf_stats:
                        module_times.append(f"{module.split('_')[0]}:{perf_stats[module]['avg']:.3f}s")
<<<<<<< HEAD
                
                if module_times:
                    self.performance_label.config(text=" | ".join(module_times))
            
            # 更新状态
            video_type = "摄像头" if self.camera_mode else "视频"
            status_text = f"{video_type}处理中 - {direction} | FPS: {self.current_fps:.1f}"
            
=======

                if module_times:
                    self.performance_label.config(text=" | ".join(module_times))

            # 更新导出按钮状态
            self._update_export_buttons_state()

            # 更新状态
            video_type = "摄像头" if self.camera_mode else "视频"
            status_text = f"{video_type}处理中 - {direction} | FPS: {self.current_fps:.1f}"

>>>>>>> upstream/main
            # 添加缓冲信息
            if hasattr(self.video_processor, 'frame_buffer'):
                buffer_len = len(self.video_processor.frame_buffer)
                max_len = self.video_processor.frame_buffer.maxlen
                buffer_percent = (buffer_len / max_len) * 100
<<<<<<< HEAD
                
=======

>>>>>>> upstream/main
                if buffer_percent < 20:
                    status_text += f" | 缓冲低({buffer_percent:.0f}%)"
                elif buffer_percent > 80:
                    status_text += f" | 缓冲高({buffer_percent:.0f}%)"
<<<<<<< HEAD
            
            self.status_var.set(status_text)
            
        except Exception as e:
            print(f"更新视频结果失败: {e}")
    
=======

            self.status_var.set(status_text)

        except Exception as e:
            print(f"更新视频结果失败: {e}")

>>>>>>> upstream/main
    def _display_image(self, image, canvas, title):
        """在Canvas上显示图像"""
        try:
            canvas.delete("all")
<<<<<<< HEAD
            
            if image is None:
                canvas.create_text(300, 200, text=f"{title}加载失败", fill="red")
                return
            
            # 转换颜色空间
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # 获取Canvas尺寸
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 600, 400
            
            # 计算缩放比例
            img_width, img_height = pil_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95  # 留出边距
            
            if scale < 1:
                new_size = (int(img_width * scale), int(img_height * scale))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 转换为Tkinter格式
            photo = ImageTk.PhotoImage(pil_image)
            
            # 居中显示
            x = (canvas_width - photo.width()) // 2
            y = (canvas_height - photo.height()) // 2
            
            canvas.create_image(x, y, anchor="nw", image=photo)
            canvas.image = photo  # 保持引用
            
=======

            if image is None:
                canvas.create_text(300, 200, text=f"{title}加载失败", fill="red")
                return

            # 转换颜色空间
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # 获取Canvas尺寸
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 600, 400

            # 计算缩放比例
            img_width, img_height = pil_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95  # 留出边距

            if scale < 1:
                new_size = (int(img_width * scale), int(img_height * scale))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

            # 转换为Tkinter格式
            photo = ImageTk.PhotoImage(pil_image)

            # 居中显示
            x = (canvas_width - photo.width()) // 2
            y = (canvas_height - photo.height()) // 2

            canvas.create_image(x, y, anchor="nw", image=photo)
            canvas.image = photo  # 保持引用

>>>>>>> upstream/main
            # 添加标题
            canvas.create_text(
                canvas_width // 2, 15,
                text=title,
                font=("微软雅黑", 10, "bold"),
                fill="#2c3e50"
            )
<<<<<<< HEAD
            
        except Exception as e:
            print(f"显示图像失败: {e}")
            canvas.create_text(150, 150, text="图像显示失败", fill="red")
    
=======

        except Exception as e:
            print(f"显示图像失败: {e}")
            canvas.create_text(150, 150, text="图像显示失败", fill="red")

>>>>>>> upstream/main
    def _clear_canvas_display(self):
        """清空画布显示"""
        self.original_canvas.delete("all")
        self.result_canvas.delete("all")
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        self.original_canvas.create_text(
            300, 200,
            text="请选择输入源",
            font=("微软雅黑", 12),
            fill="#7f8c8d"
        )
<<<<<<< HEAD
        
=======

>>>>>>> upstream/main
        self.result_canvas.create_text(
            300, 200,
            text="检测结果将显示在这里",
            font=("微软雅黑", 12),
            fill="#7f8c8d"
        )
<<<<<<< HEAD
    
    def _redetect(self):
        """重新检测"""
        if self.current_image_path and not self.is_processing and not self.is_video_mode:
            self._process_image_with_recovery(self.current_image_path)
    
    def _on_parameter_change(self, value):
        """参数变化回调"""
        sensitivity = self.sensitivity_var.get()
        
=======

    def _redetect(self):
        """重新检测当前图像"""
        if not self.current_image_path or self.is_processing:
            return

        self._process_image_threaded(self.current_image_path)

    def _export_result_image(self):
        """导出带标注的结果图片"""
        success = self.export_manager.export_image(self.result_image, self.root)
        if success:
            self.status_var.set(f"图片已导出")

    def _update_export_buttons_state(self):
        """更新导出按钮状态"""
        has_result = self.result_image is not None and self.last_detection_result is not None

        state = "normal" if has_result else "disabled"
        self.export_image_btn.config(state=state)
        self.export_json_btn.config(state=state)
        self.export_csv_btn.config(state=state)

    def _export_json_report(self):
        """导出JSON格式的检测报告"""
        success = self.export_manager.export_json_report(
            self.last_detection_result,
            self.current_image,
            self.current_image_path,
            self.root
        )
        if success:
            self.status_var.set(f"JSON报告已导出")

    def _on_parameter_change(self, value):
        """参数变化回调"""
        sensitivity = self.sensitivity_var.get()

>>>>>>> upstream/main
        # 根据敏感度调整参数
        updates = {
            'canny_threshold1': int(30 + sensitivity * 40),
            'canny_threshold2': int(80 + sensitivity * 100),
            'hough_threshold': int(20 + (1 - sensitivity) * 30)
        }
<<<<<<< HEAD
        
        # 更新配置
        self.config_manager.update_config(updates, save_to_file=True)
        
        print(f"参数更新: 敏感度={sensitivity:.2f}, 更新={updates}")
        
        # 如果已有图像，自动重新检测
        if self.current_image_path and not self.is_processing and not self.is_video_mode:
            self._redetect()
    
    def _on_scene_change(self, event):
        """场景选择变化"""
        scene = self.scene_var.get()
        
=======

        # 更新配置
        self.config_manager.update_config(updates, save_to_file=True)

        print(f"参数更新: 敏感度={sensitivity:.2f}, 更新={updates}")

        # 如果已有图像，自动重新检测
        if self.current_image_path and not self.is_processing and not self.is_video_mode:
            self._redetect()

    def _export_csv_report(self):
        """导出CSV格式的检测报告"""
        success = self.export_manager.export_csv_report(
            self.last_detection_result,
            self.current_image,
            self.current_image_path,
            self.root
        )
        if success:
            self.status_var.set(f"CSV报告已导出")


    def _batch_export_folder(self):
        """批量导出文件夹中的所有图片"""
        self.batch_processor.process_folder(self.root)


    def _on_scene_change(self, event):
        """场景选择变化"""
        scene = self.scene_var.get()

>>>>>>> upstream/main
        if scene == "高速公路":
            new_config = SceneConfig.get_scene_config('highway')
        elif scene == "城市道路":
            new_config = SceneConfig.get_scene_config('urban')
        elif scene == "乡村道路":
            new_config = SceneConfig.get_scene_config('rural')
        else:  # 自动
            return
<<<<<<< HEAD
        
        # 更新配置管理器
        self.config_manager.update_config(new_config.to_dict(), save_to_file=True)
        
        print(f"场景切换为: {scene}")
        self.status_var.set(f"场景已切换为: {scene}")
        
        # 重新检测
        if self.current_image_path and not self.is_processing and not self.is_video_mode:
            self._redetect()
    
=======

        # 更新配置管理器
        self.config_manager.update_config(new_config.to_dict(), save_to_file=True)

        print(f"场景切换为: {scene}")
        self.status_var.set(f"场景已切换为: {scene}")

        # 重新检测
        if self.current_image_path and not self.is_processing and not self.is_video_mode:
            self._redetect()

>>>>>>> upstream/main
    def _show_error(self, error_msg):
        """显示错误"""
        messagebox.showerror("错误", f"处理失败: {error_msg}")
        self.status_var.set("处理失败")
        self.error_count += 1
        self.error_label.config(text=f"错误: {self.error_count}")
<<<<<<< HEAD
    
=======

>>>>>>> upstream/main
    def _on_closing(self):
        """窗口关闭事件"""
        try:
            # 停止配置监控
            if hasattr(self, 'config_manager'):
                self.config_manager.stop_watching()
<<<<<<< HEAD
            
            # 停止视频处理
            if self.is_video_mode:
                self._stop_video()
            
=======

            # 停止视频处理
            if self.is_video_mode:
                self._stop_video()

>>>>>>> upstream/main
            # 释放摄像头资源
            if hasattr(self, 'video_processor') and self.video_processor:
                print("正在释放摄像头资源...")
                self.video_processor.release()
<<<<<<< HEAD
            
            # 关闭性能监控窗口
            if self.perf_window is not None and self.perf_window.winfo_exists():
                self.perf_window.destroy()
            
            # 关闭窗口
            self.root.destroy()
            print("应用程序已关闭")
            
=======

            # 关闭性能监控窗口
            if hasattr(self, 'perf_window_manager'):
                self.perf_window_manager.destroy()

            # 关闭窗口
            self.root.destroy()
            print("应用程序已关闭")

>>>>>>> upstream/main
        except Exception as e:
            print(f"关闭应用程序时出错: {e}")
            self.root.destroy()


def main():
    """主函数"""
    try:
        # 创建主窗口
        root = tk.Tk()
<<<<<<< HEAD
        
        # 创建应用程序实例
        app = LaneDetectionApp(root)
        
        # 运行主循环
        root.mainloop()
        
=======

        # 创建应用程序实例
        app = LaneDetectionApp(root)

        # 运行主循环
        root.mainloop()

>>>>>>> upstream/main
    except Exception as e:
        print(f"应用程序启动失败: {e}")
        traceback.print_exc()
        messagebox.showerror("致命错误", f"应用程序启动失败: {str(e)}")


if __name__ == "__main__":
    main()