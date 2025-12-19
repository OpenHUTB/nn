"""
主应用程序模块 - 支持图像、视频和摄像头实时识别
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time
from collections import deque
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# 类型注解导入
from typing import Dict, Any, Optional, Tuple, List

# 导入各个模块
from config import AppConfig, SceneConfig
from image_processor import SmartImageProcessor, RoadDetector
from lane_detector import LaneDetector
from direction_analyzer import DirectionAnalyzer
from visualizer import Visualizer
from video_processor import VideoProcessor

class LaneDetectionApp:
    """道路方向识别系统主应用程序"""
    
    def __init__(self, root):
        self.root = root
        self._setup_window()
        
        # 初始化配置
        self.config = AppConfig()
        
        # 初始化各个模块
        self.image_processor = SmartImageProcessor(self.config)
        self.road_detector = RoadDetector(self.config)
        self.lane_detector = LaneDetector(self.config)
        self.direction_analyzer = DirectionAnalyzer(self.config)
        self.visualizer = Visualizer(self.config)
        self.video_processor = VideoProcessor(self.config)
        
        # 状态变量
        self.current_image = None
        self.current_image_path = None
        self.is_processing = False
        self.is_video_mode = False
        self.processing_history = deque(maxlen=10)
        
        # 视频相关变量
        self.video_file_path = None
        self.camera_mode = False
        
        # 性能统计
        self.processing_times = []
        self.frame_counter = 0
        self.last_fps_update = time.time()
        self.current_fps = 0
        
        # 创建界面
        self._create_ui()
        
        print("道路方向识别系统已启动（支持视频/摄像头）")
    
    def _setup_window(self):
        """设置窗口"""
        self.root.title("道路方向识别系统 - 支持视频/摄像头")
        self.root.geometry("1400x850")
        self.root.minsize(1200, 700)
        
        # 设置窗口居中
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_ui(self):
        """创建用户界面"""
        # 主容器
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
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
    
    def _create_title_bar(self, parent):
        """创建标题栏"""
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill="x", pady=(0, 10))
        
        # 标题
        title_label = ttk.Label(
            title_frame,
            text="道路方向识别系统（支持视频/摄像头）",
            font=("微软雅黑", 16, "bold"),
            foreground="#2c3e50"
        )
        title_label.pack(side="left")
        
        # 模式指示器
        self.mode_label = ttk.Label(
            title_frame,
            text="[图像模式]",
            font=("微软雅黑", 10),
            foreground="#3498db"
        )
        self.mode_label.pack(side="right", padx=(0, 10))
    
    def _create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(
            parent,
            text="控制面板",
            padding="15",
            relief="groove"
        )
        control_frame.pack_propagate(False)
        control_frame.config(width=350)
        
        # 输入模式选择
        mode_frame = ttk.LabelFrame(control_frame, text="输入模式", padding="10")
        mode_frame.pack(fill="x", pady=(0, 15))
        
        # 模式选择按钮
        mode_buttons_frame = ttk.Frame(mode_frame)
        mode_buttons_frame.pack()
        
        self.image_mode_btn = ttk.Button(
            mode_buttons_frame,
            text="图像模式",
            command=self._switch_to_image_mode,
            width=12
        )
        self.image_mode_btn.pack(side="left", padx=(0, 5))
        
        self.video_mode_btn = ttk.Button(
            mode_buttons_frame,
            text="视频模式",
            command=self._switch_to_video_mode,
            width=12
        )
        self.video_mode_btn.pack(side="left", padx=(0, 5))
        
        self.camera_mode_btn = ttk.Button(
            mode_buttons_frame,
            text="摄像头模式",
            command=self._switch_to_camera_mode,
            width=12
        )
        self.camera_mode_btn.pack(side="left")
        
        # 文件操作区域
        self.file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding="10")
        self.file_frame.pack(fill="x", pady=(0, 15))
        
        # 选择图片按钮
        self.select_image_btn = ttk.Button(
            self.file_frame,
            text="选择图片",
            command=self._select_image,
            width=20
        )
        self.select_image_btn.pack(pady=(0, 10))
        
        # 重新检测按钮
        self.redetect_btn = ttk.Button(
            self.file_frame,
            text="重新检测",
            command=self._redetect,
            width=20,
            state="disabled"
        )
        self.redetect_btn.pack(pady=(0, 10))
        
        # 视频控制区域（初始隐藏）
        self.video_frame = ttk.LabelFrame(control_frame, text="视频控制", padding="10")
        
        # 选择视频按钮
        self.select_video_btn = ttk.Button(
            self.video_frame,
            text="选择视频文件",
            command=self._select_video,
            width=20
        )
        self.select_video_btn.pack(pady=(0, 10))
        
        # 视频控制按钮
        self.video_control_frame = ttk.Frame(self.video_frame)
        self.video_control_frame.pack()
        
        self.play_btn = ttk.Button(
            self.video_control_frame,
            text="开始",
            command=self._play_video,
            width=8,
            state="disabled"
        )
        self.play_btn.pack(side="left", padx=(0, 5))
        
        self.pause_btn = ttk.Button(
            self.video_control_frame,
            text="暂停",
            command=self._pause_video,
            width=8,
            state="disabled"
        )
        self.pause_btn.pack(side="left", padx=(0, 5))
        
        self.stop_btn = ttk.Button(
            self.video_control_frame,
            text="停止",
            command=self._stop_video,
            width=8,
            state="disabled"
        )
        self.stop_btn.pack(side="left")
        
        # 文件信息显示
        self.file_info_label = ttk.Label(
            self.file_frame,
            text="未选择图片",
            wraplength=300,
            foreground="#3498db"
        )
        self.file_info_label.pack()
        
        # 参数调节区域
        param_frame = ttk.LabelFrame(control_frame, text="参数调节", padding="10")
        param_frame.pack(fill="x", pady=(0, 15))
        
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
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(control_frame, text="检测结果", padding="10")
        result_frame.pack(fill="x")
        
        # 方向显示
        self.direction_label = ttk.Label(
            result_frame,
            text="等待检测...",
            font=("微软雅黑", 14, "bold"),
            foreground="#2c3e50"
        )
        self.direction_label.pack(anchor="w", pady=(0, 5))
        
        # 置信度显示
        self.confidence_label = ttk.Label(
            result_frame,
            text="",
            font=("微软雅黑", 11)
        )
        self.confidence_label.pack(anchor="w", pady=(0, 5))
        
        # 检测质量显示
        self.quality_label = ttk.Label(
            result_frame,
            text="",
            font=("微软雅黑", 10),
            foreground="#7f8c8d"
        )
        self.quality_label.pack(anchor="w", pady=(0, 5))
        
        # 处理时间/FPS显示
        self.time_label = ttk.Label(
            result_frame,
            text="",
            font=("微软雅黑", 9),
            foreground="#95a5a6"
        )
        self.time_label.pack(anchor="w")
        
        return control_frame
    
    def _create_display_panel(self, parent):
        """创建显示面板"""
        display_frame = ttk.Frame(parent)
        
        # 图像显示区域
        images_frame = ttk.Frame(display_frame)
        images_frame.pack(fill="both", expand=True)
        
        # 原图显示
        original_frame = ttk.LabelFrame(
            images_frame,
            text="原始图像",
            padding="5",
            relief="groove"
        )
        original_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
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
        
        # 结果图显示
        result_frame = ttk.LabelFrame(
            images_frame,
            text="检测结果",
            padding="5",
            relief="groove"
        )
        result_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
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
        
        return display_frame
    
    def _create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent, relief="sunken", borderwidth=1)
        status_frame.pack(fill="x", pady=(10, 0))
        
        # 进度条
        self.progress_bar = ttk.Progressbar(
            status_frame,
            mode='indeterminate',
            length=200
        )
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(5, 10), pady=5)
        
        # 状态文本
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("微软雅黑", 9)
        )
        status_label.pack(side="right", padx=(0, 10), pady=5)
    
    def _switch_to_image_mode(self):
        """切换到图像模式"""
        if self.is_video_mode:
            self._stop_video()
        
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
    
    def _switch_to_video_mode(self):
        """切换到视频模式"""
        if self.is_video_mode and self.camera_mode:
            self._stop_video()
        
        self.is_video_mode = True
        self.camera_mode = False
        self.mode_label.config(text="[视频模式]", foreground="#e74c3c")
        
        # 隐藏图像控制，显示视频控制
        self.file_frame.pack_forget()
        self.video_frame.pack(fill="x", pady=(0, 15))
        
        # 更新按钮状态
        self.select_video_btn.config(state="normal")
        self.play_btn.config(state="disabled")
        self.pause_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        
        self.status_var.set("已切换到视频模式")
    
    def _switch_to_camera_mode(self):
        """切换到摄像头模式"""
        if self.is_video_mode:
            self._stop_video()
        
        self.is_video_mode = True
        self.camera_mode = True
        self.mode_label.config(text="[摄像头模式]", foreground="#9b59b6")
        
        # 隐藏图像控制，显示视频控制
        self.file_frame.pack_forget()
        self.video_frame.pack(fill="x", pady=(0, 15))
        
        # 更新按钮状态
        self.select_video_btn.config(state="normal")
        self.play_btn.config(state="normal")
        self.pause_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        
        # 尝试打开摄像头
        self._open_camera()
    
    def _open_camera(self):
        """打开摄像头"""
        try:
            if self.video_processor.open_camera():
                self.play_btn.config(state="normal")
                self.status_var.set("摄像头已打开")
            else:
                messagebox.showerror("错误", "无法打开摄像头，请检查摄像头连接")
                self._switch_to_image_mode()
        except Exception as e:
            messagebox.showerror("错误", f"打开摄像头失败: {str(e)}")
            self._switch_to_image_mode()
    
    def _select_image(self):
        """选择图片"""
        if self.is_processing:
            messagebox.showwarning("提示", "正在处理中，请稍候...")
            return
        
        file_types = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("所有文件", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="选择道路图片",
            filetypes=file_types
        )
        
        if file_path:
            self.current_image_path = file_path
            self._load_image(file_path)
    
    def _select_video(self):
        """选择视频文件"""
        if self.is_processing and self.is_video_mode:
            messagebox.showwarning("提示", "正在处理视频，请先停止当前处理")
            return
        
        file_types = [
            ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv"),
            ("所有文件", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="选择道路视频",
            filetypes=file_types
        )
        
        if file_path:
            self.video_file_path = file_path
            self._open_video(file_path)
    
    def _open_video(self, file_path: str):
        """打开视频文件"""
        if self.video_processor.open_video_file(file_path):
            self.file_info_label.config(text=os.path.basename(file_path))
            self.play_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.status_var.set(f"视频已加载: {os.path.basename(file_path)}")
        else:
            messagebox.showerror("错误", "无法打开视频文件")
    
    def _play_video(self):
        """播放视频"""
        if self.is_processing and not self.is_video_mode:
            return
        
        if self.camera_mode and self.video_processor.video_capture is None:
            self._open_camera()
        
        if self.video_processor.start_processing(self._process_video_frame):
            self.is_processing = True
            self.play_btn.config(state="disabled")
            self.pause_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            self.status_var.set("视频处理中...")
        else:
            messagebox.showerror("错误", "无法开始视频处理")
    
    def _pause_video(self):
        """暂停视频"""
        if self.is_video_mode and self.video_processor.is_playing:
            self.video_processor.pause()
            self.play_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.status_var.set("视频已暂停")
    
    def _stop_video(self):
        """停止视频"""
        if self.is_video_mode:
            self.video_processor.stop()
            self.is_processing = False
            self.play_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.status_var.set("视频已停止")
            
            # 清空显示
            self._clear_canvas_display()
    
    def _process_video_frame(self, frame: np.ndarray, frame_info: Dict[str, Any]):
        """处理视频帧"""
        try:
            start_time = time.time()
            
            # 预处理帧
            processed_frame, roi_info = self.image_processor.preprocess_frame(frame)
            
            # 道路检测
            road_info = self.road_detector.detect_road(processed_frame, roi_info.get('mask', np.ones(processed_frame.shape[:2], dtype=np.uint8)))
            
            # 车道线检测
            lane_info = self.lane_detector.detect(processed_frame, roi_info.get('mask', np.ones(processed_frame.shape[:2], dtype=np.uint8)))
            
            # 方向分析
            direction_info = self.direction_analyzer.analyze(road_info, lane_info)
            
            # 创建可视化
            visualization = self.visualizer.create_visualization(
                processed_frame, road_info, lane_info, direction_info, 
                is_video=True, frame_info=frame_info
            )
            
            processing_time = time.time() - start_time
            
            # 在主线程中更新UI
            self.root.after(0, self._update_video_results, 
                          processed_frame, visualization, direction_info, 
                          lane_info, processing_time, frame_info)
            
            # 更新性能统计
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 10:
                self.processing_times.pop(0)
            
        except Exception as e:
            print(f"视频帧处理失败: {e}")
    
    def _load_image(self, file_path: str):
        """加载图像"""
        try:
            # 更新界面状态
            self.status_var.set("正在加载图片...")
            self.file_info_label.config(text=os.path.basename(file_path))
            self.redetect_btn.config(state="normal")
            
            # 在后台线程中处理
            thread = threading.Thread(target=self._process_image, args=(file_path,))
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")
            self.status_var.set("加载失败")
    
    def _process_image(self, file_path: str):
        """处理图像"""
        start_time = time.time()
        
        try:
            # 标记为处理中
            self.is_processing = True
            self.root.after(0, self._update_processing_state, True)
            
            # 1. 图像预处理
            result = self.image_processor.load_and_preprocess(file_path)
            if result is None:
                raise ValueError("无法处理图像")
            
            self.current_image, roi_info = result
            
            # 2. 道路检测
            road_info = self.road_detector.detect_road(self.current_image, roi_info['mask'])
            
            # 3. 车道线检测
            lane_info = self.lane_detector.detect(self.current_image, roi_info['mask'])
            
            # 4. 方向分析
            direction_info = self.direction_analyzer.analyze(road_info, lane_info)
            
            # 5. 创建可视化
            visualization = self.visualizer.create_visualization(
                self.current_image, road_info, lane_info, direction_info
            )
            
            processing_time = time.time() - start_time
            
            # 在主线程中更新UI
            self.root.after(0, self._update_results, 
                          direction_info, lane_info, visualization, processing_time)
            
            # 更新统计信息
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 10:
                self.processing_times.pop(0)
            
            # 记录处理历史
            self.processing_history.append({
                'file': file_path,
                'time': processing_time,
                'direction': direction_info['direction'],
                'confidence': direction_info['confidence']
            })
            
        except Exception as e:
            print(f"处理失败: {e}")
            self.root.after(0, self._show_error, str(e))
            
        finally:
            self.is_processing = False
            self.root.after(0, self._update_processing_state, False)
    
    def _update_processing_state(self, is_processing: bool):
        """更新处理状态"""
        if is_processing:
            self.progress_bar.start()
            self.status_var.set("正在分析...")
            self.redetect_btn.config(state="disabled")
        else:
            self.progress_bar.stop()
            self.status_var.set("分析完成")
            self.redetect_btn.config(state="normal")
    
    def _update_results(self, direction_info: Dict[str, Any], lane_info: Dict[str, Any],
                       visualization: np.ndarray, processing_time: float):
        """更新结果（图像模式）"""
        try:
            # 显示图像
            self._display_image(self.current_image, self.original_canvas, "原始图像")
            self._display_image(visualization, self.result_canvas, "检测结果")
            
            # 获取信息
            direction = direction_info['direction']
            confidence = direction_info['confidence']
            quality = lane_info.get('detection_quality', 0.0)
            
            # 更新方向信息
            self.direction_label.config(text=f"方向: {direction}")
            
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
            
            self.confidence_label.config(text=confidence_text, foreground=color)
            
            # 设置检测质量
            self.quality_label.config(text=f"检测质量: {quality:.1%}")
            
            # 设置处理时间
            self.time_label.config(text=f"处理时间: {processing_time:.3f}秒")
            
            # 更新状态
            self.status_var.set(f"分析完成 - {direction}")
            
            print(f"处理完成: 方向={direction}, 置信度={confidence:.1%}, 耗时={processing_time:.3f}s")
            
        except Exception as e:
            print(f"更新结果失败: {e}")
            self.status_var.set("更新结果失败")
    
    def _update_video_results(self, original_frame: np.ndarray, visualization: np.ndarray,
                            direction_info: Dict[str, Any], lane_info: Dict[str, Any],
                            processing_time: float, frame_info: Dict[str, Any]):
        """更新视频结果"""
        try:
            # 显示图像
            self._display_image(original_frame, self.original_canvas, "原始视频")
            self._display_image(visualization, self.result_canvas, "实时检测")
            
            # 获取信息
            direction = direction_info['direction']
            confidence = direction_info['confidence']
            quality = lane_info.get('detection_quality', 0.0)
            
            # 更新方向信息
            self.direction_label.config(text=f"方向: {direction}")
            
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
            
            self.confidence_label.config(text=confidence_text, foreground=color)
            
            # 设置检测质量
            self.quality_label.config(text=f"检测质量: {quality:.1%}")
            
            # 计算FPS
            self.frame_counter += 1
            current_time = time.time()
            if current_time - self.last_fps_update >= 1.0:
                self.current_fps = self.frame_counter / (current_time - self.last_fps_update)
                self.last_fps_update = current_time
                self.frame_counter = 0
            
            # 设置处理时间和FPS
            time_text = f"处理时间: {processing_time:.3f}s | FPS: {self.current_fps:.1f}"
            self.time_label.config(text=time_text)
            
            # 更新状态
            video_type = "摄像头" if self.camera_mode else "视频"
            self.status_var.set(f"{video_type}处理中 - {direction} | FPS: {self.current_fps:.1f}")
            
        except Exception as e:
            print(f"更新视频结果失败: {e}")
    
    def _display_image(self, image: np.ndarray, canvas: tk.Canvas, title: str):
        """在Canvas上显示图像"""
        try:
            canvas.delete("all")
            
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
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
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
            
        except Exception as e:
            print(f"显示图像失败: {e}")
            canvas.create_text(150, 150, text="图像显示失败", fill="red")
    
    def _clear_canvas_display(self):
        """清空画布显示"""
        self.original_canvas.delete("all")
        self.result_canvas.delete("all")
        
        self.original_canvas.create_text(
            300, 200,
            text="请选择输入源",
            font=("微软雅黑", 12),
            fill="#7f8c8d"
        )
        
        self.result_canvas.create_text(
            300, 200,
            text="检测结果将显示在这里",
            font=("微软雅黑", 12),
            fill="#7f8c8d"
        )
    
    def _redetect(self):
        """重新检测"""
        if self.current_image_path and not self.is_processing and not self.is_video_mode:
            self._process_image(self.current_image_path)
    
    def _on_parameter_change(self, value):
        """参数变化回调"""
        sensitivity = self.sensitivity_var.get()
        
        # 根据敏感度调整参数
        self.config.canny_threshold1 = int(30 + sensitivity * 40)
        self.config.canny_threshold2 = int(80 + sensitivity * 100)
        self.config.hough_threshold = int(20 + (1 - sensitivity) * 30)
        
        print(f"参数更新: 敏感度={sensitivity:.2f}")
        
        # 如果已有图像，自动重新检测
        if self.current_image_path and not self.is_processing and not self.is_video_mode:
            self._redetect()
    
    def _on_scene_change(self, event):
        """场景选择变化"""
        scene = self.scene_var.get()
        
        if scene == "高速公路":
            config = SceneConfig.get_scene_config('highway')
        elif scene == "城市道路":
            config = SceneConfig.get_scene_config('urban')
        elif scene == "乡村道路":
            config = SceneConfig.get_scene_config('rural')
        else:  # 自动
            return
        
        # 更新配置
        self.config = config
        
        # 重新初始化模块
        self.image_processor = SmartImageProcessor(self.config)
        self.road_detector = RoadDetector(self.config)
        self.lane_detector = LaneDetector(self.config)
        self.direction_analyzer = DirectionAnalyzer(self.config)
        self.visualizer = Visualizer(self.config)
        self.video_processor = VideoProcessor(self.config)
        
        print(f"场景切换为: {scene}")
        self.status_var.set(f"场景已切换为: {scene}")
        
        # 重新检测
        if self.current_image_path and not self.is_processing and not self.is_video_mode:
            self._redetect()
    
    def _show_error(self, error_msg: str):
        """显示错误"""
        messagebox.showerror("错误", f"处理失败: {error_msg}")
        self.status_var.set("处理失败")
    
    def _on_closing(self):
        """窗口关闭事件"""
        if self.is_video_mode:
            self._stop_video()
            self.video_processor.release()
        
        self.root.destroy()
        print("应用程序已关闭")

def main():
    """主函数"""
    try:
        # 创建主窗口
        root = tk.Tk()
        
        # 创建应用程序实例
        app = LaneDetectionApp(root)
        
        # 运行主循环
        root.mainloop()
        
    except Exception as e:
        print(f"应用程序启动失败: {e}")
        messagebox.showerror("致命错误", f"应用程序启动失败: {str(e)}")

if __name__ == "__main__":
    # 导入SceneConfig
    from config import SceneConfig
    main()