"""
主应用程序模块 - 负责界面和流程控制
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time
from typing import Dict, Any 
from collections import deque
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# 导入各个模块
from config import AppConfig
from image_processor import SmartImageProcessor, RoadDetector
from lane_detector import LaneDetector
from direction_analyzer import DirectionAnalyzer
from visualizer import Visualizer

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
        
        # 状态变量
        self.current_image = None
        self.current_image_path = None
        self.is_processing = False
        self.processing_history = deque(maxlen=10)
        
        # 性能统计
        self.processing_times = []
        
        # 创建界面
        self._create_ui()
        
        print("道路方向识别系统已启动")
    
    def _setup_window(self):
        """设置窗口"""
        self.root.title("道路方向识别系统")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)
        
        # 设置窗口居中
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
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
            text="道路方向识别系统",
            font=("微软雅黑", 16, "bold"),
            foreground="#2c3e50"
        )
        title_label.pack(side="left")
        
        # 版本信息
        version_label = ttk.Label(
            title_frame,
            text="v2.0",
            font=("微软雅黑", 10),
            foreground="#7f8c8d"
        )
        version_label.pack(side="right")
    
    def _create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(
            parent,
            text="控制面板",
            padding="15",
            relief="groove"
        )
        control_frame.pack_propagate(False)
        control_frame.config(width=300)
        
        # 文件操作区域
        file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding="10")
        file_frame.pack(fill="x", pady=(0, 15))
        
        # 选择图片按钮
        select_btn = ttk.Button(
            file_frame,
            text="选择图片",
            command=self._select_image,
            width=20
        )
        select_btn.pack(pady=(0, 10))
        
        # 重新检测按钮
        self.redetect_btn = ttk.Button(
            file_frame,
            text="重新检测",
            command=self._redetect,
            width=20,
            state="disabled"
        )
        self.redetect_btn.pack(pady=(0, 10))
        
        # 文件信息显示
        self.file_info_label = ttk.Label(
            file_frame,
            text="未选择图片",
            wraplength=250,
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
            length=250
        )
        sensitivity_scale.pack(fill="x", pady=(0, 10))
        
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
        
        # 处理时间显示
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
            text="请选择道路图片",
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
        """更新结果"""
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
    
    def _redetect(self):
        """重新检测"""
        if self.current_image_path and not self.is_processing:
            self._process_image(self.current_image_path)
    
    def _on_parameter_change(self, value):
        """参数变化回调"""
        # 更新配置
        sensitivity = self.sensitivity_var.get()
        
        # 根据敏感度调整参数
        self.config.canny_threshold1 = int(30 + sensitivity * 40)
        self.config.canny_threshold2 = int(80 + sensitivity * 100)
        self.config.hough_threshold = int(20 + (1 - sensitivity) * 30)
        
        print(f"参数更新: 敏感度={sensitivity:.2f}")
        
        # 如果已有图像，自动重新检测
        if self.current_image_path and not self.is_processing:
            self._redetect()
    
    def _show_error(self, error_msg: str):
        """显示错误"""
        messagebox.showerror("错误", f"处理失败: {error_msg}")
        self.status_var.set("处理失败")


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
    main()