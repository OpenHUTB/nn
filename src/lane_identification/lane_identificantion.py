import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import time
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lane_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LaneDetection")

@dataclass
class DetectionConfig:
    """检测配置参数"""
    # 颜色分割参数
    hsv_lower: Tuple[int, int, int] = (0, 0, 50)
    hsv_upper: Tuple[int, int, int] = (180, 50, 200)
    
    # 边缘检测参数
    canny_low: int = 50
    canny_high: int = 150
    
    # 霍夫变换参数
    hough_rho: int = 1
    hough_theta: float = np.pi/180
    hough_threshold: int = 30
    hough_min_length: int = 20
    hough_max_gap: int = 50
    
    # 方向判断阈值
    width_ratio_threshold: float = 0.7
    center_deviation_threshold: float = 0.15
    
    # 形态学操作参数
    morph_kernel_size: int = 5
    blur_kernel_size: int = 5

class ImageProcessor:
    """图像处理工具类"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
    
    def resize_image(self, image: np.ndarray, max_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """调整图像尺寸"""
        h, w = image.shape[:2]
        if w > max_size[0] or h > max_size[1]:
            scale = min(max_size[0] / w, max_size[1] / h)
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """增强图像对比度"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """去除阴影"""
        rgb_planes = cv2.split(image)
        result_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            result_planes.append(diff_img)
        return cv2.merge(result_planes)

class RoadDetector:
    """道路检测器"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.image_processor = ImageProcessor(config)
        self.last_processing_time = 0
        
    def detect_road_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """检测道路区域"""
        try:
            # 图像预处理
            enhanced = self.image_processor.enhance_contrast(image)
            shadow_removed = self.image_processor.remove_shadows(enhanced)
            
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(shadow_removed, cv2.COLOR_BGR2HSV)
            
            # 创建道路掩码
            road_mask = cv2.inRange(hsv, 
                                  np.array(self.config.hsv_lower), 
                                  np.array(self.config.hsv_upper))
            
            # 形态学操作
            kernel = np.ones((self.config.morph_kernel_size, 
                            self.config.morph_kernel_size), np.uint8)
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
            
            return road_mask
            
        except Exception as e:
            logger.error(f"道路区域检测失败: {str(e)}")
            return None
    
    def extract_road_contour(self, road_mask: np.ndarray) -> Optional[np.ndarray]:
        """提取道路轮廓"""
        try:
            contours, _ = cv2.findContours(
                road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None
            
            # 按面积排序并选择最大的几个轮廓
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
            
            # 合并相关轮廓
            merged_contour = np.vstack(contours)
            
            # 计算凸包
            hull = cv2.convexHull(merged_contour)
            
            # 简化轮廓（减少点数）
            epsilon = 0.01 * cv2.arcLength(hull, True)
            simplified_hull = cv2.approxPolyDP(hull, epsilon, True)
            
            return simplified_hull
            
        except Exception as e:
            logger.error(f"轮廓提取失败: {str(e)}")
            return None
    
    def detect_lane_lines(self, image: np.ndarray) -> Dict[str, List]:
        """检测车道线"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, 
                                  (self.config.blur_kernel_size, self.config.blur_kernel_size), 0)
            edges = cv2.Canny(blur, self.config.canny_low, self.config.canny_high)
            
            # 定义ROI区域
            height, width = image.shape[:2]
            roi_vertices = np.array([[
                (width * 0.1, height * 0.95),
                (width * 0.4, height * 0.6),
                (width * 0.6, height * 0.6),
                (width * 0.9, height * 0.95)
            ]], dtype=np.int32)
            
            mask = np.zeros_like(edges)
            cv2.fillPoly(mask, roi_vertices, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # 霍夫变换检测直线
            lines = cv2.HoughLinesP(
                masked_edges,
                self.config.hough_rho,
                self.config.hough_theta,
                self.config.hough_threshold,
                minLineLength=self.config.hough_min_length,
                maxLineGap=self.config.hough_max_gap
            )
            
            return self._classify_lines(lines, width) if lines is not None else {"left": [], "right": []}
            
        except Exception as e:
            logger.error(f"车道线检测失败: {str(e)}")
            return {"left": [], "right": []}
    
    def _classify_lines(self, lines: np.ndarray, image_width: int) -> Dict[str, List]:
        """分类左右车道线"""
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if x2 - x1 == 0:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # 过滤水平线和异常斜率
            if abs(slope) < 0.3 or abs(slope) > 2.0:
                continue
                
            # 根据斜率分类
            if slope < 0 and x1 < image_width * 0.6 and x2 < image_width * 0.6:
                left_lines.append((x1, y1, x2, y2, slope))
            elif slope > 0 and x1 > image_width * 0.4 and x2 > image_width * 0.4:
                right_lines.append((x1, y1, x2, y2, slope))
        
        return {"left": left_lines, "right": right_lines}

class DirectionAnalyzer:
    """方向分析器"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.direction_history = []
        self.history_size = 5
    
    def analyze_from_contour(self, contour: np.ndarray, image_size: Tuple[int, int]) -> str:
        """基于轮廓分析方向"""
        if contour is None or len(contour) < 3:
            return "未知方向"
        
        width, height = image_size
        
        try:
            # 计算轮廓的几何特征
            contour_points = contour.reshape(-1, 2)
            
            # 计算质心
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return "未知方向"
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # 分析轮廓在不同高度的宽度
            width_analysis = self._analyze_contour_width(contour_points, height)
            
            # 综合判断方向
            direction = self._determine_direction(cx, cy, width, height, width_analysis)
            
            # 更新历史记录
            self._update_direction_history(direction)
            
            # 使用历史记录平滑结果
            return self._get_smoothed_direction()
            
        except Exception as e:
            logger.error(f"方向分析失败: {str(e)}")
            return "未知方向"
    
    def _analyze_contour_width(self, contour_points: np.ndarray, image_height: int) -> Dict[str, float]:
        """分析轮廓宽度特征"""
        analysis = {}
        
        # 在不同高度分析宽度
        heights = [image_height * 0.3, image_height * 0.5, image_height * 0.7]
        
        for i, h in enumerate(heights):
            points_at_height = [p for p in contour_points if abs(p[1] - h) < 10]
            if len(points_at_height) >= 2:
                min_x = min(p[0] for p in points_at_height)
                max_x = max(p[0] for p in points_at_height)
                analysis[f"width_{i}"] = max_x - min_x
                analysis[f"center_{i}"] = (min_x + max_x) / 2
        
        return analysis
    
    def _determine_direction(self, cx: int, cy: int, width: int, height: int, 
                           width_analysis: Dict[str, float]) -> str:
        """确定道路方向"""
        image_center_x = width / 2
        
        # 基于质心位置判断
        deviation_ratio = (cx - image_center_x) / (width / 2)
        
        if abs(deviation_ratio) < self.config.center_deviation_threshold:
            base_direction = "直行"
        elif deviation_ratio > 0:
            base_direction = "右转"
        else:
            base_direction = "左转"
        
        # 基于宽度变化验证
        if len(width_analysis) >= 2:
            width_keys = [k for k in width_analysis.keys() if k.startswith('width_')]
            if len(width_keys) >= 2:
                top_width = width_analysis.get('width_0', 0)
                bottom_width = width_analysis.get('width_2', 0)
                
                if top_width > 0 and bottom_width > 0:
                    width_ratio = top_width / bottom_width
                    if width_ratio < self.config.width_ratio_threshold:
                        # 道路变窄，可能转弯
                        center_keys = [k for k in width_analysis.keys() if k.startswith('center_')]
                        if len(center_keys) >= 2:
                            top_center = width_analysis.get('center_0', image_center_x)
                            if top_center < image_center_x:
                                return "左转"
                            else:
                                return "右转"
        
        return base_direction
    
    def _update_direction_history(self, direction: str):
        """更新方向历史记录"""
        self.direction_history.append(direction)
        if len(self.direction_history) > self.history_size:
            self.direction_history.pop(0)
    
    def _get_smoothed_direction(self) -> str:
        """获取平滑后的方向（基于历史记录）"""
        if len(self.direction_history) == 0:
            return "未知方向"
        
        # 返回最近的方向
        return self.direction_history[-1]

class VisualizationEngine:
    """可视化引擎"""
    
    def __init__(self):
        self.colors = {
            'contour': (0, 255, 255),  # 黄色 - 轮廓
            'road_area': (0, 255, 0),   # 绿色 - 道路区域
            'left_lane': (0, 0, 255),   # 红色 - 左车道线
            'right_lane': (255, 0, 0),  # 蓝色 - 右车道线
            'direction': (0, 0, 255),   # 红色 - 方向指示
            'roi': (0, 255, 255),       # 黄色 - ROI区域
            'text': (255, 255, 255)     # 白色 - 文本
        }
    
    def draw_detection_results(self, image: np.ndarray, contour: np.ndarray, 
                             lane_lines: Dict[str, List], direction: str,
                             processing_time: float) -> np.ndarray:
        """绘制检测结果"""
        result = image.copy()
        height, width = result.shape[:2]
        
        # 绘制道路轮廓和区域
        if contour is not None:
            self._draw_road_contour(result, contour)
        
        # 绘制车道线
        self._draw_lane_lines(result, lane_lines)
        
        # 添加信息文本
        self._add_info_text(result, direction, processing_time, lane_lines)
        
        # 绘制方向指示器
        self._draw_direction_indicator(result, direction, width, height)
        
        return result
    
    def _draw_road_contour(self, image: np.ndarray, contour: np.ndarray):
        """绘制道路轮廓"""
        # 绘制轮廓线
        cv2.drawContours(image, [contour], -1, self.colors['contour'], 3)
        
        # 填充道路区域（半透明）
        overlay = image.copy()
        cv2.fillPoly(overlay, [contour], self.colors['road_area'])
        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
    
    def _draw_lane_lines(self, image: np.ndarray, lane_lines: Dict[str, List]):
        """绘制车道线"""
        for side, lines in lane_lines.items():
            color = self.colors['left_lane'] if side == 'left' else self.colors['right_lane']
            for line in lines:
                x1, y1, x2, y2, slope = line
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    def _add_info_text(self, image: np.ndarray, direction: str, 
                      processing_time: float, lane_lines: Dict[str, List]):
        """添加信息文本"""
        text_color = self.colors['text']
        bg_color = (0, 0, 0)
        
        # 方向信息
        direction_text = f"方向: {direction}"
        cv2.putText(image, direction_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        # 处理时间
        time_text = f"处理时间: {processing_time:.2f}秒"
        cv2.putText(image, time_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # 检测统计
        left_count = len(lane_lines.get('left', []))
        right_count = len(lane_lines.get('right', []))
        stats_text = f"左线: {left_count}, 右线: {right_count}"
        cv2.putText(image, stats_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    def _draw_direction_indicator(self, image: np.ndarray, direction: str, 
                                width: int, height: int):
        """绘制方向指示器"""
        center_x, center_y = width // 2, height // 2
        arrow_length = min(width, height) // 6
        
        if direction == "左转":
            end_point = (center_x - arrow_length, center_y)
        elif direction == "右转":
            end_point = (center_x + arrow_length, center_y)
        else:  # 直行
            end_point = (center_x, center_y - arrow_length)
        
        cv2.arrowedLine(image, (center_x, center_y), end_point, 
                       self.colors['direction'], 8, tipLength=0.3)

class LaneDetectionApp:
    """主应用程序"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("智能道路方向识别系统")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # 初始化组件
        self.config = DetectionConfig()
        self.road_detector = RoadDetector(self.config)
        self.direction_analyzer = DirectionAnalyzer(self.config)
        self.visualization_engine = VisualizationEngine()
        self.image_processor = ImageProcessor(self.config)
        
        # 状态变量
        self.current_image_path = None
        self.original_image = None
        self.is_processing = False
        
        # 创建UI
        self._create_ui()
        
        # 加载配置
        self._load_config()
        
        logger.info("应用程序初始化完成")
    
    def _create_ui(self):
        """创建用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 创建控件
        self._create_control_panel(main_frame)
        self._create_image_display(main_frame)
        self._create_status_bar(main_frame)
    
    def _create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 文件操作按钮
        file_frame = ttk.Frame(control_frame)
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="选择图片", 
                  command=self._select_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="重新检测", 
                  command=self._redetect).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="保存结果", 
                  command=self._save_result).pack(side=tk.LEFT)
        
        # 文件路径显示
        self.file_path_var = tk.StringVar(value="未选择图片")
        ttk.Label(file_frame, textvariable=self.file_path_var).pack(side=tk.LEFT, padx=(20, 0))
        
        # 参数调整
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Label(param_frame, text="敏感度:").grid(row=0, column=0, sticky=tk.W)
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        ttk.Scale(param_frame, from_=0.1, to=1.0, variable=self.sensitivity_var,
                 command=self._on_sensitivity_change).grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # 结果显示
        self.result_var = tk.StringVar(value="等待检测...")
        result_label = ttk.Label(control_frame, textvariable=self.result_var, 
                                font=("Arial", 12, "bold"), foreground="blue")
        result_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
    
    def _create_image_display(self, parent):
        """创建图像显示区域"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # 原图显示
        original_frame = ttk.LabelFrame(display_frame, text="原始图像", padding="5")
        original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        
        self.original_label = ttk.Label(original_frame, text="请选择道路图片", 
                                       relief="solid", background="white")
        self.original_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 结果图显示
        result_frame = ttk.LabelFrame(display_frame, text="检测结果", padding="5")
        result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        self.result_label = ttk.Label(result_frame, text="检测结果将显示在这里", 
                                     relief="solid", background="white")
        self.result_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def _create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 进度条
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 状态文本
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief="sunken")
        status_label.pack(side=tk.RIGHT, padx=(10, 0))
    
    def _load_config(self):
        """加载配置"""
        try:
            if os.path.exists("config.json"):
                with open("config.json", "r") as f:
                    config_data = json.load(f)
                    # 更新配置参数
                    # 这里可以添加配置加载逻辑
                    logger.info("配置加载成功")
        except Exception as e:
            logger.warning(f"配置加载失败: {str(e)}")
    
    def _save_config(self):
        """保存配置"""
        try:
            config_data = {
                "sensitivity": self.sensitivity_var.get()
            }
            with open("config.json", "w") as f:
                json.dump(config_data, f, indent=2)
            logger.info("配置保存成功")
        except Exception as e:
            logger.error(f"配置保存失败: {str(e)}")
    
    def _select_image(self):
        """选择图片文件"""
        if self.is_processing:
            messagebox.showwarning("警告", "正在处理中，请稍候...")
            return
        
        file_types = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("所有文件", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(title="选择道路图片", filetypes=file_types)
        
        if file_path:
            self.current_image_path = file_path
            self.file_path_var.set(os.path.basename(file_path))
            self._load_and_display_original_image(file_path)
            self._start_detection()
    
    def _load_and_display_original_image(self, file_path: str):
        """加载并显示原始图像"""
        try:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                raise ValueError("无法读取图像文件")
            
            # 调整图像尺寸
            self.original_image = self.image_processor.resize_image(self.original_image)
            
            # 显示图像
            self._display_image(self.original_image, self.original_label)
            
            self.status_var.set("图片加载成功")
            logger.info(f"图片加载成功: {file_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
            logger.error(f"图片加载失败: {str(e)}")
    
    def _display_image(self, image: np.ndarray, label: ttk.Label):
        """在标签中显示图像"""
        try:
            # 转换颜色空间
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # 调整尺寸以适应显示区域
            label_width = label.winfo_width() or 400
            label_height = label.winfo_height() or 300
            
            pil_image.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)
            
            # 转换为Tkinter格式
            photo = ImageTk.PhotoImage(pil_image)
            
            # 更新标签
            label.configure(image=photo, text="")
            label.image = photo  # 保持引用
            
        except Exception as e:
            logger.error(f"图像显示失败: {str(e)}")
            label.configure(image="", text="图像显示失败")
    
    def _start_detection(self):
        """开始检测"""
        if self.is_processing or self.original_image is None:
            return
        
        self.is_processing = True
        self.progress.start()
        self.status_var.set("正在分析道路方向...")
        self.result_var.set("检测中...")
        
        # 在后台线程中执行检测
        thread = threading.Thread(target=self._detection_thread)
        thread.daemon = True
        thread.start()
    
    def _detection_thread(self):
        """检测线程"""
        try:
            start_time = time.time()
            
            # 检测道路区域
            road_mask = self.road_detector.detect_road_region(self.original_image)
            
            # 提取道路轮廓
            road_contour = self.road_detector.extract_road_contour(road_mask) if road_mask is not None else None
            
            # 检测车道线
            lane_lines = self.road_detector.detect_lane_lines(self.original_image)
            
            # 分析方向
            direction = self.direction_analyzer.analyze_from_contour(
                road_contour, (self.original_image.shape[1], self.original_image.shape[0])
            )
            
            processing_time = time.time() - start_time
            self.road_detector.last_processing_time = processing_time
            
            # 生成结果图像
            result_image = self.visualization_engine.draw_detection_results(
                self.original_image, road_contour, lane_lines, direction, processing_time
            )
            
            # 在主线程中更新UI
            self.root.after(0, self._update_results, direction, result_image, processing_time)
            
        except Exception as e:
            logger.error(f"检测过程出错: {str(e)}")
            self.root.after(0, self._show_error, f"检测失败: {str(e)}")
    
    def _update_results(self, direction: str, result_image: np.ndarray, processing_time: float):
        """更新检测结果"""
        self.is_processing = False
        self.progress.stop()
        
        # 显示结果图像
        self._display_image(result_image, self.result_label)
        
        # 更新结果文本
        self.result_var.set(f"检测结果: {direction}")
        self.status_var.set(f"分析完成 - 耗时: {processing_time:.2f}秒")
        
        # 根据方向设置文本颜色
        color_map = {
            "左转": "red",
            "右转": "red", 
            "直行": "green",
            "未知方向": "orange"
        }
        self.result_label.configure(foreground=color_map.get(direction, "black"))
        
        logger.info(f"检测完成: {direction}, 耗时: {processing_time:.2f}秒")
    
    def _show_error(self, error_msg: str):
        """显示错误信息"""
        self.is_processing = False
        self.progress.stop()
        messagebox.showerror("错误", error_msg)
        self.status_var.set("检测失败")
        self.result_var.set("检测失败")
        logger.error(f"检测错误: {error_msg}")
    
    def _redetect(self):
        """重新检测"""
        if self.current_image_path and not self.is_processing:
            self._start_detection()
    
    def _save_result(self):
        """保存结果"""
        if self.current_image_path and hasattr(self.result_label, 'image'):
            file_path = filedialog.asksaveasfilename(
                title="保存结果图片",
                defaultextension=".jpg",
                filetypes=[("JPEG文件", "*.jpg"), ("PNG文件", "*.png"), ("所有文件", "*.*")]
            )
            
            if file_path:
                try:
                    # 这里需要保存结果图像，需要额外处理
                    messagebox.showinfo("成功", "保存功能待实现")
                    logger.info(f"结果保存到: {file_path}")
                except Exception as e:
                    messagebox.showerror("错误", f"保存失败: {str(e)}")
                    logger.error(f"保存失败: {str(e)}")
    
    def _on_sensitivity_change(self, value):
        """敏感度变化回调"""
        sensitivity = float(value)
        # 根据敏感度调整配置参数
        self.config.width_ratio_threshold = 0.3 + sensitivity * 0.4
        self.config.center_deviation_threshold = 0.1 + sensitivity * 0.1
        logger.info(f"敏感度调整为: {sensitivity:.2f}")
    
    def run(self):
        """运行应用程序"""
        try:
            self.root.mainloop()
        except Exception as e:
            logger.error(f"应用程序运行错误: {str(e)}")
        finally:
            self._save_config()

def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = LaneDetectionApp(root)
        app.run()
    except Exception as e:
        logger.critical(f"应用程序启动失败: {str(e)}")
        messagebox.showerror("致命错误", f"应用程序启动失败: {str(e)}")

if __name__ == "__main__":
    main()