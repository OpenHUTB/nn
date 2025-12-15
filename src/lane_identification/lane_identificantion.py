import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable
from collections import deque

# ==================== 配置类 ====================
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
    
    # 形态学操作参数
    morph_kernel_size: int = 5
    blur_kernel_size: int = 5
    
    # ROI参数
    roi_top_ratio: float = 0.4
    roi_bottom_ratio: float = 0.9
    
    # 方向判断阈值
    width_ratio_threshold: float = 0.7
    center_deviation_threshold: float = 0.15
    
    # 路径预测参数
    prediction_steps: int = 5
    prediction_distance: float = 0.7
    
    # 性能优化参数
    max_image_size: Tuple[int, int] = (1200, 800)
    cache_size: int = 3

# ==================== 图像处理器 ====================
class ImageProcessor:
    """高效图像处理器"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self._cache = {}  # 简单的处理结果缓存
        
    def process_image(self, image_path: str) -> Optional[np.ndarray]:
        """加载并预处理图像"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # 缓存键
            cache_key = f"{image_path}_{image.shape}"
            
            # 检查缓存
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # 调整图像尺寸
            processed = self._resize_image(image)
            
            # 增强对比度
            processed = self._enhance_contrast(processed)
            
            # 去除阴影
            processed = self._remove_shadows(processed)
            
            # 更新缓存
            if len(self._cache) >= self.config.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = processed
            
            return processed
            
        except Exception as e:
            print(f"图像处理错误: {e}")
            return None
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """智能调整图像尺寸"""
        h, w = image.shape[:2]
        max_w, max_h = self.config.max_image_size
        
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """快速对比度增强"""
        # 使用YUV颜色空间进行亮度调整
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0]
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(y_channel)
        
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    def _remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """快速阴影去除"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用形态学操作移除阴影
        dilated = cv2.dilate(gray, np.ones((3, 3), np.uint8))
        blurred = cv2.medianBlur(dilated, 15)
        diff = 255 - cv2.absdiff(gray, blurred)
        normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # 将处理后的灰度图转回BGR
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

# ==================== 道路检测器 ====================
class RoadDetector:
    """高效道路检测器"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        
    def detect_road(self, image: np.ndarray) -> Dict[str, Any]:
        """快速道路检测"""
        height, width = image.shape[:2]
        
        try:
            # 1. 提取ROI
            roi_vertices = self._get_roi_vertices(width, height)
            roi_image = self._extract_roi(image, roi_vertices)
            
            # 2. 多方法道路检测
            road_mask = self._detect_road_mask(roi_image)
            
            # 3. 提取道路轮廓
            road_contour = self._extract_contour(road_mask)
            
            return {
                'roi_vertices': roi_vertices,
                'road_mask': road_mask,
                'road_contour': road_contour,
                'image_size': (width, height)
            }
            
        except Exception as e:
            print(f"道路检测错误: {e}")
            return {'roi_vertices': None, 'road_mask': None, 'road_contour': None}
    
    def _get_roi_vertices(self, width: int, height: int) -> np.ndarray:
        """计算ROI顶点"""
        return np.array([[
            (width * 0.1, height * self.config.roi_bottom_ratio),
            (width * 0.4, height * self.config.roi_top_ratio),
            (width * 0.6, height * self.config.roi_top_ratio),
            (width * 0.9, height * self.config.roi_bottom_ratio)
        ]], dtype=np.int32)
    
    def _extract_roi(self, image: np.ndarray, roi_vertices: np.ndarray) -> np.ndarray:
        """提取ROI区域"""
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, roi_vertices, (255, 255, 255))
        return cv2.bitwise_and(image, mask)
    
    def _detect_road_mask(self, roi_image: np.ndarray) -> np.ndarray:
        """检测道路掩码"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        
        # 创建道路掩码
        lower = np.array(self.config.hsv_lower)
        upper = np.array(self.config.hsv_upper)
        road_mask = cv2.inRange(hsv, lower, upper)
        
        # 形态学优化
        kernel = np.ones((self.config.morph_kernel_size, self.config.morph_kernel_size), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        
        return road_mask
    
    def _extract_contour(self, road_mask: np.ndarray) -> Optional[np.ndarray]:
        """提取道路轮廓"""
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 简化轮廓
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        return simplified

# ==================== 车道线检测器 ====================
class LaneDetector:
    """高效车道线检测器"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def detect_lanes(self, image: np.ndarray, roi_vertices: np.ndarray) -> Dict[str, Any]:
        """检测车道线"""
        height, width = image.shape[:2]
        
        try:
            # 1. 预处理图像
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (self.config.blur_kernel_size, self.config.blur_kernel_size), 0)
            
            # 2. 边缘检测
            edges = cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)
            
            # 3. 应用ROI
            mask = np.zeros_like(edges)
            cv2.fillPoly(mask, roi_vertices, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # 4. 霍夫变换检测直线
            lines = cv2.HoughLinesP(
                masked_edges,
                self.config.hough_rho,
                self.config.hough_theta,
                self.config.hough_threshold,
                minLineLength=self.config.hough_min_length,
                maxLineGap=self.config.hough_max_gap
            )
            
            if lines is None:
                return self._empty_result()
            
            # 5. 分类和处理车道线
            left_lines, right_lines = self._classify_lines(lines, width)
            
            # 6. 拟合车道线
            left_lane = self._fit_lane(left_lines, height) if left_lines else None
            right_lane = self._fit_lane(right_lines, height) if right_lines else None
            
            # 7. 预测未来路径
            future_path = self._predict_path(left_lane, right_lane, height) if left_lane and right_lane else None
            
            return {
                'left_lines': left_lines,
                'right_lines': right_lines,
                'left_lane': left_lane,
                'right_lane': right_lane,
                'future_path': future_path,
                'num_lines': len(lines)
            }
            
        except Exception as e:
            print(f"车道线检测错误: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'left_lines': [], 'right_lines': [],
            'left_lane': None, 'right_lane': None,
            'future_path': None, 'num_lines': 0
        }
    
    def _classify_lines(self, lines: np.ndarray, image_width: int) -> Tuple[List, List]:
        """分类左右车道线"""
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 跳过垂直线
            if x2 == x1:
                continue
            
            # 计算斜率
            slope = (y2 - y1) / (x2 - x1)
            
            # 过滤水平线
            if abs(slope) < 0.3:
                continue
            
            # 分类左右车道线
            if slope < 0 and x1 < image_width * 0.6 and x2 < image_width * 0.6:
                left_lines.append((x1, y1, x2, y2, slope))
            elif slope > 0 and x1 > image_width * 0.4 and x2 > image_width * 0.4:
                right_lines.append((x1, y1, x2, y2, slope))
        
        return left_lines, right_lines
    
    def _fit_lane(self, lines: List, height: int) -> Optional[Dict]:
        """拟合单条车道线"""
        if len(lines) < 2:
            return None
        
        # 收集所有点
        x_points, y_points = [], []
        for x1, y1, x2, y2, _ in lines:
            x_points.extend([x1, x2])
            y_points.extend([y1, y2])
        
        # 线性拟合
        coeffs = np.polyfit(y_points, x_points, 1)
        poly_func = np.poly1d(coeffs)
        
        # 计算车道线端点
        y_bottom, y_top = height, int(height * self.config.roi_top_ratio)
        x_bottom, x_top = int(poly_func(y_bottom)), int(poly_func(y_top))
        
        return {
            'func': poly_func,
            'points': [(x_bottom, y_bottom), (x_top, y_top)],
            'confidence': min(len(lines) / 8.0, 1.0)
        }
    
    def _predict_path(self, left_lane: Dict, right_lane: Dict, height: int) -> Dict:
        """预测未来路径"""
        left_func = left_lane['func']
        right_func = right_lane['func']
        
        # 计算中心线函数
        def center_func(y):
            return (left_func(y) + right_func(y)) / 2
        
        # 生成预测点
        current_y = height
        target_y = int(height * (1 - self.config.prediction_distance))
        y_values = np.linspace(current_y, target_y, self.config.prediction_steps)
        
        # 计算路径点
        path_points = []
        for y in y_values:
            x = center_func(y)
            path_points.append((int(x), int(y)))
        
        return {
            'center_path': path_points,
            'prediction_distance': self.config.prediction_distance
        }

# ==================== 方向分析器 ====================
class DirectionAnalyzer:
    """智能方向分析器"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.history = deque(maxlen=5)  # 历史记录队列
        self.confidence_history = deque(maxlen=5)
    
    def analyze(self, road_contour: np.ndarray, lane_info: Dict[str, Any], 
                image_size: Tuple[int, int]) -> Dict[str, Any]:
        """分析道路方向"""
        width, height = image_size
        
        try:
            # 1. 基于轮廓分析
            contour_dir, contour_conf = self._analyze_contour(road_contour, width, height)
            
            # 2. 基于车道线分析
            lane_dir, lane_conf = self._analyze_lanes(lane_info, width, height)
            
            # 3. 基于路径预测分析
            path_dir, path_conf = self._analyze_path(lane_info, width, height)
            
            # 4. 综合判断
            directions = [contour_dir, lane_dir, path_dir]
            confidences = [contour_conf, lane_conf, path_conf]
            
            # 过滤无效结果
            valid_indices = [i for i, conf in enumerate(confidences) 
                           if conf > 0.3 and directions[i] != "未知"]
            
            if not valid_indices:
                return self._default_result()
            
            # 加权投票
            final_dir, final_conf = self._weighted_vote(
                [directions[i] for i in valid_indices],
                [confidences[i] for i in valid_indices]
            )
            
            # 5. 历史平滑
            smoothed_dir, smoothed_conf = self._smooth_with_history(final_dir, final_conf)
            
            return {
                'direction': smoothed_dir,
                'confidence': smoothed_conf,
                'source': '综合判断',
                'details': {
                    'contour': contour_dir,
                    'lanes': lane_dir,
                    'path': path_dir
                }
            }
            
        except Exception as e:
            print(f"方向分析错误: {e}")
            return self._default_result()
    
    def _default_result(self) -> Dict[str, Any]:
        """返回默认结果"""
        return {
            'direction': '未知',
            'confidence': 0.0,
            'source': '错误',
            'details': {}
        }
    
    def _analyze_contour(self, contour: np.ndarray, width: int, height: int) -> Tuple[str, float]:
        """基于轮廓分析方向"""
        if contour is None or len(contour) < 3:
            return "未知", 0.0
        
        # 计算轮廓质心
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return "未知", 0.0
        
        cx = int(M["m10"] / M["m00"])
        
        # 判断方向
        center_x = width / 2
        deviation = (cx - center_x) / (width / 2)
        
        if abs(deviation) < self.config.center_deviation_threshold:
            return "直行", 0.7 - abs(deviation)
        elif deviation > 0:
            return "右转", min(0.8, abs(deviation))
        else:
            return "左转", min(0.8, abs(deviation))
    
    def _analyze_lanes(self, lane_info: Dict[str, Any], width: int, height: int) -> Tuple[str, float]:
        """基于车道线分析方向"""
        if not lane_info.get('left_lane') or not lane_info.get('right_lane'):
            return "未知", 0.0
        
        left_func = lane_info['left_lane']['func']
        right_func = lane_info['right_lane']['func']
        
        # 计算顶部中心点
        y_top = int(height * self.config.roi_top_ratio)
        center_top = (left_func(y_top) + right_func(y_top)) / 2
        
        # 判断方向
        center_x = width / 2
        deviation = (center_top - center_x) / (width / 2)
        
        if abs(deviation) < 0.1:
            return "直行", 0.7
        elif deviation > 0:
            return "右转", min(0.8, abs(deviation))
        else:
            return "左转", min(0.8, abs(deviation))
    
    def _analyze_path(self, lane_info: Dict[str, Any], width: int, height: int) -> Tuple[str, float]:
        """基于路径预测分析方向"""
        if not lane_info.get('future_path'):
            return "未知", 0.0
        
        path = lane_info['future_path']['center_path']
        if len(path) < 2:
            return "未知", 0.0
        
        # 分析路径走向
        start_x, start_y = path[0]
        end_x, end_y = path[-1]
        
        # 计算水平偏移
        horizontal_shift = end_x - start_x
        
        # 判断方向
        if abs(horizontal_shift) < width * 0.05:
            return "直行", 0.6
        elif horizontal_shift > 0:
            confidence = min(0.8, abs(horizontal_shift) / (width * 0.2))
            return "右转", confidence
        else:
            confidence = min(0.8, abs(horizontal_shift) / (width * 0.2))
            return "左转", confidence
    
    def _weighted_vote(self, directions: List[str], confidences: List[float]) -> Tuple[str, float]:
        """加权投票"""
        scores = {}
        for dir, conf in zip(directions, confidences):
            scores[dir] = scores.get(dir, 0) + conf
        
        best_dir = max(scores.items(), key=lambda x: x[1])[0]
        total_score = sum(scores.values())
        confidence = scores[best_dir] / total_score if total_score > 0 else 0
        
        return best_dir, confidence
    
    def _smooth_with_history(self, direction: str, confidence: float) -> Tuple[str, float]:
        """历史平滑"""
        self.history.append(direction)
        self.confidence_history.append(confidence)
        
        if len(self.history) == self.history.maxlen:
            # 统计最频繁的方向
            from collections import Counter
            freq = Counter(self.history)
            most_common = freq.most_common(1)[0]
            
            if most_common[1] >= 3:  # 至少出现3次
                # 计算平均置信度
                indices = [i for i, d in enumerate(self.history) if d == most_common[0]]
                avg_conf = sum(self.confidence_history[i] for i in indices) / len(indices)
                return most_common[0], avg_conf
        
        return direction, confidence

# ==================== 可视化引擎 ====================
class Visualizer:
    """高效可视化引擎"""
    
    def __init__(self):
        self.colors = {
            'contour': (0, 255, 255),    # 黄色 - 轮廓
            'road_area': (0, 255, 0),     # 绿色 - 道路区域
            'left_lane': (255, 0, 0),     # 蓝色 - 左车道线
            'right_lane': (0, 0, 255),    # 红色 - 右车道线
            'center_line': (255, 255, 0), # 青色 - 中心线
            'future_path': (255, 0, 255), # 紫色 - 未来路径
            'direction': (0, 0, 255),     # 红色 - 方向指示
            'roi': (0, 255, 255),         # 黄色 - ROI区域
            'text': (255, 255, 255),      # 白色 - 文本
            'text_bg': (0, 0, 0, 180)     # 黑色半透明背景
        }
    
    def draw_results(self, image: np.ndarray, road_info: Dict[str, Any], 
                     lane_info: Dict[str, Any], direction_info: Dict[str, Any]) -> np.ndarray:
        """绘制检测结果"""
        result = image.copy()
        
        # 绘制道路信息
        if road_info.get('road_contour') is not None:
            self._draw_road(result, road_info)
        
        # 绘制车道线信息
        self._draw_lanes(result, lane_info)
        
        # 绘制文本信息
        self._draw_info(result, direction_info)
        
        return result
    
    def _draw_road(self, image: np.ndarray, road_info: Dict[str, Any]):
        """绘制道路信息"""
        # 绘制ROI边界
        if road_info.get('roi_vertices') is not None:
            cv2.polylines(image, [road_info['roi_vertices']], True, self.colors['roi'], 2)
        
        # 绘制道路轮廓
        contour = road_info['road_contour']
        cv2.drawContours(image, [contour], -1, self.colors['contour'], 3)
        
        # 填充道路区域（半透明）
        overlay = image.copy()
        cv2.fillPoly(overlay, [contour], self.colors['road_area'])
        cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)
    
    def _draw_lanes(self, image: np.ndarray, lane_info: Dict[str, Any]):
        """绘制车道线信息"""
        # 绘制原始车道线段
        for side, color in [('left_lines', self.colors['left_lane']), 
                          ('right_lines', self.colors['right_lane'])]:
            for line in lane_info.get(side, []):
                x1, y1, x2, y2, _ = line
                cv2.line(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制拟合的车道线
        for side, color in [('left_lane', self.colors['left_lane']), 
                          ('right_lane', self.colors['right_lane'])]:
            lane = lane_info.get(side)
            if lane and 'points' in lane:
                points = lane['points']
                if len(points) == 2:
                    cv2.line(image, points[0], points[1], color, 4)
        
        # 绘制未来路径
        future_path = lane_info.get('future_path')
        if future_path and 'center_path' in future_path:
            path_points = future_path['center_path']
            if len(path_points) >= 2:
                # 绘制路径线
                for i in range(len(path_points) - 1):
                    cv2.line(image, path_points[i], path_points[i + 1], 
                            self.colors['future_path'], 3, cv2.LINE_AA)
                
                # 绘制路径点
                for point in path_points:
                    cv2.circle(image, point, 4, self.colors['future_path'], -1)
    
    def _draw_info(self, image: np.ndarray, direction_info: Dict[str, Any]):
        """绘制文本信息"""
        height, width = image.shape[:2]
        direction = direction_info.get('direction', '未知')
        confidence = direction_info.get('confidence', 0.0)
        
        # 绘制半透明背景
        bg_height = 100
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # 绘制方向文本
        direction_text = f"方向: {direction}"
        confidence_text = f"置信度: {confidence:.1%}"
        
        # 根据置信度设置颜色
        if confidence > 0.7:
            color = (0, 255, 0)  # 绿色
        elif confidence > 0.5:
            color = (0, 165, 255)  # 橙色
        else:
            color = (0, 0, 255)  # 红色
        
        # 绘制文本
        y_offset = 30
        cv2.putText(image, direction_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        cv2.putText(image, confidence_text, (20, y_offset + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 绘制方向指示器
        self._draw_direction_indicator(image, direction, confidence, width, height)

    def _draw_direction_indicator(self, image: np.ndarray, direction: str, 
                                 confidence: float, width: int, height: int):
        """绘制方向指示器"""
        center_x, center_y = width // 2, height - 100
        arrow_length = 80
        
        if direction == "左转":
            end_point = (center_x - arrow_length, center_y)
            color = (0, 0, 255)  # 红色
        elif direction == "右转":
            end_point = (center_x + arrow_length, center_y)
            color = (0, 0, 255)  # 红色
        else:  # 直行
            end_point = (center_x, center_y - arrow_length)
            color = (0, 255, 0)  # 绿色
        
        # 根据置信度调整箭头大小
        thickness = int(6 * confidence + 3)
        cv2.arrowedLine(image, (center_x, center_y), end_point, color, thickness, tipLength=0.3)

# ==================== 主应用程序 ====================
class LaneDetectionApp:
    """主应用程序"""
    
    def __init__(self, root):
        self.root = root
        self._setup_window()
        
        # 初始化组件
        self.config = DetectionConfig()
        self.image_processor = ImageProcessor(self.config)
        self.road_detector = RoadDetector(self.config)
        self.lane_detector = LaneDetector(self.config)
        self.direction_analyzer = DirectionAnalyzer(self.config)
        self.visualizer = Visualizer()
        
        # 状态变量
        self.current_image = None
        self.is_processing = False
        
        # 创建界面
        self._create_ui()
        
        print("道路方向识别系统已启动")
    
    def _setup_window(self):
        """设置窗口"""
        self.root.title("智能道路方向识别系统")
        self.root.geometry("1200x700")
        self.root.minsize(1000, 600)
        self.root.configure(bg="#f0f0f0")
    
    def _create_ui(self):
        """创建用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # 配置权重
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # 控制面板
        self._create_control_panel(main_frame)
        
        # 图像显示区域
        self._create_display_area(main_frame)
        
        # 状态栏
        self._create_status_bar(main_frame)
    
    def _create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # 按钮框架
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", pady=(0, 10))
        
        # 按钮
        ttk.Button(button_frame, text="选择图片", 
                  command=self._select_image, width=15).pack(side="left", padx=(0, 10))
        ttk.Button(button_frame, text="重新检测", 
                  command=self._redetect, width=15).pack(side="left")
        
        # 文件路径显示
        self.file_label = ttk.Label(control_frame, text="未选择图片", 
                                   font=("Arial", 10), foreground="blue")
        self.file_label.pack(anchor="w", pady=(0, 10))
        
        # 参数调整
        self._create_parameter_controls(control_frame)
        
        # 结果显示
        self.result_label = ttk.Label(control_frame, text="等待检测...", 
                                     font=("Arial", 12, "bold"), foreground="blue")
        self.result_label.pack(anchor="w", pady=(5, 0))
        
        self.confidence_label = ttk.Label(control_frame, text="", 
                                         font=("Arial", 10), foreground="green")
        self.confidence_label.pack(anchor="w")
    
    def _create_parameter_controls(self, parent):
        """创建参数控制"""
        param_frame = ttk.Frame(parent)
        param_frame.pack(fill="x", pady=(0, 10))
        
        # 敏感度调整
        ttk.Label(param_frame, text="检测敏感度:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        sensitivity_scale = ttk.Scale(param_frame, from_=0.1, to=1.0, 
                                     variable=self.sensitivity_var, orient="horizontal",
                                     command=self._on_parameter_change, length=200)
        sensitivity_scale.grid(row=0, column=1, sticky="ew", padx=(0, 20))
        
        # 预测距离调整
        ttk.Label(param_frame, text="预测距离:").grid(row=0, column=2, sticky="w", padx=(0, 10))
        self.prediction_var = tk.DoubleVar(value=self.config.prediction_distance)
        prediction_scale = ttk.Scale(param_frame, from_=0.3, to=0.9,
                                   variable=self.prediction_var, orient="horizontal",
                                   command=self._on_parameter_change, length=200)
        prediction_scale.grid(row=0, column=3, sticky="ew")
    
    def _create_display_area(self, parent):
        """创建图像显示区域"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_columnconfigure(1, weight=1)
        
        # 原图显示
        original_frame = ttk.LabelFrame(display_frame, text="原始图像", padding="5")
        original_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        original_frame.grid_rowconfigure(0, weight=1)
        original_frame.grid_columnconfigure(0, weight=1)
        
        self.original_canvas = tk.Canvas(original_frame, bg="white", highlightthickness=1)
        self.original_canvas.grid(row=0, column=0, sticky="nsew")
        self.original_canvas.create_text(300, 200, text="请选择道路图片", 
                                        font=("Arial", 14), fill="gray")
        
        # 结果图显示
        result_frame = ttk.LabelFrame(display_frame, text="检测结果", padding="5")
        result_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        result_frame.grid_rowconfigure(0, weight=1)
        result_frame.grid_columnconfigure(0, weight=1)
        
        self.result_canvas = tk.Canvas(result_frame, bg="white", highlightthickness=1)
        self.result_canvas.grid(row=0, column=0, sticky="nsew")
        self.result_canvas.create_text(300, 200, text="检测结果将显示在这里", 
                                      font=("Arial", 14), fill="gray")
    
    def _create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        
        # 进度条
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        # 状态文本
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                relief="sunken", padding=(5, 2))
        status_label.pack(side="right")
    
    def _select_image(self):
        """选择图片"""
        if self.is_processing:
            messagebox.showwarning("提示", "正在处理中，请稍候...")
            return
        
        file_types = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp"),
            ("所有文件", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(title="选择道路图片", filetypes=file_types)
        
        if file_path:
            self.file_label.config(text=os.path.basename(file_path))
            self._load_image(file_path)
            self._start_detection()
    
    def _load_image(self, file_path: str):
        """加载图像"""
        try:
            self.current_image = self.image_processor.process_image(file_path)
            if self.current_image is None:
                raise ValueError("无法读取图像")
            
            self._display_image(self.current_image, self.original_canvas)
            self.status_var.set("图片加载成功")
            
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
            print(f"图片加载失败: {e}")
    
    def _display_image(self, image: np.ndarray, canvas: tk.Canvas):
        """在Canvas上显示图像"""
        try:
            canvas.delete("all")
            
            # 转换颜色空间
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # 获取Canvas尺寸
            canvas_width = canvas.winfo_width() or 400
            canvas_height = canvas.winfo_height() or 300
            
            # 调整图像大小
            pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            # 转换为Tkinter格式
            photo = ImageTk.PhotoImage(pil_image)
            
            # 计算居中位置
            x = (canvas_width - photo.width()) // 2
            y = (canvas_height - photo.height()) // 2
            
            # 显示图像
            canvas.create_image(x, y, anchor="nw", image=photo)
            canvas.image = photo  # 保持引用
            
        except Exception as e:
            print(f"图像显示失败: {e}")
            canvas.create_text(150, 150, text="图像显示失败", fill="red")
    
    def _start_detection(self):
        """开始检测"""
        if self.is_processing or self.current_image is None:
            return
        
        self.is_processing = True
        self.progress.start()
        self.status_var.set("正在分析道路方向...")
        self.result_label.config(text="检测中...", foreground="blue")
        self.confidence_label.config(text="")
        
        # 在后台线程中执行检测
        thread = threading.Thread(target=self._detection_thread)
        thread.daemon = True
        thread.start()
    
    def _detection_thread(self):
        """检测线程"""
        try:
            start_time = time.time()
            
            # 1. 检测道路
            road_info = self.road_detector.detect_road(self.current_image)
            
            # 2. 检测车道线
            lane_info = self.lane_detector.detect_lanes(
                self.current_image, road_info.get('roi_vertices', np.array([]))
            )
            
            # 3. 分析方向
            direction_info = self.direction_analyzer.analyze(
                road_info.get('road_contour'), 
                lane_info, 
                road_info.get('image_size', self.current_image.shape[1::-1])
            )
            
            # 4. 生成结果图像
            result_image = self.visualizer.draw_results(
                self.current_image, road_info, lane_info, direction_info
            )
            
            processing_time = time.time() - start_time
            
            # 在主线程中更新UI
            self.root.after(0, self._update_results, direction_info, result_image, processing_time)
            
        except Exception as e:
            print(f"检测过程出错: {e}")
            self.root.after(0, self._show_error, str(e))
    
    def _update_results(self, direction_info: Dict[str, Any], 
                       result_image: np.ndarray, processing_time: float):
        """更新结果"""
        self.is_processing = False
        self.progress.stop()
        
        # 显示结果图像
        self._display_image(result_image, self.result_canvas)
        
        # 更新文本信息
        direction = direction_info.get('direction', '未知')
        confidence = direction_info.get('confidence', 0.0)
        
        self.result_label.config(text=f"检测结果: {direction}")
        
        if confidence > 0:
            confidence_text = f"置信度: {confidence:.1%} | 耗时: {processing_time:.2f}秒"
            self.confidence_label.config(text=confidence_text)
            
            # 根据置信度设置颜色
            if confidence > 0.7:
                color = "green"
            elif confidence > 0.5:
                color = "orange"
            else:
                color = "red"
            
            self.confidence_label.config(foreground=color)
        
        self.status_var.set("分析完成")
        
        print(f"检测完成: {direction}, 置信度: {confidence:.1%}, 耗时: {processing_time:.2f}秒")
    
    def _show_error(self, error_msg: str):
        """显示错误"""
        self.is_processing = False
        self.progress.stop()
        
        messagebox.showerror("错误", f"检测失败: {error_msg}")
        self.status_var.set("检测失败")
        self.result_label.config(text="检测失败", foreground="red")
        self.confidence_label.config(text="")
    
    def _redetect(self):
        """重新检测"""
        if self.current_image is not None and not self.is_processing:
            self._start_detection()
    
    def _on_parameter_change(self, value):
        """参数变化回调"""
        # 更新配置
        sensitivity = self.sensitivity_var.get()
        prediction = self.prediction_var.get()
        
        self.config.width_ratio_threshold = 0.3 + sensitivity * 0.4
        self.config.center_deviation_threshold = 0.1 + sensitivity * 0.1
        self.config.prediction_distance = prediction
        
        print(f"参数更新 - 敏感度: {sensitivity:.2f}, 预测距离: {prediction:.2f}")
        
        # 自动重新检测
        if self.current_image and not self.is_processing:
            self._start_detection()

def main():
    """主函数"""
    try:
        # 创建主窗口
        root = tk.Tk()
        
        # 设置窗口图标和样式
        root.iconbitmap(default=None)  # 可以设置图标文件路径
        
        # 创建应用程序
        app = LaneDetectionApp(root)
        
        # 运行主循环
        root.mainloop()
        
    except Exception as e:
        print(f"应用程序启动失败: {e}")
        messagebox.showerror("致命错误", f"应用程序启动失败: {str(e)}")

if __name__ == "__main__":
    main()