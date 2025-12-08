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
    
    # 新算法参数
    roi_top_ratio: float = 0.4  # ROI顶部位置
    roi_bottom_ratio: float = 0.9  # ROI底部位置
    lane_width_ratio: float = 0.3  # 车道宽度比例
    min_contour_area_ratio: float = 0.02  # 最小轮廓面积比例

class EnhancedImageProcessor:
    """增强图像处理器"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
    
    def resize_image(self, image: np.ndarray, max_size: Tuple[int, int] = (1200, 800)) -> np.ndarray:
        """调整图像尺寸"""
        h, w = image.shape[:2]
        if w > max_size[0] or h > max_size[1]:
            scale = min(max_size[0] / w, max_size[1] / h)
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image
    
    def adaptive_enhancement(self, image: np.ndarray) -> np.ndarray:
        """自适应图像增强"""
        # 1. 直方图均衡化
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 2. 自适应伽马校正
        def adaptive_gamma_correction(img, gamma=1.0):
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(img, table)
        
        # 根据图像亮度自适应调整gamma
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 50:  # 太暗
            enhanced = adaptive_gamma_correction(enhanced, 0.7)
        elif mean_brightness > 200:  # 太亮
            enhanced = adaptive_gamma_correction(enhanced, 1.3)
        
        return enhanced
    
    def remove_shadows_and_glare(self, image: np.ndarray) -> np.ndarray:
        """去除阴影和反光"""
        # 使用同态滤波增强细节
        def homomorphic_filter(img):
            # 转换到对数域
            img_log = np.log1p(np.float32(img))
            
            # 傅里叶变换
            img_fft = np.fft.fft2(img_log)
            
            # 创建高斯高通滤波器
            rows, cols = img.shape[:2]
            crow, ccol = rows // 2, cols // 2
            d0 = 30
            gaussian_high = 1 - np.exp(-((np.arange(rows)[:, None] - crow)**2 + (np.arange(cols) - ccol)**2) / (2 * d0**2))
            
            # 应用滤波器
            img_fft_filtered = img_fft * gaussian_high
            
            # 逆傅里叶变换
            img_filtered = np.fft.ifft2(img_fft_filtered)
            img_filtered = np.exp(np.real(img_filtered)) - 1
            
            # 归一化
            img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            return img_filtered
        
        # 对每个通道分别处理
        if len(image.shape) == 3:
            b, g, r = cv2.split(image)
            b_filtered = homomorphic_filter(b)
            g_filtered = homomorphic_filter(g)
            r_filtered = homomorphic_filter(r)
            result = cv2.merge([b_filtered, g_filtered, r_filtered])
        else:
            result = homomorphic_filter(image)
        
        return result
    
    def extract_roi(self, image: np.ndarray) -> np.ndarray:
        """提取感兴趣区域"""
        height, width = image.shape[:2]
        
        # 定义梯形ROI区域，更专注于道路区域
        top_width = int(width * 0.15)
        bottom_width = int(width * 0.9)
        top_height = int(height * self.config.roi_top_ratio)
        bottom_height = int(height * self.config.roi_bottom_ratio)
        
        roi_vertices = np.array([[
            ((width - bottom_width) // 2, bottom_height),
            ((width - top_width) // 2, top_height),
            ((width + top_width) // 2, top_height),
            ((width + bottom_width) // 2, bottom_height)
        ]], dtype=np.int32)
        
        # 创建掩码
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, roi_vertices, (255, 255, 255))
        
        # 应用掩码
        roi_image = cv2.bitwise_and(image, mask)
        
        return roi_image, roi_vertices

class EnhancedRoadDetector:
    """增强道路检测器"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.image_processor = EnhancedImageProcessor(config)
        self.last_processing_time = 0
        
    def detect_road_region(self, image: np.ndarray) -> Dict[str, Any]:
        """检测道路区域 - 改进的多方法融合"""
        height, width = image.shape[:2]
        min_area = height * width * self.config.min_contour_area_ratio
        
        try:
            # 1. 图像预处理
            enhanced = self.image_processor.adaptive_enhancement(image)
            shadow_removed = self.image_processor.remove_shadows_and_glare(enhanced)
            
            # 2. 提取ROI
            roi_image, roi_vertices = self.image_processor.extract_roi(shadow_removed)
            
            # 3. 多方法道路检测
            methods_results = []
            
            # 方法1: HSV颜色空间分割
            hsv_result = self._detect_by_hsv(roi_image)
            methods_results.append(hsv_result)
            
            # 方法2: 基于边缘检测
            edge_result = self._detect_by_edges(roi_image)
            methods_results.append(edge_result)
            
            # 方法3: 基于灰度阈值
            gray_result = self._detect_by_gray(roi_image)
            methods_results.append(gray_result)
            
            # 4. 融合结果
            fused_mask = self._fuse_masks(methods_results)
            
            # 5. 形态学优化
            kernel = np.ones((self.config.morph_kernel_size, self.config.morph_kernel_size), np.uint8)
            fused_mask = cv2.morphologyEx(fused_mask, cv2.MORPH_CLOSE, kernel)
            fused_mask = cv2.morphologyEx(fused_mask, cv2.MORPH_OPEN, kernel)
            
            # 6. 提取轮廓
            contours, _ = cv2.findContours(fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    valid_contours.append(contour)
            
            if valid_contours:
                # 合并所有有效轮廓
                combined_contour = np.vstack(valid_contours)
                
                # 使用凸包简化轮廓
                hull = cv2.convexHull(combined_contour)
                
                # 使用多边形近似进一步简化
                epsilon = 0.005 * cv2.arcLength(hull, True)
                simplified_hull = cv2.approxPolyDP(hull, epsilon, True)
                
                return {
                    'mask': fused_mask,
                    'contour': simplified_hull,
                    'roi_vertices': roi_vertices,
                    'method_scores': [r['confidence'] for r in methods_results]
                }
            
            return {'mask': fused_mask, 'contour': None, 'roi_vertices': roi_vertices}
            
        except Exception as e:
            logger.error(f"道路区域检测失败: {str(e)}")
            return {'mask': None, 'contour': None, 'roi_vertices': None}
    
    def _detect_by_hsv(self, image: np.ndarray) -> Dict[str, Any]:
        """HSV颜色空间检测"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 多种颜色范围（适应不同道路颜色）
        color_ranges = [
            (np.array([0, 0, 50]), np.array([180, 50, 200])),  # 深色道路
            (np.array([0, 0, 100]), np.array([180, 30, 220])),  # 中等颜色
            (np.array([0, 0, 150]), np.array([180, 20, 240]))   # 浅色道路
        ]
        
        masks = []
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)
        
        # 合并所有掩码
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # 计算置信度（基于非零像素比例）
        total_pixels = mask.shape[0] * mask.shape[1]
        road_pixels = np.count_nonzero(combined_mask)
        confidence = road_pixels / total_pixels
        
        return {'mask': combined_mask, 'confidence': confidence}
    
    def _detect_by_edges(self, image: np.ndarray) -> Dict[str, Any]:
        """基于边缘检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Canny边缘检测
        edges = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)
        
        # 结合边缘和阈值
        combined = cv2.bitwise_or(thresh, edges)
        
        # 形态学操作连接边缘
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # 填充闭合区域
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(combined)
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                cv2.drawContours(filled_mask, [contour], -1, 255, -1)
        
        # 计算置信度
        total_pixels = filled_mask.shape[0] * filled_mask.shape[1]
        road_pixels = np.count_nonzero(filled_mask)
        confidence = road_pixels / total_pixels
        
        return {'mask': filled_mask, 'confidence': confidence}
    
    def _detect_by_gray(self, image: np.ndarray) -> Dict[str, Any]:
        """基于灰度阈值"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Otsu自动阈值
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 均值偏移分割
        shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
        gray_shifted = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        _, shift_thresh = cv2.threshold(gray_shifted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 结合两种方法
        combined = cv2.bitwise_and(otsu_thresh, shift_thresh)
        
        # 计算置信度
        total_pixels = combined.shape[0] * combined.shape[1]
        road_pixels = np.count_nonzero(combined)
        confidence = road_pixels / total_pixels
        
        return {'mask': combined, 'confidence': confidence}
    
    def _fuse_masks(self, method_results: List[Dict[str, Any]]) -> np.ndarray:
        """融合多个方法的掩码"""
        masks = [r['mask'] for r in method_results]
        confidences = [r['confidence'] for r in method_results]
        
        if not masks:
            return np.zeros((100, 100), dtype=np.uint8)
        
        # 加权融合
        total_confidence = sum(confidences)
        if total_confidence > 0:
            weights = [c / total_confidence for c in confidences]
            
            # 初始化融合掩码
            fused = np.zeros_like(masks[0], dtype=np.float32)
            
            for mask, weight in zip(masks, weights):
                fused += mask.astype(np.float32) * weight
            
            # 转换为二值图像
            fused_binary = (fused > 127).astype(np.uint8) * 255
            
            return fused_binary
        else:
            return masks[0]
    
    def detect_lane_lines_enhanced(self, image: np.ndarray, roi_vertices: np.ndarray) -> Dict[str, Any]:
        """增强车道线检测"""
        height, width = image.shape[:2]
        
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # 高斯模糊
            blur = cv2.GaussianBlur(gray, (self.config.blur_kernel_size, 
                                         self.config.blur_kernel_size), 0)
            
            # 边缘检测 - 使用Sobel算子增强车道线
            sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
            
            # 自适应阈值
            _, binary = cv2.threshold(gradient_magnitude, 0, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 应用ROI掩码
            mask = np.zeros_like(binary)
            cv2.fillPoly(mask, roi_vertices, 255)
            masked_edges = cv2.bitwise_and(binary, mask)
            
            # 概率霍夫变换检测直线
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=2,
                theta=np.pi/180,
                threshold=50,
                minLineLength=30,
                maxLineGap=100
            )
            
            if lines is None:
                return {"left_lines": [], "right_lines": [], "center_line": None}
            
            # 改进的线段分类和过滤
            left_segments, right_segments = self._classify_lanes_enhanced(lines, width)
            
            # 拟合车道线
            left_lane = self._fit_lane_line(left_segments, height)
            right_lane = self._fit_lane_line(right_segments, height)
            
            # 计算中心线
            center_line = self._calculate_center_line(left_lane, right_lane, height) \
                if left_lane and right_lane else None
            
            return {
                "left_lines": left_segments,
                "right_lines": right_segments,
                "left_lane": left_lane,
                "right_lane": right_lane,
                "center_line": center_line,
                "num_lines": len(lines)
            }
            
        except Exception as e:
            logger.error(f"车道线检测失败: {str(e)}")
            return {"left_lines": [], "right_lines": [], "center_line": None}
    
    def _classify_lanes_enhanced(self, lines: np.ndarray, image_width: int) -> Tuple[List, List]:
        """改进的车道线分类"""
        left_segments = []
        right_segments = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 过滤水平线
            if abs(y2 - y1) < 10:
                continue
            
            # 计算斜率和截距
            if x2 - x1 == 0:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 过滤不合理斜率的线段
            if abs(slope) < 0.2 or abs(slope) > 2.0:
                continue
            
            # 根据斜率和位置分类
            if slope < 0:  # 左车道线
                if x1 < image_width * 0.6 and x2 < image_width * 0.6:
                    left_segments.append({
                        'points': [(x1, y1), (x2, y2)],
                        'slope': slope,
                        'length': length,
                        'midpoint': ((x1 + x2)//2, (y1 + y2)//2)
                    })
            else:  # 右车道线
                if x1 > image_width * 0.4 and x2 > image_width * 0.4:
                    right_segments.append({
                        'points': [(x1, y1), (x2, y2)],
                        'slope': slope,
                        'length': length,
                        'midpoint': ((x1 + x2)//2, (y1 + y2)//2)
                    })
        
        # 过滤异常值（基于聚类）
        left_segments = self._filter_outliers(left_segments)
        right_segments = self._filter_outliers(right_segments)
        
        return left_segments, right_segments
    
    def _filter_outliers(self, segments: List) -> List:
        """过滤异常线段（基于斜率和位置聚类）"""
        if len(segments) < 2:
            return segments
        
        # 提取斜率
        slopes = [seg['slope'] for seg in segments]
        
        # 使用标准差过滤
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        
        filtered = []
        for seg in segments:
            if abs(seg['slope'] - mean_slope) < 2 * std_slope:
                filtered.append(seg)
        
        return filtered
    
    def _fit_lane_line(self, segments: List, image_height: int) -> Optional[Dict]:
        """拟合车道线"""
        if len(segments) < 2:
            return None
        
        # 收集所有点
        all_x = []
        all_y = []
        
        for seg in segments:
            for point in seg['points']:
                all_x.append(point[0])
                all_y.append(point[1])
        
        # 二次多项式拟合（适应弯曲车道）
        try:
            coeffs = np.polyfit(all_y, all_x, 2)  # x = ay² + by + c
            poly_func = np.poly1d(coeffs)
            
            # 生成拟合点
            y_top = int(image_height * 0.4)
            y_bottom = image_height
            
            x_top = int(poly_func(y_top))
            x_bottom = int(poly_func(y_bottom))
            
            return {
                'coefficients': coeffs,
                'points': [(x_bottom, y_bottom), (x_top, y_top)],
                'func': poly_func,
                'confidence': min(len(segments) / 10.0, 1.0)
            }
        except:
            # 如果二次拟合失败，使用线性拟合
            coeffs = np.polyfit(all_y, all_x, 1)
            poly_func = np.poly1d(coeffs)
            
            y_top = int(image_height * 0.4)
            y_bottom = image_height
            
            x_top = int(poly_func(y_top))
            x_bottom = int(poly_func(y_bottom))
            
            return {
                'coefficients': coeffs,
                'points': [(x_bottom, y_bottom), (x_top, y_top)],
                'func': poly_func,
                'confidence': min(len(segments) / 10.0, 0.8)  # 线性拟合置信度较低
            }
    
    def _calculate_center_line(self, left_lane: Dict, right_lane: Dict, image_height: int) -> Dict:
        """计算中心线"""
        if not left_lane or not right_lane:
            return None
        
        try:
            # 获取左右车道线函数
            left_func = left_lane['func']
            right_func = right_lane['func']
            
            # 计算中心线函数（平均）
            def center_func(y):
                return (left_func(y) + right_func(y)) / 2
            
            # 生成中心线点
            y_top = int(image_height * 0.4)
            y_bottom = image_height
            
            x_top = int(center_func(y_top))
            x_bottom = int(center_func(y_bottom))
            
            return {
                'func': center_func,
                'points': [(x_bottom, y_bottom), (x_top, y_top)]
            }
        except:
            return None

class EnhancedDirectionAnalyzer:
    """增强方向分析器"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.direction_history = []
        self.confidence_history = []
        self.history_size = 5
        
    def analyze_direction(self, road_info: Dict[str, Any], lane_info: Dict[str, Any], 
                         image_size: Tuple[int, int]) -> Dict[str, Any]:
        """综合分析道路方向 - 多特征融合"""
        width, height = image_size
        
        # 1. 基于道路轮廓的方向分析
        contour_direction, contour_confidence = self._analyze_from_contour(
            road_info.get('contour'), image_size
        )
        
        # 2. 基于车道线的方向分析
        lane_direction, lane_confidence = self._analyze_from_lanes(lane_info, image_size)
        
        # 3. 基于消失点的方向分析
        vanishing_direction, vanishing_confidence = self._analyze_vanishing_point(
            lane_info, image_size
        )
        
        # 4. 综合判断
        directions = [contour_direction, lane_direction, vanishing_direction]
        confidences = [contour_confidence, lane_confidence, vanishing_confidence]
        
        # 加权投票
        final_direction, final_confidence = self._weighted_vote(directions, confidences)
        
        # 5. 使用历史记录平滑
        if final_confidence > 0.5:
            smoothed_direction, smoothed_confidence = self._smooth_with_history(
                final_direction, final_confidence
            )
            
            return {
                'direction': smoothed_direction,
                'confidence': smoothed_confidence,
                'contour_direction': contour_direction,
                'lane_direction': lane_direction,
                'vanishing_direction': vanishing_direction,
                'raw_confidences': confidences
            }
        else:
            return {
                'direction': final_direction,
                'confidence': final_confidence,
                'contour_direction': contour_direction,
                'lane_direction': lane_direction,
                'vanishing_direction': vanishing_direction,
                'raw_confidences': confidences
            }
    
    def _analyze_from_contour(self, contour: np.ndarray, image_size: Tuple[int, int]) -> Tuple[str, float]:
        """基于道路轮廓分析方向"""
        if contour is None or len(contour) < 3:
            return "未知方向", 0.0
        
        width, height = image_size
        
        try:
            contour_points = contour.reshape(-1, 2)
            
            # 计算轮廓的几何特征
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return "未知方向", 0.0
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # 1. 质心位置分析
            image_center_x = width / 2
            deviation_ratio = (cx - image_center_x) / (width / 2)
            
            if abs(deviation_ratio) < 0.1:
                centroid_direction = "直行"
                centroid_confidence = 0.7 - abs(deviation_ratio)
            elif deviation_ratio > 0:
                centroid_direction = "右转"
                centroid_confidence = min(0.7, abs(deviation_ratio))
            else:
                centroid_direction = "左转"
                centroid_confidence = min(0.7, abs(deviation_ratio))
            
            # 2. 轮廓形状分析
            shape_direction, shape_confidence = self._analyze_contour_shape(
                contour_points, height, width
            )
            
            # 3. 融合结果
            if centroid_direction == shape_direction:
                confidence = (centroid_confidence + shape_confidence) / 2
                return centroid_direction, confidence
            else:
                # 取置信度更高的结果
                if centroid_confidence > shape_confidence:
                    return centroid_direction, centroid_confidence * 0.8
                else:
                    return shape_direction, shape_confidence * 0.8
            
        except Exception as e:
            logger.error(f"轮廓分析失败: {str(e)}")
            return "未知方向", 0.0
    
    def _analyze_contour_shape(self, contour_points: np.ndarray, height: int, width: int) -> Tuple[str, float]:
        """分析轮廓形状"""
        # 在不同高度分析轮廓宽度
        heights = [height * 0.3, height * 0.5, height * 0.7]
        widths = []
        centers = []
        
        for h in heights:
            points_at_height = [p for p in contour_points if abs(p[1] - h) < 10]
            if len(points_at_height) >= 2:
                min_x = min(p[0] for p in points_at_height)
                max_x = max(p[0] for p in points_at_height)
                widths.append(max_x - min_x)
                centers.append((min_x + max_x) / 2)
        
        if len(widths) >= 2:
            # 分析宽度变化
            width_ratio = widths[0] / widths[-1]  # 顶部宽度 / 底部宽度
            
            if width_ratio < 0.6:  # 明显变窄
                if centers[0] < width / 2:
                    return "左转", 0.8
                else:
                    return "右转", 0.8
            elif width_ratio > 1.4:  # 明显变宽
                if centers[0] < width / 2:
                    return "右转", 0.7
                else:
                    return "左转", 0.7
            else:  # 宽度变化不大
                # 分析中心线偏移
                center_deviation = centers[0] - width / 2
                if abs(center_deviation) < width * 0.1:
                    return "直行", 0.6
                elif center_deviation > 0:
                    return "右转", 0.7
                else:
                    return "左转", 0.7
        
        return "未知方向", 0.0
    
    def _analyze_from_lanes(self, lane_info: Dict[str, Any], image_size: Tuple[int, int]) -> Tuple[str, float]:
        """基于车道线分析方向"""
        width, height = image_size
        
        if not lane_info.get('left_lane') or not lane_info.get('right_lane'):
            return "未知方向", 0.0
        
        left_lane = lane_info['left_lane']
        right_lane = lane_info['right_lane']
        
        try:
            # 1. 计算车道中心线
            left_func = left_lane['func']
            right_func = right_lane['func']
            
            def lane_center_func(y):
                return (left_func(y) + right_func(y)) / 2
            
            # 2. 分析顶部和底部的中心线位置
            y_top = int(height * 0.4)
            y_bottom = height
            
            center_top = lane_center_func(y_top)
            center_bottom = lane_center_func(y_bottom)
            
            # 3. 计算偏移和收敛角度
            image_center = width / 2
            
            top_deviation = center_top - image_center
            bottom_deviation = center_bottom - image_center
            
            # 偏移方向
            if abs(top_deviation) < width * 0.1 and abs(bottom_deviation) < width * 0.1:
                direction = "直行"
                confidence = max(0.7 - abs(top_deviation/image_center), 0.4)
            elif top_deviation > 0:
                direction = "右转"
                confidence = min(0.8, abs(top_deviation/image_center))
            else:
                direction = "左转"
                confidence = min(0.8, abs(top_deviation/image_center))
            
            # 4. 考虑车道宽度变化
            top_width = right_func(y_top) - left_func(y_top)
            bottom_width = right_func(y_bottom) - left_func(y_bottom)
            
            width_ratio = top_width / bottom_width if bottom_width > 0 else 1
            
            # 宽度变化对置信度的影响
            if direction == "直行" and width_ratio < 0.8:
                confidence *= 0.8  # 可能误判
            elif direction in ["左转", "右转"] and width_ratio > 0.9:
                confidence *= 0.8  # 可能误判
            
            return direction, confidence
            
        except Exception as e:
            logger.error(f"车道线分析失败: {str(e)}")
            return "未知方向", 0.0
    
    def _analyze_vanishing_point(self, lane_info: Dict[str, Any], image_size: Tuple[int, int]) -> Tuple[str, float]:
        """基于消失点分析方向"""
        width, height = image_size
        
        left_lines = lane_info.get('left_lines', [])
        right_lines = lane_info.get('right_lines', [])
        
        if len(left_lines) < 2 or len(right_lines) < 2:
            return "未知方向", 0.0
        
        try:
            # 收集所有线段的端点
            all_left_points = []
            all_right_points = []
            
            for seg in left_lines:
                all_left_points.extend(seg['points'])
            
            for seg in right_lines:
                all_right_points.extend(seg['points'])
            
            # 拟合左右车道线的延长线
            if len(all_left_points) >= 2 and len(all_right_points) >= 2:
                # 提取坐标
                left_x = [p[0] for p in all_left_points]
                left_y = [p[1] for p in all_left_points]
                right_x = [p[0] for p in all_right_points]
                right_y = [p[1] for p in all_right_points]
                
                # 线性拟合
                left_coeffs = np.polyfit(left_y, left_x, 1)
                right_coeffs = np.polyfit(right_y, right_x, 1)
                
                # 计算消失点（两条线的交点）
                # 解方程：a1*y + b1 = a2*y + b2
                a1, b1 = left_coeffs[0], left_coeffs[1]
                a2, b2 = right_coeffs[0], right_coeffs[1]
                
                if a1 != a2:
                    vp_y = (b2 - b1) / (a1 - a2)
                    vp_x = a1 * vp_y + b1
                    
                    # 判断消失点位置
                    if vp_y < 0:  # 消失点在图像上方
                        if vp_x < width * 0.4:
                            return "左转", 0.8
                        elif vp_x > width * 0.6:
                            return "右转", 0.8
                        else:
                            return "直行", 0.7
                    else:
                        # 消失点在图像内，根据位置判断
                        if vp_x < width * 0.4:
                            return "左转", 0.6
                        elif vp_x > width * 0.6:
                            return "右转", 0.6
                        else:
                            return "直行", 0.5
            
            return "未知方向", 0.0
            
        except Exception as e:
            logger.error(f"消失点分析失败: {str(e)}")
            return "未知方向", 0.0
    
    def _weighted_vote(self, directions: List[str], confidences: List[float]) -> Tuple[str, float]:
        """加权投票决定最终方向"""
        valid_results = []
        for dir, conf in zip(directions, confidences):
            if dir != "未知方向" and conf > 0.3:
                valid_results.append((dir, conf))
        
        if not valid_results:
            return "未知方向", 0.0
        
        # 统计每个方向的加权得分
        scores = {}
        for dir, conf in valid_results:
            if dir not in scores:
                scores[dir] = 0
            scores[dir] += conf
        
        # 选择得分最高的方向
        best_direction = max(scores.items(), key=lambda x: x[1])[0]
        best_score = scores[best_direction]
        
        # 归一化置信度
        total_score = sum(scores.values())
        final_confidence = best_score / total_score if total_score > 0 else 0
        
        return best_direction, final_confidence
    
    def _smooth_with_history(self, direction: str, confidence: float) -> Tuple[str, float]:
        """使用历史记录平滑方向"""
        self.direction_history.append(direction)
        self.confidence_history.append(confidence)
        
        if len(self.direction_history) > self.history_size:
            self.direction_history.pop(0)
            self.confidence_history.pop(0)
        
        # 当有足够历史记录时，进行平滑
        if len(self.direction_history) == self.history_size:
            # 计算每个方向的出现频率和平均置信度
            direction_stats = {}
            
            for i, dir in enumerate(self.direction_history):
                if dir not in direction_stats:
                    direction_stats[dir] = {'count': 0, 'total_confidence': 0}
                direction_stats[dir]['count'] += 1
                direction_stats[dir]['total_confidence'] += self.confidence_history[i]
            
            # 选择最频繁且高置信度的方向
            best_dir = None
            best_score = 0
            
            for dir, stats in direction_stats.items():
                avg_confidence = stats['total_confidence'] / stats['count']
                score = stats['count'] * avg_confidence  # 加权得分
                
                if score > best_score:
                    best_score = score
                    best_dir = dir
            
            if best_dir and best_score > 1.0:
                smoothed_confidence = min(1.0, best_score / self.history_size)
                return best_dir, smoothed_confidence
        
        return direction, confidence

class EnhancedVisualizationEngine:
    """增强可视化引擎"""
    
    def __init__(self):
        self.colors = {
            'contour': (0, 255, 255),  # 黄色 - 轮廓
            'road_area': (0, 255, 0),   # 绿色 - 道路区域
            'left_lane': (255, 0, 0),   # 蓝色 - 左车道线
            'right_lane': (0, 0, 255),  # 红色 - 右车道线
            'center_line': (255, 255, 0), # 青色 - 中心线
            'direction': (0, 0, 255),   # 红色 - 方向指示
            'roi': (0, 255, 255),       # 黄色 - ROI区域
            'text': (255, 255, 255),    # 白色 - 文本
            'vanishing_point': (255, 0, 255) # 紫色 - 消失点
        }
    
    def draw_detection_results(self, image: np.ndarray, road_info: Dict[str, Any], 
                             lane_info: Dict[str, Any], direction_info: Dict[str, Any]) -> np.ndarray:
        """绘制增强的检测结果"""
        result = image.copy()
        height, width = result.shape[:2]
        
        # 绘制ROI区域
        if road_info.get('roi_vertices') is not None:
            cv2.polylines(result, [road_info['roi_vertices']], True, self.colors['roi'], 2)
        
        # 绘制道路轮廓和区域
        if road_info.get('contour') is not None:
            self._draw_road_contour(result, road_info['contour'])
        
        # 绘制车道线
        self._draw_enhanced_lane_lines(result, lane_info)
        
        # 绘制消失点
        if 'vanishing_point' in lane_info:
            self._draw_vanishing_point(result, lane_info['vanishing_point'])
        
        # 添加增强的信息文本
        self._draw_enhanced_info_text(result, direction_info, height, width)
        
        # 绘制方向指示器
        direction = direction_info.get('direction', '未知方向')
        confidence = direction_info.get('confidence', 0.0)
        self._draw_enhanced_direction_indicator(result, direction, confidence, width, height)
        
        return result
    
    def _draw_road_contour(self, image: np.ndarray, contour: np.ndarray):
        """绘制道路轮廓"""
        # 绘制轮廓线
        cv2.drawContours(image, [contour], -1, self.colors['contour'], 3)
        
        # 填充道路区域（半透明）
        overlay = image.copy()
        cv2.fillPoly(overlay, [contour], self.colors['road_area'])
        cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)
    
    def _draw_enhanced_lane_lines(self, image: np.ndarray, lane_info: Dict[str, Any]):
        """绘制增强的车道线"""
        # 绘制原始线段
        for side in ['left_lines', 'right_lines']:
            color = self.colors['left_lane'] if side == 'left_lines' else self.colors['right_lane']
            for segment in lane_info.get(side, []):
                points = segment['points']
                cv2.line(image, tuple(points[0]), tuple(points[1]), color, 2)
        
        # 绘制拟合的车道线
        for side, color_key in [('left_lane', 'left_lane'), ('right_lane', 'right_lane')]:
            if lane_info.get(side):
                lane = lane_info[side]
                points = lane.get('points', [])
                if len(points) == 2:
                    cv2.line(image, tuple(points[0]), tuple(points[1]), 
                            self.colors[color_key], 4)
        
        # 绘制中心线
        if lane_info.get('center_line'):
            center = lane_info['center_line']
            points = center.get('points', [])
            if len(points) == 2:
                cv2.line(image, tuple(points[0]), tuple(points[1]), 
                        self.colors['center_line'], 3, cv2.LINE_AA)
    
    def _draw_vanishing_point(self, image: np.ndarray, vanishing_point: Tuple[float, float]):
        """绘制消失点"""
        vp_x, vp_y = vanishing_point
        if 0 <= vp_x < image.shape[1] and 0 <= vp_y < image.shape[0]:
            cv2.circle(image, (int(vp_x), int(vp_y)), 10, self.colors['vanishing_point'], -1)
            cv2.circle(image, (int(vp_x), int(vp_y)), 12, self.colors['text'], 2)
    
    def _draw_enhanced_info_text(self, image: np.ndarray, direction_info: Dict[str, Any], height: int, width: int):
        """绘制增强的信息文本"""
        direction = direction_info.get('direction', '未知方向')
        confidence = direction_info.get('confidence', 0.0)
        
        # 背景框
        bg_color = (0, 0, 0)
        text_color = self.colors['text']
        
        # 主方向信息
        direction_text = f"方向: {direction}"
        confidence_text = f"置信度: {confidence:.2%}"
        
        # 详细信息
        details_texts = []
        for key in ['contour_direction', 'lane_direction', 'vanishing_direction']:
            if key in direction_info and direction_info[key] != "未知方向":
                details_texts.append(f"{key.split('_')[0]}: {direction_info[key]}")
        
        # 绘制主信息
        font_scale_large = 0.9
        font_scale_small = 0.6
        
        cv2.putText(image, direction_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, text_color, 2)
        
        # 根据置信度设置颜色
        conf_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.5 else (0, 0, 255)
        cv2.putText(image, confidence_text, (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, conf_color, 2)
        
        # 绘制详细信息
        y_offset = 90
        for i, detail_text in enumerate(details_texts):
            cv2.putText(image, detail_text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, text_color, 1)
    
    def _draw_enhanced_direction_indicator(self, image: np.ndarray, direction: str, 
                                          confidence: float, width: int, height: int):
        """绘制增强的方向指示器"""
        center_x, center_y = width // 2, height // 2
        arrow_length = min(width, height) // 4
        
        # 根据置信度调整箭头大小和颜色
        arrow_thickness = int(6 * confidence + 3)
        alpha = int(255 * confidence)
        
        if direction == "左转":
            end_point = (center_x - arrow_length, center_y)
            arrow_color = (0, 0, 255, alpha)  # 红色，带透明度
        elif direction == "右转":
            end_point = (center_x + arrow_length, center_y)
            arrow_color = (0, 0, 255, alpha)  # 红色，带透明度
        else:  # 直行
            end_point = (center_x, center_y - arrow_length)
            arrow_color = (0, 255, 0, alpha)  # 绿色，带透明度
        
        # 绘制箭头（使用叠加层实现透明度）
        overlay = image.copy()
        cv2.arrowedLine(overlay, (center_x, center_y), end_point, 
                       arrow_color[:3], arrow_thickness, tipLength=0.3)
        
        # 根据透明度混合
        cv2.addWeighted(overlay, alpha/255, image, 1 - alpha/255, 0, image)

class EnhancedLaneDetectionApp:
    """增强道路方向识别系统"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("智能道路方向识别系统 - 增强版")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # 初始化增强组件
        self.config = DetectionConfig()
        self.road_detector = EnhancedRoadDetector(self.config)
        self.direction_analyzer = EnhancedDirectionAnalyzer(self.config)
        self.visualization_engine = EnhancedVisualizationEngine()
        self.image_processor = EnhancedImageProcessor(self.config)
        
        # 状态变量
        self.current_image_path = None
        self.original_image = None
        self.is_processing = False
        
        # 创建UI
        self._create_enhanced_ui()
        
        logger.info("增强版应用程序初始化完成")
    
    def _create_enhanced_ui(self):
        """创建增强的用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 创建控件
        self._create_enhanced_control_panel(main_frame)
        self._create_image_display(main_frame)
        self._create_status_bar(main_frame)
    
    def _create_enhanced_control_panel(self, parent):
        """创建增强的控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 文件操作按钮
        file_frame = ttk.Frame(control_frame)
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="选择图片", 
                  command=self._select_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="重新检测", 
                  command=self._redetect).pack(side=tk.LEFT)
        
        # 文件路径显示
        self.file_path_var = tk.StringVar(value="未选择图片")
        ttk.Label(file_frame, textvariable=self.file_path_var).pack(side=tk.LEFT, padx=(20, 0))
        
        # 算法参数调整
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 敏感度调整
        ttk.Label(param_frame, text="检测敏感度:").grid(row=0, column=0, sticky=tk.W)
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        ttk.Scale(param_frame, from_=0.1, to=1.0, variable=self.sensitivity_var,
                 command=self._on_sensitivity_change, orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # 结果显示
        self.result_var = tk.StringVar(value="等待检测...")
        self.confidence_var = tk.StringVar(value="")
        
        result_frame = ttk.Frame(control_frame)
        result_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(result_frame, textvariable=self.result_var, 
                 font=("Arial", 12, "bold"), foreground="blue").pack(side=tk.LEFT)
        ttk.Label(result_frame, textvariable=self.confidence_var, 
                 font=("Arial", 10), foreground="green").pack(side=tk.LEFT, padx=(20, 0))
    
    def _create_image_display(self, parent):
        """创建图像显示区域"""
        # 这部分与之前版本相同，保持不变
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # 原图显示
        original_frame = ttk.LabelFrame(display_frame, text="原始图像", padding="10")
        original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        
        self.original_canvas = tk.Canvas(
            original_frame,
            width=600,
            height=500,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.original_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 结果图显示
        result_frame = ttk.LabelFrame(display_frame, text="检测结果", padding="10")
        result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        self.result_canvas = tk.Canvas(
            result_frame,
            width=600,
            height=500,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.result_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 初始提示文本
        self.original_canvas.create_text(300, 250, text="请选择道路图片", font=("Arial", 16), fill="gray")
        self.result_canvas.create_text(300, 250, text="检测结果将显示在这里", font=("Arial", 16), fill="gray")
    
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
            self.original_image = self.image_processor.resize_image(self.original_image, (1200, 800))
            
            # 显示图像
            self._display_image_on_canvas(self.original_image, self.original_canvas, "原始图像")
            
            self.status_var.set("图片加载成功")
            logger.info(f"图片加载成功: {file_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
            logger.error(f"图片加载失败: {str(e)}")
    
    def _display_image_on_canvas(self, image: np.ndarray, canvas: tk.Canvas, title: str):
        """在Canvas上显示图像"""
        try:
            # 清除Canvas
            canvas.delete("all")
            
            # 转换颜色空间
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # 转换为Tkinter格式
            photo = ImageTk.PhotoImage(pil_image)
            
            # 在Canvas上显示图像
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo  # 保持引用
            
            # 更新滚动区域
            canvas.config(scrollregion=canvas.bbox(tk.ALL))
            
        except Exception as e:
            logger.error(f"图像显示失败: {str(e)}")
            canvas.create_text(300, 250, text=f"{title}显示失败", font=("Arial", 16), fill="red")
    
    def _start_detection(self):
        """开始检测"""
        if self.is_processing or self.original_image is None:
            return
        
        self.is_processing = True
        self.progress.start()
        self.status_var.set("正在分析道路方向...")
        self.result_var.set("检测中...")
        self.confidence_var.set("")
        
        # 在后台线程中执行检测
        thread = threading.Thread(target=self._enhanced_detection_thread)
        thread.daemon = True
        thread.start()
    
    def _enhanced_detection_thread(self):
        """增强的检测线程"""
        try:
            start_time = time.time()
            
            # 1. 检测道路区域（多方法融合）
            road_info = self.road_detector.detect_road_region(self.original_image)
            
            # 2. 检测车道线（增强算法）
            lane_info = self.road_detector.detect_lane_lines_enhanced(
                self.original_image, road_info.get('roi_vertices', np.array([]))
            )
            
            # 3. 分析方向（多特征融合）
            direction_info = self.direction_analyzer.analyze_direction(
                road_info, lane_info, 
                (self.original_image.shape[1], self.original_image.shape[0])
            )
            
            processing_time = time.time() - start_time
            
            # 4. 生成结果图像
            result_image = self.visualization_engine.draw_detection_results(
                self.original_image, road_info, lane_info, direction_info
            )
            
            # 在主线程中更新UI
            self.root.after(0, self._update_enhanced_results, direction_info, result_image, processing_time)
            
        except Exception as e:
            logger.error(f"检测过程出错: {str(e)}")
            self.root.after(0, self._show_error, f"检测失败: {str(e)}")
    
    def _update_enhanced_results(self, direction_info: Dict[str, Any], result_image: np.ndarray, processing_time: float):
        """更新增强的检测结果"""
        self.is_processing = False
        self.progress.stop()
        
        # 显示结果图像
        self._display_image_on_canvas(result_image, self.result_canvas, "检测结果")
        
        # 更新结果文本
        direction = direction_info.get('direction', '未知方向')
        confidence = direction_info.get('confidence', 0.0)
        
        self.result_var.set(f"检测结果: {direction}")
        self.confidence_var.set(f" (置信度: {confidence:.2%})")
        
        # 根据置信度设置文本颜色
        if confidence > 0.7:
            color = "green"
        elif confidence > 0.5:
            color = "orange"
        else:
            color = "red"
        
        self.confidence_var.set(f" (置信度: {confidence:.2%})")
        
        self.status_var.set(f"分析完成 - 耗时: {processing_time:.2f}秒")
        
        logger.info(f"检测完成: {direction}, 置信度: {confidence:.2%}, 耗时: {processing_time:.2f}秒")
    
    def _show_error(self, error_msg: str):
        """显示错误信息"""
        self.is_processing = False
        self.progress.stop()
        messagebox.showerror("错误", error_msg)
        self.status_var.set("检测失败")
        self.result_var.set("检测失败")
        self.confidence_var.set("")
        logger.error(f"检测错误: {error_msg}")
    
    def _redetect(self):
        """重新检测"""
        if self.current_image_path and not self.is_processing:
            self._start_detection()
    
    def _on_sensitivity_change(self, value):
        """敏感度变化回调"""
        sensitivity = float(value)
        # 根据敏感度调整配置参数
        self.config.width_ratio_threshold = 0.3 + sensitivity * 0.4
        self.config.center_deviation_threshold = 0.1 + sensitivity * 0.1
        logger.info(f"敏感度调整为: {sensitivity:.2f}")
        
        # 如果已有图像，自动重新检测
        if self.current_image_path and not self.is_processing:
            self._start_detection()
    
    def run(self):
        """运行应用程序"""
        try:
            self.root.mainloop()
        except Exception as e:
            logger.error(f"应用程序运行错误: {str(e)}")

def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = EnhancedLaneDetectionApp(root)
        app.run()
    except Exception as e:
        logger.critical(f"应用程序启动失败: {str(e)}")
        messagebox.showerror("致命错误", f"应用程序启动失败: {str(e)}")

if __name__ == "__main__":
    main()