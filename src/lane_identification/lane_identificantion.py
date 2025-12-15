import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable, Deque
from collections import deque, defaultdict
import math

# ==================== 配置管理 ====================
@dataclass
class AppConfig:
    """应用配置参数"""
    # 性能参数
    max_image_size: Tuple[int, int] = (1200, 800)
    cache_size: int = 5
    batch_size: int = 50  # 批量处理时每批的图像数量
    
    # 图像处理参数
    adaptive_clip_limit: float = 2.0
    adaptive_grid_size: Tuple[int, int] = (8, 8)
    gaussian_kernel: Tuple[int, int] = (5, 5)
    
    # 检测参数
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    hough_threshold: int = 30
    hough_min_length: int = 20
    hough_max_gap: int = 50
    min_contour_area: float = 0.01  # 最小轮廓面积比例
    
    # 方向分析参数
    deviation_threshold: float = 0.15
    width_ratio_threshold: float = 0.7
    confidence_threshold: float = 0.5
    
    # 路径预测参数
    prediction_steps: int = 8
    prediction_distance: float = 0.8
    min_prediction_points: int = 3
    
    # 界面参数
    ui_refresh_rate: int = 100  # UI刷新率(ms)
    animation_duration: int = 300  # 动画持续时间(ms)

# ==================== 图像处理优化 ====================
class SmartImageProcessor:
    """智能图像处理器 - 使用缓存和预处理优化"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._cache = {}
        self._cache_order = deque(maxlen=config.cache_size)
        self._preprocess_cache = {}
    
    def load_and_preprocess(self, image_path: str) -> Optional[Tuple[np.ndarray, Any]]:
        """加载并预处理图像"""
        try:
            # 检查缓存
            if image_path in self._cache:
                return self._cache[image_path]
            
            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                return None
            
            # 获取图像信息
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # 智能调整尺寸
            processed = self._smart_resize(image)
            
            # 自适应预处理
            processed = self._adaptive_preprocessing(processed)
            
            # 计算ROI区域
            roi_info = self._calculate_roi(processed.shape)
            
            # 更新缓存
            self._update_cache(image_path, (processed, roi_info))
            
            return processed, roi_info
            
        except Exception as e:
            print(f"图像处理失败: {e}")
            return None
    
    def _smart_resize(self, image: np.ndarray) -> np.ndarray:
        """智能调整图像尺寸"""
        height, width = image.shape[:2]
        max_w, max_h = self.config.max_image_size
        
        # 计算缩放比例，保持宽高比
        scale_w = max_w / width if width > max_w else 1.0
        scale_h = max_h / height if height > max_h else 1.0
        scale = min(scale_w, scale_h)
        
        if scale < 1.0:
            new_size = (int(width * scale), int(height * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        return image
    
    def _adaptive_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """自适应图像预处理"""
        # 1. 转换为YUV颜色空间
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0]
        
        # 2. 自适应直方图均衡化
        clahe = cv2.createCLAHE(
            clipLimit=self.config.adaptive_clip_limit,
            tileGridSize=self.config.adaptive_grid_size
        )
        yuv[:, :, 0] = clahe.apply(y_channel)
        
        # 3. 转换回BGR
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # 4. 自适应去噪
        noise_level = self._estimate_noise_level(y_channel)
        if noise_level > 30:
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """估计图像噪声水平"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用拉普拉斯算子估计噪声
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        noise_level = np.std(laplacian)
        return float(noise_level)
    
    def _calculate_roi(self, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """计算ROI区域"""
        height, width = image_shape[:2]
        
        # 动态ROI计算
        roi_top = int(height * 0.35)  # 稍微提高ROI顶部
        roi_bottom = int(height * 0.92)  # 稍微降低ROI底部
        roi_width = int(width * 0.85)
        
        vertices = np.array([[
            ((width - roi_width) // 2, roi_bottom),
            ((width - roi_width) // 2 + int(roi_width * 0.3), roi_top),
            ((width - roi_width) // 2 + int(roi_width * 0.7), roi_top),
            ((width + roi_width) // 2, roi_bottom)
        ]], dtype=np.int32)
        
        return {
            'vertices': vertices,
            'mask': self._create_roi_mask(vertices, image_shape[:2]),
            'bounds': (roi_top, roi_bottom, roi_width)
        }
    
    def _create_roi_mask(self, vertices: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """创建ROI掩码"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, vertices, 255)
        return mask
    
    def _update_cache(self, key: str, value: Any):
        """更新缓存"""
        if len(self._cache) >= self.config.cache_size:
            oldest = self._cache_order.popleft()
            self._cache.pop(oldest, None)
        
        self._cache[key] = value
        self._cache_order.append(key)
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_order.clear()
        self._preprocess_cache.clear()

# ==================== 高级道路检测器 ====================
class AdvancedRoadDetector:
    """高级道路检测器 - 使用多尺度分析和机器学习启发式算法"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.previous_results = deque(maxlen=3)  # 历史结果缓冲
        self.adaptive_params = {
            'canny_low': config.canny_threshold1,
            'canny_high': config.canny_threshold2,
            'hough_threshold': config.hough_threshold
        }
    
    def detect(self, image: np.ndarray, roi_info: Dict[str, Any]) -> Dict[str, Any]:
        """执行道路检测"""
        try:
            # 多尺度分析
            results = []
            for scale in [1.0, 0.8, 0.6]:
                scaled_results = self._detect_at_scale(image, roi_info, scale)
                if scaled_results:
                    results.append(scaled_results)
            
            if not results:
                return self._create_empty_result()
            
            # 结果融合
            fused_result = self._fuse_results(results)
            
            # 更新历史
            self.previous_results.append(fused_result)
            
            # 平滑处理
            if len(self.previous_results) > 1:
                fused_result = self._temporal_smooth(fused_result)
            
            return fused_result
            
        except Exception as e:
            print(f"道路检测失败: {e}")
            return self._create_empty_result()
    
    def _detect_at_scale(self, image: np.ndarray, roi_info: Dict[str, Any], 
                        scale: float) -> Optional[Dict[str, Any]]:
        """在特定尺度下进行检测"""
        if scale != 1.0:
            height, width = image.shape[:2]
            new_size = (int(width * scale), int(height * scale))
            scaled_image = cv2.resize(image, new_size, cv2.INTER_AREA)
        else:
            scaled_image = image
        
        # 提取ROI区域
        roi_region = self._extract_roi_region(scaled_image, roi_info, scale)
        
        # 多方法道路检测
        detection_results = []
        
        # 方法1: 基于颜色的检测
        color_result = self._detect_by_color(roi_region)
        if color_result['confidence'] > 0.3:
            detection_results.append(color_result)
        
        # 方法2: 基于纹理的检测
        texture_result = self._detect_by_texture(roi_region)
        if texture_result['confidence'] > 0.3:
            detection_results.append(texture_result)
        
        # 方法3: 基于边缘的检测
        edge_result = self._detect_by_edges(roi_region)
        if edge_result['confidence'] > 0.3:
            detection_results.append(edge_result)
        
        if not detection_results:
            return None
        
        # 结果融合
        fused = self._fuse_detection_results(detection_results)
        
        # 提取特征
        features = self._extract_features(fused, scaled_image.shape)
        
        return {
            'scale': scale,
            'fused_mask': fused['mask'],
            'features': features,
            'confidence': fused['confidence'],
            'detection_methods': len(detection_results)
        }
    
    def _extract_roi_region(self, image: np.ndarray, roi_info: Dict[str, Any], 
                           scale: float) -> np.ndarray:
        """提取ROI区域"""
        if scale != 1.0:
            # 缩放ROI顶点
            vertices = roi_info['vertices'].copy()
            vertices = (vertices * scale).astype(np.int32)
            mask = self._create_scaled_mask(vertices, image.shape[:2])
        else:
            mask = roi_info['mask']
        
        return cv2.bitwise_and(image, image, mask=mask)
    
    def _create_scaled_mask(self, vertices: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """创建缩放后的掩码"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, vertices, 255)
        return mask
    
    def _detect_by_color(self, roi_region: np.ndarray) -> Dict[str, Any]:
        """基于颜色检测道路"""
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(roi_region, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # 自适应阈值
        mean_l = np.mean(l_channel)
        std_l = np.std(l_channel)
        
        # 道路颜色范围
        lower_bound = max(0, int(mean_l - std_l * 1.5))
        upper_bound = min(255, int(mean_l + std_l * 1.5))
        
        # 创建掩码
        mask = cv2.inRange(l_channel, lower_bound, upper_bound)
        
        # 形态学优化
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 计算置信度
        area_ratio = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
        confidence = min(1.0, area_ratio * 2.0)
        
        return {'mask': mask, 'confidence': confidence, 'method': 'color'}
    
    def _detect_by_texture(self, roi_region: np.ndarray) -> Dict[str, Any]:
        """基于纹理检测道路"""
        gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
        
        # 计算局部二值模式(LBP)特征
        radius = 3
        n_points = 8 * radius
        
        # 简单的纹理分析：使用Gabor滤波器
        kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        
        # 阈值化
        _, mask = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 计算置信度
        uniformity = self._calculate_uniformity(mask)
        confidence = min(1.0, uniformity * 1.5)
        
        return {'mask': mask, 'confidence': confidence, 'method': 'texture'}
    
    def _calculate_uniformity(self, mask: np.ndarray) -> float:
        """计算纹理均匀性"""
        # 计算局部方差
        kernel = np.ones((5, 5), np.float32) / 25
        mean = cv2.filter2D(mask.astype(np.float32), -1, kernel)
        variance = cv2.filter2D((mask.astype(np.float32) - mean) ** 2, -1, kernel)
        
        # 均匀性 = 1 / (1 + 平均方差)
        avg_variance = np.mean(variance)
        return 1.0 / (1.0 + avg_variance)
    
    def _detect_by_edges(self, roi_region: np.ndarray) -> Dict[str, Any]:
        """基于边缘检测道路"""
        gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
        
        # 自适应Canny边缘检测
        median_intensity = np.median(gray)
        lower = max(0, int(0.66 * median_intensity))
        upper = min(255, int(1.33 * median_intensity))
        
        edges = cv2.Canny(gray, lower, upper)
        
        # 形态学闭合连接边缘
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 填充闭合区域
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(edges)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 过滤小区域
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # 计算置信度
        edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
        confidence = min(1.0, 1.0 - edge_density * 2.0)  # 道路区域边缘应较少
        
        return {'mask': mask, 'confidence': confidence, 'method': 'edges'}
    
    def _fuse_detection_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """融合多个检测结果"""
        masks = [r['mask'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # 加权融合
        total_confidence = sum(confidences)
        weights = [c / total_confidence for c in confidences]
        
        # 初始化融合掩码
        fused = np.zeros_like(masks[0], dtype=np.float32)
        for mask, weight in zip(masks, weights):
            fused += mask.astype(np.float32) * weight
        
        # 转换为二值图像
        fused_binary = (fused > 127).astype(np.uint8) * 255
        
        # 最终形态学优化
        kernel = np.ones((5, 5), np.uint8)
        fused_binary = cv2.morphologyEx(fused_binary, cv2.MORPH_CLOSE, kernel)
        
        return {
            'mask': fused_binary,
            'confidence': np.mean(confidences),
            'methods': len(results)
        }
    
    def _extract_features(self, detection_result: Dict[str, Any], 
                         image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """提取道路特征"""
        mask = detection_result['mask']
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        # 找到最大轮廓
        main_contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓特征
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        rect = cv2.boundingRect(main_contour)
        
        # 计算凸包和简化
        hull = cv2.convexHull(main_contour)
        epsilon = 0.01 * perimeter
        simplified = cv2.approxPolyDP(hull, epsilon, True)
        
        # 计算质心
        M = cv2.moments(main_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = rect[0] + rect[2] // 2, rect[1] + rect[3] // 2
        
        # 计算方向特征
        _, _, angle = cv2.fitEllipse(main_contour) if len(main_contour) >= 5 else (0, 0, 0)
        
        return {
            'contour': simplified,
            'centroid': (cx, cy),
            'area': area,
            'perimeter': perimeter,
            'bounding_rect': rect,
            'orientation': angle,
            'solidity': area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        }
    
    def _fuse_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """融合多尺度结果"""
        if not results:
            return self._create_empty_result()
        
        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 加权平均特征
        total_weight = sum(r['confidence'] for r in results)
        weighted_features = defaultdict(list)
        
        for result in results:
            weight = result['confidence'] / total_weight
            for key, value in result['features'].items():
                if isinstance(value, (int, float)):
                    weighted_features[key].append(value * weight)
                elif key == 'centroid' and isinstance(value, tuple):
                    weighted_features['centroid_x'].append(value[0] * weight)
                    weighted_features['centroid_y'].append(value[1] * weight)
        
        # 计算加权平均值
        fused_features = {}
        for key, values in weighted_features.items():
            if key.startswith('centroid_'):
                base_key = key.replace('_x', '').replace('_y', '')
                if base_key not in fused_features:
                    fused_features[base_key] = [0, 0]
                if key.endswith('_x'):
                    fused_features[base_key][0] = sum(values)
                else:
                    fused_features[base_key][1] = sum(values)
            else:
                fused_features[key] = sum(values)
        
        # 使用最高置信度的轮廓
        best_result = results[0]
        
        return {
            **best_result,
            'features': fused_features,
            'confidence': np.mean([r['confidence'] for r in results]),
            'num_scales': len(results)
        }
    
    def _temporal_smooth(self, current_result: Dict[str, Any]) -> Dict[str, Any]:
        """时间平滑处理"""
        if len(self.previous_results) < 2:
            return current_result
        
        # 对特征进行指数移动平均
        alpha = 0.7  # 平滑因子
        smoothed_features = current_result['features'].copy()
        
        for prev_result in list(self.previous_results)[:-1]:  # 排除当前结果
            for key, value in prev_result['features'].items():
                if key in smoothed_features:
                    if isinstance(value, (int, float)):
                        smoothed_features[key] = alpha * smoothed_features[key] + (1 - alpha) * value
        
        current_result['features'] = smoothed_features
        return current_result
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'features': {},
            'confidence': 0.0,
            'num_scales': 0,
            'detection_methods': 0
        }

# ==================== 智能车道线检测器 ====================
class SmartLaneDetector:
    """智能车道线检测器 - 使用自适应算法和机器学习启发式方法"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.lane_history = deque(maxlen=5)
        self.adaptive_params = {
            'canny_multiplier': 1.0,
            'hough_threshold_multiplier': 1.0
        }
    
    def detect(self, image: np.ndarray, roi_mask: np.ndarray) -> Dict[str, Any]:
        """检测车道线"""
        try:
            # 预处理
            processed = self._preprocess_for_lanes(image, roi_mask)
            
            # 多方法检测
            detection_methods = [
                self._detect_with_canny,
                self._detect_with_sobel,
                self._detect_with_gradient
            ]
            
            all_lines = []
            for method in detection_methods:
                lines = method(processed)
                if lines is not None and len(lines) > 0:
                    all_lines.extend(lines)
            
            if not all_lines:
                return self._create_empty_lane_result()
            
            # 分类和过滤
            left_lines, right_lines = self._classify_and_filter(all_lines, image.shape[1])
            
            # 拟合车道线
            left_lane = self._fit_lane_model(left_lines, image.shape)
            right_lane = self._fit_lane_model(right_lines, image.shape)
            
            # 验证车道线
            left_lane, right_lane = self._validate_lanes(left_lane, right_lane, image.shape)
            
            # 预测路径
            future_path = self._predict_future_path(left_lane, right_lane, image.shape)
            
            # 更新历史
            result = {
                'left_lines': left_lines,
                'right_lines': right_lines,
                'left_lane': left_lane,
                'right_lane': right_lane,
                'future_path': future_path,
                'detection_quality': self._calculate_detection_quality(left_lane, right_lane)
            }
            
            self.lane_history.append(result)
            
            # 时间平滑
            if len(self.lane_history) > 1:
                result = self._temporal_smooth_lanes(result)
            
            return result
            
        except Exception as e:
            print(f"车道线检测失败: {e}")
            return self._create_empty_lane_result()
    
    def _preprocess_for_lanes(self, image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """为车道线检测预处理图像"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用ROI
        gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 自适应去噪
        noise_level = np.std(enhanced)
        if noise_level > 30:
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def _detect_with_canny(self, image: np.ndarray) -> List[np.ndarray]:
        """使用Canny边缘检测"""
        # 自适应Canny阈值
        median = np.median(image)
        lower = int(max(0, 0.66 * median))
        upper = int(min(255, 1.33 * median))
        
        edges = cv2.Canny(image, lower, upper)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=int(self.config.hough_threshold * self.adaptive_params['hough_threshold_multiplier']),
            minLineLength=self.config.hough_min_length,
            maxLineGap=self.config.hough_max_gap
        )
        
        return lines if lines is not None else []
    
    def _detect_with_sobel(self, image: np.ndarray) -> List[np.ndarray]:
        """使用Sobel算子检测"""
        # 计算梯度
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值和方向
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(np.abs(sobely), np.abs(sobelx))
        
        # 过滤垂直方向梯度
        vertical_mask = (direction > np.pi/4) & (direction < 3*np.pi/4)
        lanes = magnitude.copy()
        lanes[~vertical_mask] = 0
        
        # 转换为8位
        lanes = np.uint8(255 * lanes / np.max(lanes))
        
        # 阈值化
        _, binary = cv2.threshold(lanes, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 霍夫变换
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi/180,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.hough_min_length,
            maxLineGap=self.config.hough_max_gap
        )
        
        return lines if lines is not None else []
    
    def _detect_with_gradient(self, image: np.ndarray) -> List[np.ndarray]:
        """使用梯度方向检测"""
        # 计算梯度
        dx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        dy = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        
        # 计算梯度方向和幅值
        magnitude = np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx)
        
        # 寻找车道线方向
        lane_directions = []
        
        # 搜索可能的方向
        for angle in np.linspace(-np.pi/3, np.pi/3, 13):  # ±60度
            mask = np.abs(direction - angle) < np.pi/18  # ±10度容差
            if np.sum(mask) > 100:  # 有足够的像素
                lane_directions.append(angle)
        
        if not lane_directions:
            return []
        
        # 合并相似方向
        merged_directions = []
        for angle in lane_directions:
            if not merged_directions or min(abs(angle - md) for md in merged_directions) > np.pi/18:
                merged_directions.append(angle)
        
        # 为每个方向创建掩码并检测直线
        all_lines = []
        for angle in merged_directions:
            mask = np.abs(direction - angle) < np.pi/18
            masked_magnitude = magnitude.copy()
            masked_magnitude[~mask] = 0
            
            # 转换为8位
            result = np.uint8(255 * masked_magnitude / np.max(masked_magnitude + 1e-6))
            
            # 霍夫变换
            lines = cv2.HoughLinesP(
                result,
                rho=1,
                theta=np.pi/180,
                threshold=self.config.hough_threshold,
                minLineLength=self.config.hough_min_length,
                maxLineGap=self.config.hough_max_gap
            )
            
            if lines is not None:
                all_lines.extend(lines)
        
        return all_lines
    
    def _classify_and_filter(self, lines: List[np.ndarray], image_width: int) -> Tuple[List, List]:
        """分类和过滤车道线"""
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算线段参数
            if x2 == x1:
                continue  # 跳过垂直线
            
            dx = x2 - x1
            dy = y2 - y1
            
            # 计算斜率和长度
            slope = dy / dx
            length = np.sqrt(dx**2 + dy**2)
            
            # 过滤标准
            if abs(slope) < 0.3:  # 太平
                continue
            if length < 20:  # 太短
                continue
            
            # 根据位置和斜率分类
            midpoint_x = (x1 + x2) / 2
            if slope < 0:  # 左车道线
                if midpoint_x < image_width * 0.6:
                    left_lines.append({
                        'points': [(x1, y1), (x2, y2)],
                        'slope': slope,
                        'length': length,
                        'midpoint': (midpoint_x, (y1 + y2) / 2)
                    })
            else:  # 右车道线
                if midpoint_x > image_width * 0.4:
                    right_lines.append({
                        'points': [(x1, y1), (x2, y2)],
                        'slope': slope,
                        'length': length,
                        'midpoint': (midpoint_x, (y1 + y2) / 2)
                    })
        
        # 进一步过滤
        left_lines = self._filter_lines(left_lines)
        right_lines = self._filter_lines(right_lines)
        
        return left_lines, right_lines
    
    def _filter_lines(self, lines: List[Dict]) -> List[Dict]:
        """过滤异常线段"""
        if len(lines) < 3:
            return lines
        
        # 计算斜率和中点的统计信息
        slopes = [line['slope'] for line in lines]
        midpoints = [line['midpoint'][0] for line in lines]
        
        slope_mean = np.mean(slopes)
        slope_std = np.std(slopes)
        midpoint_mean = np.mean(midpoints)
        midpoint_std = np.std(midpoints)
        
        # 过滤异常值
        filtered = []
        for line in lines:
            slope_ok = abs(line['slope'] - slope_mean) < 2 * slope_std
            midpoint_ok = abs(line['midpoint'][0] - midpoint_mean) < 2 * midpoint_std
            
            if slope_ok and midpoint_ok:
                filtered.append(line)
        
        return filtered
    
    def _fit_lane_model(self, lines: List[Dict], image_shape: Tuple[int, ...]) -> Optional[Dict]:
        """拟合车道线模型"""
        if len(lines) < 2:
            return None
        
        # 收集所有点
        x_points, y_points = [], []
        for line in lines:
            for (x, y) in line['points']:
                x_points.append(x)
                y_points.append(y)
        
        # 尝试二次拟合
        try:
            # 二次多项式拟合: x = a*y^2 + b*y + c
            coeffs = np.polyfit(y_points, x_points, 2)
            poly_func = np.poly1d(coeffs)
            model_type = 'quadratic'
        except:
            # 降级为线性拟合
            coeffs = np.polyfit(y_points, x_points, 1)
            poly_func = np.poly1d(coeffs)
            model_type = 'linear'
        
        # 生成车道线点
        height, width = image_shape[:2]
        y_bottom = height
        y_top = int(height * 0.4)
        
        x_bottom = int(poly_func(y_bottom))
        x_top = int(poly_func(y_top))
        
        # 限制在图像范围内
        x_bottom = max(0, min(width, x_bottom))
        x_top = max(0, min(width, x_top))
        
        # 计算置信度
        confidence = min(len(lines) / 10.0, 1.0)
        
        return {
            'func': poly_func,
            'coeffs': coeffs.tolist() if hasattr(coeffs, 'tolist') else coeffs,
            'points': [(x_bottom, y_bottom), (x_top, y_top)],
            'model_type': model_type,
            'confidence': confidence,
            'num_lines': len(lines)
        }
    
    def _validate_lanes(self, left_lane: Optional[Dict], right_lane: Optional[Dict],
                       image_shape: Tuple[int, ...]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """验证车道线合理性"""
        height, width = image_shape[:2]
        
        if left_lane is None or right_lane is None:
            return left_lane, right_lane
        
        # 检查车道宽度
        if left_lane['model_type'] == 'quadratic' and right_lane['model_type'] == 'quadratic':
            # 计算几个点的宽度
            y_points = np.linspace(height * 0.4, height, 5)
            widths = []
            
            for y in y_points:
                left_x = left_lane['func'](y)
                right_x = right_lane['func'](y)
                widths.append(right_x - left_x)
            
            avg_width = np.mean(widths)
            std_width = np.std(widths)
            
            # 宽度应合理且变化不大
            min_reasonable_width = width * 0.15
            max_reasonable_width = width * 0.8
            
            if avg_width < min_reasonable_width or avg_width > max_reasonable_width or std_width > width * 0.2:
                # 宽度不合理，降低置信度
                left_lane['confidence'] *= 0.7
                right_lane['confidence'] *= 0.7
        
        # 检查车道线交叉
        if left_lane['points'][0][0] > right_lane['points'][0][0] or left_lane['points'][1][0] > right_lane['points'][1][0]:
            # 车道线交叉，降低置信度
            left_lane['confidence'] *= 0.6
            right_lane['confidence'] *= 0.6
        
        return left_lane, right_lane
    
    def _predict_future_path(self, left_lane: Optional[Dict], right_lane: Optional[Dict],
                           image_shape: Tuple[int, ...]) -> Optional[Dict]:
        """预测未来路径"""
        if left_lane is None or right_lane is None:
            return None
        
        try:
            height, width = image_shape[:2]
            
            # 计算中心线函数
            def center_func(y):
                left_x = left_lane['func'](y)
                right_x = right_lane['func'](y)
                return (left_x + right_x) / 2
            
            # 生成预测点
            current_y = height
            target_y = int(height * (1 - self.config.prediction_distance))
            
            if target_y <= 0:
                return None
            
            y_values = np.linspace(current_y, target_y, self.config.prediction_steps)
            path_points = []
            
            for y in y_values:
                x = center_func(y)
                # 确保点在图像内
                x = max(0, min(width, x))
                path_points.append((int(x), int(y)))
            
            # 计算路径曲率
            if len(path_points) >= 3:
                curvature = self._calculate_curvature(path_points)
            else:
                curvature = 0.0
            
            return {
                'center_path': path_points,
                'left_boundary': self._generate_boundary(left_lane, y_values, width),
                'right_boundary': self._generate_boundary(right_lane, y_values, width),
                'curvature': curvature,
                'prediction_length': len(path_points)
            }
            
        except Exception as e:
            print(f"路径预测失败: {e}")
            return None
    
    def _generate_boundary(self, lane: Dict, y_values: np.ndarray, image_width: int) -> List[Tuple[int, int]]:
        """生成边界点"""
        boundary = []
        for y in y_values:
            x = lane['func'](y)
            x = max(0, min(image_width, x))
            boundary.append((int(x), int(y)))
        return boundary
    
    def _calculate_curvature(self, points: List[Tuple[int, int]]) -> float:
        """计算路径曲率"""
        if len(points) < 3:
            return 0.0
        
        # 将点转换为numpy数组
        pts = np.array(points, dtype=np.float32)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # 计算一阶和二阶导数
        dx = np.gradient(x)
        dy = np.gradient(y)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        # 计算曲率
        curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**1.5
        
        # 返回平均曲率
        return float(np.mean(curvature[np.isfinite(curvature)]))
    
    def _calculate_detection_quality(self, left_lane: Optional[Dict], right_lane: Optional[Dict]) -> float:
        """计算检测质量"""
        quality = 0.0
        
        if left_lane is not None:
            quality += left_lane['confidence'] * 0.5
        
        if right_lane is not None:
            quality += right_lane['confidence'] * 0.5
        
        if left_lane is not None and right_lane is not None:
            # 两侧都有车道线，额外加分
            quality += 0.1
            
            # 模型类型一致加分
            if left_lane['model_type'] == right_lane['model_type']:
                quality += 0.1
        
        return min(quality, 1.0)
    
    def _temporal_smooth_lanes(self, current_result: Dict[str, Any]) -> Dict[str, Any]:
        """时间平滑车道线"""
        if len(self.lane_history) < 2:
            return current_result
        
        alpha = 0.6  # 平滑因子
        
        # 对车道线进行指数移动平均
        if current_result['left_lane'] and len(self.lane_history) > 0:
            # 获取历史结果
            prev_results = list(self.lane_history)[:-1]
            
            # 对系数进行平滑
            coeffs = np.array(current_result['left_lane']['coeffs'])
            for prev in prev_results:
                if prev['left_lane']:
                    prev_coeffs = np.array(prev['left_lane']['coeffs'])
                    if len(coeffs) == len(prev_coeffs):
                        coeffs = alpha * coeffs + (1 - alpha) * prev_coeffs
            
            current_result['left_lane']['coeffs'] = coeffs.tolist()
            current_result['left_lane']['func'] = np.poly1d(coeffs)
        
        # 同样处理右车道线
        if current_result['right_lane'] and len(self.lane_history) > 0:
            coeffs = np.array(current_result['right_lane']['coeffs'])
            for prev in list(self.lane_history)[:-1]:
                if prev['right_lane']:
                    prev_coeffs = np.array(prev['right_lane']['coeffs'])
                    if len(coeffs) == len(prev_coeffs):
                        coeffs = alpha * coeffs + (1 - alpha) * prev_coeffs
            
            current_result['right_lane']['coeffs'] = coeffs.tolist()
            current_result['right_lane']['func'] = np.poly1d(coeffs)
        
        return current_result
    
    def _create_empty_lane_result(self) -> Dict[str, Any]:
        """创建空的车道线结果"""
        return {
            'left_lines': [],
            'right_lines': [],
            'left_lane': None,
            'right_lane': None,
            'future_path': None,
            'detection_quality': 0.0
        }

# ==================== 高级方向分析器 ====================
class AdvancedDirectionAnalyzer:
    """高级方向分析器 - 使用多特征融合和机器学习"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.history = deque(maxlen=8)
        self.feature_weights = {
            'contour_centroid': 0.25,
            'lane_convergence': 0.35,
            'path_curvature': 0.25,
            'historical_consistency': 0.15
        }
    
    def analyze(self, road_features: Dict[str, Any], lane_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析道路方向"""
        try:
            # 特征提取
            features = self._extract_features(road_features, lane_info)
            
            # 方向预测
            direction_probs = self._predict_direction(features)
            
            # 置信度计算
            confidence = self._calculate_confidence(features, direction_probs)
            
            # 获取最终方向
            final_direction = max(direction_probs.items(), key=lambda x: x[1])[0]
            
            # 历史平滑
            final_direction, confidence = self._apply_historical_smoothing(final_direction, confidence)
            
            # 创建结果
            result = {
                'direction': final_direction,
                'confidence': confidence,
                'probabilities': direction_probs,
                'features': features,
                'reasoning': self._generate_reasoning(features, direction_probs)
            }
            
            # 更新历史
            self.history.append(result)
            
            return result
            
        except Exception as e:
            print(f"方向分析失败: {e}")
            return self._create_default_result()
    
    def _extract_features(self, road_features: Dict[str, Any], lane_info: Dict[str, Any]) -> Dict[str, Any]:
        """提取特征"""
        features = {}
        
        # 1. 轮廓质心特征
        if 'centroid' in road_features:
            cx, cy = road_features['centroid']
            features['contour_centroid_x'] = cx
            features['contour_centroid_y'] = cy
        
        # 2. 车道线收敛特征
        if lane_info['left_lane'] and lane_info['right_lane']:
            features['lane_convergence'] = self._calculate_lane_convergence(lane_info)
            features['lane_width'] = self._calculate_lane_width(lane_info)
            features['lane_symmetry'] = self._calculate_lane_symmetry(lane_info)
        
        # 3. 路径曲率特征
        if lane_info['future_path']:
            features['path_curvature'] = lane_info['future_path']['curvature']
            features['path_straightness'] = self._calculate_path_straightness(lane_info['future_path'])
        
        # 4. 历史一致性特征
        if self.history:
            features['historical_consistency'] = self._calculate_historical_consistency()
        
        return features
    
    def _calculate_lane_convergence(self, lane_info: Dict[str, Any]) -> float:
        """计算车道线收敛度"""
        left_func = lane_info['left_lane']['func']
        right_func = lane_info['right_lane']['func']
        
        # 计算顶部和底部的车道宽度
        y_bottom = 600  # 假设图像高度
        y_top = int(y_bottom * 0.4)
        
        width_bottom = right_func(y_bottom) - left_func(y_bottom)
        width_top = right_func(y_top) - left_func(y_top)
        
        # 计算收敛比
        if width_bottom > 0:
            convergence = width_top / width_bottom
            return float(convergence)
        
        return 1.0
    
    def _calculate_lane_width(self, lane_info: Dict[str, Any]) -> float:
        """计算平均车道宽度"""
        left_func = lane_info['left_lane']['func']
        right_func = lane_info['right_lane']['func']
        
        # 在多个位置采样宽度
        y_values = np.linspace(600 * 0.4, 600, 5)  # 假设图像高度600
        widths = [right_func(y) - left_func(y) for y in y_values]
        
        return float(np.mean(widths))
    
    def _calculate_lane_symmetry(self, lane_info: Dict[str, Any]) -> float:
        """计算车道对称性"""
        left_func = lane_info['left_lane']['func']
        right_func = lane_info['right_lane']['func']
        
        # 计算中心线
        def center_func(y):
            return (left_func(y) + right_func(y)) / 2
        
        # 在多个位置检查对称性
        y_values = np.linspace(600 * 0.4, 600, 5)
        symmetry_scores = []
        
        for y in y_values:
            center = center_func(y)
            left_dist = center - left_func(y)
            right_dist = right_func(y) - center
            
            if left_dist + right_dist > 0:
                symmetry = 1 - abs(left_dist - right_dist) / (left_dist + right_dist)
                symmetry_scores.append(symmetry)
        
        return float(np.mean(symmetry_scores) if symmetry_scores else 0.5)
    
    def _calculate_path_straightness(self, future_path: Dict[str, Any]) -> float:
        """计算路径直线度"""
        if not future_path['center_path'] or len(future_path['center_path']) < 2:
            return 1.0
        
        points = np.array(future_path['center_path'])
        
        # 拟合直线
        if len(points) >= 2:
            x = points[:, 0]
            y = points[:, 1]
            
            # 线性拟合
            coeffs = np.polyfit(y, x, 1)
            poly_func = np.poly1d(coeffs)
            
            # 计算R²值
            residuals = x - poly_func(y)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((x - np.mean(x))**2)
            
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
                return float(r_squared)
        
        return 0.5
    
    def _calculate_historical_consistency(self) -> float:
        """计算历史一致性"""
        if len(self.history) < 2:
            return 0.5
        
        # 检查最近几次方向的一致性
        recent_directions = [h['direction'] for h in list(self.history)[-3:]]
        
        if len(recent_directions) >= 2:
            # 计算一致性
            from collections import Counter
            freq = Counter(recent_directions)
            most_common_count = max(freq.values())
            consistency = most_common_count / len(recent_directions)
            return consistency
        
        return 0.5
    
    def _predict_direction(self, features: Dict[str, Any]) -> Dict[str, float]:
        """预测方向概率"""
        # 基于规则的推理
        probabilities = {'直行': 0.3, '左转': 0.35, '右转': 0.35}
        
        # 1. 基于轮廓质心
        if 'contour_centroid_x' in features:
            centroid_x = features['contour_centroid_x']
            # 假设图像宽度为800
            deviation = (centroid_x - 400) / 400  # 归一化到[-1, 1]
            
            if abs(deviation) < 0.1:
                probabilities['直行'] += 0.3
            elif deviation > 0.1:
                probabilities['右转'] += abs(deviation) * 0.5
            else:
                probabilities['左转'] += abs(deviation) * 0.5
        
        # 2. 基于车道线收敛
        if 'lane_convergence' in features:
            convergence = features['lane_convergence']
            
            if convergence < 0.6:  # 明显收敛
                if 'lane_symmetry' in features and features['lane_symmetry'] < 0.7:
                    # 不对称收敛，表示转弯
                    if features.get('lane_width', 200) < 250:  # 窄车道可能转弯
                        if features.get('lane_symmetry', 0.5) < 0.6:
                            probabilities['左转'] += 0.2
                        else:
                            probabilities['右转'] += 0.2
            else:
                probabilities['直行'] += 0.2
        
        # 3. 基于路径曲率
        if 'path_curvature' in features:
            curvature = features['path_curvature']
            
            if abs(curvature) < 0.001:
                probabilities['直行'] += 0.2
            elif curvature > 0:
                probabilities['右转'] += min(0.3, curvature * 100)
            else:
                probabilities['左转'] += min(0.3, abs(curvature) * 100)
        
        # 4. 基于历史一致性
        if 'historical_consistency' in features:
            consistency = features['historical_consistency']
            # 历史一致性高，保持当前方向
            for direction in probabilities:
                probabilities[direction] += consistency * 0.1
        
        # 归一化概率
        total = sum(probabilities.values())
        if total > 0:
            for direction in probabilities:
                probabilities[direction] /= total
        
        return probabilities
    
    def _calculate_confidence(self, features: Dict[str, Any], probabilities: Dict[str, float]) -> float:
        """计算置信度"""
        confidence_factors = []
        
        # 1. 概率分布清晰度
        max_prob = max(probabilities.values())
        min_prob = min(probabilities.values())
        clarity = (max_prob - min_prob) / max_prob if max_prob > 0 else 0
        confidence_factors.append(clarity * 0.4)
        
        # 2. 特征质量
        feature_quality = 0.0
        if 'lane_convergence' in features:
            feature_quality += 0.3
        if 'path_curvature' in features:
            feature_quality += 0.2
        if 'historical_consistency' in features:
            feature_quality += 0.1
        
        confidence_factors.append(feature_quality * 0.3)
        
        # 3. 历史一致性
        if 'historical_consistency' in features:
            confidence_factors.append(features['historical_consistency'] * 0.3)
        
        # 综合置信度
        confidence = sum(confidence_factors)
        return min(max(confidence, 0.0), 1.0)
    
    def _apply_historical_smoothing(self, direction: str, confidence: float) -> Tuple[str, float]:
        """应用历史平滑"""
        if len(self.history) < 2:
            return direction, confidence
        
        # 获取最近的历史结果
        recent_history = list(self.history)[-3:]
        
        # 统计方向频率
        direction_counts = {}
        for result in recent_history:
            d = result['direction']
            direction_counts[d] = direction_counts.get(d, 0) + 1
        
        # 如果有明显的主要方向
        if direction_counts:
            most_common_direction = max(direction_counts.items(), key=lambda x: x[1])[0]
            frequency = direction_counts[most_common_direction] / len(recent_history)
            
            # 如果某个方向出现频率超过阈值，使用该方向
            if frequency > 0.7 and most_common_direction != direction:
                # 平滑过渡
                if confidence < 0.6:  # 当前置信度较低时更容易改变
                    return most_common_direction, confidence * 0.9
        
        return direction, confidence
    
    def _generate_reasoning(self, features: Dict[str, Any], probabilities: Dict[str, float]) -> str:
        """生成推理说明"""
        reasons = []
        
        # 基于特征添加原因
        if 'lane_convergence' in features:
            convergence = features['lane_convergence']
            if convergence < 0.6:
                reasons.append("车道明显收敛")
            elif convergence > 1.4:
                reasons.append("车道明显发散")
            else:
                reasons.append("车道基本平行")
        
        if 'path_curvature' in features:
            curvature = features['path_curvature']
            if abs(curvature) < 0.001:
                reasons.append("路径基本直线")
            elif curvature > 0:
                reasons.append("路径向右弯曲")
            else:
                reasons.append("路径向左弯曲")
        
        # 基于概率添加原因
        max_direction = max(probabilities.items(), key=lambda x: x[1])
        if max_direction[1] > 0.5:
            reasons.append(f"{max_direction[0]}概率较高")
        
        return "，".join(reasons) if reasons else "特征不明显"
    
    def _create_default_result(self) -> Dict[str, Any]:
        """创建默认结果"""
        return {
            'direction': '未知',
            'confidence': 0.0,
            'probabilities': {'直行': 0.33, '左转': 0.33, '右转': 0.33},
            'features': {},
            'reasoning': '检测失败'
        }

# ==================== 智能可视化引擎 ====================
class SmartVisualizer:
    """智能可视化引擎 - 使用渐变动画和智能布局"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.colors = {
            # 道路相关
            'road_area': (0, 180, 0, 100),      # 半透明绿色
            'road_boundary': (0, 255, 255, 200), # 黄色边界
            'road_highlight': (0, 255, 0, 50),   # 高亮绿色
            
            # 车道线
            'left_lane': (255, 100, 100, 200),   # 浅红色
            'right_lane': (100, 100, 255, 200),  # 浅蓝色
            'center_line': (255, 255, 0, 180),   # 青色
            
            # 路径预测
            'future_path': (255, 0, 255, 180),   # 紫色
            'prediction_points': (255, 150, 255, 220), # 浅紫色
            
            # 界面元素
            'text_primary': (255, 255, 255, 255), # 白色
            'text_secondary': (200, 200, 200, 255), # 灰色
            'text_highlight': (0, 255, 255, 255), # 青色高亮
            
            # 状态指示
            'status_good': (0, 255, 0, 255),     # 绿色
            'status_warning': (255, 165, 0, 255), # 橙色
            'status_error': (255, 0, 0, 255),    # 红色
            
            # 背景
            'overlay_bg': (0, 0, 0, 180),        # 半透明黑色
            'panel_bg': (30, 30, 40, 220),       # 深色面板
        }
        
        self.animation_state = {}
    
    def create_visualization(self, image: np.ndarray, road_info: Dict[str, Any],
                           lane_info: Dict[str, Any], direction_info: Dict[str, Any]) -> np.ndarray:
        """创建可视化结果"""
        try:
            # 创建副本
            visualization = image.copy()
            
            # 绘制道路区域
            if road_info.get('features', {}).get('contour'):
                visualization = self._draw_road_area(visualization, road_info)
            
            # 绘制车道线
            visualization = self._draw_lanes(visualization, lane_info)
            
            # 绘制路径预测
            if lane_info.get('future_path'):
                visualization = self._draw_future_path(visualization, lane_info['future_path'])
            
            # 绘制信息面板
            visualization = self._draw_info_panel(visualization, direction_info, lane_info)
            
            # 绘制方向指示器
            visualization = self._draw_direction_indicator(visualization, direction_info)
            
            # 应用全局效果
            visualization = self._apply_global_effects(visualization)
            
            return visualization
            
        except Exception as e:
            print(f"可视化创建失败: {e}")
            return image
    
    def _draw_road_area(self, image: np.ndarray, road_info: Dict[str, Any]) -> np.ndarray:
        """绘制道路区域"""
        contour = road_info['features'].get('contour')
        if contour is None:
            return image
        
        # 创建道路区域图层
        road_layer = image.copy()
        
        # 填充道路区域
        cv2.drawContours(road_layer, [contour], -1, self.colors['road_area'][:3], -1)
        
        # 绘制道路边界
        cv2.drawContours(road_layer, [contour], -1, self.colors['road_boundary'][:3], 3)
        
        # 混合图层
        alpha = self.colors['road_area'][3] / 255.0
        cv2.addWeighted(road_layer, alpha, image, 1 - alpha, 0, image)
        
        return image
    
    def _draw_lanes(self, image: np.ndarray, lane_info: Dict[str, Any]) -> np.ndarray:
        """绘制车道线"""
        # 绘制原始检测线段
        for side, color_key in [('left_lines', 'left_lane'), ('right_lines', 'right_lane')]:
            lines = lane_info.get(side, [])
            color = self.colors[color_key]
            
            for line in lines:
                points = line['points']
                cv2.line(image, points[0], points[1], color[:3], 2, cv2.LINE_AA)
        
        # 绘制拟合的车道线
        lane_layer = image.copy()
        
        for side, color_key in [('left_lane', 'left_lane'), ('right_lane', 'right_lane')]:
            lane = lane_info.get(side)
            if lane and 'points' in lane:
                points = lane['points']
                if len(points) == 2:
                    color = self.colors[color_key]
                    thickness = 4 + int(lane.get('confidence', 0.5) * 4)
                    cv2.line(lane_layer, points[0], points[1], color[:3], thickness, cv2.LINE_AA)
        
        # 绘制中心线
        if lane_info.get('center_line'):
            center = lane_info['center_line']
            if 'points' in center and len(center['points']) == 2:
                color = self.colors['center_line']
                cv2.line(lane_layer, center['points'][0], center['points'][1], 
                        color[:3], 3, cv2.LINE_AA)
        
        # 混合车道线图层
        cv2.addWeighted(lane_layer, 0.7, image, 0.3, 0, image)
        
        return image
    
    def _draw_future_path(self, image: np.ndarray, future_path: Dict[str, Any]) -> np.ndarray:
        """绘制未来路径"""
        path_layer = image.copy()
        path_points = future_path.get('center_path', [])
        
        if len(path_points) < 2:
            return image
        
        # 绘制路径线
        color = self.colors['future_path']
        thickness = 5
        
        # 绘制渐变路径
        for i in range(len(path_points) - 1):
            # 计算透明度渐变
            alpha = 0.5 + 0.5 * (i / (len(path_points) - 1))
            line_color = tuple(int(c * alpha) for c in color[:3])
            
            cv2.line(path_layer, path_points[i], path_points[i + 1], 
                    line_color, thickness - i, cv2.LINE_AA)
        
        # 绘制路径点
        point_color = self.colors['prediction_points']
        for i, point in enumerate(path_points):
            radius = 4 - int(i / len(path_points) * 2)
            cv2.circle(path_layer, point, radius, point_color[:3], -1)
        
        # 混合路径图层
        cv2.addWeighted(path_layer, 0.6, image, 0.4, 0, image)
        
        return image
    
    def _draw_info_panel(self, image: np.ndarray, direction_info: Dict[str, Any],
                        lane_info: Dict[str, Any]) -> np.ndarray:
        """绘制信息面板"""
        height, width = image.shape[:2]
        
        # 创建半透明背景
        panel_height = 120
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), self.colors['overlay_bg'][:3], -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 绘制主信息
        direction = direction_info.get('direction', '未知')
        confidence = direction_info.get('confidence', 0.0)
        
        # 根据置信度选择颜色
        if confidence > 0.7:
            color = self.colors['status_good'][:3]
        elif confidence > 0.4:
            color = self.colors['status_warning'][:3]
        else:
            color = self.colors['status_error'][:3]
        
        # 绘制方向文本
        font_scale = 1.2
        thickness = 2
        
        direction_text = f"方向: {direction}"
        cv2.putText(image, direction_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # 绘制置信度
        confidence_text = f"置信度: {confidence:.1%}"
        cv2.putText(image, confidence_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness - 1)
        
        # 绘制检测质量
        quality = lane_info.get('detection_quality', 0.0)
        quality_text = f"检测质量: {quality:.1%}"
        cv2.putText(image, quality_text, (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text_secondary'][:3], 1)
        
        # 绘制概率分布（右侧）
        if 'probabilities' in direction_info:
            probabilities = direction_info['probabilities']
            start_x = width - 200
            start_y = 30
            
            for i, (direction, prob) in enumerate(probabilities.items()):
                y = start_y + i * 25
                prob_text = f"{direction}: {prob:.1%}"
                cv2.putText(image, prob_text, (start_x, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text_primary'][:3], 1)
        
        return image
    
    def _draw_direction_indicator(self, image: np.ndarray, direction_info: Dict[str, Any]) -> np.ndarray:
        """绘制方向指示器"""
        height, width = image.shape[:2]
        direction = direction_info.get('direction', '未知')
        confidence = direction_info.get('confidence', 0.0)
        
        # 指示器位置
        center_x = width // 2
        indicator_y = height - 150
        
        # 创建指示器图层
        indicator_layer = np.zeros_like(image)
        
        # 根据方向绘制不同形状
        if direction == "左转":
            # 绘制左转箭头
            points = np.array([
                (center_x, indicator_y),
                (center_x - 80, indicator_y),
                (center_x - 60, indicator_y - 40),
                (center_x - 100, indicator_y - 40),
                (center_x - 120, indicator_y),
                (center_x - 200, indicator_y),
                (center_x - 100, indicator_y + 80),
                (center_x, indicator_y + 80)
            ])
            color = (0, 100, 255)  # 橙色
        elif direction == "右转":
            # 绘制右转箭头
            points = np.array([
                (center_x, indicator_y),
                (center_x + 80, indicator_y),
                (center_x + 60, indicator_y - 40),
                (center_x + 100, indicator_y - 40),
                (center_x + 120, indicator_y),
                (center_x + 200, indicator_y),
                (center_x + 100, indicator_y + 80),
                (center_x, indicator_y + 80)
            ])
            color = (0, 100, 255)  # 橙色
        else:  # 直行或未知
            # 绘制直行箭头
            points = np.array([
                (center_x - 60, indicator_y + 40),
                (center_x, indicator_y - 40),
                (center_x + 60, indicator_y + 40),
                (center_x + 40, indicator_y + 40),
                (center_x + 40, indicator_y + 120),
                (center_x - 40, indicator_y + 120),
                (center_x - 40, indicator_y + 40)
            ])
            color = (0, 255, 0)  # 绿色
        
        # 绘制指示器
        cv2.fillPoly(indicator_layer, [points], color)
        
        # 根据置信度调整透明度
        alpha = 0.3 + confidence * 0.5
        cv2.addWeighted(indicator_layer, alpha, image, 1 - alpha, 0, image)
        
        # 绘制边框
        cv2.polylines(image, [points], True, (255, 255, 255), 2, cv2.LINE_AA)
        
        return image
    
    def _apply_global_effects(self, image: np.ndarray) -> np.ndarray:
        """应用全局效果"""
        # 轻微锐化
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # 混合原始图像和锐化图像
        cv2.addWeighted(sharpened, 0.3, image, 0.7, 0, image)
        
        return image

# ==================== 主应用程序 ====================
class AdvancedLaneDetectionApp:
    """高级道路方向识别系统主应用程序"""
    
    def __init__(self, root):
        self.root = root
        self._setup_window()
        
        # 初始化配置和组件
        self.config = AppConfig()
        self.image_processor = SmartImageProcessor(self.config)
        self.road_detector = AdvancedRoadDetector(self.config)
        self.lane_detector = SmartLaneDetector(self.config)
        self.direction_analyzer = AdvancedDirectionAnalyzer(self.config)
        self.visualizer = SmartVisualizer(self.config)
        
        # 状态变量
        self.current_image = None
        self.current_image_path = None
        self.is_processing = False
        self.processing_history = deque(maxlen=10)
        
        # 性能监控
        self.processing_times = []
        self.average_time = 0
        
        # 创建界面
        self._create_ui()
        
        print("高级道路方向识别系统已启动")
    
    def _setup_window(self):
        """设置窗口"""
        self.root.title("🚗 智能道路方向识别系统")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)
        
        # 设置窗口图标（如果有的话）
        try:
            self.root.iconbitmap(default=None)
        except:
            pass
        
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
            text="智能道路方向识别系统",
            font=("微软雅黑", 16, "bold"),
            foreground="#2c3e50"
        )
        title_label.pack(side="left")
        
        # 版本信息
        version_label = ttk.Label(
            title_frame,
            text="v2.0 - 高级版",
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
            text="📁 选择图片",
            command=self._select_image,
            width=20
        )
        select_btn.pack(pady=(0, 10))
        
        # 重新检测按钮
        self.redetect_btn = ttk.Button(
            file_frame,
            text="🔄 重新检测",
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
        
        # 预测距离调节
        ttk.Label(param_frame, text="预测距离:").pack(anchor="w", pady=(0, 5))
        self.prediction_var = tk.DoubleVar(value=self.config.prediction_distance)
        prediction_scale = ttk.Scale(
            param_frame,
            from_=0.3,
            to=0.9,
            variable=self.prediction_var,
            orient="horizontal",
            command=self._on_parameter_change,
            length=250
        )
        prediction_scale.pack(fill="x", pady=(0, 10))
        
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
        
        # 统计信息区域
        stats_frame = ttk.LabelFrame(
            display_frame,
            text="统计信息",
            padding="10",
            relief="groove"
        )
        stats_frame.pack(fill="x", pady=(10, 0))
        
        # 创建统计信息显示
        self._create_stats_display(stats_frame)
        
        return display_frame
    
    def _create_stats_display(self, parent):
        """创建统计信息显示"""
        stats_grid = ttk.Frame(parent)
        stats_grid.pack(fill="x")
        
        # 处理次数
        ttk.Label(stats_grid, text="处理次数:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.process_count_label = ttk.Label(stats_grid, text="0")
        self.process_count_label.grid(row=0, column=1, sticky="w", padx=(0, 30))
        
        # 平均处理时间
        ttk.Label(stats_grid, text="平均时间:").grid(row=0, column=2, sticky="w", padx=(0, 10))
        self.avg_time_label = ttk.Label(stats_grid, text="0.00s")
        self.avg_time_label.grid(row=0, column=3, sticky="w", padx=(0, 30))
        
        # 缓存命中率
        ttk.Label(stats_grid, text="缓存命中:").grid(row=0, column=4, sticky="w", padx=(0, 10))
        self.cache_hit_label = ttk.Label(stats_grid, text="0%")
        self.cache_hit_label.grid(row=0, column=5, sticky="w")
    
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
            road_info = self.road_detector.detect(self.current_image, roi_info)
            
            # 3. 车道线检测
            lane_info = self.lane_detector.detect(self.current_image, roi_info['mask'])
            
            # 4. 方向分析
            direction_info = self.direction_analyzer.analyze(road_info['features'], lane_info)
            
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
            self.average_time = np.mean(self.processing_times)
            
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
            
            # 更新方向信息
            direction = direction_info['direction']
            confidence = direction_info['confidence']
            quality = lane_info.get('detection_quality', 0.0)
            
            # 设置方向文本和颜色
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
            
            # 更新统计信息
            self.process_count_label.config(text=str(len(self.processing_history)))
            self.avg_time_label.config(text=f"{self.average_time:.3f}s")
            
            # 更新缓存命中率（这里简化处理）
            cache_hit_rate = len(self.image_processor._cache) / self.image_processor.config.cache_size
            self.cache_hit_label.config(text=f"{cache_hit_rate:.1%}")
            
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
            canvas.update()
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
        prediction = self.prediction_var.get()
        
        # 根据敏感度调整参数
        self.config.canny_threshold1 = int(50 * (0.5 + sensitivity * 0.5))
        self.config.canny_threshold2 = int(150 * (0.5 + sensitivity * 0.5))
        self.config.hough_threshold = int(30 * (1.5 - sensitivity * 0.5))
        self.config.prediction_distance = prediction
        
        print(f"参数更新: 敏感度={sensitivity:.2f}, 预测距离={prediction:.2f}")
        
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
        app = AdvancedLaneDetectionApp(root)
        
        # 运行主循环
        root.mainloop()
        
    except Exception as e:
        print(f"应用程序启动失败: {e}")
        messagebox.showerror("致命错误", f"应用程序启动失败: {str(e)}")

if __name__ == "__main__":
    main()