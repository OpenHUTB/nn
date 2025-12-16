import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Deque
from collections import deque, defaultdict
import math
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置管理 ====================
@dataclass
class AppConfig:
    """应用配置参数 - 移除 __slots__ 以支持属性修改"""
    
    # 性能参数
    max_image_size: Tuple[int, int] = (1200, 800)
    cache_size: int = 5
    batch_size: int = 50
    
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
    min_contour_area: float = 0.01
    
    # 方向分析参数
    deviation_threshold: float = 0.15
    width_ratio_threshold: float = 0.7
    confidence_threshold: float = 0.5
    
    # 路径预测参数
    prediction_steps: int = 8
    prediction_distance: float = 0.8
    min_prediction_points: int = 3
    
    # 界面参数
    ui_refresh_rate: int = 100
    animation_duration: int = 300

# ==================== 图像处理优化 ====================
class SmartImageProcessor:
    """智能图像处理器 - 优化内存使用和性能"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._cache = {}
        self._cache_order = deque(maxlen=config.cache_size)
        self._roi_cache = {}
        
        # 预定义常用滤波器核
        self._kernels = {
            'morph_close_3x3': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            'morph_open_3x3': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            'morph_close_5x5': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            'gaussian_5x5': cv2.getGaussianKernel(5, 0),
        }
    
    def load_and_preprocess(self, image_path: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """优化版：使用缓存和并行预处理"""
        try:
            # 检查缓存
            if image_path in self._cache:
                return self._cache[image_path]
            
            # 异步读取图像
            image = self._load_image_optimized(image_path)
            if image is None:
                return None
            
            # 并行预处理
            processed, roi_info = self._parallel_preprocess(image)
            
            # 更新缓存
            self._update_cache(image_path, (processed, roi_info))
            
            return processed, roi_info
            
        except Exception as e:
            print(f"图像处理失败: {e}")
            return None
    
    def _load_image_optimized(self, image_path: str) -> Optional[np.ndarray]:
        """优化图像加载"""
        # 检查文件大小，避免加载过大文件
        try:
            file_size = os.path.getsize(image_path)
            if file_size > 50 * 1024 * 1024:  # 50MB
                print(f"警告: 图像文件过大 ({file_size / 1024 / 1024:.1f}MB)")
                return None
            
            # 使用优化参数读取
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                return None
            
            # 检查图像深度和通道
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            return image
            
        except Exception as e:
            print(f"加载图像失败: {e}")
            return None
    
    def _parallel_preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """并行预处理"""
        # 调整尺寸
        resized = self._smart_resize(image)
        
        # 并行执行预处理步骤
        with ThreadPoolExecutor(max_workers=2) as executor:
            enhanced_future = executor.submit(self._enhance_image, resized)
            roi_future = executor.submit(self._calculate_roi, resized.shape)
            
            enhanced = enhanced_future.result()
            roi_info = roi_future.result()
        
        return enhanced, roi_info
    
    def _smart_resize(self, image: np.ndarray) -> np.ndarray:
        """优化版：智能调整尺寸"""
        height, width = image.shape[:2]
        max_w, max_h = self.config.max_image_size
        
        # 如果图像小于最大尺寸，直接返回副本
        if width <= max_w and height <= max_h:
            return image.copy()
        
        # 计算保持宽高比的缩放比例
        scale = min(max_w / width, max_h / height)
        new_size = (int(width * scale), int(height * scale))
        
        # 使用适当的插值方法
        if scale < 1.0:
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """优化版：图像增强"""
        # 1. 转换为YUV并处理Y通道
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0]
        
        # 2. 使用CLAHE增强对比度
        clahe = cv2.createCLAHE(
            clipLimit=self.config.adaptive_clip_limit,
            tileGridSize=self.config.adaptive_grid_size
        )
        yuv[:, :, 0] = clahe.apply(y_channel)
        
        # 3. 转换回BGR
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # 4. 选择性去噪
        if self._estimate_noise_level(y_channel) > 25:
            enhanced = cv2.bilateralFilter(
                enhanced, 
                d=9, 
                sigmaColor=75, 
                sigmaSpace=75
            )
        
        return enhanced
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """快速噪声估计"""
        # 使用块状方法加速计算
        h, w = image.shape
        block_size = 32
        
        if h < 100 or w < 100:
            # 小图像直接计算
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            return float(np.std(laplacian))
        
        # 随机采样块进行估计
        noise_levels = []
        for _ in range(10):
            y = np.random.randint(0, h - block_size)
            x = np.random.randint(0, w - block_size)
            block = image[y:y+block_size, x:x+block_size]
            laplacian = cv2.Laplacian(block, cv2.CV_64F)
            noise_levels.append(np.std(laplacian))
        
        return float(np.mean(noise_levels))
    
    def _calculate_roi(self, image_shape: Tuple[int, int]) -> Dict:
        """优化ROI计算"""
        height, width = image_shape[:2]
        
        # 预计算的ROI参数
        roi_top = int(height * 0.35)
        roi_bottom = int(height * 0.92)
        roi_width = int(width * 0.85)
        
        # 创建顶点数组
        vertices = np.array([[
            ((width - roi_width) // 2, roi_bottom),
            ((width - roi_width) // 2 + int(roi_width * 0.3), roi_top),
            ((width - roi_width) // 2 + int(roi_width * 0.7), roi_top),
            ((width + roi_width) // 2, roi_bottom)
        ]], dtype=np.int32)
        
        # 创建掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, vertices, 255)
        
        return {
            'vertices': vertices,
            'mask': mask,
            'bounds': (roi_top, roi_bottom, roi_width)
        }
    
    def _update_cache(self, key: str, value: Any):
        """优化缓存更新"""
        if key in self._cache:
            self._cache_order.remove(key)
        
        if len(self._cache) >= self.config.cache_size:
            oldest = self._cache_order.popleft()
            del self._cache[oldest]
        
        self._cache[key] = value
        self._cache_order.append(key)
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_order.clear()
        self._roi_cache.clear()

# ==================== 高级道路检测器优化 ====================
class AdvancedRoadDetector:
    """优化版：减少计算复杂度和内存使用"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.previous_results = deque(maxlen=3)
        
        # 预编译常用函数
        self._detection_methods = [
            self._detect_by_color,
            self._detect_by_texture,
            self._detect_by_edges
        ]
    
    def detect(self, image: np.ndarray, roi_info: Dict) -> Dict[str, Any]:
        """优化版检测：使用早期终止和结果缓存"""
        try:
            # 提取ROI
            roi_region = cv2.bitwise_and(image, image, mask=roi_info['mask'])
            
            # 多尺度检测
            scales = [1.0, 0.8, 0.6]
            scale_results = []
            
            for scale in scales:
                if scale != 1.0:
                    # 快速缩放
                    new_size = (int(roi_region.shape[1] * scale), 
                              int(roi_region.shape[0] * scale))
                    scaled = cv2.resize(roi_region, new_size, cv2.INTER_AREA)
                else:
                    scaled = roi_region
                
                # 并行执行检测方法
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(method, scaled) 
                             for method in self._detection_methods]
                    
                    detection_results = []
                    for future in futures:
                        result = future.result()
                        if result['confidence'] > 0.3:
                            detection_results.append(result)
                
                # 早期终止：如果检测结果太少，跳过当前尺度
                if len(detection_results) < 2:
                    continue
                
                # 融合结果
                fused = self._fuse_detection_results(detection_results)
                
                # 提取特征
                features = self._extract_features_optimized(fused, scaled.shape)
                
                scale_results.append({
                    'scale': scale,
                    'fused_mask': fused['mask'],
                    'features': features,
                    'confidence': fused['confidence'],
                    'detection_methods': len(detection_results)
                })
            
            if not scale_results:
                return self._create_empty_result()
            
            # 融合多尺度结果
            fused_result = self._fuse_results_optimized(scale_results)
            
            # 时间平滑
            if self.previous_results:
                fused_result = self._temporal_smooth(fused_result)
            
            # 更新历史
            self.previous_results.append(fused_result)
            
            return fused_result
            
        except Exception as e:
            print(f"道路检测失败: {e}")
            return self._create_empty_result()
    
    def _detect_by_color(self, roi_region: np.ndarray) -> Dict[str, Any]:
        """优化版颜色检测"""
        # 使用LAB颜色空间
        lab = cv2.cvtColor(roi_region, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # 快速统计
        mean_l = np.mean(l_channel)
        std_l = np.std(l_channel)
        
        # 自适应阈值
        lower = max(0, int(mean_l - std_l * 1.5))
        upper = min(255, int(mean_l + std_l * 1.5))
        
        # 创建掩码
        mask = cv2.inRange(l_channel, lower, upper)
        
        # 快速形态学操作
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 计算置信度
        area_ratio = np.count_nonzero(mask) / (mask.size + 1e-6)
        confidence = min(1.0, area_ratio * 2.0)
        
        return {'mask': mask, 'confidence': confidence, 'method': 'color'}
    
    def _detect_by_texture(self, roi_region: np.ndarray) -> Dict[str, Any]:
        """优化版纹理检测"""
        # 转换为灰度
        gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
        
        # 使用简单的方差纹理分析
        kernel_size = 5
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        variance = cv2.blur(gray.astype(np.float32)**2, (kernel_size, kernel_size))
        variance = variance - mean.astype(np.float32)**2
        
        # 阈值化
        mean_var = np.mean(variance)
        mask = (variance < mean_var * 0.5).astype(np.uint8) * 255
        
        # 计算均匀性
        uniformity = 1.0 / (1.0 + np.mean(variance))
        confidence = min(1.0, uniformity * 1.5)
        
        return {'mask': mask, 'confidence': confidence, 'method': 'texture'}
    
    def _detect_by_edges(self, roi_region: np.ndarray) -> Dict[str, Any]:
        """优化版边缘检测"""
        gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
        
        # 自适应Canny
        median = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        
        edges = cv2.Canny(gray, lower, upper)
        
        # 填充边缘区域
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(edges)
        
        # 快速轮廓填充
        min_area = gray.shape[0] * gray.shape[1] * 0.001
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # 置信度计算
        edge_density = np.count_nonzero(edges) / (edges.size + 1e-6)
        confidence = min(1.0, 1.0 - edge_density * 2.0)
        
        return {'mask': mask, 'confidence': confidence, 'method': 'edges'}
    
    def _fuse_detection_results(self, results: List[Dict]) -> Dict[str, Any]:
        """优化版结果融合"""
        if not results:
            return {'mask': np.zeros((1, 1), dtype=np.uint8), 'confidence': 0.0}
        
        # 权重计算
        confidences = np.array([r['confidence'] for r in results])
        weights = confidences / (np.sum(confidences) + 1e-6)
        
        # 加权融合
        fused = np.zeros_like(results[0]['mask'], dtype=np.float32)
        for r, weight in zip(results, weights):
            fused += r['mask'].astype(np.float32) * weight
        
        # 二值化
        fused_binary = (fused > 127).astype(np.uint8) * 255
        
        # 形态学优化
        kernel = np.ones((5, 5), np.uint8)
        fused_binary = cv2.morphologyEx(fused_binary, cv2.MORPH_CLOSE, kernel)
        
        return {
            'mask': fused_binary,
            'confidence': float(np.mean(confidences)),
            'methods': len(results)
        }
    
    def _extract_features_optimized(self, detection_result: Dict, 
                                   image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """优化版特征提取"""
        mask = detection_result['mask']
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {}
        
        # 找到最大轮廓
        main_contour = max(contours, key=cv2.contourArea)
        
        # 快速轮廓近似
        epsilon = 0.01 * cv2.arcLength(main_contour, True)
        approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # 计算凸包
        hull = cv2.convexHull(approx_contour)
        
        # 计算基本特征
        area = cv2.contourArea(approx_contour)
        perimeter = cv2.arcLength(approx_contour, True)
        rect = cv2.boundingRect(approx_contour)
        
        # 计算质心
        M = cv2.moments(approx_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = rect[0] + rect[2] // 2, rect[1] + rect[3] // 2
        
        # 计算方向
        angle = 0
        if len(approx_contour) >= 5:
            _, _, angle = cv2.fitEllipse(approx_contour)
        
        # 计算坚实度
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            'contour': approx_contour,
            'centroid': (cx, cy),
            'area': area,
            'perimeter': perimeter,
            'bounding_rect': rect,
            'orientation': angle,
            'solidity': solidity
        }
    
    def _fuse_results_optimized(self, results: List[Dict]) -> Dict[str, Any]:
        """优化版多尺度结果融合"""
        if not results:
            return self._create_empty_result()
        
        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 使用最佳结果
        best_result = results[0]
        
        # 加权平均特征
        total_confidence = sum(r['confidence'] for r in results)
        weighted_features = {}
        
        # 只对数值特征进行加权平均
        for key in ['area', 'perimeter', 'orientation', 'solidity']:
            if key in best_result['features']:
                weighted_value = sum(r['features'].get(key, 0) * r['confidence'] 
                                   for r in results)
                weighted_features[key] = weighted_value / total_confidence
        
        # 保持非数值特征
        for key in ['contour', 'centroid', 'bounding_rect']:
            if key in best_result['features']:
                weighted_features[key] = best_result['features'][key]
        
        return {
            'features': weighted_features,
            'confidence': np.mean([r['confidence'] for r in results]),
            'num_scales': len(results),
            'detection_methods': best_result.get('detection_methods', 0)
        }
    
    def _temporal_smooth(self, current_result: Dict[str, Any]) -> Dict[str, Any]:
        """时间平滑"""
        if len(self.previous_results) < 2:
            return current_result
        
        alpha = 0.7  # 平滑因子
        smoothed_features = current_result['features'].copy()
        
        # 只对数值特征进行平滑
        for prev_result in list(self.previous_results)[:-1]:
            for key in ['area', 'perimeter', 'orientation', 'solidity']:
                if key in smoothed_features and key in prev_result['features']:
                    smoothed_features[key] = (alpha * smoothed_features[key] + 
                                            (1 - alpha) * prev_result['features'][key])
        
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

# ==================== 智能车道线检测器优化 ====================
class SmartLaneDetector:
    """优化版：减少计算复杂度和内存使用"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.lane_history = deque(maxlen=5)
        
        # 预计算常用值
        self._precomputed = {
            'angles': np.linspace(-np.pi/3, np.pi/3, 13),
            'y_points': np.linspace(0.4, 1.0, 5)
        }
    
    def detect(self, image: np.ndarray, roi_mask: np.ndarray) -> Dict[str, Any]:
        """优化版车道线检测"""
        try:
            # 预处理
            processed = self._preprocess_optimized(image, roi_mask)
            
            # 并行检测
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    'canny': executor.submit(self._detect_with_canny, processed),
                    'sobel': executor.submit(self._detect_with_sobel, processed),
                    'gradient': executor.submit(self._detect_with_gradient, processed)
                }
                
                # 收集结果
                all_lines = []
                for future in futures.values():
                    lines = future.result()
                    if lines is not None and len(lines) > 0:
                        all_lines.extend(lines)
            
            if not all_lines:
                return self._create_empty_lane_result()
            
            # 分类和过滤
            left_lines, right_lines = self._classify_and_filter_optimized(all_lines, image.shape[1])
            
            # 拟合车道线
            left_lane = self._fit_lane_model_optimized(left_lines, image.shape)
            right_lane = self._fit_lane_model_optimized(right_lines, image.shape)
            
            # 验证车道线
            left_lane, right_lane = self._validate_lanes_optimized(left_lane, right_lane, image.shape)
            
            # 预测路径
            future_path = None
            if left_lane and right_lane:
                future_path = self._predict_future_path_optimized(left_lane, right_lane, image.shape)
            
            # 创建结果
            result = {
                'left_lines': left_lines,
                'right_lines': right_lines,
                'left_lane': left_lane,
                'right_lane': right_lane,
                'future_path': future_path,
                'detection_quality': self._calculate_detection_quality(left_lane, right_lane)
            }
            
            # 更新历史
            self.lane_history.append(result)
            
            # 时间平滑
            if len(self.lane_history) > 1:
                result = self._temporal_smooth_lanes(result)
            
            return result
            
        except Exception as e:
            print(f"车道线检测失败: {e}")
            return self._create_empty_lane_result()
    
    def _preprocess_optimized(self, image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """优化版预处理"""
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用ROI
        gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
        
        # CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 选择性去噪
        if np.std(enhanced) > 30:
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def _detect_with_canny(self, image: np.ndarray) -> List[np.ndarray]:
        """优化版Canny检测"""
        # 快速中值估计
        median = np.median(image)
        lower = int(max(0, 0.66 * median))
        upper = int(min(255, 1.33 * median))
        
        edges = cv2.Canny(image, lower, upper)
        
        # 霍夫变换
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.hough_min_length,
            maxLineGap=self.config.hough_max_gap
        )
        
        return [] if lines is None else lines.tolist()
    
    def _detect_with_sobel(self, image: np.ndarray) -> List[np.ndarray]:
        """优化版Sobel检测"""
        # 计算梯度
        sobelx = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)
        
        # 计算梯度方向和幅值
        magnitude = np.sqrt(sobelx.astype(np.float32)**2 + sobely.astype(np.float32)**2)
        direction = np.arctan2(np.abs(sobely), np.abs(sobelx))
        
        # 过滤垂直方向梯度
        vertical_mask = (direction > np.pi/4) & (direction < 3*np.pi/4)
        lanes = np.where(vertical_mask, magnitude, 0)
        
        # 转换为8位
        lanes = np.uint8(255 * lanes / (np.max(lanes) + 1e-6))
        
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
        
        return [] if lines is None else lines.tolist()
    
    def _detect_with_gradient(self, image: np.ndarray) -> List[np.ndarray]:
        """优化版梯度方向检测"""
        # 计算梯度
        dx = cv2.Scharr(image, cv2.CV_16S, 1, 0)
        dy = cv2.Scharr(image, cv2.CV_16S, 0, 1)
        
        # 计算梯度方向
        direction = np.arctan2(dy, dx)
        
        # 预计算的搜索角度
        all_lines = []
        for angle in self._precomputed['angles']:
            mask = np.abs(direction - angle) < np.pi/18
            
            if np.sum(mask) > 100:  # 有足够的像素
                # 计算幅值
                magnitude = np.sqrt(dx.astype(np.float32)**2 + dy.astype(np.float32)**2)
                masked_magnitude = np.where(mask, magnitude, 0)
                
                # 转换为8位
                result = np.uint8(255 * masked_magnitude / (np.max(masked_magnitude) + 1e-6))
                
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
                    all_lines.extend(lines.tolist())
        
        return all_lines
    
    def _classify_and_filter_optimized(self, lines: List[List], image_width: int) -> Tuple[List, List]:
        """优化版分类和过滤"""
        left_lines, right_lines = [], []
        
        for line in lines:
            if len(line[0]) != 4:
                continue
                
            x1, y1, x2, y2 = line[0]
            
            # 跳过垂直线
            if x2 == x1:
                continue
            
            # 计算斜率
            dx = x2 - x1
            dy = y2 - y1
            slope = dy / dx
            
            # 过滤标准
            if abs(slope) < 0.3:
                continue
            
            length = np.sqrt(dx**2 + dy**2)
            if length < 20:
                continue
            
            # 分类
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
        
        return self._filter_lines_optimized(left_lines), self._filter_lines_optimized(right_lines)
    
    def _filter_lines_optimized(self, lines: List[Dict]) -> List[Dict]:
        """优化版过滤"""
        if len(lines) < 3:
            return lines
        
        # 提取特征
        slopes = np.array([line['slope'] for line in lines])
        midpoints = np.array([line['midpoint'][0] for line in lines])
        
        # 计算统计信息
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
    
    def _fit_lane_model_optimized(self, lines: List[Dict], image_shape: Tuple[int, ...]) -> Optional[Dict]:
        """优化版车道线拟合"""
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
            'coeffs': coeffs.tolist(),
            'points': [(x_bottom, y_bottom), (x_top, y_top)],
            'model_type': model_type,
            'confidence': confidence,
            'num_lines': len(lines)
        }
    
    def _validate_lanes_optimized(self, left_lane: Optional[Dict], right_lane: Optional[Dict],
                                image_shape: Tuple[int, ...]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """优化版车道线验证"""
        if left_lane is None or right_lane is None:
            return left_lane, right_lane
        
        height, width = image_shape[:2]
        
        # 检查车道宽度
        if left_lane['model_type'] == 'quadratic' and right_lane['model_type'] == 'quadratic':
            # 在几个点采样宽度
            y_points = self._precomputed['y_points'] * height
            widths = []
            
            for y in y_points:
                left_x = left_lane['func'](y)
                right_x = right_lane['func'](y)
                widths.append(right_x - left_x)
            
            avg_width = np.mean(widths)
            std_width = np.std(widths)
            
            # 检查宽度合理性
            min_width = width * 0.15
            max_width = width * 0.8
            
            if avg_width < min_width or avg_width > max_width or std_width > width * 0.2:
                left_lane['confidence'] *= 0.7
                right_lane['confidence'] *= 0.7
        
        # 检查车道线交叉
        if left_lane['points'][0][0] > right_lane['points'][0][0]:
            left_lane['confidence'] *= 0.6
            right_lane['confidence'] *= 0.6
        
        return left_lane, right_lane
    
    def _predict_future_path_optimized(self, left_lane: Dict, right_lane: Dict,
                                     image_shape: Tuple[int, ...]) -> Optional[Dict]:
        """优化版路径预测"""
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
                x = max(0, min(width, x))
                path_points.append((int(x), int(y)))
            
            # 计算曲率
            curvature = 0.0
            if len(path_points) >= 3:
                curvature = self._calculate_curvature_optimized(path_points)
            
            return {
                'center_path': path_points,
                'left_boundary': [(int(left_lane['func'](y)), int(y)) for y in y_values],
                'right_boundary': [(int(right_lane['func'](y)), int(y)) for y in y_values],
                'curvature': curvature,
                'prediction_length': len(path_points)
            }
            
        except Exception as e:
            print(f"路径预测失败: {e}")
            return None
    
    def _calculate_curvature_optimized(self, points: List[Tuple[int, int]]) -> float:
        """优化版曲率计算"""
        if len(points) < 3:
            return 0.0
        
        # 转换为numpy数组
        pts = np.array(points, dtype=np.float32)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # 计算导数
        dx = np.gradient(x)
        dy = np.gradient(y)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        # 计算曲率
        denominator = (dx**2 + dy**2)**1.5
        curvature = np.abs(dx * d2y - d2x * dy) / np.maximum(denominator, 1e-6)
        
        # 返回有效曲率的平均值
        valid_curvature = curvature[np.isfinite(curvature)]
        return 0.0 if len(valid_curvature) == 0 else float(np.mean(valid_curvature))
    
    def _calculate_detection_quality(self, left_lane: Optional[Dict], right_lane: Optional[Dict]) -> float:
        """计算检测质量"""
        quality = 0.0
        
        if left_lane is not None:
            quality += left_lane['confidence'] * 0.5
        
        if right_lane is not None:
            quality += right_lane['confidence'] * 0.5
        
        if left_lane is not None and right_lane is not None:
            quality += 0.1
            if left_lane['model_type'] == right_lane['model_type']:
                quality += 0.1
        
        return min(quality, 1.0)
    
    def _temporal_smooth_lanes(self, current_result: Dict[str, Any]) -> Dict[str, Any]:
        """时间平滑"""
        if len(self.lane_history) < 2:
            return current_result
        
        alpha = 0.6
        
        # 对左车道线系数进行平滑
        if current_result['left_lane'] and len(self.lane_history) > 0:
            coeffs = np.array(current_result['left_lane']['coeffs'])
            for prev in list(self.lane_history)[:-1]:
                if prev['left_lane']:
                    prev_coeffs = np.array(prev['left_lane']['coeffs'])
                    if len(coeffs) == len(prev_coeffs):
                        coeffs = alpha * coeffs + (1 - alpha) * prev_coeffs
            
            current_result['left_lane']['coeffs'] = coeffs.tolist()
            current_result['left_lane']['func'] = np.poly1d(coeffs)
        
        # 对右车道线系数进行平滑
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

# ==================== 主应用程序优化 ====================
class AdvancedLaneDetectionApp:
    """优化版主应用程序 - 减少内存使用和提升响应速度"""
    
    def __init__(self, root):
        self.root = root
        self._setup_window()
        
        # 初始化配置和组件
        self.config = AppConfig()
        self.image_processor = SmartImageProcessor(self.config)
        self.road_detector = AdvancedRoadDetector(self.config)
        self.lane_detector = SmartLaneDetector(self.config)
        
        # 状态变量
        self.current_image = None
        self.current_image_path = None
        self.is_processing = False
        self.processing_history = deque(maxlen=10)
        
        # 性能监控
        self.processing_times = deque(maxlen=20)
        self.average_time = 0
        
        # 创建界面
        self._create_optimized_ui()
        
        print("🚗 智能道路方向识别系统已启动")
    
    def _setup_window(self):
        """设置窗口"""
        self.root.title("🚗 智能道路方向识别系统")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)
        
        # 窗口居中
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - width) // 2
        y = (self.root.winfo_screenheight() - height) // 2
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_optimized_ui(self):
        """创建优化版UI"""
        # 使用更简单的布局
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 标题
        title_label = ttk.Label(
            main_frame,
            text="智能道路方向识别系统",
            font=("微软雅黑", 16, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        # 主要内容区域
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill="both", expand=True)
        
        # 左侧控制面板
        self._create_control_panel(content_frame)
        
        # 右侧显示区域
        self._create_display_panel(content_frame)
        
        # 状态栏
        self._create_status_bar(main_frame)
    
    def _create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding=10)
        control_frame.pack(side="left", fill="y", padx=(0, 10))
        
        # 文件操作
        ttk.Button(
            control_frame,
            text="选择图片",
            command=self._select_image,
            width=20
        ).pack(pady=(0, 10))
        
        self.redetect_btn = ttk.Button(
            control_frame,
            text="重新检测",
            command=self._redetect,
            width=20,
            state="disabled"
        )
        self.redetect_btn.pack(pady=(0, 10))
        
        self.file_info_label = ttk.Label(control_frame, text="未选择图片")
        self.file_info_label.pack(pady=(0, 20))
        
        # 参数调节
        param_frame = ttk.LabelFrame(control_frame, text="参数调节", padding=10)
        param_frame.pack(fill="x", pady=(0, 20))
        
        ttk.Label(param_frame, text="检测敏感度:").pack(anchor="w")
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        ttk.Scale(
            param_frame,
            from_=0.1,
            to=1.0,
            variable=self.sensitivity_var,
            orient="horizontal",
            command=self._on_parameter_change
        ).pack(fill="x", pady=(0, 10))
        
        ttk.Label(param_frame, text="预测距离:").pack(anchor="w")
        self.prediction_var = tk.DoubleVar(value=self.config.prediction_distance)
        ttk.Scale(
            param_frame,
            from_=0.3,
            to=0.9,
            variable=self.prediction_var,
            orient="horizontal",
            command=self._on_parameter_change
        ).pack(fill="x", pady=(0, 10))
        
        # 结果显示
        result_frame = ttk.LabelFrame(control_frame, text="检测结果", padding=10)
        result_frame.pack(fill="x")
        
        self.direction_label = ttk.Label(
            result_frame,
            text="等待检测...",
            font=("微软雅黑", 14, "bold")
        )
        self.direction_label.pack(anchor="w", pady=(0, 5))
        
        self.confidence_label = ttk.Label(result_frame, text="")
        self.confidence_label.pack(anchor="w", pady=(0, 5))
        
        self.quality_label = ttk.Label(result_frame, text="")
        self.quality_label.pack(anchor="w", pady=(0, 5))
        
        self.time_label = ttk.Label(result_frame, text="")
        self.time_label.pack(anchor="w")
    
    def _create_display_panel(self, parent):
        """创建显示面板"""
        display_frame = ttk.Frame(parent)
        display_frame.pack(side="right", fill="both", expand=True)
        
        # 图像显示区域
        images_frame = ttk.Frame(display_frame)
        images_frame.pack(fill="both", expand=True)
        
        # 原图显示
        original_frame = ttk.LabelFrame(images_frame, text="原始图像", padding=5)
        original_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.original_canvas = tk.Canvas(original_frame, bg="#f0f0f0")
        self.original_canvas.pack(fill="both", expand=True)
        
        # 结果图显示
        result_frame = ttk.LabelFrame(images_frame, text="检测结果", padding=5)
        result_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        self.result_canvas = tk.Canvas(result_frame, bg="#f0f0f0")
        self.result_canvas.pack(fill="both", expand=True)
        
        # 统计信息
        stats_frame = ttk.LabelFrame(display_frame, text="统计信息", padding=10)
        stats_frame.pack(fill="x", pady=(10, 0))
        
        self._create_stats_display(stats_frame)
    
    def _create_stats_display(self, parent):
        """创建统计信息显示"""
        stats_grid = ttk.Frame(parent)
        stats_grid.pack(fill="x")
        
        ttk.Label(stats_grid, text="处理次数:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.process_count_label = ttk.Label(stats_grid, text="0")
        self.process_count_label.grid(row=0, column=1, sticky="w", padx=(0, 30))
        
        ttk.Label(stats_grid, text="平均时间:").grid(row=0, column=2, sticky="w", padx=(0, 10))
        self.avg_time_label = ttk.Label(stats_grid, text="0.00s")
        self.avg_time_label.grid(row=0, column=3, sticky="w")
    
    def _create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent, relief="sunken", borderwidth=1)
        status_frame.pack(fill="x", pady=(10, 0))
        
        self.progress_bar = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side="right", padx=5, pady=5)
    
    def _select_image(self):
        """选择图片"""
        if self.is_processing:
            return
        
        file_path = filedialog.askopenfilename(
            title="选择道路图片",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self._load_image(file_path)
    
    def _load_image(self, file_path: str):
        """加载图像"""
        try:
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
            
            processing_time = time.time() - start_time
            
            # 在主线程中更新UI
            self.root.after(0, self._update_results, 
                          road_info, lane_info, processing_time)
            
            # 更新统计信息
            self.processing_times.append(processing_time)
            self.average_time = np.mean(self.processing_times) if self.processing_times else 0
            
            # 记录处理历史
            self.processing_history.append({
                'file': file_path,
                'time': processing_time
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
    
    def _update_results(self, road_info: Dict, lane_info: Dict, processing_time: float):
        """更新结果"""
        try:
            # 显示图像
            self._display_image(self.current_image, self.original_canvas)
            
            # 创建可视化结果
            visualization = self._create_visualization(road_info, lane_info)
            self._display_image(visualization, self.result_canvas)
            
            # 更新信息
            quality = lane_info.get('detection_quality', 0.0)
            confidence = road_info.get('confidence', 0.0)
            
            # 判断方向（简化版）
            direction = self._determine_direction(road_info, lane_info)
            
            self.direction_label.config(text=f"方向: {direction}")
            
            # 设置置信度颜色
            if confidence > 0.7:
                color = "green"
            elif confidence > 0.4:
                color = "orange"
            else:
                color = "red"
            
            self.confidence_label.config(text=f"置信度: {confidence:.1%}", foreground=color)
            self.quality_label.config(text=f"检测质量: {quality:.1%}")
            self.time_label.config(text=f"处理时间: {processing_time:.3f}秒")
            
            # 更新统计信息
            self.process_count_label.config(text=str(len(self.processing_history)))
            self.avg_time_label.config(text=f"{self.average_time:.3f}s")
            
            self.status_var.set(f"分析完成 - {direction}")
            
            print(f"处理完成: 方向={direction}, 置信度={confidence:.1%}, 耗时={processing_time:.3f}s")
            
        except Exception as e:
            print(f"更新结果失败: {e}")
            self.status_var.set("更新结果失败")
    
    def _create_visualization(self, road_info: Dict, lane_info: Dict) -> np.ndarray:
        """创建可视化图像"""
        if self.current_image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 创建副本
        visualization = self.current_image.copy()
        
        # 绘制道路区域
        if 'contour' in road_info['features']:
            contour = road_info['features']['contour']
            cv2.drawContours(visualization, [contour], -1, (0, 180, 0), -1)
            cv2.drawContours(visualization, [contour], -1, (0, 255, 255), 2)
        
        # 绘制车道线
        if lane_info['left_lane']:
            points = lane_info['left_lane']['points']
            if len(points) == 2:
                cv2.line(visualization, points[0], points[1], (255, 100, 100), 4)
        
        if lane_info['right_lane']:
            points = lane_info['right_lane']['points']
            if len(points) == 2:
                cv2.line(visualization, points[0], points[1], (100, 100, 255), 4)
        
        # 绘制路径预测
        if lane_info['future_path']:
            path_points = lane_info['future_path']['center_path']
            for i in range(len(path_points) - 1):
                cv2.line(visualization, path_points[i], path_points[i + 1], 
                        (255, 0, 255), 3)
        
        return visualization
    
    def _determine_direction(self, road_info: Dict, lane_info: Dict) -> str:
        """判断方向"""
        if lane_info['left_lane'] and lane_info['right_lane']:
            left_func = lane_info['left_lane']['func']
            right_func = lane_info['right_lane']['func']
            
            # 计算底部和顶部的中心点
            height = self.current_image.shape[0]
            y_bottom = height
            y_top = int(height * 0.4)
            
            bottom_center = (left_func(y_bottom) + right_func(y_bottom)) / 2
            top_center = (left_func(y_top) + right_func(y_top)) / 2
            
            # 判断方向
            deviation = (top_center - bottom_center) / self.current_image.shape[1]
            
            if abs(deviation) < 0.05:
                return "直行"
            elif deviation > 0:
                return "右转"
            else:
                return "左转"
        
        return "未知"
    
    def _display_image(self, image: np.ndarray, canvas: tk.Canvas):
        """显示图像"""
        try:
            canvas.delete("all")
            
            if image is None or image.size == 0:
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
            
            # 缩放图像
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
    
    def _redetect(self):
        """重新检测"""
        if self.current_image_path and not self.is_processing:
            self._process_image(self.current_image_path)
    
    def _on_parameter_change(self, value):
        """参数变化回调"""
        # 更新配置
        sensitivity = self.sensitivity_var.get()
        prediction = self.prediction_var.get()
        
        self.config.canny_threshold1 = int(50 * (0.5 + sensitivity * 0.5))
        self.config.canny_threshold2 = int(150 * (0.5 + sensitivity * 0.5))
        self.config.hough_threshold = int(30 * (1.5 - sensitivity * 0.5))
        self.config.prediction_distance = prediction
        
        # 如果已有图像，重新检测
        if self.current_image_path and not self.is_processing:
            self._redetect()
    
    def _show_error(self, error_msg: str):
        """显示错误"""
        messagebox.showerror("错误", f"处理失败: {error_msg}")
        self.status_var.set("处理失败")

def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = AdvancedLaneDetectionApp(root)
        root.mainloop()
        
    except Exception as e:
        print(f"应用程序启动失败: {e}")
        messagebox.showerror("错误", f"应用程序启动失败: {str(e)}")

if __name__ == "__main__":
    main()