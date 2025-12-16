import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Deque
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置管理 ====================
@dataclass
class AppConfig:
    """应用配置参数"""
    # 性能参数
    max_image_size: Tuple[int, int] = (1200, 800)
    cache_size: int = 5
    
    # 图像处理参数
    adaptive_clip_limit: float = 2.0
    adaptive_grid_size: Tuple[int, int] = (8, 8)
    
    # 检测参数
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    hough_threshold: int = 30
    hough_min_length: int = 20
    hough_max_gap: int = 50
    min_contour_area: float = 0.01
    
    # 方向分析参数
    deviation_threshold: float = 0.15
    
    # 路径预测参数
    prediction_steps: int = 8
    prediction_distance: float = 0.8

# ==================== 内存池管理器 ====================
class MemoryPool:
    """内存池管理器，重用数组减少内存分配"""
    
    def __init__(self):
        self.pool = {}
        self.max_pool_size = 10
    
    def get_array(self, shape: Tuple[int, ...], dtype=np.uint8):
        """从内存池获取或创建数组"""
        key = (shape, dtype)
        
        if key in self.pool and self.pool[key]:
            return self.pool[key].pop()
        
        return np.zeros(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray):
        """将数组返回到内存池"""
        if array is None:
            return
        
        key = (array.shape, array.dtype)
        
        if key not in self.pool:
            self.pool[key] = []
        
        # 清理数组内容
        array.fill(0)
        
        # 限制内存池大小
        if len(self.pool[key]) < self.max_pool_size:
            self.pool[key].append(array)
    
    def clear(self):
        """清空内存池"""
        self.pool.clear()

# ==================== 图像处理优化 ====================
class OptimizedImageProcessor:
    """进一步优化的图像处理器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.memory_pool = MemoryPool()
        self._cache = {}
        self._cache_order = deque(maxlen=config.cache_size)
        
        # 预计算的ROI模板
        self._roi_templates = {}
        
        # 预计算的CLAHE对象
        self.clahe_yuv = cv2.createCLAHE(
            clipLimit=config.adaptive_clip_limit,
            tileGridSize=config.adaptive_grid_size
        )
        self.clahe_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def load_and_preprocess(self, image_path: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """超优化版本：使用内存池和模板重用"""
        try:
            # 缓存检查
            if image_path in self._cache:
                return self._cache[image_path]
            
            # 快速加载图像
            image = self._fast_load_image(image_path)
            if image is None:
                return None
            
            # 智能调整尺寸
            image = self._resize_with_pool(image)
            
            # 并行预处理
            enhanced, roi_info = self._parallel_preprocess_fast(image)
            
            # 更新缓存
            self._update_cache(image_path, (enhanced, roi_info))
            
            return enhanced, roi_info
            
        except Exception as e:
            print(f"图像处理失败: {e}")
            return None
    
    def _fast_load_image(self, image_path: str) -> Optional[np.ndarray]:
        """快速图像加载"""
        try:
            # 检查文件大小
            if os.path.getsize(image_path) > 50 * 1024 * 1024:
                return None
            
            # 使用内存映射方式读取
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                return None
            
            # 确保正确的数据类型
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            return image
            
        except Exception as e:
            print(f"快速加载失败: {e}")
            return None
    
    def _resize_with_pool(self, image: np.ndarray) -> np.ndarray:
        """使用内存池的调整尺寸"""
        height, width = image.shape[:2]
        max_w, max_h = self.config.max_image_size
        
        # 无需调整大小
        if width <= max_w and height <= max_h:
            return image.copy()
        
        # 计算缩放比例
        scale = min(max_w / width, max_h / height)
        new_size = (int(width * scale), int(height * scale))
        
        # 使用内存池分配结果数组
        if scale < 1.0:
            # 缩小使用区域插值
            resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        else:
            # 放大使用线性插值
            resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def _parallel_preprocess_fast(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """并行预处理快速版"""
        # 并行执行增强和ROI计算
        with ThreadPoolExecutor(max_workers=2) as executor:
            enhanced_future = executor.submit(self._enhance_image_fast, image)
            roi_future = executor.submit(self._get_roi_template, image.shape)
            
            enhanced = enhanced_future.result()
            roi_info = roi_future.result()
        
        return enhanced, roi_info
    
    def _enhance_image_fast(self, image: np.ndarray) -> np.ndarray:
        """快速图像增强"""
        # 对于小图像，使用简化增强
        if image.shape[0] < 300 or image.shape[1] < 300:
            # 转换为灰度并增强
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enhanced_gray = self.clahe_gray.apply(gray)
            enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            return enhanced
        
        # 对于大图像，使用完整增强
        # 转换为YUV
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0]
        
        # CLAHE增强
        yuv[:, :, 0] = self.clahe_yuv.apply(y_channel)
        
        # 转换回BGR
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # 选择性去噪
        if self._estimate_noise_fast(y_channel) > 25:
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def _estimate_noise_fast(self, image: np.ndarray) -> float:
        """超快速噪声估计"""
        # 使用下采样加速计算
        if image.shape[0] > 100 and image.shape[1] > 100:
            # 下采样到100x100
            small = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
            laplacian = cv2.Laplacian(small, cv2.CV_64F)
            return float(np.std(laplacian))
        
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return float(np.std(laplacian))
    
    def _get_roi_template(self, shape: Tuple[int, int]) -> Dict:
        """获取或创建ROI模板"""
        key = shape[:2]
        
        if key in self._roi_templates:
            return self._roi_templates[key]
        
        # 创建新的ROI模板
        height, width = key
        roi_info = self._create_roi_template(height, width)
        self._roi_templates[key] = roi_info
        
        return roi_info
    
    def _create_roi_template(self, height: int, width: int) -> Dict:
        """创建ROI模板"""
        roi_top = int(height * 0.35)
        roi_bottom = int(height * 0.92)
        roi_width = int(width * 0.85)
        
        vertices = np.array([[
            ((width - roi_width) // 2, roi_bottom),
            ((width - roi_width) // 2 + int(roi_width * 0.3), roi_top),
            ((width - roi_width) // 2 + int(roi_width * 0.7), roi_top),
            ((width + roi_width) // 2, roi_bottom)
        ]], dtype=np.int32)
        
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
            # 移动到最近使用
            self._cache_order.remove(key)
        elif len(self._cache) >= self.config.cache_size:
            # 移除最旧的
            oldest = self._cache_order.popleft()
            del self._cache[oldest]
        
        self._cache[key] = value
        self._cache_order.append(key)

# ==================== 道路检测器优化 ====================
class OptimizedRoadDetector:
    """超优化道路检测器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.previous_results = deque(maxlen=2)  # 减少历史记录长度
        
        # 预计算检测方法
        self._detection_methods = [
            ('color', self._detect_color_fast),
            ('edges', self._detect_edges_fast)
        ]
        
        # 预分配内存
        self._temp_arrays = {}
    
    def detect(self, image: np.ndarray, roi_info: Dict) -> Dict[str, Any]:
        """超快速检测"""
        try:
            # 提取ROI区域
            roi_region = cv2.bitwise_and(image, image, mask=roi_info['mask'])
            
            # 单尺度检测（足够准确）
            results = []
            for method_name, method_func in self._detection_methods:
                result = method_func(roi_region)
                if result['confidence'] > 0.2:  # 降低阈值
                    results.append(result)
            
            if not results:
                return self._create_empty_result()
            
            # 快速融合结果
            fused = self._fuse_results_fast(results)
            
            # 提取基本特征
            features = self._extract_features_fast(fused, image.shape)
            
            # 时间平滑
            if self.previous_results:
                fused = self._apply_temporal_smoothing(fused)
            
            # 更新历史
            self.previous_results.append(fused)
            
            return {
                'features': features,
                'confidence': fused['confidence'],
                'num_methods': len(results)
            }
            
        except Exception as e:
            print(f"道路检测失败: {e}")
            return self._create_empty_result()
    
    def _detect_color_fast(self, roi_region: np.ndarray) -> Dict[str, Any]:
        """快速颜色检测"""
        # 转换为灰度
        gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值
        mean = np.mean(gray)
        std = np.std(gray)
        
        lower = max(0, int(mean - std * 1.2))
        upper = min(255, int(mean + std * 1.2))
        
        # 创建掩码
        mask = cv2.inRange(gray, lower, upper)
        
        # 快速形态学
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 置信度计算
        area_ratio = np.count_nonzero(mask) / mask.size
        confidence = min(1.0, area_ratio * 1.5)
        
        return {'mask': mask, 'confidence': confidence, 'method': 'color'}
    
    def _detect_edges_fast(self, roi_region: np.ndarray) -> Dict[str, Any]:
        """快速边缘检测"""
        gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
        
        # 自适应Canny
        median = np.median(gray)
        lower = int(max(0, 0.5 * median))
        upper = int(min(255, 1.5 * median))
        
        edges = cv2.Canny(gray, lower, upper)
        
        # 填充边缘区域
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(edges)
        
        # 只处理大轮廓
        min_area = gray.shape[0] * gray.shape[1] * 0.002
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # 置信度计算
        edge_density = np.count_nonzero(edges) / edges.size
        confidence = min(1.0, 1.0 - edge_density * 1.5)
        
        return {'mask': mask, 'confidence': confidence, 'method': 'edges'}
    
    def _fuse_results_fast(self, results: List[Dict]) -> Dict[str, Any]:
        """快速结果融合"""
        if len(results) == 1:
            return results[0]
        
        # 简单加权平均
        weights = np.array([r['confidence'] for r in results])
        weights = weights / np.sum(weights)
        
        # 融合掩码
        fused_mask = np.zeros_like(results[0]['mask'], dtype=np.float32)
        for r, w in zip(results, weights):
            fused_mask += r['mask'].astype(np.float32) * w
        
        # 二值化
        fused_binary = (fused_mask > 127).astype(np.uint8) * 255
        
        # 形态学优化
        kernel = np.ones((3, 3), np.uint8)
        fused_binary = cv2.morphologyEx(fused_binary, cv2.MORPH_CLOSE, kernel)
        
        return {
            'mask': fused_binary,
            'confidence': float(np.mean([r['confidence'] for r in results]))
        }
    
    def _extract_features_fast(self, detection_result: Dict, 
                               image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """快速特征提取"""
        mask = detection_result['mask']
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {}
        
        # 最大轮廓
        main_contour = max(contours, key=cv2.contourArea)
        
        # 简单轮廓近似
        epsilon = 0.02 * cv2.arcLength(main_contour, True)
        approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # 计算基本特征
        rect = cv2.boundingRect(approx_contour)
        area = cv2.contourArea(approx_contour)
        
        # 质心
        M = cv2.moments(approx_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = rect[0] + rect[2] // 2, rect[1] + rect[3] // 2
        
        return {
            'contour': approx_contour,
            'centroid': (cx, cy),
            'area': area,
            'bounding_rect': rect
        }
    
    def _apply_temporal_smoothing(self, current_result: Dict[str, Any]) -> Dict[str, Any]:
        """时间平滑"""
        alpha = 0.6  # 较弱的平滑
        
        # 对特征进行指数移动平均
        if self.previous_results:
            prev_result = self.previous_results[-1]
            current_result['confidence'] = (
                alpha * current_result['confidence'] + 
                (1 - alpha) * prev_result['confidence']
            )
        
        return current_result
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """空结果"""
        return {
            'features': {},
            'confidence': 0.0,
            'num_methods': 0
        }

# ==================== 车道线检测器优化 ====================
class OptimizedLaneDetector:
    """超优化车道线检测器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.lane_history = deque(maxlen=3)  # 减少历史长度
        
        # 预计算参数
        self._angles = np.linspace(-np.pi/3, np.pi/3, 7)  # 减少角度数量
        
        # 预分配内存
        self._temp_buffers = {}
    
    def detect(self, image: np.ndarray, roi_mask: np.ndarray) -> Dict[str, Any]:
        """快速车道线检测"""
        try:
            # 快速预处理
            processed = self._preprocess_fast(image, roi_mask)
            
            # 使用单一检测方法（Canny+霍夫）
            lines = self._detect_lines_fast(processed)
            
            if len(lines) < 2:
                return self._create_empty_lane_result()
            
            # 分类车道线
            left_lines, right_lines = self._classify_lines_fast(lines, image.shape[1])
            
            # 拟合车道线
            left_lane = self._fit_lane_fast(left_lines, image.shape)
            right_lane = self._fit_lane_fast(right_lines, image.shape)
            
            # 验证车道线
            if left_lane and right_lane:
                left_lane, right_lane = self._validate_lanes_fast(left_lane, right_lane, image.shape)
            
            # 计算检测质量
            quality = self._calculate_quality_fast(left_lane, right_lane)
            
            result = {
                'left_lane': left_lane,
                'right_lane': right_lane,
                'detection_quality': quality,
                'has_left': left_lane is not None,
                'has_right': right_lane is not None
            }
            
            # 时间平滑
            if self.lane_history:
                result = self._smooth_result(result)
            
            self.lane_history.append(result)
            
            return result
            
        except Exception as e:
            print(f"车道线检测失败: {e}")
            return self._create_empty_lane_result()
    
    def _preprocess_fast(self, image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """快速预处理"""
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用ROI
        gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
        
        # 快速增强
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def _detect_lines_fast(self, image: np.ndarray) -> List[List]:
        """快速直线检测"""
        # 自适应Canny
        median = np.median(image)
        lower = int(max(0, 0.6 * median))
        upper = int(min(255, 1.4 * median))
        
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
    
    def _classify_lines_fast(self, lines: List[List], image_width: int) -> Tuple[List, List]:
        """快速车道线分类"""
        left_lines, right_lines = [], []
        
        for line in lines:
            if len(line[0]) != 4:
                continue
                
            x1, y1, x2, y2 = line[0]
            
            # 跳过水平线
            if abs(y2 - y1) < 10:
                continue
            
            # 计算斜率
            if x2 == x1:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # 过滤标准
            if abs(slope) < 0.2:
                continue
            
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 15:
                continue
            
            # 分类
            midpoint_x = (x1 + x2) / 2
            if slope < 0:  # 左车道线
                if midpoint_x < image_width * 0.7:
                    left_lines.append((x1, y1, x2, y2, slope, length))
            else:  # 右车道线
                if midpoint_x > image_width * 0.3:
                    right_lines.append((x1, y1, x2, y2, slope, length))
        
        # 过滤异常值
        left_lines = self._filter_outliers_fast(left_lines)
        right_lines = self._filter_outliers_fast(right_lines)
        
        return left_lines, right_lines
    
    def _filter_outliers_fast(self, lines: List[Tuple]) -> List[Tuple]:
        """快速过滤异常值"""
        if len(lines) < 3:
            return lines
        
        # 计算斜率中值和标准差
        slopes = [line[4] for line in lines]
        median_slope = np.median(slopes)
        mad_slope = np.median(np.abs(slopes - median_slope))
        
        # 过滤离群值
        filtered = []
        for line in lines:
            if abs(line[4] - median_slope) < 2 * mad_slope:
                filtered.append(line)
        
        return filtered
    
    def _fit_lane_fast(self, lines: List[Tuple], image_shape: Tuple[int, ...]) -> Optional[Dict]:
        """快速车道线拟合"""
        if len(lines) < 2:
            return None
        
        # 收集所有点
        x_points, y_points = [], []
        for line in lines:
            x1, y1, x2, y2, _, _ = line
            x_points.extend([x1, x2])
            y_points.extend([y1, y2])
        
        # 线性拟合
        try:
            coeffs = np.polyfit(y_points, x_points, 1)
            poly_func = np.poly1d(coeffs)
        except:
            return None
        
        # 生成车道线点
        height, width = image_shape[:2]
        y_bottom = height
        y_top = int(height * 0.4)
        
        x_bottom = int(poly_func(y_bottom))
        x_top = int(poly_func(y_top))
        
        # 限制范围
        x_bottom = max(0, min(width, x_bottom))
        x_top = max(0, min(width, x_top))
        
        # 计算置信度
        confidence = min(len(lines) / 8.0, 1.0)
        
        return {
            'func': poly_func,
            'coeffs': coeffs.tolist(),
            'points': [(x_bottom, y_bottom), (x_top, y_top)],
            'confidence': confidence,
            'num_lines': len(lines)
        }
    
    def _validate_lanes_fast(self, left_lane: Dict, right_lane: Dict, 
                             image_shape: Tuple[int, ...]) -> Tuple[Dict, Dict]:
        """快速车道线验证"""
        height, width = image_shape[:2]
        
        # 检查车道宽度
        y_mid = height // 2
        left_x = left_lane['func'](y_mid)
        right_x = right_lane['func'](y_mid)
        lane_width = right_x - left_x
        
        min_width = width * 0.1
        max_width = width * 0.7
        
        if lane_width < min_width or lane_width > max_width:
            left_lane['confidence'] *= 0.8
            right_lane['confidence'] *= 0.8
        
        # 检查交叉
        if left_lane['points'][0][0] > right_lane['points'][0][0]:
            left_lane['confidence'] *= 0.7
            right_lane['confidence'] *= 0.7
        
        return left_lane, right_lane
    
    def _calculate_quality_fast(self, left_lane: Optional[Dict], right_lane: Optional[Dict]) -> float:
        """快速质量计算"""
        quality = 0.0
        
        if left_lane is not None:
            quality += left_lane['confidence'] * 0.4
        
        if right_lane is not None:
            quality += right_lane['confidence'] * 0.4
        
        if left_lane is not None and right_lane is not None:
            quality += 0.2
        
        return min(quality, 1.0)
    
    def _smooth_result(self, current_result: Dict[str, Any]) -> Dict[str, Any]:
        """结果平滑"""
        if not self.lane_history:
            return current_result
        
        alpha = 0.5
        
        # 平滑左车道线
        if current_result['left_lane'] and self.lane_history[-1]['left_lane']:
            prev_coeffs = np.array(self.lane_history[-1]['left_lane']['coeffs'])
            curr_coeffs = np.array(current_result['left_lane']['coeffs'])
            
            if len(prev_coeffs) == len(curr_coeffs):
                smoothed_coeffs = alpha * curr_coeffs + (1 - alpha) * prev_coeffs
                current_result['left_lane']['coeffs'] = smoothed_coeffs.tolist()
                current_result['left_lane']['func'] = np.poly1d(smoothed_coeffs)
        
        # 平滑右车道线
        if current_result['right_lane'] and self.lane_history[-1]['right_lane']:
            prev_coeffs = np.array(self.lane_history[-1]['right_lane']['coeffs'])
            curr_coeffs = np.array(current_result['right_lane']['coeffs'])
            
            if len(prev_coeffs) == len(curr_coeffs):
                smoothed_coeffs = alpha * curr_coeffs + (1 - alpha) * prev_coeffs
                current_result['right_lane']['coeffs'] = smoothed_coeffs.tolist()
                current_result['right_lane']['func'] = np.poly1d(smoothed_coeffs)
        
        return current_result
    
    def _create_empty_lane_result(self) -> Dict[str, Any]:
        """空车道线结果"""
        return {
            'left_lane': None,
            'right_lane': None,
            'detection_quality': 0.0,
            'has_left': False,
            'has_right': False
        }

# ==================== 方向分析器优化 ====================
class OptimizedDirectionAnalyzer:
    """简化但高效的方向分析器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.history = deque(maxlen=5)
    
    def analyze(self, road_features: Dict, lane_info: Dict) -> Dict[str, Any]:
        """快速方向分析"""
        try:
            # 提取关键特征
            direction = self._determine_direction_simple(road_features, lane_info)
            
            # 计算置信度
            confidence = self._calculate_confidence_simple(road_features, lane_info)
            
            # 历史平滑
            if self.history:
                direction, confidence = self._apply_history_smoothing(direction, confidence)
            
            result = {
                'direction': direction,
                'confidence': confidence,
                'is_straight': direction == '直行',
                'is_left': direction == '左转',
                'is_right': direction == '右转'
            }
            
            self.history.append(result)
            
            return result
            
        except Exception as e:
            print(f"方向分析失败: {e}")
            return {
                'direction': '未知',
                'confidence': 0.0,
                'is_straight': False,
                'is_left': False,
                'is_right': False
            }
    
    def _determine_direction_simple(self, road_features: Dict, lane_info: Dict) -> str:
        """简单方向判断"""
        # 检查是否有车道线
        has_left = lane_info.get('has_left', False)
        has_right = lane_info.get('has_right', False)
        
        if not has_left or not has_right:
            return '未知'
        
        left_lane = lane_info['left_lane']
        right_lane = lane_info['right_lane']
        
        if left_lane is None or right_lane is None:
            return '未知'
        
        # 计算车道线收敛
        height = 600  # 假设图像高度
        
        # 底部和顶部位置
        y_bottom = height
        y_top = int(height * 0.4)
        
        # 计算车道宽度
        bottom_width = right_lane['func'](y_bottom) - left_lane['func'](y_bottom)
        top_width = right_lane['func'](y_top) - left_lane['func'](y_top)
        
        # 计算收敛比
        if bottom_width > 0:
            convergence_ratio = top_width / bottom_width
            
            # 判断方向
            if convergence_ratio > 0.9 and convergence_ratio < 1.1:
                return '直行'
            elif convergence_ratio < 0.9:
                return '左转'
            else:
                return '右转'
        
        # 基于质心判断
        if 'centroid' in road_features:
            centroid_x = road_features['centroid'][0]
            image_width = 800  # 假设图像宽度
            
            # 计算偏差
            deviation = (centroid_x - image_width/2) / (image_width/2)
            
            if abs(deviation) < 0.1:
                return '直行'
            elif deviation > 0:
                return '右转'
            else:
                return '左转'
        
        return '未知'
    
    def _calculate_confidence_simple(self, road_features: Dict, lane_info: Dict) -> float:
        """简单置信度计算"""
        confidence_factors = []
        
        # 1. 道路检测置信度
        if 'confidence' in road_features:
            confidence_factors.append(road_features['confidence'] * 0.3)
        
        # 2. 车道线质量
        lane_quality = lane_info.get('detection_quality', 0.0)
        confidence_factors.append(lane_quality * 0.4)
        
        # 3. 历史一致性
        if self.history:
            recent_directions = [h['direction'] for h in self.history[-3:]]
            if recent_directions:
                from collections import Counter
                most_common = Counter(recent_directions).most_common(1)[0]
                consistency = most_common[1] / len(recent_directions)
                confidence_factors.append(consistency * 0.3)
        
        # 综合置信度
        confidence = sum(confidence_factors) if confidence_factors else 0.0
        return min(confidence, 1.0)
    
    def _apply_history_smoothing(self, direction: str, confidence: float) -> Tuple[str, float]:
        """历史平滑"""
        if not self.history:
            return direction, confidence
        
        # 检查最近的历史
        recent_history = list(self.history)[-2:]
        
        if recent_history:
            # 计算方向频率
            direction_counts = {}
            for result in recent_history:
                d = result['direction']
                direction_counts[d] = direction_counts.get(d, 0) + 1
            
            # 如果有明显的主要方向
            if direction_counts:
                most_common = max(direction_counts.items(), key=lambda x: x[1])
                frequency = most_common[1] / len(recent_history)
                
                if frequency > 0.5 and most_common[0] != direction and confidence < 0.6:
                    # 平滑过渡
                    return most_common[0], confidence * 0.8
        
        return direction, confidence

# ==================== 可视化引擎优化 ====================
class OptimizedVisualizer:
    """高效可视化引擎"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.colors = {
            'road_area': (0, 180, 0),
            'road_boundary': (0, 255, 255),
            'left_lane': (255, 100, 100),
            'right_lane': (100, 100, 255),
            'center_path': (255, 0, 255),
            'text_good': (0, 255, 0),
            'text_warning': (255, 165, 0),
            'text_error': (255, 0, 0)
        }
        
        # 预分配的叠加层
        self._overlay_cache = {}
    
    def create_visualization(self, image: np.ndarray, road_info: Dict,
                           lane_info: Dict, direction_info: Dict) -> np.ndarray:
        """快速可视化"""
        try:
            # 创建副本
            visualization = image.copy()
            
            # 并行绘制
            with ThreadPoolExecutor(max_workers=2) as executor:
                # 绘制道路区域
                if 'contour' in road_info['features']:
                    executor.submit(self._draw_road_fast, visualization, road_info).result()
                
                # 绘制车道线
                executor.submit(self._draw_lanes_fast, visualization, lane_info).result()
            
            # 绘制信息面板
            self._draw_info_fast(visualization, direction_info, lane_info)
            
            return visualization
            
        except Exception as e:
            print(f"可视化失败: {e}")
            return image
    
    def _draw_road_fast(self, image: np.ndarray, road_info: Dict):
        """快速绘制道路"""
        contour = road_info['features'].get('contour')
        if contour is None:
            return
        
        # 创建半透明覆盖层
        overlay = image.copy()
        cv2.drawContours(overlay, [contour], -1, self.colors['road_area'], -1)
        cv2.drawContours(overlay, [contour], -1, self.colors['road_boundary'], 2)
        
        # 混合
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
    
    def _draw_lanes_fast(self, image: np.ndarray, lane_info: Dict):
        """快速绘制车道线"""
        # 绘制左车道线
        if lane_info['left_lane']:
            points = lane_info['left_lane']['points']
            if len(points) == 2:
                cv2.line(image, points[0], points[1], self.colors['left_lane'], 3, cv2.LINE_AA)
        
        # 绘制右车道线
        if lane_info['right_lane']:
            points = lane_info['right_lane']['points']
            if len(points) == 2:
                cv2.line(image, points[0], points[1], self.colors['right_lane'], 3, cv2.LINE_AA)
        
        # 绘制中心线（如果有）
        if lane_info['left_lane'] and lane_info['right_lane']:
            left_func = lane_info['left_lane']['func']
            right_func = lane_info['right_lane']['func']
            
            height = image.shape[0]
            y_bottom = height
            y_top = int(height * 0.4)
            
            # 计算中心点
            bottom_center = int((left_func(y_bottom) + right_func(y_bottom)) / 2)
            top_center = int((left_func(y_top) + right_func(y_top)) / 2)
            
            cv2.line(image, (bottom_center, y_bottom), (top_center, y_top), 
                    self.colors['center_path'], 2, cv2.LINE_AA)
    
    def _draw_info_fast(self, image: np.ndarray, direction_info: Dict, lane_info: Dict):
        """快速绘制信息"""
        height, width = image.shape[:2]
        
        # 创建半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # 方向信息
        direction = direction_info['direction']
        confidence = direction_info['confidence']
        
        # 选择颜色
        if confidence > 0.7:
            color = self.colors['text_good']
        elif confidence > 0.4:
            color = self.colors['text_warning']
        else:
            color = self.colors['text_error']
        
        # 绘制方向文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"方向: {direction}", (20, 40), 
                   font, 1.0, color, 2)
        
        # 绘制置信度
        cv2.putText(image, f"置信度: {confidence:.1%}", (20, 70), 
                   font, 0.7, color, 1)
        
        # 绘制检测质量
        quality = lane_info.get('detection_quality', 0.0)
        cv2.putText(image, f"质量: {quality:.1%}", (width - 120, 40), 
                   font, 0.6, (200, 200, 200), 1)

# ==================== 主应用程序 ====================
class OptimizedLaneDetectionApp:
    """优化版主应用程序"""
    
    def __init__(self, root):
        self.root = root
        self._setup_window()
        
        # 初始化组件
        self.config = AppConfig()
        self.image_processor = OptimizedImageProcessor(self.config)
        self.road_detector = OptimizedRoadDetector(self.config)
        self.lane_detector = OptimizedLaneDetector(self.config)
        self.direction_analyzer = OptimizedDirectionAnalyzer(self.config)
        self.visualizer = OptimizedVisualizer(self.config)
        
        # 状态变量
        self.current_image = None
        self.current_image_path = None
        self.is_processing = False
        self.processing_count = 0
        self.total_processing_time = 0
        
        # 创建界面
        self._create_optimized_ui()
        
        print("🚗 超级优化版道路方向识别系统已启动")
    
    def _setup_window(self):
        """设置窗口"""
        self.root.title("🚗 超级优化版道路方向识别系统")
        self.root.geometry("1200x700")
        
        # 窗口居中
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width, window_height = 1200, 700
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    def _create_optimized_ui(self):
        """创建优化版UI"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # 标题
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(
            title_frame,
            text="超级优化版道路方向识别系统",
            font=("微软雅黑", 14, "bold")
        ).pack(side="left")
        
        # 内容区域
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
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10", width=250)
        control_frame.pack(side="left", fill="y", padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 文件操作
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Button(
            file_frame,
            text="📁 选择图片",
            command=self._select_image,
            width=18
        ).pack(pady=(0, 10))
        
        self.redetect_btn = ttk.Button(
            file_frame,
            text="🔄 重新检测",
            command=self._redetect,
            width=18,
            state="disabled"
        )
        self.redetect_btn.pack(pady=(0, 10))
        
        self.file_label = ttk.Label(file_frame, text="未选择图片", wraplength=220)
        self.file_label.pack()
        
        # 参数调节
        param_frame = ttk.LabelFrame(control_frame, text="参数调节", padding="10")
        param_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(param_frame, text="检测敏感度:").pack(anchor="w")
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        ttk.Scale(
            param_frame,
            from_=0.1,
            to=1.0,
            variable=self.sensitivity_var,
            orient="horizontal",
            command=self._on_param_change
        ).pack(fill="x", pady=(0, 10))
        
        ttk.Label(param_frame, text="预测距离:").pack(anchor="w")
        self.prediction_var = tk.DoubleVar(value=self.config.prediction_distance)
        ttk.Scale(
            param_frame,
            from_=0.3,
            to=0.9,
            variable=self.prediction_var,
            orient="horizontal",
            command=self._on_param_change
        ).pack(fill="x")
        
        # 结果展示
        result_frame = ttk.LabelFrame(control_frame, text="检测结果", padding="10")
        result_frame.pack(fill="x")
        
        self.direction_label = ttk.Label(
            result_frame,
            text="等待检测...",
            font=("微软雅黑", 12, "bold")
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
        
        # 图像显示
        images_frame = ttk.Frame(display_frame)
        images_frame.pack(fill="both", expand=True)
        
        # 原始图像
        original_frame = ttk.LabelFrame(images_frame, text="原始图像", padding="5")
        original_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.original_canvas = tk.Canvas(original_frame, bg="#f0f0f0")
        self.original_canvas.pack(fill="both", expand=True)
        
        # 结果图像
        result_frame = ttk.LabelFrame(images_frame, text="检测结果", padding="5")
        result_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        self.result_canvas = tk.Canvas(result_frame, bg="#f0f0f0")
        self.result_canvas.pack(fill="both", expand=True)
        
        # 统计信息
        stats_frame = ttk.LabelFrame(display_frame, text="性能统计", padding="10")
        stats_frame.pack(fill="x", pady=(10, 0))
        
        self._create_stats_panel(stats_frame)
    
    def _create_stats_panel(self, parent):
        """创建统计面板"""
        stats_grid = ttk.Frame(parent)
        stats_grid.pack(fill="x")
        
        ttk.Label(stats_grid, text="处理次数:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.count_label = ttk.Label(stats_grid, text="0")
        self.count_label.grid(row=0, column=1, sticky="w", padx=(0, 30))
        
        ttk.Label(stats_grid, text="平均时间:").grid(row=0, column=2, sticky="w", padx=(0, 10))
        self.avg_time_label = ttk.Label(stats_grid, text="0.00s")
        self.avg_time_label.grid(row=0, column=3, sticky="w")
        
        ttk.Label(stats_grid, text="缓存大小:").grid(row=0, column=4, sticky="w", padx=(40, 10))
        self.cache_label = ttk.Label(stats_grid, text="0")
        self.cache_label.grid(row=0, column=5, sticky="w")
    
    def _create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent, relief="sunken")
        status_frame.pack(fill="x", pady=(10, 0))
        
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(side="left", fill="x", expand=True, padx=5, pady=3)
        
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="right", padx=5, pady=3)
    
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
            self._process_image_async()
    
    def _process_image_async(self):
        """异步处理图像"""
        self.status_var.set("正在加载...")
        self.file_label.config(text=os.path.basename(self.current_image_path))
        self.redetect_btn.config(state="normal")
        
        # 启动处理线程
        thread = threading.Thread(target=self._process_image, daemon=True)
        thread.start()
    
    def _process_image(self):
        """处理图像"""
        start_time = time.time()
        
        try:
            self.is_processing = True
            self.root.after(0, self._update_ui_processing, True)
            
            # 1. 图像预处理
            result = self.image_processor.load_and_preprocess(self.current_image_path)
            if result is None:
                raise ValueError("无法处理图像")
            
            self.current_image, roi_info = result
            
            # 2. 并行检测
            with ThreadPoolExecutor(max_workers=2) as executor:
                road_future = executor.submit(self.road_detector.detect, self.current_image, roi_info)
                lane_future = executor.submit(self.lane_detector.detect, self.current_image, roi_info['mask'])
                
                road_info = road_future.result()
                lane_info = lane_future.result()
            
            # 3. 方向分析
            direction_info = self.direction_analyzer.analyze(road_info['features'], lane_info)
            
            # 4. 创建可视化
            visualization = self.visualizer.create_visualization(
                self.current_image, road_info, lane_info, direction_info
            )
            
            processing_time = time.time() - start_time
            
            # 更新统计
            self.processing_count += 1
            self.total_processing_time += processing_time
            
            # 在主线程更新UI
            self.root.after(0, self._update_results, 
                          visualization, direction_info, lane_info, processing_time)
            
        except Exception as e:
            print(f"处理失败: {e}")
            self.root.after(0, self._show_error, str(e))
        finally:
            self.is_processing = False
            self.root.after(0, self._update_ui_processing, False)
    
    def _update_ui_processing(self, is_processing: bool):
        """更新UI处理状态"""
        if is_processing:
            self.progress.start()
            self.status_var.set("正在分析...")
            self.redetect_btn.config(state="disabled")
        else:
            self.progress.stop()
            self.status_var.set("就绪")
            self.redetect_btn.config(state="normal")
    
    def _update_results(self, visualization: np.ndarray, direction_info: Dict,
                       lane_info: Dict, processing_time: float):
        """更新结果"""
        try:
            # 显示图像
            self._display_image(self.current_image, self.original_canvas)
            self._display_image(visualization, self.result_canvas)
            
            # 更新结果信息
            direction = direction_info['direction']
            confidence = direction_info['confidence']
            quality = lane_info.get('detection_quality', 0.0)
            
            self.direction_label.config(text=f"方向: {direction}")
            
            # 设置颜色
            if confidence > 0.7:
                color = "green"
            elif confidence > 0.4:
                color = "orange"
            else:
                color = "red"
            
            self.confidence_label.config(text=f"置信度: {confidence:.1%}", foreground=color)
            self.quality_label.config(text=f"检测质量: {quality:.1%}")
            self.time_label.config(text=f"处理时间: {processing_time:.3f}s")
            
            # 更新统计
            avg_time = self.total_processing_time / self.processing_count if self.processing_count > 0 else 0
            self.count_label.config(text=str(self.processing_count))
            self.avg_time_label.config(text=f"{avg_time:.3f}s")
            self.cache_label.config(text=str(len(self.image_processor._cache)))
            
            self.status_var.set(f"完成 - {direction}")
            
            print(f"处理完成: {direction}, 置信度: {confidence:.1%}, 耗时: {processing_time:.3f}s")
            
        except Exception as e:
            print(f"更新结果失败: {e}")
            self.status_var.set("更新失败")
    
    def _display_image(self, image: np.ndarray, canvas: tk.Canvas):
        """显示图像"""
        try:
            canvas.delete("all")
            
            if image is None or image.size == 0:
                return
            
            # 转换颜色
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # 获取Canvas尺寸
            canvas.update()
            width = canvas.winfo_width()
            height = canvas.winfo_height()
            
            if width <= 1 or height <= 1:
                width, height = 500, 350
            
            # 计算缩放
            img_width, img_height = pil_image.size
            scale = min(width / img_width, height / img_height)
            
            if scale < 1:
                new_size = (int(img_width * scale), int(img_height * scale))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 显示
            photo = ImageTk.PhotoImage(pil_image)
            x = (width - photo.width()) // 2
            y = (height - photo.height()) // 2
            
            canvas.create_image(x, y, anchor="nw", image=photo)
            canvas.image = photo
            
        except Exception as e:
            print(f"显示图像失败: {e}")
    
    def _redetect(self):
        """重新检测"""
        if self.current_image_path and not self.is_processing:
            self._process_image_async()
    
    def _on_param_change(self, value):
        """参数变化处理"""
        sensitivity = self.sensitivity_var.get()
        prediction = self.prediction_var.get()
        
        # 更新配置
        self.config.canny_threshold1 = int(40 + sensitivity * 40)
        self.config.canny_threshold2 = int(100 + sensitivity * 80)
        self.config.hough_threshold = int(20 + (1 - sensitivity) * 20)
        self.config.prediction_distance = prediction
        
        # 重新检测
        if self.current_image_path and not self.is_processing:
            self._redetect()
    
    def _show_error(self, error_msg: str):
        """显示错误"""
        messagebox.showerror("错误", f"处理失败: {error_msg}")
        self.status_var.set("失败")

# ==================== 主函数 ====================
def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = OptimizedLaneDetectionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"启动失败: {e}")
        messagebox.showerror("错误", f"启动失败: {e}")

if __name__ == "__main__":
    main()