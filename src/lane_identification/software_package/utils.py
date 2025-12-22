"""
工具函数模块 - 提供通用函数和工具类
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Optional, Dict, Any
from collections import deque
from dataclasses import dataclass
from enum import Enum


class Timer:
    """计时器类"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = 0
        self.end_time = 0
        self.total_time = 0
        self.call_count = 0
        
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self
        
    def stop(self):
        """停止计时并返回耗时"""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.total_time += elapsed
        self.call_count += 1
        return elapsed
        
    def avg_time(self) -> float:
        """计算平均耗时"""
        return self.total_time / self.call_count if self.call_count > 0 else 0
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_samples: int = 100):
        self.metrics: Dict[str, deque] = {}
        self.max_samples = max_samples
        self.timers: Dict[str, Timer] = {}
        
    def start_timer(self, name: str) -> Timer:
        """开始计时器"""
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name].start()
    
    def stop_timer(self, name: str) -> float:
        """停止计时器并返回耗时"""
        if name in self.timers:
            elapsed = self.timers[name].stop()
            self.record_metric(f"{name}_time", elapsed)
            return elapsed
        return 0.0
    
    def record_metric(self, metric_name: str, value: float):
        """记录性能指标"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=self.max_samples)
        self.metrics[metric_name].append(value)
        
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """获取统计信息"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {'avg': 0, 'min': 0, 'max': 0, 'std': 0, 'p95': 0}
            
        values = list(self.metrics[metric_name])
        avg_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)
        std_val = np.std(values)
        p95_val = np.percentile(values, 95) if len(values) >= 2 else avg_val
        
        return {
            'avg': avg_val,
            'min': min_val,
            'max': max_val,
            'std': std_val,
            'p95': p95_val,
            'count': len(values)
        }
        
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取性能摘要"""
        summary = {}
        for metric in self.metrics:
            summary[metric] = self.get_statistics(metric)
        return summary
    
    def reset(self):
        """重置监控器"""
        self.metrics.clear()
        self.timers.clear()


def safe_resize(image: np.ndarray, target_size: Tuple[int, int], 
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """安全的图像缩放"""
    if image is None:
        return None
        
    if image.shape[:2] == target_size:
        return image.copy()
    
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 如果尺寸不匹配，填充到目标尺寸
        if new_w != target_w or new_h != target_h:
            padded = np.zeros((target_h, target_w, 3) if len(resized.shape) == 3 else (target_h, target_w), 
                            dtype=resized.dtype)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            if len(resized.shape) == 3:
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized
            else:
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
            return padded
        return resized
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def calculate_fps(frame_times: List[float], window: int = 10) -> float:
    """计算FPS"""
    if not frame_times or len(frame_times) < 2:
        return 0.0
    
    recent_times = frame_times[-window:]
    avg_time = np.mean(recent_times)
    return 1.0 / avg_time if avg_time > 0 else 0.0


def create_transparent_overlay(base_image: np.ndarray, 
                               overlay_image: np.ndarray,
                               alpha: float = 0.5,
                               position: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """创建透明覆盖层"""
    if base_image is None or overlay_image is None:
        return base_image
        
    # 创建副本
    result = base_image.copy()
    
    # 获取覆盖层尺寸和位置
    oh, ow = overlay_image.shape[:2]
    bh, bw = base_image.shape[:2]
    
    x, y = position
    x_end = min(x + ow, bw)
    y_end = min(y + oh, bh)
    
    # 计算实际覆盖区域
    if x_end <= x or y_end <= y:
        return result
    
    overlay_region = overlay_image[0:y_end-y, 0:x_end-x]
    base_region = base_image[y:y_end, x:x_end]
    
    # 混合图像
    if len(base_region.shape) == 3 and len(overlay_region.shape) == 3:
        blended = cv2.addWeighted(base_region, 1-alpha, overlay_region, alpha, 0)
    else:
        # 灰度图像处理
        if len(base_region.shape) == 2:
            base_region = cv2.cvtColor(base_region, cv2.COLOR_GRAY2BGR)
        if len(overlay_region.shape) == 2:
            overlay_region = cv2.cvtColor(overlay_region, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(base_region, 1-alpha, overlay_region, alpha, 0)
    
    result[y:y_end, x:x_end] = blended
    return result


def normalize_vector(v: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """向量归一化"""
    norm = np.linalg.norm(v)
    return v / (norm + epsilon)


def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量之间的角度（度）"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    dot_product = np.dot(v1, v2)
    cos_angle = dot_product / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.arccos(cos_angle) * 180 / np.pi


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """计算两个矩形框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def adaptive_threshold(image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """自适应阈值处理"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c
    )


def smooth_polynomial(coeffs_history: List[np.ndarray], smoothing_factor: float = 0.7) -> np.ndarray:
    """平滑多项式系数"""
    if not coeffs_history:
        return np.array([])
    
    weights = np.array([smoothing_factor ** i for i in range(len(coeffs_history))])
    weights = weights / weights.sum()
    
    smoothed = np.zeros_like(coeffs_history[0])
    for i, coeffs in enumerate(coeffs_history):
        smoothed += coeffs * weights[len(coeffs_history) - i - 1]
    
    return smoothed


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """创建高斯核"""
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    return kernel / kernel.sum()


def draw_text_with_background(image: np.ndarray, text: str, position: Tuple[int, int],
                              font_scale: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255),
                              bg_color: Tuple[int, int, int] = (0, 0, 0, 128), 
                              thickness: int = 2, padding: int = 5) -> np.ndarray:
    """在图像上绘制带背景的文字"""
    if image is None:
        return None
        
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 获取文字尺寸
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # 计算背景矩形位置
    x, y = position
    bg_top_left = (x - padding, y - text_height - padding)
    bg_bottom_right = (x + text_width + padding, y + padding)
    
    # 绘制背景
    cv2.rectangle(result, bg_top_left, bg_bottom_right, bg_color, -1)
    
    # 绘制文字
    cv2.putText(result, text, (x, y - baseline), font, font_scale, color, thickness)
    
    return result


def calculate_histogram_similarity(hist1: np.ndarray, hist2: np.ndarray, 
                                  method: int = cv2.HISTCMP_CORREL) -> float:
    """计算直方图相似度"""
    if hist1 is None or hist2 is None:
        return 0.0
    
    # 归一化直方图
    hist1_norm = cv2.normalize(hist1, hist1).flatten()
    hist2_norm = cv2.normalize(hist2, hist2).flatten()
    
    return cv2.compareHist(hist1_norm, hist2_norm, method)


@dataclass
class KalmanFilter1D:
    """一维卡尔曼滤波器"""
    process_variance: float = 1e-5
    measurement_variance: float = 1e-1
    estimate: float = 0.0
    error_covariance: float = 1.0
    
    def update(self, measurement: float) -> float:
        """更新滤波器"""
        # 预测
        prediction = self.estimate
        error_covariance_pred = self.error_covariance + self.process_variance
        
        # 更新
        kalman_gain = error_covariance_pred / (error_covariance_pred + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error_covariance = (1 - kalman_gain) * error_covariance_pred
        
        return self.estimate