"""
图像处理模块 - 负责图像加载、预处理和缓存
支持图像、视频和摄像头输入
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any
import os
from config import AppConfig

class SmartImageProcessor:
    """智能图像处理器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._cache = {}
        self._cache_order = deque(maxlen=config.cache_size)
    
    def load_and_preprocess(self, image_path: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """加载并预处理图像"""
        try:
            # 检查缓存
            if image_path in self._cache:
                self._cache_order.remove(image_path)
                self._cache_order.append(image_path)
                return self._cache[image_path]
            
            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"无法读取图像: {image_path}")
                return None
            
            # 智能调整尺寸
            processed = self._smart_resize(image)
            
            # 自适应预处理
            processed = self._adaptive_preprocessing(processed)
            
            # 计算ROI区域
            roi_info = self._calculate_roi(processed.shape)
            
            # 更新缓存
            result = (processed, roi_info)
            self._update_cache(image_path, result)
            
            return result
            
        except Exception as e:
            print(f"图像处理失败 {image_path}: {e}")
            return None
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """预处理视频帧"""
        try:
            # 智能调整尺寸
            processed = self._smart_resize(frame)
            
            # 自适应预处理
            processed = self._adaptive_preprocessing(processed)
            
            # 计算ROI区域
            roi_info = self._calculate_roi(processed.shape)
            
            return processed, roi_info
            
        except Exception as e:
            print(f"帧预处理失败: {e}")
            return frame, {}
    
    def _smart_resize(self, image: np.ndarray) -> np.ndarray:
        """智能调整图像尺寸"""
        height, width = image.shape[:2]
        max_w, max_h = self.config.max_image_size
        
        # 计算最佳缩放比例
        scale_w = max_w / width if width > max_w else 1.0
        scale_h = max_h / height if height > max_h else 1.0
        scale = min(scale_w, scale_h)
        
        if scale < 1.0:
            new_size = (int(width * scale), int(height * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        return image
    
    def _adaptive_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """自适应图像预处理"""
        # 转换为YUV颜色空间
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0]
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(
            clipLimit=self.config.adaptive_clip_limit,
            tileGridSize=self.config.adaptive_grid_size
        )
        yuv[:, :, 0] = clahe.apply(y_channel)
        
        # 转换回BGR
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # 自适应去噪
        noise_level = self._estimate_noise_level(y_channel)
        if noise_level > 30:
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """估计图像噪声水平"""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return float(np.std(laplacian))
    
    def _calculate_roi(self, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """计算ROI区域"""
        height, width = image_shape[:2]
        
        # 动态ROI计算
        roi_top = int(height * 0.35)
        roi_bottom = int(height * 0.92)
        roi_width = int(width * 0.85)
        
        vertices = np.array([[
            ((width - roi_width) // 2, roi_bottom),
            ((width - roi_width) // 2 + int(roi_width * 0.3), roi_top),
            ((width - roi_width) // 2 + int(roi_width * 0.7), roi_top),
            ((width + roi_width) // 2, roi_bottom)
        ]], dtype=np.int32)
        
        # 创建掩码
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, vertices, 255)
        
        return {
            'vertices': vertices,
            'mask': mask,
            'bounds': (roi_top, roi_bottom, roi_width)
        }
    
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


class RoadDetector:
    """道路检测器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def detect_road(self, image: np.ndarray, roi_mask: np.ndarray) -> Dict[str, Any]:
        """检测道路区域"""
        try:
            # 应用ROI
            roi_region = cv2.bitwise_and(image, image, mask=roi_mask)
            
            # 转换为灰度图
            gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
            
            # 自适应阈值
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学操作
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'contour': None, 'confidence': 0.0}
            
            # 找到最大轮廓
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            
            # 计算凸包和坚实度
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # 计算质心
            M = cv2.moments(main_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            
            # 计算置信度
            confidence = min(1.0, area / (roi_mask.shape[0] * roi_mask.shape[1]) * 2)
            
            return {
                'contour': main_contour,
                'centroid': (cx, cy),
                'area': area,
                'solidity': solidity,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"道路检测失败: {e}")
            return {'contour': None, 'confidence': 0.0}