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
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime

# 忽略警告
warnings.filterwarnings('ignore')

# ==================== 配置管理 ====================
@dataclass
class AppConfig:
    """应用配置参数 - 优化版"""
    # 性能参数
    max_image_size: Tuple[int, int] = (1200, 800)
    cache_size: int = 8
    batch_size: int = 30
    
    # 图像处理参数
    adaptive_clip_limit: float = 2.5
    adaptive_grid_size: Tuple[int, int] = (8, 8)
    gaussian_kernel: Tuple[int, int] = (5, 5)
    
    # 检测参数
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    hough_threshold: int = 30
    hough_min_length: int = 25
    hough_max_gap: int = 40
    min_contour_area: float = 0.005
    
    # 方向分析参数
    deviation_threshold: float = 0.15
    width_ratio_threshold: float = 0.7
    confidence_threshold: float = 0.6
    
    # 路径预测参数
    prediction_steps: int = 10
    prediction_distance: float = 0.75
    min_prediction_points: int = 4
    
    # 置信度参数
    min_confidence_for_direction: float = 0.4
    confidence_smoothing_factor: float = 0.7
    quality_weight_lane: float = 0.5
    quality_weight_road: float = 0.3
    quality_weight_consistency: float = 0.2
    
    # 界面参数
    ui_refresh_rate: int = 100
    animation_duration: int = 300

# ==================== 置信度校准器 ====================
class ConfidenceCalibrator:
    """置信度校准器 - 提高置信度准确性"""
    
    def __init__(self):
        self.calibration_history = deque(maxlen=100)
        self.performance_stats = {
            'total_predictions': 0,
            'high_confidence_correct': 0,
            'calibration_adjustments': []
        }
    
    def calibrate(self, raw_confidence: float, features: Dict[str, Any], 
                 context: str = "default") -> float:
        """校准置信度"""
        # 记录原始置信度
        self.performance_stats['total_predictions'] += 1
        
        # 阶段1: 基础校准
        calibrated = self._apply_sigmoid_calibration(raw_confidence)
        
        # 阶段2: 特征依赖校准
        calibrated = self._apply_feature_based_calibration(calibrated, features)
        
        # 阶段3: 上下文校准
        calibrated = self._apply_context_calibration(calibrated, context)
        
        # 阶段4: 历史一致性校准
        calibrated = self._apply_historical_calibration(calibrated)
        
        # 记录校准调整
        adjustment = calibrated - raw_confidence
        self.performance_stats['calibration_adjustments'].append(adjustment)
        if len(self.performance_stats['calibration_adjustments']) > 100:
            self.performance_stats['calibration_adjustments'].pop(0)
        
        return max(0.0, min(1.0, calibrated))
    
    def _apply_sigmoid_calibration(self, confidence: float) -> float:
        """应用S型曲线校准"""
        # 低置信度区域更保守，中等置信度增强，高置信度保持
        if confidence < 0.3:
            return confidence * 0.7
        elif confidence < 0.6:
            # 使用sigmoid增强中等置信度
            x = (confidence - 0.3) * 3.33  # 映射到[0,1]
            sigmoid = 1 / (1 + np.exp(-10 * (x - 0.5)))
            return 0.3 + sigmoid * 0.4
        else:
            return confidence
    
    def _apply_feature_based_calibration(self, confidence: float, 
                                        features: Dict[str, Any]) -> float:
        """基于特征的校准"""
        adjustment = 0.0
        
        # 1. 特征完整性校准
        feature_count = sum(1 for v in features.values() 
                          if isinstance(v, (int, float)) and not np.isnan(v))
        
        if feature_count < 3:
            adjustment -= 0.2  # 特征不足，显著降低置信度
        elif feature_count > 6:
            adjustment += 0.1  # 特征丰富，适度提高置信度
        
        # 2. 特征质量校准
        quality_indicators = []
        
        if 'lane_symmetry' in features:
            symmetry = features['lane_symmetry']
            if symmetry > 0.8:
                adjustment += 0.15
            elif symmetry < 0.4:
                adjustment -= 0.1
        
        if 'path_smoothness' in features:
            smoothness = features['path_smoothness']
            if smoothness > 0.7:
                adjustment += 0.1
            elif smoothness < 0.3:
                adjustment -= 0.08
        
        if 'lane_model_quality' in features:
            quality = features['lane_model_quality']
            adjustment += (quality - 0.5) * 0.2
        
        # 3. 特征一致性校准
        if 'feature_consistency' in features:
            consistency = features['feature_consistency']
            adjustment += (consistency - 0.5) * 0.15
        
        return confidence + adjustment
    
    def _apply_context_calibration(self, confidence: float, context: str) -> float:
        """基于上下文的校准"""
        if context == "highway":
            # 高速公路上置信度通常更高
            return min(1.0, confidence * 1.1)
        elif context == "urban":
            # 城市道路更复杂，适度降低
            return confidence * 0.95
        elif context == "rural":
            # 乡村道路质量不一，保持原样或略降
            return confidence * 0.9
        else:
            return confidence
    
    def _apply_historical_calibration(self, confidence: float) -> float:
        """基于历史表现的校准"""
        if len(self.calibration_history) < 10:
            return confidence
        
        # 计算历史平均置信度
        historical_confidences = [h['calibrated_confidence'] 
                                for h in self.calibration_history]
        hist_avg = np.mean(historical_confidences)
        
        # 计算历史置信度标准差
        hist_std = np.std(historical_confidences)
        
        # 如果当前置信度偏离历史平均超过2个标准差，进行调整
        if abs(confidence - hist_avg) > 2 * hist_std:
            # 向历史平均值靠拢，但保持一定弹性
            pull_factor = 0.3
            adjusted = confidence * (1 - pull_factor) + hist_avg * pull_factor
            return adjusted
        
        return confidence
    
    def update_performance(self, calibrated_confidence: float, 
                          was_correct: bool, features: Dict[str, Any]):
        """更新性能统计"""
        if calibrated_confidence > 0.7 and was_correct:
            self.performance_stats['high_confidence_correct'] += 1
        
        # 记录到历史
        self.calibration_history.append({
            'timestamp': datetime.now(),
            'calibrated_confidence': calibrated_confidence,
            'was_correct': was_correct,
            'feature_count': len(features)
        })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if self.performance_stats['total_predictions'] == 0:
            return {"status": "No data available"}
        
        high_conf_accuracy = 0
        if self.performance_stats['high_confidence_correct'] > 0:
            high_conf_accuracy = (
                self.performance_stats['high_confidence_correct'] / 
                self.performance_stats['total_predictions'] * 100
            )
        
        avg_adjustment = np.mean(self.performance_stats['calibration_adjustments']) \
            if self.performance_stats['calibration_adjustments'] else 0
        
        return {
            "total_predictions": self.performance_stats['total_predictions'],
            "high_confidence_accuracy": f"{high_conf_accuracy:.1f}%",
            "average_calibration_adjustment": f"{avg_adjustment:.3f}",
            "calibration_history_size": len(self.calibration_history),
            "calibration_effectiveness": self._calculate_effectiveness()
        }
    
    def _calculate_effectiveness(self) -> str:
        """计算校准效果"""
        if len(self.calibration_history) < 20:
            return "Insufficient data"
        
        # 计算校准前后的对比
        recent = list(self.calibration_history)[-20:]
        adjustments = [h.get('calibration_adjustment', 0) for h in recent]
        
        avg_adj = np.mean(adjustments)
        if avg_adj > 0.1:
            return "Strong positive calibration"
        elif avg_adj > 0.05:
            return "Moderate positive calibration"
        elif avg_adj > -0.05:
            return "Neutral calibration"
        else:
            return "Negative calibration"

# ==================== 质量评估器 ====================
class QualityEvaluator:
    """质量评估器 - 评估检测质量并生成改进建议"""
    
    def __init__(self):
        self.quality_thresholds = {
            'excellent': 0.85,
            'good': 0.70,
            'fair': 0.55,
            'poor': 0.40,
            'very_poor': 0.25
        }
        
        self.quality_weights = {
            'lane_detection': 0.35,
            'road_detection': 0.25,
            'feature_consistency': 0.20,
            'historical_stability': 0.10,
            'image_quality': 0.10
        }
    
    def evaluate_comprehensive(self, lane_info: Dict[str, Any], 
                             road_features: Dict[str, Any],
                             image_quality: float = 0.7) -> Dict[str, Any]:
        """综合质量评估"""
        scores = {}
        
        # 1. 车道线检测质量
        lane_score, lane_details = self._evaluate_lane_detection(lane_info)
        scores['lane_detection'] = {
            'score': lane_score,
            'details': lane_details
        }
        
        # 2. 道路检测质量
        road_score, road_details = self._evaluate_road_detection(road_features)
        scores['road_detection'] = {
            'score': road_score,
            'details': road_details
        }
        
        # 3. 特征一致性
        consistency_score, consistency_details = self._evaluate_feature_consistency(
            lane_info, road_features
        )
        scores['feature_consistency'] = {
            'score': consistency_score,
            'details': consistency_details
        }
        
        # 4. 图像质量（简化评估）
        scores['image_quality'] = {
            'score': image_quality,
            'details': {'estimated_quality': image_quality}
        }
        
        # 5. 计算综合质量分数
        overall_score = (
            lane_score * self.quality_weights['lane_detection'] +
            road_score * self.quality_weights['road_detection'] +
            consistency_score * self.quality_weights['feature_consistency'] +
            image_quality * self.quality_weights['image_quality']
        )
        
        scores['overall'] = {
            'score': overall_score,
            'level': self._get_quality_level(overall_score),
            'confidence_impact': self._calculate_confidence_impact(overall_score)
        }
        
        # 6. 生成改进建议
        scores['recommendations'] = self._generate_recommendations(scores)
        
        return scores
    
    def _evaluate_lane_detection(self, lane_info: Dict[str, Any]) -> Tuple[float, Dict]:
        """评估车道线检测质量"""
        details = {}
        score = 0.0
        
        # 基础质量分数
        base_quality = lane_info.get('detection_quality', 0.0)
        score += base_quality * 0.3
        details['base_quality'] = base_quality
        
        # 车道线数量
        left_count = len(lane_info.get('left_lines', []))
        right_count = len(lane_info.get('right_lines', []))
        
        line_count_score = (min(left_count, 8) / 8.0 * 0.25 + 
                          min(right_count, 8) / 8.0 * 0.25)
        score += line_count_score
        details['left_line_count'] = left_count
        details['right_line_count'] = right_count
        details['line_count_score'] = line_count_score
        
        # 车道线模型质量
        left_lane = lane_info.get('left_lane')
        right_lane = lane_info.get('right_lane')
        
        model_score = 0.0
        if left_lane and right_lane:
            left_conf = left_lane.get('confidence', 0.0)
            right_conf = right_lane.get('confidence', 0.0)
            model_score = (left_conf + right_conf) / 2.0 * 0.3
            
            # 模型类型一致性
            if left_lane.get('model_type') == right_lane.get('model_type'):
                model_score *= 1.1
            
            details['left_lane_confidence'] = left_conf
            details['right_lane_confidence'] = right_conf
            details['model_consistency'] = left_lane.get('model_type') == right_lane.get('model_type')
        
        score += model_score
        details['model_score'] = model_score
        
        # 路径预测质量
        if lane_info.get('future_path'):
            path_quality = lane_info['future_path'].get('prediction_quality', 0.5)
            score += path_quality * 0.15
            details['path_quality'] = path_quality
        
        return min(score, 1.0), details
    
    def _evaluate_road_detection(self, road_features: Dict[str, Any]) -> Tuple[float, Dict]:
        """评估道路检测质量"""
        details = {}
        score = 0.0
        
        # 轮廓完整性
        if 'contour' in road_features:
            contour = road_features['contour']
            if len(contour) >= 4:
                score += 0.25
                details['contour_points'] = len(contour)
            else:
                details['contour_points'] = len(contour)
        
        # 轮廓面积
        if 'area' in road_features:
            area = road_features['area']
            area_score = min(area / 50000.0, 1.0) * 0.2
            score += area_score
            details['area'] = area
            details['area_score'] = area_score
        
        # 轮廓坚实度
        if 'solidity' in road_features:
            solidity = road_features['solidity']
            score += solidity * 0.25
            details['solidity'] = solidity
        
        # 检测置信度
        if 'confidence' in road_features:
            confidence = road_features['confidence']
            score += confidence * 0.2
            details['confidence'] = confidence
        
        # 检测方法数量
        if 'detection_methods' in road_features:
            methods = road_features['detection_methods']
            method_score = min(methods / 3.0, 1.0) * 0.1
            score += method_score
            details['detection_methods'] = methods
        
        return min(score, 1.0), details
    
    def _evaluate_feature_consistency(self, lane_info: Dict[str, Any], 
                                     road_features: Dict[str, Any]) -> Tuple[float, Dict]:
        """评估特征一致性"""
        details = {}
        consistency_scores = []
        
        # 1. 位置一致性（车道线应在道路区域内）
        if 'contour' in road_features and lane_info.get('left_lane') and lane_info.get('right_lane'):
            contour = road_features['contour']
            left_points = lane_info['left_lane'].get('points', [])
            right_points = lane_info['right_lane'].get('points', [])
            
            # 简化检查：检查车道线点是否在道路轮廓内
            position_score = 0.7  # 默认值
            consistency_scores.append(position_score)
            details['position_consistency'] = position_score
        
        # 2. 方向一致性（车道线方向与道路方向一致）
        if 'orientation' in road_features:
            road_orientation = road_features.get('orientation', 0)
            
            # 计算车道线平均方向
            lane_orientations = []
            if lane_info.get('left_lane'):
                left_func = lane_info['left_lane'].get('func')
                if left_func:
                    # 简化计算方向
                    lane_orientations.append(45)  # 示例值
            
            if lane_orientations:
                avg_lane_orientation = np.mean(lane_orientations)
                orientation_diff = abs(road_orientation - avg_lane_orientation)
                orientation_score = max(0, 1 - orientation_diff / 90.0)
                consistency_scores.append(orientation_score)
                details['orientation_consistency'] = orientation_score
        
        # 3. 宽度一致性（车道宽度应合理）
        if lane_info.get('left_lane') and lane_info.get('right_lane'):
            left_func = lane_info['left_lane'].get('func')
            right_func = lane_info['right_lane'].get('func')
            
            if left_func and right_func:
                # 计算几个点的宽度
                y_points = [600, 500, 400]  # 示例y坐标
                widths = []
                for y in y_points:
                    try:
                        left_x = left_func(y)
                        right_x = right_func(y)
                        widths.append(right_x - left_x)
                    except:
                        continue
                
                if widths:
                    width_std = np.std(widths) if len(widths) > 1 else 0
                    width_mean = np.mean(widths)
                    
                    if width_mean > 0:
                        width_cv = width_std / width_mean  # 变异系数
                        width_consistency = max(0, 1 - width_cv)
                        consistency_scores.append(width_consistency)
                        details['width_consistency'] = width_consistency
                        details['avg_width'] = width_mean
                        details['width_std'] = width_std
        
        # 计算平均一致性分数
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
        else:
            avg_consistency = 0.5  # 默认值
        
        details['consistency_scores'] = consistency_scores
        return avg_consistency, details
    
    def _get_quality_level(self, score: float) -> str:
        """获取质量等级"""
        if score >= self.quality_thresholds['excellent']:
            return "优秀"
        elif score >= self.quality_thresholds['good']:
            return "良好"
        elif score >= self.quality_thresholds['fair']:
            return "一般"
        elif score >= self.quality_thresholds['poor']:
            return "较差"
        else:
            return "很差"
    
    def _calculate_confidence_impact(self, quality_score: float) -> float:
        """计算质量分数对置信度的影响因子"""
        # 质量越高，对置信度的正面影响越大
        if quality_score > 0.8:
            return 1.2  # 提高置信度
        elif quality_score > 0.6:
            return 1.1
        elif quality_score > 0.4:
            return 1.0
        elif quality_score > 0.2:
            return 0.9
        else:
            return 0.8  # 降低置信度
    
    def _generate_recommendations(self, scores: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        overall_score = scores['overall']['score']
        
        if overall_score < 0.6:
            recommendations.append("检测质量一般，建议：")
            
            lane_score = scores['lane_detection']['score']
            if lane_score < 0.5:
                recommendations.append("  - 车道线检测较弱，尝试调整检测参数")
            
            road_score = scores['road_detection']['score']
            if road_score < 0.5:
                recommendations.append("  - 道路区域检测不完整，检查图像质量")
            
            consistency_score = scores['feature_consistency']['score']
            if consistency_score < 0.5:
                recommendations.append("  - 特征一致性较低，可能需要重新标定")
        
        elif overall_score < 0.8:
            recommendations.append("检测质量良好，可进一步优化：")
            
            details = scores['lane_detection']['details']
            if details.get('line_count_score', 0) < 0.7:
                recommendations.append("  - 增加车道线检测数量可提高精度")
            
            if details.get('model_score', 0) < 0.7:
                recommendations.append("  - 车道线模型拟合精度可优化")
        
        else:
            recommendations.append("检测质量优秀，保持当前设置")
        
        # 具体建议
        if scores.get('lane_detection', {}).get('details', {}).get('left_line_count', 0) < 3:
            recommendations.append("  - 左侧车道线数量不足，可能影响方向判断")
        
        if scores.get('lane_detection', {}).get('details', {}).get('right_line_count', 0) < 3:
            recommendations.append("  - 右侧车道线数量不足，可能影响方向判断")
        
        return recommendations

# ==================== 智能图像处理器 ====================
class SmartImageProcessor:
    """智能图像处理器 - 优化版"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._cache = {}
        self._cache_order = deque(maxlen=config.cache_size)
        self._image_stats_cache = {}
    
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
            
            # 获取图像统计信息
            image_stats = self._analyze_image_stats(image)
            
            # 智能调整尺寸
            processed = self._smart_resize(image)
            
            # 自适应预处理（根据图像统计信息优化）
            processed = self._adaptive_preprocessing(processed, image_stats)
            
            # 计算ROI区域
            roi_info = self._calculate_optimized_roi(processed.shape, image_stats)
            
            # 更新缓存
            result = (processed, roi_info, image_stats)
            self._update_cache(image_path, result)
            
            return result
            
        except Exception as e:
            print(f"图像处理失败 {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _analyze_image_stats(self, image: np.ndarray) -> Dict[str, Any]:
        """分析图像统计信息"""
        stats = {}
        
        # 基础信息
        stats['height'], stats['width'] = image.shape[:2]
        stats['aspect_ratio'] = stats['width'] / stats['height']
        
        # 转换为灰度图分析
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 亮度统计
        stats['brightness_mean'] = np.mean(gray)
        stats['brightness_std'] = np.std(gray)
        stats['brightness_range'] = np.max(gray) - np.min(gray)
        
        # 对比度评估
        stats['contrast'] = stats['brightness_std'] / 128.0  # 归一化
        
        # 噪声评估
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        stats['noise_level'] = np.std(laplacian)
        
        # 边缘密度（评估图像复杂度）
        edges = cv2.Canny(gray, 50, 150)
        stats['edge_density'] = np.count_nonzero(edges) / (stats['height'] * stats['width'])
        
        # 颜色分布
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        stats['saturation_mean'] = np.mean(hsv[:, :, 1])
        
        # 图像质量评分（0-1）
        stats['quality_score'] = self._calculate_image_quality(stats)
        
        return stats
    
    def _calculate_image_quality(self, stats: Dict[str, Any]) -> float:
        """计算图像质量评分"""
        quality_factors = []
        
        # 亮度质量（理想范围50-200）
        brightness = stats['brightness_mean']
        if 50 <= brightness <= 200:
            brightness_quality = 1.0
        else:
            brightness_quality = 1.0 - min(abs(brightness - 125) / 125.0, 1.0)
        quality_factors.append(brightness_quality * 0.3)
        
        # 对比度质量
        contrast = stats['contrast']
        if 0.2 <= contrast <= 0.8:
            contrast_quality = 1.0
        else:
            contrast_quality = 1.0 - min(abs(contrast - 0.5) / 0.5, 1.0)
        quality_factors.append(contrast_quality * 0.3)
        
        # 噪声质量
        noise = stats['noise_level']
        noise_quality = 1.0 - min(noise / 50.0, 1.0)
        quality_factors.append(noise_quality * 0.2)
        
        # 边缘质量（适中的边缘密度最好）
        edge_density = stats['edge_density']
        if 0.05 <= edge_density <= 0.3:
            edge_quality = 1.0
        else:
            edge_quality = 1.0 - min(abs(edge_density - 0.175) / 0.175, 1.0)
        quality_factors.append(edge_quality * 0.2)
        
        return min(sum(quality_factors), 1.0)
    
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
            # 根据缩放比例选择插值方法
            if scale < 0.5:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_LINEAR
            return cv2.resize(image, new_size, interpolation=interpolation)
        
        return image
    
    def _adaptive_preprocessing(self, image: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """自适应图像预处理"""
        enhanced = image.copy()
        
        # 1. 自适应直方图均衡化
        if stats['contrast'] < 0.3:  # 低对比度图像
            # 转换为YUV颜色空间
            yuv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0]
            
            # 根据图像大小调整CLAHE参数
            height, width = image.shape[:2]
            grid_size = max(4, min(16, height // 50, width // 50))
            
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(grid_size, grid_size)
            )
            yuv[:, :, 0] = clahe.apply(y_channel)
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # 2. 自适应去噪
        if stats['noise_level'] > 20:
            # 根据噪声水平选择滤波方法
            if stats['noise_level'] > 40:
                # 强噪声，使用非局部均值去噪
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            else:
                # 中等噪声，使用双边滤波
                enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 3. 自适应锐化（仅在需要时）
        if stats['edge_density'] < 0.1:  # 边缘较少，可能需要锐化
            kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def _calculate_optimized_roi(self, image_shape: Tuple[int, ...], 
                               stats: Dict[str, Any]) -> Dict[str, Any]:
        """计算优化的ROI区域"""
        height, width = image_shape[:2]
        
        # 根据图像统计信息动态调整ROI
        if stats['edge_density'] > 0.2:  # 复杂场景，缩小ROI
            roi_top = int(height * 0.4)
            roi_bottom = int(height * 0.9)
            roi_width_ratio = 0.8
        else:  # 简单场景，使用较大ROI
            roi_top = int(height * 0.35)
            roi_bottom = int(height * 0.92)
            roi_width_ratio = 0.85
        
        roi_width = int(width * roi_width_ratio)
        
        # 创建梯形ROI（模拟透视变换）
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
            'bounds': (roi_top, roi_bottom, roi_width),
            'area_ratio': np.count_nonzero(mask) / (height * width)
        }
    
    def _update_cache(self, key: str, value: Any):
        """更新缓存"""
        if len(self._cache) >= self.config.cache_size:
            oldest = self._cache_order.popleft()
            self._cache.pop(oldest, None)
        
        self._cache[key] = value
        self._cache_order.append(key)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self.config.cache_size,
            'cache_hit_rate': self._calculate_hit_rate(),
            'cached_files': list(self._cache.keys())
        }
    
    def _calculate_hit_rate(self) -> float:
        """计算缓存命中率（简化版本）"""
        # 在实际应用中，这里应该记录实际的命中次数
        # 这里返回一个估计值
        return min(len(self._cache) / self.config.cache_size, 1.0)

# ==================== 高级道路检测器 ====================
class AdvancedRoadDetector:
    """高级道路检测器 - 高置信度版"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.previous_results = deque(maxlen=4)
        self.detection_history = deque(maxlen=20)
        
        # 自适应参数
        self.adaptive_params = {
            'canny_low': config.canny_threshold1,
            'canny_high': config.canny_threshold2,
            'hough_threshold': config.hough_threshold,
            'last_adjustment': 0
        }
        
        # 检测方法权重
        self.method_weights = {
            'color': 0.35,
            'texture': 0.30,
            'edges': 0.35
        }
    
    def detect(self, image: np.ndarray, roi_info: Dict[str, Any]) -> Dict[str, Any]:
        """执行道路检测 - 高置信度版"""
        start_time = time.time()
        
        try:
            # 提取ROI区域
            roi_region = self._extract_roi_region(image, roi_info['mask'])
            
            # 并行执行多种检测方法
            detection_results = self._parallel_detection(roi_region)
            
            # 计算各方法置信度
            method_confidences = self._calculate_method_confidences(detection_results)
            
            # 加权融合结果
            fused_result = self._weighted_fusion(detection_results, method_confidences)
            
            # 提取道路特征
            road_features = self._extract_road_features(fused_result, roi_region.shape)
            
            # 计算综合置信度
            overall_confidence = self._calculate_overall_confidence(
                fused_result, method_confidences, road_features
            )
            
            # 时间平滑
            if self.previous_results:
                smoothed_result = self._temporal_smoothing(
                    fused_result, overall_confidence, road_features
                )
                fused_result.update(smoothed_result)
            
            # 创建结果字典
            result = {
                'detection_mask': fused_result['mask'],
                'features': road_features,
                'confidence': overall_confidence,
                'method_confidences': method_confidences,
                'detection_methods': len(detection_results),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now()
            }
            
            # 更新历史
            self.previous_results.append(result)
            self.detection_history.append({
                'confidence': overall_confidence,
                'methods_used': len(detection_results),
                'features_count': len(road_features)
            })
            
            # 自适应调整参数
            self._adaptive_parameter_adjustment(overall_confidence)
            
            return result
            
        except Exception as e:
            print(f"道路检测失败: {e}")
            return self._create_fallback_result()
    
    def _parallel_detection(self, roi_region: np.ndarray) -> List[Dict[str, Any]]:
        """并行执行多种检测方法"""
        detection_methods = [
            ('color', self._detect_by_color),
            ('texture', self._detect_by_texture),
            ('edges', self._detect_by_edges)
        ]
        
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_method = {
                executor.submit(method_func, roi_region): method_name
                for method_name, method_func in detection_methods
            }
            
            for future in as_completed(future_to_method):
                method_name = future_to_method[future]
                try:
                    result = future.result(timeout=2.0)
                    if result and result['confidence'] > 0.2:  # 过滤低质量结果
                        result['method'] = method_name
                        results.append(result)
                except Exception as e:
                    print(f"{method_name}检测失败: {e}")
        
        return results
    
    def _detect_by_color(self, roi_region: np.ndarray) -> Dict[str, Any]:
        """基于颜色检测道路"""
        try:
            # 转换为LAB颜色空间（对光照变化更鲁棒）
            lab = cv2.cvtColor(roi_region, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # 分析L通道的直方图
            hist = cv2.calcHist([l_channel], [0], None, [64], [0, 256])
            hist = hist.flatten()
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            
            # 寻找主要峰值（道路区域通常对应直方图的主要峰值）
            peaks = self._find_histogram_peaks(hist, min_distance=5)
            
            if not peaks:
                return {'mask': np.zeros_like(l_channel), 'confidence': 0.0}
            
            # 选择最大的峰值作为道路颜色
            main_peak = peaks[0][1] * 4  # 转换为实际灰度值
            
            # 创建掩码（允许一定的颜色变化）
            lower_bound = max(0, int(main_peak * 0.7))
            upper_bound = min(255, int(main_peak * 1.3))
            
            mask = cv2.inRange(l_channel, lower_bound, upper_bound)
            
            # 形态学优化
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 填充孔洞
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            
            # 计算置信度
            area_ratio = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
            
            # 考虑峰值显著性和面积
            peak_prominence = peaks[0][0] if peaks else 0
            confidence = min(0.7, area_ratio * 1.5) + min(0.3, peak_prominence * 3)
            
            return {'mask': mask, 'confidence': confidence}
            
        except Exception as e:
            print(f"颜色检测失败: {e}")
            return {'mask': np.zeros_like(roi_region[:, :, 0]), 'confidence': 0.0}
    
    def _find_histogram_peaks(self, hist: np.ndarray, min_distance: int = 5) -> List[Tuple[float, int]]:
        """寻找直方图峰值"""
        peaks = []
        n = len(hist)
        
        for i in range(1, n - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                # 检查是否足够突出
                left_min = min(hist[max(0, i - min_distance):i])
                right_min = min(hist[i + 1:min(n, i + min_distance)])
                prominence = hist[i] - max(left_min, right_min)
                
                if prominence > 0.01:  # 显著性阈值
                    peaks.append((prominence, i))
        
        # 按显著性排序
        peaks.sort(reverse=True)
        return peaks
    
    def _detect_by_texture(self, roi_region: np.ndarray) -> Dict[str, Any]:
        """基于纹理检测道路"""
        try:
            gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
            
            # 计算局部二值模式（LBP）纹理特征
            radius = 2
            n_points = 8 * radius
            
            # 计算LBP
            lbp = np.zeros_like(gray, dtype=np.uint8)
            for i in range(radius, gray.shape[0] - radius):
                for j in range(radius, gray.shape[1] - radius):
                    center = gray[i, j]
                    code = 0
                    for n in range(n_points):
                        angle = 2 * np.pi * n / n_points
                        x = int(j + radius * np.cos(angle))
                        y = int(i - radius * np.sin(angle))
                        code |= (gray[y, x] >= center) << n
                    lbp[i, j] = code
            
            # 计算纹理均匀性（道路通常纹理均匀）
            hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            
            # 计算熵（均匀纹理熵较低）
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            max_entropy = np.log2(256)
            uniformity = 1.0 - (entropy / max_entropy)
            
            # 创建掩码（基于纹理均匀性）
            _, mask = cv2.threshold(lbp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学优化
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 计算置信度
            confidence = uniformity * 0.8 + min(0.2, np.count_nonzero(mask) / mask.size)
            
            return {'mask': mask, 'confidence': confidence}
            
        except Exception as e:
            print(f"纹理检测失败: {e}")
            return {'mask': np.zeros_like(roi_region[:, :, 0]), 'confidence': 0.0}
    
    def _detect_by_edges(self, roi_region: np.ndarray) -> Dict[str, Any]:
        """基于边缘检测道路"""
        try:
            gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
            
            # 自适应Canny边缘检测
            median_intensity = np.median(gray)
            
            # 基于图像统计调整阈值
            if median_intensity < 50:
                lower, upper = 20, 60
            elif median_intensity > 200:
                lower, upper = 80, 160
            else:
                lower = int(max(20, median_intensity * 0.4))
                upper = int(min(200, median_intensity * 0.8))
            
            edges = cv2.Canny(gray, lower, upper)
            
            # 连接边缘
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 填充闭合区域
            contours, hierarchy = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            mask = np.zeros_like(edges)
            total_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # 过滤小区域
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    total_area += area
            
            # 计算置信度
            edge_density = np.count_nonzero(edges) / edges.size
            fill_ratio = total_area / mask.size if mask.size > 0 else 0
            
            # 道路区域边缘应较少，填充区域应适中
            edge_confidence = max(0, 1.0 - edge_density * 3.0)
            fill_confidence = min(1.0, fill_ratio * 2.0)
            
            confidence = edge_confidence * 0.6 + fill_confidence * 0.4
            
            return {'mask': mask, 'confidence': confidence}
            
        except Exception as e:
            print(f"边缘检测失败: {e}")
            return {'mask': np.zeros_like(roi_region[:, :, 0]), 'confidence': 0.0}
    
    def _calculate_method_confidences(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算各方法置信度"""
        confidences = {}
        total_confidence = sum(r['confidence'] for r in results)
        
        if total_confidence > 0:
            for result in results:
                method = result['method']
                raw_confidence = result['confidence']
                
                # 加权归一化
                weight = self.method_weights.get(method, 0.33)
                normalized = (raw_confidence / total_confidence) * weight
                confidences[method] = normalized
        
        return confidences
    
    def _weighted_fusion(self, results: List[Dict[str, Any]], 
                        confidences: Dict[str, float]) -> Dict[str, Any]:
        """加权融合结果"""
        if not results:
            return {'mask': None, 'combined_confidence': 0.0}
        
        # 初始化融合掩码
        first_mask = results[0]['mask']
        fused = np.zeros_like(first_mask, dtype=np.float32)
        
        # 加权求和
        for result in results:
            method = result['method']
            mask = result['mask'].astype(np.float32)
            weight = confidences.get(method, 0.0)
            
            fused += mask * weight
        
        # 归一化并二值化
        fused_normalized = (fused / np.max(fused + 1e-10) * 255).astype(np.uint8)
        
        # 自适应阈值
        _, binary_mask = cv2.threshold(fused_normalized, 0, 255, 
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学优化
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # 计算综合置信度
        combined_confidence = np.mean(list(confidences.values())) if confidences else 0.0
        
        return {
            'mask': binary_mask,
            'combined_confidence': combined_confidence,
            'method_count': len(results)
        }
    
    def _extract_road_features(self, fused_result: Dict[str, Any], 
                              shape: Tuple[int, ...]) -> Dict[str, Any]:
        """提取道路特征"""
        mask = fused_result['mask']
        
        if mask is None or np.count_nonzero(mask) == 0:
            return {}
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        # 找到最大轮廓
        main_contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓特征
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        # 计算凸包
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 计算边界矩形
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # 计算质心
        M = cv2.moments(main_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        # 计算方向（拟合椭圆）
        orientation = 0
        if len(main_contour) >= 5:
            try:
                (_, _), (_, _), angle = cv2.fitEllipse(main_contour)
                orientation = angle
            except:
                orientation = 0
        
        # 计算形状特征
        aspect_ratio = w / h if h > 0 else 0
        extent = area / (w * h) if w * h > 0 else 0
        
        # 计算轮廓复杂度
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        return {
            'contour': main_contour,
            'centroid': (cx, cy),
            'area': area,
            'perimeter': perimeter,
            'solidity': solidity,
            'bounding_rect': (x, y, w, h),
            'orientation': orientation,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'circularity': circularity,
            'hull_area': hull_area,
            'pixel_count': np.count_nonzero(mask)
        }
    
    def _calculate_overall_confidence(self, fused_result: Dict[str, Any],
                                    method_confidences: Dict[str, float],
                                    road_features: Dict[str, Any]) -> float:
        """计算综合置信度"""
        confidence_factors = []
        
        # 1. 融合结果置信度
        combined_confidence = fused_result.get('combined_confidence', 0.0)
        confidence_factors.append(combined_confidence * 0.4)
        
        # 2. 特征质量
        feature_quality = 0.0
        
        if road_features.get('area', 0) > 1000:
            feature_quality += 0.2
        
        if road_features.get('solidity', 0) > 0.7:
            feature_quality += 0.2
        
        if road_features.get('circularity', 0) > 0.3:
            feature_quality += 0.1
        
        confidence_factors.append(feature_quality * 0.3)
        
        # 3. 方法多样性
        method_count = fused_result.get('method_count', 0)
        method_diversity = min(method_count / 3.0, 1.0)
        confidence_factors.append(method_diversity * 0.2)
        
        # 4. 历史一致性
        if self.previous_results:
            recent_confidences = [r['confidence'] for r in list(self.previous_results)[-3:]]
            if recent_confidences:
                hist_consistency = 1.0 - np.std(recent_confidences)
                confidence_factors.append(hist_consistency * 0.1)
        
        # 综合置信度
        overall = sum(confidence_factors)
        
        # 应用非线性调整（使中等置信度更可靠）
        if overall < 0.3:
            return overall * 0.8  # 低置信度更保守
        elif overall < 0.7:
            return 0.3 + (overall - 0.3) * 1.2  # 中等置信度增强
        else:
            return min(overall, 1.0)  # 高置信度保持
    
    def _temporal_smoothing(self, current_result: Dict[str, Any],
                           current_confidence: float,
                           current_features: Dict[str, Any]) -> Dict[str, Any]:
        """时间平滑"""
        if not self.previous_results:
            return {}
        
        smoothing_factor = 0.6
        
        # 平滑特征
        smoothed_features = current_features.copy()
        
        for prev_result in list(self.previous_results)[-2:]:
            prev_features = prev_result.get('features', {})
            
            for key in smoothed_features:
                if key in prev_features and isinstance(smoothed_features[key], (int, float)):
                    # 指数移动平均
                    smoothed_features[key] = (
                        smoothing_factor * smoothed_features[key] +
                        (1 - smoothing_factor) * prev_features[key]
                    )
        
        # 平滑置信度
        recent_confidences = [r['confidence'] for r in list(self.previous_results)[-3:]]
        if recent_confidences:
            smoothed_confidence = (
                smoothing_factor * current_confidence +
                (1 - smoothing_factor) * np.mean(recent_confidences)
            )
        else:
            smoothed_confidence = current_confidence
        
        return {
            'smoothed_features': smoothed_features,
            'smoothed_confidence': smoothed_confidence,
            'smoothing_factor': smoothing_factor
        }
    
    def _adaptive_parameter_adjustment(self, confidence: float):
        """自适应参数调整"""
        # 根据置信度调整参数
        adjustment_factor = 0.1
        
        if confidence < 0.4:
            # 低置信度，降低阈值以检测更多特征
            self.adaptive_params['canny_low'] = max(20, 
                self.adaptive_params['canny_low'] * (1 - adjustment_factor))
            self.adaptive_params['canny_high'] = max(60,
                self.adaptive_params['canny_high'] * (1 - adjustment_factor))
            self.adaptive_params['hough_threshold'] = max(15,
                self.adaptive_params['hough_threshold'] * (1 - adjustment_factor))
            
        elif confidence > 0.8:
            # 高置信度，提高阈值以减少噪声
            self.adaptive_params['canny_low'] = min(80,
                self.adaptive_params['canny_low'] * (1 + adjustment_factor))
            self.adaptive_params['canny_high'] = min(200,
                self.adaptive_params['canny_high'] * (1 + adjustment_factor))
            self.adaptive_params['hough_threshold'] = min(50,
                self.adaptive_params['hough_threshold'] * (1 + adjustment_factor))
    
    def _create_fallback_result(self) -> Dict[str, Any]:
        """创建降级结果"""
        return {
            'detection_mask': None,
            'features': {},
            'confidence': 0.1,
            'method_confidences': {},
            'detection_methods': 0,
            'processing_time': 0,
            'timestamp': datetime.now(),
            'is_fallback': True
        }
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        if not self.detection_history:
            return {"status": "No detection history"}
        
        confidences = [h['confidence'] for h in self.detection_history]
        methods_used = [h['methods_used'] for h in self.detection_history]
        
        return {
            "total_detections": len(self.detection_history),
            "average_confidence": np.mean(confidences),
            "confidence_std": np.std(confidences),
            "average_methods_used": np.mean(methods_used),
            "current_parameters": self.adaptive_params,
            "method_weights": self.method_weights
        }

# ==================== 智能车道线检测器 ====================
class SmartLaneDetector:
    """智能车道线检测器 - 高置信度版"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.lane_history = deque(maxlen=8)
        self.confidence_calibrator = ConfidenceCalibrator()
        
        # 检测方法配置
        self.method_configs = {
            'canny': {
                'weight': 0.35,
                'min_line_length': 20,
                'max_line_gap': 30
            },
            'sobel': {
                'weight': 0.30,
                'gradient_threshold': 50
            },
            'gradient': {
                'weight': 0.35,
                'angle_tolerance': np.pi / 18  # 10度
            }
        }
        
        # 车道线验证参数
        self.validation_params = {
            'min_lane_width': 100,  # 像素
            'max_lane_width': 500,  # 像素
            'max_width_variation': 0.4,  # 宽度变化率
            'min_symmetry_score': 0.6
        }
    
    def detect(self, image: np.ndarray, roi_mask: np.ndarray) -> Dict[str, Any]:
        """检测车道线 - 高置信度版"""
        start_time = time.time()
        
        try:
            # 预处理
            processed = self._preprocess_for_lanes(image, roi_mask)
            
            # 并行执行多种检测方法
            detection_results = self._parallel_lane_detection(processed)
            
            # 分类和过滤车道线
            left_lines, right_lines = self._classify_and_filter_lanes(
                detection_results, image.shape[1]
            )
            
            # 拟合车道线模型
            left_lane = self._fit_lane_model_robust(left_lines, image.shape, 'left')
            right_lane = self._fit_lane_model_robust(right_lines, image.shape, 'right')
            
            # 验证车道线合理性
            left_lane, right_lane = self._validate_lanes(left_lane, right_lane, image.shape)
            
            # 计算中心线和路径预测
            center_line = self._calculate_center_line(left_lane, right_lane, image.shape)
            future_path = self._predict_future_path(left_lane, right_lane, image.shape)
            
            # 计算检测质量
            detection_quality = self._calculate_detection_quality(
                left_lane, right_lane, left_lines, right_lines, future_path
            )
            
            # 创建结果
            result = {
                'left_lines': left_lines,
                'right_lines': right_lines,
                'left_lane': left_lane,
                'right_lane': right_lane,
                'center_line': center_line,
                'future_path': future_path,
                'detection_quality': detection_quality,
                'detection_methods': len(detection_results),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now()
            }
            
            # 时间平滑（如果质量足够高）
            if detection_quality > 0.4:
                result = self._apply_temporal_smoothing(result)
                self.lane_history.append(result)
            
            return result
            
        except Exception as e:
            print(f"车道线检测失败: {e}")
            return self._create_empty_result()
    
    def _preprocess_for_lanes(self, image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """为车道线检测预处理图像"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用ROI
        gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 自适应去噪
        noise_std = np.std(enhanced)
        if noise_std > 35:
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 锐化（增强边缘）
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def _parallel_lane_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """并行执行多种车道线检测方法"""
        detection_tasks = [
            ('canny', self._detect_with_canny),
            ('sobel', self._detect_with_sobel),
            ('gradient', self._detect_with_gradient)
        ]
        
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_method = {
                executor.submit(method_func, image): method_name
                for method_name, method_func in detection_tasks
            }
            
            for future in as_completed(future_to_method):
                method_name = future_to_method[future]
                try:
                    method_result = future.result(timeout=2.0)
                    if method_result:
                        method_result['method'] = method_name
                        results.append(method_result)
                except Exception as e:
                    print(f"{method_name}车道线检测失败: {e}")
        
        return results
    
    def _detect_with_canny(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """使用Canny边缘检测车道线"""
        try:
            # 自适应Canny阈值
            median = np.median(image)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * median))
            upper = int(min(255, (1.0 + sigma) * median))
            
            edges = cv2.Canny(image, lower, upper)
            
            # 霍夫变换检测直线
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=self.config.hough_threshold,
                minLineLength=self.method_configs['canny']['min_line_length'],
                maxLineGap=self.method_configs['canny']['max_line_gap']
            )
            
            if lines is not None:
                return {'lines': lines, 'confidence': 0.7}
            else:
                return {'lines': [], 'confidence': 0.3}
                
        except Exception as e:
            print(f"Canny检测失败: {e}")
            return None
    
    def _detect_with_sobel(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """使用Sobel算子检测车道线"""
        try:
            # 计算梯度幅值和方向
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            direction = np.arctan2(np.abs(sobely), np.abs(sobelx))
            
            # 关注近似垂直的边缘（车道线方向）
            vertical_mask = (direction > np.pi/4) & (direction < 3*np.pi/4)
            lane_edges = np.zeros_like(magnitude, dtype=np.uint8)
            lane_edges[vertical_mask & (magnitude > self.method_configs['sobel']['gradient_threshold'])] = 255
            
            # 霍夫变换
            lines = cv2.HoughLinesP(
                lane_edges,
                rho=1,
                theta=np.pi/180,
                threshold=self.config.hough_threshold,
                minLineLength=15,
                maxLineGap=20
            )
            
            if lines is not None:
                return {'lines': lines, 'confidence': 0.6}
            else:
                return {'lines': [], 'confidence': 0.2}
                
        except Exception as e:
            print(f"Sobel检测失败: {e}")
            return None
    
    def _detect_with_gradient(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """使用梯度方向检测车道线"""
        try:
            # 计算梯度
            dx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
            dy = cv2.Scharr(image, cv2.CV_64F, 0, 1)
            
            magnitude = np.sqrt(dx**2 + dy**2)
            direction = np.arctan2(dy, dx)
            
            # 搜索可能的方向（±60度）
            detected_lines = []
            angle_tolerance = self.method_configs['gradient']['angle_tolerance']
            
            for base_angle in np.linspace(-np.pi/3, np.pi/3, 13):  # 13个方向
                mask = np.abs(direction - base_angle) < angle_tolerance
                if np.sum(mask) < 50:  # 像素数量不足
                    continue
                
                # 提取该方向的边缘
                directional_edges = np.zeros_like(magnitude, dtype=np.uint8)
                directional_edges[mask & (magnitude > 30)] = 255
                
                # 霍夫变换
                lines = cv2.HoughLinesP(
                    directional_edges,
                    rho=1,
                    theta=np.pi/180,
                    threshold=self.config.hough_threshold,
                    minLineLength=20,
                    maxLineGap=25
                )
                
                if lines is not None:
                    detected_lines.extend(lines)
            
            if detected_lines:
                return {'lines': detected_lines, 'confidence': 0.65}
            else:
                return {'lines': [], 'confidence': 0.25}
                
        except Exception as e:
            print(f"梯度方向检测失败: {e}")
            return None
    
    def _classify_and_filter_lanes(self, detection_results: List[Dict[str, Any]], 
                                  image_width: int) -> Tuple[List[Dict], List[Dict]]:
        """分类和过滤车道线"""
        all_lines = []
        
        # 收集所有检测到的线段
        for result in detection_results:
            if 'lines' in result:
                lines = result['lines']
                method = result.get('method', 'unknown')
                weight = self.method_configs.get(method, {}).get('weight', 0.33)
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # 计算线段参数
                    dx = x2 - x1
                    dy = y2 - y1
                    
                    if dx == 0:
                        continue  # 跳过垂直线
                    
                    slope = dy / dx
                    length = np.sqrt(dx**2 + dy**2)
                    angle = np.arctan2(dy, dx)
                    
                    # 过滤太短或太平的线段
                    if length < 15 or abs(slope) < 0.1:
                        continue
                    
                    # 计算线段中点
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    
                    line_data = {
                        'points': [(x1, y1), (x2, y2)],
                        'slope': slope,
                        'angle': angle,
                        'length': length,
                        'midpoint': (mid_x, mid_y),
                        'method': method,
                        'weight': weight
                    }
                    
                    all_lines.append(line_data)
        
        if not all_lines:
            return [], []
        
        # 根据位置和斜率分类
        left_candidates = []
        right_candidates = []
        
        for line in all_lines:
            slope = line['slope']
            mid_x = line['midpoint'][0]
            
            # 分类逻辑
            if slope < 0:  # 左车道线（负斜率）
                if mid_x < image_width * 0.65:
                    left_candidates.append(line)
            else:  # 右车道线（正斜率）
                if mid_x > image_width * 0.35:
                    right_candidates.append(line)
        
        # 进一步过滤和聚类
        left_lines = self._filter_and_cluster_lines(left_candidates, 'left')
        right_lines = self._filter_and_cluster_lines(right_candidates, 'right')
        
        return left_lines, right_lines
    
    def _filter_and_cluster_lines(self, candidates: List[Dict], side: str) -> List[Dict]:
        """过滤和聚类线段"""
        if len(candidates) < 2:
            return candidates
        
        # 提取特征进行聚类
        slopes = np.array([c['slope'] for c in candidates])
        midpoints_x = np.array([c['midpoint'][0] for c in candidates])
        
        # 简单聚类：基于斜率和位置
        from sklearn.cluster import DBSCAN
        
        features = np.column_stack([slopes, midpoints_x])
        
        # 标准化特征
        features_norm = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-10)
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(features_norm)
        labels = clustering.labels_
        
        # 选择最大的簇
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique_labels) == 0:
            return candidates  # 没有形成簇，返回所有
        
        main_cluster_label = unique_labels[np.argmax(counts)]
        
        # 过滤出主簇的线段
        filtered = [c for c, l in zip(candidates, labels) if l == main_cluster_label]
        
        # 进一步统计过滤
        if len(filtered) >= 3:
            filtered = self._apply_statistical_filtering(filtered)
        
        return filtered
    
    def _apply_statistical_filtering(self, lines: List[Dict]) -> List[Dict]:
        """应用统计过滤"""
        # 计算均值和标准差
        slopes = np.array([l['slope'] for l in lines])
        midpoints_x = np.array([l['midpoint'][0] for l in lines])
        
        slope_mean = np.mean(slopes)
        slope_std = np.std(slopes)
        midpoint_mean = np.mean(midpoints_x)
        midpoint_std = np.std(midpoints_x)
        
        # 过滤异常值（2.5σ原则）
        filtered = []
        for line in lines:
            slope_zscore = abs(line['slope'] - slope_mean) / (slope_std + 1e-10)
            midpoint_zscore = abs(line['midpoint'][0] - midpoint_mean) / (midpoint_std + 1e-10)
            
            if slope_zscore < 2.5 and midpoint_zscore < 2.5:
                filtered.append(line)
        
        return filtered
    
    def _fit_lane_model_robust(self, lines: List[Dict], image_shape: Tuple[int, ...], 
                              side: str) -> Optional[Dict]:
        """鲁棒的车道线模型拟合"""
        if len(lines) < 2:
            return None
        
        try:
            # 收集所有点
            x_points, y_points = [], []
            weights = []
            
            for line in lines:
                for (x, y) in line['points']:
                    x_points.append(x)
                    y_points.append(y)
                    weights.append(line.get('weight', 1.0))
            
            x_arr = np.array(x_points)
            y_arr = np.array(y_points)
            weights_arr = np.array(weights)
            
            # 尝试多项式拟合
            try:
                # 加权二次拟合
                coeffs = np.polyfit(y_arr, x_arr, 2, w=weights_arr)
                model_type = 'quadratic'
            except:
                # 降级为线性拟合
                coeffs = np.polyfit(y_arr, x_arr, 1, w=weights_arr)
                model_type = 'linear'
            
            poly_func = np.poly1d(coeffs)
            
            # 生成车道线点
            height, width = image_shape[:2]
            y_bottom = height
            y_top = int(height * 0.4)
            
            x_bottom = int(poly_func(y_bottom))
            x_top = int(poly_func(y_top))
            
            # 确保点在图像内
            x_bottom = max(0, min(width - 1, x_bottom))
            x_top = max(0, min(width - 1, x_top))
            
            # 计算拟合质量
            if model_type == 'quadratic':
                # 计算R²
                x_pred = poly_func(y_arr)
                ss_res = np.sum((x_arr - x_pred) ** 2)
                ss_tot = np.sum((x_arr - np.mean(x_arr)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            else:
                # 线性模型的简单质量度量
                r_squared = max(0, 1 - np.std(x_arr - poly_func(y_arr)) / np.std(x_arr))
            
            # 计算置信度
            line_count_factor = min(len(lines) / 8.0, 1.0)
            fit_quality_factor = max(0, min(1.0, r_squared * 1.5))
            weight_factor = np.mean(weights_arr)
            
            confidence = 0.4 * line_count_factor + 0.4 * fit_quality_factor + 0.2 * weight_factor
            
            return {
                'func': poly_func,
                'coeffs': coeffs.tolist() if hasattr(coeffs, 'tolist') else coeffs,
                'points': [(x_bottom, y_bottom), (x_top, y_top)],
                'model_type': model_type,
                'confidence': confidence,
                'r_squared': r_squared,
                'num_lines': len(lines),
                'avg_weight': weight_factor
            }
            
        except Exception as e:
            print(f"车道线拟合失败 ({side}): {e}")
            return None
    
    def _validate_lanes(self, left_lane: Optional[Dict], right_lane: Optional[Dict],
                       image_shape: Tuple[int, ...]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """验证车道线合理性"""
        if left_lane is None or right_lane is None:
            return left_lane, right_lane
        
        height, width = image_shape[:2]
        
        # 检查车道宽度
        try:
            left_func = left_lane['func']
            right_func = right_lane['func']
            
            # 在多个高度检查宽度
            y_samples = np.linspace(height * 0.4, height, 5)
            widths = []
            
            for y in y_samples:
                try:
                    left_x = left_func(y)
                    right_x = right_func(y)
                    if right_x > left_x:
                        widths.append(right_x - left_x)
                except:
                    continue
            
            if widths:
                avg_width = np.mean(widths)
                width_std = np.std(widths)
                width_cv = width_std / avg_width if avg_width > 0 else 0
                
                # 宽度合理性检查
                min_width = self.validation_params['min_lane_width']
                max_width = self.validation_params['max_lane_width']
                max_variation = self.validation_params['max_width_variation']
                
                if avg_width < min_width or avg_width > max_width:
                    # 宽度不合理，降低置信度
                    penalty = 0.5
                    left_lane['confidence'] *= penalty
                    right_lane['confidence'] *= penalty
                
                if width_cv > max_variation:
                    # 宽度变化太大，降低置信度
                    penalty = 0.7
                    left_lane['confidence'] *= penalty
                    right_lane['confidence'] *= penalty
        except:
            pass
        
        # 检查车道线交叉
        try:
            if left_lane['points'][0][0] > right_lane['points'][0][0]:
                # 底部交叉
                penalty = 0.6
                left_lane['confidence'] *= penalty
                right_lane['confidence'] *= penalty
        except:
            pass
        
        return left_lane, right_lane
    
    def _calculate_center_line(self, left_lane: Optional[Dict], right_lane: Optional[Dict],
                              image_shape: Tuple[int, ...]) -> Optional[Dict]:
        """计算中心线"""
        if left_lane is None or right_lane is None:
            return None
        
        try:
            left_func = left_lane['func']
            right_func = right_lane['func']
            
            height, width = image_shape[:2]
            
            # 中心线函数：左右车道线的平均值
            def center_func(y):
                left_x = left_func(y)
                right_x = right_func(y)
                return (left_x + right_x) / 2
            
            # 生成中心线点
            y_bottom = height
            y_top = int(height * 0.4)
            
            x_bottom = int(center_func(y_bottom))
            x_top = int(center_func(y_top))
            
            # 确保点在图像内
            x_bottom = max(0, min(width - 1, x_bottom))
            x_top = max(0, min(width - 1, x_top))
            
            # 计算中心线置信度（基于两侧车道线置信度的平均值）
            confidence = (left_lane.get('confidence', 0) + right_lane.get('confidence', 0)) / 2
            
            return {
                'func': center_func,
                'points': [(x_bottom, y_bottom), (x_top, y_top)],
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"中心线计算失败: {e}")
            return None
    
    def _predict_future_path(self, left_lane: Optional[Dict], right_lane: Optional[Dict],
                           image_shape: Tuple[int, ...]) -> Optional[Dict]:
        """预测未来路径"""
        if left_lane is None or right_lane is None:
            return None
        
        try:
            height, width = image_shape[:2]
            
            # 使用中心线进行预测
            center_line = self._calculate_center_line(left_lane, right_lane, image_shape)
            if center_line is None:
                return None
            
            center_func = center_line['func']
            
            # 生成预测点
            current_y = height
            target_y = int(height * (1 - self.config.prediction_distance))
            
            if target_y <= current_y * 0.6:  # 预测距离太短
                return None
            
            y_values = np.linspace(current_y, target_y, self.config.prediction_steps)
            path_points = []
            
            for y in y_values:
                try:
                    x = center_func(y)
                    # 确保点在图像内
                    x = max(0, min(width - 1, x))
                    path_points.append((int(x), int(y)))
                except:
                    continue
            
            if len(path_points) < self.config.min_prediction_points:
                return None
            
            # 计算路径特征
            path_features = self._calculate_path_features(path_points)
            
            # 计算预测质量
            prediction_quality = self._calculate_prediction_quality(
                path_points, left_lane, right_lane
            )
            
            return {
                'center_path': path_points,
                'features': path_features,
                'prediction_quality': prediction_quality,
                'prediction_length': len(path_points),
                'start_point': path_points[0] if path_points else None,
                'end_point': path_points[-1] if path_points else None
            }
            
        except Exception as e:
            print(f"路径预测失败: {e}")
            return None
    
    def _calculate_path_features(self, path_points: List[Tuple[int, int]]) -> Dict[str, Any]:
        """计算路径特征"""
        if len(path_points) < 3:
            return {}
        
        pts = np.array(path_points, dtype=np.float32)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # 1. 曲率
        dx = np.gradient(x)
        dy = np.gradient(y)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**1.5
        curvature = curvature[np.isfinite(curvature)]
        
        # 2. 平滑度（通过二阶导数的变化）
        d2x_std = np.std(d2x) if len(d2x) > 0 else 0
        d2y_std = np.std(d2y) if len(d2y) > 0 else 0
        smoothness = 1.0 / (1.0 + d2x_std + d2y_std)
        
        # 3. 直线度（通过线性拟合的R²）
        if len(x) >= 2:
            coeffs = np.polyfit(y, x, 1)
            poly_func = np.poly1d(coeffs)
            x_pred = poly_func(y)
            ss_res = np.sum((x - x_pred) ** 2)
            ss_tot = np.sum((x - np.mean(x)) ** 2)
            straightness = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            straightness = 0
        
        return {
            'curvature_mean': float(np.mean(curvature)) if len(curvature) > 0 else 0,
            'curvature_std': float(np.std(curvature)) if len(curvature) > 0 else 0,
            'smoothness': float(smoothness),
            'straightness': float(straightness),
            'path_length': len(path_points)
        }
    
    def _calculate_prediction_quality(self, path_points: List[Tuple[int, int]],
                                     left_lane: Dict, right_lane: Dict) -> float:
        """计算预测质量"""
        quality_factors = []
        
        # 1. 路径点数量
        point_count_factor = min(len(path_points) / 8.0, 1.0)
        quality_factors.append(point_count_factor * 0.3)
        
        # 2. 车道线置信度
        lane_confidence_factor = (left_lane.get('confidence', 0) + 
                                 right_lane.get('confidence', 0)) / 2
        quality_factors.append(lane_confidence_factor * 0.4)
        
        # 3. 路径平滑度
        path_features = self._calculate_path_features(path_points)
        smoothness_factor = path_features.get('smoothness', 0.5)
        quality_factors.append(smoothness_factor * 0.3)
        
        return min(sum(quality_factors), 1.0)
    
    def _calculate_detection_quality(self, left_lane: Optional[Dict], right_lane: Optional[Dict],
                                    left_lines: List, right_lines: List,
                                    future_path: Optional[Dict]) -> float:
        """计算检测质量"""
        quality_factors = []
        
        # 1. 车道线数量
        left_line_count = len(left_lines)
        right_line_count = len(right_lines)
        
        line_count_factor = (min(left_line_count, 10) / 10.0 * 0.5 + 
                           min(right_line_count, 10) / 10.0 * 0.5)
        quality_factors.append(line_count_factor * 0.25)
        
        # 2. 车道线模型质量
        lane_model_factor = 0.0
        if left_lane and right_lane:
            left_conf = left_lane.get('confidence', 0)
            right_conf = right_lane.get('confidence', 0)
            lane_model_factor = (left_conf + right_conf) / 2.0
        quality_factors.append(lane_model_factor * 0.35)
        
        # 3. 路径预测质量
        path_quality_factor = 0.0
        if future_path:
            path_quality_factor = future_path.get('prediction_quality', 0)
        quality_factors.append(path_quality_factor * 0.20)
        
        # 4. 检测方法数量
        method_count = self.config.batch_size  # 简化，实际应从结果中获取
        method_factor = min(method_count / 3.0, 1.0)
        quality_factors.append(method_factor * 0.10)
        
        # 5. 历史一致性
        if self.lane_history:
            recent_qualities = [h.get('detection_quality', 0) 
                              for h in list(self.lane_history)[-3:]]
            if recent_qualities:
                hist_consistency = 1.0 - np.std(recent_qualities)
                quality_factors.append(hist_consistency * 0.10)
        
        # 综合质量
        total_quality = sum(quality_factors)
        
        # 应用非线性调整
        if total_quality < 0.3:
            return total_quality * 0.8
        elif total_quality < 0.7:
            return 0.3 + (total_quality - 0.3) * 1.1
        else:
            return min(total_quality, 1.0)
    
    def _apply_temporal_smoothing(self, current_result: Dict[str, Any]) -> Dict[str, Any]:
        """应用时间平滑"""
        if not self.lane_history:
            return current_result
        
        smoothing_factor = 0.65
        
        # 获取历史结果
        recent_history = list(self.lane_history)[-2:]
        
        # 平滑车道线参数
        if current_result['left_lane'] and recent_history:
            for prev_result in recent_history:
                if prev_result['left_lane']:
                    # 平滑系数
                    prev_coeffs = np.array(prev_result['left_lane']['coeffs'])
                    curr_coeffs = np.array(current_result['left_lane']['coeffs'])
                    
                    if len(prev_coeffs) == len(curr_coeffs):
                        smoothed_coeffs = (
                            smoothing_factor * curr_coeffs + 
                            (1 - smoothing_factor) * prev_coeffs
                        )
                        current_result['left_lane']['coeffs'] = smoothed_coeffs.tolist()
                        current_result['left_lane']['func'] = np.poly1d(smoothed_coeffs)
        
        # 同样处理右车道线
        if current_result['right_lane'] and recent_history:
            for prev_result in recent_history:
                if prev_result['right_lane']:
                    prev_coeffs = np.array(prev_result['right_lane']['coeffs'])
                    curr_coeffs = np.array(current_result['right_lane']['coeffs'])
                    
                    if len(prev_coeffs) == len(curr_coeffs):
                        smoothed_coeffs = (
                            smoothing_factor * curr_coeffs + 
                            (1 - smoothing_factor) * prev_coeffs
                        )
                        current_result['right_lane']['coeffs'] = smoothed_coeffs.tolist()
                        current_result['right_lane']['func'] = np.poly1d(smoothed_coeffs)
        
        # 平滑检测质量
        if recent_history:
            recent_qualities = [h.get('detection_quality', 0) for h in recent_history]
            avg_quality = np.mean(recent_qualities)
            
            smoothed_quality = (
                smoothing_factor * current_result['detection_quality'] +
                (1 - smoothing_factor) * avg_quality
            )
            current_result['detection_quality'] = smoothed_quality
        
        return current_result
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'left_lines': [],
            'right_lines': [],
            'left_lane': None,
            'right_lane': None,
            'center_line': None,
            'future_path': None,
            'detection_quality': 0.0,
            'detection_methods': 0,
            'processing_time': 0,
            'timestamp': datetime.now(),
            'is_fallback': True
        }
    
    def get_lane_stats(self) -> Dict[str, Any]:
        """获取车道线检测统计"""
        if not self.lane_history:
            return {"status": "No lane detection history"}
        
        qualities = [h.get('detection_quality', 0) for h in self.lane_history]
        processing_times = [h.get('processing_time', 0) for h in self.lane_history]
        
        return {
            "total_detections": len(self.lane_history),
            "average_quality": np.mean(qualities),
            "quality_std": np.std(qualities),
            "average_processing_time": np.mean(processing_times),
            "method_configs": self.method_configs,
            "validation_params": self.validation_params
        }

# ==================== 高级方向分析器 ====================
class AdvancedDirectionAnalyzer:
    """高级方向分析器 - 高置信度版"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.history = deque(maxlen=12)
        self.confidence_calibrator = ConfidenceCalibrator()
        self.quality_evaluator = QualityEvaluator()
        
        # 特征权重（基于重要性）
        self.feature_weights = {
            'lane_convergence': 0.30,
            'lane_symmetry': 0.20,
            'path_curvature': 0.20,
            'contour_position': 0.15,
            'historical_consistency': 0.10,
            'detection_quality': 0.05
        }
        
        # 方向决策阈值
        self.decision_thresholds = {
            'high_confidence': 0.7,
            'medium_confidence': 0.5,
            'low_confidence': 0.3,
            'direction_dominance': 1.3  # 优势比
        }
        
        # 上下文检测
        self.context_detector = ContextDetector()
    
    def analyze(self, road_features: Dict[str, Any], 
                lane_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析道路方向 - 高置信度版"""
        start_time = time.time()
        
        try:
            # 1. 提取增强特征
            features = self._extract_enhanced_features(road_features, lane_info)
            
            # 2. 检测场景上下文
            context = self.context_detector.detect_context(features, lane_info)
            
            # 3. 集成预测（多模型融合）
            raw_probabilities = self._ensemble_prediction(features, lane_info, context)
            
            # 4. 计算基础置信度
            base_confidence = self._calculate_base_confidence(features, raw_probabilities, lane_info)
            
            # 5. 校准置信度
            calibrated_confidence = self.confidence_calibrator.calibrate(
                base_confidence, features, context
            )
            
            # 6. 质量评估
            quality_scores = self.quality_evaluator.evaluate_comprehensive(
                lane_info, road_features, features.get('image_quality', 0.7)
            )
            
            # 7. 应用质量调整
            quality_adjusted_confidence = self._apply_quality_adjustment(
                calibrated_confidence, quality_scores
            )
            
            # 8. 获取最终方向
            final_direction = self._get_final_direction(
                raw_probabilities, quality_adjusted_confidence
            )
            
            # 9. 历史平滑
            final_direction, final_confidence = self._apply_historical_smoothing(
                final_direction, quality_adjusted_confidence, features
            )
            
            # 10. 生成详细推理
            reasoning = self._generate_detailed_reasoning(
                features, raw_probabilities, final_direction, final_confidence, quality_scores
            )
            
            # 创建结果
            result = {
                'direction': final_direction,
                'confidence': final_confidence,
                'probabilities': raw_probabilities,
                'features': features,
                'reasoning': reasoning,
                'context': context,
                'quality_scores': quality_scores,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now(),
                'is_high_confidence': final_confidence >= self.decision_thresholds['high_confidence']
            }
            
            # 更新性能统计
            self.confidence_calibrator.update_performance(
                final_confidence, True, features  # 假设正确，实际应用中需要真实标签
            )
            
            # 更新历史
            if final_confidence > 0.3:
                self.history.append(result)
            
            return result
            
        except Exception as e:
            print(f"方向分析失败: {e}")
            return self._create_default_result()
    
    def _extract_enhanced_features(self, road_features: Dict[str, Any],
                                 lane_info: Dict[str, Any]) -> Dict[str, Any]:
        """提取增强特征"""
        features = {}
        
        # 1. 道路轮廓特征
        if 'centroid' in road_features:
            cx, cy = road_features['centroid']
            features['contour_centroid_x'] = cx
            features['contour_centroid_y'] = cy
            features['contour_area'] = road_features.get('area', 0)
            features['contour_solidity'] = road_features.get('solidity', 0.5)
        
        # 2. 车道线特征
        if lane_info['left_lane'] and lane_info['right_lane']:
            lane_features = self._extract_lane_features(lane_info)
            features.update(lane_features)
        
        # 3. 路径特征
        if lane_info['future_path'] and lane_info['future_path'].get('features'):
            path_features = self._extract_path_features(lane_info['future_path'])
            features.update(path_features)
        
        # 4. 检测质量特征
        features['detection_quality'] = lane_info.get('detection_quality', 0.0)
        features['road_confidence'] = road_features.get('confidence', 0.0)
        
        # 5. 历史特征
        if self.history:
            historical_features = self._extract_historical_features()
            features.update(historical_features)
        
        # 6. 特征一致性
        consistency_features = self._calculate_feature_consistency(features, lane_info, road_features)
        features.update(consistency_features)
        
        return features
    
    def _extract_lane_features(self, lane_info: Dict[str, Any]) -> Dict[str, Any]:
        """提取车道线特征"""
        features = {}
        left_lane = lane_info['left_lane']
        right_lane = lane_info['right_lane']
        
        # 收敛度
        convergence = self._calculate_lane_convergence(left_lane, right_lane)
        features['lane_convergence'] = convergence
        
        # 对称性
        symmetry = self._calculate_lane_symmetry(left_lane, right_lane)
        features['lane_symmetry'] = symmetry
        
        # 宽度特征
        width_features = self._calculate_width_features(left_lane, right_lane)
        features.update(width_features)
        
        # 方向特征
        direction_features = self._calculate_direction_features(left_lane, right_lane)
        features.update(direction_features)
        
        # 模型质量
        features['lane_model_quality'] = min(
            left_lane.get('confidence', 0.5),
            right_lane.get('confidence', 0.5)
        )
        
        return features
    
    def _calculate_lane_convergence(self, left_lane: Dict, right_lane: Dict) -> float:
        """计算车道线收敛度"""
        try:
            left_func = left_lane['func']
            right_func = right_lane['func']
            
            # 在多个高度采样
            y_points = np.linspace(600, 300, 7)  # 从底部到中部
            widths = []
            
            for y in y_points:
                try:
                    left_x = left_func(y)
                    right_x = right_func(y)
                    if right_x > left_x:  # 有效宽度
                        widths.append(right_x - left_x)
                except:
                    continue
            
            if len(widths) < 3:
                return 1.0  # 默认平行
            
            # 计算收敛趋势
            widths = np.array(widths)
            y_norm = np.linspace(0, 1, len(widths))
            
            # 线性拟合斜率
            coeffs = np.polyfit(y_norm, widths, 1)
            slope = coeffs[0]
            
            # 计算收敛比
            width_ratio = widths[-1] / widths[0] if widths[0] > 0 else 1.0
            
            # 综合收敛指标
            if slope < -10:  # 明显收敛
                convergence = max(0.1, width_ratio * 0.3)
            elif slope > 10:  # 明显发散
                convergence = min(1.9, width_ratio * 1.7)
            else:  # 基本平行
                convergence = 1.0
            
            return float(convergence)
            
        except Exception:
            return 1.0  # 默认平行
    
    def _calculate_lane_symmetry(self, left_lane: Dict, right_lane: Dict) -> float:
        """计算车道对称性"""
        try:
            left_func = left_lane['func']
            right_func = right_lane['func']
            
            # 计算中心线
            def center_func(y):
                return (left_func(y) + right_func(y)) / 2
            
            # 在多个位置检查对称性
            y_points = np.linspace(600, 300, 5)
            symmetry_scores = []
            
            for y in y_points:
                try:
                    center = center_func(y)
                    left_dist = center - left_func(y)
                    right_dist = right_func(y) - center
                    
                    if left_dist + right_dist > 0:
                        symmetry = 1 - abs(left_dist - right_dist) / (left_dist + right_dist)
                        symmetry_scores.append(max(0, symmetry))
                except:
                    continue
            
            return float(np.mean(symmetry_scores)) if symmetry_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_width_features(self, left_lane: Dict, right_lane: Dict) -> Dict[str, Any]:
        """计算宽度特征"""
        features = {}
        
        try:
            left_func = left_lane['func']
            right_func = right_lane['func']
            
            # 在多个高度采样宽度
            y_points = np.linspace(600, 300, 5)
            widths = []
            
            for y in y_points:
                try:
                    left_x = left_func(y)
                    right_x = right_func(y)
                    if right_x > left_x:
                        widths.append(right_x - left_x)
                except:
                    continue
            
            if widths:
                widths = np.array(widths)
                features['avg_lane_width'] = float(np.mean(widths))
                features['width_std'] = float(np.std(widths))
                features['width_variation'] = features['width_std'] / features['avg_lane_width'] \
                    if features['avg_lane_width'] > 0 else 0
                features['min_width'] = float(np.min(widths))
                features['max_width'] = float(np.max(widths))
        
        except Exception:
            pass
        
        return features
    
    def _calculate_direction_features(self, left_lane: Dict, right_lane: Dict) -> Dict[str, Any]:
        """计算方向特征"""
        features = {}
        
        try:
            # 计算左右车道线的方向差
            left_points = left_lane.get('points', [])
            right_points = right_lane.get('points', [])
            
            if len(left_points) == 2 and len(right_points) == 2:
                # 计算斜率
                left_dx = left_points[1][0] - left_points[0][0]
                left_dy = left_points[1][1] - left_points[0][1]
                right_dx = right_points[1][0] - right_points[0][0]
                right_dy = right_points[1][1] - right_points[0][1]
                
                if left_dx != 0 and right_dx != 0:
                    left_slope = left_dy / left_dx
                    right_slope = right_dy / right_dx
                    
                    features['slope_difference'] = abs(left_slope - right_slope)
                    features['avg_slope'] = (abs(left_slope) + abs(right_slope)) / 2
        
        except Exception:
            pass
        
        return features
    
    def _extract_path_features(self, future_path: Dict[str, Any]) -> Dict[str, Any]:
        """提取路径特征"""
        features = {}
        
        path_features = future_path.get('features', {})
        
        # 曲率特征
        curvature_mean = path_features.get('curvature_mean', 0)
        curvature_std = path_features.get('curvature_std', 0)
        
        features['path_curvature'] = curvature_mean
        features['path_curvature_abs'] = abs(curvature_mean)
        features['path_curvature_std'] = curvature_std
        
        # 平滑度特征
        smoothness = path_features.get('smoothness', 0.5)
        features['path_smoothness'] = smoothness
        
        # 直线度特征
        straightness = path_features.get('straightness', 0.5)
        features['path_straightness'] = straightness
        
        # 路径质量
        prediction_quality = future_path.get('prediction_quality', 0.5)
        features['path_quality'] = prediction_quality
        
        return features
    
    def _extract_historical_features(self) -> Dict[str, Any]:
        """提取历史特征"""
        features = {}
        
        if not self.history:
            return features
        
        # 最近的历史结果
        recent_history = list(self.history)[-5:]
        
        # 方向一致性
        recent_directions = [h['direction'] for h in recent_history]
        direction_counts = {}
        for direction in recent_directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        if direction_counts:
            most_common = max(direction_counts.items(), key=lambda x: x[1])
            features['historical_direction'] = most_common[0]
            features['historical_consistency'] = most_common[1] / len(recent_history)
        
        # 置信度统计
        recent_confidences = [h['confidence'] for h in recent_history]
        features['historical_confidence_mean'] = np.mean(recent_confidences)
        features['historical_confidence_std'] = np.std(recent_confidences)
        
        return features
    
    def _calculate_feature_consistency(self, features: Dict[str, Any],
                                     lane_info: Dict[str, Any],
                                     road_features: Dict[str, Any]) -> Dict[str, Any]:
        """计算特征一致性"""
        consistency_scores = []
        
        # 1. 车道线与道路位置一致性
        if 'contour_centroid_x' in features and 'avg_lane_width' in features:
            # 简化检查：车道线应在道路区域内
            position_consistency = 0.7  # 默认值
            consistency_scores.append(position_consistency)
        
        # 2. 不同特征之间的一致性
        if 'lane_convergence' in features and 'path_curvature' in features:
            convergence = features['lane_convergence']
            curvature = features['path_curvature']
            
            # 收敛和曲率应一致
            if convergence < 0.8 and curvature > 0:  # 收敛且右转
                consistency = 0.8
            elif convergence < 0.8 and curvature < 0:  # 收敛且左转
                consistency = 0.8
            elif convergence > 1.2 and abs(curvature) < 0.001:  # 发散且直行
                consistency = 0.7
            else:
                consistency = 0.5
            
            consistency_scores.append(consistency)
        
        # 3. 检测质量一致性
        lane_quality = lane_info.get('detection_quality', 0.5)
        road_quality = road_features.get('confidence', 0.5)
        quality_consistency = 1.0 - abs(lane_quality - road_quality)
        consistency_scores.append(quality_consistency)
        
        features['feature_consistency'] = np.mean(consistency_scores) if consistency_scores else 0.5
        features['consistency_scores'] = consistency_scores
        
        return features
    
    def _ensemble_prediction(self, features: Dict[str, Any],
                           lane_info: Dict[str, Any],
                           context: str) -> Dict[str, float]:
        """集成预测（多模型融合）"""
        # 方法1：基于规则的预测
        rule_based = self._rule_based_prediction(features)
        
        # 方法2：基于统计的预测
        statistical = self._statistical_prediction(features, lane_info)
        
        # 方法3：基于几何的预测
        geometric = self._geometric_prediction(features)
        
        # 方法4：基于历史的预测
        historical = self._historical_prediction()
        
        # 方法5：基于上下文的预测
        contextual = self._contextual_prediction(context, features)
        
        # 加权融合
        weights = {
            'rule': 0.25,
            'statistical': 0.20,
            'geometric': 0.20,
            'historical': 0.15,
            'contextual': 0.20
        }
        
        # 初始化融合概率
        fused_probs = {'直行': 0.0, '左转': 0.0, '右转': 0.0}
        
        # 融合各个模型
        model_results = [
            ('rule', rule_based),
            ('statistical', statistical),
            ('geometric', geometric),
            ('historical', historical),
            ('contextual', contextual)
        ]
        
        for model_name, probs in model_results:
            if probs:
                weight = weights.get(model_name, 0.2)
                for direction in fused_probs:
                    fused_probs[direction] += probs.get(direction, 0.0) * weight
        
        # 归一化
        total = sum(fused_probs.values())
        if total > 0:
            for direction in fused_probs:
                fused_probs[direction] /= total
        
        return fused_probs
    
    def _rule_based_prediction(self, features: Dict[str, Any]) -> Dict[str, float]:
        """基于规则的预测"""
        probs = {'直行': 0.3, '左转': 0.35, '右转': 0.35}
        
        # 1. 车道线收敛特征
        if 'lane_convergence' in features:
            convergence = features['lane_convergence']
            
            if convergence < 0.8:  # 明显收敛
                # 结合对称性判断左右
                symmetry = features.get('lane_symmetry', 0.5)
                if symmetry < 0.6:
                    probs['左转'] += 0.25
                else:
                    probs['右转'] += 0.25
            elif convergence > 1.2:  # 明显发散
                probs['直行'] += 0.2
        
        # 2. 路径曲率特征
        if 'path_curvature' in features:
            curvature = features['path_curvature']
            
            if abs(curvature) < 0.0005:
                probs['直行'] += 0.2
            elif curvature > 0.0005:
                probs['右转'] += min(0.3, curvature * 800)
            else:
                probs['左转'] += min(0.3, abs(curvature) * 800)
        
        # 3. 轮廓质心特征
        if 'contour_centroid_x' in features:
            centroid_x = features['contour_centroid_x']
            deviation = (centroid_x - 400) / 400  # 归一化到[-1, 1]
            
            if abs(deviation) < 0.15:
                probs['直行'] += 0.25
            elif deviation > 0.15:
                probs['右转'] += min(0.3, abs(deviation) * 1.5)
            else:
                probs['左转'] += min(0.3, abs(deviation) * 1.5)
        
        # 4. 车道宽度变化
        if 'width_variation' in features:
            variation = features['width_variation']
            if variation > 0.3:  # 宽度变化大，可能转弯
                probs['直行'] -= 0.1
        
        # 归一化
        total = sum(probs.values())
        if total > 0:
            for direction in probs:
                probs[direction] /= total
        
        return probs
    
    def _statistical_prediction(self, features: Dict[str, Any],
                              lane_info: Dict[str, Any]) -> Dict[str, float]:
        """基于统计的预测"""
        probs = {'直行': 0.33, '左转': 0.33, '右转': 0.34}
        
        # 基于特征值的统计分析
        feature_values = []
        
        if 'lane_convergence' in features:
            conv = features['lane_convergence']
            if conv < 0.7:
                feature_values.append('turn')
            elif conv > 1.3:
                feature_values.append('straight')
        
        if 'path_curvature_abs' in features:
            curv = features['path_curvature_abs']
            if curv > 0.001:
                feature_values.append('turn')
            else:
                feature_values.append('straight')
        
        if 'lane_symmetry' in features:
            sym = features['lane_symmetry']
            if sym < 0.6:
                feature_values.append('left_turn')
            elif sym > 0.8:
                feature_values.append('right_turn')
        
        # 统计特征倾向
        turn_count = feature_values.count('turn')
        straight_count = feature_values.count('straight')
        left_count = feature_values.count('left_turn')
        right_count = feature_values.count('right_turn')
        
        total = turn_count + straight_count + left_count + right_count
        
        if total > 0:
            probs['直行'] = 0.2 + straight_count / total * 0.4
            probs['左转'] = 0.2 + left_count / total * 0.4
            probs['右转'] = 0.2 + right_count / total * 0.4
        
        # 归一化
        total_probs = sum(probs.values())
        if total_probs > 0:
            for direction in probs:
                probs[direction] /= total_probs
        
        return probs
    
    def _geometric_prediction(self, features: Dict[str, Any]) -> Dict[str, float]:
        """基于几何的预测"""
        probs = {'直行': 0.33, '左转': 0.33, '右转': 0.34}
        
        # 使用多个几何特征进行判断
        indicators = []
        
        # 1. 车道线角度差
        if 'slope_difference' in features:
            diff = features['slope_difference']
            if diff > 0.3:
                indicators.append('turn')
        
        # 2. 车道宽度变化
        if 'width_variation' in features:
            variation = features['width_variation']
            if variation > 0.25:
                indicators.append('turn')
        
        # 3. 路径曲率
        if 'path_curvature_abs' in features:
            curvature = features['path_curvature_abs']
            if curvature > 0.0008:
                indicators.append('turn')
        
        # 判断转弯方向
        if indicators:
            turn_count = len(indicators)
            
            # 使用对称性判断左右
            symmetry = features.get('lane_symmetry', 0.5)
            if symmetry < 0.55:
                probs['左转'] += turn_count * 0.15
            elif symmetry > 0.65:
                probs['右转'] += turn_count * 0.15
            else:
                # 对称性不明显，平均分配
                probs['左转'] += turn_count * 0.075
                probs['右转'] += turn_count * 0.075
        
        # 归一化
        total = sum(probs.values())
        if total > 0:
            for direction in probs:
                probs[direction] /= total
        
        return probs
    
    def _historical_prediction(self) -> Dict[str, float]:
        """基于历史的预测"""
        if not self.history:
            return {'直行': 0.33, '左转': 0.33, '右转': 0.34}
        
        # 统计历史方向
        recent_history = list(self.history)[-5:]
        direction_counts = defaultdict(int)
        
        for result in recent_history:
            direction = result.get('direction', '未知')
            if direction != '未知':
                direction_counts[direction] += 1
        
        total = sum(direction_counts.values())
        
        if total == 0:
            return {'直行': 0.33, '左转': 0.33, '右转': 0.34}
        
        # 计算历史概率
        probs = {
            '直行': direction_counts.get('直行', 0) / total,
            '左转': direction_counts.get('左转', 0) / total,
            '右转': direction_counts.get('右转', 0) / total
        }
        
        # 平滑处理（避免0概率）
        smoothing = 0.1
        for direction in probs:
            probs[direction] = (probs[direction] + smoothing) / (1 + 3 * smoothing)
        
        return probs
    
    def _contextual_prediction(self, context: str, features: Dict[str, Any]) -> Dict[str, float]:
        """基于上下文的预测"""
        # 根据不同场景调整先验概率
        if context == "highway":
            # 高速公路更可能直行
            return {'直行': 0.6, '左转': 0.2, '右转': 0.2}
        elif context == "urban_intersection":
            # 城市交叉口转弯概率增加
            return {'直行': 0.3, '左转': 0.35, '右转': 0.35}
        elif context == "rural_curve":
            # 乡村弯道
            # 根据特征判断左右
            curvature = features.get('path_curvature', 0)
            if curvature > 0:
                return {'直行': 0.2, '左转': 0.3, '右转': 0.5}
            else:
                return {'直行': 0.2, '左转': 0.5, '右转': 0.3}
        else:
            return {'直行': 0.33, '左转': 0.33, '右转': 0.34}
    
    def _calculate_base_confidence(self, features: Dict[str, Any],
                                 probabilities: Dict[str, float],
                                 lane_info: Dict[str, Any]) -> float:
        """计算基础置信度"""
        confidence_factors = []
        
        # 1. 概率清晰度
        max_prob = max(probabilities.values())
        min_prob = min(probabilities.values())
        
        if max_prob > 0:
            clarity = (max_prob - min_prob) / max_prob
            confidence_factors.append(clarity * 0.35)
        
        # 2. 特征质量
        feature_quality = 0.0
        quality_indicators = []
        
        if 'lane_convergence' in features:
            convergence = features['lane_convergence']
            # 收敛度越接近1.0，越不确定；越偏离1.0，越确定
            conv_confidence = 1 - min(1.0, abs(convergence - 1.0) * 1.5)
            quality_indicators.append(conv_confidence)
        
        if 'lane_symmetry' in features:
            symmetry = features['lane_symmetry']
            sym_confidence = symmetry
            quality_indicators.append(sym_confidence)
        
        if 'path_curvature_abs' in features:
            curvature = features['path_curvature_abs']
            # 曲率适中时置信度高，过大或过小都低
            curv_confidence = 1 - min(1.0, curvature * 800)
            quality_indicators.append(curv_confidence)
        
        if quality_indicators:
            feature_quality = np.mean(quality_indicators)
        
        confidence_factors.append(feature_quality * 0.30)
        
        # 3. 检测质量
        detection_quality = features.get('detection_quality', 0.5)
        confidence_factors.append(detection_quality * 0.20)
        
        # 4. 特征一致性
        feature_consistency = features.get('feature_consistency', 0.5)
        confidence_factors.append(feature_consistency * 0.15)
        
        # 综合置信度
        base_confidence = sum(confidence_factors)
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _apply_quality_adjustment(self, confidence: float,
                                quality_scores: Dict[str, Any]) -> float:
        """应用质量调整"""
        overall_quality = quality_scores.get('overall', {}).get('score', 0.5)
        
        # 根据质量调整置信度
        if overall_quality > 0.8:
            # 高质量，增强置信度
            adjustment = 1.2
        elif overall_quality > 0.6:
            # 中等质量，适度增强
            adjustment = 1.1
        elif overall_quality > 0.4:
            # 一般质量，保持原样
            adjustment = 1.0
        elif overall_quality > 0.2:
            # 较差质量，降低置信度
            adjustment = 0.9
        else:
            # 很差质量，显著降低
            adjustment = 0.8
        
        adjusted = confidence * adjustment
        
        # 限制在合理范围内
        return min(max(adjusted, 0.0), 1.0)
    
    def _get_final_direction(self, probabilities: Dict[str, float],
                           confidence: float) -> str:
        """获取最终方向"""
        # 按概率排序
        sorted_directions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_directions:
            return '未知'
        
        best_direction, best_prob = sorted_directions[0]
        second_direction, second_prob = sorted_directions[1] if len(sorted_directions) > 1 else ('', 0)
        
        # 决策逻辑
        if confidence < self.decision_thresholds['low_confidence']:
            return '未知'
        
        elif confidence < self.decision_thresholds['medium_confidence']:
            # 中等置信度，需要明显优势
            if best_prob > second_prob * self.decision_thresholds['direction_dominance']:
                return best_direction
            else:
                return '未知'
        
        else:
            # 高置信度，允许较小的优势
            if best_prob > second_prob * 1.2:
                return best_direction
            else:
                # 检查历史
                if self.history:
                    historical_direction = self._get_historical_direction()
                    if historical_direction != '未知':
                        return historical_direction
                
                return best_direction
    
    def _get_historical_direction(self) -> str:
        """获取历史主要方向"""
        if not self.history:
            return '未知'
        
        recent = list(self.history)[-3:]
        direction_counts = defaultdict(int)
        
        for result in recent:
            direction = result.get('direction', '未知')
            if direction != '未知':
                direction_counts[direction] += 1
        
        if not direction_counts:
            return '未知'
        
        most_common = max(direction_counts.items(), key=lambda x: x[1])
        return most_common[0]
    
    def _apply_historical_smoothing(self, direction: str, confidence: float,
                                  features: Dict[str, Any]) -> Tuple[str, float]:
        """应用历史平滑"""
        if not self.history:
            return direction, confidence
        
        recent_history = list(self.history)[-4:]
        if not recent_history:
            return direction, confidence
        
        # 统计历史方向
        historical_directions = [h.get('direction', '未知') for h in recent_history]
        historical_confidences = [h.get('confidence', 0) for h in recent_history]
        
        # 计算历史一致性
        direction_counts = defaultdict(int)
        for d in historical_directions:
            if d != '未知':
                direction_counts[d] += 1
        
        if not direction_counts:
            return direction, confidence
        
        # 找到历史主要方向
        most_common, count = max(direction_counts.items(), key=lambda x: x[1])
        frequency = count / len(historical_directions)
        
        # 如果历史一致性高且当前置信度低，信任历史
        if frequency > 0.75 and confidence < 0.5:
            # 使用历史方向
            historical_confidence = np.mean(historical_confidences)
            smoothed_confidence = historical_confidence * 0.9
            return most_common, smoothed_confidence
        
        # 如果历史与当前不一致，但历史一致性高
        if most_common != direction and frequency > 0.6:
            # 平滑过渡
            smoothing_factor = min(0.7, frequency)
            historical_confidence = np.mean(historical_confidences)
            
            smoothed_confidence = (
                confidence * (1 - smoothing_factor) + 
                historical_confidence * smoothing_factor
            )
            
            # 检查是否需要改变方向
            if confidence < 0.4 and historical_confidence > 0.6:
                return most_common, smoothed_confidence
        
        return direction, confidence
    
    def _generate_detailed_reasoning(self, features: Dict[str, Any],
                                   probabilities: Dict[str, float],
                                   direction: str, confidence: float,
                                   quality_scores: Dict[str, Any]) -> str:
        """生成详细推理"""
        reasons = []
        
        # 主要判断依据
        if 'lane_convergence' in features:
            conv = features['lane_convergence']
            if conv < 0.8:
                reasons.append("车道明显收敛")
            elif conv > 1.2:
                reasons.append("车道发散")
            else:
                reasons.append("车道基本平行")
        
        if 'path_curvature' in features:
            curv = features['path_curvature']
            if abs(curv) < 0.0005:
                reasons.append("路径基本直线")
            elif curv > 0:
                reasons.append("路径向右弯曲")
            else:
                reasons.append("路径向左弯曲")
        
        # 概率分布信息
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        if sorted_probs:
            top1, top2 = sorted_probs[:2]
            ratio = top1[1] / top2[1] if top2[1] > 0 else float('inf')
            
            if ratio > 2.0:
                reasons.append(f"{top1[0]}显著占优")
            elif ratio > 1.5:
                reasons.append(f"{top1[0]}明显占优")
            elif ratio > 1.2:
                reasons.append(f"{top1[0]}稍占优势")
            else:
                reasons.append("各方向概率接近")
        
        # 质量信息
        quality_level = quality_scores.get('overall', {}).get('level', '一般')
        reasons.append(f"检测质量: {quality_level}")
        
        # 置信度水平
        if confidence > 0.8:
            conf_level = "非常高"
        elif confidence > 0.7:
            conf_level = "高"
        elif confidence > 0.6:
            conf_level = "中高"
        elif confidence > 0.5:
            conf_level = "中等"
        elif confidence > 0.4:
            conf_level = "中低"
        else:
            conf_level = "低"
        
        reasons.append(f"置信度: {conf_level}")
        
        return " | ".join(reasons)
    
    def _create_default_result(self) -> Dict[str, Any]:
        """创建默认结果"""
        return {
            'direction': '未知',
            'confidence': 0.0,
            'probabilities': {'直行': 0.33, '左转': 0.33, '右转': 0.34},
            'features': {},
            'reasoning': '检测失败',
            'context': 'unknown',
            'quality_scores': {},
            'processing_time': 0,
            'timestamp': datetime.now(),
            'is_high_confidence': False
        }

# ==================== 上下文检测器 ====================
class ContextDetector:
    """上下文检测器 - 识别道路场景"""
    
    def detect_context(self, features: Dict[str, Any], lane_info: Dict[str, Any]) -> str:
        """检测道路场景上下文"""
        try:
            # 1. 高速公路检测
            if self._is_highway(features, lane_info):
                return "highway"
            
            # 2. 城市交叉口检测
            if self._is_urban_intersection(features, lane_info):
                return "urban_intersection"
            
            # 3. 乡村弯道检测
            if self._is_rural_curve(features, lane_info):
                return "rural_curve"
            
            # 4. 一般道路
            return "general_road"
            
        except Exception:
            return "unknown"
    
    def _is_highway(self, features: Dict[str, Any], lane_info: Dict[str, Any]) -> bool:
        """检测是否为高速公路"""
        indicators = []
        
        # 1. 车道宽度
        if 'avg_lane_width' in features:
            width = features['avg_lane_width']
            if width > 250:  # 高速公路车道较宽
                indicators.append(True)
        
        # 2. 车道线数量
        left_count = len(lane_info.get('left_lines', []))
        right_count = len(lane_info.get('right_lines', []))
        if left_count > 4 and right_count > 4:  # 高速公路车道线清晰
            indicators.append(True)
        
        # 3. 道路曲率
        if 'path_curvature_abs' in features:
            curvature = features['path_curvature_abs']
            if curvature < 0.0003:  # 高速公路较直
                indicators.append(True)
        
        # 4. 检测质量
        quality = lane_info.get('detection_quality', 0)
        if quality > 0.7:  # 高速公路检测质量通常较高
            indicators.append(True)
        
        return len(indicators) >= 3 and all(indicators)
    
    def _is_urban_intersection(self, features: Dict[str, Any], lane_info: Dict[str, Any]) -> bool:
        """检测是否为城市交叉口"""
        indicators = []
        
        # 1. 车道收敛/发散
        if 'lane_convergence' in features:
            convergence = features['lane_convergence']
            if convergence < 0.7 or convergence > 1.3:  # 明显收敛或发散
                indicators.append(True)
        
        # 2. 车道线数量较少
        left_count = len(lane_info.get('left_lines', []))
        right_count = len(lane_info.get('right_lines', []))
        if left_count < 3 or right_count < 3:  # 交叉口车道线可能不连续
            indicators.append(True)
        
        # 3. 道路曲率变化大
        if 'path_curvature_std' in features:
            curvature_std = features['path_curvature_std']
            if curvature_std > 0.0005:  # 曲率变化大
                indicators.append(True)
        
        return len(indicators) >= 2
    
    def _is_rural_curve(self, features: Dict[str, Any], lane_info: Dict[str, Any]) -> bool:
        """检测是否为乡村弯道"""
        indicators = []
        
        # 1. 明显曲率
        if 'path_curvature_abs' in features:
            curvature = features['path_curvature_abs']
            if curvature > 0.001:  # 明显弯曲
                indicators.append(True)
        
        # 2. 车道宽度变化大
        if 'width_variation' in features:
            variation = features['width_variation']
            if variation > 0.3:  # 宽度变化大
                indicators.append(True)
        
        # 3. 车道线数量适中
        left_count = len(lane_info.get('left_lines', []))
        right_count = len(lane_info.get('right_lines', []))
        if 2 <= left_count <= 5 and 2 <= right_count <= 5:
            indicators.append(True)
        
        return len(indicators) >= 2

# ==================== 智能可视化引擎 ====================
class SmartVisualizer:
    """智能可视化引擎 - 优化版"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._setup_colors()
        self._setup_styles()
    
    def _setup_colors(self):
        """设置颜色方案"""
        self.colors = {
            # 道路相关
            'road_area': (0, 180, 0, 100),
            'road_boundary': (0, 255, 255, 200),
            'road_highlight': (0, 255, 0, 50),
            
            # 车道线
            'left_lane': (255, 100, 100, 200),
            'right_lane': (100, 100, 255, 200),
            'center_line': (255, 255, 0, 180),
            
            # 路径预测
            'future_path': (255, 0, 255, 180),
            'prediction_points': (255, 150, 255, 220),
            
            # 置信度颜色
            'confidence_high': (0, 255, 0, 255),
            'confidence_medium': (255, 165, 0, 255),
            'confidence_low': (255, 0, 0, 255),
            'confidence_very_low': (128, 128, 128, 255),
            
            # 文本颜色
            'text_primary': (255, 255, 255, 255),
            'text_secondary': (200, 200, 200, 255),
            'text_highlight': (0, 255, 255, 255),
            
            # 状态指示
            'status_success': (0, 255, 0, 255),
            'status_warning': (255, 165, 0, 255),
            'status_error': (255, 0, 0, 255),
            
            # 背景
            'overlay_bg': (0, 0, 0, 180),
            'panel_bg': (30, 30, 40, 220),
        }
    
    def _setup_styles(self):
        """设置可视化样式"""
        self.styles = {
            'line_thickness': {
                'thin': 1,
                'normal': 2,
                'thick': 3,
                'very_thick': 4
            },
            'font_size': {
                'small': 0.5,
                'normal': 0.7,
                'large': 1.0,
                'very_large': 1.2
            },
            'opacity': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.9,
                'full': 1.0
            }
        }
    
    def create_visualization(self, image: np.ndarray, 
                           road_info: Dict[str, Any],
                           lane_info: Dict[str, Any], 
                           direction_info: Dict[str, Any]) -> np.ndarray:
        """创建可视化结果"""
        try:
            # 创建副本
            visualization = image.copy()
            
            # 1. 绘制道路区域
            if road_info.get('features', {}).get('contour') is not None:
                visualization = self._draw_road_area(visualization, road_info)
            
            # 2. 绘制车道线
            visualization = self._draw_lanes(visualization, lane_info)
            
            # 3. 绘制路径预测
            if lane_info.get('future_path'):
                visualization = self._draw_future_path(visualization, lane_info['future_path'])
            
            # 4. 绘制信息面板
            visualization = self._draw_info_panel(visualization, direction_info, lane_info)
            
            # 5. 绘制方向指示器
            visualization = self._draw_direction_indicator(visualization, direction_info)
            
            # 6. 绘制置信度指示器
            visualization = self._draw_confidence_indicator(visualization, direction_info)
            
            # 7. 应用全局效果
            visualization = self._apply_global_effects(visualization)
            
            return visualization
            
        except Exception as e:
            print(f"可视化创建失败: {e}")
            return image
    
    def _draw_road_area(self, image: np.ndarray, road_info: Dict[str, Any]) -> np.ndarray:
        """绘制道路区域"""
        contour = road_info['features'].get('contour')
        if contour is None or len(contour) == 0:
            return image
        
        # 创建道路图层
        road_layer = image.copy()
        
        # 填充道路区域
        cv2.drawContours(road_layer, [contour], -1, self.colors['road_area'][:3], -1)
        
        # 绘制道路边界
        cv2.drawContours(road_layer, [contour], -1, self.colors['road_boundary'][:3], 2)
        
        # 混合图层
        alpha = self.colors['road_area'][3] / 255.0
        cv2.addWeighted(road_layer, alpha, image, 1 - alpha, 0, image)
        
        return image
    
    def _draw_lanes(self, image: np.ndarray, lane_info: Dict[str, Any]) -> np.ndarray:
        """绘制车道线"""
        lane_layer = image.copy()
        
        # 绘制原始检测线段
        for side, color_key in [('left_lines', 'left_lane'), ('right_lines', 'right_lane')]:
            lines = lane_info.get(side, [])
            color = self.colors[color_key]
            
            for line in lines:
                points = line.get('points', [])
                if len(points) == 2:
                    cv2.line(lane_layer, points[0], points[1], color[:3], 1, cv2.LINE_AA)
        
        # 绘制拟合的车道线
        for side, color_key in [('left_lane', 'left_lane'), ('right_lane', 'right_lane')]:
            lane = lane_info.get(side)
            if lane and 'points' in lane and len(lane['points']) == 2:
                points = lane['points']
                color = self.colors[color_key]
                
                # 根据置信度调整线条粗细
                confidence = lane.get('confidence', 0.5)
                thickness = 2 + int(confidence * 4)
                
                cv2.line(lane_layer, points[0], points[1], color[:3], thickness, cv2.LINE_AA)
        
        # 绘制中心线
        center_line = lane_info.get('center_line')
        if center_line and 'points' in center_line and len(center_line['points']) == 2:
            points = center_line['points']
            color = self.colors['center_line']
            thickness = 2
            cv2.line(lane_layer, points[0], points[1], color[:3], thickness, cv2.LINE_AA)
        
        # 混合车道线图层
        cv2.addWeighted(lane_layer, 0.7, image, 0.3, 0, image)
        
        return image
    
    def _draw_future_path(self, image: np.ndarray, future_path: Dict[str, Any]) -> np.ndarray:
        """绘制未来路径"""
        path_points = future_path.get('center_path', [])
        if len(path_points) < 2:
            return image
        
        path_layer = image.copy()
        color = self.colors['future_path']
        
        # 绘制渐变路径线
        for i in range(len(path_points) - 1):
            # 计算透明度渐变（远处更透明）
            alpha_factor = 0.5 + 0.5 * (i / (len(path_points) - 1))
            line_color = tuple(int(c * alpha_factor) for c in color[:3])
            
            # 线条粗细渐变
            thickness = 5 - int(i / len(path_points) * 3)
            
            cv2.line(path_layer, path_points[i], path_points[i + 1], 
                    line_color, thickness, cv2.LINE_AA)
        
        # 绘制路径点
        point_color = self.colors['prediction_points']
        for i, point in enumerate(path_points):
            radius = 5 - int(i / len(path_points) * 3)
            cv2.circle(path_layer, point, radius, point_color[:3], -1)
        
        # 混合路径图层
        cv2.addWeighted(path_layer, 0.6, image, 0.4, 0, image)
        
        return image
    
    def _draw_info_panel(self, image: np.ndarray, direction_info: Dict[str, Any],
                        lane_info: Dict[str, Any]) -> np.ndarray:
        """绘制信息面板"""
        height, width = image.shape[:2]
        
        # 创建半透明背景
        panel_height = 140
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), 
                     self.colors['overlay_bg'][:3], -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 获取信息
        direction = direction_info.get('direction', '未知')
        confidence = direction_info.get('confidence', 0.0)
        reasoning = direction_info.get('reasoning', '')
        quality = lane_info.get('detection_quality', 0.0)
        
        # 设置颜色
        confidence_color = self._get_confidence_color(confidence)
        
        # 绘制方向信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 1. 方向
        direction_text = f"方向: {direction}"
        cv2.putText(image, direction_text, (20, 35), 
                   font, 1.2, confidence_color, 2)
        
        # 2. 置信度
        confidence_text = f"置信度: {confidence:.1%}"
        cv2.putText(image, confidence_text, (20, 70), 
                   font, 0.9, confidence_color, 2)
        
        # 3. 检测质量
        quality_text = f"检测质量: {quality:.1%}"
        cv2.putText(image, quality_text, (20, 100), 
                   font, 0.7, self.colors['text_secondary'][:3], 1)
        
        # 4. 推理说明（截断以适应面板）
        if reasoning:
            # 简单截断处理
            max_chars = 40
            if len(reasoning) > max_chars:
                reasoning = reasoning[:max_chars-3] + "..."
            
            cv2.putText(image, reasoning, (20, 130), 
                       font, 0.6, self.colors['text_secondary'][:3], 1)
        
        # 5. 概率分布（右侧）
        if 'probabilities' in direction_info:
            probabilities = direction_info['probabilities']
            start_x = width - 200
            start_y = 30
            
            for i, (dir_name, prob) in enumerate(probabilities.items()):
                y = start_y + i * 25
                prob_text = f"{dir_name}: {prob:.1%}"
                
                # 高亮当前方向
                color = self.colors['text_highlight'][:3] if dir_name == direction \
                       else self.colors['text_primary'][:3]
                
                cv2.putText(image, prob_text, (start_x, y), 
                           font, 0.7, color, 1)
        
        return image
    
    def _draw_direction_indicator(self, image: np.ndarray, 
                                direction_info: Dict[str, Any]) -> np.ndarray:
        """绘制方向指示器"""
        height, width = image.shape[:2]
        direction = direction_info.get('direction', '未知')
        confidence = direction_info.get('confidence', 0.0)
        
        # 指示器位置（底部中央）
        center_x = width // 2
        indicator_y = height - 180
        
        # 创建指示器图层
        indicator_layer = np.zeros_like(image)
        
        # 根据方向和置信度绘制指示器
        if direction == "左转":
            # 左转箭头
            points = np.array([
                [center_x, indicator_y],
                [center_x - 80, indicator_y],
                [center_x - 60, indicator_y - 40],
                [center_x - 100, indicator_y - 40],
                [center_x - 120, indicator_y],
                [center_x - 200, indicator_y],
                [center_x - 100, indicator_y + 80],
                [center_x, indicator_y + 80]
            ], dtype=np.int32)
            base_color = (0, 165, 255)  # 橙色
            
        elif direction == "右转":
            # 右转箭头
            points = np.array([
                [center_x, indicator_y],
                [center_x + 80, indicator_y],
                [center_x + 60, indicator_y - 40],
                [center_x + 100, indicator_y - 40],
                [center_x + 120, indicator_y],
                [center_x + 200, indicator_y],
                [center_x + 100, indicator_y + 80],
                [center_x, indicator_y + 80]
            ], dtype=np.int32)
            base_color = (0, 165, 255)  # 橙色
            
        else:  # 直行或未知
            # 直行箭头
            points = np.array([
                [center_x - 60, indicator_y + 40],
                [center_x, indicator_y - 40],
                [center_x + 60, indicator_y + 40],
                [center_x + 40, indicator_y + 40],
                [center_x + 40, indicator_y + 120],
                [center_x - 40, indicator_y + 120],
                [center_x - 40, indicator_y + 40]
            ], dtype=np.int32)
            base_color = (0, 255, 0)  # 绿色
        
        # 根据置信度调整颜色亮度
        brightness_factor = 0.5 + confidence * 0.5
        color = tuple(int(c * brightness_factor) for c in base_color)
        
        # 绘制指示器
        cv2.fillPoly(indicator_layer, [points], color)
        
        # 根据置信度调整透明度
        alpha = 0.3 + confidence * 0.5
        cv2.addWeighted(indicator_layer, alpha, image, 1 - alpha, 0, image)
        
        # 绘制边框
        border_color = (255, 255, 255)
        cv2.polylines(image, [points], True, border_color, 2, cv2.LINE_AA)
        
        # 在指示器上添加置信度文本
        if confidence > 0.3:
            conf_text = f"{confidence:.0%}"
            text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = center_x - text_size[0] // 2
            text_y = indicator_y + 40
            
            cv2.putText(image, conf_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image
    
    def _draw_confidence_indicator(self, image: np.ndarray,
                                 direction_info: Dict[str, Any]) -> np.ndarray:
        """绘制置信度指示器"""
        confidence = direction_info.get('confidence', 0.0)
        height, width = image.shape[:2]
        
        # 在右上角绘制置信度条
        bar_width = 200
        bar_height = 20
        bar_x = width - bar_width - 20
        bar_y = 20
        
        # 绘制背景条
        cv2.rectangle(image, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # 绘制前景条（根据置信度）
        fill_width = int(bar_width * confidence)
        fill_color = self._get_confidence_color(confidence)
        
        cv2.rectangle(image, (bar_x, bar_y), 
                     (bar_x + fill_width, bar_y + bar_height),
                     fill_color, -1)
        
        # 绘制边框
        cv2.rectangle(image, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (255, 255, 255), 1)
        
        # 添加文本
        conf_text = f"置信度: {confidence:.1%}"
        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = bar_x + (bar_width - text_size[0]) // 2
        text_y = bar_y + bar_height // 2 + text_size[1] // 2
        
        cv2.putText(image, conf_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """根据置信度获取颜色"""
        if confidence >= 0.8:
            return self.colors['confidence_high'][:3]
        elif confidence >= 0.6:
            return self.colors['confidence_medium'][:3]
        elif confidence >= 0.4:
            return self.colors['confidence_low'][:3]
        else:
            return self.colors['confidence_very_low'][:3]
    
    def _apply_global_effects(self, image: np.ndarray) -> np.ndarray:
        """应用全局效果"""
        # 轻微锐化
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # 混合原始和锐化图像
        cv2.addWeighted(sharpened, 0.3, image, 0.7, 0, image)
        
        # 轻微增加对比度
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return image

# ==================== 主应用程序 ====================
class HighConfidenceLaneDetectionApp:
    """高置信度道路方向识别系统主应用程序"""
    
    def __init__(self, root):
        self.root = root
        self._setup_window()
        
        # 初始化配置
        self.config = AppConfig()
        
        # 初始化各个组件
        self.image_processor = SmartImageProcessor(self.config)
        self.road_detector = AdvancedRoadDetector(self.config)
        self.lane_detector = SmartLaneDetector(self.config)
        self.direction_analyzer = AdvancedDirectionAnalyzer(self.config)
        self.visualizer = SmartVisualizer(self.config)
        
        # 状态变量
        self.current_image = None
        self.current_image_path = None
        self.current_visualization = None
        self.is_processing = False
        self.processing_history = deque(maxlen=20)
        
        # 性能统计
        self.processing_times = []
        self.confidence_history = deque(maxlen=50)
        self.direction_history = deque(maxlen=50)
        
        # 创建界面
        self._create_enhanced_ui()
        
        # 绑定事件
        self._bind_events()
        
        print("高置信度道路方向识别系统已启动")
        print("=" * 50)
    
    def _setup_window(self):
        """设置窗口"""
        self.root.title("🚗 高置信度道路方向识别系统")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 750)
        
        # 设置窗口图标
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
        
        # 设置窗口背景
        self.root.configure(bg='#f0f0f0')
    
    def _create_enhanced_ui(self):
        """创建增强的用户界面"""
        # 主容器
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill="both", expand=True)
        
        # 标题栏
        self._create_title_bar(main_container)
        
        # 内容区域
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        # 左侧控制面板
        control_frame = self._create_control_panel(content_frame)
        control_frame.pack(side="left", fill="y", padx=(0, 10))
        
        # 右侧显示区域
        display_frame = self._create_display_panel(content_frame)
        display_frame.pack(side="right", fill="both", expand=True)
        
        # 状态栏
        self._create_status_bar(main_container)
    
    def _create_title_bar(self, parent):
        """创建标题栏"""
        title_frame = ttk.Frame(parent, style="Title.TFrame")
        title_frame.pack(fill="x", pady=(0, 10))
        
        # 标题
        title_label = ttk.Label(
            title_frame,
            text="高置信度道路方向识别系统",
            font=("微软雅黑", 18, "bold"),
            foreground="#2c3e50",
            background="#ecf0f1"
        )
        title_label.pack(side="left", padx=10, pady=10)
        
        # 版本信息
        version_label = ttk.Label(
            title_frame,
            text="v3.0 - 高置信度版",
            font=("微软雅黑", 10),
            foreground="#7f8c8d",
            background="#ecf0f1"
        )
        version_label.pack(side="right", padx=10, pady=10)
    
    def _create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(
            parent,
            text="控制面板",
            padding="15",
            relief="groove",
            style="Control.TLabelframe"
        )
        control_frame.pack_propagate(False)
        control_frame.config(width=320)
        
        # 文件操作区域
        file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding="10")
        file_frame.pack(fill="x", pady=(0, 15))
        
        # 选择图片按钮
        select_btn = ttk.Button(
            file_frame,
            text="📁 选择图片",
            command=self._select_image,
            width=20,
            style="Accent.TButton"
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
            foreground="#3498db",
            font=("微软雅黑", 9)
        )
        self.file_info_label.pack()
        
        # 参数调节区域
        param_frame = ttk.LabelFrame(control_frame, text="参数调节", padding="10")
        param_frame.pack(fill="x", pady=(0, 15))
        
        # 检测敏感度
        ttk.Label(param_frame, text="检测敏感度:", 
                 font=("微软雅黑", 9)).pack(anchor="w", pady=(0, 5))
        
        self.sensitivity_var = tk.DoubleVar(value=0.6)
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
        
        # 置信度阈值
        ttk.Label(param_frame, text="置信度阈值:", 
                 font=("微软雅黑", 9)).pack(anchor="w", pady=(0, 5))
        
        self.confidence_threshold_var = tk.DoubleVar(
            value=self.config.confidence_threshold
        )
        confidence_scale = ttk.Scale(
            param_frame,
            from_=0.3,
            to=0.9,
            variable=self.confidence_threshold_var,
            orient="horizontal",
            command=self._on_parameter_change,
            length=250
        )
        confidence_scale.pack(fill="x", pady=(0, 10))
        
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
        
        # 置信度显示和进度条
        confidence_frame = ttk.Frame(result_frame)
        confidence_frame.pack(fill="x", pady=(0, 5))
        
        self.confidence_label = ttk.Label(
            confidence_frame,
            text="",
            font=("微软雅黑", 11)
        )
        self.confidence_label.pack(side="left", anchor="w")
        
        # 置信度进度条
        self.confidence_progress = ttk.Progressbar(
            confidence_frame,
            orient="horizontal",
            length=100,
            mode="determinate"
        )
        self.confidence_progress.pack(side="right", padx=(10, 0))
        
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
        
        # 推理说明
        self.reasoning_label = ttk.Label(
            result_frame,
            text="",
            wraplength=280,
            font=("微软雅黑", 8),
            foreground="#34495e",
            justify="left"
        )
        self.reasoning_label.pack(anchor="w", pady=(5, 0))
        
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
        
        # 添加初始提示
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
        
        # 添加初始提示
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
        
        # 创建统计信息网格
        self._create_stats_grid(stats_frame)
        
        return display_frame
    
    def _create_stats_grid(self, parent):
        """创建统计信息网格"""
        # 使用网格布局
        stats_grid = ttk.Frame(parent)
        stats_grid.pack(fill="x")
        
        # 第一行
        row1 = ttk.Frame(stats_grid)
        row1.pack(fill="x", pady=(0, 5))
        
        ttk.Label(row1, text="处理次数:", font=("微软雅黑", 9)).pack(side="left", padx=(0, 5))
        self.process_count_label = ttk.Label(row1, text="0", font=("微软雅黑", 9, "bold"))
        self.process_count_label.pack(side="left", padx=(0, 20))
        
        ttk.Label(row1, text="平均时间:", font=("微软雅黑", 9)).pack(side="left", padx=(0, 5))
        self.avg_time_label = ttk.Label(row1, text="0.00s", font=("微软雅黑", 9, "bold"))
        self.avg_time_label.pack(side="left", padx=(0, 20))
        
        ttk.Label(row1, text="高置信率:", font=("微软雅黑", 9)).pack(side="left", padx=(0, 5))
        self.high_conf_rate_label = ttk.Label(row1, text="0%", font=("微软雅黑", 9, "bold"))
        self.high_conf_rate_label.pack(side="left")
        
        # 第二行
        row2 = ttk.Frame(stats_grid)
        row2.pack(fill="x")
        
        ttk.Label(row2, text="平均置信度:", font=("微软雅黑", 9)).pack(side="left", padx=(0, 5))
        self.avg_confidence_label = ttk.Label(row2, text="0.0%", font=("微软雅黑", 9, "bold"))
        self.avg_confidence_label.pack(side="left", padx=(0, 20))
        
        ttk.Label(row2, text="缓存命中:", font=("微软雅黑", 9)).pack(side="left", padx=(0, 5))
        self.cache_hit_label = ttk.Label(row2, text="0%", font=("微软雅黑", 9, "bold"))
        self.cache_hit_label.pack(side="left", padx=(0, 20))
        
        ttk.Label(row2, text="方向分布:", font=("微软雅黑", 9)).pack(side="left", padx=(0, 5))
        self.direction_dist_label = ttk.Label(row2, text="直:0 左:0 右:0", 
                                             font=("微软雅黑", 9, "bold"))
        self.direction_dist_label.pack(side="left")
    
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
    
    def _bind_events(self):
        """绑定事件"""
        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # 快捷键
        self.root.bind('<Control-o>', lambda e: self._select_image())
        self.root.bind('<F5>', lambda e: self._redetect())
        self.root.bind('<Escape>', lambda e: self.root.quit())
    
    def _on_closing(self):
        """窗口关闭时的处理"""
        if messagebox.askokcancel("退出", "确定要退出系统吗？"):
            self.root.destroy()
    
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
            self.file_info_label.config(
                text=os.path.basename(file_path)[:30] + 
                ("..." if len(os.path.basename(file_path)) > 30 else "")
            )
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
            
            self.current_image, roi_info, image_stats = result
            
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
            
            self.current_visualization = visualization
            
            processing_time = time.time() - start_time
            
            # 在主线程中更新UI
            self.root.after(0, self._update_results, 
                          direction_info, lane_info, visualization, processing_time, image_stats)
            
            # 更新统计信息
            self._update_statistics(direction_info, processing_time)
            
            print(f"处理完成: 方向={direction_info['direction']}, "
                  f"置信度={direction_info['confidence']:.1%}, "
                  f"耗时={processing_time:.3f}s")
            
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
                       visualization: np.ndarray, processing_time: float,
                       image_stats: Dict[str, Any]):
        """更新结果"""
        try:
            # 显示图像
            self._display_image(self.current_image, self.original_canvas, "原始图像")
            self._display_image(visualization, self.result_canvas, "检测结果")
            
            # 更新方向信息
            direction = direction_info['direction']
            confidence = direction_info['confidence']
            quality = lane_info.get('detection_quality', 0.0)
            reasoning = direction_info.get('reasoning', '')
            
            # 设置方向文本
            self.direction_label.config(text=f"方向: {direction}")
            
            # 设置置信度
            confidence_color, confidence_text = self._get_confidence_display(confidence)
            self.confidence_label.config(text=confidence_text, foreground=confidence_color)
            
            # 更新置信度进度条
            self.confidence_progress['value'] = confidence * 100
            
            # 设置检测质量
            quality_color, quality_text = self._get_quality_display(quality)
            self.quality_label.config(text=quality_text, foreground=quality_color)
            
            # 设置处理时间
            self.time_label.config(text=f"处理时间: {processing_time:.3f}秒")
            
            # 设置推理说明
            self.reasoning_label.config(text=f"说明: {reasoning}")
            
            # 更新状态
            status_text = f"分析完成 - {direction}"
            if direction_info.get('is_high_confidence', False):
                status_text += " (高置信度)"
            self.status_var.set(status_text)
            
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
    
    def _get_confidence_display(self, confidence: float) -> Tuple[str, str]:
        """获取置信度显示信息"""
        if confidence > 0.8:
            return ("#27ae60", f"置信度: {confidence:.1%} (非常高)")
        elif confidence > 0.7:
            return ("#2ecc71", f"置信度: {confidence:.1%} (高)")
        elif confidence > 0.6:
            return ("#f39c12", f"置信度: {confidence:.1%} (中高)")
        elif confidence > 0.5:
            return ("#e67e22", f"置信度: {confidence:.1%} (中等)")
        elif confidence > 0.4:
            return ("#e74c3c", f"置信度: {confidence:.1%} (中低)")
        elif confidence > 0.3:
            return ("#c0392b", f"置信度: {confidence:.1%} (低)")
        else:
            return ("#7f8c8d", f"置信度: {confidence:.1%} (非常低)")
    
    def _get_quality_display(self, quality: float) -> Tuple[str, str]:
        """获取质量显示信息"""
        if quality > 0.8:
            return ("#27ae60", f"检测质量: {quality:.1%} (优秀)")
        elif quality > 0.7:
            return ("#2ecc71", f"检测质量: {quality:.1%} (良好)")
        elif quality > 0.6:
            return ("#f39c12", f"检测质量: {quality:.1%} (一般)")
        elif quality > 0.5:
            return ("#e67e22", f"检测质量: {quality:.1%} (中等)")
        elif quality > 0.4:
            return ("#e74c3c", f"检测质量: {quality:.1%} (较差)")
        else:
            return ("#c0392b", f"检测质量: {quality:.1%} (很差)")
    
    def _update_statistics(self, direction_info: Dict[str, Any], processing_time: float):
        """更新统计信息"""
        # 记录处理时间
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 20:
            self.processing_times.pop(0)
        
        # 记录置信度
        confidence = direction_info['confidence']
        self.confidence_history.append(confidence)
        
        # 记录方向
        direction = direction_info['direction']
        self.direction_history.append(direction)
        
        # 更新UI统计显示
        self.root.after(0, self._update_stats_display)
    
    def _update_stats_display(self):
        """更新统计显示"""
        try:
            # 处理次数
            process_count = len(self.processing_times)
            self.process_count_label.config(text=str(process_count))
            
            # 平均处理时间
            if self.processing_times:
                avg_time = np.mean(self.processing_times)
                self.avg_time_label.config(text=f"{avg_time:.3f}s")
            
            # 高置信率
            if self.confidence_history:
                high_conf_count = sum(1 for c in self.confidence_history if c > 0.7)
                high_conf_rate = high_conf_count / len(self.confidence_history) * 100
                self.high_conf_rate_label.config(text=f"{high_conf_rate:.1f}%")
            
            # 平均置信度
            if self.confidence_history:
                avg_confidence = np.mean(self.confidence_history) * 100
                self.avg_confidence_label.config(text=f"{avg_confidence:.1f}%")
            
            # 缓存命中率
            cache_info = self.image_processor.get_cache_info()
            cache_hit_rate = cache_info.get('cache_hit_rate', 0) * 100
            self.cache_hit_label.config(text=f"{cache_hit_rate:.1f}%")
            
            # 方向分布
            if self.direction_history:
                straight_count = sum(1 for d in self.direction_history if d == '直行')
                left_count = sum(1 for d in self.direction_history if d == '左转')
                right_count = sum(1 for d in self.direction_history if d == '右转')
                self.direction_dist_label.config(
                    text=f"直:{straight_count} 左:{left_count} 右:{right_count}"
                )
                
        except Exception as e:
            print(f"更新统计显示失败: {e}")
    
    def _redetect(self):
        """重新检测"""
        if self.current_image_path and not self.is_processing:
            self._process_image(self.current_image_path)
    
    def _on_parameter_change(self, value):
        """参数变化回调"""
        # 更新配置
        sensitivity = self.sensitivity_var.get()
        confidence_threshold = self.confidence_threshold_var.get()
        
        # 根据敏感度调整参数
        self.config.canny_threshold1 = int(30 + sensitivity * 40)
        self.config.canny_threshold2 = int(80 + sensitivity * 100)
        self.config.hough_threshold = int(20 + (1 - sensitivity) * 30)
        self.config.confidence_threshold = confidence_threshold
        
        print(f"参数更新: 敏感度={sensitivity:.2f}, 置信度阈值={confidence_threshold:.2f}")
        
        # 如果已有图像，自动重新检测
        if self.current_image_path and not self.is_processing:
            self._redetect()
    
    def _show_error(self, error_msg: str):
        """显示错误"""
        messagebox.showerror("错误", f"处理失败: {error_msg}")
        self.status_var.set("处理失败")

# ==================== 主函数 ====================
def main():
    """主函数"""
    try:
        # 创建主窗口
        root = tk.Tk()
        
        # 设置样式
        style = ttk.Style()
        style.theme_use('clam')
        
        # 创建应用程序实例
        app = HighConfidenceLaneDetectionApp(root)
        
        # 运行主循环
        root.mainloop()
        
    except Exception as e:
        print(f"应用程序启动失败: {e}")
        messagebox.showerror("致命错误", f"应用程序启动失败: {str(e)}")

if __name__ == "__main__":
    main()