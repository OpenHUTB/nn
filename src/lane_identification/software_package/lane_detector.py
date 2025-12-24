"""
车道线检测模块 - 负责检测和拟合车道线
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict, Any
from config import AppConfig

class LaneDetector:
    """车道线检测器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.lane_history = deque(maxlen=5)
    
    def detect(self, image: np.ndarray, roi_mask: np.ndarray) -> Dict[str, Any]:
        """检测车道线"""
        try:
            # 预处理
            processed = self._preprocess_for_lanes(image, roi_mask)
            
            # 边缘检测
            edges = self._detect_edges(processed)
            
            # 霍夫变换检测直线
            lines = self._hough_transform(edges)
            
            if lines is None or len(lines) == 0:
                return self._create_empty_result()
            
            # 分类和过滤车道线
            left_lines, right_lines = self._classify_and_filter(lines, image.shape[1])
            
            # 拟合车道线
            left_lane = self._fit_lane_model(left_lines, image.shape)
            right_lane = self._fit_lane_model(right_lines, image.shape)
            
            # 验证车道线
            left_lane, right_lane = self._validate_lanes(left_lane, right_lane, image.shape)
            
            # 计算中心线
            center_line = self._calculate_center_line(left_lane, right_lane, image.shape)
            
            # 预测路径
            future_path = self._predict_future_path(left_lane, right_lane, image.shape)
            
            # 计算检测质量
            detection_quality = self._calculate_detection_quality(left_lane, right_lane)
            
            # 创建结果
            result = {
                'left_lines': left_lines,
                'right_lines': right_lines,
                'left_lane': left_lane,
                'right_lane': right_lane,
                'center_line': center_line,
                'future_path': future_path,
                'detection_quality': detection_quality
            }
            
            # 时间平滑
            result = self._apply_temporal_smoothing(result)
            
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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """边缘检测"""
        # 自适应Canny阈值
        median = np.median(image)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        
        return cv2.Canny(image, lower, upper)
    
    def _hough_transform(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """霍夫变换检测直线"""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.hough_min_length,
            maxLineGap=self.config.hough_max_gap
        )
        return lines
    
    def _classify_and_filter(self, lines: np.ndarray, image_width: int) -> Tuple[List, List]:
        """分类和过滤车道线"""
        left_lines, right_lines = [], []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if x2 == x1:
                continue  # 跳过垂直线
            
            dx = x2 - x1
            dy = y2 - y1
            slope = dy / dx
            length = np.sqrt(dx**2 + dy**2)
            
            # 过滤条件
            if abs(slope) < 0.3 or length < 20:
                continue
            
            midpoint_x = (x1 + x2) / 2
            
            # 分类
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
        
        return left_lines, right_lines
    
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
        
        # 多项式拟合
        try:
            coeffs = np.polyfit(y_points, x_points, 2)
            model_type = 'quadratic'
        except:
            try:
                coeffs = np.polyfit(y_points, x_points, 1)
                model_type = 'linear'
            except:
                return None
        
        poly_func = np.poly1d(coeffs)
        
        # 生成车道线点
        height, width = image_shape[:2]
        y_bottom = height
        y_top = int(height * 0.4)
        
        try:
            x_bottom = int(poly_func(y_bottom))
            x_top = int(poly_func(y_top))
            
            # 限制范围
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
        except:
            return None
    
    def _validate_lanes(self, left_lane: Optional[Dict], right_lane: Optional[Dict],
                       image_shape: Tuple[int, ...]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """验证车道线合理性"""
        if left_lane is None or right_lane is None:
            return left_lane, right_lane
        
        height, width = image_shape[:2]
        
        try:
            # 检查车道宽度
            left_func = left_lane['func']
            right_func = right_lane['func']
            
            y_points = np.linspace(height * 0.4, height, 5)
            widths = []
            
            for y in y_points:
                left_x = left_func(y)
                right_x = right_func(y)
                if right_x > left_x:
                    widths.append(right_x - left_x)
            
            if widths:
                avg_width = np.mean(widths)
                std_width = np.std(widths)
                
                # 宽度合理性检查
                min_reasonable_width = width * 0.15
                max_reasonable_width = width * 0.8
                
                if avg_width < min_reasonable_width or avg_width > max_reasonable_width or std_width > width * 0.2:
                    left_lane['confidence'] *= 0.7
                    right_lane['confidence'] *= 0.7
            
            # 检查车道线交叉
            if left_lane['points'][0][0] > right_lane['points'][0][0]:
                left_lane['confidence'] *= 0.6
                right_lane['confidence'] *= 0.6
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
            
            def center_func(y):
                left_x = left_func(y)
                right_x = right_func(y)
                return (left_x + right_x) / 2
            
            # 生成中心线点
            y_bottom = height
            y_top = int(height * 0.4)
            
            x_bottom = int(center_func(y_bottom))
            x_top = int(center_func(y_top))
            
            x_bottom = max(0, min(width, x_bottom))
            x_top = max(0, min(width, x_top))
            
            confidence = (left_lane.get('confidence', 0) + right_lane.get('confidence', 0)) / 2
            
            return {
                'func': center_func,
                'points': [(x_bottom, y_bottom), (x_top, y_top)],
                'confidence': confidence
            }
            
        except Exception:
            return None
    
    def _predict_future_path(self, left_lane: Optional[Dict], right_lane: Optional[Dict],
                           image_shape: Tuple[int, ...]) -> Optional[Dict]:
        """预测未来路径"""
        if left_lane is None or right_lane is None:
            return None
        
        try:
            height, width = image_shape[:2]
            
            # 计算中心线
            center_line = self._calculate_center_line(left_lane, right_lane, image_shape)
            if center_line is None:
                return None
            
            center_func = center_line['func']
            
            # 生成预测点
            current_y = height
            target_y = int(height * (1 - self.config.prediction_distance))
            
            if target_y <= current_y * 0.6:
                return None
            
            y_values = np.linspace(current_y, target_y, self.config.prediction_steps)
            path_points = []
            
            for y in y_values:
                try:
                    x = center_func(y)
                    x = max(0, min(width, x))
                    path_points.append((int(x), int(y)))
                except:
                    continue
            
            if len(path_points) < self.config.min_prediction_points:
                return None
            
            return {
                'center_path': path_points,
                'prediction_length': len(path_points)
            }
            
        except Exception:
            return None
    
    def _calculate_detection_quality(self, left_lane: Optional[Dict], right_lane: Optional[Dict]) -> float:
        """计算检测质量"""
        quality = 0.0
        
        if left_lane is not None:
            quality += left_lane.get('confidence', 0) * 0.5
        
        if right_lane is not None:
            quality += right_lane.get('confidence', 0) * 0.5
        
        if left_lane is not None and right_lane is not None:
            quality += 0.1
            if left_lane.get('model_type') == right_lane.get('model_type'):
                quality += 0.1
        
        return min(quality, 1.0)
    
    def _apply_temporal_smoothing(self, current_result: Dict[str, Any]) -> Dict[str, Any]:
        """应用时间平滑"""
        if len(self.lane_history) < 2:
            self.lane_history.append(current_result)
            return current_result
        
        smoothing_factor = 0.6
        
        # 平滑车道线参数
        if current_result['left_lane'] and current_result['left_lane'].get('coeffs'):
            for prev_result in list(self.lane_history)[-2:]:
                if prev_result['left_lane'] and prev_result['left_lane'].get('coeffs'):
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
        if current_result['right_lane'] and current_result['right_lane'].get('coeffs'):
            for prev_result in list(self.lane_history)[-2:]:
                if prev_result['right_lane'] and prev_result['right_lane'].get('coeffs'):
                    prev_coeffs = np.array(prev_result['right_lane']['coeffs'])
                    curr_coeffs = np.array(current_result['right_lane']['coeffs'])
                    
                    if len(prev_coeffs) == len(curr_coeffs):
                        smoothed_coeffs = (
                            smoothing_factor * curr_coeffs + 
                            (1 - smoothing_factor) * prev_coeffs
                        )
                        current_result['right_lane']['coeffs'] = smoothed_coeffs.tolist()
                        current_result['right_lane']['func'] = np.poly1d(smoothed_coeffs)
        
        self.lane_history.append(current_result)
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
            'detection_quality': 0.0
        }