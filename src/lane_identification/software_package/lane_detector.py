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

    def detect(self, image: np.ndarray, roi_mask: np.ndarray, light_condition: str = 'day') -> Dict[str, Any]:
        """检测车道线 - 支持多车道场景"""
        try:
            processed = self._preprocess_for_lanes(image, roi_mask, light_condition)
            edges = self._detect_edges(processed, light_condition)

            lines = self._hough_transform(edges)

            if lines is None or len(lines) == 0:
                return self._create_empty_result()

            left_lines, right_lines = self._classify_and_filter(lines, image.shape[1])

            # 新增：合并相似的线段
            left_lines = self._merge_similar_lines(left_lines)
            right_lines = self._merge_similar_lines(right_lines)

            # 选择主车道和邻车道
            primary_left, primary_right, neighbor_left, neighbor_right = \
                self._select_primary_lanes(left_lines, right_lines, image.shape[1], image.shape[0])

            # 使用主车道线进行拟合
            left_lane = self._fit_lane_model(primary_left, image.shape)
            right_lane = self._fit_lane_model(primary_right, image.shape)

            left_lane, right_lane = self._validate_lanes(left_lane, right_lane, image.shape)
            center_line = self._calculate_center_line(left_lane, right_lane, image.shape)
            future_path = self._predict_future_path(left_lane, right_lane, image.shape)
            detection_quality = self._calculate_detection_quality(left_lane, right_lane)

            # 计算车道统计信息
            all_lines = left_lines + right_lines
            lane_stats = self._calculate_lane_statistics(all_lines, image.shape)

            result = {
                'left_lines': left_lines,
                'right_lines': right_lines,
                'primary_left_lines': primary_left,
                'primary_right_lines': primary_right,
                'neighbor_left_lines': neighbor_left,
                'neighbor_right_lines': neighbor_right,
                'left_lane': left_lane,
                'right_lane': right_lane,
                'center_line': center_line,
                'future_path': future_path,
                'detection_quality': detection_quality,
                'lane_statistics': lane_stats
            }

            result = self._apply_temporal_smoothing(result)
            return result

        except Exception as e:
            print(f"车道线检测失败: {e}")
            return self._create_empty_result()

    def _merge_similar_lines(self, lines: List[Dict], distance_threshold: float = 20.0, angle_threshold: float = 0.1) -> \
    List[Dict]:
        """合并相似的车道线段

        Args:
            lines: 待合并的线段列表
            distance_threshold: 距离阈值（像素）
            angle_threshold: 角度阈值（斜率差）

        Returns:
            合并后的线段列表
        """
        if len(lines) < 2:
            return lines

        merged_lines = []
        used_indices = set()

        for i, line1 in enumerate(lines):
            if i in used_indices:
                continue

            similar_lines = [line1]
            used_indices.add(i)

            for j, line2 in enumerate(lines):
                if j in used_indices or j == i:
                    continue

                # 检查斜率是否相似
                if abs(line1['slope'] - line2['slope']) > angle_threshold:
                    continue

                # 检查中点距离
                mid1 = np.array(line1['midpoint'])
                mid2 = np.array(line2['midpoint'])
                distance = np.linalg.norm(mid1 - mid2)

                if distance <= distance_threshold:
                    similar_lines.append(line2)
                    used_indices.add(j)

            # 合并相似线段：取平均斜率和端点
            if len(similar_lines) > 0:
                avg_slope = np.mean([l['slope'] for l in similar_lines])
                avg_midpoint = np.mean([l['midpoint'] for l in similar_lines], axis=0)
                avg_length = np.mean([l['length'] for l in similar_lines])

                # 使用最长的线段作为代表
                representative = max(similar_lines, key=lambda x: x['length'])

                merged_lines.append({
                    'points': representative['points'],
                    'slope': avg_slope,
                    'length': avg_length,
                    'midpoint': tuple(avg_midpoint)
                })

        return merged_lines

    def _preprocess_for_lanes(self, image: np.ndarray, roi_mask: np.ndarray, light_condition: str) -> np.ndarray:
        """为车道线检测预处理图像"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=roi_mask)

        # 夜间模式二次增强
        if light_condition == 'night':
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        return gray

    def _detect_edges(self, image: np.ndarray, light_condition: str) -> np.ndarray:
        """边缘检测"""
        # 根据光照动态调整阈值
        if light_condition == 'night':
            lower, upper = 30, 100
        elif light_condition == 'dusk':
            lower, upper = 40, 120
        else:
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
                continue

            dx = x2 - x1
            dy = y2 - y1
            slope = dy / dx
            length = np.sqrt(dx ** 2 + dy ** 2)

            # 过滤条件 - 提高阈值减少碎片
            if abs(slope) < 0.3 or length < 30:
                continue

            # 增加：过滤过于水平的线段（可能是横向标记）
            if abs(slope) > 3.0:
                continue

            midpoint_x = (x1 + x2) / 2

            # 分类
            if slope < 0:
                if midpoint_x < image_width * 0.6:
                    left_lines.append({
                        'points': [(x1, y1), (x2, y2)],
                        'slope': slope,
                        'length': length,
                        'midpoint': (midpoint_x, (y1 + y2) / 2)
                    })
            else:
                if midpoint_x > image_width * 0.4:
                    right_lines.append({
                        'points': [(x1, y1), (x2, y2)],
                        'slope': slope,
                        'length': length,
                        'midpoint': (midpoint_x, (y1 + y2) / 2)
                    })

        return left_lines, right_lines

    def _select_primary_lanes(self, left_lines: List[Dict], right_lines: List[Dict],
                              image_width: int, image_height: int) -> Tuple[List, List, List, List]:
        """从多条候选线中选择主车道和邻车道

        Returns:
            primary_left: 主车道左边界线（最靠近中心的）
            primary_right: 主车道右边界线（最靠近中心的）
            neighbor_left: 左侧邻车道线列表
            neighbor_right: 右侧邻车道线列表
        """
        if not left_lines and not right_lines:
            return [], [], [], []

        center_x = image_width / 2

        # 按到中点的距离排序
        left_lines_sorted = sorted(left_lines, key=lambda x: abs(x['midpoint'][0] - center_x))
        right_lines_sorted = sorted(right_lines, key=lambda x: abs(x['midpoint'][0] - center_x))

        # 选择最靠近中心的前2条作为主车道边界
        primary_left = left_lines_sorted[:2] if len(left_lines_sorted) >= 2 else left_lines_sorted
        primary_right = right_lines_sorted[:2] if len(right_lines_sorted) >= 2 else right_lines_sorted

        # 其余的作为邻车道
        neighbor_left = left_lines_sorted[2:] if len(left_lines_sorted) > 2 else []
        neighbor_right = right_lines_sorted[2:] if len(right_lines_sorted) > 2 else []

        return primary_left, primary_right, neighbor_left, neighbor_right

    def _calculate_lane_statistics(self, all_lines: List[Dict], image_shape: Tuple[int, ...]) -> Dict:
        """计算车道统计信息"""
        height, width = image_shape[:2]

        if not all_lines:
            return {
                'total_detected_lines': 0,
                'estimated_lanes': 1,
                'is_multi_lane': False,
                'avg_lane_width': 0,
                'lane_positions': []
            }

        midpoints = sorted([line['midpoint'][0] for line in all_lines])

        # 基于相邻车道线间距分组判断车道数量
        # 合理车道宽度为图像宽度的8%-35%
        min_lane_width = width * 0.08
        max_lane_width = width * 0.35

        lane_groups = []
        current_group = [midpoints[0]]

        for i in range(1, len(midpoints)):
            gap = midpoints[i] - midpoints[i - 1]

            if min_lane_width <= gap <= max_lane_width:
                current_group.append(midpoints[i])
            else:
                if len(current_group) >= 2:
                    lane_groups.append(current_group)
                current_group = [midpoints[i]]

        if len(current_group) >= 2:
            lane_groups.append(current_group)

        if not lane_groups:
            estimated_lanes = 1
        else:
            estimated_lanes = min(len(lane_groups), 4)

        lane_widths = []
        for group in lane_groups:
            for i in range(len(group) - 1):
                lane_widths.append(group[i + 1] - group[i])

        avg_lane_width = np.mean(lane_widths) if lane_widths else 0

        return {
            'total_detected_lines': len(all_lines),
            'estimated_lanes': estimated_lanes,
            'is_multi_lane': estimated_lanes > 2,
            'avg_lane_width': avg_lane_width,
            'lane_positions': midpoints
        }

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
        except (ValueError, TypeError):
            try:
                coeffs = np.polyfit(y_points, x_points, 1)
                model_type = 'linear'
            except (ValueError, TypeError):
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

            # 计算置信度 - 基于线段数量和长度
            avg_length = np.mean([line['length'] for line in lines])
            length_score = min(avg_length / 100.0, 1.0)
            count_score = min(len(lines) / 8.0, 1.0)
            confidence = 0.6 * count_score + 0.4 * length_score

            return {
                'func': poly_func,
                'coeffs': coeffs.tolist(),
                'points': [(x_bottom, y_bottom), (x_top, y_top)],
                'model_type': model_type,
                'confidence': confidence,
                'num_lines': len(lines)
            }
        except (IndexError, OverflowError):
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
            'primary_left_lines': [],
            'primary_right_lines': [],
            'neighbor_left_lines': [],
            'neighbor_right_lines': [],
            'left_lane': None,
            'right_lane': None,
            'center_line': None,
            'future_path': None,
            'detection_quality': 0.0,
            'lane_statistics': {
                'total_detected_lines': 0,
                'estimated_lanes': 1,
                'is_multi_lane': False,
                'avg_lane_width': 0,
                'lane_positions': []
            }
        }