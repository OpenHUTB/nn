"""
方向分析模块 - 负责分析道路方向并计算置信度
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, Any, Tuple, List
from config import AppConfig

class DirectionAnalyzer:
    """方向分析器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.history = deque(maxlen=10)
        
        # 特征权重
        self.feature_weights = {
            'lane_convergence': 0.35,
            'lane_symmetry': 0.20,
            'path_curvature': 0.20,
            'contour_position': 0.15,
            'historical_consistency': 0.10
        }
    
    def analyze(self, road_features: Dict[str, Any], lane_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析道路方向"""
        try:
            # 提取特征
            features = self._extract_features(road_features, lane_info)
            
            # 方向预测
            direction_probs = self._predict_direction(features)
            
            # 置信度计算
            confidence = self._calculate_confidence(features, direction_probs, lane_info)
            
            # 获取最终方向
            final_direction = self._get_final_direction(direction_probs, confidence)
            
            # 历史平滑
            final_direction, confidence = self._apply_historical_smoothing(final_direction, confidence)
            
            # 生成推理说明
            reasoning = self._generate_reasoning(features, direction_probs, final_direction)
            
            # 创建结果
            result = {
                'direction': final_direction,
                'confidence': confidence,
                'probabilities': direction_probs,
                'features': features,
                'reasoning': reasoning
            }
            
            # 更新历史
            if confidence > 0.3:
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
        
        # 2. 车道线特征
        if lane_info['left_lane'] and lane_info['right_lane']:
            features['lane_convergence'] = self._calculate_lane_convergence(lane_info)
            features['lane_symmetry'] = self._calculate_lane_symmetry(lane_info)
            features['lane_width'] = self._calculate_lane_width(lane_info)
        
        # 3. 路径特征
        if lane_info['future_path']:
            features['path_curvature'] = self._calculate_path_curvature(lane_info['future_path'])
            features['path_straightness'] = self._calculate_path_straightness(lane_info['future_path'])
        
        # 4. 历史特征
        if self.history:
            features['historical_consistency'] = self._calculate_historical_consistency()
        
        # 5. 检测质量
        features['detection_quality'] = lane_info.get('detection_quality', 0.0)
        
        return features
    
    def _calculate_lane_convergence(self, lane_info: Dict[str, Any]) -> float:
        """计算车道线收敛度"""
        left_func = lane_info['left_lane']['func']
        right_func = lane_info['right_lane']['func']
        
        # 在顶部和底部计算宽度
        y_bottom = 600  # 假设图像高度
        y_top = int(y_bottom * 0.4)
        
        width_bottom = right_func(y_bottom) - left_func(y_bottom)
        width_top = right_func(y_top) - left_func(y_top)
        
        if width_bottom > 0:
            return float(width_top / width_bottom)
        
        return 1.0
    
    def _calculate_lane_symmetry(self, lane_info: Dict[str, Any]) -> float:
        """计算车道对称性"""
        left_func = lane_info['left_lane']['func']
        right_func = lane_info['right_lane']['func']
        
        def center_func(y):
            return (left_func(y) + right_func(y)) / 2
        
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
    
    def _calculate_lane_width(self, lane_info: Dict[str, Any]) -> float:
        """计算平均车道宽度"""
        left_func = lane_info['left_lane']['func']
        right_func = lane_info['right_lane']['func']
        
        y_values = np.linspace(600 * 0.4, 600, 5)
        widths = [right_func(y) - left_func(y) for y in y_values]
        
        return float(np.mean(widths))
    
    def _calculate_path_curvature(self, future_path: Dict[str, Any]) -> float:
        """计算路径曲率"""
        path_points = future_path.get('center_path', [])
        
        if len(path_points) < 3:
            return 0.0
        
        pts = np.array(path_points, dtype=np.float32)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # 计算导数
        dx = np.gradient(x)
        dy = np.gradient(y)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        # 计算曲率
        curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**1.5
        
        # 返回平均曲率
        return float(np.mean(curvature[np.isfinite(curvature)]))
    
    def _calculate_path_straightness(self, future_path: Dict[str, Any]) -> float:
        """计算路径直线度"""
        path_points = future_path.get('center_path', [])
        
        if len(path_points) < 2:
            return 1.0
        
        pts = np.array(path_points, dtype=np.float32)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # 线性拟合
        coeffs = np.polyfit(y, x, 1)
        poly_func = np.poly1d(coeffs)
        
        # 计算R²值
        residuals = x - poly_func(y)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((x - np.mean(x))**2)
        
        if ss_tot > 0:
            return float(1 - (ss_res / ss_tot))
        
        return 0.5
    
    def _calculate_historical_consistency(self) -> float:
        """计算历史一致性"""
        if len(self.history) < 2:
            return 0.5
        
        recent_directions = [h['direction'] for h in list(self.history)[-3:]]
        
        if len(recent_directions) >= 2:
            from collections import Counter
            freq = Counter(recent_directions)
            most_common_count = max(freq.values())
            return most_common_count / len(recent_directions)
        
        return 0.5
    
    def _predict_direction(self, features: Dict[str, Any]) -> Dict[str, float]:
        """预测方向概率"""
        # 基础概率
        probabilities = {'直行': 0.3, '左转': 0.35, '右转': 0.35}
        
        # 1. 基于轮廓质心
        if 'contour_centroid_x' in features:
            centroid_x = features['contour_centroid_x']
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
                symmetry = features.get('lane_symmetry', 0.5)
                if symmetry < 0.6:
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
        
        # 归一化
        total = sum(probabilities.values())
        if total > 0:
            for direction in probabilities:
                probabilities[direction] /= total
        
        return probabilities
    
    def _calculate_confidence(self, features: Dict[str, Any], 
                            probabilities: Dict[str, float],
                            lane_info: Dict[str, Any]) -> float:
        """计算置信度"""
        confidence_factors = []
        
        # 1. 概率清晰度
        max_prob = max(probabilities.values())
        min_prob = min(probabilities.values())
        if max_prob > 0:
            clarity = (max_prob - min_prob) / max_prob
            confidence_factors.append(clarity * 0.4)
        
        # 2. 特征质量
        feature_quality = 0.0
        quality_indicators = []
        
        if 'lane_convergence' in features:
            convergence = features['lane_convergence']
            conv_confidence = 1 - min(1.0, abs(convergence - 1.0) * 1.5)
            quality_indicators.append(conv_confidence)
        
        if 'lane_symmetry' in features:
            symmetry = features['lane_symmetry']
            quality_indicators.append(symmetry)
        
        if 'path_curvature' in features:
            curvature = features['path_curvature']
            curv_confidence = 1 - min(1.0, abs(curvature) * 100)
            quality_indicators.append(curv_confidence)
        
        if quality_indicators:
            feature_quality = np.mean(quality_indicators)
        
        confidence_factors.append(feature_quality * 0.3)
        
        # 3. 检测质量
        detection_quality = lane_info.get('detection_quality', 0.5)
        confidence_factors.append(detection_quality * 0.2)
        
        # 4. 历史一致性
        if 'historical_consistency' in features:
            consistency = features['historical_consistency']
            confidence_factors.append(consistency * 0.1)
        
        # 综合置信度
        confidence = sum(confidence_factors)
        
        # 非线性调整
        if confidence < 0.3:
            confidence = confidence * 0.8
        elif confidence < 0.7:
            confidence = 0.3 + (confidence - 0.3) * 1.1
        
        return min(max(confidence, 0.0), 1.0)
    
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
        if confidence < self.config.min_confidence_for_direction:
            return '未知'
        elif confidence < self.config.confidence_threshold:
            # 需要明显优势
            if best_prob > second_prob * 1.5:
                return best_direction
            else:
                return '未知'
        else:
            # 允许较小优势
            if best_prob > second_prob * 1.2:
                return best_direction
            else:
                # 参考历史
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
        
        return max(direction_counts.items(), key=lambda x: x[1])[0]
    
    def _apply_historical_smoothing(self, direction: str, confidence: float) -> Tuple[str, float]:
        """应用历史平滑"""
        if len(self.history) < 2:
            return direction, confidence
        
        recent_history = list(self.history)[-4:]
        
        # 统计历史方向
        direction_counts = defaultdict(int)
        for result in recent_history:
            d = result['direction']
            direction_counts[d] += 1
        
        if not direction_counts:
            return direction, confidence
        
        most_common, count = max(direction_counts.items(), key=lambda x: x[1])
        frequency = count / len(recent_history)
        
        # 如果历史一致性高且当前置信度低，信任历史
        if frequency > 0.75 and confidence < 0.5:
            historical_confidences = [h.get('confidence', 0) for h in recent_history]
            historical_confidence = np.mean(historical_confidences)
            return most_common, historical_confidence * 0.9
        
        # 如果历史与当前不一致，但历史一致性高
        if most_common != direction and frequency > 0.6:
            smoothing_factor = min(0.7, frequency)
            historical_confidences = [h.get('confidence', 0) for h in recent_history]
            historical_confidence = np.mean(historical_confidences)
            
            smoothed_confidence = (
                confidence * (1 - smoothing_factor) + 
                historical_confidence * smoothing_factor
            )
            
            if confidence < 0.4 and historical_confidence > 0.6:
                return most_common, smoothed_confidence
        
        return direction, confidence
    
    def _generate_reasoning(self, features: Dict[str, Any], 
                          probabilities: Dict[str, float],
                          final_direction: str) -> str:
        """生成推理说明"""
        reasons = []
        
        # 基于特征添加原因
        if 'lane_convergence' in features:
            convergence = features['lane_convergence']
            if convergence < 0.8:
                reasons.append("车道明显收敛")
            elif convergence > 1.2:
                reasons.append("车道发散")
            else:
                reasons.append("车道基本平行")
        
        if 'path_curvature' in features:
            curvature = features['path_curvature']
            if abs(curvature) < 0.0005:
                reasons.append("路径基本直线")
            elif curvature > 0:
                reasons.append("路径向右弯曲")
            else:
                reasons.append("路径向左弯曲")
        
        # 基于概率添加原因
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        if sorted_probs:
            top1, top2 = sorted_probs[:2]
            if top1[1] > top2[1] * 1.5:
                reasons.append(f"{top1[0]}明显占优")
            elif top1[1] > top2[1] * 1.2:
                reasons.append(f"{top1[0]}稍占优势")
        
        return "，".join(reasons) if reasons else "特征不明显"
    
    def _create_default_result(self) -> Dict[str, Any]:
        """创建默认结果"""
        return {
            'direction': '未知',
            'confidence': 0.0,
            'probabilities': {'直行': 0.33, '左转': 0.33, '右转': 0.34},
            'features': {},
            'reasoning': '检测失败'
        }