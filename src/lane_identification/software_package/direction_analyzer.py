"""
方向分析模块 - 稳定优化版
已移除日志文件生成功能
"""

import numpy as np
from collections import deque, defaultdict
import cv2

class DirectionAnalyzer:
    """方向分析器 - 稳定优化版本"""
    
    def __init__(self, config=None):
        if config is None:
            # 创建默认配置
            class DefaultConfig:
                min_confidence_for_direction = 0.25
                confidence_threshold = 0.5
                
            self.config = DefaultConfig()
        else:
            self.config = config
        
        self.history = deque(maxlen=10)
        self.direction_history = deque(maxlen=8)
        self.confidence_history = deque(maxlen=8)
        
        # 特征权重
        self.feature_weights = {
            'lane_convergence': 0.30,
            'lane_symmetry': 0.20,
            'lane_balance': 0.15,
            'centroid_offset': 0.20,
            'path_curvature': 0.15
        }
    
    def analyze(self, road_features, lane_info):
        """分析道路方向 - 主方法"""
        try:
            # 提取特征
            features = self._extract_features(road_features, lane_info)
            
            # 如果没有足够特征，使用回退策略
            if len(features) < 2:
                return self._fallback_direction_analysis(road_features, lane_info)
            
            # 方向预测
            direction_probs = self._predict_direction_improved(features)
            
            # 置信度计算
            confidence = self._calculate_confidence_improved(features, direction_probs, lane_info)
            
            # 获取最终方向
            final_direction = self._get_final_direction_improved(direction_probs, confidence)
            
            # 历史平滑
            final_direction, confidence = self._apply_historical_smoothing(final_direction, confidence)
            
            # 生成推理说明
            reasoning = self._generate_detailed_reasoning(features, direction_probs, final_direction, confidence)
            
            # 创建结果
            result = {
                'direction': final_direction,
                'confidence': confidence,
                'probabilities': direction_probs,
                'features': features,
                'reasoning': reasoning
            }
            
            # 更新历史
            if confidence > 0.2:
                self.history.append(result)
                self.direction_history.append(final_direction)
                self.confidence_history.append(confidence)
            
            return result
            
        except Exception as e:
            # 只打印错误，不生成日志文件
            print(f"方向分析失败: {e}")
            return self._create_default_result()
    
    def _extract_features(self, road_features, lane_info):
        """提取特征"""
        features = {}
        
        # 1. 道路质心特征
        if 'centroid' in road_features and road_features['centroid'] is not None:
            try:
                cx, cy = road_features['centroid']
                centroid_offset = (cx - 400) / 400
                features['centroid_offset'] = max(-1.0, min(1.0, centroid_offset))
            except:
                pass
        
        # 2. 道路坚实度特征
        if 'solidity' in road_features:
            features['road_solidity'] = road_features['solidity']
        
        # 3. 道路面积特征
        if 'area' in road_features:
            features['road_area'] = road_features['area']
        
        # 4. 车道线特征
        left_lane = lane_info.get('left_lane')
        right_lane = lane_info.get('right_lane')
        
        if left_lane and right_lane:
            try:
                # 车道收敛度
                convergence = self._calculate_lane_convergence_safe(left_lane, right_lane)
                features['lane_convergence'] = convergence
                
                # 车道对称性
                symmetry = self._calculate_lane_symmetry_safe(left_lane, right_lane)
                features['lane_symmetry'] = symmetry
                
                # 车道平衡性
                balance = self._calculate_lane_balance(left_lane, right_lane)
                features['lane_balance'] = balance
                
            except Exception as e:
                # 只打印错误，不生成日志文件
                pass
        
        # 5. 路径特征
        future_path = lane_info.get('future_path')
        if future_path and 'center_path' in future_path:
            try:
                path_points = future_path.get('center_path', [])
                if len(path_points) >= 3:
                    curvature = self._calculate_path_curvature_simple(path_points)
                    features['path_curvature'] = curvature
            except Exception as e:
                # 只打印错误，不生成日志文件
                pass
        
        # 6. 检测质量
        detection_quality = lane_info.get('detection_quality', 0.0)
        features['detection_quality'] = detection_quality
        
        # 7. 历史一致性
        if self.direction_history:
            consistency = self._calculate_historical_consistency()
            features['historical_consistency'] = consistency
        
        return features
    
    def _calculate_lane_convergence_safe(self, left_lane, right_lane):
        """安全计算车道线收敛度"""
        try:
            left_func = left_lane.get('func')
            right_func = right_lane.get('func')
            
            if left_func and right_func:
                y_bottom = 600
                y_top = 240
                
                try:
                    left_bottom = float(left_func(y_bottom))
                    right_bottom = float(right_func(y_bottom))
                    left_top = float(left_func(y_top))
                    right_top = float(right_func(y_top))
                    
                    width_bottom = right_bottom - left_bottom
                    width_top = right_top - left_top
                    
                    if width_bottom > 0:
                        convergence = width_top / width_bottom
                        return max(0.3, min(3.0, convergence))
                except:
                    pass
        except:
            pass
        
        return 1.0
    
    def _calculate_lane_symmetry_safe(self, left_lane, right_lane):
        """安全计算车道对称性"""
        try:
            left_func = left_lane.get('func')
            right_func = right_lane.get('func')
            
            if left_func and right_func:
                y = 450
                
                try:
                    left_x = float(left_func(y))
                    right_x = float(right_func(y))
                    
                    center = 400
                    
                    left_dist = center - left_x
                    right_dist = right_x - center
                    
                    if left_dist + right_dist > 0:
                        symmetry = 1 - abs(left_dist - right_dist) / (left_dist + right_dist)
                        return max(0, min(1, symmetry))
                except:
                    pass
        except:
            pass
        
        return 0.5
    
    def _calculate_lane_balance(self, left_lane, right_lane):
        """计算车道平衡性"""
        try:
            left_func = left_lane.get('func')
            right_func = right_lane.get('func')
            
            if left_func and right_func:
                y = 525
                
                try:
                    left_x = float(left_func(y))
                    right_x = float(right_func(y))
                    
                    lane_center = (left_x + right_x) / 2
                    image_center = 400
                    
                    offset_ratio = (lane_center - image_center) / 200
                    
                    balance = 1 - min(1.0, abs(offset_ratio))
                    return max(0, balance)
                except:
                    pass
        except:
            pass
        
        return 0.5
    
    def _calculate_path_curvature_simple(self, path_points):
        """简单计算路径曲率"""
        if len(path_points) < 3:
            return 0.0
        
        try:
            p1 = np.array(path_points[0])
            p2 = np.array(path_points[len(path_points)//2])
            p3 = np.array(path_points[-1])
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = max(-1, min(1, cos_angle))
                angle = np.arccos(cos_angle)
                
                return min(1.0, angle / (np.pi / 3))
        except:
            pass
        
        return 0.0
    
    def _calculate_historical_consistency(self):
        """计算历史一致性"""
        if len(self.direction_history) < 2:
            return 0.5
        
        recent_directions = list(self.direction_history)
        
        direction_counts = defaultdict(int)
        for direction in recent_directions:
            direction_counts[direction] += 1
        
        most_common_count = max(direction_counts.values())
        consistency = most_common_count / len(recent_directions)
        
        return consistency
    
    def _predict_direction_improved(self, features):
        """改进的方向预测"""
        probabilities = {'直行': 0.4, '左转': 0.3, '右转': 0.3}
        
        # 1. 基于道路质心偏移
        if 'centroid_offset' in features:
            offset = features['centroid_offset']
            
            if abs(offset) < 0.15:
                probabilities['直行'] += 0.25
            elif offset < -0.15:
                probabilities['左转'] += abs(offset) * 0.8
            elif offset > 0.15:
                probabilities['右转'] += offset * 0.8
        
        # 2. 基于车道收敛度
        if 'lane_convergence' in features:
            convergence = features['lane_convergence']
            
            if convergence < 0.7:
                probabilities['左转'] += (0.7 - convergence) * 0.4
                probabilities['右转'] += (0.7 - convergence) * 0.4
                probabilities['直行'] -= 0.2
            elif convergence > 1.3:
                probabilities['直行'] += 0.2
            elif 0.9 <= convergence <= 1.1:
                probabilities['直行'] += 0.15
        
        # 3. 基于车道对称性
        if 'lane_symmetry' in features:
            symmetry = features['lane_symmetry']
            
            if symmetry > 0.7:
                probabilities['直行'] += 0.15
            elif symmetry < 0.4:
                if 'centroid_offset' in features:
                    offset = features['centroid_offset']
                    if offset < -0.1:
                        probabilities['左转'] += 0.15
                    elif offset > 0.1:
                        probabilities['右转'] += 0.15
        
        # 4. 基于车道平衡性
        if 'lane_balance' in features:
            balance = features['lane_balance']
            
            if balance < 0.4:
                if 'centroid_offset' in features:
                    offset = features['centroid_offset']
                    if offset < -0.1:
                        probabilities['左转'] += 0.2
                    elif offset > 0.1:
                        probabilities['右转'] += 0.2
            elif balance > 0.7:
                probabilities['直行'] += 0.1
        
        # 5. 基于路径曲率
        if 'path_curvature' in features:
            curvature = features['path_curvature']
            
            if curvature > 0.3:
                if 'centroid_offset' in features:
                    offset = features['centroid_offset']
                    if offset < 0:
                        probabilities['左转'] += curvature * 0.5
                    else:
                        probabilities['右转'] += curvature * 0.5
            elif curvature < 0.1:
                probabilities['直行'] += 0.15
        
        # 确保概率非负
        for direction in probabilities:
            probabilities[direction] = max(0.01, probabilities[direction])
        
        # 归一化
        total = sum(probabilities.values())
        if total > 0:
            for direction in probabilities:
                probabilities[direction] /= total
        
        return probabilities
    
    def _calculate_confidence_improved(self, features, probabilities, lane_info):
        """改进的置信度计算"""
        confidence_factors = []
        
        # 1. 概率清晰度
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_probs) >= 2:
            best_prob = sorted_probs[0][1]
            second_prob = sorted_probs[1][1]
            
            if best_prob > 0:
                clarity = (best_prob - second_prob) / best_prob
                clarity_score = min(1.0, clarity * 2)
                confidence_factors.append(clarity_score * 0.4)
        
        # 2. 特征质量
        feature_scores = []
        
        if 'centroid_offset' in features:
            offset = abs(features['centroid_offset'])
            if offset > 0.2:
                feature_scores.append(0.9)
            elif offset > 0.1:
                feature_scores.append(0.7)
            elif offset > 0.05:
                feature_scores.append(0.5)
            else:
                feature_scores.append(0.3)
        
        if 'lane_convergence' in features:
            convergence = features['lane_convergence']
            if convergence < 0.7 or convergence > 1.3:
                feature_scores.append(0.8)
            elif 0.9 <= convergence <= 1.1:
                feature_scores.append(0.7)
            else:
                feature_scores.append(0.5)
        
        if 'lane_symmetry' in features:
            symmetry = features['lane_symmetry']
            if symmetry > 0.8 or symmetry < 0.3:
                feature_scores.append(0.8)
            elif 0.4 <= symmetry <= 0.6:
                feature_scores.append(0.5)
            else:
                feature_scores.append(0.6)
        
        if feature_scores:
            avg_feature_score = np.mean(feature_scores)
            confidence_factors.append(avg_feature_score * 0.3)
        
        # 3. 检测质量
        detection_quality = lane_info.get('detection_quality', 0.5)
        confidence_factors.append(detection_quality * 0.2)
        
        # 4. 特征一致性
        consistency_score = self._evaluate_feature_consistency(features, probabilities)
        confidence_factors.append(consistency_score * 0.1)
        
        # 综合置信度
        if confidence_factors:
            confidence = np.mean(confidence_factors)
            
            if confidence < 0.3:
                confidence = confidence * 0.9
            elif confidence < 0.6:
                confidence = 0.3 + (confidence - 0.3) * 1.3
            else:
                confidence = 0.6 + (confidence - 0.6) * 1.1
            
            return min(max(confidence, 0.0), 1.0)
        else:
            return 0.5
    
    def _evaluate_feature_consistency(self, features, probabilities):
        """评估特征一致性"""
        if not features:
            return 0.5
        
        best_direction = max(probabilities.items(), key=lambda x: x[1])[0]
        
        consistency_scores = []
        
        if best_direction == '直行':
            if 'centroid_offset' in features:
                offset = abs(features['centroid_offset'])
                if offset < 0.1:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(1 - min(1.0, offset))
            
            if 'lane_convergence' in features:
                convergence = features['lane_convergence']
                if 0.9 <= convergence <= 1.1:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(1 - min(1.0, abs(convergence - 1) / 0.5))
            
            if 'lane_symmetry' in features:
                symmetry = features['lane_symmetry']
                consistency_scores.append(symmetry)
        
        elif best_direction == '左转':
            if 'centroid_offset' in features:
                offset = features['centroid_offset']
                if offset < -0.1:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(max(0, 1 - (offset + 0.1) / 0.3))
            
            if 'lane_convergence' in features:
                convergence = features['lane_convergence']
                if convergence < 0.9:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(max(0, 1 - (convergence - 0.9) / 0.4))
            
            if 'lane_symmetry' in features:
                symmetry = features['lane_symmetry']
                if symmetry < 0.6:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(max(0, 1 - (symmetry - 0.6) / 0.4))
        
        elif best_direction == '右转':
            if 'centroid_offset' in features:
                offset = features['centroid_offset']
                if offset > 0.1:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(max(0, 1 - (0.1 - offset) / 0.3))
            
            if 'lane_convergence' in features:
                convergence = features['lane_convergence']
                if convergence < 0.9:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(max(0, 1 - (convergence - 0.9) / 0.4))
            
            if 'lane_symmetry' in features:
                symmetry = features['lane_symmetry']
                if symmetry < 0.6:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(max(0, 1 - (symmetry - 0.6) / 0.4))
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _get_final_direction_improved(self, probabilities, confidence):
        """改进的最终方向决策"""
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_probs:
            return '直行'
        
        best_direction, best_prob = sorted_probs[0]
        second_direction, second_prob = sorted_probs[1] if len(sorted_probs) > 1 else ('', 0)
        
        min_confidence = getattr(self.config, 'min_confidence_for_direction', 0.25)
        
        if confidence < 0.15:
            return '未知'
        elif confidence < min_confidence:
            if best_prob > 0.5 and best_prob > second_prob * 1.8:
                return best_direction
            elif best_prob > 0.6:
                return best_direction
            else:
                return '未知'
        elif confidence < 0.5:
            if best_prob > second_prob * 1.3:
                return best_direction
            elif best_prob > 0.4:
                return best_direction
            else:
                if self.direction_history:
                    historical_direction = self._get_historical_direction()
                    if historical_direction != '未知':
                        return historical_direction
                return best_direction
        else:
            if best_prob > second_prob * 1.2:
                return best_direction
            elif best_prob > 0.35:
                return best_direction
            else:
                if self.direction_history:
                    historical_direction = self._get_historical_direction()
                    if historical_direction != '未知':
                        return historical_direction
                return best_direction
    
    def _get_historical_direction(self):
        """获取历史主要方向"""
        if not self.direction_history:
            return '未知'
        
        recent = list(self.direction_history)[-3:]
        direction_counts = defaultdict(int)
        
        for direction in recent:
            if direction != '未知':
                direction_counts[direction] += 1
        
        if not direction_counts:
            return '未知'
        
        most_common = max(direction_counts.items(), key=lambda x: x[1])
        return most_common[0]
    
    def _apply_historical_smoothing(self, direction, confidence):
        """应用历史平滑"""
        if len(self.direction_history) < 2:
            return direction, confidence
        
        recent_directions = list(self.direction_history)[-4:]
        recent_confidences = list(self.confidence_history)[-4:]
        
        direction_counts = defaultdict(int)
        for d in recent_directions:
            if d != '未知':
                direction_counts[d] += 1
        
        if not direction_counts:
            return direction, confidence
        
        most_common, count = max(direction_counts.items(), key=lambda x: x[1])
        frequency = count / len(recent_directions)
        
        if recent_confidences:
            historical_confidence = np.mean(recent_confidences)
        else:
            historical_confidence = 0.5
        
        if frequency > 0.75 and confidence < 0.4:
            smoothed_confidence = historical_confidence * 0.8
            return most_common, smoothed_confidence
        
        elif most_common != direction and frequency > 0.6 and confidence < 0.5:
            smoothing_factor = min(0.6, frequency)
            smoothed_confidence = (
                confidence * (1 - smoothing_factor) + 
                historical_confidence * smoothing_factor
            )
            
            if historical_confidence > 0.6:
                return most_common, smoothed_confidence
        
        return direction, confidence
    
    def _generate_detailed_reasoning(self, features, probabilities, final_direction, confidence):
        """生成详细推理说明"""
        reasoning_parts = []
        
        key_features = []
        
        if 'centroid_offset' in features:
            offset = features['centroid_offset']
            if offset < -0.15:
                key_features.append("道路明显偏左")
            elif offset < -0.05:
                key_features.append("道路略偏左")
            elif offset > 0.15:
                key_features.append("道路明显偏右")
            elif offset > 0.05:
                key_features.append("道路略偏右")
            else:
                key_features.append("道路居中")
        
        if 'lane_convergence' in features:
            conv = features['lane_convergence']
            if conv < 0.7:
                key_features.append("车道明显收敛")
            elif conv > 1.3:
                key_features.append("车道发散")
            elif 0.9 <= conv <= 1.1:
                key_features.append("车道平行")
        
        if 'lane_symmetry' in features:
            symmetry = features['lane_symmetry']
            if symmetry > 0.8:
                key_features.append("车道对称")
            elif symmetry < 0.4:
                key_features.append("车道不对称")
        
        if key_features:
            reasoning_parts.append("特征：" + "，".join(key_features))
        
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_probs) >= 2:
            best_dir, best_prob = sorted_probs[0]
            second_dir, second_prob = sorted_probs[1]
            
            if best_prob > second_prob * 1.5:
                reasoning_parts.append(f"{best_dir}明显占优({best_prob:.0%})")
            elif best_prob > second_prob * 1.2:
                reasoning_parts.append(f"{best_dir}稍占优势({best_prob:.0%})")
            else:
                reasoning_parts.append("方向接近")
        
        if confidence >= 0.7:
            confidence_text = "高置信度"
        elif confidence >= 0.5:
            confidence_text = "中等置信度"
        elif confidence >= 0.3:
            confidence_text = "低置信度"
        else:
            confidence_text = "置信度不足"
        
        reasoning_parts.append(confidence_text)
        
        if final_direction != '未知':
            reasoning_parts.append(f"决策：{final_direction}")
        else:
            reasoning_parts.append("决策：无法确定")
        
        return " | ".join(reasoning_parts)
    
    def _fallback_direction_analysis(self, road_features, lane_info):
        """回退策略：当特征不足时使用"""
        direction = '直行'
        confidence = 0.3
        
        if 'centroid' in road_features and road_features['centroid'] is not None:
            cx, cy = road_features['centroid']
            if cx < 350:
                direction = '左转'
                confidence = 0.4
            elif cx > 450:
                direction = '右转'
                confidence = 0.4
        
        left_lane = lane_info.get('left_lane')
        right_lane = lane_info.get('right_lane')
        
        if left_lane and not right_lane:
            direction = '右转'
            confidence = 0.35
        elif right_lane and not left_lane:
            direction = '左转'
            confidence = 0.35
        
        detection_quality = lane_info.get('detection_quality', 0.0)
        if detection_quality > 0.7:
            confidence = min(1.0, confidence * 1.3)
        
        if direction == '直行':
            probabilities = {'直行': 0.7, '左转': 0.15, '右转': 0.15}
        elif direction == '左转':
            probabilities = {'左转': 0.7, '直行': 0.15, '右转': 0.15}
        else:
            probabilities = {'右转': 0.7, '直行': 0.15, '左转': 0.15}
        
        return {
            'direction': direction,
            'confidence': confidence,
            'probabilities': probabilities,
            'features': {'fallback': True},
            'reasoning': '特征不足，使用回退策略'
        }
    
    def _create_default_result(self):
        """创建默认结果"""
        return {
            'direction': '直行',
            'confidence': 0.2,
            'probabilities': {'直行': 0.5, '左转': 0.25, '右转': 0.25},
            'features': {},
            'reasoning': '检测失败，使用默认值'
        }