"""
方向分析模块 - 负责分析道路方向并计算置信度
稳定优化版
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, Any, Tuple, List
import cv2

class DirectionAnalyzer:
    """方向分析器 - 稳定优化版本"""
    
    def __init__(self, config=None):
        if config is None:
            # 创建默认配置
            class DefaultConfig:
                min_confidence_for_direction = 0.25  # 降低阈值以提高判断率
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
        
        print("方向分析器已初始化，使用优化配置")
    
    def analyze(self, road_features: Dict[str, Any], lane_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析道路方向 - 主方法"""
        try:
            # 调试信息
            # print(f"道路特征可用: {list(road_features.keys())}")
            # print(f"车道信息可用: {list(lane_info.keys())}")
            
            # 提取特征
            features = self._extract_features(road_features, lane_info)
            
            # 如果没有足够特征，使用回退策略
            if len(features) < 2:
                # print("特征不足，使用回退策略")
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
            if confidence > 0.2:  # 降低历史记录门槛
                self.history.append(result)
                self.direction_history.append(final_direction)
                self.confidence_history.append(confidence)
            
            # 调试信息
            # print(f"方向分析结果: {final_direction}, 置信度: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            print(f"方向分析失败: {e}")
            import traceback
            traceback.print_exc()
            return self._create_default_result()
    
    def _extract_features(self, road_features: Dict[str, Any], lane_info: Dict[str, Any]) -> Dict[str, float]:
        """提取特征"""
        features = {}
        
        # 1. 道路质心特征
        if 'centroid' in road_features and road_features['centroid'] is not None:
            try:
                cx, cy = road_features['centroid']
                # 假设图像宽度为800，进行归一化
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
                
                # 车道宽度
                width = self._calculate_lane_width_safe(left_lane, right_lane)
                features['lane_width'] = width
                
            except Exception as e:
                print(f"车道特征提取失败: {e}")
        
        # 5. 路径特征
        future_path = lane_info.get('future_path')
        if future_path and 'center_path' in future_path:
            try:
                path_points = future_path.get('center_path', [])
                if len(path_points) >= 3:
                    curvature = self._calculate_path_curvature_simple(path_points)
                    features['path_curvature'] = curvature
                    
                    straightness = self._calculate_path_straightness_simple(path_points)
                    features['path_straightness'] = straightness
            except Exception as e:
                print(f"路径特征提取失败: {e}")
        
        # 6. 检测质量
        detection_quality = lane_info.get('detection_quality', 0.0)
        features['detection_quality'] = detection_quality
        
        # 7. 历史一致性
        if self.direction_history:
            consistency = self._calculate_historical_consistency()
            features['historical_consistency'] = consistency
        
        return features
    
    def _calculate_lane_convergence_safe(self, left_lane: Dict[str, Any], right_lane: Dict[str, Any]) -> float:
        """安全计算车道线收敛度"""
        try:
            left_func = left_lane.get('func')
            right_func = right_lane.get('func')
            
            if left_func and right_func:
                # 在顶部和底部计算宽度
                y_bottom = 600  # 图像底部
                y_top = 240    # 图像顶部 (600*0.4)
                
                try:
                    left_bottom = float(left_func(y_bottom))
                    right_bottom = float(right_func(y_bottom))
                    left_top = float(left_func(y_top))
                    right_top = float(right_func(y_top))
                    
                    width_bottom = right_bottom - left_bottom
                    width_top = right_top - left_top
                    
                    if width_bottom > 0:
                        convergence = width_top / width_bottom
                        # 限制在合理范围内
                        return max(0.3, min(3.0, convergence))
                except:
                    pass
        except:
            pass
        
        return 1.0  # 默认值
    
    def _calculate_lane_symmetry_safe(self, left_lane: Dict[str, Any], right_lane: Dict[str, Any]) -> float:
        """安全计算车道对称性"""
        try:
            left_func = left_lane.get('func')
            right_func = right_lane.get('func')
            
            if left_func and right_func:
                # 在图像中间采样
                y = 450  # 600 * 0.75
                
                try:
                    left_x = float(left_func(y))
                    right_x = float(right_func(y))
                    
                    # 假设图像中心在400
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
        
        return 0.5  # 默认值
    
    def _calculate_lane_balance(self, left_lane: Dict[str, Any], right_lane: Dict[str, Any]) -> float:
        """计算车道平衡性"""
        try:
            left_func = left_lane.get('func')
            right_func = right_lane.get('func')
            
            if left_func and right_func:
                # 在图像下半部分采样
                y = 525  # 600 * 0.875
                
                try:
                    left_x = float(left_func(y))
                    right_x = float(right_func(y))
                    
                    # 计算车道中心
                    lane_center = (left_x + right_x) / 2
                    image_center = 400  # 假设图像宽度800，中心在400
                    
                    # 计算偏移比例
                    offset_ratio = (lane_center - image_center) / 200  # 归一化到[-2, 2]
                    
                    # 转换为平衡分数（1表示居中，0表示严重偏移）
                    balance = 1 - min(1.0, abs(offset_ratio))
                    return max(0, balance)
                except:
                    pass
        except:
            pass
        
        return 0.5  # 默认值
    
    def _calculate_lane_width_safe(self, left_lane: Dict[str, Any], right_lane: Dict[str, Any]) -> float:
        """安全计算车道宽度"""
        try:
            left_func = left_lane.get('func')
            right_func = right_lane.get('func')
            
            if left_func and right_func:
                # 在图像底部采样
                y = 600
                
                try:
                    left_x = float(left_func(y))
                    right_x = float(right_func(y))
                    
                    if right_x > left_x:
                        width = right_x - left_x
                        # 归一化到0-1范围，假设合理宽度在200-400之间
                        normalized = (width - 200) / 200
                        return max(0, min(1, normalized))
                except:
                    pass
        except:
            pass
        
        return 0.5  # 默认值
    
    def _calculate_path_curvature_simple(self, path_points: List[Tuple[int, int]]) -> float:
        """简单计算路径曲率"""
        if len(path_points) < 3:
            return 0.0
        
        try:
            # 使用前中后三个点计算曲率
            p1 = np.array(path_points[0])
            p2 = np.array(path_points[len(path_points)//2])
            p3 = np.array(path_points[-1])
            
            # 计算向量
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 计算夹角
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = max(-1, min(1, cos_angle))
                angle = np.arccos(cos_angle)
                
                # 归一化到0-1范围
                return min(1.0, angle / (np.pi / 3))
        except:
            pass
        
        return 0.0
    
    def _calculate_path_straightness_simple(self, path_points: List[Tuple[int, int]]) -> float:
        """简单计算路径直线度"""
        if len(path_points) < 2:
            return 1.0
        
        try:
            # 计算首尾距离与总路径长度的比例
            total_length = 0
            for i in range(len(path_points) - 1):
                x1, y1 = path_points[i]
                x2, y2 = path_points[i+1]
                total_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            start_x, start_y = path_points[0]
            end_x, end_y = path_points[-1]
            direct_length = np.sqrt((end_x-start_x)**2 + (end_y-start_y)**2)
            
            if total_length > 0:
                straightness = direct_length / total_length
                return max(0, min(1, straightness))
        except:
            pass
        
        return 0.5
    
    def _calculate_historical_consistency(self) -> float:
        """计算历史一致性"""
        if len(self.direction_history) < 2:
            return 0.5
        
        recent_directions = list(self.direction_history)
        
        # 统计方向频率
        direction_counts = defaultdict(int)
        for direction in recent_directions:
            direction_counts[direction] += 1
        
        most_common_count = max(direction_counts.values())
        consistency = most_common_count / len(recent_directions)
        
        return consistency
    
    def _predict_direction_improved(self, features: Dict[str, float]) -> Dict[str, float]:
        """改进的方向预测"""
        # 基础概率
        probabilities = {'直行': 0.4, '左转': 0.3, '右转': 0.3}
        
        # 1. 基于道路质心偏移
        if 'centroid_offset' in features:
            offset = features['centroid_offset']
            
            if abs(offset) < 0.15:  # 基本居中
                probabilities['直行'] += 0.25
            elif offset < -0.15:  # 偏左
                probabilities['左转'] += abs(offset) * 0.8
            elif offset > 0.15:  # 偏右
                probabilities['右转'] += offset * 0.8
        
        # 2. 基于车道收敛度
        if 'lane_convergence' in features:
            convergence = features['lane_convergence']
            
            if convergence < 0.7:  # 明显收敛
                probabilities['左转'] += (0.7 - convergence) * 0.4
                probabilities['右转'] += (0.7 - convergence) * 0.4
                probabilities['直行'] -= 0.2
            elif convergence > 1.3:  # 明显发散
                probabilities['直行'] += 0.2
            elif 0.9 <= convergence <= 1.1:  # 基本平行
                probabilities['直行'] += 0.15
        
        # 3. 基于车道对称性
        if 'lane_symmetry' in features:
            symmetry = features['lane_symmetry']
            
            if symmetry > 0.7:  # 对称性好
                probabilities['直行'] += 0.15
            elif symmetry < 0.4:  # 对称性差
                # 结合质心偏移判断
                if 'centroid_offset' in features:
                    offset = features['centroid_offset']
                    if offset < -0.1:
                        probabilities['左转'] += 0.15
                    elif offset > 0.1:
                        probabilities['右转'] += 0.15
        
        # 4. 基于车道平衡性
        if 'lane_balance' in features:
            balance = features['lane_balance']
            
            if balance < 0.4:  # 严重不平衡
                if 'centroid_offset' in features:
                    offset = features['centroid_offset']
                    if offset < -0.1:
                        probabilities['左转'] += 0.2
                    elif offset > 0.1:
                        probabilities['右转'] += 0.2
            elif balance > 0.7:  # 平衡性好
                probabilities['直行'] += 0.1
        
        # 5. 基于路径曲率
        if 'path_curvature' in features:
            curvature = features['path_curvature']
            
            if curvature > 0.3:  # 明显弯曲
                if 'centroid_offset' in features:
                    offset = features['centroid_offset']
                    if offset < 0:
                        probabilities['左转'] += curvature * 0.5
                    else:
                        probabilities['右转'] += curvature * 0.5
            elif curvature < 0.1:  # 基本直线
                probabilities['直行'] += 0.15
        
        # 确保概率非负
        for direction in probabilities:
            probabilities[direction] = max(0.01, probabilities[direction])
        
        # 归一化
        total = sum(probabilities.values())
        if total > 0:
            for direction in probabilities:
                probabilities[direction] /= total
        
        # 确保总和为1（浮点误差）
        total = sum(probabilities.values())
        if total > 0:
            for direction in probabilities:
                probabilities[direction] /= total
        
        return probabilities
    
    def _calculate_confidence_improved(self, features: Dict[str, float],
                                    probabilities: Dict[str, float],
                                    lane_info: Dict[str, Any]) -> float:
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
        
        # 检查特征是否明显
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
            
            # 非线性调整
            if confidence < 0.3:
                confidence = confidence * 0.9  # 稍微降低低置信度
            elif confidence < 0.6:
                confidence = 0.3 + (confidence - 0.3) * 1.3  # 增强中等置信度
            else:
                confidence = 0.6 + (confidence - 0.6) * 1.1  # 稍微增强高置信度
            
            return min(max(confidence, 0.0), 1.0)
        else:
            return 0.5
    
    def _evaluate_feature_consistency(self, features: Dict[str, float],
                                    probabilities: Dict[str, float]) -> float:
        """评估特征一致性"""
        if not features:
            return 0.5
        
        # 确定优势方向
        best_direction = max(probabilities.items(), key=lambda x: x[1])[0]
        
        consistency_scores = []
        
        if best_direction == '直行':
            # 直行特征：居中、平行、对称
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
            # 左转特征：偏左、收敛、不对称
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
            # 右转特征：偏右、收敛、不对称
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
    
    def _get_final_direction_improved(self, probabilities: Dict[str, float],
                                    confidence: float) -> str:
        """改进的最终方向决策"""
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_probs:
            return '直行'  # 默认直行
        
        best_direction, best_prob = sorted_probs[0]
        second_direction, second_prob = sorted_probs[1] if len(sorted_probs) > 1 else ('', 0)
        
        # 获取最小置信度阈值
        min_confidence = getattr(self.config, 'min_confidence_for_direction', 0.25)
        
        # 决策逻辑（更宽松）
        if confidence < 0.15:  # 极低置信度
            return '未知'
        elif confidence < min_confidence:  # 低置信度
            # 如果概率优势明显，仍可判断
            if best_prob > 0.5 and best_prob > second_prob * 1.8:
                return best_direction
            elif best_prob > 0.6:  # 概率超过60%
                return best_direction
            else:
                return '未知'
        elif confidence < 0.5:  # 中等置信度
            if best_prob > second_prob * 1.3:
                return best_direction
            elif best_prob > 0.4:  # 概率超过40%
                return best_direction
            else:
                # 参考历史
                if self.direction_history:
                    historical_direction = self._get_historical_direction()
                    if historical_direction != '未知':
                        return historical_direction
                return best_direction  # 即使优势不明显，也返回最佳方向
        else:  # 高置信度
            if best_prob > second_prob * 1.2:
                return best_direction
            elif best_prob > 0.35:  # 概率超过35%
                return best_direction
            else:
                # 参考历史
                if self.direction_history:
                    historical_direction = self._get_historical_direction()
                    if historical_direction != '未知':
                        return historical_direction
                return best_direction  # 总是返回最佳方向
    
    def _get_historical_direction(self) -> str:
        """获取历史主要方向"""
        if not self.direction_history:
            return '未知'
        
        recent = list(self.direction_history)[-3:]  # 最近3次
        direction_counts = defaultdict(int)
        
        for direction in recent:
            if direction != '未知':
                direction_counts[direction] += 1
        
        if not direction_counts:
            return '未知'
        
        # 返回出现最频繁的方向
        most_common = max(direction_counts.items(), key=lambda x: x[1])
        return most_common[0]
    
    def _apply_historical_smoothing(self, direction: str, confidence: float) -> Tuple[str, float]:
        """应用历史平滑"""
        if len(self.direction_history) < 2:
            return direction, confidence
        
        recent_directions = list(self.direction_history)[-4:]
        recent_confidences = list(self.confidence_history)[-4:]
        
        # 统计历史方向
        direction_counts = defaultdict(int)
        for d in recent_directions:
            if d != '未知':
                direction_counts[d] += 1
        
        if not direction_counts:
            return direction, confidence
        
        most_common, count = max(direction_counts.items(), key=lambda x: x[1])
        frequency = count / len(recent_directions)
        
        # 计算历史平均置信度
        if recent_confidences:
            historical_confidence = np.mean(recent_confidences)
        else:
            historical_confidence = 0.5
        
        # 应用平滑规则
        if frequency > 0.75 and confidence < 0.4:
            # 历史一致性很高，当前置信度低，信任历史
            smoothed_confidence = historical_confidence * 0.8
            return most_common, smoothed_confidence
        
        elif most_common != direction and frequency > 0.6 and confidence < 0.5:
            # 历史与当前不一致，但历史一致性高，当前置信度不高
            smoothing_factor = min(0.6, frequency)
            smoothed_confidence = (
                confidence * (1 - smoothing_factor) + 
                historical_confidence * smoothing_factor
            )
            
            if historical_confidence > 0.6:
                return most_common, smoothed_confidence
        
        return direction, confidence
    
    def _generate_detailed_reasoning(self, features: Dict[str, float],
                                   probabilities: Dict[str, float],
                                   final_direction: str, confidence: float) -> str:
        """生成详细推理说明"""
        reasoning_parts = []
        
        # 添加关键特征说明
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
        
        # 添加概率说明
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
        
        # 添加置信度说明
        if confidence >= 0.7:
            confidence_text = "高置信度"
        elif confidence >= 0.5:
            confidence_text = "中等置信度"
        elif confidence >= 0.3:
            confidence_text = "低置信度"
        else:
            confidence_text = "置信度不足"
        
        reasoning_parts.append(confidence_text)
        
        # 添加最终决策说明
        if final_direction != '未知':
            reasoning_parts.append(f"决策：{final_direction}")
        else:
            reasoning_parts.append("决策：无法确定")
        
        return " | ".join(reasoning_parts)
    
    def _fallback_direction_analysis(self, road_features: Dict[str, Any],
                                   lane_info: Dict[str, Any]) -> Dict[str, Any]:
        """回退策略：当特征不足时使用"""
        direction = '直行'  # 默认
        confidence = 0.3
        
        # 1. 基于道路质心
        if 'centroid' in road_features and road_features['centroid'] is not None:
            cx, cy = road_features['centroid']
            if cx < 350:  # 质心偏左
                direction = '左转'
                confidence = 0.4
            elif cx > 450:  # 质心偏右
                direction = '右转'
                confidence = 0.4
        
        # 2. 基于车道线
        left_lane = lane_info.get('left_lane')
        right_lane = lane_info.get('right_lane')
        
        if left_lane and not right_lane:
            direction = '右转'  # 只能看到左车道线，可能右转
            confidence = 0.35
        elif right_lane and not left_lane:
            direction = '左转'  # 只能看到右车道线，可能左转
            confidence = 0.35
        
        # 3. 基于检测质量
        detection_quality = lane_info.get('detection_quality', 0.0)
        if detection_quality > 0.7:
            confidence = min(1.0, confidence * 1.3)
        
        # 创建概率分布
        if direction == '直行':
            probabilities = {'直行': 0.7, '左转': 0.15, '右转': 0.15}
        elif direction == '左转':
            probabilities = {'左转': 0.7, '直行': 0.15, '右转': 0.15}
        else:  # 右转
            probabilities = {'右转': 0.7, '直行': 0.15, '左转': 0.15}
        
        return {
            'direction': direction,
            'confidence': confidence,
            'probabilities': probabilities,
            'features': {'fallback': True},
            'reasoning': '特征不足，使用回退策略'
        }
    
    def _create_default_result(self) -> Dict[str, Any]:
        """创建默认结果"""
        return {
            'direction': '直行',
            'confidence': 0.2,
            'probabilities': {'直行': 0.5, '左转': 0.25, '右转': 0.25},
            'features': {},
            'reasoning': '检测失败，使用默认值'
        }