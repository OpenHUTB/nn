"""
质量评估模块 - 评估检测质量并生成改进建议
从旧版 lane_identification.py 迁移
"""

import numpy as np
from typing import Dict, Any, Tuple, List


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

        # 4. 图像质量
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

        base_quality = lane_info.get('detection_quality', 0.0)
        score += base_quality * 0.3
        details['base_quality'] = base_quality

        left_count = len(lane_info.get('left_lines', []))
        right_count = len(lane_info.get('right_lines', []))

        line_count_score = (min(left_count, 8) / 8.0 * 0.25 +
                            min(right_count, 8) / 8.0 * 0.25)
        score += line_count_score
        details['left_line_count'] = left_count
        details['right_line_count'] = right_count
        details['line_count_score'] = line_count_score

        left_lane = lane_info.get('left_lane')
        right_lane = lane_info.get('right_lane')

        model_score = 0.0
        if left_lane and right_lane:
            left_conf = left_lane.get('confidence', 0.0)
            right_conf = right_lane.get('confidence', 0.0)
            model_score = (left_conf + right_conf) / 2.0 * 0.3

            if left_lane.get('model_type') == right_lane.get('model_type'):
                model_score *= 1.1

            details['left_lane_confidence'] = left_conf
            details['right_lane_confidence'] = right_conf
            details['model_consistency'] = left_lane.get('model_type') == right_lane.get('model_type')

        score += model_score
        details['model_score'] = model_score

        if lane_info.get('future_path'):
            path_quality = lane_info['future_path'].get('prediction_quality', 0.5)
            score += path_quality * 0.15
            details['path_quality'] = path_quality

        return min(score, 1.0), details

    def _evaluate_road_detection(self, road_features: Dict[str, Any]) -> Tuple[float, Dict]:
        """评估道路检测质量"""
        details = {}
        score = 0.0

        if 'contour' in road_features:
            contour = road_features['contour']
            if len(contour) >= 4:
                score += 0.25
                details['contour_points'] = len(contour)
            else:
                details['contour_points'] = len(contour)

        if 'area' in road_features:
            area = road_features['area']
            area_score = min(area / 50000.0, 1.0) * 0.2
            score += area_score
            details['area'] = area
            details['area_score'] = area_score

        if 'solidity' in road_features:
            solidity = road_features['solidity']
            score += solidity * 0.25
            details['solidity'] = solidity

        if 'confidence' in road_features:
            confidence = road_features['confidence']
            score += confidence * 0.2
            details['confidence'] = confidence

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

        if 'contour' in road_features and lane_info.get('left_lane') and lane_info.get('right_lane'):
            position_score = 0.7
            consistency_scores.append(position_score)
            details['position_consistency'] = position_score

        if 'orientation' in road_features:
            road_orientation = road_features.get('orientation', 0)
            lane_orientations = []
            if lane_info.get('left_lane'):
                lane_orientations.append(45)

            if lane_orientations:
                avg_lane_orientation = np.mean(lane_orientations)
                orientation_diff = abs(road_orientation - avg_lane_orientation)
                orientation_score = max(0, 1 - orientation_diff / 90.0)
                consistency_scores.append(orientation_score)
                details['orientation_consistency'] = orientation_score

        if lane_info.get('left_lane') and lane_info.get('right_lane'):
            left_func = lane_info['left_lane'].get('func')
            right_func = lane_info['right_lane'].get('func')

            if left_func and right_func:
                y_points = [600, 500, 400]
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
                        width_cv = width_std / width_mean
                        width_consistency = max(0, 1 - width_cv)
                        consistency_scores.append(width_consistency)
                        details['width_consistency'] = width_consistency
                        details['avg_width'] = width_mean
                        details['width_std'] = width_std

        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
        else:
            avg_consistency = 0.5

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
        if quality_score > 0.8:
            return 1.2
        elif quality_score > 0.6:
            return 1.1
        elif quality_score > 0.4:
            return 1.0
        elif quality_score > 0.2:
            return 0.9
        else:
            return 0.8

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

        if scores.get('lane_detection', {}).get('details', {}).get('left_line_count', 0) < 3:
            recommendations.append("  - 左侧车道线数量不足，可能影响方向判断")

        if scores.get('lane_detection', {}).get('details', {}).get('right_line_count', 0) < 3:
            recommendations.append("  - 右侧车道线数量不足，可能影响方向判断")

        return recommendations
