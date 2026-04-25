"""
置信度校准模块 - 提高置信度准确性
从旧版 lane_identification.py 迁移
"""

import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, Any


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
        if confidence < 0.3:
            return confidence * 0.7
        elif confidence < 0.6:
            x = (confidence - 0.3) * 3.33
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
            adjustment -= 0.2
        elif feature_count > 6:
            adjustment += 0.1

        # 2. 特征质量校准
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
            return min(1.0, confidence * 1.1)
        elif context == "urban":
            return confidence * 0.95
        elif context == "rural":
            return confidence * 0.9
        else:
            return confidence

    def _apply_historical_calibration(self, confidence: float) -> float:
        """基于历史表现的校准"""
        if len(self.calibration_history) < 10:
            return confidence

        historical_confidences = [h['calibrated_confidence']
                                  for h in self.calibration_history]
        hist_avg = np.mean(historical_confidences)
        hist_std = np.std(historical_confidences)

        if abs(confidence - hist_avg) > 2 * hist_std:
            pull_factor = 0.3
            adjusted = confidence * (1 - pull_factor) + hist_avg * pull_factor
            return adjusted

        return confidence

    def update_performance(self, calibrated_confidence: float,
                           was_correct: bool, features: Dict[str, Any]):
        """更新性能统计"""
        if calibrated_confidence > 0.7 and was_correct:
            self.performance_stats['high_confidence_correct'] += 1

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
        }
