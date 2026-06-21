"""
飞行评分系统 - 根据飞行稳定性、手势准确率、轨迹平滑度进行综合评分

评分维度：
1. 飞行稳定性 (40分) - 速度波动率、姿态稳定性
2. 手势准确率 (35分) - 置信度均值、有效命令占比
3. 轨迹平滑度 (25分) - 加速度变化率、急停急转检测

评级标准：
- S级: 90-100分 (大师级飞行)
- A级: 80-89分  (优秀飞行)
- B级: 65-79分  (良好飞行)
- C级: 50-64分  (一般飞行)
- D级: 0-49分   (需要改进)
"""

import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


class FlightScorer:
    """飞行评分系统"""

    # 评分维度权重
    WEIGHT_STABILITY = 0.40    # 飞行稳定性
    WEIGHT_ACCURACY = 0.35     # 手势准确率
    WEIGHT_SMOOTHNESS = 0.25   # 轨迹平滑度

    def __init__(self):
        # ===== 飞行稳定性相关 =====
        self.speed_history: List[float] = []           # 速度历史 (m/s)
        self.acceleration_history: List[float] = []    # 加速度历史 (m/s²)
        self.velocity_samples: deque = deque(maxlen=60) # 速度采样 (最近60帧)
        self.stability_score: float = 0.0

        # ===== 手势准确率相关 =====
        self.confidence_history: List[float] = []       # 置信度历史
        self.total_gesture_attempts: int = 0            # 总尝试次数
        self.valid_commands: int = 0                    # 有效命令数
        self.accuracy_score: float = 0.0

        # ===== 轨迹平滑度相关 =====
        self.position_history: deque = deque(maxlen=120) # 位置历史
        self.jerk_history: List[float] = []             # 急动度历史 (加速度变化率)
        self.sudden_stops: int = 0                      # 急停次数
        self.sharp_turns: int = 0                       # 急转弯次数
        self.smoothness_score: float = 0.0

        # ===== 总评分 =====
        self.total_score: float = 0.0
        self.grade: str = "N/A"
        self.finalized: bool = False

        # ===== 额外统计 =====
        self.max_speed: float = 0.0
        self.max_acceleration: float = 0.0
        self.max_jerk: float = 0.0

        # 阈值常量
        self.SMOOTH_SPEED_THRESHOLD = 0.3      # 速度波动阈值 (m/s)
        self.SUDDEN_STOP_THRESHOLD = 5.0       # 急停加速度阈值 (m/s²)
        self.SHARP_TURN_JERK_THRESHOLD = 10.0  # 急转弯急动度阈值 (m/s³)
        self.JERK_SMOOTH_THRESHOLD = 2.0       # 平滑急动度阈值

    def update(self, drone_state: dict, gesture_confidence: float,
               command_executed: bool, dt: float):
        """
        每帧更新评分数据

        Args:
            drone_state: 无人机状态字典
            gesture_confidence: 当前手势置信度 (0-1)
            command_executed: 是否成功执行了命令
            dt: 时间增量
        """
        if dt <= 0:
            return

        position = np.array(drone_state['position'])
        velocity = drone_state['velocity']
        speed = float(np.linalg.norm(velocity))

        # ===== 稳定性数据 =====
        self.speed_history.append(speed)
        self.velocity_samples.append(velocity.copy())

        # 计算加速度
        if len(self.speed_history) >= 2:
            acceleration = abs(self.speed_history[-1] - self.speed_history[-2]) / dt
            self.acceleration_history.append(acceleration)
            if acceleration > self.max_acceleration:
                self.max_acceleration = acceleration

        if speed > self.max_speed:
            self.max_speed = speed

        # ===== 手势准确率数据 =====
        if gesture_confidence > 0:
            self.confidence_history.append(gesture_confidence)
            self.total_gesture_attempts += 1
            if command_executed:
                self.valid_commands += 1

        # ===== 轨迹平滑度数据 =====
        self.position_history.append(position.copy())

        # 计算急动度 (Jerk = 加速度变化率)
        if len(self.acceleration_history) >= 2:
            jerk = abs(self.acceleration_history[-1] - self.acceleration_history[-2]) / dt
            self.jerk_history.append(jerk)
            if jerk > self.max_jerk:
                self.max_jerk = jerk

            # 检测急停
            if self.acceleration_history[-1] > self.SUDDEN_STOP_THRESHOLD:
                self.sudden_stops += 1

            # 检测急转弯
            if jerk > self.SHARP_TURN_JERK_THRESHOLD:
                self.sharp_turns += 1

    def finalize(self) -> Tuple[float, str]:
        """结束评分，计算最终分数和等级"""
        self._calculate_stability_score()
        self._calculate_accuracy_score()
        self._calculate_smoothness_score()

        # 加权总分
        self.total_score = (
            self.stability_score * self.WEIGHT_STABILITY +
            self.accuracy_score * self.WEIGHT_ACCURACY +
            self.smoothness_score * self.WEIGHT_SMOOTHNESS
        )

        # 确定等级
        self.grade = self._get_grade(self.total_score)
        self.finalized = True

        return self.total_score, self.grade

    def _calculate_stability_score(self):
        """计算飞行稳定性分数 (0-100)"""
        if len(self.speed_history) < 3:
            self.stability_score = 50.0  # 数据不足给基础分
            return

        speeds = np.array(self.speed_history)
        accelerations = np.array(self.acceleration_history)

        # 1. 速度波动率 (占60%) - 标准差越小越稳定
        speed_std = float(np.std(speeds))
        speed_stability = max(0, 100 - speed_std * 30)  # std=3.3 时得0分

        # 2. 加速度平滑度 (占40%) - 加速度方差越小越稳定
        if len(accelerations) > 0:
            accel_std = float(np.std(accelerations))
            accel_stability = max(0, 100 - accel_std * 20)  # std=5 时得0分
        else:
            accel_stability = 50

        self.stability_score = speed_stability * 0.6 + accel_stability * 0.4

    def _calculate_accuracy_score(self):
        """计算手势准确率分数 (0-100)"""
        if self.total_gesture_attempts < 5:
            self.accuracy_score = 50.0  # 数据不足给基础分
            return

        # 1. 平均置信度 (占50%)
        avg_confidence = float(np.mean(self.confidence_history))
        confidence_score = avg_confidence * 100  # 置信度直接映射

        # 2. 有效命令率 (占50%)
        command_rate = self.valid_commands / max(self.total_gesture_attempts, 1)
        command_score = command_rate * 100

        self.accuracy_score = confidence_score * 0.5 + command_score * 0.5

    def _calculate_smoothness_score(self):
        """计算轨迹平滑度分数 (0-100)"""
        if len(self.jerk_history) < 3:
            self.smoothness_score = 50.0
            return

        jerks = np.array(self.jerk_history)
        flight_duration = len(self.jerk_history)  # 近似帧数

        # 1. 急动度均值 (占40%) - 越小越平滑
        avg_jerk = float(np.mean(jerks))
        jerk_score = max(0, 100 - avg_jerk * 10)

        # 2. 急停惩罚 (占30%) - 每10秒超过1次急停开始扣分
        stops_per_10s = self.sudden_stops / max(flight_duration / 600, 1)  # 假设60fps
        stop_penalty = min(30, stops_per_10s * 10)  # 最多扣30分
        stop_score = 30 - stop_penalty

        # 3. 急转弯惩罚 (占30%) - 每10秒超过2次急转开始扣分
        turns_per_10s = self.sharp_turns / max(flight_duration / 600, 1)
        turn_penalty = min(30, turns_per_10s * 8)
        turn_score = 30 - turn_penalty

        self.smoothness_score = jerk_score * 0.4 + stop_score + turn_score

    @staticmethod
    def _get_grade(score: float) -> str:
        """根据分数获取等级"""
        if score >= 90:
            return "S"
        elif score >= 80:
            return "A"
        elif score >= 65:
            return "B"
        elif score >= 50:
            return "C"
        else:
            return "D"

    def get_realtime_score(self) -> dict:
        """获取实时评分（用于界面显示，不修改内部状态）"""
        # 临时计算各维度分数
        temp_scorer = FlightScorer.__new__(FlightScorer)
        temp_scorer.speed_history = self.speed_history.copy()
        temp_scorer.acceleration_history = self.acceleration_history.copy()
        temp_scorer.confidence_history = self.confidence_history.copy()
        temp_scorer.total_gesture_attempts = self.total_gesture_attempts
        temp_scorer.valid_commands = self.valid_commands
        temp_scorer.jerk_history = self.jerk_history.copy()
        temp_scorer.sudden_stops = self.sudden_stops
        temp_scorer.sharp_turns = self.sharp_turns

        temp_scorer._calculate_stability_score()
        temp_scorer._calculate_accuracy_score()
        temp_scorer._calculate_smoothness_score()

        total = (
            temp_scorer.stability_score * self.WEIGHT_STABILITY +
            temp_scorer.accuracy_score * self.WEIGHT_ACCURACY +
            temp_scorer.smoothness_score * self.WEIGHT_SMOOTHNESS
        )

        return {
            'total': total,
            'grade': self._get_grade(total),
            'stability': temp_scorer.stability_score,
            'accuracy': temp_scorer.accuracy_score,
            'smoothness': temp_scorer.smoothness_score,
            'weights': {
                'stability': self.WEIGHT_STABILITY,
                'accuracy': self.WEIGHT_ACCURACY,
                'smoothness': self.WEIGHT_SMOOTHNESS,
            }
        }

    def get_report(self) -> dict:
        """获取完整评分报告"""
        if not self.finalized:
            self.finalize()

        avg_confidence = float(np.mean(self.confidence_history)) if self.confidence_history else 0
        command_rate = (self.valid_commands / max(self.total_gesture_attempts, 1) * 100
                        ) if self.total_gesture_attempts > 0 else 0

        return {
            'total_score': self.total_score,
            'grade': self.grade,
            'stability_score': self.stability_score,
            'accuracy_score': self.accuracy_score,
            'smoothness_score': self.smoothness_score,
            'details': {
                'stability': {
                    'speed_std': float(np.std(self.speed_history)) if len(self.speed_history) > 1 else 0,
                    'max_speed': self.max_speed,
                    'max_acceleration': self.max_acceleration,
                    'samples': len(self.speed_history),
                },
                'accuracy': {
                    'avg_confidence': avg_confidence,
                    'total_attempts': self.total_gesture_attempts,
                    'valid_commands': self.valid_commands,
                    'command_rate': command_rate,
                },
                'smoothness': {
                    'avg_jerk': float(np.mean(self.jerk_history)) if self.jerk_history else 0,
                    'max_jerk': self.max_jerk,
                    'sudden_stops': self.sudden_stops,
                    'sharp_turns': self.sharp_turns,
                }
            },
            'weights': {
                'stability': self.WEIGHT_STABILITY,
                'accuracy': self.WEIGHT_ACCURACY,
                'smoothness': self.WEIGHT_SMOOTHNESS,
            }
        }

    def print_report(self):
        """打印评分报告到控制台"""
        report = self.get_report()

        print("\n" + "=" * 60)
        print("              飞 行 评 分 报 告")
        print("=" * 60)

        # 总评
        grade_display = {
            'S': '⭐ 大师级飞行 (S)',
            'A': '🌟 优秀飞行 (A)',
            'B': '👍 良好飞行 (B)',
            'C': '📝 一般飞行 (C)',
            'D': '💪 需要改进 (D)',
        }
        print(f"\n  🏆 综合评分: {report['total_score']:.1f}/100")
        print(f"  🎖  飞行等级: {grade_display.get(report['grade'], report['grade'])}")
        print()

        # 各维度得分
        print("-" * 60)
        print("  维度评分明细:")
        print(f"  ├─ 飞行稳定性: {report['stability_score']:.1f}/100 (权重 {self.WEIGHT_STABILITY*100:.0f}%)")
        print(f"  ├─ 手势准确率: {report['accuracy_score']:.1f}/100 (权重 {self.WEIGHT_ACCURACY*100:.0f}%)")
        print(f"  └─ 轨迹平滑度: {report['smoothness_score']:.1f}/100 (权重 {self.WEIGHT_SMOOTHNESS*100:.0f}%)")
        print()

        # 详细数据
        details = report['details']
        print("-" * 60)
        print("  详细分析:")

        stab = details['stability']
        print(f"  [稳定性] 速度波动(std): {stab['speed_std']:.3f} m/s")
        print(f"            最大速度: {stab['max_speed']:.2f} m/s")
        print(f"            最大加速度: {stab['max_acceleration']:.2f} m/s²")

        acc = details['accuracy']
        print(f"  [准确率] 平均置信度: {acc['avg_confidence']:.2%}")
        print(f"            有效命令率: {acc['command_rate']:.1f}% ({acc['valid_commands']}/{acc['total_attempts']})")

        smt = details['smoothness']
        print(f"  [平滑度] 平均急动度: {smt['avg_jerk']:.3f} m/s³")
        print(f"            急停次数: {smt['sudden_stops']}")
        print(f"            急转弯次数: {smt['sharp_turns']}")

        # 改进建议
        print()
        print("-" * 60)
        print("  改进建议:")
        suggestions = self._get_suggestions(report)
        for i, sug in enumerate(suggestions, 1):
            print(f"  {i}. {sug}")

        print("=" * 60)

    def _get_suggestions(self, report: dict) -> List[str]:
        """根据评分生成改进建议"""
        suggestions = []

        if report['stability_score'] < 70:
            suggestions.append("放慢飞行速度，减少急加速操作，保持匀速飞行")
        if report['stability_score'] < 50:
            suggestions.append("避免频繁切换方向，给无人机稳定的飞行节奏")

        if report['accuracy_score'] < 70:
            suggestions.append("确保手势在摄像头前清晰可见，提高手势识别准确度")
        if report['accuracy_score'] < 50:
            suggestions.append("调整摄像头角度，保持手势稳定不抖动")

        if report['smoothness_score'] < 70:
            suggestions.append("避免急停急转，使用渐进式加速和减速")
        if report['smoothness_score'] < 50:
            suggestions.append("规划平滑的飞行路线，减少突然改变方向")

        if not suggestions:
            suggestions.append("表现优秀！继续保持稳定精准的飞行控制")

        return suggestions

    def reset(self):
        """重置评分系统"""
        self.__init__()
