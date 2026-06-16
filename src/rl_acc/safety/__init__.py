"""
安全约束模块 - Safety Constraints Module
实现ACC系统的多层次安全约束机制
包括距离约束、速度约束、加速度约束、TTC约束等
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    """安全等级枚举"""
    SAFE = "safe"           # 安全
    CAUTION = "caution"     # 注意
    WARNING = "warning"     # 警告
    DANGER = "danger"       # 危险
    EMERGENCY = "emergency" # 紧急


@dataclass
class SafetyConstraints:
    """安全约束参数"""
    # 距离约束
    min_safe_distance: float = 10.0      # 最小安全距离（米）
    warning_distance: float = 20.0       # 警告距离（米）
    caution_distance: float = 30.0        # 注意距离（米）

    # 速度约束
    max_speed: float = 35.0              # 最大速度（米/秒）
    min_speed: float = 0.0               # 最小速度（米/秒）

    # 加速度约束
    max_acceleration: float = 2.0         # 最大加速度（米/秒²）
    max_deceleration: float = -5.0        # 最大减速度（米/秒²）
    comfortable_acceleration: float = 1.5  # 舒适加速度（米/秒²）
    comfortable_deceleration: float = -2.0 # 舒适减速度（米/秒²）

    # TTC约束
    ttc_danger_threshold: float = 3.0      # TTC危险阈值（秒）
    ttc_warning_threshold: float = 5.0    # TTC警告阈值（秒）

    # 反应时间
    driver_reaction_time: float = 1.5     # 驾驶员反应时间（秒）


@dataclass
class SafetyStatus:
    """安全状态"""
    level: SafetyLevel
    distance_safe: bool
    speed_safe: bool
    acceleration_safe: bool
    ttc_safe: bool
    message: str
    recommended_action: str


class SafetyConstraintsManager:
    """
    安全约束管理器
    实现多层次安全约束检查和控制
    """

    def __init__(self, constraints: Optional[SafetyConstraints] = None):
        """
        初始化安全约束管理器

        Args:
            constraints: 安全约束参数，如果为None则使用默认值
        """
        self.constraints = constraints or SafetyConstraints()

    def check_distance_safety(self,
                             distance: float,
                             ego_speed: float,
                             target_speed: float) -> Tuple[bool, SafetyLevel, str]:
        """
        检查距离安全性

        Args:
            distance: 与前车距离（米）
            ego_speed: 自车速度（米/秒）
            target_speed: 前车速度（米/秒）

        Returns:
            (是否安全, 安全等级, 消息)
        """
        if distance <= 0:
            return False, SafetyLevel.EMERGENCY, "距离无效"

        # 计算安全距离
        relative_speed = ego_speed - target_speed
        if relative_speed > 0:
            # 正在接近，需要更大的安全距离
            time_gap = distance / ego_speed if ego_speed > 0 else 0
            safe_distance = self.constraints.min_safe_distance + ego_speed * time_gap * 0.5
        else:
            # 正在远离或保持距离
            safe_distance = self.constraints.min_safe_distance

        # 分级判断
        if distance < safe_distance * 0.5:
            return False, SafetyLevel.EMERGENCY, f"距离过近: {distance:.1f}m < {safe_distance*0.5:.1f}m"
        elif distance < safe_distance:
            return False, SafetyLevel.DANGER, f"距离危险: {distance:.1f}m < {safe_distance:.1f}m"
        elif distance < self.constraints.warning_distance:
            return False, SafetyLevel.WARNING, f"距离警告: {distance:.1f}m < {self.constraints.warning_distance:.1f}m"
        elif distance < self.constraints.caution_distance:
            return True, SafetyLevel.CAUTION, f"距离注意: {distance:.1f}m"
        else:
            return True, SafetyLevel.SAFE, f"距离安全: {distance:.1f}m"

    def check_speed_safety(self,
                          ego_speed: float,
                          target_speed: Optional[float] = None) -> Tuple[bool, SafetyLevel, str]:
        """
        检查速度安全性

        Args:
            ego_speed: 自车速度（米/秒）
            target_speed: 前车速度（米/秒），如果为None则只检查绝对速度

        Returns:
            (是否安全, 安全等级, 消息)
        """
        # 检查绝对速度
        if ego_speed > self.constraints.max_speed:
            return False, SafetyLevel.WARNING, f"超速: {ego_speed:.1f}m/s > {self.constraints.max_speed:.1f}m/s"
        elif ego_speed < self.constraints.min_speed:
            return False, SafetyLevel.WARNING, f"速度过低: {ego_speed:.1f}m/s < {self.constraints.min_speed:.1f}m/s"

        # 检查与前车速度差
        if target_speed is not None:
            speed_diff = ego_speed - target_speed
            if speed_diff > 10:  # 超过前车速度10m/s
                return True, SafetyLevel.CAUTION, f"速度差较大: +{speed_diff:.1f}m/s"

        return True, SafetyLevel.SAFE, f"速度正常: {ego_speed:.1f}m/s"

    def check_acceleration_safety(self,
                                 acceleration: float) -> Tuple[bool, SafetyLevel, str]:
        """
        检查加速度安全性

        Args:
            acceleration: 加速度值（米/秒²）

        Returns:
            (是否安全, 安全等级, 消息)
        """
        # 检查是否超出物理限制
        if acceleration > self.constraints.max_acceleration:
            return False, SafetyLevel.WARNING, f"加速度过大: {acceleration:.2f}m/s²"
        elif acceleration < self.constraints.max_deceleration:
            return False, SafetyLevel.WARNING, f"减速度过大: {acceleration:.2f}m/s²"

        # 检查是否舒适
        if acceleration > self.constraints.comfortable_acceleration:
            return True, SafetyLevel.CAUTION, f"加速度较高: {acceleration:.2f}m/s²"
        elif acceleration < -abs(self.constraints.comfortable_deceleration):
            return True, SafetyLevel.CAUTION, f"减速度较高: {acceleration:.2f}m/s²"

        return True, SafetyLevel.SAFE, f"加速度正常: {acceleration:.2f}m/s²"

    def check_ttc_safety(self,
                        distance: float,
                        relative_speed: float) -> Tuple[bool, SafetyLevel, float, str]:
        """
        检查TTC（碰撞时间）安全性

        Args:
            distance: 与前车距离（米）
            relative_speed: 相对速度（米/秒），正数表示接近

        Returns:
            (是否安全, 安全等级, TTC值, 消息)
        """
        # TTC计算：distance / relative_speed
        # 只在正在接近时计算TTC
        if relative_speed <= 0:
            return True, SafetyLevel.SAFE, float('inf'), "目标远离或静止"

        TTC = distance / relative_speed

        # 分级判断
        if TTC < 1.0:
            return False, SafetyLevel.EMERGENCY, TTC, f"紧急: TTC={TTC:.2f}s < 1.0s"
        elif TTC < self.constraints.ttc_danger_threshold:
            return False, SafetyLevel.DANGER, TTC, f"危险: TTC={TTC:.2f}s < {self.constraints.ttc_danger_threshold:.1f}s"
        elif TTC < self.constraints.ttc_warning_threshold:
            return True, SafetyLevel.WARNING, TTC, f"警告: TTC={TTC:.2f}s < {self.constraints.ttc_warning_threshold:.1f}s"
        else:
            return True, SafetyLevel.SAFE, TTC, f"安全: TTC={TTC:.2f}s"

    def calculate_safe_acceleration(self,
                                   distance: float,
                                   ego_speed: float,
                                   target_speed: float) -> float:
        """
        计算安全的加速度

        Args:
            distance: 与前车距离（米）
            ego_speed: 自车速度（米/秒）
            target_speed: 前车速度（米/秒）

        Returns:
            安全的加速度值（米/秒²）
        """
        relative_speed = ego_speed - target_speed

        # 计算最小安全距离（基于2秒反应时间）
        min_distance = max(self.constraints.min_safe_distance,
                          ego_speed * self.constraints.driver_reaction_time +
                          ego_speed * abs(relative_speed) * 0.5)

        # 如果距离足够，不需要减速
        if distance > min_distance * 1.5:
            return min(self.constraints.comfortable_acceleration, 0.5)

        # 紧急制动情况
        if distance < min_distance * 0.5:
            return self.constraints.max_deceleration

        # 计算需要的减速度
        decel_needed = (ego_speed ** 2 - target_speed ** 2) / (2 * (distance - min_distance * 0.3))
        decel_needed = max(decel_needed, self.constraints.max_deceleration)
        decel_needed = min(decel_needed, 0)

        return decel_needed

    def get_safe_distance(self,
                         ego_speed: float,
                         target_speed: float) -> float:
        """
        计算安全距离

        Args:
            ego_speed: 自车速度（米/秒）
            target_speed: 前车速度（米/秒）

        Returns:
            安全距离（米）
        """
        # 使用基于时间的距离模型
        time_gap = 2.0  # 2秒时间间隔
        relative_speed = abs(ego_speed - target_speed)

        # 基础安全距离
        base_distance = self.constraints.min_safe_distance

        # 基于速度的距离
        speed_distance = ego_speed * time_gap

        # 基于相对速度的距离
        relative_distance = relative_speed * 1.0

        safe_distance = base_distance + speed_distance + relative_distance

        return max(safe_distance, self.constraints.min_safe_distance)

    def comprehensive_check(self,
                           distance: float,
                           ego_speed: float,
                           target_speed: float,
                           acceleration: float) -> SafetyStatus:
        """
        综合安全检查

        Args:
            distance: 与前车距离（米）
            ego_speed: 自车速度（米/秒）
            target_speed: 前车速度（米/秒）
            acceleration: 当前加速度（米/秒²）

        Returns:
            SafetyStatus: 综合安全状态
        """
        # 执行各项检查
        dist_safe, dist_level, dist_msg = self.check_distance_safety(distance, ego_speed, target_speed)
        speed_safe, speed_level, speed_msg = self.check_speed_safety(ego_speed, target_speed)
        accel_safe, accel_level, accel_msg = self.check_acceleration_safety(acceleration)
        ttc_safe, ttc_level, TTC, ttc_msg = self.check_ttc_safety(distance, ego_speed - target_speed)

        # 确定最高安全等级
        levels = [dist_level, speed_level, accel_level, ttc_level]
        level_priority = {
            SafetyLevel.EMERGENCY: 4,
            SafetyLevel.DANGER: 3,
            SafetyLevel.WARNING: 2,
            SafetyLevel.CAUTION: 1,
            SafetyLevel.SAFE: 0
        }

        max_level = max(levels, key=lambda x: level_priority[x])

        # 生成综合消息
        messages = [dist_msg, speed_msg, accel_msg, ttc_msg]
        all_safe = dist_safe and speed_safe and accel_safe and ttc_safe

        if all_safe and max_level == SafetyLevel.SAFE:
            message = "所有检查通过，系统安全"
            action = "保持当前状态"
        elif max_level == SafetyLevel.EMERGENCY:
            message = "紧急情况！立即制动！"
            action = "紧急制动"
        elif max_level == SafetyLevel.DANGER:
            message = "危险！需要减速"
            action = "大幅减速"
        elif max_level == SafetyLevel.WARNING:
            message = "警告！注意安全"
            action = "适当减速"
        else:
            message = "注意：保持警惕"
            action = "谨慎驾驶"

        return SafetyStatus(
            level=max_level,
            distance_safe=dist_safe,
            speed_safe=speed_safe,
            acceleration_safe=accel_safe,
            ttc_safe=ttc_safe,
            message=message,
            recommended_action=action
        )

    def constrain_action(self,
                       desired_acceleration: float,
                       distance: float,
                       ego_speed: float,
                       target_speed: float) -> float:
        """
        约束动作，确保安全

        Args:
            desired_acceleration: 期望的加速度（米/秒²）
            distance: 与前车距离（米）
            ego_speed: 自车速度（米/秒）
            target_speed: 前车速度（米/秒）

        Returns:
            约束后的安全加速度（米/秒²）
        """
        # 计算安全加速度
        safe_accel = self.calculate_safe_acceleration(distance, ego_speed, target_speed)

        # 如果期望加速度不安全，使用安全加速度
        if desired_acceleration > self.constraints.max_acceleration:
            return self.constraints.max_acceleration
        elif desired_acceleration < safe_accel:
            return safe_accel
        elif desired_acceleration < self.constraints.max_deceleration:
            return self.constraints.max_deceleration

        return desired_acceleration
