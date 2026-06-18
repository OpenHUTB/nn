"""
自动紧急制动系统 - Automatic Emergency Braking (AEB)
实现AEB功能，包括前方碰撞预警和自动制动
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AEBState(Enum):
    """AEB状态"""
    OFF = "off"                     # 关闭
    STANDBY = "standby"             # 待机
    WARNING = "warning"              # 警告
    PARTIAL_BRAKING = "partial"     # 部分制动
    FULL_BRAKING = "full"           # 全力制动


@dataclass
class AEBParameters:
    """AEB参数"""
    # 碰撞预警时间
    fcw_warning_time: float = 2.0   # 预警时间（秒）

    # 制动开始距离
    partial_braking_distance: float = 30.0   # 部分制动开始距离（米）
    full_braking_distance: float = 15.0     # 全力制动开始距离（米）

    # 制动减速度
    partial_braking_decel: float = -3.0      # 部分制动减速度（米/秒²）
    full_braking_decel: float = -6.0         # 全力制动减速度（米/秒²）

    # 最小速度阈值
    min_speed_for_aeb: float = 5.0           # AEB生效的最小速度（米/秒）


class AEBController:
    """
    AEB控制器
    实现自动紧急制动功能
    """

    def __init__(self, parameters: Optional[AEBParameters] = None):
        """
        初始化AEB控制器

        Args:
            parameters: AEB参数，如果为None则使用默认值
        """
        self.params = parameters or AEBParameters()
        self.state = AEBState.STANDBY
        self.last_warning_time = 0.0

    def calculate_braking_level(self,
                               distance: float,
                               relative_speed: float) -> Tuple[AEBState, float]:
        """
        计算制动等级

        Args:
            distance: 与前车距离（米）
            relative_speed: 相对速度（米/秒），正数表示接近

        Returns:
            (AEB状态, 建议减速度)
        """
        # 如果目标在远离或静止，不需要制动
        if relative_speed <= 0:
            self.state = AEBState.STANDBY
            return AEBState.STANDBY, 0.0

        # 计算TTC
        TTC = distance / relative_speed if relative_speed > 0 else float('inf')

        # 状态判断
        if TTC > self.params.fcw_warning_time and distance > self.params.partial_braking_distance:
            # 预警阶段
            self.state = AEBState.WARNING
            return AEBState.WARNING, 0.0

        elif distance <= self.params.full_braking_distance:
            # 全力制动
            self.state = AEBState.FULL_BRAKING
            return AEBState.FULL_BRAKING, self.params.full_braking_decel

        elif distance <= self.params.partial_braking_distance:
            # 部分制动
            self.state = AEBState.PARTIAL_BRAKING
            # 根据距离线性调整减速度
            ratio = (distance - self.params.full_braking_distance) / \
                    (self.params.partial_braking_distance - self.params.full_braking_distance)
            decel = self.params.full_braking_decel + \
                   (self.params.partial_braking_decel - self.params.full_braking_decel) * ratio
            return AEBState.PARTIAL_BRAKING, decel

        else:
            # 预警
            self.state = AEBState.WARNING
            return AEBState.WARNING, -1.0  # 轻微制动

    def should_activate(self,
                      distance: float,
                      relative_speed: float,
                      ego_speed: float) -> Tuple[bool, AEBState, str]:
        """
        判断AEB是否应该激活

        Args:
            distance: 与前车距离（米）
            relative_speed: 相对速度（米/秒）
            ego_speed: 自车速度（米/秒）

        Returns:
            (是否激活, AEB状态, 原因)
        """
        # 检查最低速度要求
        if ego_speed < self.params.min_speed_for_aeb:
            return False, AEBState.OFF, f"速度过低: {ego_speed:.1f}m/s < {self.params.min_speed_for_aeb:.1f}m/s"

        # 检查距离
        if distance > self.params.partial_braking_distance * 1.5:
            return False, AEBState.STANDBY, "距离充足"

        # 检查是否正在接近
        if relative_speed <= 0:
            return False, AEBState.STANDBY, "目标远离"

        # 计算TTC
        TTC = distance / relative_speed if relative_speed > 0 else float('inf')

        # TTC判断
        if TTC < 1.0:
            return True, AEBState.FULL_BRAKING, f"紧急: TTC={TTC:.2f}s"
        elif TTC < 2.0:
            return True, AEBState.PARTIAL_BRAKING, f"危险: TTC={TTC:.2f}s"
        elif TTC < self.params.fcw_warning_time:
            return True, AEBState.WARNING, f"警告: TTC={TTC:.2f}s"
        else:
            return False, AEBState.WARNING, f"预警: TTC={TTC:.2f}s"

    def get_braking_force(self,
                        state: AEBState) -> float:
        """
        获取制动力度

        Args:
            state: AEB状态

        Returns:
            制动减速度（米/秒²）
        """
        if state == AEBState.FULL_BRAKING:
            return self.params.full_braking_decel
        elif state == AEBState.PARTIAL_BRAKING:
            return self.params.partial_braking_decel
        elif state == AEBState.WARNING:
            return -1.0
        else:
            return 0.0

    def reset(self):
        """重置AEB状态"""
        self.state = AEBState.STANDBY
        self.last_warning_time = 0.0
