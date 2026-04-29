"""
Lyapunov-based Potential Shaping for Lane Change

理论基础：
1. 侧向动力学的Lyapunov函数
2. 纵向安全的障碍函数（带饱和机制）
3. 策略不变性保证

与现有系统集成：
- 作为ImprovedRewardSystem的扩展
- 保持现有奖励结构不变
- 添加理论驱动的塑形奖励
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class LaneChangeDynamicsParams:
    """换道动力学参数"""
    # 侧向动力学参数
    omega_n: float = 2.0          # 自然频率 (rad/s)
    zeta: float = 0.7             # 阻尼比

    # 纵向安全参数
    ttc_min: float = 2.5          # 最小TTC (s)
    d_min: float = 5.0            # 最小安全距离 (m)
    d_comfortable: float = 20.0   # 舒适距离 (m)

    # 塑形权重
    alpha_lateral: float = 0.5    # 侧向势能权重
    alpha_longitudinal: float = 0.3  # 纵向势能权重
    sigma_safety: float = 5.0     # 安全势能陡峭度

    # ⭐ 新增：饱和参数
    max_barrier_value: float = 10.0  # 最大障碍值（防止指数爆炸）

    # 环境参数
    lane_width: float = 3.5       # 车道宽度 (m)


class LyapunovLateralPotential:
    """
    侧向Lyapunov势能函数

    基于二阶侧向动力学：
    d̈ = -2ζω_n ḋ - ω_n² (d - d_target) + u_lat

    Lyapunov函数：
    V(d, ḋ) = ½(d - d_target)² + ½(ḋ/ω_n)²

    性质：V̇ ≤ 0（渐近稳定）
    """

    def __init__(self, params: LaneChangeDynamicsParams):
        self.params = params
        self.omega_n = params.omega_n
        self.alpha = params.alpha_lateral

    def compute_potential(self, d: float, d_dot: float, d_target: float) -> float:
        """
        计算侧向势能

        Args:
            d: 当前横向位置 (m)
            d_dot: 当前横向速度 (m/s)
            d_target: 目标横向位置 (m)

        Returns:
            potential: 势能值（负值，越大越好）
        """
        d_error = d - d_target

        # Lyapunov函数 V = ½(d - d_target)² + ½(ḋ/ω_n)²
        V_lateral = 0.5 * (d_error ** 2) + 0.5 * (d_dot / self.omega_n) ** 2

        # 势能为负的Lyapunov函数（目标位置势能最大）
        potential = -self.alpha * V_lateral

        return potential

    def compute_shaping_reward(self,
                              d_curr: float, d_dot_curr: float,
                              d_next: float, d_dot_next: float,
                              d_target: float,
                              gamma: float = 0.99) -> float:
        """
        计算塑形奖励：γΦ(s') - Φ(s)

        Args:
            d_curr, d_dot_curr: 当前状态
            d_next, d_dot_next: 下一状态
            d_target: 目标位置
            gamma: 折扣因子

        Returns:
            shaping_reward: 塑形奖励
        """
        phi_curr = self.compute_potential(d_curr, d_dot_curr, d_target)
        phi_next = self.compute_potential(d_next, d_dot_next, d_target)

        return gamma * phi_next - phi_curr


class SafetyBarrierPotential:
    """
    纵向安全障碍势能函数（改进版，带饱和机制）

    基于指数障碍函数，距离过近时势能急剧下降
    Φ(Δs) = -α · min(barrier, max_value)

    其中 barrier = exp(-(Δs - d_safe)/σ)

    ⭐ 改进：添加饱和机制防止指数爆炸
    """

    def __init__(self, params: LaneChangeDynamicsParams):
        self.params = params
        self.alpha = params.alpha_longitudinal
        self.sigma = params.sigma_safety
        self.ttc_min = params.ttc_min
        self.d_min = params.d_min
        self.d_comfortable = params.d_comfortable
        self.max_barrier_value = params.max_barrier_value  # ⭐ 饱和阈值

    def compute_safe_distance(self, v_ego: float, v_lead: float) -> float:
        """
        基于TTC计算安全距离

        Args:
            v_ego: 自车速度 (m/s)
            v_lead: 前车速度 (m/s)

        Returns:
            safe_distance: 安全距离 (m)
        """
        relative_velocity = v_ego - v_lead

        if relative_velocity <= 0:
            # 不接近前车，使用最小安全距离
            return self.d_min
        else:
            # 基于TTC的安全距离
            ttc_based = self.ttc_min * relative_velocity
            return max(self.d_min, ttc_based)

    def compute_potential(self, delta_s: float, v_ego: float, v_lead: float) -> float:
        """
        计算纵向势能（带饱和机制）

        Args:
            delta_s: 与前车距离 (m)
            v_ego: 自车速度 (m/s)
            v_lead: 前车速度 (m/s)

        Returns:
            potential: 势能值
        """
        # 距离足够远时，势能为0
        if delta_s >= self.d_comfortable:
            return 0.0

        # 计算安全距离
        d_safe = self.compute_safe_distance(v_ego, v_lead)

        # ⭐ 计算指数，但限制在合理范围内
        exponent = -(delta_s - d_safe) / self.sigma

        # ⭐ 饱和机制：限制指数值在[-10, 10]范围内
        # 这样 exp(exponent) 在 [4.5e-5, 22026] 范围内
        exponent = np.clip(exponent, -10.0, 10.0)

        # 指数障碍函数
        barrier = np.exp(exponent)

        # ⭐ 再次限制barrier值
        barrier = min(barrier, self.max_barrier_value)

        potential = -self.alpha * barrier

        return potential

    def compute_shaping_reward(self,
                              delta_s_curr: float, v_ego_curr: float, v_lead_curr: float,
                              delta_s_next: float, v_ego_next: float, v_lead_next: float,
                              gamma: float = 0.99) -> float:
        """
        计算塑形奖励

        Args:
            *_curr: 当前状态
            *_next: 下一状态
            gamma: 折扣因子

        Returns:
            shaping_reward: 塑形奖励
        """
        phi_curr = self.compute_potential(delta_s_curr, v_ego_curr, v_lead_curr)
        phi_next = self.compute_potential(delta_s_next, v_ego_next, v_lead_next)

        return gamma * phi_next - phi_curr


class LyapunovPotentialShaping:
    """
    完整的Lyapunov势能塑形系统

    组合侧向和纵向势能，提供理论保证的奖励塑形
    """

    def __init__(self, params: LaneChangeDynamicsParams = None):
        if params is None:
            params = LaneChangeDynamicsParams()

        self.params = params
        self.lateral_potential = LyapunovLateralPotential(params)
        self.longitudinal_potential = SafetyBarrierPotential(params)

        # 统计
        self.total_shaping_reward = 0.0
        self.episode_shaping_rewards = []

        # 历史状态（用于计算塑形奖励）
        self.last_state = None

        print("🎯 Lyapunov Potential Shaping initialized")
        print(f"   Lateral ω_n: {params.omega_n:.2f} rad/s")
        print(f"   Lateral ζ: {params.zeta:.2f}")
        print(f"   Safety TTC_min: {params.ttc_min:.2f} s")
        print(f"   Lateral weight α_lat: {params.alpha_lateral:.2f}")
        print(f"   Longitudinal weight α_long: {params.alpha_longitudinal:.2f}")
        print(f"   Max barrier value: {params.max_barrier_value:.2f}")  # ⭐ 新增

    def compute_shaping_reward(self,
                              state_curr: Dict,
                              state_next: Dict,
                              gamma: float = 0.99) -> Tuple[float, Dict]:
        """
        计算塑形奖励（主接口）

        Args:
            state_curr: 当前状态字典
                - 'd': 横向位置
                - 'd_dot': 横向速度
                - 'd_target': 目标横向位置
                - 'delta_s': 与前车距离
                - 'v_ego': 自车速度
                - 'v_lead': 前车速度
                - 'has_lead': 是否有前车
            state_next: 下一状态字典（同上）
            gamma: 折扣因子

        Returns:
            total_shaping: 总塑形奖励
            details: 详细信息字典
        """
        # 侧向塑形
        shaping_lateral = self.lateral_potential.compute_shaping_reward(
            state_curr['d'], state_curr['d_dot'],
            state_next['d'], state_next['d_dot'],
            state_curr['d_target'], gamma
        )

        # 纵向塑形（仅当有前车时）
        shaping_longitudinal = 0.0
        if state_curr.get('has_lead', False) and state_next.get('has_lead', False):
            shaping_longitudinal = self.longitudinal_potential.compute_shaping_reward(
                state_curr['delta_s'], state_curr['v_ego'], state_curr['v_lead'],
                state_next['delta_s'], state_next['v_ego'], state_next['v_lead'],
                gamma
            )

        # 总塑形奖励
        total_shaping = shaping_lateral + shaping_longitudinal
        self.total_shaping_reward += total_shaping

        # 详细信息
        details = {
            'shaping_lateral': shaping_lateral,
            'shaping_longitudinal': shaping_longitudinal,
            'total_shaping': total_shaping,
        }

        return total_shaping, details

    def reset_episode(self):
        """Episode结束时重置"""
        self.episode_shaping_rewards.append(self.total_shaping_reward)
        self.total_shaping_reward = 0.0
        self.last_state = None

    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.episode_shaping_rewards:
            return {}

        return {
            'mean_shaping_reward': np.mean(self.episode_shaping_rewards),
            'std_shaping_reward': np.std(self.episode_shaping_rewards),
            'total_episodes': len(self.episode_shaping_rewards),
        }