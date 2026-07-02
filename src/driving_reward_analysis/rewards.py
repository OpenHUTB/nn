"""自动驾驶奖励函数定义模块.

实现三类核心奖励函数：
- 速度跟踪奖励 (Speed Tracking Reward)
- 距离控制奖励 (Distance Control Reward)
- 舒适性奖励 (Comfort Reward)
"""

import numpy as np


def speed_reward(
    velocity: np.ndarray,
    config: dict,
) -> np.ndarray:
    """速度跟踪奖励函数.

    基于高斯分布建模，车辆速度越接近目标速度奖励越高.
    R_speed(v) = -((v - v_target) / sigma)^2

    Args:
        velocity: 当前速度数组 (m/s)
        config: 速度奖励配置，包含 target_speed 和 tolerance

    Returns:
        速度奖励值数组
    """
    target = config.get("target_speed", 25.0)
    sigma = config.get("tolerance", 5.0)

    reward = -((velocity - target) / sigma) ** 2
    return np.clip(reward, -100, 0)


def distance_reward(
    distance: np.ndarray,
    config: dict,
) -> np.ndarray:
    """距离控制奖励函数.

    分段线性函数建模：
    - d < critical: 严重惩罚 (-100)
    - critical <= d < safe: 线性增长
    - d >= safe: 最大奖励 (0)

    Args:
        distance: 当前前车距离数组 (m)
        config: 距离奖励配置，包含 safe_distance 和 critical_distance

    Returns:
        距离奖励值数组
    """
    safe = config.get("safe_distance", 15.0)
    critical = config.get("critical_distance", 5.0)

    reward = np.zeros_like(distance, dtype=np.float64)

    # d < critical: 严重惩罚
    mask_critical = distance < critical
    reward[mask_critical] = -100.0

    # critical <= d < safe: 线性增长
    mask_linear = (distance >= critical) & (distance < safe)
    reward[mask_linear] = -10.0 * (safe - distance[mask_linear])

    # d >= safe: 0 (最大奖励)
    # 默认为 0

    return reward


def comfort_reward(
    acceleration: np.ndarray,
    config: dict,
) -> np.ndarray:
    """舒适性奖励函数.

    基于加速度的二次函数建模，零加速度时奖励最高.
    R_comfort(a) = -a^2

    Args:
        acceleration: 当前加速度数组 (m/s²)
        config: 舒适性奖励配置，包含 max_acceleration

    Returns:
        舒适性奖励值数组
    """
    max_acc = config.get("max_acceleration", 2.0)

    reward = -(acceleration**2)
    return np.clip(reward, -(max_acc**2) * 2, 0)
