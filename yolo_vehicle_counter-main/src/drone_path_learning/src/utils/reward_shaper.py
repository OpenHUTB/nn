"""无人机路径规划的奖励塑形函数"""

import numpy as np
from typing import Dict, Tuple, List


class RewardShaper:
    """无人机导航的奖励函数计算"""

    def __init__(
        self,
        target_points: List[np.ndarray] = None,
        distance_reward_scale: float = 0.1,
        collision_penalty: float = -100.0,
        success_reward: float = 50.0,
        step_reward: float = 1.0,
        max_distance: float = 100.0,
    ):
        """
        初始化奖励塑形工具。

        Args:
            target_points: 目标航点列表（每个都是形状为 (3,) 的 np.ndarray）
            distance_reward_scale: 距离惩罚的规模
            collision_penalty: 碰撞惩罚
            success_reward: 达到目标的奖励
            step_reward: 每步的小奖励（探索奖励）
            max_distance: 最大距离阈值
        """
        self.target_points = target_points or [
            np.array([0, 0, -10]),
            np.array([50, 0, -10]),
            np.array([50, 50, -10]),
        ]
        self.distance_reward_scale = distance_reward_scale
        self.collision_penalty = collision_penalty
        self.success_reward = success_reward
        self.step_reward = step_reward
        self.max_distance = max_distance
        self.current_target_idx = 0

    def compute_reward(
        self,
        position: np.ndarray,
        collision: bool,
        reached_target: bool,
        info: Dict = None,
    ) -> Tuple[float, bool]:
        """
        计算当前步的奖励。

        Args:
            position: 当前无人机位置（3D 数组）
            collision: 是否发生碰撞
            reached_target: 是否达到目标
            info: 附加信息字典

        返回值:
            reward: 计算的奖励
            done: 回合终止标志
        """
        reward = 0.0
        done = False

        # 1. 碰撞惩罚（最高优先级）
        if collision:
            reward = self.collision_penalty
            done = True
            return float(reward), done

        # 2. 基于距离的惩罚（与距离目标远而获得的负奖励）
        min_dist = self._min_distance_to_target(position)
        distance_penalty = -min_dist * self.distance_reward_scale
        reward += distance_penalty

        # 3. 步奖励（鼓励探测）
        reward += self.step_reward

        # 4. 目标达到（成功奖励）
        if reached_target:
            reward += self.success_reward
            done = True

        # 5. 超时检查（可选：如果离太远太久则完成）
        if min_dist > self.max_distance:
            reward -= 10.0  # 离太远的额外惩罚

        return float(reward), done

    def _min_distance_to_target(self, position: np.ndarray) -> float:
        """计算到任何目标点的最小距离"""
        if not self.target_points:
            return 0.0

        targets = np.asarray(self.target_points, dtype=np.float32)
        distances = np.linalg.norm(
            targets - np.asarray(position, dtype=np.float32), axis=1
        )
        return float(np.min(distances))

    def should_reset_target(self, position: np.ndarray, threshold: float = 2.0) -> bool:
        """检查是否达到当前目标并移动到下一个"""
        if len(self.target_points) == 0:
            return False

        current_target = self.target_points[self.current_target_idx]
        dist = np.linalg.norm(position - current_target)

        if dist < threshold:
            if self.current_target_idx < len(self.target_points) - 1:
                self.current_target_idx += 1
                return True
        return False

    def reset_targets(self):
        """为新回合重置目标序列"""
        self.current_target_idx = 0

    def get_current_target(self) -> np.ndarray:
        """获取当前目标位置"""
        if self.current_target_idx < len(self.target_points):
            return self.target_points[self.current_target_idx]
        return self.target_points[-1]


class SimpleRewardShaper(RewardShaper):
    """具有最少功能的简化奖励塑形工具"""

    def compute_reward(
        self,
        position: np.ndarray,
        collision: bool,
        reached_target: bool,
        info: Dict = None,
    ) -> Tuple[float, bool]:
        """简化的奖励计算"""
        reward = 0.0
        done = False

        if collision:
            reward = -100.0
            done = True
        elif reached_target:
            reward = 50.0
            done = True
        else:
            # 简单的距离惩罚
            min_dist = self._min_distance_to_target(position)
            reward = -min_dist * 0.1 + 1.0  # -距离 + 每步奖励

        return float(reward), done
