"""用于训练监控和检查点保存的自定义回调"""

import json
import os
from typing import Dict, Any
import numpy as np
import logging
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class CustomCallback(BaseCallback):
    """用于详细日志记录和监控的自定义回调"""

    def __init__(self, log_dir: str = "./data/logs/", verbose: int = 1):
        """
        初始化自定义回调。

        Args:
            log_dir: 保存日志的目录
            verbose: 详细级别
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 统计指标跟踪
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_collisions = []
        self.episode_success = []

    def _on_step(self) -> bool:
        """环境每一步后调用"""
        # 检查回合是否完成
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        if isinstance(dones, np.ndarray):
            dones = dones.tolist()

        # 处理单个环境
        if not isinstance(dones, list):
            dones = [dones]
        if not isinstance(infos, list):
            infos = [infos]

        # 处理已完成的回合
        for done, info in zip(dones, infos):
            if done and "episode" in info:
                episode_info = info["episode"]

                # 提取回合统计信息
                ep_reward = episode_info.get("r", 0.0)
                ep_length = episode_info.get("l", 0)

                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # 检查碰撞或成功标志（如果在信息中可用）
                collision = info.get("collision", False)
                success = info.get("success", False)

                self.episode_collisions.append(collision)
                self.episode_success.append(success)

                # 定期记录
                if len(self.episode_rewards) % 10 == 0:
                    self._save_stats()

                    if self.verbose >= 1:
                        mean_reward = np.mean(self.episode_rewards[-100:])
                        mean_length = np.mean(self.episode_lengths[-100:])
                        collision_rate = np.mean(self.episode_collisions[-100:])
                        success_rate = np.mean(self.episode_success[-100:])

                        print(
                            f"回合 {len(self.episode_rewards)}: "
                            f"奖励={mean_reward:.2f}, "
                            f"长度={mean_length:.0f}, "
                            f"碰撞={collision_rate:.2%}, "
                            f"成功={success_rate:.2%}"
                        )

        return True

    def _on_training_end(self) -> None:
        """训练结束时调用"""
        self._save_stats()
        logger.info(f"训练完成。统计信息已保存到 {self.log_dir}")

    def _save_stats(self):
        """将训练统计信息保存到JSON"""
        stats = {
            "total_episodes": len(self.episode_rewards),
            "mean_reward": (
                float(np.mean(self.episode_rewards)) if self.episode_rewards else 0
            ),
            "std_reward": (
                float(np.std(self.episode_rewards))
                if len(self.episode_rewards) > 1
                else 0
            ),
            "max_reward": (
                float(np.max(self.episode_rewards)) if self.episode_rewards else 0
            ),
            "min_reward": (
                float(np.min(self.episode_rewards)) if self.episode_rewards else 0
            ),
            "mean_length": (
                float(np.mean(self.episode_lengths)) if self.episode_lengths else 0
            ),
            "collision_rate": (
                float(np.mean(self.episode_collisions))
                if self.episode_collisions
                else 0
            ),
            "success_rate": (
                float(np.mean(self.episode_success)) if self.episode_success else 0
            ),
        }

        stats_file = os.path.join(self.log_dir, "training_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)


class ProgressCallback(BaseCallback):
    """简单进度回调"""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.last_n_calls = 0

    def _on_step(self) -> bool:
        """定期记录进度"""
        n_calls = self.num_timesteps

        # 每10000步记录一次
        if n_calls > self.last_n_calls + 10000:
            self.last_n_calls = n_calls
            if self.verbose >= 1:
                print(f"训练步: {n_calls}")

        return True


class RewardThresholdCallback(BaseCallback):
    """如果达到奖励阈值则停止训练"""

    def __init__(self, threshold: float = 0.0, patience: int = 50, verbose: int = 0):
        """
        初始化奖励阈值回调。

        Args:
            threshold: 停止的奖励阈值
            patience: 改进无进展前的回合数
            verbose: 详细级别
        """
        super().__init__(verbose)
        self.threshold = threshold
        self.patience = patience
        self.episode_rewards = []
        self.best_mean_reward = -np.inf
        self.patience_counter = 0

    def _on_step(self) -> bool:
        """检查奖励阈值"""
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        if isinstance(dones, np.ndarray):
            dones = dones.tolist()
        if not isinstance(dones, list):
            dones = [dones]
        if not isinstance(infos, list):
            infos = [infos]

        for done, info in zip(dones, infos):
            if done and "episode" in info:
                ep_reward = info["episode"].get("r", 0.0)
                self.episode_rewards.append(ep_reward)

                # 检查是否达到阈值
                if len(self.episode_rewards) >= 10:
                    mean_reward = np.mean(self.episode_rewards[-10:])

                    if mean_reward > self.threshold:
                        if self.verbose >= 1:
                            print(f"Threshold reached! Mean reward: {mean_reward:.2f}")
                        return False

                    # 检查耐心值
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1

                    if self.patience_counter >= self.patience:
                        if self.verbose >= 1:
                            print(
                                f"No improvement for {self.patience} episodes. Stopping."
                            )
                        return False

        return True


class EpisodeStatisticsCallback(BaseCallback):
    """跟踪详细回合统计信息"""

    def __init__(self, log_dir: str = "./data/logs/", verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.episode_data = []

    def _on_step(self) -> bool:
        """收集回合数据"""
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        if isinstance(dones, np.ndarray):
            dones = dones.tolist()
        if not isinstance(dones, list):
            dones = [dones]
        if not isinstance(infos, list):
            infos = [infos]

        for done, info in zip(dones, infos):
            if done and "episode" in info:
                episode_info = {
                    "episode": len(self.episode_data),
                    "reward": float(info["episode"].get("r", 0.0)),
                    "length": int(info["episode"].get("l", 0)),
                    "timestep": int(self.num_timesteps),
                    **{k: v for k, v in info.items() if k not in ["episode"]},
                }
                self.episode_data.append(episode_info)

                # 定期保存
                if len(self.episode_data) % 50 == 0:
                    self._save_data()

        return True

    def _on_training_end(self) -> None:
        """在训练结束时保存回合数据"""
        self._save_data()

    def _save_data(self):
        """将回合数据保存到 JSON"""
        data_file = os.path.join(self.log_dir, "episode_data.json")
        with open(data_file, "w") as f:
            json.dump(self.episode_data, f, indent=2)
