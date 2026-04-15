"""无人机强化学习评估的性能指标计算"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """为无人机回合计算各种性能指标"""

    @staticmethod
    def success_rate(episodes: List[Dict]) -> float:
        """
        计算成功率（在没有碰撞的情况下达到目标）。

        Args:
            episodes: 回合字典列表

        返回值:
            成功率 (0-1)
        """
        if len(episodes) == 0:
            return 0.0

        successes = sum(
            1
            for ep in episodes
            if not ep.get("collision", False) and ep.get("success", False)
        )
        return successes / len(episodes)

    @staticmethod
    def collision_rate(episodes: List[Dict]) -> float:
        """
        计算碰撞率。

        Args:
            episodes: 回合字典列表

        返回值:
            碰撞率 (0-1)
        """
        if len(episodes) == 0:
            return 0.0

        collisions = sum(1 for ep in episodes if ep.get("collision", False))
        return collisions / len(episodes)

    @staticmethod
    def average_path_length(
        trajectories: List[List[Tuple[float, float, float]]],
    ) -> float:
        """
        计算平均路径长度（欧几里得距离）。

        Args:
            trajectories: 位置列表的列表

        返回值:
            平均路径长度
        """
        if len(trajectories) == 0:
            return 0.0

        path_lengths = []
        for trajectory in trajectories:
            if len(trajectory) < 2:
                continue

            positions = np.array(trajectory)
            diffs = np.diff(positions, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            path_length = np.sum(distances)
            path_lengths.append(path_length)

        return np.mean(path_lengths) if path_lengths else 0.0

    @staticmethod
    def path_efficiency(
        trajectories: List[List[Tuple[float, float, float]]],
        straight_line_distances: List[float],
    ) -> float:
        """
        计算路径效率（直线距离/实际路径）。

        Args:
            trajectories: 位置列表的列表
            straight_line_distances: 到目标的直线距离

        返回值:
            路径效率比率（越接近 1 越好）
        """
        if len(trajectories) != len(straight_line_distances):
            return 0.0

        efficiencies = []
        for trajectory, straight_dist in zip(trajectories, straight_line_distances):
            if len(trajectory) < 2 or straight_dist == 0:
                continue

            positions = np.array(trajectory)
            diffs = np.diff(positions, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            actual_path = np.sum(distances)

            efficiency = straight_dist / actual_path if actual_path > 0 else 0
            efficiencies.append(efficiency)

        return np.mean(efficiencies) if efficiencies else 0.0

    @staticmethod
    def smoothness(trajectory: List[Tuple[float, float, float]]) -> float:
        """
        计算轨迹平滑度（越低越平滑）。

        使用二阶导数（曲率）。

        Args:
            trajectory: 3D 位置列表

        返回值:
            平滑度指标
        """
        if len(trajectory) < 3:
            return 0.0

        positions = np.array(trajectory)

        # 计算一阶差分（速度）
        v1 = np.diff(positions, axis=0)

        # 计算二阶差分（加速度）
        if len(v1) < 2:
            return 0.0

        v2 = np.diff(v1, axis=0)

        # 计算加速度模长
        accelerations = np.linalg.norm(v2, axis=1)

        # 平滑度取平均加速度的反比
        mean_accel = np.mean(accelerations)
        smoothness = 1.0 / (1.0 + mean_accel)  # 归一化到 [0, 1]

        return float(smoothness)

    @staticmethod
    def average_smoothness(
        trajectories: List[List[Tuple[float, float, float]]],
    ) -> float:
        """计算所有轨迹的平均平滑度"""
        if len(trajectories) == 0:
            return 0.0

        smoothness_scores = [
            PerformanceMetrics.smoothness(traj) for traj in trajectories
        ]
        return np.mean(smoothness_scores)

    @staticmethod
    def stability(episode_rewards: List[float]) -> float:
        """
        计算训练稳定性（方差越小越稳定）。

        Args:
            episode_rewards: 回合奖励列表

        返回值:
            稳定性评分 (0-1): 1 表示非常稳定
        """
        if len(episode_rewards) < 2:
            return 0.0

        # 使用变异系数（标准差 / 均值）
        mean_reward = np.mean(episode_rewards)
        if mean_reward == 0:
            return 0.0

        std_reward = np.std(episode_rewards)
        cv = std_reward / abs(mean_reward)

        # 转换为稳定性评分
        stability = 1.0 / (1.0 + cv)  # 归一化到 [0, 1]
        return float(stability)

    @staticmethod
    def compute_all_metrics(
        episodes: List[Dict],
        trajectories: List[List[Tuple[float, float, float]]],
        episode_rewards: List[float],
    ) -> Dict[str, float]:
        """
        一次性计算全部指标。

        Args:
            episodes: 回合数据列表
            trajectories: 轨迹位置列表
            episode_rewards: 奖励列表

        返回值:
            包含所有指标的字典
        """
        metrics = {
            "success_rate": PerformanceMetrics.success_rate(episodes),
            "collision_rate": PerformanceMetrics.collision_rate(episodes),
            "average_path_length": PerformanceMetrics.average_path_length(trajectories),
            "average_smoothness": PerformanceMetrics.average_smoothness(trajectories),
            "training_stability": PerformanceMetrics.stability(episode_rewards),
        }

        return metrics


def compute_metrics_from_results(
    results_file: str, trajectories_file: str = None
) -> Dict[str, float]:
    """
    从评估结果 JSON 文件计算指标。

    Args:
        results_file: 评估结果 JSON 路径
        trajectories_file: 轨迹 JSON 路径

    返回值:
        包含计算结果的字典
    """
    # 读取评估结果
    with open(results_file, "r") as f:
        eval_results = json.load(f)

    episodes = [
        {
            "success": eval_results["episode_successes"][i],
            "collision": eval_results["episode_collisions"][i],
        }
        for i in range(eval_results["n_episodes"])
    ]

    episode_rewards = eval_results["episode_rewards"]

    # 如有轨迹文件则读取
    trajectories = []
    if trajectories_file and os.path.exists(trajectories_file):
        with open(trajectories_file, "r") as f:
            traj_data = json.load(f)
        trajectories = [t.get("positions", []) for t in traj_data]

    # 计算指标
    metrics = PerformanceMetrics.compute_all_metrics(
        episodes, trajectories, episode_rewards
    )

    # 补充评估结果中的基础指标
    metrics.update(
        {
            "mean_reward": eval_results["mean_reward"],
            "std_reward": eval_results["std_reward"],
            "mean_episode_length": eval_results["mean_length"],
        }
    )

    return metrics


def print_metrics(metrics: Dict[str, float]):
    """格式化输出指标"""
    logger.info("\n" + "=" * 50)
    logger.info("PERFORMANCE METRICS")
    logger.info("=" * 50)

    for key, value in metrics.items():
        if "rate" in key or key in ["training_stability", "average_smoothness"]:
            logger.info(f"{key:.<40} {value:>7.1%}")
        else:
            logger.info(f"{key:.<40} {value:>7.2f}")

    logger.info("=" * 50 + "\n")


def main():
    """命令行入口"""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="计算性能指标")
    parser.add_argument("--results", type=str, required=True, help="评估结果 JSON 路径")
    parser.add_argument("--trajectories", type=str, default=None, help="轨迹 JSON 路径")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    metrics = compute_metrics_from_results(args.results, args.trajectories)
    print_metrics(metrics)

    # 保存指标
    output_file = os.path.splitext(args.results)[0] + "_metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"指标已保存到 {output_file}")


if __name__ == "__main__":
    main()
