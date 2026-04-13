"""已训练无人机强化学习模型的评估脚本"""

import argparse
import logging
import os
from typing import List, Dict, Tuple
import json
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    config_path: str = None,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    save_trajectories: bool = True,
    results_dir: str = "./data/results/",
) -> Dict:
    """
    评估训练过的强化学习模型。

    Args:
        model_path: 已训练模型的路径
        config_path: 配置YAML的路径
        n_episodes: 评估的回合数
        deterministic: 使用确定性策略
        render: 评估期间是否渲染
        save_trajectories: 保存回合轨迹
        results_dir: 保存结果的目录

    返回值:
        包含评估结果的字典
    """
    from stable_baselines3 import PPO, DQN
    from ..envs.base_drone_env import AirSimDroneEnv
    from ..envs.wrappers import create_wrapped_env

    os.makedirs(results_dir, exist_ok=True)

    # 加载配置
    if config_path is None:
        # 默认复用训练配置文件
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "training", "config.yaml"
        )
        config_path = os.path.abspath(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    success_distance_threshold = float(
        config.get("evaluation", {}).get("success_distance_threshold", 2.0)
    )

    logger.info(f"从 {model_path} 加载模型")

    # 根据模型路径或配置确定算法
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    else:
        try:
            model = PPO.load(model_path)
        except:
            model = DQN.load(model_path)

    # 创建评估环境
    logger.info("正在创建评估环境")
    env_config = config.get("environment", {})
    obs_config = config.get("observation", {})
    reward_config = config.get("reward", {})
    target_points = [np.array(pt) for pt in env_config.get("target_points", [])]

    env = AirSimDroneEnv(
        ip_address=env_config.get("ip_address", "127.0.0.1"),
        image_shape=tuple(env_config.get("image_shape", [84, 84])),
        max_steps=config.get("training", {}).get("max_steps_per_episode", 500),
        target_points=target_points if target_points else None,
        action_duration=env_config.get("action_duration", 0.5),
        control_step_delay=float(env_config.get("control_step_delay", 0.02)),
        velocity_step=env_config.get("velocity_step", 5.0),
        target_reached_threshold=reward_config.get("success_distance_threshold", 2.0),
        reward_config=reward_config,
        verbose=False,
    )

    env = create_wrapped_env(
        env,
        frame_stack=obs_config.get("frame_stack", 4),
        normalize_obs=obs_config.get("normalize", True),
        resize_obs=obs_config.get("resize", False),
        resize_shape=tuple(obs_config.get("resize_shape", [84, 84])),
        clip_reward=bool(config.get("action", {}).get("clip", False)),
        action_repeat=config.get("action", {}).get("repeat", 1),
    )

    # 运行评估
    logger.info(f"为 {n_episodes} 个回合运行评估")

    results = {
        "episode_rewards": [],
        "episode_lengths": [],
        "episode_successes": [],
        "episode_collisions": [],
        "trajectories": [] if save_trajectories else None,
    }

    for ep in range(n_episodes):
        logger.info(f"回合 {ep + 1}/{n_episodes}")

        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        trajectory = [env.unwrapped.state["position"].copy()]

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if save_trajectories:
                trajectory.append(env.unwrapped.state["position"].copy())

            done = terminated or truncated

        # 记录回合统计
        results["episode_rewards"].append(episode_reward)
        results["episode_lengths"].append(episode_length)
        results["episode_collisions"].append(info.get("collision", False))

        # 仅在真实到达目标阈值且无碰撞时判定成功。
        success = (
            not info.get("collision", False)
            and float(info.get("distance_to_target", np.inf))
            <= success_distance_threshold
        )
        results["episode_successes"].append(success)

        if save_trajectories:
            results["trajectories"].append(
                {
                    "episode": ep,
                    "reward": float(episode_reward),
                    "length": episode_length,
                    "positions": [pos.tolist() for pos in trajectory],
                }
            )

        logger.info(
            f"  奖励: {episode_reward:.2f}, 步长: {episode_length}, "
            f"成功: {success}, 碰撞: {info.get('collision', False)}"
        )

    # 计算统计数据
    rewards = results["episode_rewards"]
    lengths = results["episode_lengths"]
    successes = results["episode_successes"]
    collisions = results["episode_collisions"]

    eval_stats = {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "episode_rewards": [float(r) for r in rewards],
        "episode_lengths": [int(l) for l in lengths],
        "episode_successes": [bool(s) for s in successes],
        "episode_collisions": [bool(c) for c in collisions],
    }

    # 保存结果
    results_file = os.path.join(results_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(eval_stats, f, indent=2)
    logger.info(f"结果已保存到 {results_file}")

    # 保存轨迹（如果请求）
    if save_trajectories and results["trajectories"]:
        traj_file = os.path.join(results_dir, "trajectories.json")
        with open(traj_file, "w") as f:
            json.dump(results["trajectories"], f, indent=2)
        logger.info(f"轨迹已保存到 {traj_file}")

    # 打印总结
    logger.info("\n" + "=" * 50)
    logger.info("评估总结")
    logger.info("=" * 50)
    logger.info(
        f"平均奖励: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}"
    )
    logger.info(f"平均回合长度: {eval_stats['mean_length']:.1f}")
    logger.info(f"成功率: {eval_stats['success_rate']:.1%}")
    logger.info(f"碰撞率: {eval_stats['collision_rate']:.1%}")
    logger.info("=" * 50 + "\n")

    env.close()
    return eval_stats


def compare_models(
    model_paths: List[str],
    config_path: str = None,
    n_episodes: int = 10,
    results_dir: str = "./data/results/",
) -> Dict:
    """
    比较多个已训练的模型。

    Args:
        model_paths: 模型路径列表
        config_path: 配置 YAML 路径
        n_episodes: 每个模型评估回合数
        results_dir: 结果保存目录

    返回值:
        对比结果
    """
    comparison = {}

    for model_path in model_paths:
        logger.info(f"\n正在评估 {model_path}")
        results = evaluate_model(
            model_path,
            config_path=config_path,
            n_episodes=n_episodes,
            results_dir=results_dir,
        )
        model_name = os.path.basename(model_path)
        comparison[model_name] = results

    # 保存对比结果
    comparison_file = os.path.join(results_dir, "model_comparison.json")
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"对比结果已保存到 {comparison_file}")

    return comparison


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="评估训练好的无人机强化学习模型")
    parser.add_argument("model_path", type=str, help="已训练模型路径")
    parser.add_argument("--config", type=str, default=None, help="配置 YAML 文件路径")
    parser.add_argument("--episodes", type=int, default=10, help="评估回合数")
    parser.add_argument(
        "--stochastic", action="store_true", help="使用随机策略（探索）"
    )
    parser.add_argument("--render", action="store_true", help="评估时渲染")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./data/results/",
        help="结果保存目录",
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        config_path=args.config,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
