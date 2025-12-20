# eval_agent.py
"""
CARLA 智能体评估脚本（4D 环境专用）
- 不使用 Monitor / DummyVecEnv
- 支持多轮评估、轨迹分离、种子复现
- 输出平均奖励与标准差
"""

import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from carla_env.carla_env_multi_obs import CarlaEnvMultiObs


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent in CARLA (4D mode)")
    parser.add_argument("--model-path", type=str, default="models/ppo_carla.zip",
                        help="Path to the trained model (.zip file)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of evaluation episodes")
    parser.add_argument("--steps-per-ep", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--keep-alive", action="store_true",
                        help="Keep last vehicle alive after evaluation")
    parser.add_argument("--log-file", type=str, default="eval_trajectory.csv",
                        help="Base name for trajectory logs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for reproducibility")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"❌ Model file not found: {args.model_path}")

    print(f"🎯 Loading model from: {args.model_path}")
    model = PPO.load(args.model_path)  # 注意：不传 env！

    rewards = []
    for ep in range(args.episodes):
        print(f"\n▶️  Starting evaluation episode {ep + 1}/{args.episodes} ...")

        # 创建独立环境实例（每轮新环境）
        env = CarlaEnvMultiObs(
            keep_alive_after_exit=(args.keep_alive and ep == args.episodes - 1),
            log_trajectory=True,
            trajectory_log_file=f"ep{ep + 1}_{args.log_file}",
            max_episode_steps=args.steps_per_ep
        )

        try:
            obs, _ = env.reset(seed=args.seed + ep)
            total_reward = 0.0

            for step in range(args.steps_per_ep):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    print(f"  ⏹️  Episode ended early at step {step + 1}")
                    break

            rewards.append(total_reward)
            print(f"  ✅ Episode {ep + 1} finished | Total Reward: {total_reward:.2f} | Steps: {step + 1}")

        except Exception as e:
            print(f"  ❌ Episode {ep + 1} failed: {e}")
            rewards.append(-1e6)  # 记录失败
        finally:
            env.close()  # 确保清理资源

    # 统计结果
    valid_rewards = [r for r in rewards if r > -1e5]
    if not valid_rewards:
        print("💥 All episodes failed!")
        return

    mean_reward = np.mean(valid_rewards)
    std_reward = np.std(valid_rewards) if len(valid_rewards) > 1 else 0.0

    print("\n" + "="*50)
    print(f"📊 Evaluation Results ({len(valid_rewards)} successful episodes):")
    print(f"   Mean Reward: {mean_reward:.2f}")
    print(f"   Std Dev:     {std_reward:.2f}")
    print(f"   Min:         {min(valid_rewards):.2f}")
    print(f"   Max:         {max(valid_rewards):.2f}")
    print("="*50)
    print(f"📁 Trajectory files saved as: ep1_{args.log_file}, ep2_{args.log_file}, ...")


if __name__ == "__main__":
    main()
