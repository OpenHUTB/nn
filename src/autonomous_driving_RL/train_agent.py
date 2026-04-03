# train_agent.py
"""
强化学习智能体训练脚本
- 环境: CarlaEnvMultiObs
- 算法: PPO (Proximal Policy Optimization)
- 特性: 自动日志、定期评估、最佳模型保存、TensorBoard 支持
"""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from carla_env.carla_env_multi_obs import CarlaEnvMultiObs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300000, help="总训练步数 (默认: 300000)")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--model_save_path", type=str, default="./checkpoints/best_model.zip", help="最佳模型保存路径")
    args = parser.parse_args()

    # 创建必要目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    print("🚀 开始训练 PPO 智能体...")
    print(f"  - 总步数: {args.timesteps:,}")
    print(f"  - 日志目录: {args.log_dir}")
    print(f"  - 模型保存路径: {args.model_save_path}")

    # 初始化训练环境（单进程，便于调试）
    env = CarlaEnvMultiObs(keep_alive_after_exit=False)  # 训练时不保留车辆
    env = Monitor(env, filename=os.path.join(args.log_dir, "train_monitor.csv"))

    # 初始化评估环境（独立实例，避免干扰训练）
    eval_env = CarlaEnvMultiObs(keep_alive_after_exit=False)
    eval_env = Monitor(eval_env, filename=os.path.join(args.log_dir, "eval_monitor.csv"))

    # 创建 PPO 模型（使用 Stable Baselines3 默认超参，适合连续控制）
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.log_dir,
        device="auto"  # 自动选择 GPU/CPU
    )

    # 设置评估回调：每 5000 步评估一次，保存最佳模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(args.model_save_path),
        log_path=args.log_dir,
        eval_freq=5000,          # 每 5000 训练步评估一次
        deterministic=True,      # 评估时使用确定性策略
        render=False,            # 不渲染（加速评估）
        n_eval_episodes=3,       # 每次评估跑 3 轮取平均
        verbose=1
    )

    # （可选）添加检查点回调：每 5 万步保存一个 checkpoint
    # checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="./checkpoints/", name_prefix="ppo_carla")

    try:
        # 开始训练
        model.learn(
            total_timesteps=args.timesteps,
            callback=eval_callback,
            tb_log_name="PPO_Carla",      # TensorBoard 中的 run 名称
            reset_num_timesteps=True,     # 从零开始计数（若续训设为 False）
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n🛑 训练被用户中断，正在保存当前模型...")
    finally:
        # 保存最终模型（非最佳）
        final_path = os.path.join(os.path.dirname(args.model_save_path), "final_model.zip")
        model.save(final_path)
        print(f"💾 最终模型已保存至: {final_path}")
        env.close()
        eval_env.close()

    print("\n✅ 训练完成！")
    print("\n📊 查看训练曲线:")
    print("   tensorboard --logdir ./logs")
    print("\n🧪 评估最佳模型:")
    print("   python eval_agent.py --model_path ./checkpoints/best_model.zip")


if __name__ == "__main__":
    main()
