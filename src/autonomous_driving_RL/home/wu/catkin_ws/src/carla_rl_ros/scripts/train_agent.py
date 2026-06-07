# train_ppo.py
"""
使用新奖励函数训练 PPO 模型
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from carla_env.carla_env_multi_obs import CarlaEnvMultiObs

if __name__ == "__main__":
    # 创建环境
    env = CarlaEnvMultiObs(
        random_spawn=True,
        max_episode_steps=1000,
        debug=False
    )

    # 日志和模型保存路径
    log_dir = "./logs/"
    checkpoint_dir = "./checkpoints/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 回调：每 10k 步保存一次
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="ppo_carla"
    )

    # 创建 PPO 模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )

    # 开始训练
    print("🚀 开始训练 PPO 模型...")
    model.learn(
        total_timesteps=500_000,
        callback=checkpoint_callback,
        tb_log_name="ppo_run"
    )

    # 保存最终模型
    model.save(os.path.join(checkpoint_dir, "best_model.zip"))
    print("✅ 训练完成，模型已保存至 ./checkpoints/best_model.zip")
    env.close()