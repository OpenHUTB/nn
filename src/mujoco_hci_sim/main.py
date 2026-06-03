# main.py（单文件，训练/测试自动判断）
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

MODEL_PATH = "ppo_humanoid_balance"

# 1. 创建环境
env = gym.make("Humanoid-v4", render_mode="human")
env = DummyVecEnv([lambda: env])

if os.path.exists(MODEL_PATH + ".zip"):
    # 已有模型：直接测试
    print("发现已训练模型，进入测试模式...")
    model = PPO.load(MODEL_PATH)
    obs = env.reset()
    for _ in range(3000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            obs = env.reset()
else:
    # 没有模型：开始训练
    print("未找到模型，开始训练...")
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=3e-4, gamma=0.99, clip_range=0.2,
        n_steps=2048, batch_size=64, n_epochs=10
    )
    model.learn(total_timesteps=3_000_000)
    model.save(MODEL_PATH)
    print("训练完成，模型已保存！")

env.close()