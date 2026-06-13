from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import os

# ==================== 创建目录 ====================
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# ==================== 环境创建函数 ====================
def make_env():
    def _init():
        env = gym.make("BipedalWalker-v3", render_mode="human")
        env = Monitor(env, "logs/")
        return env
    return _init

# ==================== 创建向量化环境（修复点！）====================
# 必须传入【函数列表】，而不是直接传函数
env_fns = [make_env()]
env = DummyVecEnv(env_fns)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
env = VecFrameStack(env, n_stack=4)

# ==================== PPO 模型 ====================
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
)

# ==================== 开始训练 ====================
print("=====================================")
print("  双足行走机器人 PPO 训练开始")
print("=====================================")

model.learn(total_timesteps=50000)

# ==================== 保存模型 ====================
model.save("models/ppo_bipedalwalker")
env.save("models/env_stats.pkl")

print("\n训练完成！模型已保存到 models/")

# ==================== 测试演示 ====================
print("\n开始演示机器人行走……")

obs = env.reset()
for _ in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
print("\n✅ 项目运行完成！")