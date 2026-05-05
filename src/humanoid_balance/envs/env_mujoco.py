import gymnasium as gym
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv

def make_env(render_mode=None):
    # 使用 Humanoid-v4，它默认的观测空间正是 376 维
    env = gym.make("Humanoid-v4", render_mode=render_mode)
    return env
