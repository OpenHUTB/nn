import gymnasium as gym
<<<<<<< HEAD

def build_env():
    env = gym.make("HumanoidStandup-v4", render_mode="human")
    # check_env(env)
    # env.action_space.seed(42)
    # env.reset(seed=42)
=======
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv

def make_env(render_mode=None):
    # 使用 Humanoid-v4，它默认的观测空间正是 376 维
    env = gym.make("Humanoid-v4", render_mode=render_mode)
>>>>>>> upstream/main
    return env
