import gym
import numpy as np
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv

class TransformObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(TransformObservation, self).__init__(env)
        # 显式定义新的观察空间维度 (44 + 332 = 376)
        self.target_obs_dim = 376  
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.target_obs_dim,), 
            dtype=np.float32
        )

    def observation(self, observation):
        # 统一处理所有观测值，确保返回 376 维
        return np.concatenate([observation, np.zeros(332)]).astype(np.float32)

    def reset(self, **kwargs):
        # 关键修复：reset 也必须通过 observation 包装器处理维度
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        # 旧版 gym 标准返回 4 个值
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

def build_env():
    # render=True 才会显示物理窗口
    env = HumanoidBulletEnv(render=True)
    env = TransformObservation(env)
    return env
