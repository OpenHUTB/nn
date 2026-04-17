import gym
import gym_airsim_multirotor

from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

env = gym.make("airsim-env-v0")

model_path = r"logs\NH_center\2026_04_16_16_37_Multirotor_CNN_FC_PPO\models\model.zip"
model = TD3.load(model_path)

env.model = model

obs = env.reset()

while True:
    action = model.predict(obs)
    obs, rewards, done, info = env.step(action[0])
    if done:
        obs = env.reset()
