import time

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

train = False
test = True
model_path = ""         # update this path with the path to the model you want to load

def euclidean_distance(obs):
    return np.sqrt(np.sum((obs['achieved_goal'] - obs['desired_goal'])**2))

if train:
    env = gym.make("FetchReach-v2")
    model = PPO("MultiInputPolicy", env, verbose=1, device="cuda")
    model.learn(total_timesteps=1_000_000)
    model.save(model_path)


if test:
    model = PPO.load(model_path)

    for _ in range(10):
        env = gym.make("FetchReach-v2", render_mode='human')
        # pick an env such that the gripper is far from the target so we can visualize the movement
        # while True:
        #     obs = env.reset()[0]
        #     dist = euclidean_distance(obs)
        #     # print(f"euclidean distance before testing: {dist}")
        #     if dist > 0.22:
        #         break
        # time.sleep(2)
        obs = env.reset()[0]
        for i in range(500):
            action, _ = model.predict(obs, deterministic=True)
            # action = env.action_space.sample()
            obs, reward, isTerminated, isTruncated, info = env.step(action)
            env.render()
            time.sleep(0.3)
            # calculate euclidean distance between gripper and target
            euclidean_distance = np.sqrt(np.sum((obs['achieved_goal'] - obs['desired_goal'])**2))
            print(f"euclidean distance: {euclidean_distance}")
            if euclidean_distance < 0.05:   # in m
                break

        env.close()


