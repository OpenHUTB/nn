import gymnasium as gym
from functools import reduce
from Agent import CarAgent
import numpy as np
import logging
import os
from datetime import datetime
import itertools

from warnings import filterwarnings
filterwarnings(action='ignore')

# Creating directory to save log files
if not os.path.exists('Logger'):
    os.mkdir('Logger')

# Configuring logger file to save as well display log data in certain format
logging.basicConfig(
    format='%(asctime)s :: [%(levelname)s] :: %(message)s',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler("Logger/" + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + ".log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CarRacing")


# Initialise Environment
env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")

observation_size = reduce(lambda x, y: x*y, env.observation_space.shape)
logger.debug(f"Observation Space: {observation_size}")

logger.debug(f"Action Space: {env.action_space.shape[0]}")
precision = 0.25

steers = [-1.0, 0.0, 1.0]
gases  = [0.0, 0.5, 1.0]
brakes = [0.0, 0.5, 0.75]  # CarRacing 刹车范围 [0,1]

logger.debug(f"Possible Actions for Steering: {steers} (count={len(steers)})")
logger.debug(f"Possible Actions for Gas: {gases} (count={len(gases)})")
logger.debug(f"Possible Actions for Brake: {brakes} (count={len(brakes)})")

action_space = []
for s in steers:
    for g in gases:
        for b in brakes:
            # 互斥：同一动作不允许同时给油且刹车
            if g > 0 and b > 0:
                continue
            action_space.append([s, g, b])

logger.debug(f"Possible Actions (after mutex filter): {len(action_space)}")

car = CarAgent(
    action_size=len(action_space),
    action_space=action_space,
    state_size=observation_size,
)

if not os.path.exists('data'):
    os.mkdir('data')

file = open("test.csv", "a")
content = f"Episode,TimeStep,Reward,Memory,Epsilon"
file.write(content)
file.close()

for episode_no in range(1000):
    logger.critical(f"Episode No.: {episode_no + 1}")

    # Initiate one sample step
    observation = env.reset()[0]
    env.render()

    terminated = False
    score = 0

    for t in range(200000):
        if t % 100 == 0:
            logger.critical(f"Time Step.: {t + 1}")

        # Action Taken
        action = car.get_action(observation)
        if t % 100 == 0:
            logger.debug(f"Action Taken: {action}")

        # Initiate the random step / action recorded in previous line
        next_observation, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float64))

        # Reward for the action taken
        logger.debug(f"Reward: {reward}")

        car.append_sample(observation, action, reward, next_observation, terminated)

        if t % 10 == 0:
            car.train_model()

        # env.render()

        score += reward
        observation = next_observation

        # If terminal state then True else False
        if terminated:
            logger.critical("Episode Terminated By Environment")
            break

        if score <= -10:
            logger.critical("Maximum Negative Reward Reached")
            break

    logger.critical(f"Timesteps covered in Episode: {t+1}")
    logger.critical(f"End of Episode {episode_no + 1}")
    logger.critical(f"Total Reward Collected For This Episode: {score}")
    logger.critical(f"Memory Length: {len(car.memory)}")
    logger.critical(f"Epsilon: {car.epsilon}")

    file = open("test.csv", "a")
    content = f"\n{episode_no+1},{t+1},{score},{len(car.memory)},{car.epsilon}"
    file.write(content)
    file.close()

    if episode_no + 1 % 10 == 0:
        car.save_model_weights(episode=episode_no + 1)

    if car.epsilon > car.epsilon_min:
        car.epsilon *= car.epsilon_decay
        logger.critical(f"Current Epsilon Value: {car.epsilon}")

env.close()
