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

# action_size_for_acceleration = [round(a, 2) for a in np.arange(0.75, 1.01, precision)]
action_size_for_acceleration = [1.0]
logger.debug(f"Possible Actions for Acceleration: {action_size_for_acceleration}")
logger.debug(f"Possible Actions for Acceleration: {len(action_size_for_acceleration)}")

# action_size_for_brake = [round(a, 2) for a in np.arange(0.25, 0.5, precision)]
action_size_for_brake = [0.75]
logger.debug(f"Possible Actions for Braking: {action_size_for_brake}")
logger.debug(f"Possible Actions for Braking: {len(action_size_for_brake)}")

# action_size_for_steer = [round(a, 2) for a in np.arange(-1., 1.01, precision)]
action_size_for_steer = [-1.0, 0.00, 1.0]
logger.debug(f"Possible Actions for Steering: {action_size_for_steer}")
logger.debug(f"Possible Actions for Steering: {len(action_size_for_steer)}")

action_space = []
for action_space_acc in action_size_for_acceleration:
    for action_space_brake in action_size_for_brake:
        for action_space_steer in action_size_for_steer:
            action_space.append([action_space_steer, action_space_acc, action_space_brake])
logger.debug(f"Possible Actions: {len(action_space)}")

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
