<<<<<<< HEAD
# ===================== Robust import for CarAgent =====================
import os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Agent import CarAgent
# =====================================================================

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from functools import reduce
import logging
from datetime import datetime
from warnings import filterwarnings
from collections import deque
filterwarnings(action='ignore')

# ---------- 可调开关 ----------
SEED = 42
PREVIEW_EVERY = 5        # 每隔多少个回合开一次“真人预览”窗口；设 0 关闭
PREVIEW_STEPS = 150     # 预览时最多跑多少步（避免看太久）
SAVE_VIDEO_EVERY = 10    # 每隔多少回合保存一次评估视频；设 0 关闭
VIDEO_DIR = "videos"     # 视频保存目录
MA_WINDOW = 20           # 滑动平均窗口
# --------------------------------

random.seed(SEED)
np.random.seed(SEED)

# 目录/日志
os.makedirs('Logger', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s :: [%(levelname)s] :: %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(f"Logger/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"),
=======
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
>>>>>>> fc901d9d1fe1398ec352cb396eef596cae0d8350
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CarRacing")

<<<<<<< HEAD
# =============== 训练环境：rgb_array（更快） ===============
env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="rgb_array")
observation_size = reduce(lambda x, y: x*y, env.observation_space.shape)
logger.info(f"Observation Space: {observation_size}")
logger.info(f"Env Action Dim: {env.action_space.shape[0]}")

# =============== 单独的“真人预览”环境（可选） ===============
eval_env = None
if PREVIEW_EVERY > 0:
    eval_env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
    logger.info("Preview env (human) is ON")

# ==== 多档离散 + 互斥过滤 ====
steers = [-1.0, 0.0, 1.0]
gases  = [0.0, 0.5, 1.0]
brakes = [0.0, 0.5, 0.75]

action_space = []
for s in steers:
    for g in gases:
        for b in brakes:
            if g > 0 and b > 0:  # 互斥
                continue
            action_space.append([s, g, b])
logger.info(f"Discrete Actions: {len(action_space)}")  # 15
=======

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
>>>>>>> fc901d9d1fe1398ec352cb396eef596cae0d8350

car = CarAgent(
    action_size=len(action_space),
    action_space=action_space,
    state_size=observation_size,
)

<<<<<<< HEAD
os.makedirs('data', exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# CSV 只在空文件时写表头
csv_path = "test.csv"
if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("Episode,TimeStep,Reward,Memory,Epsilon,MAReward")

# 滑动平均（看趋势更直观）
ma_rewards = deque(maxlen=MA_WINDOW)

def normalize(obs):
    return obs.astype(np.float32) / 255.0

def run_eval_episode(eval_env_like, steps=1500, epsilon_eval=0.0, human=False):
    """用当前策略做一次评估；human=True 时实时渲染。返回(总回报, 步数)。"""
    old_eps = car.epsilon
    car.epsilon = epsilon_eval  # 0=纯利用
    obs, _ = eval_env_like.reset(seed=SEED)
    obs = normalize(obs)
    total, k = 0.0, 0
    for k in range(steps):
        action = car.get_action(obs)
        obs, r, term, trunc, _ = eval_env_like.step(np.array(action, dtype=np.float32))
        obs = normalize(obs)
        total += r
        if human:
            eval_env_like.render()
        if term or trunc:
            break
    car.epsilon = old_eps
    return total, k + 1

try:
    for episode_no in range(1000):
        logger.info(f"Episode {episode_no + 1}")

        # Reset + normalize
        observation, _ = env.reset(seed=SEED)
        observation = normalize(observation)

        score = 0.0

        for t in range(200000):
            # 选动作 & 执行
            action = car.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(
                np.array(action, dtype=np.float32)
            )
            next_observation = normalize(next_observation)

            # 经验入池 & 训练
            car.append_sample(observation, action, reward, next_observation, terminated)
            if t % 10 == 0:
                car.train_model()

            score += reward
            observation = next_observation

            if terminated or truncated:
                break
            if score <= -10:
                logger.info("Early stop: negative reward gate")
                break

        # 滑动平均
        ma_rewards.append(score)
        ma = float(np.mean(ma_rewards)) if len(ma_rewards) > 0 else score
        logger.info(f"Steps: {t + 1} | Reward: {score:.2f} | MA({MA_WINDOW}): {ma:.2f} | Mem: {len(car.memory)} | Eps: {car.epsilon:.4f}")

        # 记录到 CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            f.write(f"\n{episode_no+1},{t+1},{score},{len(car.memory)},{car.epsilon},{ma}")

        # 定期保存权重
        if (episode_no + 1) % 10 == 0:
            car.save_model_weights(episode=episode_no + 1)

        # 预览：开窗口看一小段（不影响训练速度）
        if eval_env is not None and (episode_no + 1) % PREVIEW_EVERY == 0:
            preview_score, preview_steps = run_eval_episode(eval_env, steps=PREVIEW_STEPS, epsilon_eval=0.0, human=True)
            logger.info(f"[PREVIEW] Steps: {preview_steps} | Reward: {preview_score:.2f}")

        # 录视频：生成 mp4，便于回放/分享
        if SAVE_VIDEO_EVERY > 0 and (episode_no + 1) % SAVE_VIDEO_EVERY == 0:
            name_prefix = f"eval_ep{episode_no + 1}"
            venv = RecordVideo(
                gym.make("CarRacing-v3", domain_randomize=False, render_mode="rgb_array"),
                video_folder=VIDEO_DIR,
                name_prefix=name_prefix
            )
            vscore, vsteps = run_eval_episode(venv, steps=PREVIEW_STEPS, epsilon_eval=0.0, human=False)
            venv.close()
            logger.info(f"[VIDEO] Saved to ./{VIDEO_DIR}/{name_prefix}*.mp4 | Steps: {vsteps} | Reward: {vscore:.2f}")

        # Epsilon 衰减
        if car.epsilon > car.epsilon_min:
            car.epsilon *= car.epsilon_decay

finally:
    env.close()
    if eval_env is not None:
        eval_env.close()
=======
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
>>>>>>> fc901d9d1fe1398ec352cb396eef596cae0d8350
