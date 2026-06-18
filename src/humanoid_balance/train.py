<<<<<<< HEAD
import argparse
import math

from src.agents.agent import save_model, get_model
from src.envs import get_env


def train(agent, args):
    total_timesteps = args.total_timesteps
    save_interval = args.save_interval
    model_name = args.model_name

    timesteps_left = total_timesteps
    it = 1

    while timesteps_left > 0:
        print("Starting new train iteration {} of {}".format(it, math.ceil(total_timesteps / save_interval)))
        agent.learn(total_timesteps=min(timesteps_left, save_interval))
        tmp_model_name = model_name + "__{}".format(it) if timesteps_left > save_interval else model_name
        save_model(agent, tmp_model_name, args)
        timesteps_left -= save_interval
        it += 1

    print("Training completed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default="mujoco",
                    help='Choose from mujoco or pybullet')
    parser.add_argument('--total-timesteps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
    parser.add_argument('--load-model', required=False, help='Preload specified model')
    parser.add_argument('--model-name', default="default_model",
                    help='Output model name (default: default_model)')
    parser.add_argument('--save-interval', default=100000, help="After what number of steps saves model")
    parser.add_argument("--device", default="auto", help="Device to use for PyTorch (default: auto)")
    return parser.parse_args()

def main(args):
    env = get_env("pybullet")
    agent = get_model(env, args)

    train(agent=agent, args=args)
    
    env.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)
=======
from stable_baselines3 import SAC
from env_mujoco import make_env
import os

# 1. 初始化环境
env = make_env()

# 2. 模型路径（把你的模型文件改名为 my_humanoid_model.zip）
model_path = "my_humanoid_model.zip"

if os.path.exists(model_path):
    # 加载已有的模型
    model = SAC.load(model_path, env=env)
    print("成功加载现有模型！")
else:
    # 如果没找到，则新建（不推荐，因为重头练很慢）
    model = SAC("MlpPolicy", env, verbose=1)

# 3. 稍微微调一下（比如练 5000 步，确保在你的电脑上能跑通）
model.learn(total_timesteps=5000)

# 4. 保存
model.save("humanoid_final_walking")  
>>>>>>> upstream/main
