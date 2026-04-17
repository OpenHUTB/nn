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
