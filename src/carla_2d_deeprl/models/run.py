import torch
import carla
import numpy as np
from PIL import Image
from itertools import count
import torchvision.transforms as T
from PIL.Image import Resampling


from .dqn import DQN
from min_carla_env.env import ACTIONS, CONFIG, CarlaEnv


import argparse
parser = argparse.ArgumentParser(description='Run dqn model.')
parser.add_argument("--path", help="model path")
args = parser.parse_args()

WIDTH = 120
HEIGHT = 120
channels = 1
model_path = args.path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize = T.Compose([T.ToPILImage(),
                    T.Resize(120, interpolation=Resampling.BICUBIC),
                    T.ToTensor()])


def process_state(state):
    state = np.ascontiguousarray(state, dtype=np.float32) / 6
    state = torch.from_numpy(state)
    state = resize(state).unsqueeze(0).to(device)
    return state


target_net = DQN(HEIGHT, WIDTH, len(ACTIONS), channel=channels).to(device)
#target_net.load_state_dict(torch.load(model_path))
target_net.eval()

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
print("connected")
num_episodes = 90000
try:
    env = CarlaEnv(client, CONFIG, demo=True, world_config={'fast': True})
    prev_act = 4
    prev_count = 0
    for i in range(num_episodes):
        state = env.reset()
        for t in count():
            state = process_state(state)
            with torch.no_grad():
                action = target_net(state).detach()
                print("action {}".format(action))
                action = action.max(1)[1].view(1, 1)
                print("action {}".format(action.item()))
            if prev_act == action:
                prev_count += 1

            if prev_count >= 5:
                prev_count = 0

            prev_act = action
            state, reward, done, _ = env.step(action.item())
            if done:
                break
finally:
    env.mw.clean_world()
