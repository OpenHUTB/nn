import math
import time
import torch
import carla
import random
import logging
import numpy as np
from PIL import Image
from itertools import count

import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from .dqn import DQN
from .replay_memory import ReplayMemory, Transition
from min_carla_env.env import ACTIONS, CarlaEnv
from min_carla_env.config import CONFIG

logger = logging.getLogger(__name__)


BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
TARGET_UPDATE = 10
OPTIM_FREQ = 2
WIDTH = 120
HEIGHT = 120
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize = T.Compose([T.ToPILImage(),
                    T.Resize(WIDTH, interpolation=Image.BICUBIC),
                    T.ToTensor()])


steps_done = 0


def select_action(state, policy_net):
    global steps_done
    global EPS_DECAY
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if steps_done >= 2000 and EPS_DECAY >= 200:
        EPS_DECAY = EPS_DECAY*0.8
        steps_done = 0
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(3)]], device=device,
                            dtype=torch.long)


def process_state(state):
    state = np.ascontiguousarray(state, dtype=np.float32) / 6
    state = torch.from_numpy(state)
    state = resize(state).unsqueeze(0).to(device)
    return state


def optimize_model(memory, policy_net, target_net, optimizer):
    """Original source:
    https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
    """
    if len(memory) < BATCH_SIZE:
        return 0
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


def main(load_path, save_path, episodes, start_episode, experiment_name,
         render, channels, debug):
    writer = SummaryWriter('runs/{}'.format(experiment_name))

    policy_net = DQN(HEIGHT, WIDTH, len(ACTIONS), channel=channels).to(device)
    target_net = DQN(HEIGHT, WIDTH, len(ACTIONS), channel=channels).to(device)
    if load_path is not None:
        policy_net.loadd_state_dict(torch.load(load_path))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters(), lr=5e-3)
    memory = ReplayMemory(8192)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    num_episodes = episodes
    model_episode = 0
    optim_step = 0
    env = None
    try:
        env = CarlaEnv(client, CONFIG, {"render": render}, debug=debug)
        for i_episode in range(start_episode, num_episodes + 1):
            # Initialize the environment and state
            state = None
            for i in range(10):
                state = env.reset()
                if state is None:
                    time.sleep(1)
                    continue
                state = process_state(state)
                break
            if state is None:
                logger.error(f"Episode {i_episode}: failed to get valid state after 10 retries")
                break
            total_reward = 0.0
            # episode_loss = 0.0
            for t in count():
                # Select and perform an action
                action = select_action(state, policy_net)
                # _, reward, done, _ = env.step(action.item())
                next_state, reward, done, _ = env.step(action.item())
                if next_state is None:
                    break

                total_reward += reward
                reward = torch.tensor([reward], device=device)

                next_state = process_state(next_state)
                diff = (state - next_state).sum()
                is_same = torch.all(torch.eq(state, next_state))
                if (is_same or (diff < 5 and diff > -5)) and not done:
                    continue

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state
                optim_step += 1
                if done or t >= 8192:
                    if optim_step >= OPTIM_FREQ:
                        # Perform one step of the optimization (on the target network)
                        loss = optimize_model(memory, policy_net, target_net, optimizer)
                        writer.add_scalar('episode-loss', loss, i_episode)
                        if save_path is not None:
                            model_episode += 1
                            # torch.save(policy_net, "{}-{}.pt".format(save_path, model_episode))
                        optim_step = 0
                    writer.add_scalar('total-reward', total_reward, i_episode)
                    writer.add_scalar('duration', t, i_episode)
                    print("episode#:", i_episode, "model episode#:", model_episode,
                          "step#:", steps_done, "total_reward:",
                          round(total_reward, 10), "duration:", t)
                    break
            # Update the target network
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                torch.save(target_net.state_dict(), "{}-target-{}.pt".format(save_path, i_episode))
    finally:
        if env:
            env.mw.clean_world()
        writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train dqn model.')
    parser.add_argument("--load", default=None, help="model load path")
    parser.add_argument("--save", default=None, help="model save path")
    parser.add_argument("--episodes", default=20000, help="Number episode size")
    parser.add_argument("--start-episode", default=0, help="starting episode")
    parser.add_argument("--experiment-name", default="experiment", help="experiment name")
    parser.add_argument("--render", default=False, help="Client rendering option")
    parser.add_argument("--channels", default=1, help="Model input channel size.")
    parser.add_argument("--debug", default=False, help="Debug environment.")
    args = parser.parse_args()
    load_path = args.load
    save_path = args.save
    episodes = int(args.episodes)
    start_eps = int(args.start_episode)
    experiment_name = args.experiment_name
    render = args.render
    channels = int(args.channels)
    debug = args.debug
    main(load_path, save_path, episodes, start_eps, experiment_name,
         render, channels, debug)
