from source.model import ActorCritic
from source.carlaenv import CarlaEnv
import source.utility as util
from source.replaybuffer import ReplayBuffer
from source.model import device
from tqdm import tqdm
import torch


class Trainer(object):
    def __init__(self):
        self.env = CarlaEnv()
        self.config = util.get_env_settings("./config.yaml")
        self.ac_net = ActorCritic(
            4, self.config['hidden_dim'], self.config['n_layers'], self.config['gamma'], self.config['lr']).to(device)
        self.replaybuffer = ReplayBuffer(self.config['buffer_size'])
        self.max_average_frames = 0

    def train(self, epoch_i):
        """on-policy actor-critic training."""
        # sample n trajectories
        print("sample trajectories")
        paths = util.sample_n_trajectories(
            self.config['sample_n'], self.env, self.ac_net, self.config['max_episode_length'], epoch_i)
        # add trajectories to replaybuffer
        print("add to replaybuffer")
        self.replaybuffer.add_rollouts(paths)
        # checkpoints
        self.save_model(paths, epoch_i)
        # sample lastest trajectories for training
        print("update policy")
        training_paths = self.replaybuffer.sample_recent_rollouts(
            self.config['training_n'])
        self.ac_net.update(training_paths, epoch_i)

    def training_loop(self):
        print("*" * 20)
        print(f"use device {device}")
        print("*" * 20)
        for i in tqdm(range(self.config['epoch']), desc="Epoch"):
            self.train(i)

    def save_model(self, paths, epoch_i):
        average_frames = util.check_average_frames(paths)
        if average_frames > self.max_average_frames:
            self.max_average_frames = average_frames
            print(f"save model. Average frames: {average_frames}")
            torch.save(self.ac_net.state_dict(),
                       f"checkpoints/a2c/model{epoch_i}.pt")
