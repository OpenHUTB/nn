from source.model import ActorCritic
from source.carlaenv import CarlaEnv
import source.utility as util
from source.replaybuffer import ReplayBuffer
from source.model import device
from tqdm import tqdm
import torch
import os


class Trainer:
    """Trainer class for A2C algorithm."""

    def __init__(self):
        self.env = CarlaEnv()
        self.config = util.get_env_settings("./config.yaml")
        self.ac_net = ActorCritic(
            4, self.config['hidden_dim'], self.config['n_layers'], 
            self.config['gamma'], self.config['lr']
        ).to(device)
        self.replaybuffer = ReplayBuffer(self.config['buffer_size'])
        self.max_average_frames = 0
        
        os.makedirs("checkpoints/a2c", exist_ok=True)

    def train(self, epoch_i: int):
        """Single training iteration."""
        paths = util.sample_n_trajectories(
            self.config['sample_n'], self.env, self.ac_net, 
            self.config['max_episode_length'], epoch_i
        )
        
        self.replaybuffer.add_rollouts(paths)
        self.save_model(paths, epoch_i)
        
        training_paths = self.replaybuffer.sample_recent_rollouts(self.config['training_n'])
        self.ac_net.update(training_paths, epoch_i)

    def training_loop(self):
        """Main training loop."""
        print(f"{'='*50}")
        print(f"Training on device: {device}")
        print(f"Total epochs: {self.config['epoch']}")
        print(f"{'='*50}")
        
        for epoch_i in tqdm(range(self.config['epoch']), desc="Training A2C"):
            self.train(epoch_i)

    def save_model(self, paths: list, epoch_i: int):
        """Save model checkpoint if performance improved."""
        average_frames = util.check_average_frames(paths)
        if average_frames > self.max_average_frames:
            self.max_average_frames = average_frames
            torch.save(self.ac_net.state_dict(), f"checkpoints/a2c/model_{epoch_i}_{int(average_frames)}.pt")
