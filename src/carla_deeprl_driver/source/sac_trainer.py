from source.replaybuffer import ReplayBuffer
from source.sac import SAC
from source import utility as util
from torch.utils.tensorboard import SummaryWriter
import torch
import os

config = util.get_env_settings("./config.yaml")


class Trainer:
    """Trainer class for SAC algorithm."""

    def __init__(self, env):
        self.env = env
        self.rb = ReplayBuffer(config['buffer_size'])
        self.sac = SAC(1, config['log_min'], config['log_max'],
                       self.rb, config['gamma'], config['soft_tau'])
        self.summarywriter = SummaryWriter("./log/sac", flush_secs=20)
        self.max_average_frames = 0
        
        os.makedirs("checkpoints/sac", exist_ok=True)

    def train(self, epoch_i: int):
        """Single training iteration."""
        paths = util.sample_n_trajectories(
            config['sample_n'], self.env, self.sac.policy_net, 
            config['max_episode_length'], epoch_i
        )
        
        self.rb.add_rollouts(paths)
        self.save_model(paths, epoch_i)
        
        training_paths = self.rb.sample_random_rollouts(config['sample_n'])
        soft_q_loss, value_loss, policy_loss = self.sac.update(training_paths)
        
        self.log_info(soft_q_loss, value_loss, policy_loss, epoch_i)

    def log_info(self, soft_q_loss: float, value_loss: float, 
                 policy_loss: float, epoch_i: int):
        """Log training metrics to TensorBoard."""
        self.summarywriter.add_scalar("Train/soft_q_loss", soft_q_loss, epoch_i)
        self.summarywriter.add_scalar("Train/value_loss", value_loss, epoch_i)
        self.summarywriter.add_scalar("Train/policy_loss", policy_loss, epoch_i)

    def save_model(self, paths: list, epoch_i: int):
        """Save model checkpoints if performance improved."""
        average_frames = util.check_average_frames(paths)
        if average_frames > self.max_average_frames:
            self.max_average_frames = average_frames
            torch.save(self.sac.soft_q_net.state_dict(), 
                       f"checkpoints/sac/softqnet_{epoch_i}_{int(average_frames)}.pt")
            torch.save(self.sac.target_value_net.state_dict(), 
                       f"checkpoints/sac/target_vnet_{epoch_i}_{int(average_frames)}.pt")
            torch.save(self.sac.policy_net.state_dict(), 
                       f"checkpoints/sac/policynet_{epoch_i}_{int(average_frames)}.pt")
