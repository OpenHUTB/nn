"""
Base Agent classes for RL training.
Contains common functionality shared by DQN and DoubleDQN agents.
"""
import os
import torch
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
import gymnasium as gym

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class SkipFrame(gym.Wrapper):
    """Skip N frames to speed up training"""
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info


class BaseDQNNetwork(nn.Module):
    """Base CNN network architecture for CarRacing"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        channel_n, height, width = in_dim

        if height != 84 or width != 84:
            raise ValueError(f"Input must be (84, 84), got ({height}, {width})")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channel_n, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BaseAgent:
    """Base class for DQN agents with common functionality"""
    
    def __init__(self, state_space_shape, action_n, config=None, config_path=None,
                 load_state=False, load_model=None):
        self.state_shape = state_space_shape
        self.action_n = action_n
        
        # Load config
        if config is None:
            if config_path is None:
                config_path = Path(__file__).parent.parent / 'configs' / 'dqn.yaml'
            elif isinstance(config_path, str):
                config_path = Path(config_path)
            
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
        
        # Hyperparameters with defaults
        self.hyperparameters = self.config.get('hyperparameters', {})
        self.gamma = self.hyperparameters.get('gamma', 0.99)
        self.epsilon = self.hyperparameters.get('epsilon_start', 1.0)
        self.epsilon_decay = self.hyperparameters.get('epsilon_decay', 0.9999)
        self.epsilon_min = self.hyperparameters.get('epsilon_min', 0.05)
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Build networks (to be implemented by subclass)
        self.policy_net = None
        self.frozen_net = None
        self._build_networks()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.hyperparameters.get('lr', 0.0001)
        )
        self.loss_fn = nn.SmoothL1Loss()
        
        # Replay buffer
        self.buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                self.hyperparameters.get('buffer_size', 100000),
                device=torch.device("cpu")
            )
        )
        
        # Training tracking
        self.act_taken = 0
        self.n_updates = 0
        
        # Paths
        repo = Path(__file__).resolve().parents[1]
        self.save_dir = str(repo / "training" / "saved_models")
        self.log_dir = str(repo / "training" / "logs")
        
        # Load model if specified
        if load_state and load_model:
            self.load(os.path.join(self.save_dir, load_model))

    def _build_networks(self):
        """Build policy and target networks - override in subclass"""
        self.policy_net = BaseDQNNetwork(self.state_shape, self.action_n).float()
        self.frozen_net = BaseDQNNetwork(self.state_shape, self.action_n).float()
        self.frozen_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net = self.policy_net.to(self.device)
        self.frozen_net = self.frozen_net.to(self.device)

    def store(self, state, action, reward, new_state, terminated):
        """Store experience in replay buffer"""
        self.buffer.add(TensorDict({
            "state": torch.tensor(state),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
            "new_state": torch.tensor(new_state),
            "terminated": torch.tensor(terminated)
        }, batch_size=[]))

    def get_samples(self, batch_size):
        """Sample batch from replay buffer"""
        batch = self.buffer.sample(batch_size)
        states = batch.get('state').float().to(self.device)
        new_states = batch.get('new_state').float().to(self.device)
        actions = batch.get('action').squeeze().to(self.device)
        rewards = batch.get('reward').squeeze().to(self.device)
        terminateds = batch.get('terminated').squeeze().to(self.device)
        return states, actions, rewards, new_states, terminateds

    def take_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_n)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action_values = self.policy_net(state)
                action_idx = torch.argmax(action_values, axis=1).item()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
            
        self.act_taken += 1
        return action_idx

    def update_net(self, batch_size):
        """Update networks - override in subclass"""
        raise NotImplementedError

    def sync_target_net(self):
        """Copy policy net weights to target net"""
        self.frozen_net.load_state_dict(self.policy_net.state_dict())

    def save(self, save_dir, filename):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{filename}.pt")
        
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'frozen_net_state_dict': self.frozen_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'n_updates': self.n_updates,
            'config': self.config
        }, model_path)
        print(f"Model weights saved to: {model_path}")

    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.frozen_net.load_state_dict(checkpoint['frozen_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.n_updates = checkpoint['n_updates']
        print(f"Loaded weights from {path}")

    def write_log(self, date_list, time_list, reward_list, length_list, 
                  loss_list, epsilon_list, log_filename='log.csv'):
        """Write training log to CSV"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        rows = [
            ['date'] + date_list,
            ['time'] + time_list,
            ['reward'] + reward_list,
            ['length'] + length_list,
            ['loss'] + loss_list,
            ['epsilon'] + epsilon_list
        ]
        with open(os.path.join(self.log_dir, log_filename), 'w') as f:
            csv.writer(f).writerows(rows)


def plot_rewards(episode_num, reward_list, n_steps):
    """Plot training rewards with moving average"""
    plt.figure(1)
    rewards_tensor = torch.tensor(reward_list, dtype=torch.float)
    
    if len(rewards_tensor) >= 11:
        eval_reward = rewards_tensor[-10:]
        mean_reward = round(torch.mean(eval_reward).item(), 2)
        std_reward = round(torch.std(eval_reward).item(), 2)
        plt.clf()
        plt.title(f'Episode #{episode_num}: {n_steps} steps, '
                  f'reward {mean_reward}±{std_reward}')
    else:
        plt.clf()
        plt.title('Training...')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_tensor.numpy())
    
    if len(rewards_tensor) >= 50:
        reward_f = rewards_tensor[:50]
        means = rewards_tensor.unfold(0, 50, 1).mean(1)
        means = torch.cat((torch.ones(49) * torch.mean(reward_f), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)
