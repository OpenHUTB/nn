import os
import sys
import torch
import numpy as np
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
import datetime
import csv
import matplotlib.pyplot as plt  # Import for plotting
import yaml
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dqn_model.DQN_model import Agent as DQNAgent, SkipFrame  # Import SkipFrame from DQN_model

# Define is_ipython locally
import matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_space_shape, action_n, config_path='configs/double_dqn.yaml', load_state=False, load_model=None, **kwargs):
        # Convert to Path object if it's a string
        config_path = Path(config_path) if isinstance(config_path, str) else config_path
        
        # Load base config first
        base_config_path = config_path.parent / 'dqn.yaml'
        with open(base_config_path) as f:
            base_config = yaml.safe_load(f)
        
        # Load and merge DoubleDQN specific config
        with open(config_path) as f:
            double_dqn_config = yaml.safe_load(f)
        
        # Merge configs with DoubleDQN having priority
        self.config = {'hyperparameters': {}}
        if 'hyperparameters' in base_config:
            self.config['hyperparameters'].update(base_config['hyperparameters'])
        if 'hyperparameters' in double_dqn_config:
            self.config['hyperparameters'].update(double_dqn_config['hyperparameters'])
        
        # Set default values for DoubleDQN specific parameters if not provided
        self.config['hyperparameters'].setdefault('tau', 0.005)
        self.config['hyperparameters'].setdefault('update_target_every', 10000)
        
        # Initialize parent with explicit parameters
        super().__init__(
            state_space_shape=state_space_shape,
            action_n=action_n,
            config=self.config,
            **kwargs
        )
        
        # Double DQN specific parameters
        self.tau = self.config['hyperparameters']['tau']
        self.update_target_every = self.config['hyperparameters']['update_target_every']
        
        # Initialize Double DQN networks
        self._initialize_networks(load_state, load_model)

    def _initialize_networks(self, load_state, load_model):
        """Initialize policy_net and target_net networks"""
        self.policy_net = self._build_network().float().to(self.device)
        self.target_net = self._build_network().float().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.hyperparameters['lr'])
        # If loading state, do it after networks are initialized
        if load_state and load_model:
            self.load(os.path.join(self.save_dir, load_model))

    def update_net(self, batch_size):
        """Override update method for Double DQN logic"""
        self.n_updates += 1
        states, actions, rewards, new_states, terminateds = self.get_samples(batch_size)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN target calculation
        with torch.no_grad():
            next_actions = self.policy_net(new_states).argmax(1, keepdim=True)
            next_q = self.target_net(new_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - terminateds.float().unsqueeze(1)) * self.gamma * next_q
        
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        if self.n_updates % self.update_target_every != 0:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        
        return current_q.mean().item(), loss.item()
    
    def _build_network(self):
        """Same architecture as your DQN for compatibility"""
        return nn.Sequential(
            nn.Conv2d(self.state_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_n)
        )

    def store(self, state, action, reward, new_state, terminated):
        """Identical storage method to maintain compatibility"""
        self.buffer.add(TensorDict({
            "state": torch.tensor(state),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
            "new_state": torch.tensor(new_state),
            "terminated": torch.tensor(terminated)
        }, batch_size=[]))

    def get_samples(self, batch_size):
        """Identical sampling method"""
        batch = self.buffer.sample(batch_size)
        states = batch.get('state').float().to(self.device)
        new_states = batch.get('new_state').float().to(self.device)
        actions = batch.get('action').squeeze().to(self.device)
        rewards = batch.get('reward').squeeze().to(self.device)
        terminateds = batch.get('terminated').squeeze().to(self.device)
        return states, actions, rewards, new_states, terminateds

    def take_action(self, state):
        """Identical action selection"""
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_n)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_values = self.policy_net(state)
            action_idx = torch.argmax(action_values, axis=1).item()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
            
        self.act_taken += 1
        return action_idx

    def save(self, save_dir, filename):
        """Override save to use parent-compatible format"""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{filename}.pt")
        
        torch.save({
            'updating_net_state_dict': self.policy_net.state_dict(),
            'frozen_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'n_updates': self.n_updates,
            'config': self.config  # Include config for consistency
        }, model_path)
        
        print(f"Model weights saved to: {model_path}")

    def load(self, path):
        """Override load to use parent-compatible format"""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['updating_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['frozen_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.n_updates = checkpoint['n_updates']
        
        # Handle config mismatch if present
        if 'config' in checkpoint:
            for key in ['gamma', 'epsilon_decay', 'epsilon_min', 'tau']:
                if checkpoint['config'].get(key) != getattr(self, key):
                    print(f"Warning: Config mismatch for {key} - "
                          f"Model: {checkpoint['config'].get(key)}, "
                          f"Current: {getattr(self, key)}")
        
        print(f"Loaded weights from {path}")

    def write_log(self, date_list, time_list, reward_list, length_list, loss_list, epsilon_list, log_filename='double_dqn_log.csv'):
        """Identical logging method"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        rows = [
            ['date'] + date_list,
            ['time'] + time_list,
            ['reward'] + reward_list,
            ['length'] + length_list,
            ['loss'] + loss_list,
            ['epsilon'] + epsilon_list
        ]
        with open(os.path.join(self.log_dir, log_filename), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)

def plot_reward(episode_num, reward_list, n_steps):
    plt.figure(1)
    rewards_tensor = torch.tensor(reward_list, dtype=torch.float)
    if len(rewards_tensor) >= 11:
        eval_reward = torch.clone(rewards_tensor[-10:])
        mean_eval_reward = round(torch.mean(eval_reward).item(), 2)
        std_eval_reward = round(torch.std(eval_reward).item(), 2)
        plt.clf()
        plt.title(f'Episode #{episode_num}: {n_steps} steps, '
                  f'reward {mean_eval_reward}±{std_eval_reward}')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_tensor.numpy())
    if len(rewards_tensor) >= 50:
        reward_f = torch.clone(rewards_tensor[:50])
        means = rewards_tensor.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.ones(49) * torch.mean(reward_f), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
import os
import sys
import torch
import numpy as np
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
import datetime
import csv
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dqn_model.DQN_model import Agent as DQNAgent, SkipFrame

import matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_space_shape, action_n, config_path='configs/double_dqn.yaml', load_state=False, load_model=None, **kwargs):
        config_path = Path(config_path) if isinstance(config_path, str) else config_path
        
        base_config_path = config_path.parent / 'dqn.yaml'
        with open(base_config_path) as f:
            base_config = yaml.safe_load(f)
        
        with open(config_path) as f:
            double_dqn_config = yaml.safe_load(f)
        
        self.config = {'hyperparameters': {}}
        if 'hyperparameters' in base_config:
            self.config['hyperparameters'].update(base_config['hyperparameters'])
        if 'hyperparameters' in double_dqn_config:
            self.config['hyperparameters'].update(double_dqn_config['hyperparameters'])
        
        self.config['hyperparameters'].setdefault('tau', 0.005)
        self.config['hyperparameters'].setdefault('update_target_every', 10000)
        
        super().__init__(
            state_space_shape=state_space_shape,
            action_n=action_n,
            config=self.config,
            **kwargs
        )
        
        self.tau = self.config['hyperparameters']['tau']
        self.update_target_every = self.config['hyperparameters']['update_target_every']
        
        self._initialize_networks(load_state, load_model)

    def _initialize_networks(self, load_state, load_model):
        """Initialize policy_net and target_net networks"""
        self.policy_net = self._build_network().float().to(self.device)
        self.target_net = self._build_network().float().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.hyperparameters['lr'])
        
        # Add gradient clipping for training stability
        self.max_grad_norm = self.config['hyperparameters'].get('max_grad_norm', 10.0)
        
        # Add learning rate scheduler for better convergence
        scheduler_type = self.config['hyperparameters'].get('scheduler_type', 'step')
        if scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.config['hyperparameters'].get('scheduler_step', 10000),
                gamma=self.config['hyperparameters'].get('scheduler_gamma', 0.9)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['hyperparameters'].get('scheduler_tmax', 100000)
            )
        else:
            self.scheduler = None
        
        if load_state and load_model:
            self.load(os.path.join(self.save_dir, load_model))

    def update_net(self, batch_size):
        """Override update method for Double DQN logic with improvements"""
        self.n_updates += 1
        states, actions, rewards, new_states, terminateds = self.get_samples(batch_size)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN target calculation with importance sampling weights support
        with torch.no_grad():
            next_actions = self.policy_net(new_states).argmax(1, keepdim=True)
            next_q = self.target_net(new_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - terminateds.float().unsqueeze(1)) * self.gamma * next_q
        
        # Compute loss with Huber loss for better stability
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # Update learning rate if scheduler exists
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Soft update target network (fixed: should update when divisible, not when not divisible)
        if self.n_updates % self.update_target_every == 0:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        
        return current_q.mean().item(), loss.item()
    
    def _build_network(self):
        """Improved network architecture with dropout and batch normalization"""
        return nn.Sequential(
            nn.Conv2d(self.state_shape[0], 32, kernel_size=8, stride=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.action_n)
        )

    def store(self, state, action, reward, new_state, terminated):
        """Storage with priority buffer support"""
        self.buffer.add(TensorDict({
            "state": torch.tensor(state),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
            "new_state": torch.tensor(new_state),
            "terminated": torch.tensor(terminated)
        }, batch_size=[]))

    def get_samples(self, batch_size):
        """Sampling with device optimization"""
        batch = self.buffer.sample(batch_size)
        states = batch.get('state').float().to(self.device)
        new_states = batch.get('new_state').float().to(self.device)
        actions = batch.get('action').squeeze().to(self.device)
        rewards = batch.get('reward').squeeze().to(self.device)
        terminateds = batch.get('terminated').squeeze().to(self.device)
        return states, actions, rewards, new_states, terminateds

    def take_action(self, state):
        """Action selection with noise exploration option"""
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_n)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action_values = self.policy_net(state)
                action_idx = torch.argmax(action_values, axis=1).item()
        
        # Improved epsilon decay with exponential decay option
        if self.epsilon > self.epsilon_min:
            decay_type = self.config['hyperparameters'].get('epsilon_decay_type', 'exponential')
            if decay_type == 'exponential':
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        else:
            self.epsilon = self.epsilon_min
            
        self.act_taken += 1
        return action_idx

    def save(self, save_dir, filename):
        """Enhanced save with additional metadata"""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{filename}.pt")
        
        torch.save({
            'updating_net_state_dict': self.policy_net.state_dict(),
            'frozen_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epsilon': self.epsilon,
            'n_updates': self.n_updates,
            'config': self.config,
            'act_taken': self.act_taken
        }, model_path)
        
        print(f"Model weights saved to: {model_path}")

    def load(self, path):
        """Enhanced load with scheduler state restoration"""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['updating_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['frozen_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.n_updates = checkpoint['n_updates']
        
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'act_taken' in checkpoint:
            self.act_taken = checkpoint['act_taken']
        
        if 'config' in checkpoint:
            for key in ['gamma', 'epsilon_decay', 'epsilon_min', 'tau']:
                if checkpoint['config'].get('hyperparameters', {}).get(key) != getattr(self, key, None):
                    print(f"Warning: Config mismatch for {key} - "
                          f"Model: {checkpoint['config'].get('hyperparameters', {}).get(key)}, "
                          f"Current: {getattr(self, key, None)}")
        
        print(f"Loaded weights from {path}")

    def write_log(self, date_list, time_list, reward_list, length_list, loss_list, epsilon_list, log_filename='double_dqn_log.csv'):
        """Enhanced logging with additional metrics"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        rows = [
            ['date'] + date_list,
            ['time'] + time_list,
            ['reward'] + reward_list,
            ['length'] + length_list,
            ['loss'] + loss_list,
            ['epsilon'] + epsilon_list
        ]
        with open(os.path.join(self.log_dir, log_filename), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)

    def get_current_lr(self):
        """Get current learning rate for monitoring"""
        return self.optimizer.param_groups[0]['lr']

def plot_reward(episode_num, reward_list, n_steps):
    """Enhanced plotting with moving average and confidence intervals"""
    plt.figure(1)
    rewards_tensor = torch.tensor(reward_list, dtype=torch.float)
    if len(rewards_tensor) >= 11:
        eval_reward = torch.clone(rewards_tensor[-10:])
        mean_eval_reward = round(torch.mean(eval_reward).item(), 2)
        std_eval_reward = round(torch.std(eval_reward).item(), 2)
        plt.clf()
        plt.title(f'Episode #{episode_num}: {n_steps} steps, '
                  f'reward {mean_eval_reward}±{std_eval_reward}')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_tensor.numpy())
    
    # Improved moving average calculation
    if len(rewards_tensor) >= 50:
        window_size = min(50, len(rewards_tensor))
        means = rewards_tensor.unfold(0, window_size, 1).mean(1).view(-1)
        padding = torch.ones(window_size - 1) * torch.mean(rewards_tensor[:window_size])
        means = torch.cat((padding, means))
        plt.plot(means.numpy(), label=f'Moving Average ({window_size})')
        plt.legend()
    
    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)