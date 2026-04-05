"""
DoubleDQN Agent - inherits from BaseAgent.
Only contains the Double DQN-specific update logic.
"""
import torch
import yaml
from pathlib import Path
from torch import nn
from dqn_model.base_agent import BaseAgent, BaseDQNNetwork, SkipFrame, plot_rewards


class DoubleDQNAgent(BaseAgent):
    """Double DQN Agent with soft target updates"""
    
    def __init__(self, state_space_shape, action_n, config_path=None, 
                 load_state=False, load_model=None, **kwargs):
        # Default config path
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'configs' / 'double_dqn.yaml'
        
        config_path = Path(config_path)
        
        # Load and merge configs (base dqn + double dqn specific)
        base_config_path = config_path.parent / 'dqn.yaml'
        with open(base_config_path) as f:
            base_config = yaml.safe_load(f)
        
        with open(config_path) as f:
            ddqn_config = yaml.safe_load(f)
        
        # Merge configs
        merged_config = {'hyperparameters': base_config.get('hyperparameters', {})}
        merged_config['hyperparameters'].update(ddqn_config.get('hyperparameters', {}))
        
        # DoubleDQN specific defaults
        merged_config['hyperparameters'].setdefault('tau', 0.005)
        merged_config['hyperparameters'].setdefault('update_target_every', 10000)
        merged_config['hyperparameters'].setdefault('max_grad_norm', 10.0)
        
        self.config = merged_config
        self.tau = merged_config['hyperparameters']['tau']
        self.update_target_every = merged_config['hyperparameters']['update_target_every']
        self.max_grad_norm = merged_config['hyperparameters']['max_grad_norm']
        
        super().__init__(
            state_space_shape=state_space_shape,
            action_n=action_n,
            config=self.config,
            config_path=None,  # Already loaded
            load_state=load_state,
            load_model=load_model
        )
        
        # Learning rate scheduler
        scheduler_type = self.hyperparameters.get('scheduler_type', 'step')
        if scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.hyperparameters.get('scheduler_step', 10000),
                gamma=self.hyperparameters.get('scheduler_gamma', 0.9)
            )
        else:
            self.scheduler = None
    
    def _build_networks(self):
        """Build policy and target networks"""
        self.policy_net = BaseDQNNetwork(self.state_shape, self.action_n).float()
        self.frozen_net = BaseDQNNetwork(self.state_shape, self.action_n).float()
        self.frozen_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net = self.policy_net.to(self.device)
        self.frozen_net = self.frozen_net.to(self.device)
    
    def update_net(self, batch_size):
        """Double DQN update with soft target updates"""
        self.n_updates += 1
        states, actions, rewards, new_states, terminateds = self.get_samples(batch_size)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use policy net to select, frozen net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(new_states).argmax(1, keepdim=True)
            next_q = self.frozen_net(new_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - terminateds.float().unsqueeze(1)) * \
                      self.gamma * next_q
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Learning rate scheduler
        if self.scheduler:
            self.scheduler.step()
        
        # Soft update target network
        if self.n_updates % self.update_target_every == 0:
            for target_param, policy_param in zip(
                self.frozen_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
                )
        
        return current_q.mean().item(), loss.item()
    
    def save(self, save_dir, filename):
        """Save with scheduler state"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{filename}.pt")
        
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'frozen_net_state_dict': self.frozen_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epsilon': self.epsilon,
            'n_updates': self.n_updates,
            'config': self.config
        }, model_path)
        print(f"Model weights saved to: {model_path}")
    
    def load(self, path):
        """Load with scheduler state"""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.frozen_net.load_state_dict(checkpoint['frozen_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.n_updates = checkpoint['n_updates']
        
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded weights from {path}")
    
    def get_current_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


# Aliases for backwards compatibility
Agent = DoubleDQNAgent
plot_reward = plot_rewards
