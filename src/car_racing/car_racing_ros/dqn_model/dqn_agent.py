"""
DQN Agent - inherits from BaseAgent.
Only contains the DQN-specific update logic.
"""
import torch
import numpy as np
from base_agent import BaseAgent, BaseDQNNetwork, SkipFrame, plot_rewards


class DQNAgent(BaseAgent):
    """Standard DQN Agent"""
    
    def _build_networks(self):
        """Build policy and target networks using base architecture"""
        self.policy_net = BaseDQNNetwork(self.state_shape, self.action_n).float()
        self.frozen_net = BaseDQNNetwork(self.state_shape, self.action_n).float()
        self.frozen_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net = self.policy_net.to(self.device)
        self.frozen_net = self.frozen_net.to(self.device)
    
    def update_net(self, batch_size):
        """Standard DQN update with fixed target network sync"""
        self.n_updates += 1
        states, actions, rewards, new_states, terminateds = self.get_samples(batch_size)
        
        # Current Q values
        current_q = self.policy_net(states)[np.arange(batch_size), actions]
        
        # Target Q values
        with torch.no_grad():
            target_q = rewards + (1 - terminateds.float()) * self.gamma * \
                      self.frozen_net(new_states).max(1)[0]
        
        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Periodic target network sync
        if self.n_updates % self.hyperparameters.get('target_update', 5000) == 0:
            self.sync_target_net()
        
        return current_q.mean().item(), loss.item()


# Aliases for backwards compatibility
Agent = DQNAgent
plot_reward = plot_rewards
