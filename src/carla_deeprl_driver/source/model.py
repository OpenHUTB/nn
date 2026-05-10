import torch
import torch.nn as nn
from torchvision import models
from torch import distributions
from torch import optim
import numpy as np
from typing import Tuple, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """Actor-Critic network for A2C algorithm with ResNet50 backbone."""

    def __init__(self, ac_dim: int, hidden_dim: int, n_layers: int, 
                 gamma: float, learning_rate: float):
        super(ActorCritic, self).__init__()
        self.ac_dim = ac_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.layers = self._build_layers()
        self.actor_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ac_dim),
            nn.Softmax(dim=1),
        )
        self.critic_layer = nn.Linear(self.hidden_dim, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def _build_layers(self) -> nn.Sequential:
        """Build fully connected layers between ResNet and output heads."""
        layers = [nn.Linear(self.resnet.fc.out_features, self.hidden_dim)]
        for _ in range(self.n_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def _process_imgs(self, imgs: torch.Tensor) -> torch.Tensor:
        """Add batch dimension if missing."""
        if len(imgs.shape) == 3:
            return imgs.unsqueeze(0)
        return imgs

    def forward(self, obs: torch.Tensor) -> Tuple[distributions.Categorical, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            obs: Input observation tensor
            
        Returns:
            Tuple of (action distribution, value estimate)
        """
        features = self.resnet(obs)
        hidden = self.layers(features)
        probs = self.actor_layer(hidden).squeeze()
        v_value = self.critic_layer(hidden).squeeze()
        action_dist = distributions.Categorical(probs)
        return action_dist, v_value

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample action from the policy.
        
        Args:
            obs: Input observation tensor
            
        Returns:
            Sampled action index
        """
        obs = self._process_imgs(obs).to(device)
        action_prob, _ = self.forward(obs)
        return action_prob.sample()

    def compute_advantage(self, rewards: np.ndarray, terminals: np.ndarray, 
                          v_current: np.ndarray) -> np.ndarray:
        """Compute advantages using TD error.
        
        Args:
            rewards: Array of rewards
            terminals: Array of terminal flags
            v_current: Array of value estimates
            
        Returns:
            Array of advantages
        """
        advantages = np.zeros_like(rewards)
        v_current = np.append(v_current, [0])
        
        for i in range(len(terminals)):
            if terminals[i] == 1:
                advantages[i] = rewards[i] - v_current[i]
            else:
                advantages[i] = rewards[i] + self.gamma * v_current[i + 1] - v_current[i]
        
        return advantages

    def update(self, paths: list, epoch_i: int):
        """Update the actor-critic network.
        
        Args:
            paths: List of trajectory dictionaries
            epoch_i: Current epoch index
        """
        from source import utility as util
        
        observations, actions, rewards, next_obs, terminals, _ = util.convert_path2list(paths)
        loss_list = []
        
        for obs, acs, rws, nextobs, terminal in zip(observations, actions, rewards, next_obs, terminals):
            obs = self._process_imgs(obs).to(device)
            nextobs = self._process_imgs(nextobs).to(device)
            
            _, v_current = self.forward(obs)
            _, v_next = self.forward(nextobs)
            
            target = self.gamma * v_next + util.totensor(rws)
            critic_loss = self.loss_fn(v_current, target)
            
            self.optimizer.zero_grad()
            critic_loss.backward()
            self.optimizer.step()
            
            pred_action, v_value = self.forward(obs)
            advantages = self.compute_advantage(rws, terminal, util.tonumpy(v_value))
            actor_loss = -torch.mean(pred_action.log_prob(acs) * util.totensor(advantages))
            
            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()
            
            loss_list.append(actor_loss.item())
        
        util.log_training(np.mean(loss_list), epoch_i)
