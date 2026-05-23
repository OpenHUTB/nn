import torch
import torch.nn as nn
import torch.optim as optim
import source.utility as util
from torch.distributions import Normal
import carla
from typing import Tuple

config = util.get_env_settings("./config.yaml")
device = util.device


class ValueNetwork(nn.Module):
    """Value network for SAC algorithm."""

    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.resnet = util.build_resnet()
        self.layers = self._build_layers()
        self.optimizer = optim.Adam(self.parameters(), lr=config['valuenet_lr'])
        self.loss_fn = nn.MSELoss()

    def _build_layers(self) -> nn.Sequential:
        layers = [nn.Linear(self.resnet.fc.out_features, config['hidden_dim'])]
        for _ in range(config['n_layers']):
            layers.append(nn.Linear(config['hidden_dim'], config['hidden_dim']))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config['hidden_dim'], 1))
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.resnet(obs.to(device))
        value = self.layers(features)
        return value


class SoftQNet(nn.Module):
    """Soft Q-network for SAC algorithm."""

    def __init__(self, action_dim: int):
        super(SoftQNet, self).__init__()
        self.resnet = util.build_resnet()
        self.layers = self._build_layers(action_dim)
        self.action_dim = action_dim
        self.optimizer = optim.Adam(self.parameters(), lr=config['softq_lr'])
        self.loss_fn = nn.MSELoss()

    def _build_layers(self, action_dim: int) -> nn.Sequential:
        layers = [nn.Linear(self.resnet.fc.out_features + action_dim, config['hidden_dim'])]
        for _ in range(config['n_layers']):
            layers.append(nn.Linear(config['hidden_dim'], config['hidden_dim']))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config['hidden_dim'], 1))
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, acs: torch.Tensor) -> torch.Tensor:
        acs = util.totensor(acs) if not isinstance(acs, torch.Tensor) else acs
        features = self.resnet(obs.to(device))
        input_tensor = torch.cat((features, acs), 1)
        q_value = self.layers(input_tensor)
        return q_value


class PolicyNet(nn.Module):
    """Policy network for SAC algorithm with Gaussian policy."""

    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super(PolicyNet, self).__init__()
        self.resnet = util.build_resnet()
        self.mean_layer = self._build_layers(action_dim)
        self.std_layer = self._build_layers(action_dim)
        self.action_dim = action_dim
        self.log_min = config['log_min']
        self.log_max = config['log_max']
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.parameters(), lr=config['policy_lr'])

    def _build_layers(self, output_dim: int) -> nn.Sequential:
        layers = [nn.Linear(self.resnet.fc.out_features, config['hidden_dim'])]
        for _ in range(config['n_layers']):
            layers.append(nn.Linear(config['hidden_dim'], config['hidden_dim']))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config['hidden_dim'], output_dim))
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.resnet(obs.to(device))
        mean = self.mean_layer(features)
        log_std = self.std_layer(features)
        log_std = torch.clamp(log_std, self.log_min, self.log_max)
        return mean, log_std

    def evaluate(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + self.epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, obs: torch.Tensor) -> carla.VehicleControl:
        with torch.no_grad():
            mean, log_std = self.forward(obs.to(device))
            std = log_std.exp()
            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)
        return carla.VehicleControl(1, util.tonumpy(action), 0)


class SAC:
    """Soft Actor-Critic algorithm implementation."""

    def __init__(self, action_dim: int, log_min: float, log_max: float, 
                 replaybuffer, gamma: float, soft_tau: float):
        self.value_net = ValueNetwork().to(device)
        self.target_value_net = ValueNetwork().to(device)
        
        for param, t_param in zip(self.value_net.parameters(), self.target_value_net.parameters()):
            t_param.data.copy_(param.data)
        
        self.soft_q_net = SoftQNet(action_dim).to(device)
        self.policy_net = PolicyNet(action_dim).to(device)
        self.replaybuffer = replaybuffer
        self.gamma = gamma
        self.soft_tau = soft_tau

    def update(self, paths: list) -> Tuple[float, float, float]:
        obs, acs, rws, next_obs, terminals, _ = util.convert_path2list(paths)
        
        soft_q_value = self.soft_q_net.forward(obs, acs)
        target_soft_q = util.totensor(rws) + self.gamma * (1 - terminals) * self.target_value_net(next_obs)
        soft_q_loss = self.soft_q_net.loss_fn(soft_q_value, target_soft_q.detach())
        
        v_value = self.value_net.forward(obs)
        sample_acs, log_prob, _, _, _ = self.policy_net.evaluate(obs)
        new_soft_q = self.soft_q_net.forward(obs, sample_acs)
        target_v = new_soft_q - log_prob
        value_loss = self.value_net.loss_fn(v_value, target_v.detach())
        
        policy_loss = torch.mean(new_soft_q.detach() - log_prob)
        
        self.soft_q_net.optimizer.zero_grad()
        soft_q_loss.backward()
        self.soft_q_net.optimizer.step()
        
        self.value_net.optimizer.zero_grad()
        value_loss.backward()
        self.value_net.optimizer.step()
        
        self.policy_net.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net.optimizer.step()
        
        for param, t_param in zip(self.value_net.parameters(), self.target_value_net.parameters()):
            t_param.data.copy_(t_param.data * (1 - self.soft_tau) + param.data * self.soft_tau)
        
        return soft_q_loss.item(), value_loss.item(), policy_loss.item()
