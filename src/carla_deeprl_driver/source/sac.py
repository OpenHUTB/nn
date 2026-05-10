import torch
import torch.nn as nn
import torch.optim as optim
import source.utility as util
from torch.distributions import Normal
import carla

config = util.get_env_settings("./config.yaml")
device = util.device

"""
SAC paper: http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf
SAC implementation instructions: https://intellabs.github.io/coach/components/agents/policy_optimization/sac.html
SAC code reference: https://github.com/higgsfield/RL-Adventure-2/blob/master/7.soft%20actor-critic.ipynb


action: steer[-1, 1] with fixed throttle 1, and no brake
"""


class ValueNetwork(nn.Module):
    def __init__(self) -> None:
        super(ValueNetwork, self).__init__()
        self.optimizer = optim.Adam(
            self.parameters(), lr=config['valuenet_lr'])
        self.resnet = util.build_resnet()
        self.layers = self.build_layers()
        self.loss_fn = nn.MSELoss()

    def build_layers(self):
        layers = []
        layers.append(
            nn.Linear(self.resnet.fc.out_features, config['hidden_dim']))
        for _ in range(config['n_layers']):
            layers.append(
                nn.Linear(config['hidden_dim'], config['hidden_dim']))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config['hidden_dim'], 1))
        return nn.Sequential(*layers)

    def forward(self, obs):
        obs = self.resnet(obs.to(device))
        value = self.layers(obs)
        return value


class SoftQNet(nn.Module):
    def __init__(self, action_dim) -> None:
        super(SoftQNet, self).__init__()
        self.optimizer = optim.Adam(self.parameters(), lr=config['softq_lr'])
        self.resnet = util.build_resnet()
        self.layers = self.build_layers()
        self.action_dim = action_dim
        self.loss_fn = nn.MSELoss()

    def build_layers(self):
        layers = []
        layers.append(nn.Linear(self.resnet.fc.out_features +
                      self.action_dim, config['hidden_dim']))
        for _ in range(config['n_layers']):
            layers.append(
                nn.Linear(config['hidden_dim'], config['hidden_dim']))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config['hidden_dim'], 1))
        return nn.Sequential(*layers)

    def forward(self, obs, acs):
        acs = util.totensor(acs)
        obs = self.resnet(obs.to(device))
        input = torch.cat((obs, acs), 1)
        q_value = self.layers(input)
        return q_value


class PolicyNet(nn.Module):
    def __init__(self, action_dim, epsilon=1e-6, log_min=-20, log_max=2):
        super(PolicyNet, self).__init__()
        self.optimizer = optim.Adam(self.parameters(), lr=config['policy_lr'])
        self.resnet = util.build_resnet()
        self.std_layer = self.build_layers()
        self.mean_layer = self.build_layers()
        self.action_dim = action_dim
        self.log_min = log_min
        self.log_max = log_max
        self.epsilon = epsilon

    def build_layers(self):
        layers = []
        layers.append(
            nn.Linear(self.resnet.fc.out_features, config['hidden_dim']))
        for _ in range(config['n_layers']):
            layers.append(
                nn.Linear(config['hidden_dim'], config['hidden_dim']))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config['hidden_dim'], self.action_dim))
        return nn.Sequential(*layers)

    def forward(self, obs):
        obs = self.resnet(obs.to(device))
        mean = self.mean_layer(obs)
        log_std = self.std_layer(obs)
        log_std = torch.clamp(log_std, self.log_min, self.log_max)
        return mean, log_std

    def evaluate(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(
            z) - torch.log(1 - action.pow(2) + self.epsilon)
        # log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, obs):
        with torch.no_grad():
            mean, log_std = self.forward(obs.to(device))
            log_std = log_std.exp()
            normal = Normal(mean, log_std)
            z = normal.sample()
            action = torch.tanh(z)
        # only control steer
        return carla.VehicleControl(1, util.tonumpy(action), 0)


class SAC(object):
    def __init__(self, action_dim, log_min, log_max, replaybuffer, gamma, soft_tau) -> None:
        self.value_net = ValueNetwork().to(device)
        self.target_value_net = ValueNetwork().to(device)
        # copy the parameters
        for param, t_param in zip(self.value_net.parameters(), self.target_value_net.parameters()):
            param.data.copy_(t_param.data)
        self.soft_q_net = SoftQNet(action_dim).to(device)
        self.policy_net = PolicyNet(action_dim, log_min, log_max).to(device)
        self.replaybuffer = replaybuffer
        self.gamma = gamma
        # self.mean_lambda = mean_lambda
        # self.std_lambda = std_lambda
        # self.z_lambda = z_lambda
        self.soft_tau = soft_tau

    def update(self, paths):
        # get rollouts
        obs, acs, rws, next_obs, terminals, _ = util.convert_path2list(paths)
        # soft q net loss
        soft_q_value = self.soft_q_net.forward(obs, acs)
        target_soft_q_value = util.totensor(
            rws) + self.gamma * (1 - terminals) * self.target_value_net(next_obs)
        soft_q_loss = self.soft_q_net.loss_fn(
            soft_q_value, target_soft_q_value.detach())
        # value net loss
        v_value = self.value_net.forward(obs)
        sample_acs, log_prob, z, mean, log_std = self.policy_net.evaluate(obs)
        new_soft_q_value = self.soft_q_net.forward(obs, sample_acs)
        target_v_net = new_soft_q_value - log_prob
        value_net_loss = self.value_net.loss_fn(v_value, target_v_net.detach())
        # policy net loss
        policy_loss = torch.mean(new_soft_q_value.detach() - log_prob)
        # policy_loss = (log_prob * (log_prob - log_prob -
        #                log_prob_target).detach()).mean()
        # mean_loss = self.mean_lambda * mean.pow(2).mean()
        # std_loss = self.std_lambda * log_std.pow(2).mean()
        # z_loss = self.z_lambda * z.pow(2).sum(1).mean()
        # policy_loss += mean_loss + std_loss + z_loss

        # update all network
        self.soft_q_net.optimizer.zero_grad()
        soft_q_loss.backward()
        self.soft_q_net.optimizer.step()

        self.value_net.optimizer.zero_grad()
        value_net_loss.backward()
        self.value_net.optimizer.step()

        self.policy_net.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net.optimizer.step()
        # update target value net
        # copy the parameters
        for param, t_param in zip(self.value_net.parameters(), self.target_value_net.parameters()):
            t_param.data.copy_(
                t_param.data * (1 - self.soft_tau) + param.data * self.soft_tau)

        return soft_q_loss.item(), value_net_loss.item(), policy_loss.item()
