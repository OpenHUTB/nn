import torch
import torch.nn as nn
from torchvision import models
from torch import distributions
from torch import optim
import numpy as np
import time
import source.utility as util
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class ActorCritic(nn.Module):
    def __init__(self, ac_dim, hidden_dim, n_layers, gamma, learning_rate):
        super(ActorCritic, self).__init__()
        self.ac_dim = ac_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.resnet = models.resnet50(pretrained=True)
        # using pretrained resnet50
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.layers = self.build_layers()
        self.actor_layer = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.ac_dim),
            nn.Softmax(dim=1),
        ])
        self.critic_layer = nn.Linear(self.hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def process_imgs(self, imgs):
        """process_imgs processes PIL images with Resnet50 and return a mini-batch tensor."""
        if len(imgs.shape) == 3:
            return imgs.unsqueeze(0)
        else:
            return imgs

    def build_layers(self):
        layers = []
        layers.append(nn.Linear(self.resnet.fc.out_features, self.hidden_dim))
        for _ in range(self.n_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, obs):
        obs = self.resnet(obs)
        middle_result = self.layers(obs)
        probs = self.actor_layer(middle_result).squeeze()
        v_value = self.critic_layer(middle_result).squeeze()
        actions_distribution = distributions.Categorical(probs)
        return actions_distribution, v_value

    def get_action(self, obs):
        obs = self.process_imgs(obs).to(device)
        action_prob, _ = self.forward(obs)
        action = action_prob.sample()
        return action

    def compute_advantage(self, rws, terminals, v_current: np.ndarray):
        advantages = np.zeros(rws.size)
        assert v_current.ndim == 1, f"v_current dimintion is not 1, got {v_current.ndim}"
        v_current = np.append(v_current, [0])
        # compute q_value(TD)
        for i in range(len(terminals)):
            if terminals[i] == 1:
                advantages[i] = rws[i] - v_current[i]
            else:
                advantages[i] = rws[i] + self.gamma * \
                    v_current[i + 1] - v_current[i]
        return advantages

    def update(self, paths, epoch_i):
        observations, actions, rewards, next_obs, terminals, frames = util.convert_path2list(
            paths)
        start = time.time()
        loss_list = []
        for i in range(len(paths)):
            obs, acs, rws, nextobs, terminal = observations[
                i], actions[i], rewards[i], next_obs[i], terminals[i]
            with torch.no_grad():
                obs = self.process_imgs(obs).to(device)
                nextobs = self.process_imgs(nextobs).to(device)
            # update critic
            print("fit v model.")
            _, v_current = self.forward(obs)
            self.optimizer.zero_grad()
            _, v_next = self.forward(nextobs)
            target = self.gamma * v_next + util.totensor(rws)
            critic_loss = self.loss_fn(v_current, target)
            critic_loss.backward()
            self.optimizer.step()
            print("fit v model done.")
            # update actor
            print("update actor")
            self.optimizer.zero_grad()
            pred_action, v_value = self.forward(obs)
            advantages = self.compute_advantage(
                rws, terminal, util.tonumpy(v_value))
            loss = -torch.mean(pred_action.log_prob(acs)
                               * util.totensor(advantages))
            loss.backward()
            self.optimizer.step()
            print(f"loss: {loss.item()}")
            loss_list.append(loss.item())
            print("update actor done.")
        end = time.time()
        util.log_training(np.mean(loss_list), epoch_i)
        print(f"time: {end - start}")
