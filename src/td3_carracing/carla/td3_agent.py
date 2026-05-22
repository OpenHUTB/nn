import torch
import numpy as np
import torch.nn.functional as F
from td3_models import Actor, Critic


class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        for i in batch:
            s, a, r, ns, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(ns, copy=False))
            done.append(np.array(d, copy=False))

        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, device, use_cnn=True):
        self.device = device
        self.use_cnn = use_cnn

        self.actor = Actor(state_dim, action_dim, max_action, use_cnn).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, use_cnn).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=1e-5)

        self.critic1 = Critic(state_dim, action_dim, use_cnn).to(device)
        self.critic2 = Critic(state_dim, action_dim, use_cnn).to(device)
        self.critic1_target = Critic(state_dim, action_dim, use_cnn).to(device)
        self.critic2_target = Critic(state_dim, action_dim, use_cnn).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.1
        self.noise_clip = 0.3
        self.policy_freq = 2
        self.total_it = 0

        # 动作平滑相关
        self.last_action = None
        self.smooth_alpha = 0.8
        self.max_steer = 1.0
        self.max_throttle = 1.0
        self.max_brake = 1.0

    def select_action(self, state, smooth=True, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        if smooth and self.last_action is not None:
            action = self.smooth_alpha * action + (1 - self.smooth_alpha) * self.last_action
            action[0] = np.clip(action[0], -self.max_steer, self.max_steer)
            action[1] = np.clip(action[1], 0.0, self.max_throttle)
            action[2] = np.clip(action[2], 0.0, self.max_brake)
        else:
            action[0] = np.clip(action[0], -self.max_steer, self.max_steer)
            action[1] = np.clip(action[1], 0.0, self.max_throttle)
            action[2] = np.clip(action[2], 0.0, self.max_brake)

        # 死区：微小转向直接置0
        if abs(action[0]) < 0.05:
            action[0] = 0.0

        self.last_action = action.copy()
        return action


    def train(self):
        if len(self.replay_buffer) < self.batch_size * 10:
            return

        self.total_it += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        # 对转向动作单独降低噪声
        noise[:, 0] = noise[:, 0] * 0.6
        # 对油门和刹车单独处理（确保非负）
        next_action = (self.actor_target(next_state) + noise)
        # 限制范围
        next_action[:, 0] = next_action[:, 0].clamp(-self.max_action[0], self.max_action[0])
        next_action[:, 1] = next_action[:, 1].clamp(0.0, self.max_action[1])
        next_action[:, 2] = next_action[:, 2].clamp(0.0, self.max_action[2])

        # 计算目标 Q 值
        target_q1 = self.critic1_target(next_state, next_action)
        target_q2 = self.critic2_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + (1 - done) * self.gamma * target_q

        # 更新 Critic
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟更新 Actor
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic1.state_dict(), filename + "_critic1.pth")
        torch.save(self.critic2.state_dict(), filename + "_critic2.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1.load_state_dict(torch.load(filename + "_critic1.pth", map_location=self.device))
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2.load_state_dict(torch.load(filename + "_critic2.pth", map_location=self.device))
        self.critic2_target.load_state_dict(self.critic2.state_dict())
