import torch
import numpy as np
import torch.nn.functional as F
from collections import deque
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


class ActionSmoother:
    """动作平滑器，使用指数移动平均"""

    def __init__(self, action_dim, smooth_factor=0.3):
        self.action_dim = action_dim
        self.smooth_factor = smooth_factor
        self.prev_action = None

    def smooth(self, action):
        if self.prev_action is None:
            self.prev_action = action.copy()
            return action

        # 指数移动平均
        smoothed = self.smooth_factor * action + (1 - self.smooth_factor) * self.prev_action
        self.prev_action = smoothed.copy()
        return smoothed

    def reset(self):
        self.prev_action = None


class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, device, use_cnn=True, action_smooth_factor=0.3):
        self.device = device
        self.use_cnn = use_cnn

        # 网络初始化
        self.actor = Actor(state_dim, action_dim, max_action, use_cnn).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, use_cnn).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic1 = Critic(state_dim, action_dim, use_cnn).to(device)
        self.critic2 = Critic(state_dim, action_dim, use_cnn).to(device)
        self.critic1_target = Critic(state_dim, action_dim, use_cnn).to(device)
        self.critic2_target = Critic(state_dim, action_dim, use_cnn).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=1e-4
        )

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

        # 动作平滑器
        self.action_smoother = ActionSmoother(action_dim, action_smooth_factor)

        # 动作延迟缓冲
        self.action_history = deque(maxlen=3)

    def select_action(self, state, apply_smoothing=True):
        """选择动作，可选应用平滑"""
        # 设置为评估模式
        self.actor.eval()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            raw_action = self.actor(state_tensor).cpu().numpy().flatten()

        # 切换回训练模式
        self.actor.train()

        if apply_smoothing:
            # 应用指数移动平均平滑
            smoothed_action = self.action_smoother.smooth(raw_action)

            # 额外限制动作变化率
            if len(self.action_history) > 0:
                prev_action = self.action_history[-1]
                max_change = 0.2  # 最大动作变化率
                change = np.abs(smoothed_action - prev_action)
                if np.any(change > max_change):
                    # 限制变化幅度
                    for i in range(len(smoothed_action)):
                        if change[i] > max_change:
                            smoothed_action[i] = prev_action[i] + np.clip(
                                smoothed_action[i] - prev_action[i], -max_change, max_change
                            )

            self.action_history.append(smoothed_action.copy())
            return smoothed_action
        else:
            return raw_action

    def reset_action_history(self):
        """重置动作历史（在episode开始时调用）"""
        self.action_smoother.reset()
        self.action_history.clear()

    def train(self):
        if len(self.replay_buffer) < self.batch_size * 10:
            return

        # 设置为训练模式
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

        self.total_it += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # 添加策略噪声
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        target_q1 = self.critic1_target(next_state, next_action)
        target_q2 = self.critic2_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

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