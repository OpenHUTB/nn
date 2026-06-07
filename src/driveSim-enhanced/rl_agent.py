#!/usr/bin/env python
# rl_agent.py - 强化学习模块（修正版：动作裁剪+油门保底+诊断）

import os
import sys
import math
import time
import random
import pickle
from collections import deque
from datetime import datetime

import numpy as np

# 添加 carla 导入
try:
    import carla
except ImportError:
    print("⚠ CARLA not available for RL environment")
    carla = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not installed, RL unavailable")


# ==============================================================================
# -- 神经网络模型定义 ------------------------------------------------------------
# ==============================================================================

class DQNNetwork(nn.Module):
    """DQN网络 - 用于离散动作空间"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)


class ActorNetwork(nn.Module):
    """Actor网络 - 用于连续动作空间 (DDPG/SAC)"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, state):
        return self.network(state)


class CriticNetwork(nn.Module):
    """Critic网络 - 用于连续动作空间"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class PPONetwork(nn.Module):
    """PPO网络 - Actor-Critic结构（修正：actor_mean 后加 Tanh）"""

    def __init__(self, state_dim, action_dim, hidden_dim=256, continuous=True):
        super(PPONetwork, self).__init__()
        self.continuous = continuous

        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor头
        if continuous:
            # 使用 Sequential 包装以确保 Tanh 激活，输出范围 [-1, 1]
            self.actor_mean = nn.Sequential(
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            )
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic头
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.feature(state)
        value = self.critic(features)

        if self.continuous:
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_log_std)
            return action_mean, action_std, value
        else:
            action_logits = self.actor(features)
            return action_logits, value

    def get_action(self, state, deterministic=False):
        features = self.feature(state)
        value = self.critic(features)

        if self.continuous:
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_log_std)

            if deterministic:
                action = action_mean
            else:
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1) if not deterministic else None
            return action, log_prob, value
        else:
            action_logits = self.actor(features)
            dist = torch.distributions.Categorical(logits=action_logits)

            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            return action, log_prob, value


# ==============================================================================
# -- 经验回放缓冲区 --------------------------------------------------------------
# ==============================================================================

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.buffer = deque(pickle.load(f), maxlen=self.buffer.maxlen)


# ==============================================================================
# -- DQN智能体 ------------------------------------------------------------------
# ==============================================================================

class DQNAgent:
    """DQN智能体 - 离散动作空间"""

    def __init__(self, state_dim, action_dim, device='cuda', lr=3e-4, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # 网络
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer()

        self.update_counter = 0
        self.target_update_freq = 100

    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = states.to(self.device)
        actions = actions.long().to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)


# ==============================================================================
# -- PPO智能体 (连续动作空间) ----------------------------------------------------
# ==============================================================================

class PPOAgent:
    """PPO智能体 - 支持连续和离散动作空间"""

    def __init__(self, state_dim, action_dim, device='cuda', lr=3e-4, gamma=0.99,
                 continuous=True, clip_epsilon=0.2, update_epochs=10):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs

        self.network = PPONetwork(state_dim, action_dim, continuous=continuous).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.continuous:
                action_mean, action_std, value = self.network(state)
                dist = torch.distributions.Normal(action_mean, action_std)

                if deterministic:
                    action = action_mean
                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                action_logits, value = self.network(state)
                dist = torch.distributions.Categorical(logits=action_logits)

                if deterministic:
                    action = torch.argmax(action_logits, dim=-1)
                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action)

        return action.cpu().numpy().squeeze(), log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns(self, last_value):
        returns = []
        gae = 0
        values = self.values + [last_value]

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.gamma * 0.95 * (1 - self.dones[step]) * gae
            returns.insert(0, gae + values[step])

        return returns

    def update(self):
        if len(self.states) == 0:
            return

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)

        with torch.no_grad():
            last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
            if self.continuous:
                _, _, last_value = self.network(last_state)
            else:
                _, last_value = self.network(last_state)

        returns = self.compute_returns(last_value.item())
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(self.update_epochs):
            if self.continuous:
                action_means, action_stds, new_values = self.network(states)
                dist = torch.distributions.Normal(action_means, action_stds)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().mean()
            else:
                action_logits, new_values = self.network(states)
                dist = torch.distributions.Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(actions.long())
                entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

        return total_loss / self.update_epochs

    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


# ==============================================================================
# -- CARLA RL环境包装器 (修正版) ------------------------------------------------
# ==============================================================================

class CARLARLEnvironment:
    """CARLA强化学习环境包装器"""

    def __init__(self, vehicle, world, hud, assisted_driving=None):
        self.vehicle = vehicle
        self.world = world
        self.hud = hud
        self.assisted_driving = assisted_driving

        self.state_dim = 12
        self.continuous_actions = True
        self.action_dim = 2

        self.prev_location = None
        self.prev_velocity = 0
        self.step_count = 0
        self.episode_reward = 0
        self.collision_occurred = False
        self.lane_invasion_occurred = False

        self.target_location = None
        self.target_reached_threshold = 5.0

    def set_target(self, location):
        self.target_location = location

    def get_state(self):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        waypoint = self.world.get_map().get_waypoint(transform.location)

        obstacle_dist = 50.0
        obstacle_angle = 0.0
        if self.assisted_driving:
            dist, angle, _ = self.assisted_driving.get_front_obstacle()
            obstacle_dist = min(dist, 50.0)
            obstacle_angle = angle

        if self.target_location is not None:
            target_vec = np.array([
                self.target_location.x - transform.location.x,
                self.target_location.y - transform.location.y
            ])
            target_dist = np.linalg.norm(target_vec)
            target_angle = math.atan2(target_vec[1], target_vec[0]) - math.radians(transform.rotation.yaw)
            target_angle = math.degrees(target_angle)
            target_angle = (target_angle + 180) % 360 - 180
        else:
            target_dist = 0
            target_angle = 0

        left_change = 0.0
        right_change = 0.0
        both_change = 0.0

        try:
            if hasattr(carla, 'LaneChange'):
                left_change = float(waypoint.lane_change in [carla.LaneChange.Left, carla.LaneChange.Both])
                right_change = float(waypoint.lane_change in [carla.LaneChange.Right, carla.LaneChange.Both])
                both_change = float(waypoint.lane_change == carla.LaneChange.Both)
        except:
            pass

        state = np.array([
            speed / 100.0,
            transform.rotation.yaw / 180.0,
            transform.location.x / 500.0,
            transform.location.y / 500.0,
            obstacle_dist / 50.0,
            obstacle_angle / 90.0,
            target_dist / 100.0,
            target_angle / 180.0,
            left_change,
            right_change,
            both_change,
            1.0 if self.collision_occurred else 0.0
        ], dtype=np.float32)

        return state

    def compute_reward(self):
        reward = 0
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        target_speed = 40.0
        speed_reward = -abs(speed - target_speed) / target_speed
        reward += speed_reward * 0.5

        if self.prev_location is not None:
            current_loc = self.vehicle.get_location()
            distance_traveled = current_loc.distance(self.prev_location)
            reward += distance_traveled * 2.0

        if self.target_location is not None:
            current_loc = self.vehicle.get_location()
            current_dist = current_loc.distance(self.target_location)
            if hasattr(self, 'prev_target_dist'):
                reward += (self.prev_target_dist - current_dist) * 3.0
            self.prev_target_dist = current_dist

        if self.collision_occurred:
            reward -= 100

        if self.lane_invasion_occurred:
            reward -= 2
            self.lane_invasion_occurred = False

        waypoint = self.world.get_map().get_waypoint(transform.location)
        lane_center = waypoint.transform.location
        lateral_offset = transform.location.distance(lane_center)
        reward -= lateral_offset * 0.2

        reward += 0.05

        return reward

    def step(self, action):
        # 关键修复1：强制裁剪动作到合法范围
        action = np.clip(action, -1.0, 1.0)

        self.step_count += 1
        self.prev_location = self.vehicle.get_location()

        # 关键修复2：先释放手刹，避免物理锁死
        # 关键修复2：先释放手刹，避免物理锁死
        release_control = carla.VehicleControl()
        release_control.hand_brake = False
        release_control.reverse = False
        release_control.manual_gear_shift = False
        self.vehicle.apply_control(release_control)

        control = carla.VehicleControl()
        control.hand_brake = False
        control.reverse = False
        if self.target_location is not None:
            dist = self.vehicle.get_location().distance(self.target_location)
            if self.step_count % 50 == 0:
                print(f"[ENV] Target distance: {dist:.2f} m")
        if self.continuous_actions:
            steer = float(action[0])
            throttle_brake = float(action[1])

            # 明确映射：正数油门，负数刹车
            if throttle_brake > -0.2:  # 大部分情况给油门
                control.throttle = float(np.clip(0.3 + 0.7 * throttle_brake, 0.2, 1.0))
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = float(np.clip(-throttle_brake * 0.8, 0.0, 0.8))
            control.steer = float(np.clip(steer * 1.2, -1.0, 1.0))

            # 关键修复3：确保有基础油门，避免“蜗牛速度”
            if control.throttle < 0.3 and control.brake == 0.0:
                control.throttle = 0.4

            control.steer = float(np.clip(steer, -1.0, 1.0))

            # 轻微探索噪声
            if not hasattr(self, 'eval_mode') or not self.eval_mode:
                if np.random.random() < 0.05:
                    control.steer += np.random.normal(0, 0.15)
                    control.steer = np.clip(control.steer, -1, 1)
        else:
            # 离散动作略...
            if action == 0:
                control.steer = -0.5; control.throttle = 0.5
            elif action == 1:
                control.steer = 0.0; control.throttle = 0.5
            elif action == 2:
                control.steer = 0.5; control.throttle = 0.5
            elif action == 3:
                control.steer = 0.0; control.throttle = 0.8
            elif action == 4:
                control.steer = 0.0; control.brake = 0.8

        self.vehicle.apply_control(control)

        # 诊断打印
        vel = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        if self.step_count % 100 == 0:
            print(f"[ENV] Step {self.step_count}: steer={control.steer:.2f}, "
                  f"throttle={control.throttle:.2f}, brake={control.brake:.2f}, speed={speed:.2f} km/h")

        next_state = self.get_state()
        reward = self.compute_reward()
        self.episode_reward += reward

        done = False
        # if self.collision_occurred:
        #     done = True
        if self.target_location is not None:
            dist = self.vehicle.get_location().distance(self.target_location)
            if dist < self.target_reached_threshold:
                reward += 100
                done = True

        return next_state, reward, done, {}

    def reset(self):
        self.step_count = 0
        self.episode_reward = 0
        self.collision_occurred = False
        self.lane_invasion_occurred = False
        self.prev_location = None
        self.prev_target_dist = None

        # 强制重置车辆控制状态
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.0
        control.steer = 0.0
        control.hand_brake = False
        control.reverse = False
        control.manual_gear_shift = False
        self.vehicle.apply_control(control)

        return self.get_state()

    def on_collision(self):
        self.collision_occurred = True

    def on_lane_invasion(self):
        self.lane_invasion_occurred = True


# ==============================================================================
# -- RL训练器 -------------------------------------------------------------------
# ==============================================================================

class RLTrainer:
    """强化学习训练器"""

    def __init__(self, env, agent, save_dir='./rl_checkpoints'):
        self.env = env
        self.agent = agent
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.episode_rewards = []
        self.best_reward = -float('inf')

    def train_episode(self, max_steps=1000):
        state = self.env.reset()
        episode_reward = 0

        for step in range(max_steps):
            if isinstance(self.agent, DQNAgent):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.memory.push(state, action, reward, next_state, done)
                loss = self.agent.update()
            else:
                action, log_prob, value = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.store_transition(state, action, log_prob, reward, done, value)

            state = next_state
            episode_reward += reward

            if done:
                break

        if isinstance(self.agent, PPOAgent):
            loss = self.agent.update()

        self.episode_rewards.append(episode_reward)
        return episode_reward

    def train(self, num_episodes=1000, save_freq=100, early_stop_patience=50):
        print(f"开始训练，共 {num_episodes} 个episodes")

        best_reward = -float('inf')
        no_improve_count = 0

        for episode in range(num_episodes):
            reward = self.train_episode()

            avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else reward
            print(f"Episode {episode + 1}/{num_episodes} | Reward: {reward:.2f} | Avg Reward: {avg_reward:.2f}")

            if reward > best_reward:
                best_reward = reward
                no_improve_count = 0
                self.agent.save(os.path.join(self.save_dir, 'best_model.pth'))
                print(f"  -> 新的最佳模型! Reward: {reward:.2f}")
            else:
                no_improve_count += 1

            if (episode + 1) % save_freq == 0:
                self.agent.save(os.path.join(self.save_dir, f'model_episode_{episode + 1}.pth'))

            if no_improve_count >= early_stop_patience:
                print(f"早停于 episode {episode + 1}")
                break

        self.agent.save(os.path.join(self.save_dir, 'final_model.pth'))
        print("训练完成!")

        return self.episode_rewards