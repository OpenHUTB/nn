import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import os

from agents.agent import Agent
from agents.dqn.config import DQNConfig
from agents.dqn.network import QNetwork
from agents.dqn.replay_buffer import ReplayBuffer
from environments.drone_env import DroneEnv


class DQNAgent(Agent):
    """DQN 强化学习智能体"""

    def __init__(self, client, move_type='velocity', config=None):
        super(DQNAgent, self).__init__(client, move_type)
        self.config = config or DQNConfig()

        # 创建环境
        self.env = DroneEnv(client, self.config)

        # 网络
        self.policy_net = QNetwork(
            self.config.STATE_DIM,
            self.config.ACTION_DIM,
            self.config.HIDDEN_DIM
        ).to(self.config.DEVICE)

        self.target_net = QNetwork(
            self.config.STATE_DIM,
            self.config.ACTION_DIM,
            self.config.HIDDEN_DIM
        ).to(self.config.DEVICE)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.LEARNING_RATE
        )

        # 经验回放
        self.buffer = ReplayBuffer(self.config.BUFFER_SIZE)

        # 探索率
        self.epsilon = self.config.EPSILON_START

        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []

    def get_state(self):
        return self.env._get_state()

    def act(self, state=None, eval_mode=False):
        """选择动作 (epsilon-greedy)"""
        if state is None:
            state = self.get_state()

        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.config.ACTION_DIM)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.DEVICE)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update_network(self):
        """更新网络参数"""
        if len(self.buffer) < self.config.BATCH_SIZE:
            return

        # 采样
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.BATCH_SIZE
        )

        states = torch.FloatTensor(states).to(self.config.DEVICE)
        actions = torch.LongTensor(actions).to(self.config.DEVICE)
        rewards = torch.FloatTensor(rewards).to(self.config.DEVICE)
        next_states = torch.FloatTensor(next_states).to(self.config.DEVICE)
        dones = torch.FloatTensor(dones).to(self.config.DEVICE)

        # 计算 Q 值
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标 Q 值 (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.config.GAMMA * next_q_values

        # 计算损失
        loss = F.mse_loss(q_values, target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, episodes=None, save_path='./models'):
        """训练 DQN"""
        episodes = episodes or self.config.MAX_EPISODES
        os.makedirs(save_path, exist_ok=True)

        total_steps = 0

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0

            for step in range(self.config.MAX_STEPS):
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)

                self.buffer.push(state, action, reward, next_state, float(done))

                loss = self.update_network()
                if loss is not None:
                    episode_loss += loss

                state = next_state
                episode_reward += reward
                steps += 1
                total_steps += 1

                # 如果连续10步平均奖励超过20，提前结束episode
                if episode > 10 and np.mean(self.episode_rewards[-10:]) > 20:
                    break

                if done:
                    break

            # 更新探索率
            self.epsilon = max(
                self.config.EPSILON_END,
                self.epsilon * self.config.EPSILON_DECAY
            )

            # 更新目标网络
            if episode % self.config.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # 记录统计
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)

            # 打印进度
            avg_reward = np.mean(self.episode_rewards[-10:])
            print(f"Episode {episode+1}/{episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg: {avg_reward:.2f} | "
                  f"Steps: {steps} | "
                  f"Epsilon: {self.epsilon:.3f}")

        # 保存模型
        self.save(save_path)

        # 打印最终统计
        if len(self.episode_rewards) > 0:
            final_avg_reward = np.mean(self.episode_rewards)
            max_reward = np.max(self.episode_rewards)
            min_reward = np.min(self.episode_rewards)
            print(f"\n=== Training Summary ===")
            print(f"Total Episodes: {len(self.episode_rewards)}")
            print(f"Final Average Reward: {final_avg_reward:.2f}")
            print(f"Max Reward: {max_reward:.2f}")
            print(f"Min Reward: {min_reward:.2f}")
            print(f"Model saved to {save_path}")
        else:
            print(f"Training completed. Model saved to {save_path}")

    def run(self, episodes=10):
        """测试模式运行"""
        self.policy_net.eval()
        total_reward = 0

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.config.MAX_STEPS):
                action = self.act(state, eval_mode=True)
                next_state, reward, done, info = self.env.step(action)

                state = next_state
                episode_reward += reward

                if done:
                    break

            total_reward += episode_reward
            print(f"Test Episode {episode+1}: Reward = {episode_reward:.2f}")

        print(f"Average Reward: {total_reward / episodes:.2f}")

    def save(self, path='./models'):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config.__dict__
        }, os.path.join(path, 'dqn_model.pth'))

    def load(self, path='./models/dqn_model.pth'):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")
