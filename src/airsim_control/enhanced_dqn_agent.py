import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import time
from collections import deque

from agents.agent import Agent
from agents.dqn.config import DQNConfig
from agents.dqn.network import QNetwork
from agents.dqn.replay_buffer import ReplayBuffer
from environments.enhanced_drone_env_fixed import EnhancedDroneEnvFixed


class EnhancedDQNAgent(Agent):
    """增强版DQN智能体，包含智能奖励函数和可视化"""

    def __init__(self, client, move_type='velocity', config=None, enable_visualization=True):
        super(EnhancedDQNAgent, self).__init__(client, move_type)
        self.config = config or DQNConfig()
        self.enable_visualization = enable_visualization

        # 创建增强环境
        self.env = EnhancedDroneEnvFixed(client, self.config, enable_visualization)

        # 网络 - 使用更深的网络
        self.policy_net = QNetwork(
            self.config.STATE_DIM + 4,  # 增加目标信息
            self.config.ACTION_DIM,
            self.config.HIDDEN_DIM
        ).to(self.config.DEVICE)

        self.target_net = QNetwork(
            self.config.STATE_DIM + 4,
            self.config.ACTION_DIM,
            self.config.HIDDEN_DIM
        ).to(self.config.DEVICE)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器 - 使用带动量的Adam
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=1e-4  # L2正则化
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=50,
            gamma=0.9
        )

        # 经验回放 - 使用优先级回放
        self.buffer = ReplayBuffer(self.config.BUFFER_SIZE)

        # 探索率 - 使用线性衰减
        self.epsilon = self.config.EPSILON_START
        self.epsilon_decay = (self.config.EPSILON_START - self.config.EPSILON_END) / self.config.MAX_EPISODES

        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []
        self.episode_collisions = []
        self.recent_rewards = deque(maxlen=10)
        self.recent_losses = deque(maxlen=100)

        # 训练参数
        self.update_count = 0
        self.best_avg_reward = -np.inf

        # 计算Q值频率
        self.target_update_freq = 4  # 每4步更新一次目标网络

    def get_state(self):
        return self.env._get_state()

    def act(self, state=None, eval_mode=False):
        """选择动作 (epsilon-greedy with noise)"""
        if state is None:
            state = self.get_state()

        if not eval_mode and np.random.random() < self.epsilon:
            # 添加一些随机性
            return np.random.randint(self.config.ACTION_DIM)

        if not eval_mode:
            # 添加高斯噪声进行探索
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.DEVICE)
                q_values = self.policy_net(state_tensor)
                noise = torch.randn_like(q_values) * 0.1
                action = (q_values + noise).argmax(dim=1).item()
                return action

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.DEVICE)
            return self.policy_net(state_tensor).argmax(dim=1).item()

    def update_network(self):
        """更新网络参数，使用Double DQN"""
        if len(self.buffer) < self.config.BATCH_SIZE:
            return None

        # 采样经验
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.BATCH_SIZE
        )

        states = torch.FloatTensor(states).to(self.config.DEVICE)
        actions = torch.LongTensor(actions).to(self.config.DEVICE)
        rewards = torch.FloatTensor(rewards).to(self.config.DEVICE)
        next_states = torch.FloatTensor(next_states).to(self.config.DEVICE)
        dones = torch.FloatTensor(dones).to(self.config.DEVICE)

        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN计算目标Q值
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.config.GAMMA * next_q_values

        # 计算Huber损失（比MSE更鲁棒）
        loss = F.huber_loss(current_q_values, target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        self.update_count += 1

        # 定期更新目标网络
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def train(self, episodes=None, save_path='./models'):
        """训练增强版DQN"""
        episodes = episodes or self.config.MAX_EPISODES
        os.makedirs(save_path, exist_ok=True)

        print("开始增强版DQN训练...")
        print("="*50)

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            episode_collisions = 0

            start_time = time.time()

            for step in range(self.config.MAX_STEPS):
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)

                # 存储经验
                self.buffer.push(state, action, reward, next_state, float(done))

                # 更新网络
                loss = self.update_network()
                if loss is not None:
                    episode_loss += loss
                    self.recent_losses.append(loss)

                state = next_state
                episode_reward += reward
                steps += 1

                if info['collision']:
                    episode_collisions += 1

                # 早停条件
                if episode > 20 and np.mean(self.recent_rewards) > 50:
                    print(f"\n在第 {episode} 轮提前停止训练！")
                    break

                if done:
                    break

            # 更新探索率（线性衰减）
            self.epsilon = max(
                self.config.EPSILON_END,
                self.epsilon - self.epsilon_decay
            )

            # 记录统计
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            self.episode_distances.append(info.get('distance_to_target', 0))
            self.episode_collisions.append(episode_collisions)
            self.recent_rewards.append(episode_reward)

            # 打印进度
            avg_reward = np.mean(self.recent_rewards)
            avg_loss = np.mean(self.recent_losses) if len(self.recent_losses) > 0 else 0

            print(f"第 {episode+1}/{episodes} 轮 | "
                  f"奖励: {episode_reward:.2f} | "
                  f"平均(10): {avg_reward:.2f} | "
                  f"步数: {steps} | "
                  f"损失: {avg_loss:.4f} | "
                  f"探索率: {self.epsilon:.3f} | "
                  f"碰撞: {episode_collisions} | "
                  f"时间: {time.time()-start_time:.2f}秒")

            # 每10个episode保存一次模型
            if episode % 10 == 0:
                self.save(save_path)

            # 更新可视化（只在训练完成后）
            if self.enable_visualization and episode == episodes - 1:
                print("正在生成训练可视化图表...")
                self.env.update_visualization(episode+1, self.episode_rewards)

            # 保存最佳模型
            current_avg = np.mean(self.recent_rewards)
            if current_avg > self.best_avg_reward:
                self.best_avg_reward = current_avg
                self.save_best_model(save_path)

        # 保存最终模型
        self.save(save_path)

        # 最终统计
        print("\n" + "="*50)
        print("训练完成！")
        print(f"最终平均奖励（最近10轮）: {np.mean(self.recent_rewards):.2f}")
        print(f"最佳平均奖励: {self.best_avg_reward:.2f}")
        print(f"平均每轮长度: {np.mean(self.episode_lengths):.2f}")
        print(f"总碰撞次数: {np.sum(self.episode_collisions)}")
        print(f"模型已保存到: {save_path}")

        # 保存最终可视化
        if self.enable_visualization:
            self.env.save_training_plot()

    def run(self, episodes=10):
        """测试模式运行"""
        self.policy_net.eval()
        total_reward = 0
        test_collisions = 0

        print("\n" + "="*30)
        print("开始测试模式")
        print("="*30)

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.config.MAX_STEPS):
                action = self.act(state, eval_mode=True)
                next_state, reward, done, info = self.env.step(action)

                state = next_state
                episode_reward += reward

                if info['collision']:
                    test_collisions += 1

                if done:
                    break

            total_reward += episode_reward
            print(f"测试轮次 {episode+1}: 奖励 = {episode_reward:.2f} | "
                  f"距离 = {info.get('distance_to_target', 0):.2f}米 | "
                  f"碰撞 = {1 if info['collision'] else 0}")

        print(f"\n测试结果:")
        print(f"平均奖励: {total_reward / episodes:.2f}")
        print(f"总碰撞次数: {test_collisions}")
        print(f"成功率: {(episodes - test_collisions) / episodes * 100:.1f}%")

    def save(self, path='./models'):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config.__dict__,
            'episode_rewards': self.episode_rewards,
            'best_avg_reward': self.best_avg_reward
        }, os.path.join(path, 'enhanced_dqn_model.pth'))

    def save_best_model(self, path):
        """保存最佳模型"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config.__dict__,
            'episode_rewards': self.episode_rewards,
            'best_avg_reward': self.best_avg_reward
        }, os.path.join(path, 'best_enhanced_dqn_model.pth'))

    def load(self, path='./models/enhanced_dqn_model.pth'):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.best_avg_reward = checkpoint.get('best_avg_reward', 0)
        print(f"Model loaded from {path}")