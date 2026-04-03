import mujoco
import numpy as np
from mujoco import viewer
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import pickle
import os
from collections import deque
import random
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ==================== 1. 数据预处理和轨迹数据集 ====================

class MotionDataset(Dataset):
    """轨迹数据集类"""
    def __init__(self, walk_data, squat_data, sequence_length=10):
        self.sequence_length = sequence_length
        
        # 首先打印数据形状进行调试
        print(f"Walk qpos shape: {walk_data['qpos'].shape}")
        print(f"Walk qvel shape: {walk_data['qvel'].shape}")
        if 'qacc' in walk_data:
            print(f"Walk qacc shape: {walk_data['qacc'].shape}")
        
        print(f"Squat qpos shape: {squat_data['qpos'].shape}")
        print(f"Squat qvel shape: {squat_data['qvel'].shape}")
        if 'qacc' in squat_data:
            print(f"Squat qacc shape: {squat_data['qacc'].shape}")
        
        # 处理行走数据
        walk_qpos = walk_data['qpos']
        walk_qvel = walk_data['qvel']
        
        # 创建状态-动作对
        states_walk = np.hstack([walk_qpos, walk_qvel])
        actions_walk = walk_qpos[1:] - walk_qpos[:-1]
        states_walk = states_walk[:-1]  # 对齐
        
        # 处理深蹲数据
        squat_qpos = squat_data['qpos']
        squat_qvel = squat_data['qvel']
        
        states_squat = np.hstack([squat_qpos, squat_qvel])
        actions_squat = squat_qpos[1:] - squat_qpos[:-1]
        states_squat = states_squat[:-1]
        
        # 合并数据
        self.states = np.vstack([states_walk, states_squat])
        self.actions = np.vstack([actions_walk, actions_squat])
        
        # 打印合并后的形状
        print(f"合并后 states shape: {self.states.shape}")
        print(f"合并后 actions shape: {self.actions.shape}")
        
        # 计算统计数据用于归一化
        self.state_mean = self.states.mean(axis=0)
        self.state_std = self.states.std(axis=0) + 1e-8
        self.action_mean = self.actions.mean(axis=0)
        self.action_std = self.actions.std(axis=0) + 1e-8
        
        # 归一化
        self.states_norm = (self.states - self.state_mean) / self.state_std
        self.actions_norm = (self.actions - self.action_mean) / self.action_std
        
        print(f"数据集大小: {len(self.states)}")
        print(f"状态维度: {self.states.shape[1]}, 动作维度: {self.actions.shape[1]}")
    
    def __len__(self):
        return len(self.states) - self.sequence_length
    
    def __getitem__(self, idx):
        # 获取序列数据
        state_seq = self.states_norm[idx:idx+self.sequence_length]
        action_seq = self.actions_norm[idx:idx+self.sequence_length]
        
        return torch.FloatTensor(state_seq), torch.FloatTensor(action_seq)
    
    def denormalize_action(self, action_norm):
        """将归一化的动作还原"""
        return action_norm * self.action_std + self.action_mean

# ==================== 2. 模仿学习模型 ====================

class BehavioralCloning(nn.Module):
    """行为克隆模型"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        print(f"初始化行为克隆模型: state_dim={state_dim}, action_dim={action_dim}")
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state):
        features = self.encoder(state)
        action = self.action_predictor(features)
        value = self.value_head(features)
        return action, value
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, path)
        print(f"模型已保存到: {path}")
    
    @classmethod
    def load(cls, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(checkpoint['state_dim'], checkpoint['action_dim'])
        model.load_state_dict(checkpoint['state_dict'])
        print(f"模型已从 {path} 加载")
        return model

# ==================== 3. 强化学习环境 ====================

class UnitreeEnv(gym.Env):
    """Unitree H1 强化学习环境"""
    def __init__(self, model_path, initial_pos=None):
        super().__init__()
        
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        
        # 打印模型信息
        print(f"模型关节数 (nq): {self.model.nq}")
        print(f"模型速度数 (nv): {self.model.nv}")
        print(f"模型执行器数 (nu): {self.model.nu}")
        
        # 初始位置
        if initial_pos is not None:
            self.data.qpos[:] = initial_pos
        
        # 状态和动作空间
        self.action_dim = self.model.nu
        self.state_dim = self.model.nq + self.model.nv
        
        print(f"环境状态维度: {self.state_dim}")
        print(f"环境动作维度: {self.action_dim}")
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # 目标速度和方向
        self.target_speed = 0.5  # m/s
        self.target_direction = 0  # 前进方向
        
        # 奖励相关参数
        self.alive_bonus = 1.0
        self.speed_weight = 1.0
        self.energy_weight = 0.001
        self.smoothness_weight = 0.1
        
        # 状态跟踪
        self.prev_qpos = None
        self.prev_action = None
        self.step_count = 0
        self.max_steps = 1000
        
        # 渲染相关
        self.viewer = None
        self.render_mode = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        if hasattr(self, 'initial_pos'):
            self.data.qpos[:] = self.initial_pos
        
        self.prev_qpos = self.data.qpos.copy()
        self.prev_action = np.zeros(self.action_dim)
        self.step_count = 0
        
        state = self._get_state()
        return state, {}
    
    def _get_state(self):
        """获取当前状态"""
        return np.concatenate([self.data.qpos, self.data.qvel])
    
    def _compute_reward(self, action):
        """计算奖励"""
        # 1. 存活奖励
        reward = self.alive_bonus
        
        # 2. 速度奖励
        pelvis_vel = self.data.qvel[:3]  # 骨盆线速度
        forward_vel = pelvis_vel[0]  # x方向速度
        speed_reward = np.exp(-np.abs(forward_vel - self.target_speed))
        reward += self.speed_weight * speed_reward
        
        # 3. 能量消耗惩罚
        torque = self.data.qfrc_actuator
        if len(torque) > 0:
            energy = np.sum(np.abs(torque * self.data.qvel[:len(torque)]))
            reward -= self.energy_weight * energy
        
        # 4. 动作平滑惩罚
        if self.prev_action is not None:
            action_diff = np.sum(np.square(action - self.prev_action))
            reward -= self.smoothness_weight * action_diff
        
        # 5. 稳定性奖励 (保持直立)
        if self.model.nq > 3:
            torso_angle = self.data.qpos[3]  # 躯干俯仰角
            stability_reward = np.exp(-10 * np.abs(torso_angle))
            reward += 0.5 * stability_reward
        
        return reward
    
    def step(self, action):
        """执行一步动作"""
        # 应用动作
        self.data.ctrl[:] = action
        
        # 模拟一步
        mujoco.mj_step(self.model, self.data)
        
        # 更新状态
        state = self._get_state()
        reward = self._compute_reward(action)
        self.step_count += 1
        
        # 检查终止条件
        terminated = False
        truncated = False
        
        # 跌倒检测
        if self.model.nq > 2:
            torso_height = self.data.qpos[2]
            if torso_height < 0.3:  # 躯干高度过低
                terminated = True
                reward -= 10.0
        
        # 最大步数限制
        if self.step_count >= self.max_steps:
            truncated = True
        
        # 更新历史记录
        self.prev_action = action.copy()
        self.prev_qpos = self.data.qpos.copy()
        
        return state, reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            if self.viewer is None:
                self.viewer = viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 3
                self.viewer.cam.azimuth = 90
                self.viewer.cam.elevation = -20
                self.viewer.cam.lookat[:] = [0, 0, 1]
            
            self.viewer.sync()
            time.sleep(0.01)
        elif mode == 'rgb_array':
            # 如果需要返回图像数组
            pass
    
    def close(self):
        """关闭渲染器"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# ==================== 4. PPO 算法 ====================

class PPOBuffer:
    """PPO经验回放缓冲区"""
    def __init__(self, state_dim, action_dim, buffer_size, gamma=0.99, lam=0.95):
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.buffer_size = buffer_size
    
    def store(self, state, action, reward, value, log_prob):
        assert self.ptr < self.buffer_size
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1
    
    def finish_path(self, last_value=0):
        """计算GAE优势函数"""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        self.path_start_idx = self.ptr
    
    def _discount_cumsum(self, x, discount):
        """计算折扣累积和"""
        cumsum = 0
        for t in reversed(range(len(x))):
            cumsum = x[t] + discount * cumsum
            x[t] = cumsum
        return x
    
    def get(self):
        """获取所有数据"""
        assert self.ptr == self.buffer_size
        self.ptr, self.path_start_idx = 0, 0
        
        # 标准化优势函数
        advantages = self.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return (
            torch.FloatTensor(self.states),
            torch.FloatTensor(self.actions),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(advantages),
            torch.FloatTensor(self.values)
        )

class PPOAgent:
    """PPO智能体"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_ratio=0.2, train_epochs=10):
        
        print(f"初始化PPO智能体: state_dim={state_dim}, action_dim={action_dim}")
        
        # 策略网络
        self.policy = BehavioralCloning(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # PPO参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.train_epochs = train_epochs
        
        # 动作标准差 (可学习)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # 训练记录
        self.episode_rewards = []
        self.episode_lengths = []
    
    def get_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_mean, value = self.policy(state_tensor)
            action_std = torch.exp(self.log_std)
            
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            
        return action.numpy()[0], log_prob.numpy()[0], value.numpy()[0]
    
    def update(self, buffer):
        """更新策略网络"""
        states, actions, old_log_probs, advantages, old_values = buffer.get()
        
        total_loss = 0
        for _ in range(self.train_epochs):
            # 计算新策略
            action_means, values = self.policy(states)
            action_stds = torch.exp(self.log_std)
            
            dist = torch.distributions.Normal(action_means, action_stds)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # 计算比率
            ratios = torch.exp(log_probs - old_log_probs)
            
            # 裁剪PPO目标
            advantages = advantages.squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            # 策略损失
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(values.squeeze(), old_values.squeeze())
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            total_loss += loss.item()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        return total_loss / self.train_epochs
    
    def save(self, path):
        """保存智能体"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'log_std': self.log_std,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'state_dim': self.policy.state_dim,
            'action_dim': self.policy.action_dim
        }
        torch.save(checkpoint, path)
        print(f"智能体已保存到: {path}")
    
    def load(self, path):
        """加载智能体"""
        checkpoint = torch.load(path, map_location='cpu')
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.log_std = checkpoint['log_std']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        print(f"智能体已从 {path} 加载")

# ==================== 5. MuJoCo 渲染器 ====================

class MuJoCoRenderer:
    """MuJoCo模型渲染器"""
    def __init__(self, model_path, camera_config=None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        # 默认相机配置
        self.default_camera = {
            'distance': 3.0,
            'azimuth': 90.0,
            'elevation': -20.0,
            'lookat': [0.0, 0.0, 1.0]
        }
        
        if camera_config:
            self.default_camera.update(camera_config)
    
    def init_viewer(self):
        """初始化查看器"""
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
            
            # 设置相机
            self.viewer.cam.distance = self.default_camera['distance']
            self.viewer.cam.azimuth = self.default_camera['azimuth']
            self.viewer.cam.elevation = self.default_camera['elevation']
            self.viewer.cam.lookat[:] = self.default_camera['lookat']
        
        return self.viewer
    
    def render_trajectory(self, qpos_trajectory, qvel_trajectory=None, fps=30):
        """渲染轨迹"""
        self.init_viewer()
        
        print(f"渲染轨迹，共 {len(qpos_trajectory)} 帧，FPS: {fps}")
        
        frame_duration = 1.0 / fps
        
        try:
            for i in range(len(qpos_trajectory)):
                # 设置状态
                self.data.qpos[:] = qpos_trajectory[i]
                if qvel_trajectory is not None and i < len(qvel_trajectory):
                    self.data.qvel[:] = qvel_trajectory[i]
                
                # 正向动力学
                mujoco.mj_forward(self.model, self.data)
                
                # 同步渲染
                self.viewer.sync()
                
                # 控制帧率
                time.sleep(frame_duration)
                
        except KeyboardInterrupt:
            print("渲染被用户中断")
        except Exception as e:
            print(f"渲染出错: {e}")
        finally:
            self.close()
    
    def render_policy(self, policy, max_steps=1000, fps=30):
        """渲染策略执行"""
        self.init_viewer()
        
        print(f"渲染策略执行，最大步数: {max_steps}, FPS: {fps}")
        
        # 重置环境
        mujoco.mj_resetData(self.model, self.data)
        
        frame_duration = 1.0 / fps
        step_count = 0
        
        try:
            while self.viewer.is_running() and step_count < max_steps:
                # 获取当前状态
                state = np.concatenate([self.data.qpos, self.data.qvel])
                
                # 通过策略获取动作
                if isinstance(policy, PPOAgent):
                    action, _, _ = policy.get_action(state)
                elif isinstance(policy, BehavioralCloning):
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        action, _ = policy(state_tensor)
                        action = action.numpy()[0]
                elif callable(policy):
                    action = policy(state)
                else:
                    raise ValueError("不支持的策略类型")
                
                # 应用动作
                self.data.ctrl[:] = action
                
                # 模拟一步
                mujoco.mj_step(self.model, self.data)
                
                # 同步渲染
                self.viewer.sync()
                
                # 控制帧率
                time.sleep(frame_duration)
                
                step_count += 1
                
                # 跌倒检测
                if self.model.nq > 2:
                    torso_height = self.data.qpos[2]
                    if torso_height < 0.3:
                        print(f"在第 {step_count} 步跌倒")
                        break
                
                # 进度显示
                if step_count % 100 == 0:
                    print(f"已执行 {step_count} 步")
        
        except KeyboardInterrupt:
            print("渲染被用户中断")
        except Exception as e:
            print(f"渲染出错: {e}")
        finally:
            self.close()
            print(f"总共执行了 {step_count} 步")
    
    def compare_rendering(self, expert_trajectory, policy, max_steps=500, fps=30):
        """对比渲染专家轨迹和策略"""
        print("开始对比渲染...")
        
        # 渲染专家轨迹
        print("\n1. 渲染专家轨迹:")
        self.render_trajectory(expert_trajectory['qpos'][:max_steps], 
                              expert_trajectory['qvel'][:max_steps] if 'qvel' in expert_trajectory else None,
                              fps=fps)
        
        time.sleep(2)  # 等待一下
        
        # 渲染策略
        print("\n2. 渲染学习到的策略:")
        self.render_policy(policy, max_steps=max_steps, fps=fps)
    
    def record_video(self, policy, output_path, max_steps=1000, fps=30):
        """录制策略执行的视频"""
        # 这里需要安装额外的库来录制视频
        print(f"录制视频到 {output_path}...")
        # 实际实现需要添加视频录制代码
        self.render_policy(policy, max_steps=max_steps, fps=fps)
    
    def close(self):
        """关闭查看器"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# ==================== 6. 训练流程 ====================

class RobotTrainer:
    """机器人训练器"""
    def __init__(self, model_path, data_paths, output_dir="./training_output"):
        self.model_path = model_path
        self.data_paths = data_paths
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        
        # 初始化组件
        self.env = None
        self.dataset = None
        self.agent = None
        self.renderer = None
        
        # 训练记录
        self.training_log = {
            "start_time": None,
            "end_time": None,
            "total_episodes": 0,
            "best_reward": -float('inf'),
            "config": {}
        }
    
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        data_walk = np.load(self.data_paths['walk'], allow_pickle=True)
        data_squat = np.load(self.data_paths['squat'], allow_pickle=True)
        
        self.dataset = MotionDataset(data_walk, data_squat)
        return self.dataset
    
    def train_imitation(self, epochs=100, batch_size=32):
        """训练模仿学习"""
        print("\n" + "="*50)
        print("模仿学习训练")
        print("="*50)
        
        if self.dataset is None:
            self.load_data()
        
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # 初始化模型
        state_dim = self.dataset.states.shape[1]
        action_dim = self.dataset.actions.shape[1]
        model = BehavioralCloning(state_dim, action_dim)
        
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for states, actions in dataloader:
                batch_count += 1
                
                states_flat = states[:, -1, :]
                actions_flat = actions[:, -1, :]
                
                predicted_actions, _ = model(states_flat)
                loss = F.mse_loss(predicted_actions, actions_flat)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
                
                # 保存检查点
                if epoch % 50 == 0:
                    checkpoint_path = os.path.join(
                        self.output_dir, 
                        "checkpoints", 
                        f"imitation_epoch_{epoch}.pth"
                    )
                    model.save(checkpoint_path)
        
        # 保存最终模型
        final_path = os.path.join(self.output_dir, "imitation_final.pth")
        model.save(final_path)
        
        # 绘制损失曲线
        self.plot_training_curve(losses, "Imitation Learning Loss", "imitation_loss.png")
        
        print("模仿学习训练完成!")
        return model, losses
    
    def train_rl(self, imitation_model, num_episodes=500):
        """训练强化学习"""
        print("\n" + "="*50)
        print("强化学习训练")
        print("="*50)
        
        if self.env is None:
            self.env = UnitreeEnv(self.model_path)
        
        # 初始化智能体
        self.agent = PPOAgent(self.env.state_dim, self.env.action_dim)
        
        # 如果维度匹配，复制模仿学习权重
        if imitation_model.state_dim == self.env.state_dim:
            print("复制模仿学习模型权重...")
            for target_param, source_param in zip(self.agent.policy.encoder.parameters(), 
                                                 imitation_model.encoder.parameters()):
                target_param.data.copy_(source_param.data)
        
        self.training_log["start_time"] = datetime.now().isoformat()
        self.training_log["total_episodes"] = num_episodes
        
        # 训练循环
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            buffer = PPOBuffer(self.env.state_dim, self.env.action_dim, buffer_size=2048)
            
            # 收集轨迹
            while True:
                action, log_prob, value = self.agent.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                buffer.store(state, action, reward, value, log_prob)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                done = terminated or truncated
                if done or episode_length >= self.env.max_steps:
                    with torch.no_grad():
                        _, last_value = self.agent.policy(torch.FloatTensor(state).unsqueeze(0))
                        last_value = last_value.item()
                    
                    buffer.finish_path(last_value)
                    break
            
            # 更新策略
            if buffer.ptr == buffer.buffer_size:
                loss = self.agent.update(buffer)
            
            # 记录
            self.agent.episode_rewards.append(episode_reward)
            self.agent.episode_lengths.append(episode_length)
            
            # 更新最佳奖励
            if episode_reward > self.training_log["best_reward"]:
                self.training_log["best_reward"] = episode_reward
            
            # 输出进度
            if episode % 10 == 0:
                avg_reward = np.mean(self.agent.episode_rewards[-10:]) if len(self.agent.episode_rewards) >= 10 else episode_reward
                print(f"Episode {episode}/{num_episodes}, "
                      f"Reward: {episode_reward:.2f}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Length: {episode_length}")
            
            # 定期保存
            if episode % 100 == 0 and episode > 0:
                self.save_checkpoint(episode)
        
        self.training_log["end_time"] = datetime.now().isoformat()
        
        # 保存最终模型
        self.save_checkpoint("final")
        
        # 绘制训练曲线
        self.plot_training_curve(self.agent.episode_rewards, "RL Training Rewards", "rl_rewards.png")
        
        print("强化学习训练完成!")
        return self.agent
    
    def save_checkpoint(self, episode):
        """保存检查点"""
        checkpoint_path = os.path.join(
            self.output_dir, 
            "checkpoints", 
            f"rl_agent_episode_{episode}.pth"
        )
        self.agent.save(checkpoint_path)
        
        # 保存训练日志
        log_path = os.path.join(self.output_dir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        print(f"检查点保存到: {checkpoint_path}")
    
    def plot_training_curve(self, data, title, filename):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.xlabel('Episode/Epoch')
        plt.ylabel('Reward/Loss')
        plt.title(title)
        plt.grid(True)
        
        # 添加平滑曲线
        if len(data) > 10:
            smooth_data = np.convolve(data, np.ones(10)/10, mode='valid')
            plt.plot(range(9, len(data)), smooth_data, 'r-', linewidth=2, label='Smoothed')
            plt.legend()
        
        save_path = os.path.join(self.output_dir, "visualizations", filename)
        plt.savefig(save_path)
        plt.close()
        print(f"训练曲线保存到: {save_path}")
    
    def visualize_results(self, expert_data=None):
        """可视化训练结果"""
        print("\n" + "="*50)
        print("可视化训练结果")
        print("="*50)
        
        if self.renderer is None:
            self.renderer = MuJoCoRenderer(self.model_path)
        
        if self.agent is None:
            print("没有训练好的模型，请先训练!")
            return
        
        # 1. 渲染学习到的策略
        print("\n1. 渲染学习到的策略:")
        self.renderer.render_policy(self.agent.policy, max_steps=1000, fps=30)
        
        # 2. 如果有专家数据，进行对比
        if expert_data is not None:
            print("\n2. 对比专家轨迹和学习策略:")
            self.renderer.compare_rendering(
                expert_data, 
                self.agent.policy, 
                max_steps=500, 
                fps=30
            )
        
        # 3. 渲染多个随机种子
        print("\n3. 不同初始状态下的策略表现:")
        for i in range(3):
            print(f"\n第 {i+1} 次测试:")
            self.renderer.render_policy(self.agent.policy, max_steps=300, fps=30)
            time.sleep(1)
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("\n" + "="*50)
        print("交互式演示")
        print("="*50)
        
        if self.agent is None:
            print("请先训练模型!")
            return
        
        print("控制指令:")
        print("  - 按 'W' 切换行走模式")
        print("  - 按 'S' 切换慢速模式")
        print("  - 按 'R' 重置机器人")
        print("  - 按 'ESC' 退出")
        
        # 这里可以添加更复杂的交互逻辑
        self.visualize_results()

# ==================== 7. 主函数 ====================

def main():
    """主函数"""
    print("="*60)
    print("机器人行走强化学习训练与渲染系统")
    print("="*60)
    
    # 配置文件
    config = {
        'model_path': "RobotH.xml",
        'data_paths': {
            'walk': "walk.npz",
            'squat': "squat.npz"
        },
        'output_dir': "./training_output_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        'train_imitation': True,
        'train_rl': True,
        'visualize': True,
        'imitation_epochs': 50,
        'rl_episodes': 200
    }
    
    # 创建训练器
    trainer = RobotTrainer(
        model_path=config['model_path'],
        data_paths=config['data_paths'],
        output_dir=config['output_dir']
    )
    
    trainer.training_log["config"] = config
    
    try:
        # 1. 加载数据
        dataset = trainer.load_data()
        
        # 2. 训练模仿学习
        if config['train_imitation']:
            imitation_model, imitation_losses = trainer.train_imitation(
                epochs=config['imitation_epochs'],
                batch_size=32
            )
        else:
            # 加载预训练的模仿学习模型
            imitation_model = BehavioralCloning.load("pretrained_imitation.pth")
        
        # 3. 训练强化学习
        if config['train_rl']:
            agent = trainer.train_rl(imitation_model, num_episodes=config['rl_episodes'])
        
        # 4. 可视化结果
        if config['visualize']:
            # 加载专家数据用于对比
            expert_data = np.load(config['data_paths']['walk'], allow_pickle=True)
            trainer.visualize_results(expert_data)
            
            # 运行交互式演示
            trainer.run_interactive_demo()
        
        print("\n" + "="*60)
        print("训练完成! 所有结果保存在:", trainer.output_dir)
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

def quick_demo():
    """快速演示：直接加载和渲染训练好的模型"""
    print("快速演示：加载和渲染训练好的模型")
    
    # 模型路径
    model_path = "RobotH.xml"
    
    # 创建渲染器
    renderer = MuJoCoRenderer(model_path)
    
    # 演示选项
    print("\n演示选项:")
    print("1. 渲染专家轨迹")
    print("2. 加载并渲染训练好的模型")
    print("3. 对比专家和模型")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    try:
        if choice == '1':
            # 加载专家数据
            walk_data = np.load("walk.npz", allow_pickle=True)
            print(f"专家轨迹长度: {len(walk_data['qpos'])} 帧")
            
            # 渲染专家轨迹
            renderer.render_trajectory(walk_data['qpos'], walk_data['qvel'], fps=30)
            
        elif choice == '2':
            # 加载训练好的模型
            model_path = input("输入模型路径 (或按回车使用默认): ").strip()
            if not model_path:
                model_path = "./training_output/checkpoints/rl_agent_final.pth"
            
            if os.path.exists(model_path):
                # 创建环境获取维度信息
                env = UnitreeEnv("RobotH.xml")
                
                # 加载智能体
                checkpoint = torch.load(model_path, map_location='cpu')
                agent = PPOAgent(checkpoint['state_dim'], checkpoint['action_dim'])
                agent.load(model_path)
                
                # 渲染策略
                renderer.render_policy(agent.policy, max_steps=1000, fps=30)
            else:
                print(f"模型文件不存在: {model_path}")
                
        elif choice == '3':
            # 加载专家数据和模型
            walk_data = np.load("walk.npz", allow_pickle=True)
            
            # 尝试加载模型
            model_path = "./training_output/checkpoints/rl_agent_final.pth"
            if os.path.exists(model_path):
                env = UnitreeEnv("RobotH.xml")
                checkpoint = torch.load(model_path, map_location='cpu')
                agent = PPOAgent(checkpoint['state_dim'], checkpoint['action_dim'])
                agent.load(model_path)
                
                # 对比渲染
                renderer.compare_rendering(walk_data, agent.policy, max_steps=500, fps=30)
            else:
                print("没有找到训练好的模型，请先训练!")
                
    except Exception as e:
        print(f"演示出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n选择运行模式:")
    print("1. 完整训练流程")
    print("2. 快速演示")
    
    mode = input("请选择 (1 或 2): ").strip()
    
    if mode == '1':
        main()
    elif mode == '2':
        quick_demo()
    else:
        print("无效选择，运行完整训练流程")
        main()