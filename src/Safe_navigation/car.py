"""
基于深度强化学习（DQN）的AirSim自动驾驶安全导航训练系统
兼容旧版AirSim服务器（版本1）的修复版
"""

import os
import sys
import time
import random
import argparse
from collections import deque
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# AirSim导入
try:
    import airsim
    print(f"✓ AirSim模块导入成功，版本: {airsim.__version__}")
except ImportError as e:
    print(f"AirSim模块导入失败: {e}")
    sys.exit(1)


# ==================== 配置类 ====================
class TrainingConfig:
    """训练配置参数"""

    def __init__(self):
        # 网络参数
        self.image_size = (84, 84)  # 输入图像尺寸
        self.state_dim = 15  # 简化状态维度
        self.action_dim = 5  # 简化动作空间

        # 训练参数
        self.total_episodes = 50  # 测试阶段用少量回合
        self.max_steps = 100
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.gamma = 0.99
        self.tau = 1e-3
        self.update_every = 4

        # 经验回放
        self.buffer_size = 2000
        self.pretrain_length = 100

        # 探索策略
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.99

        # 安全参数
        self.collision_penalty = -5.0
        self.max_speed = 10.0

        # 路径参数
        self.model_save_path = "./models"
        self.log_path = "./logs"
        self.save_interval = 10

        # AirSim参数
        self.ip_address = "127.0.0.1"


# ==================== 神经网络架构 ====================
class SimpleDQN(nn.Module):
    """简化版DQN网络"""

    def __init__(self, state_dim, action_dim, image_channels=3):
        super(SimpleDQN, self).__init__()

        # 视觉编码器
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # 计算卷积层输出维度
        with torch.no_grad():
            sample = torch.zeros(1, image_channels, 84, 84)
            conv_out = self.visual_encoder(sample)
            self.visual_feature_dim = conv_out.shape[1]

        # 状态处理器
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
        )

        # 特征融合层
        fusion_input_dim = self.visual_feature_dim + 32
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
        )

        # 动作价值头
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, image, state):
        # 处理视觉输入
        visual_features = self.visual_encoder(image)

        # 处理状态输入
        state_features = self.state_processor(state)

        # 特征融合
        combined = torch.cat([visual_features, state_features], dim=1)
        fused_features = self.fusion_layer(combined)

        # 输出动作价值
        q_values = self.value_stream(fused_features)

        return q_values


# ==================== 经验回放缓冲区 ====================
class ReplayBuffer:
    """简化版经验回放缓冲区"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self):
        return len(self.buffer)


# ==================== DQN智能体 ====================
class DQNAgent:
    """DQN智能体"""

    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = torch.device(device)

        # 初始化网络
        self.policy_net = SimpleDQN(config.state_dim, config.action_dim).to(self.device)
        self.target_net = SimpleDQN(config.state_dim, config.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(config.buffer_size)

        # 训练参数
        self.epsilon = config.epsilon_start
        self.steps_done = 0

        print(f"初始化DQN智能体，设备: {self.device}")

    def select_action(self, state_image, state_vector, eval_mode=False):
        """选择动作"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.config.action_dim)

        with torch.no_grad():
            image_tensor = torch.FloatTensor(state_image).unsqueeze(0).to(self.device)
            vector_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)

            q_values = self.policy_net(image_tensor, vector_tensor)
            return q_values.argmax(1).item()

    def train_step(self):
        """训练步骤"""
        if len(self.memory) < self.config.pretrain_length:
            return 0

        batch = self.memory.sample(self.config.batch_size)
        if batch is None:
            return 0

        # 解构批数据
        states_img, states_vec, actions, rewards, next_states_img, next_states_vec, dones = zip(*batch)

        # 转换为张量
        states_img = torch.FloatTensor(np.array(states_img)).to(self.device)
        states_vec = torch.FloatTensor(np.array(states_vec)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_img = torch.FloatTensor(np.array(next_states_img)).to(self.device)
        next_states_vec = torch.FloatTensor(np.array(next_states_vec)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 计算当前Q值
        current_q = self.policy_net(states_img, states_vec)
        current_q = current_q.gather(1, actions)

        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states_img, next_states_vec)
            next_q_max = next_q.max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.config.gamma * next_q_max

        # 计算损失
        loss = F.smooth_l1_loss(current_q, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 更新探索率
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, episode, path=None):
        """保存模型"""
        if path is None:
            path = self.config.model_save_path

        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f'airsim_dqn_episode_{episode}.pth')

        torch.save({
            'episode': episode,
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, model_path)

        print(f"模型已保存: {model_path}")

    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

        print(f"模型已加载: {model_path}")


# ==================== AirSim环境封装（兼容旧版） ====================
class AirSimSafetyEnv:
    """兼容旧版AirSim服务器的环境封装"""

    def __init__(self, config):
        self.config = config
        self.client = None
        self.step_count = 0
        self.total_reward = 0
        self.collisions = 0

        print("初始化AirSim环境...")

    def connect(self):
        """连接到AirSim服务器（兼容旧版）"""
        try:
            print(f"尝试连接AirSim服务器 {self.config.ip_address}...")

            # 创建客户端
            self.client = airsim.CarClient()
            self.client.confirmConnection()
            print("✓ 连接成功!")

            # 尝试启用API控制（兼容旧版）
            try:
                self.client.enableApiControl(True)
                print("✓ API控制已启用")
            except Exception as api_error:
                print(f"⚠️  API控制启用失败（可能是旧版）: {api_error}")
                print("尝试继续运行...")

            # 重置车辆
            self.client.reset()
            print("✓ 车辆已重置")

            # 等待稳定
            time.sleep(1.0)

            return True

        except Exception as e:
            print(f"❌ 连接失败: {e}")
            print("\n请确保:")
            print("1. AirSim仿真环境正在运行（如AirSimNH.exe）")
            print("2. 已选择汽车模式")
            print("3. 服务器版本兼容（使用AirSim 1.2.6客户端）")
            return False

    def get_camera_image(self):
        """获取摄像头图像"""
        try:
            # 获取场景图像
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            if responses and len(responses) > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)

                if img1d.size > 0:
                    img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
                    resized = cv2.resize(img_rgb, self.config.image_size)
                    normalized = resized.astype(np.float32) / 255.0
                    normalized = np.transpose(normalized, (2, 0, 1))

                    return normalized

        except Exception as e:
            print(f"获取摄像头图像失败: {e}")

        # 返回空白图像
        return np.zeros((3, *self.config.image_size), dtype=np.float32)

    def get_vehicle_state(self):
        """获取车辆状态"""
        try:
            # 获取车辆状态
            car_state = self.client.getCarState()

            # 获取碰撞信息
            collision_info = self.client.simGetCollisionInfo()

            state_info = {
                'speed': car_state.speed,
                'velocity': [
                    car_state.kinematics_estimated.linear_velocity.x_val,
                    car_state.kinematics_estimated.linear_velocity.y_val,
                    car_state.kinematics_estimated.linear_velocity.z_val
                ],
                'position': [
                    car_state.kinematics_estimated.position.x_val,
                    car_state.kinematics_estimated.position.y_val,
                    car_state.kinematics_estimated.position.z_val
                ],
                'collision': collision_info.has_collided,
                'collision_count': collision_info.collision_count,
            }

            return state_info

        except Exception as e:
            print(f"获取车辆状态失败: {e}")
            return None

    def create_state_vector(self, state_info):
        """创建状态向量"""
        if state_info is None:
            return np.zeros(self.config.state_dim, dtype=np.float32)

        state_vector = []

        # 速度信息
        state_vector.append(state_info['speed'] / self.config.max_speed)
        state_vector.extend([v / 10.0 for v in state_info['velocity'][:2]])

        # 位置信息
        state_vector.extend([p / 100.0 for p in state_info['position'][:2]])

        # 碰撞信息
        state_vector.append(float(state_info['collision']))

        # 补全到指定维度
        while len(state_vector) < self.config.state_dim:
            state_vector.append(np.random.uniform(-0.1, 0.1))

        state_vector = state_vector[:self.config.state_dim]

        return np.array(state_vector, dtype=np.float32)

    def get_state(self):
        """获取当前状态"""
        try:
            image_state = self.get_camera_image()
            state_info = self.get_vehicle_state()
            state_vector = self.create_state_vector(state_info)

            safety_flags = {
                'collision': state_info['collision'] if state_info else False,
            }

            return image_state, state_vector, safety_flags

        except Exception as e:
            print(f"获取状态失败: {e}")
            # 返回默认状态
            image_state = np.zeros((3, *self.config.image_size), dtype=np.float32)
            state_vector = np.zeros(self.config.state_dim, dtype=np.float32)
            safety_flags = {'collision': False}
            return image_state, state_vector, safety_flags

    def apply_action(self, action_idx):
        """应用动作到车辆"""
        # 简化动作空间
        steer_actions = [-0.3, -0.1, 0.0, 0.1, 0.3]
        throttle_actions = [0.0, 0.2, 0.5, 0.8, 1.0]

        # 确保动作索引在范围内
        action_idx = min(action_idx, len(steer_actions) * len(throttle_actions) - 1)
        steer_idx = action_idx % len(steer_actions)
        throttle_idx = min(action_idx // len(steer_actions), len(throttle_actions) - 1)

        # 创建控制命令
        car_controls = airsim.CarControls()
        car_controls.steering = steer_actions[steer_idx]
        car_controls.throttle = throttle_actions[throttle_idx]
        car_controls.brake = 0.0

        # 应用控制
        try:
            self.client.setCarControls(car_controls)
            return car_controls
        except Exception as e:
            print(f"应用控制命令失败: {e}")
            return car_controls

    def calculate_reward(self, current_state_info, safety_flags):
        """计算奖励函数"""
        if current_state_info is None:
            return 0.0

        reward = 0.0
        speed = current_state_info['speed']

        # 基础移动奖励
        if speed > 0.1:
            reward += 0.1

        # 碰撞惩罚
        if safety_flags['collision']:
            reward += self.config.collision_penalty
            self.collisions += 1
            print(f"⚠️ 发生碰撞! 惩罚: {self.config.collision_penalty}")

        # 生存奖励
        reward += 0.01

        return reward

    def step(self, action_idx):
        """执行一步环境交互"""
        self.step_count += 1

        # 获取当前状态
        prev_image, prev_vector, prev_safety = self.get_state()

        # 应用动作
        control = self.apply_action(action_idx)

        # 等待环境响应
        time.sleep(0.1)

        # 获取新状态
        current_image, current_vector, current_safety = self.get_state()

        # 获取状态信息用于计算奖励
        current_state_info = self.get_vehicle_state()

        # 计算奖励
        reward = self.calculate_reward(current_state_info, current_safety)
        self.total_reward += reward

        # 检查是否终止
        done = False
        if current_safety['collision']:
            done = True
            print("💥 终止: 发生碰撞")
        elif self.step_count >= self.config.max_steps:
            done = True
            print("⏱️ 终止: 达到最大步数")

        return (current_image, current_vector, reward,
                prev_image, prev_vector, done, current_safety)

    def reset(self):
        """重置环境"""
        self.step_count = 0
        self.total_reward = 0
        self.collisions = 0

        try:
            self.client.reset()
            time.sleep(1.0)  # 等待重置完成
        except Exception as e:
            print(f"重置环境失败: {e}")

        image_state, vector_state, safety_flags = self.get_state()

        return image_state, vector_state, safety_flags

    def close(self):
        """关闭环境"""
        try:
            print("AirSim环境已关闭")
        except:
            pass


# ==================== 训练函数 ====================
def train_dqn_safety_navigation(resume_model=None):
    """主训练函数"""
    config = TrainingConfig()

    # 创建TensorBoard记录器
    log_dir = f"./logs/airsim_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    # 初始化环境和智能体
    env = AirSimSafetyEnv(config)
    agent = DQNAgent(config, device='cpu')

    # 恢复训练（如果指定）
    if resume_model and os.path.exists(resume_model):
        try:
            agent.load_model(resume_model)
            print(f"从模型恢复训练: {resume_model}")
        except:
            print("模型加载失败，从头开始训练")
    else:
        print("从头开始训练")

    # 连接AirSim服务器
    print("\n" + "=" * 60)
    print("正在连接AirSim服务器...")

    if not env.connect():
        print("无法连接到AirSim服务器，退出训练")
        return

    print("=" * 60)

    # 训练循环
    print(f"\n开始深度强化学习安全导航训练")
    print(f"总回合数: {config.total_episodes}")
    print(f"最大步数: {config.max_steps}")
    print(f"设备: {agent.device}")
    print("=" * 60)

    for episode in range(config.total_episodes):
        # 重置环境
        try:
            image_state, vector_state, safety_flags = env.reset()
        except Exception as e:
            print(f"重置环境失败: {e}")
            break

        episode_reward = 0
        episode_steps = 0
        episode_losses = []
        episode_start_time = time.time()

        # 回合循环
        done = False
        while not done and episode_steps < config.max_steps:
            try:
                # 选择动作
                action = agent.select_action(image_state, vector_state)

                # 执行动作，获取新状态和奖励
                (next_image, next_vector, reward,
                 prev_image, prev_vector, done, next_safety) = env.step(action)

                # 存储经验
                experience = (
                    prev_image, prev_vector, action, reward,
                    next_image, next_vector, done
                )
                agent.memory.push(experience)

                # 训练智能体
                if agent.steps_done % config.update_every == 0:
                    loss = agent.train_step()
                    if loss > 0:
                        episode_losses.append(loss)

                # 更新状态
                image_state, vector_state = next_image, next_vector
                episode_reward += reward
                episode_steps += 1
                agent.steps_done += 1

                # 简单进度显示
                if episode_steps % 10 == 0:
                    print(f"  步数: {episode_steps}, 奖励: {episode_reward:.2f}, 探索率: {agent.epsilon:.3f}")

            except Exception as e:
                print(f"回合执行出错: {e}")
                done = True

        # 计算回合统计
        episode_time = time.time() - episode_start_time
        avg_loss = np.mean(episode_losses) if episode_losses else 0

        # 记录训练数据
        writer.add_scalar('Reward/Episode', episode_reward, episode)
        writer.add_scalar('Loss/Episode', avg_loss, episode)
        writer.add_scalar('Exploration/Epsilon', agent.epsilon, episode)
        writer.add_scalar('Steps/Episode_Steps', episode_steps, episode)

        # 打印回合总结
        print(f"\n回合 {episode + 1}/{config.total_episodes}")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  步数: {episode_steps}")
        print(f"  时间: {episode_time:.1f}s")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  碰撞次数: {env.collisions}")
        print(f"  探索率: {agent.epsilon:.3f}")

        # 保存模型
        if (episode + 1) % config.save_interval == 0:
            agent.save_model(episode + 1)

        # 更新目标网络
        if (episode + 1) % 5 == 0:
            agent.update_target_network()

    # 保存最终模型
    agent.save_model(config.total_episodes)

    # 关闭环境
    env.close()
    writer.close()

    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print(f"模型已保存至: {config.model_save_path}")
    print(f"训练日志: {log_dir}")
    print("=" * 60)


# ==================== 评估函数 ====================
def evaluate_model(model_path, eval_episodes=3):
    """评估训练好的模型"""
    config = TrainingConfig()
    env = AirSimSafetyEnv(config)
    agent = DQNAgent(config, device='cpu')

    # 加载模型
    try:
        agent.load_model(model_path)
        agent.epsilon = 0.01  # 评估时探索率低
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 连接环境
    if not env.connect():
        print("无法连接到AirSim服务器")
        return

    results = []

    for episode in range(eval_episodes):
        # 重置环境
        image_state, vector_state, _ = env.reset()

        episode_reward = 0
        episode_steps = 0

        done = False
        while not done and episode_steps < config.max_steps:
            try:
                # 选择动作（评估模式）
                action = agent.select_action(image_state, vector_state, eval_mode=True)

                # 执行动作
                (next_image, next_vector, reward,
                 _, _, done, _) = env.step(action)

                # 更新
                image_state, vector_state = next_image, next_vector
                episode_reward += reward
                episode_steps += 1

                # 简单显示
                if episode_steps % 10 == 0:
                    print(f"  评估步数: {episode_steps}, 奖励: {episode_reward:.2f}")

            except Exception as e:
                print(f"评估出错: {e}")
                done = True

        results.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'steps': episode_steps,
        })

        print(f"评估回合 {episode + 1}/{eval_episodes}: 奖励={episode_reward:.2f}")

    # 计算平均指标
    if results:
        avg_reward = np.mean([r['reward'] for r in results])

        print("\n" + "=" * 60)
        print("评估结果总结:")
        print(f"平均回合奖励: {avg_reward:.2f}")
        print("=" * 60)

    env.close()


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 70)
    print("基于深度强化学习的AirSim自动驾驶安全导航系统")
    print("版本：兼容旧版AirSim服务器 (1.2.6)")
    print("=" * 70)

    # 创建必要的目录
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("将使用CPU运行")

    # 用户选择
    choice = None
    valid_choices = ['1', '2', '3']

    while choice not in valid_choices:
        print("\n请选择运行模式:")
        print("1. 训练新模型 (推荐先测试连接)")
        print("2. 恢复训练")
        print("3. 评估模型")

        user_input = input("\n请输入选择 (1-3): ").strip()

        if user_input:
            first_char = user_input[0]
            if first_char in valid_choices:
                choice = first_char
            else:
                print(f"错误: 输入 '{user_input}' 无效。请输入 1, 2 或 3。")
        else:
            print("错误: 输入不能为空。请输入 1, 2 或 3。")

    # 根据选择执行
    if choice == '1':
        print("\n" + "=" * 60)
        print("重要提示:")
        print("1. 请确保AirSim仿真环境正在运行")
        print("2. 已选择汽车模式 (Car Mode)")
        print("3. 初始训练只有50回合，用于测试连接")
        print("=" * 60)

        confirm = input("\n确认AirSim环境已启动？(y/n): ").strip().lower()
        if confirm == 'y':
            train_dqn_safety_navigation()
        else:
            print("请先启动AirSim环境再运行程序")
    elif choice == '2':
        model_path = None
        while not model_path:
            model_path = input("请输入模型路径 (例如: ./models/airsim_dqn_episode_10.pth): ").strip()
            if not model_path:
                print("错误: 输入不能为空。")
                continue

            if os.path.exists(model_path):
                print(f"从模型恢复训练: {model_path}")
                train_dqn_safety_navigation(resume_model=model_path)
            else:
                print(f"错误: 模型文件不存在: {model_path}")
                model_path = None
    elif choice == '3':
        model_path = None
        while not model_path:
            model_path = input("请输入要评估的模型路径: ").strip()
            if not model_path:
                print("错误: 输入不能为空。")
                continue

            if os.path.exists(model_path):
                print(f"评估模型: {model_path}")
                evaluate_model(model_path)
            else:
                print(f"错误: 模型文件不存在: {model_path}")
                model_path = None


                # 在代码开头添加测试函数
                def test_airsim_connection():
                    """测试AirSim连接"""
                    try:
                        import airsim
                        client = airsim.CarClient()
                        client.confirmConnection()
                        print("✓ 成功连接到AirSim服务器！")

                        # 获取车辆状态
                        state = client.getCarState()
                        print(f"车辆速度: {state.speed}")

                        client.enableApiControl(True)
                        print("API控制已启用")

                        return True
                    except Exception as e:
                        print(f"❌ 连接失败: {e}")
                        print("\n请确保:")
                        print("1. AirSim仿真环境正在运行")
                        print("2. 已选择汽车模式")
                        print("3. AirSim服务器IP地址正确")
                        return False


                # 在主程序中调用
                if __name__ == "__main__":
                    print("测试AirSim连接...")
                    if test_airsim_connection():
                        print("\n连接测试通过！可以开始训练。")
                    else:
                        print("\n连接测试失败，请检查AirSim环境。")