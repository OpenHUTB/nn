#基于深度学习的无人机控制与可视化系统
import pygame
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pygame.locals import *
import random
import math

# 初始化pygame
pygame.init()

# 设置中文字体
pygame.font.init()
font_path = pygame.font.match_font('simsun')  # 尝试匹配宋体
if not font_path:
    # 如果找不到宋体，使用默认字体
    font = pygame.font.Font(None, 36)
else:
    font = pygame.font.Font(font_path, 24)

# 屏幕设置
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("无人机深度学习控制系统")

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)


# 无人机类
class Drone:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.z = 100  # 高度
        self.angle = 0  # 角度（度）
        self.speed = 2
        self.size = 20
        self.max_height = 300
        self.min_height = 50

    def move(self, dx, dy, dz=0):
        # 根据当前角度调整移动方向
        rad = math.radians(self.angle)
        self.x += dx * math.cos(rad) - dy * math.sin(rad)
        self.y += dx * math.sin(rad) + dy * math.cos(rad)

        # 限制在屏幕内
        self.x = max(self.size, min(WIDTH - self.size, self.x))
        self.y = max(self.size, min(HEIGHT - self.size, self.y))

        # 调整高度
        self.z = max(self.min_height, min(self.max_height, self.z + dz))

    def rotate(self, delta_angle):
        self.angle = (self.angle + delta_angle) % 360

    def draw(self, surface):
        # 绘制无人机（根据高度调整大小和颜色深浅）
        size_factor = self.z / self.max_height
        draw_size = int(self.size * (0.5 + size_factor * 0.5))

        # 无人机中心点
        center = (int(self.x), int(self.y))

        # 绘制无人机机身
        pygame.draw.circle(surface,
                           (int(50 + size_factor * 205),
                            int(50 + size_factor * 105),
                            int(50 + size_factor * 205)),
                           center, draw_size)

        # 绘制无人机旋翼
        rad = math.radians(self.angle)
        rotor_length = draw_size * 0.8

        # 前旋翼
        front_x = self.x + math.cos(rad) * rotor_length
        front_y = self.y + math.sin(rad) * rotor_length
        pygame.draw.line(surface, BLACK, center, (front_x, front_y), 3)

        # 后旋翼
        back_x = self.x - math.cos(rad) * rotor_length
        back_y = self.y - math.sin(rad) * rotor_length
        pygame.draw.line(surface, BLACK, center, (back_x, back_y), 3)

        # 左旋翼
        left_x = self.x - math.sin(rad) * rotor_length
        left_y = self.y + math.cos(rad) * rotor_length
        pygame.draw.line(surface, BLACK, center, (left_x, left_y), 3)

        # 右旋翼
        right_x = self.x + math.sin(rad) * rotor_length
        right_y = self.y - math.cos(rad) * rotor_length
        pygame.draw.line(surface, BLACK, center, (right_x, right_y), 3)

        # 显示高度
        height_text = font.render(f"高度: {int(self.z)}", True, BLACK)
        surface.blit(height_text, (self.x + draw_size, self.y - draw_size))


# 目标点类
class Target:
    def __init__(self):
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(50, HEIGHT - 50)
        self.z = random.randint(80, 250)
        self.radius = 15

    def draw(self, surface):
        # 绘制目标点
        pygame.draw.circle(surface, RED, (self.x, self.y), self.radius, 2)
        pygame.draw.circle(surface, RED, (self.x, self.y), 3)

        # 显示目标高度
        z_text = font.render(f"Z: {self.z}", True, RED)
        surface.blit(z_text, (self.x + self.radius, self.y - self.radius))

    def reset(self):
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(50, HEIGHT - 50)
        self.z = random.randint(80, 250)


# 深度学习控制器模型
class DroneController(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, output_size=4):
        super(DroneController, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # 输出范围[-1, 1]

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


# 强化学习环境
class DroneEnv:
    def __init__(self):
        self.drone = Drone()
        self.target = Target()
        self.max_steps = 500
        self.current_step = 0
        self.reset()

    def reset(self):
        self.drone = Drone()
        self.target.reset()
        self.current_step = 0
        return self.get_state()

    def get_state(self):
        # 状态包括无人机位置、角度、目标位置
        return np.array([
            self.drone.x / WIDTH,
            self.drone.y / HEIGHT,
            self.drone.z / self.drone.max_height,
            self.drone.angle / 360,
            self.target.x / WIDTH,
            self.target.y / HEIGHT,
            self.target.z / self.drone.max_height
        ])

    def step(self, action):
        # 动作：[前进/后退, 左右移动, 旋转, 高度调整]
        forward_back = action[0] * self.drone.speed
        left_right = action[1] * self.drone.speed
        rotate = action[2] * 5  # 旋转角度
        height_adjust = action[3] * 3  # 高度调整

        # 执行动作
        self.drone.move(forward_back, left_right, height_adjust)
        self.drone.rotate(rotate)

        # 计算距离目标的距离
        distance = math.sqrt(
            (self.drone.x - self.target.x) ** 2 +
            (self.drone.y - self.target.y) ** 2 +
            ((self.drone.z - self.target.z) * 0.5) ** 2  # 高度权重稍低
        )

        # 计算奖励
        reward = 100.0 / (1.0 + distance)  # 距离越近奖励越高

        # 检查是否到达目标
        done = distance < 30 or self.current_step >= self.max_steps

        if done and self.current_step < self.max_steps:
            reward += 100  # 到达目标额外奖励

        self.current_step += 1

        return self.get_state(), reward, done

    def render(self, surface):
        # 绘制环境
        surface.fill(WHITE)

        # 绘制网格背景
        for x in range(0, WIDTH, 50):
            pygame.draw.line(surface, GRAY, (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, 50):
            pygame.draw.line(surface, GRAY, (0, y), (WIDTH, y), 1)

        # 绘制目标和无人机
        self.target.draw(surface)
        self.drone.draw(surface)


# 训练代理
class Agent:
    def __init__(self, state_size=7, action_size=4, lr=0.001, gamma=0.99):
        self.model = DroneController(input_size=state_size, output_size=action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma  # 折扣因子
        self.memory = []  # 存储经验

    def get_action(self, state, epsilon=0.1):
        # epsilon-贪婪策略
        if random.random() < epsilon:
            # 随机动作
            return np.random.uniform(-1, 1, size=4)
        else:
            # 模型预测动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = self.model(state_tensor).numpy()[0]
            return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0.0

        # 随机采样批次
        batch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor([s for s, _, _, _, _ in batch])
        actions = torch.FloatTensor([a for _, a, _, _, _ in batch])
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch])
        next_states = torch.FloatTensor([ns for _, _, _, ns, _ in batch])
        dones = torch.FloatTensor([d for _, _, _, _, d in batch])

        # 计算目标Q值
        with torch.no_grad():
            next_q = self.model(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_q.max(dim=1)[0]

        # 计算当前Q值
        current_q = self.model(states).gather(1, actions.argmax(dim=1).unsqueeze(1)).squeeze(1)

        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# 主函数
def main():
    clock = pygame.time.Clock()
    env = DroneEnv()
    agent = Agent()

    episodes = 1000
    batch_size = 32
    epsilon = 1.0  # 初始探索率
    epsilon_decay = 0.995
    epsilon_min = 0.01

    # 记录训练信息
    total_rewards = []
    avg_rewards = []
    losses = []

    # 训练模式开关
    training_mode = True
    show_info = True

    running = True
    current_episode = 0
    state = env.reset()
    total_reward = 0

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                # 按T切换训练模式
                if event.key == K_t:
                    training_mode = not training_mode
                    print(f"训练模式: {'开启' if training_mode else '关闭'}")
                # 按I切换信息显示
                elif event.key == K_i:
                    show_info = not show_info
                # 按R重置环境
                elif event.key == K_r:
                    state = env.reset()
                    total_reward = 0

            # 手动控制（非训练模式）
            if not training_mode:
                keys = pygame.key.get_pressed()
                action = [0, 0, 0, 0]

                if keys[K_w]:
                    action[0] = 1  # 前进
                elif keys[K_s]:
                    action[0] = -1  # 后退

                if keys[K_a]:
                    action[1] = 1  # 左移
                elif keys[K_d]:
                    action[1] = -1  # 右移

                if keys[K_q]:
                    action[2] = 1  # 左转
                elif keys[K_e]:
                    action[2] = -1  # 右转

                if keys[K_SPACE]:
                    action[3] = 1  # 上升
                elif keys[K_LSHIFT]:
                    action[3] = -1  # 下降

                next_state, reward, done = env.step(action)
                total_reward += reward
                state = next_state

                if done:
                    state = env.reset()
                    total_reward = 0

        # 训练模式
        if training_mode and running:
            # 获取动作
            action = agent.get_action(state, epsilon)

            # 执行动作
            next_state, reward, done = env.step(action)
            total_reward += reward

            # 存储经验
            agent.remember(state, action, reward, next_state, done)

            # 训练模型
            loss = agent.train(batch_size)
            if loss > 0:
                losses.append(loss)

            state = next_state

            #  episode结束
            if done:
                current_episode += 1
                total_rewards.append(total_reward)

                # 计算平均奖励
                window_size = min(10, len(total_rewards))
                avg_reward = sum(total_rewards[-window_size:]) / window_size
                avg_rewards.append(avg_reward)

                # 衰减探索率
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

                # 打印进度
                if current_episode % 10 == 0:
                    print(
                        f"Episode {current_episode}/{episodes}, 奖励: {total_reward:.2f}, 平均奖励: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

                # 重置环境
                state = env.reset()
                total_reward = 0

                # 达到最大训练轮次
                if current_episode >= episodes:
                    print("训练完成!")
                    training_mode = False

        # 渲染环境
        env.render(screen)

        # 显示信息
        if show_info:
            mode_text = font.render(f"模式: {'训练' if training_mode else '手动'}", True, BLACK)
            screen.blit(mode_text, (10, 10))

            if training_mode and current_episode > 0:
                episode_text = font.render(f"轮次: {current_episode}/{episodes}", True, BLACK)
                screen.blit(episode_text, (10, 40))

                reward_text = font.render(f"当前奖励: {total_reward:.1f}", True, BLACK)
                screen.blit(reward_text, (10, 70))

                if len(avg_rewards) > 0:
                    avg_text = font.render(f"平均奖励: {avg_rewards[-1]:.1f}", True, BLACK)
                    screen.blit(avg_text, (10, 100))

                epsilon_text = font.render(f"探索率: {epsilon:.3f}", True, BLACK)
                screen.blit(epsilon_text, (10, 130))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()