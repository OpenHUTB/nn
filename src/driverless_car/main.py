# 基于深度学习的无人机控制与可视化系统
# 核心功能：通过强化学习训练无人机自主飞向目标点，支持训练模式和手动控制模式切换

# ====================== 库导入及说明 ======================
# pygame：用于创建图形化界面、处理用户输入和实时渲染无人机与环境
import pygame
# sys：用于系统级操作，如退出程序
import sys
# numpy：用于数值计算，处理状态向量和动作向量的数组操作
import numpy as np
# torch：PyTorch深度学习框架，用于构建和训练神经网络控制器
import torch
# torch.nn：PyTorch的神经网络模块，用于定义网络层和模型结构
import torch.nn as nn
# torch.optim：PyTorch的优化器模块，用于模型参数更新（如Adam优化器）
import torch.optim as optim
# pygame.locals：包含pygame的常量定义（如按键、事件类型），简化事件处理
from pygame.locals import *
# random：用于生成随机数，如目标点位置、训练时的探索动作、经验回放采样
import random
# math：用于数学计算，如角度转换（弧度/角度）、距离计算、三角函数运算
import math

# ====================== 初始化配置 ======================
# 初始化pygame，必须在使用pygame其他功能前调用
pygame.init()

# 初始化pygame字体系统，用于在界面上显示文字信息（如高度、奖励值）
pygame.font.init()
# 尝试匹配系统中的宋体字体（支持中文显示）
font_path = pygame.font.match_font('simsun')
if not font_path:
    # 如果找不到宋体，使用pygame默认字体，字号36
    font = pygame.font.Font(None, 36)
else:
    # 使用找到的宋体字体，字号24（更适合显示详细信息）
    font = pygame.font.Font(font_path, 24)

# 屏幕尺寸设置（宽度800像素，高度600像素）
WIDTH, HEIGHT = 800, 600
# 创建游戏窗口，指定尺寸
screen = pygame.display.set_mode((WIDTH, HEIGHT))
# 设置窗口标题
pygame.display.set_caption("无人机深度学习控制系统")

# 颜色定义（RGB格式），用于界面元素绘制
WHITE = (255, 255, 255)    # 背景色
BLACK = (0, 0, 0)          # 文字、线条颜色
RED = (255, 0, 0)          # 目标点颜色
GREEN = (0, 255, 0)        # 预留颜色（未使用）
BLUE = (0, 0, 255)         # 预留颜色（未使用）
YELLOW = (255, 255, 0)     # 预留颜色（未使用）
GRAY = (200, 200, 200)     # 网格线颜色


# ====================== 无人机类（Drone） ======================
# 封装无人机的属性（位置、角度、高度等）和行为（移动、旋转、绘制）
class Drone:
    def __init__(self):
        # 初始位置：屏幕中心（x,y坐标）
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.z = 100  # 初始高度（z轴坐标）
        self.angle = 0  # 初始角度（0度，朝向屏幕上方），单位：度
        self.speed = 2  # 平面移动速度（像素/帧）
        self.size = 20  # 基础尺寸（用于绘制）
        self.max_height = 300  # 最大飞行高度限制
        self.min_height = 50   # 最小飞行高度限制

    def move(self, dx, dy, dz=0):
        """
        根据输入的移动量和当前角度，更新无人机的平面位置和高度
        :param dx: 前后移动量（正：前进，负：后退）
        :param dy: 左右移动量（正：左移，负：右移）
        :param dz: 高度调整量（正：上升，负：下降）
        """
        # 将角度转换为弧度（math三角函数要求输入弧度）
        rad = math.radians(self.angle)
        # 基于当前朝向，计算实际x、y方向的位移（极坐标转直角坐标）
        self.x += dx * math.cos(rad) - dy * math.sin(rad)
        self.y += dx * math.sin(rad) + dy * math.cos(rad)

        # 限制无人机在屏幕内（避免超出边界）
        self.x = max(self.size, min(WIDTH - self.size, self.x))
        self.y = max(self.size, min(HEIGHT - self.size, self.y))

        # 限制高度在最小和最大高度之间
        self.z = max(self.min_height, min(self.max_height, self.z + dz))

    def rotate(self, delta_angle):
        """
        调整无人机的朝向角度
        :param delta_angle: 角度变化量（正：左转，负：右转）
        """
        # 取模360，确保角度始终在[0, 360)范围内
        self.angle = (self.angle + delta_angle) % 360

    def draw(self, surface):
        """
        在指定表面（屏幕）上绘制无人机
        :param surface: 绘制目标表面（此处为screen窗口）
        """
        # 根据高度调整绘制尺寸（高度越高，视觉上越小）
        size_factor = self.z / self.max_height
        draw_size = int(self.size * (0.5 + size_factor * 0.5))

        # 无人机中心点坐标（整数化，避免绘制模糊）
        center = (int(self.x), int(self.y))

        # 绘制无人机机身（圆形），颜色深浅随高度变化（高度越高越亮）
        pygame.draw.circle(surface,
                           (int(50 + size_factor * 205),  # R通道
                            int(50 + size_factor * 105),  # G通道
                            int(50 + size_factor * 205)),  # B通道
                           center, draw_size)

        # 绘制无人机旋翼（十字形，沿当前朝向）
        rad = math.radians(self.angle)
        rotor_length = draw_size * 0.8  # 旋翼长度为机身尺寸的0.8倍

        # 前旋翼（沿当前朝向）
        front_x = self.x + math.cos(rad) * rotor_length
        front_y = self.y + math.sin(rad) * rotor_length
        pygame.draw.line(surface, BLACK, center, (front_x, front_y), 3)

        # 后旋翼（与前旋翼相反方向）
        back_x = self.x - math.cos(rad) * rotor_length
        back_y = self.y - math.sin(rad) * rotor_length
        pygame.draw.line(surface, BLACK, center, (back_x, back_y), 3)

        # 左旋翼（垂直于当前朝向左侧）
        left_x = self.x - math.sin(rad) * rotor_length
        left_y = self.y + math.cos(rad) * rotor_length
        pygame.draw.line(surface, BLACK, center, (left_x, left_y), 3)

        # 右旋翼（垂直于当前朝向右侧）
        right_x = self.x + math.sin(rad) * rotor_length
        right_y = self.y - math.cos(rad) * rotor_length
        pygame.draw.line(surface, BLACK, center, (right_x, right_y), 3)

        # 显示无人机当前高度（文字绘制在机身右侧）
        height_text = font.render(f"高度: {int(self.z)}", True, BLACK)
        surface.blit(height_text, (self.x + draw_size, self.y - draw_size))


# ====================== 目标点类（Target） ======================
# 封装目标点的属性（位置、高度）和行为（绘制、重置位置）
class Target:
    def __init__(self):
        """初始化目标点位置（随机生成，避免靠近屏幕边界）"""
        self.x = random.randint(50, WIDTH - 50)  # x坐标：50~750
        self.y = random.randint(50, HEIGHT - 50) # y坐标：50~550
        self.z = random.randint(80, 250)         # 高度：80~250（在无人机高度限制内）
        self.radius = 15  # 目标点绘制半径

    def draw(self, surface):
        """在指定表面绘制目标点"""
        # 绘制目标点外圈（红色空心圆）
        pygame.draw.circle(surface, RED, (self.x, self.y), self.radius, 2)
        # 绘制目标点中心（红色实心小圆）
        pygame.draw.circle(surface, RED, (self.x, self.y), 3)

        # 显示目标点高度（文字绘制在目标点右侧）
        z_text = font.render(f"Z: {self.z}", True, RED)
        surface.blit(z_text, (self.x + self.radius, self.y - self.radius))

    def reset(self):
        """重置目标点位置（训练时episode结束后调用）"""
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(50, HEIGHT - 50)
        self.z = random.randint(80, 250)


# ====================== 深度学习控制器模型（DroneController） ======================
# 基于全连接神经网络的无人机控制器，输入状态向量，输出动作向量
class DroneController(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, output_size=4):
        """
        初始化神经网络结构
        :param input_size: 输入维度（状态向量长度，实际使用时为7，此处默认值预留）
        :param hidden_size: 隐藏层神经元数量（32个，平衡性能和复杂度）
        :param output_size: 输出维度（动作向量长度：前后、左右、旋转、高度调整）
        """
        super(DroneController, self).__init__()
        # 第一层全连接层：输入层 -> 隐藏层1
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 第二层全连接层：隐藏层1 -> 隐藏层2（加深网络，提升表达能力）
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 第三层全连接层：隐藏层2 -> 输出层
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()  # 激活函数（引入非线性，提升网络拟合能力）
        self.tanh = nn.Tanh()  # 输出激活函数（将动作值限制在[-1,1]范围内）

    def forward(self, x):
        """
        前向传播过程（输入状态向量，输出动作向量）
        :param x: 输入状态向量（tensor类型）
        :return: 输出动作向量（tensor类型，值范围[-1,1]）
        """
        x = self.relu(self.fc1(x))  # 输入层 -> 隐藏层1，ReLU激活
        x = self.relu(self.fc2(x))  # 隐藏层1 -> 隐藏层2，ReLU激活
        x = self.tanh(self.fc3(x))  # 隐藏层2 -> 输出层，Tanh激活
        return x


# ====================== 强化学习环境（DroneEnv） ======================
# 封装强化学习的环境逻辑：状态获取、动作执行、奖励计算、环境渲染
class DroneEnv:
    def __init__(self):
        self.drone = Drone()  # 初始化无人机实例
        self.target = Target()  # 初始化目标点实例
        self.max_steps = 500  # 每个episode的最大步数（避免无限循环）
        self.current_step = 0  # 当前episode的已执行步数
        self.reset()  # 重置环境到初始状态

    def reset(self):
        """重置环境（新episode开始时调用）"""
        self.drone = Drone()  # 重置无人机位置和状态
        self.target.reset()  # 重置目标点位置
        self.current_step = 0  # 重置步数计数器
        return self.get_state()  # 返回初始状态向量

    def get_state(self):
        """
        获取当前环境状态向量（归一化到[0,1]范围，提升模型训练稳定性）
        状态向量组成：[无人机x坐标, 无人机y坐标, 无人机高度, 无人机角度, 目标x坐标, 目标y坐标, 目标高度]
        """
        return np.array([
            self.drone.x / WIDTH,  # 无人机x归一化（相对于屏幕宽度）
            self.drone.y / HEIGHT,  # 无人机y归一化（相对于屏幕高度）
            self.drone.z / self.drone.max_height,  # 无人机高度归一化（相对于最大高度）
            self.drone.angle / 360,  # 无人机角度归一化（相对于360度）
            self.target.x / WIDTH,  # 目标x归一化
            self.target.y / HEIGHT,  # 目标y归一化
            self.target.z / self.drone.max_height  # 目标高度归一化
        ])

    def step(self, action):
        """
        执行动作，更新环境状态，返回下一个状态、奖励和结束标志
        :param action: 动作向量（长度4，值范围[-1,1]），对应：[前后, 左右, 旋转, 高度调整]
        :return: next_state（下一个状态向量）、reward（奖励值）、done（是否结束）
        """
        # 将动作向量映射到实际操作量（缩放因子控制动作强度）
        forward_back = action[0] * self.drone.speed  # 前后移动
        left_right = action[1] * self.drone.speed    # 左右移动
        rotate = action[2] * 5  # 旋转角度（缩放5倍，使旋转更明显）
        height_adjust = action[3] * 3  # 高度调整（缩放3倍，控制上升/下降速度）

        # 执行动作（移动+旋转）
        self.drone.move(forward_back, left_right, height_adjust)
        self.drone.rotate(rotate)

        # 计算无人机与目标点的三维距离（高度方向权重减半，因为平面视觉更重要）
        distance = math.sqrt(
            (self.drone.x - self.target.x) ** 2 +
            (self.drone.y - self.target.y) ** 2 +
            ((self.drone.z - self.target.z) * 0.5) ** 2
        )

        # 计算奖励：距离越近，奖励越高（100/(1+距离)确保奖励为正，且随距离递减）
        reward = 100.0 / (1.0 + distance)

        # 判断episode是否结束：到达目标（距离<30）或步数耗尽（超过max_steps）
        done = distance < 30 or self.current_step >= self.max_steps

        # 到达目标额外奖励（鼓励无人机快速到达目标）
        if done and self.current_step < self.max_steps:
            reward += 100

        # 更新步数计数器
        self.current_step += 1

        return self.get_state(), reward, done

    def render(self, surface):
        """渲染环境（绘制背景、网格、目标点、无人机）"""
        surface.fill(WHITE)  # 填充背景为白色

        # 绘制网格背景（50像素间隔，提升视觉定位效果）
        for x in range(0, WIDTH, 50):
            pygame.draw.line(surface, GRAY, (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, 50):
            pygame.draw.line(surface, GRAY, (0, y), (WIDTH, y), 1)

        # 绘制目标点和无人机（目标点先画，避免被无人机遮挡）
        self.target.draw(surface)
        self.drone.draw(surface)


# ====================== 训练代理（Agent） ======================
# 封装强化学习代理的逻辑：动作选择、经验存储、模型训练
class Agent:
    def __init__(self, state_size=7, action_size=4, lr=0.001, gamma=0.99):
        """
        初始化代理
        :param state_size: 状态向量长度（7维）
        :param action_size: 动作向量长度（4维）
        :param lr: 学习率（控制模型参数更新速度）
        :param gamma: 折扣因子（控制未来奖励的权重，0.99表示重视长期奖励）
        """
        self.model = DroneController(input_size=state_size, output_size=action_size)  # 深度学习控制器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Adam优化器（自适应学习率，训练稳定）
        self.gamma = gamma  # 折扣因子
        self.memory = []  # 经验回放缓冲区（存储(state, action, reward, next_state, done)）

    def get_action(self, state, epsilon=0.1):
        """
        基于epsilon-贪婪策略选择动作（平衡探索和利用）
        :param state: 当前状态向量
        :param epsilon: 探索概率（0~1，越大越倾向于随机动作）
        :return: 动作向量（长度4，值范围[-1,1]）
        """
        if random.random() < epsilon:
            # 探索：随机生成动作（均匀分布在[-1,1]）
            return np.random.uniform(-1, 1, size=4)
        else:
            # 利用：模型预测最优动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 转换为tensor并增加batch维度
            with torch.no_grad():  # 禁用梯度计算（仅预测，不训练）
                action = self.model(state_tensor).numpy()[0]  # tensor转numpy数组
            return action

    def remember(self, state, action, reward, next_state, done):
        """将经验存储到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        """
        从经验回放缓冲区采样训练模型（DQN算法核心）
        :param batch_size: 批量采样大小（32，平衡训练效率和稳定性）
        :return: 训练损失值（用于监控训练效果）
        """
        # 缓冲区经验不足时，不训练
        if len(self.memory) < batch_size:
            return 0.0

        # 从缓冲区随机采样batch_size条经验（打破时序相关性，提升训练稳定性）
        batch = random.sample(self.memory, batch_size)

        # 提取批量数据并转换为tensor（便于PyTorch计算）
        states = torch.FloatTensor([s for s, _, _, _, _ in batch])
        actions = torch.FloatTensor([a for _, a, _, _, _ in batch])
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch])
        next_states = torch.FloatTensor([ns for _, _, _, ns, _ in batch])
        dones = torch.FloatTensor([d for _, _, _, _, d in batch])

        # 计算目标Q值（基于下一状态的最大Q值，Bellman方程）
        with torch.no_grad():  # 目标Q值不参与梯度计算
            next_q = self.model(next_states)  # 下一状态的Q值预测
            # 目标Q值 = 即时奖励 + 折扣因子 * 下一状态最大Q值（done时，下一状态无奖励，故乘(1-done)）
            target_q = rewards + (1 - dones) * self.gamma * next_q.max(dim=1)[0]

        # 计算当前Q值（模型对当前状态-动作的Q值预测）
        current_q = self.model(states).gather(1, actions.argmax(dim=1).unsqueeze(1)).squeeze(1)

        # 计算损失（均方误差MSE，衡量当前Q值与目标Q值的差距）
        loss = nn.MSELoss()(current_q, target_q)

        # 反向传播更新模型参数
        self.optimizer.zero_grad()  # 清空梯度（避免梯度累积）
        loss.backward()  # 计算梯度
        self.optimizer.step()  # 更新参数

        return loss.item()  # 返回损失值（标量）


# ====================== 主函数（程序入口） ======================
def main():
    clock = pygame.time.Clock()  # 时钟对象，控制游戏帧率
    env = DroneEnv()  # 初始化环境
    agent = Agent()  # 初始化训练代理

    # 训练参数配置
    episodes = 1000  # 总训练轮次（episode）
    batch_size = 32  # 训练批量大小
    epsilon = 1.0  # 初始探索率（1.0表示完全探索）
    epsilon_decay = 0.995  # 探索率衰减系数（每轮衰减5%）
    epsilon_min = 0.01  # 最小探索率（避免完全停止探索）

    # 训练信息记录（用于监控训练进度）
    total_rewards = []  # 每轮总奖励
    avg_rewards = []    # 最近10轮平均奖励
    losses = []         # 训练损失值

    # 模式控制开关
    training_mode = True  # 是否开启训练模式（True：训练，False：手动控制）
    show_info = True      # 是否显示训练/状态信息

    running = True  # 程序运行标志
    current_episode = 0  # 当前训练轮次
    state = env.reset()  # 初始化环境，获取初始状态
    total_reward = 0  # 当前轮次总奖励

    # 主循环（程序核心逻辑）
    while running:
        # 事件处理（键盘输入、窗口关闭）
        for event in pygame.event.get():
            if event.type == QUIT:
                # 关闭窗口事件：退出程序
                running = False
            elif event.type == KEYDOWN:
                # 按键事件处理
                if event.key == K_t:
                    # 按T键：切换训练模式/手动控制模式
                    training_mode = not training_mode
                    print(f"训练模式: {'开启' if training_mode else '关闭'}")
                elif event.key == K_i:
                    # 按I键：切换信息显示/隐藏
                    show_info = not show_info
                elif event.key == K_r:
                    # 按R键：重置环境
                    state = env.reset()
                    total_reward = 0

            # 手动控制逻辑（训练模式关闭时生效）
            if not training_mode:
                keys = pygame.key.get_pressed()  # 获取当前按下的所有按键
                action = [0, 0, 0, 0]  # 初始化动作向量（默认无动作）

                # W/S：前后移动
                if keys[K_w]:
                    action[0] = 1  # 前进
                elif keys[K_s]:
                    action[0] = -1  # 后退

                # A/D：左右移动
                if keys[K_a]:
                    action[1] = 1  # 左移
                elif keys[K_d]:
                    action[1] = -1  # 右移

                # Q/E：左右旋转
                if keys[K_q]:
                    action[2] = 1  # 左转
                elif keys[K_e]:
                    action[2] = -1  # 右转

                # 空格/左Shift：上升/下降
                if keys[K_SPACE]:
                    action[3] = 1  # 上升
                elif keys[K_LSHIFT]:
                    action[3] = -1  # 下降

                # 执行动作，更新状态和奖励
                next_state, reward, done = env.step(action)
                total_reward += reward
                state = next_state

                # 任务结束时重置环境
                if done:
                    state = env.reset()
                    total_reward = 0

        # 训练模式逻辑（训练模式开启时生效）
        if training_mode and running:
            # 获取动作（基于epsilon-贪婪策略）
            action = agent.get_action(state, epsilon)

            # 执行动作，获取下一个状态、奖励和结束标志
            next_state, reward, done = env.step(action)
            total_reward += reward

            # 存储经验到回放缓冲区
            agent.remember(state, action, reward, next_state, done)

            # 训练模型，获取损失值
            loss = agent.train(batch_size)
            if loss > 0:
                losses.append(loss)

            # 更新当前状态
            state = next_state

            # 当前episode结束处理
            if done:
                current_episode += 1
                total_rewards.append(total_reward)  # 记录当前轮次总奖励

                # 计算最近10轮平均奖励（平滑奖励曲线，便于观察训练趋势）
                window_size = min(10, len(total_rewards))
                avg_reward = sum(total_rewards[-window_size:]) / window_size
                avg_rewards.append(avg_reward)

                # 衰减探索率（逐渐从探索转向利用）
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

                # 每10轮打印一次训练进度
                if current_episode % 10 == 0:
                    print(
                        f"Episode {current_episode}/{episodes}, 奖励: {total_reward:.2f}, 平均奖励: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

                # 重置环境，准备下一轮训练
                state = env.reset()
                total_reward = 0

                # 达到总训练轮次，关闭训练模式
                if current_episode >= episodes:
                    print("训练完成!")
                    training_mode = False

        # 渲染环境（实时绘制界面）
        env.render(screen)

        # 显示信息（训练模式下显示轮次、奖励、探索率等；手动模式下显示当前模式）
        if show_info:
            mode_text = font.render(f"模式: {'训练' if training_mode else '手动'}", True, BLACK)
            screen.blit(mode_text, (10, 10))

            if training_mode and current_episode > 0:
                # 训练模式信息显示
                episode_text = font.render(f"轮次: {current_episode}/{episodes}", True, BLACK)
                screen.blit(episode_text, (10, 40))

                reward_text = font.render(f"当前奖励: {total_reward:.1f}", True, BLACK)
                screen.blit(reward_text, (10, 70))

                if len(avg_rewards) > 0:
                    avg_text = font.render(f"平均奖励: {avg_rewards[-1]:.1f}", True, BLACK)
                    screen.blit(avg_text, (10, 100))

                epsilon_text = font.render(f"探索率: {epsilon:.3f}", True, BLACK)
                screen.blit(epsilon_text, (10, 130))

        # 更新屏幕显示（将绘制的内容刷新到窗口）
        pygame.display.flip()
        # 控制帧率为60帧/秒（确保界面流畅，训练速度稳定）
        clock.tick(60)

    # 退出程序（释放pygame资源）
    pygame.quit()
    sys.exit()


# 程序入口：当脚本直接运行时，执行main函数
if __name__ == "__main__":
    main()