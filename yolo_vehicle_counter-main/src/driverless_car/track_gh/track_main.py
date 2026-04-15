import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import random

# 自定义网格环境
class DroneGridEnv(gym.Env):
    """
    无人机网格路径规划环境
    网格大小：grid_size x grid_size
    状态：无人机当前坐标 (x, y)
    动作：0(上), 1(下), 2(左), 3(右)
    奖励：靠近终点+1，到达终点+100，撞障碍物/边界-50，每步-1（鼓励最短路径）
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=10):
        super(DroneGridEnv, self).__init__()
        self.grid_size = grid_size  # 网格大小
        # 定义动作空间：上下左右4个动作
        self.action_space = spaces.Discrete(4)
        # 定义状态空间：坐标(x, y)，范围0到grid_size-1
        self.observation_space = spaces.Tuple((
            spaces.Discrete(grid_size),
            spaces.Discrete(grid_size)
        ))

        # 初始化起点、终点、障碍物
        self.start = (0, 0)
        self.end = (grid_size - 1, grid_size - 1)
        # 随机生成障碍物（数量为网格的10%）
        self.obstacles = set()
        obstacle_num = int(grid_size * grid_size * 0.1)
        while len(self.obstacles) < obstacle_num:
            x = random.randint(0, grid_size - 1)
            y = random.randint(0, grid_size - 1)
            # 障碍物不能在起点和终点
            if (x, y) != self.start and (x, y) != self.end:
                self.obstacles.add((x, y))

        # 无人机当前位置
        self.current_pos = None
        # 初始化绘图对象（避免重复创建）
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        # 初始化颜色条（只创建一次）
        self.cbar = None
        # 定义颜色映射（更美观的配色）
        self.cmap = plt.cm.Spectral_r  # 替换为更协调的配色
        # 定义标签对应的数值（用于颜色映射）
        self.label_values = {
            '空位置': 0,
            '起点': 1,
            '终点': 2,
            '障碍物': 3,
            '路径': 4
        }
        self.value_labels = {v: k for k, v in self.label_values.items()}

    def reset(self, seed=None, options=None):
        """重置环境，返回初始状态（适配gymnasium的reset接口）"""
        super().reset(seed=seed)
        self.current_pos = self.start
        return self.current_pos, {}  # gymnasium要求返回(state, info)

    def step(self, action):
        """执行动作，返回新状态、奖励、是否结束、是否截断、额外信息（适配gymnasium）"""
        x, y = self.current_pos
        # 根据动作更新位置
        if action == 0:  # 上
            new_x, new_y = x - 1, y
        elif action == 1:  # 下
            new_x, new_y = x + 1, y
        elif action == 2:  # 左
            new_x, new_y = x, y - 1
        elif action == 3:  # 右
            new_x, new_y = x, y + 1
        else:
            new_x, new_y = x, y  # 无效动作，位置不变

        # 检查是否越界
        if new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size:
            reward = -50  # 撞边界扣分
            terminated = False
            truncated = False
            new_pos = (x, y)  # 位置不变
        # 检查是否撞障碍物
        elif (new_x, new_y) in self.obstacles:
            reward = -50  # 撞障碍物扣分
            terminated = False
            truncated = False
            new_pos = (x, y)  # 位置不变
        # 到达终点
        elif (new_x, new_y) == self.end:
            reward = 100  # 到达终点大加分
            terminated = True
            truncated = False
            new_pos = (new_x, new_y)
        # 普通移动
        else:
            new_pos = (new_x, new_y)
            # 计算与终点的距离（曼哈顿距离），靠近终点加分
            old_dist = abs(x - self.end[0]) + abs(y - self.end[1])
            new_dist = abs(new_x - self.end[0]) + abs(new_y - self.end[1])
            reward = 1 if new_dist < old_dist else -1  # 靠近+1，远离-1
            reward -= 1  # 每步扣1，鼓励最短路径
            terminated = False
            truncated = False

        self.current_pos = new_pos
        return new_pos, reward, terminated, truncated, {}  # gymnasium的step返回格式

    def render(self, path=None):
        """可视化网格和路径（优化后的版本）"""
        self.ax.clear()  # 清空当前轴

        # 创建网格矩阵
        grid = np.zeros((self.grid_size, self.grid_size))
        # 标记起点、终点、障碍物
        grid[self.start] = self.label_values['起点']
        grid[self.end] = self.label_values['终点']
        for (x, y) in self.obstacles:
            grid[x, y] = self.label_values['障碍物']
        # 标记路径
        if path:
            for (x, y) in path:
                if (x, y) not in [self.start, self.end]:  # 不覆盖起点和终点
                    grid[x, y] = self.label_values['路径']

        # 绘制网格（使用pcolormesh，支持网格线）
        im = self.ax.pcolormesh(
            grid,
            cmap=self.cmap,
            vmin=0,
            vmax=max(self.label_values.values()),
            edgecolors='white',  # 网格线颜色
            linewidths=2  # 网格线宽度
        )

        # 只创建一次颜色条，后续更新即可
        if self.cbar is None:
            self.cbar = plt.colorbar(im, ax=self.ax, shrink=0.8)
            # 设置颜色条标签
            self.cbar.set_ticks(list(self.label_values.values()))
            self.cbar.set_ticklabels(list(self.label_values.keys()))
        else:
            self.cbar.update_normal(im)

        # 添加坐标标注（每个单元格的中心）
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # 显示坐标 (i, j)
                self.ax.text(
                    j + 0.5,  # x坐标（列）
                    i + 0.5,  # y坐标（行）
                    f'({i}, {j})',
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=8,
                    fontweight='bold'
                )

        # 绘制路径箭头（展示移动方向）
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                # 绘制箭头（从当前点到下一个点）
                self.ax.arrow(
                    y1 + 0.5,  # 起点x（列）
                    x1 + 0.5,  # 起点y（行）
                    y2 - y1,    # x方向偏移
                    x2 - x1,    # y方向偏移
                    head_width=0.1,  # 箭头宽度
                    head_length=0.1, # 箭头长度
                    fc='red',   # 箭头颜色
                    ec='red',
                    alpha=0.7
                )

        # 设置坐标轴
        self.ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
        self.ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        # 反转y轴，让(0,0)在左上角（符合网格坐标习惯）
        self.ax.invert_yaxis()
        # 设置标题和标签
        self.ax.set_title(
            '无人机网格路径规划',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        self.ax.set_xlabel('列坐标 (y)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('行坐标 (x)', fontsize=12, fontweight='bold')
        # 调整刻度标签
        self.ax.tick_params(axis='both', which='major', labelsize=10)

        # 强制刷新画布
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(block=False)
        plt.pause(0.1)  # 短暂暂停，便于观察

    def close(self):
        """关闭绘图对象"""
        plt.close(self.fig)

# Q-Learning智能体
class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        """
        初始化Q-Learning智能体
        :param state_space: 状态空间
        :param action_space: 动作空间（Discrete类型）
        :param learning_rate: 学习率α
        :param gamma: 折扣因子γ
        :param epsilon: 探索率ε（ε-贪心策略）
        """
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space = action_space
        self.n_actions = action_space.n

        # 初始化Q表：字典，键为状态(x, y)，值为动作对应的Q值（列表）
        self.q_table = {}

    def get_q_value(self, state):
        """获取状态对应的Q值，若不存在则初始化"""
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(self.n_actions)]
        return self.q_table[state]

    def choose_action(self, state):
        """ε-贪心策略选择动作"""
        if random.uniform(0, 1) < self.epsilon:
            # 探索：随机选择动作
            action = self.action_space.sample()
        else:
            # 利用：选择Q值最大的动作
            q_values = self.get_q_value(state)
            action = np.argmax(q_values)
        return action

    def learn(self, state, action, reward, next_state, done):
        """更新Q表"""
        current_q = self.get_q_value(state)[action]
        if done:
            target_q = reward  # 到达终点，无后续奖励
        else:
            next_q_values = self.get_q_value(next_state)
            target_q = reward + self.gamma * np.max(next_q_values)  # Q-Learning的目标Q值

        # 更新Q值
        self.get_q_value(state)[action] += self.lr * (target_q - current_q)

# 主程序：训练并测试
if __name__ == "__main__":
    # 解决matplotlib后端问题（针对PyCharm）
    plt.switch_backend('TkAgg')  # 切换到TkAgg后端，避免InterAgg的问题

    # 1. 初始化环境和智能体
    grid_size = 10
    env = DroneGridEnv(grid_size=grid_size)
    agent = QLearningAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=0.1,
        gamma=0.9,
        epsilon=0.1
    )

    # 2. 训练智能体
    episodes = 1000  # 训练轮数
    rewards_history = []  # 记录每轮的总奖励

    for episode in range(episodes):
        state, _ = env.reset()  # 适配gymnasium的reset返回
        total_reward = 0
        done = False

        while not done:
            # 选择动作
            action = agent.choose_action(state)
            # 执行动作（适配gymnasium的step返回）
            next_state, reward, terminated, truncated, _ = env.step(action)
            # 只要terminated或truncated，就视为结束
            done = terminated or truncated
            # 学习更新Q表
            agent.learn(state, action, reward, next_state, done)
            # 更新状态和总奖励
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        # 每100轮打印一次训练情况
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")

    # 绘制奖励变化曲线（优化样式）
    plt.figure(figsize=(12, 6))
    # 绘制奖励曲线，添加平滑效果
    plt.plot(rewards_history, color='blue', alpha=0.5, label='每轮奖励')
    # 计算移动平均（窗口大小为20），展示趋势
    window_size = 20
    if len(rewards_history) >= window_size:
        moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards_history)), moving_avg, color='red', linewidth=2, label=f'{window_size}轮移动平均')
    plt.xlabel('训练轮数 (Episode)', fontsize=12, fontweight='bold')
    plt.ylabel('总奖励 (Total Reward)', fontsize=12, fontweight='bold')
    plt.title('无人机路径规划训练奖励变化', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)  # 阻塞显示，看完后关闭再继续

    # 3. 测试智能体（展示路径）
    state, _ = env.reset()
    path = [state]
    done = False

    while not done:
        # 测试时仅利用，不探索（ε=0）
        q_values = agent.get_q_value(state)
        action = np.argmax(q_values)
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        path.append(next_state)
        state = next_state
        # 可视化每一步路径
        env.render(path=path)

    # 最终展示路径（阻塞显示，保持窗口）
    env.render(path=path)
    plt.show(block=True)
    # 关闭环境
    env.close()