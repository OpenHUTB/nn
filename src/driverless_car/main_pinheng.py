# 导入必要的库
import numpy as np  # 数值计算库，用于矩阵和数组操作
import matplotlib.pyplot as plt  # 绘图库，用于静态和动态可视化
from matplotlib.animation import FuncAnimation  # 动态动画绘制工具
import torch  # PyTorch深度学习框架核心库
import torch.nn as nn  # 神经网络层定义模块
import torch.optim as optim  # 优化器模块，用于模型训练
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器，用于批处理训练
import random  # 随机数生成器，用于初始化无人机状态
from mpl_toolkits.mplot3d import Axes3D  # 3D绘图工具包

# 设置matplotlib中文字体和负号显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]  # 支持中文显示的字体列表
plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题


# ===================== 无人机物理模型类 =====================
# 功能：模拟无人机的动力学和运动学特性，实现状态重置和一步动作执行
class DroneModel:
    def __init__(self):
        # 无人机物理参数配置
        self.mass = 1.0  # 无人机质量，单位kg
        self.inertia = np.array([0.1, 0.1, 0.2])  # 三轴转动惯量 [滚转, 俯仰, 偏航]
        self.gravity = 9.81  # 重力加速度，单位m/s²
        self.dt = 0.01  # 仿真时间步长，单位s，越小仿真越精确但速度越慢
        self.reset()  # 初始化时重置无人机状态

    def reset(self):
        """
        重置无人机状态到初始值
        返回值：初始化后的状态数组
        状态数组格式(共12维):
        [x, y, z, 滚转角, 俯仰角, 偏航角, 线速度vx, vy, vz, 角速度wx, wy, wz]
        """
        # 位置初始化：x=0, y=0, z=1m（初始高度1米）
        # 姿态初始化：滚转和俯仰角在[-0.025, 0.025]rad小范围随机扰动，偏航角为0
        # 线速度和角速度初始化为0
        self.state = np.array([
            0.0, 0.0, 1.0,  # 位置 [x, y, z]
            0.05 * random.random() - 0.025,  # 滚转角 (roll)
            0.05 * random.random() - 0.025,  # 俯仰角 (pitch)
            0.0,  # 偏航角 (yaw)
            0.0, 0.0, 0.0,  # 线速度 [vx, vy, vz]
            0.0, 0.0, 0.0   # 角速度 [wx, wy, wz]
        ])
        return self.state

    def step(self, action):
        """
        执行一步动作，更新无人机状态
        参数:
            action: 长度为4的数组，代表四个螺旋桨的推力大小
        返回值:
            state: 更新后的无人机状态
            reward: 本次动作的奖励值（用于强化学习训练）
            done: 布尔值，是否达到终止条件（无人机失控）
        """
        f1, f2, f3, f4 = action  # 解包四个螺旋桨推力

        # 1. 计算总推力和三轴力矩
        thrust = (f1 + f2 + f3 + f4)  # 总推力（垂直机体坐标系z轴）
        # 滚转力矩：由对角螺旋桨推力差产生，臂长系数0.1
        torque_x = (f2 + f4 - f1 - f3) * 0.1
        # 俯仰力矩：由前后螺旋桨推力差产生，臂长系数0.1
        torque_y = (f1 + f2 - f3 - f4) * 0.1
        # 偏航力矩：由相邻螺旋桨推力差产生，系数0.05
        torque_z = (f2 + f3 - f1 - f4) * 0.05

        # 2. 解包当前姿态角
        roll, pitch, yaw = self.state[3:6]

        # 3. 计算线加速度（机体坐标系转世界坐标系）
        # 基于欧拉角的坐标变换，将机体坐标系的推力转换为世界坐标系的加速度
        ax = (np.cos(yaw) * np.sin(pitch) + np.sin(yaw) * np.sin(roll) * np.cos(pitch)) * thrust / self.mass
        ay = (np.sin(yaw) * np.sin(pitch) - np.cos(yaw) * np.sin(roll) * np.cos(pitch)) * thrust / self.mass
        az = self.gravity - (np.cos(roll) * np.cos(pitch)) * thrust / self.mass  # 重力补偿

        # 4. 计算角加速度（力矩/转动惯量）
        angular_acc = np.array([
            torque_x / self.inertia[0],
            torque_y / self.inertia[1],
            torque_z / self.inertia[2]
        ])

        # 5. 更新状态（欧拉积分法：状态 = 原状态 + 变化率 * 时间步长）
        self.state[6:9] += np.array([ax, ay, az]) * self.dt  # 更新线速度
        self.state[0:3] += self.state[6:9] * self.dt  # 更新位置
        self.state[9:12] += angular_acc * self.dt  # 更新角速度
        self.state[3:6] += self.state[9:12] * self.dt  # 更新姿态角

        # 6. 姿态角范围限制：将角度约束在[-π, π]范围内
        self.state[3:6] = np.mod(self.state[3:6] + np.pi, 2 * np.pi) - np.pi

        # 7. 计算奖励值：惩罚姿态倾斜、角速度过大、位置偏移和高度偏离
        # 奖励公式设计原则：越接近目标状态（悬停），奖励值越高
        reward = - (
            np.sum(np.square(self.state[3:5])) +  # 惩罚滚转/俯仰角倾斜（权重1）
            0.1 * np.sum(np.square(self.state[9:11])) +  # 惩罚角速度（权重0.1）
            0.01 * np.sum(np.square(self.state[0:2])) +  # 惩罚xy平面位置偏移（权重0.01）
            0.1 * np.square(self.state[2] - 1.0)  # 惩罚高度偏离1m目标值（权重0.1）
        )

        # 8. 判断终止条件：无人机失控
        done = (
            abs(self.state[3]) > np.pi / 4 or  # 滚转角超过45度
            abs(self.state[4]) > np.pi / 4 or  # 俯仰角超过45度
            self.state[2] < 0.2  # 高度低于0.2米
        )

        return self.state, reward, done


# ===================== 深度学习控制器模型 =====================
# 功能：基于全连接神经网络的无人机控制器，输入状态输出四个螺旋桨推力
class DroneController(nn.Module):
    def __init__(self, state_dim=12, action_dim=4):
        """
        初始化神经网络结构
        参数:
            state_dim: 输入状态维度，默认12维
            action_dim: 输出动作维度，默认4维（四个螺旋桨推力）
        """
        super(DroneController, self).__init__()
        # 定义三层全连接神经网络
        self.fc1 = nn.Linear(state_dim, 64)  # 输入层→隐藏层1：12→64
        self.fc2 = nn.Linear(64, 64)  # 隐藏层1→隐藏层2：64→64
        self.fc3 = nn.Linear(64, action_dim)  # 隐藏层2→输出层：64→4
        self.activation = nn.Tanh()  # 激活函数：输出范围[-1, 1]

    def forward(self, x):
        """
        前向传播：输入状态计算输出动作
        参数:
            x: 输入状态张量，shape=(batch_size, state_dim)
        返回值:
            输出动作张量，shape=(batch_size, action_dim)，推力范围[-15, 15]
        """
        x = self.activation(self.fc1(x))  # 第一层+激活
        x = self.activation(self.fc2(x))  # 第二层+激活
        x = self.activation(self.fc3(x))  # 第三层+激活
        return x * 15.0  # 缩放输出范围到[-15, 15]（螺旋桨最大推力15N）


# ===================== 无人机数据集类 =====================
# 功能：生成训练数据，使用规则控制器生成示范动作，用于监督学习训练
class DroneDataset(Dataset):
    def __init__(self, size=10000):
        """
        初始化数据集
        参数:
            size: 数据集大小，默认10000条样本
        """
        self.size = size
        self.drone = DroneModel()  # 实例化无人机模型
        self.data = []  # 存储样本数据：[(state, action), ...]

        # 生成size条训练样本
        for _ in range(size):
            state = self.drone.reset()  # 随机初始化无人机状态
            roll, pitch = state[3], state[4]  # 获取当前姿态角

            # 规则控制器生成目标动作：基于姿态误差的PD控制（简化版）
            # 控制逻辑：姿态角倾斜时，调整对应螺旋桨推力以修正姿态
            f1 = 5.0 + 10.0 * pitch + 10.0 * roll
            f2 = 5.0 + 10.0 * pitch - 10.0 * roll
            f3 = 5.0 - 10.0 * pitch - 10.0 * roll
            f4 = 5.0 - 10.0 * pitch + 10.0 * roll

            # 推力范围限制：避免推力为负数或过大
            action = np.clip([f1, f2, f3, f4], 0, 15)
            self.data.append((state, action))  # 保存状态-动作对

    def __len__(self):
        """返回数据集大小（必须实现）"""
        return self.size

    def __getitem__(self, idx):
        """
        获取单个样本（必须实现）
        参数:
            idx: 样本索引
        返回值:
            状态张量和动作张量（float32类型，符合PyTorch要求）
        """
        state, action = self.data[idx]
        return torch.FloatTensor(state), torch.FloatTensor(action)


# ===================== 模型训练函数 =====================
def train_model(epochs=50):
    """
    训练深度学习控制器模型
    参数:
        epochs: 训练轮数，默认50轮
    返回值:
        训练完成的模型
    """
    # 1. 创建数据集和数据加载器
    dataset = DroneDataset()  # 生成训练数据
    # 数据加载器：批大小64，打乱数据顺序
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. 初始化模型、损失函数和优化器
    model = DroneController()  # 实例化控制器模型
    criterion = nn.MSELoss()  # 损失函数：均方误差（监督学习）
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器：Adam，学习率0.001

    # 3. 开始训练循环
    for epoch in range(epochs):
        running_loss = 0.0  # 累计每轮的损失值
        for states, actions in dataloader:  # 遍历每个批次
            optimizer.zero_grad()  # 清零梯度（避免梯度累积）

            outputs = model(states)  # 模型预测动作
            loss = criterion(outputs, actions)  # 计算预测值与真实值的损失

            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()  # 累加损失值

        # 每10轮打印一次训练进度
        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(dataloader)
            print(f'第 {epoch + 1:2d} 轮训练 | 平均损失: {avg_loss:.6f}')

    return model  # 返回训练好的模型


# ===================== 无人机3D可视化函数 =====================
def draw_drone(ax, state):
    """
    在3D坐标轴上绘制无人机的当前姿态
    参数:
        ax: matplotlib的3D坐标轴对象
        state: 无人机当前状态数组
    返回值:
        更新后的3D坐标轴对象
    """
    # 解包当前位置和姿态角
    x, y, z = state[0], state[1], state[2]
    roll, pitch, yaw = state[3], state[4], state[5]

    # 无人机几何参数
    body_length = 0.5  # 机身长度（米）
    arm_length = 0.3  # 机臂长度（米）
    center = np.array([x, y, z])  # 无人机中心位置

    # 计算旋转矩阵：机体坐标系→世界坐标系（基于欧拉角）
    cosr, sinr = np.cos(roll), np.sin(roll)
    cosp, sinp = np.cos(pitch), np.sin(pitch)
    cosy, siny = np.cos(yaw), np.sin(yaw)
    # 欧拉角旋转矩阵（Z-Y-X顺序：偏航→俯仰→滚转）
    R = np.array([
        [cosy * cosp, cosy * sinp * sinr - siny * cosr, cosy * sinp * cosr + siny * sinr],
        [siny * cosp, siny * sinp * sinr + cosy * cosr, siny * sinp * cosr - cosy * sinr],
        [-sinp, cosp * sinr, cosp * cosr]
    ])

    # 计算无人机机体关键点在世界坐标系的位置
    front = center + R @ np.array([body_length / 2, 0, 0])  # 机头
    back = center + R @ np.array([-body_length / 2, 0, 0])  # 机尾
    left = center + R @ np.array([0, arm_length, 0])  # 左侧
    right = center + R @ np.array([0, -arm_length, 0])  # 右侧
    top = center + R @ np.array([0, 0, 0.1])  # 机顶（用于绘制垂直方向）

    # 绘制无人机结构
    ax.clear()  # 清空之前的绘制内容
    ax.plot([front[0], back[0]], [front[1], back[1]], [front[2], back[2]], 'b-', linewidth=3, label='机身')
    ax.plot([left[0], right[0]], [left[1], right[1]], [left[2], right[2]], 'b-', linewidth=3, label='机臂')
    ax.plot([center[0], top[0]], [center[1], top[1]], [center[2], top[2]], 'r-', linewidth=2, label='垂向')

    # 绘制四个螺旋桨位置（绿色散点）
    props = [
        center + R @ np.array([body_length / 4, arm_length, 0]),  # 前左
        center + R @ np.array([-body_length / 4, arm_length, 0]),  # 后左
        center + R @ np.array([-body_length / 4, -arm_length, 0]),  # 后右
        center + R @ np.array([body_length / 4, -arm_length, 0])  # 前右
    ]
    for p in props:
        ax.scatter(p[0], p[1], p[2], color='g', s=50, label='螺旋桨' if p is props[0] else "")

    # 设置3D坐标轴范围和标签
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])
    ax.set_xlabel('X轴 (m)')
    ax.set_ylabel('Y轴 (m)')
    ax.set_zlabel('Z轴 (m)')
    ax.set_title('无人机平衡控制可视化')
    ax.legend(loc='upper right')

    return ax


# ===================== 实时控制可视化函数 =====================
def visualize_control(model):
    """
    实时运行训练好的模型，可视化无人机控制效果
    参数:
        model: 训练完成的DroneController模型
    """
    drone = DroneModel()  # 实例化无人机模型
    state = drone.reset()  # 重置初始状态

    # 创建绘图窗口，2x2子图布局
    fig = plt.figure(figsize=(12, 8))

    # 子图1：3D无人机姿态可视化
    ax3d = fig.add_subplot(221, projection='3d')

    # 子图2：姿态角变化曲线（滚转、俯仰、偏航）
    ax_angles = fig.add_subplot(222)
    ax_angles.set_title('姿态角度变化')
    ax_angles.set_ylim([-0.8, 0.8])
    ax_angles.set_ylabel('角度 (rad)')
    ax_angles.grid(True, alpha=0.3)
    lines_angles = []
    for color, label in zip(['r', 'g', 'b'], ['滚转角', '俯仰角', '偏航角']):
        line, = ax_angles.plot([], [], color=color, label=label)
        lines_angles.append(line)
    ax_angles.legend()

    # 子图3：位置变化曲线（x、y、z）
    ax_pos = fig.add_subplot(223)
    ax_pos.set_title('位置变化')
    ax_pos.set_ylim([-0.5, 1.5])
    ax_pos.set_ylabel('位置 (m)')
    ax_pos.set_xlabel('时间 (s)')
    ax_pos.grid(True, alpha=0.3)
    lines_pos = []
    for color, label in zip(['r', 'g', 'b'], ['X', 'Y', 'Z']):
        line, = ax_pos.plot([], [], color=color, label=label)
        lines_pos.append(line)
    ax_pos.legend()

    # 子图4：四个螺旋桨推力变化曲线
    ax_actions = fig.add_subplot(224)
    ax_actions.set_title('螺旋桨推力变化')
    ax_actions.set_ylim([0, 20])
    ax_actions.set_ylabel('推力 (N)')
    ax_actions.set_xlabel('时间 (s)')
    ax_actions.grid(True, alpha=0.3)
    lines_actions = []
    for i in range(4):
        line, = ax_actions.plot([], [], label=f'螺旋桨 {i+1}')
        lines_actions.append(line)
    ax_actions.legend()

    # 历史数据缓存：用于绘制曲线
    history = {
        'angles': np.zeros((3, 0)),  # 姿态角历史 (3, time_steps)
        'pos': np.zeros((3, 0)),  # 位置历史 (3, time_steps)
        'actions': np.zeros((4, 0)),  # 推力历史 (4, time_steps)
        'time': np.array([])  # 时间戳历史
    }
    max_history = 100  # 最大缓存长度（只显示最近100个时间步）
    time = 0  # 累计时间

    def update(frame):
        """
        动画更新函数：每帧执行一次，更新无人机状态和绘图
        参数:
            frame: 动画帧数（matplotlib自动传入）
        返回值:
            所有需要更新的绘图元素
        """
        nonlocal state, time, history  # 引用外部变量

        # 1. 使用训练好的模型计算控制动作（关闭梯度计算，加快速度）
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)  # numpy→tensor
            action = model(state_tensor).numpy()  # tensor→numpy

        # 2. 执行动作，更新无人机状态
        state, reward, done = drone.step(action)

        # 3. 更新历史数据缓存
        time += drone.dt  # 累计时间
        history['time'] = np.append(history['time'], time)
        history['angles'] = np.hstack((history['angles'], state[3:6].reshape(-1, 1)))
        history['pos'] = np.hstack((history['pos'], state[0:3].reshape(-1, 1)))
        history['actions'] = np.hstack((history['actions'], action.reshape(-1, 1)))

        # 4. 裁剪历史数据，只保留最近max_history个数据点
        if len(history['time']) > max_history:
            history['time'] = history['time'][-max_history:]
            history['angles'] = history['angles'][:, -max_history:]
            history['pos'] = history['pos'][:, -max_history:]
            history['actions'] = history['actions'][:, -max_history:]

        # 5. 更新3D无人机姿态图
        draw_drone(ax3d, state)

        # 6. 更新姿态角曲线
        for i, line in enumerate(lines_angles):
            line.set_data(history['time'], history['angles'][i])

        # 7. 更新位置曲线
        for i, line in enumerate(lines_pos):
            line.set_data(history['time'], history['pos'][i])

        # 8. 更新推力曲线
        for i, line in enumerate(lines_actions):
            line.set_data(history['time'], history['actions'][i])

        # 9. 自动调整x轴范围（跟随时间变化）
        for ax in [ax_angles, ax_pos, ax_actions]:
            ax.set_xlim([history['time'][0], history['time'][-1]])

        # 10. 无人机失控时停止动画
        if done:
            print("\n无人机失去平衡！动画停止")
            ani.event_source.stop()  # 停止动画

        # 返回所有更新的绘图元素（用于blitting优化）
        return ax3d, *lines_angles, *lines_pos, *lines_actions

    # 创建动画对象：每50ms更新一次，无限循环
    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()  # 自动调整子图间距
    plt.show()  # 显示动画窗口


# ===================== 主函数入口 =====================
if __name__ == "__main__":
    print("=" * 50)
    print("开始训练无人机平衡控制模型...")
    print("=" * 50)
    model = train_model(epochs=50)  # 训练模型（50轮）

    print("\n" + "=" * 50)
    print("训练完成！开始实时可视化控制过程...")
    print("=" * 50)
    visualize_control(model)  # 启动可视化