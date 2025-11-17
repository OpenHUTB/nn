import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 无人机物理模型参数
class DroneModel:
    def __init__(self):
        self.mass = 1.0  # 质量(kg)
        self.inertia = np.array([0.1, 0.1, 0.2])  # 转动惯量
        self.gravity = 9.81  # 重力加速度
        self.dt = 0.01  # 时间步长
        self.reset()

    def reset(self):
        # 状态: [x, y, z, 滚转角, 俯仰角, 偏航角, 线速度, 角速度]
        self.state = np.array([0.0, 0.0, 1.0,  # 位置
                               0.05 * random.random() - 0.025,  # 滚转角
                               0.05 * random.random() - 0.025,  # 俯仰角
                               0.0,  # 偏航角
                               0.0, 0.0, 0.0,  # 线速度
                               0.0, 0.0, 0.0])  # 角速度
        return self.state

    def step(self, action):
        # action: 四个螺旋桨的推力
        f1, f2, f3, f4 = action

        # 计算合力和力矩
        thrust = (f1 + f2 + f3 + f4)
        torque_x = (f2 + f4 - f1 - f3) * 0.1
        torque_y = (f1 + f2 - f3 - f4) * 0.1
        torque_z = (f2 + f3 - f1 - f4) * 0.05

        # 更新位置和姿态
        roll, pitch, yaw = self.state[3:6]

        # 线加速度计算
        ax = (np.cos(yaw) * np.sin(pitch) + np.sin(yaw) * np.sin(roll) * np.cos(pitch)) * thrust / self.mass
        ay = (np.sin(yaw) * np.sin(pitch) - np.cos(yaw) * np.sin(roll) * np.cos(pitch)) * thrust / self.mass
        az = self.gravity - (np.cos(roll) * np.cos(pitch)) * thrust / self.mass

        # 角加速度计算
        angular_acc = np.array([torque_x / self.inertia[0],
                                torque_y / self.inertia[1],
                                torque_z / self.inertia[2]])

        # 更新速度
        self.state[6:9] += np.array([ax, ay, az]) * self.dt

        # 更新位置
        self.state[0:3] += self.state[6:9] * self.dt

        # 更新角速度
        self.state[9:12] += angular_acc * self.dt

        # 更新姿态
        self.state[3:6] += self.state[9:12] * self.dt

        # 限制角度范围在[-pi, pi]
        self.state[3:6] = np.mod(self.state[3:6] + np.pi, 2 * np.pi) - np.pi

        # 计算奖励: 越稳定奖励越高
        reward = - (np.sum(np.square(self.state[3:5])) + 0.1 * np.sum(np.square(self.state[9:11]))
                    + 0.01 * np.sum(np.square(self.state[0:2])) + 0.1 * np.square(self.state[2] - 1.0))

        # 判断是否失败
        done = (abs(self.state[3]) > np.pi / 4 or  # 滚转角过大
                abs(self.state[4]) > np.pi / 4 or  # 俯仰角过大
                self.state[2] < 0.2)  # 高度过低

        return self.state, reward, done


# 深度学习控制器模型
class DroneController(nn.Module):
    def __init__(self, state_dim=12, action_dim=4):
        super(DroneController, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))  # 输出范围[-1,1]
        return x * 15.0  # 推力范围[-15,15]


# 生成训练数据
class DroneDataset(Dataset):
    def __init__(self, size=10000):
        self.size = size
        self.drone = DroneModel()
        self.data = []

        for _ in range(size):
            state = self.drone.reset()
            # 简单的规则控制器生成目标动作
            roll, pitch = state[3], state[4]
            # 基于姿态误差的控制动作
            f1 = 5.0 + 10.0 * pitch + 10.0 * roll
            f2 = 5.0 + 10.0 * pitch - 10.0 * roll
            f3 = 5.0 - 10.0 * pitch - 10.0 * roll
            f4 = 5.0 - 10.0 * pitch + 10.0 * roll
            action = np.clip([f1, f2, f3, f4], 0, 15)

            self.data.append((state, action))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        state, action = self.data[idx]
        return torch.FloatTensor(state), torch.FloatTensor(action)


# 训练模型
def train_model(epochs=50):
    dataset = DroneDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = DroneController()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        for states, actions in dataloader:
            optimizer.zero_grad()

            outputs = model(states)
            loss = criterion(outputs, actions)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'第 {epoch + 1} 轮训练, 损失: {running_loss / len(dataloader):.6f}')

    return model


# 可视化无人机
def draw_drone(ax, state):
    ax.clear()
    x, y, z = state[0], state[1], state[2]
    roll, pitch, yaw = state[3], state[4], state[5]

    # 无人机机体尺寸
    body_length = 0.5
    arm_length = 0.3

    # 无人机中心位置
    center = np.array([x, y, z])

    # 旋转矩阵 (从机体坐标系到世界坐标系)
    cosr, sinr = np.cos(roll), np.sin(roll)
    cosp, sinp = np.cos(pitch), np.sin(pitch)
    cosy, siny = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cosy * cosp, cosy * sinp * sinr - siny * cosr, cosy * sinp * cosr + siny * sinr],
        [siny * cosp, siny * sinp * sinr + cosy * cosr, siny * sinp * cosr - cosy * sinr],
        [-sinp, cosp * sinr, cosp * cosr]
    ])

    # 机体关键点 (机体坐标系)
    front = center + R @ np.array([body_length / 2, 0, 0])
    back = center + R @ np.array([-body_length / 2, 0, 0])
    left = center + R @ np.array([0, arm_length, 0])
    right = center + R @ np.array([0, -arm_length, 0])
    top = center + R @ np.array([0, 0, 0.1])

    # 绘制无人机
    ax.plot([front[0], back[0]], [front[1], back[1]], [front[2], back[2]], 'b-', linewidth=3)
    ax.plot([left[0], right[0]], [left[1], right[1]], [left[2], right[2]], 'b-', linewidth=3)
    ax.plot([center[0], top[0]], [center[1], top[1]], [center[2], top[2]], 'r-', linewidth=2)

    # 绘制四个螺旋桨
    props = [
        center + R @ np.array([body_length / 4, arm_length, 0]),
        center + R @ np.array([-body_length / 4, arm_length, 0]),
        center + R @ np.array([-body_length / 4, -arm_length, 0]),
        center + R @ np.array([body_length / 4, -arm_length, 0])
    ]

    for p in props:
        ax.scatter(p[0], p[1], p[2], color='g', s=50)

    # 设置坐标轴
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('无人机平衡控制可视化')

    return ax


# 实时可视化控制过程
def visualize_control(model):
    drone = DroneModel()
    state = drone.reset()

    fig = plt.figure(figsize=(12, 8))

    # 3D 无人机姿态图
    ax3d = fig.add_subplot(221, projection='3d')

    # 姿态角度图
    ax_angles = fig.add_subplot(222)
    ax_angles.set_title('姿态角度')
    ax_angles.set_ylim([-0.8, 0.8])
    ax_angles.set_ylabel('角度 (rad)')
    lines_angles = []
    for color in ['r', 'g', 'b']:
        line, = ax_angles.plot([], [], color=color, label=['滚转角', '俯仰角', '偏航角'][len(lines_angles)])
        lines_angles.append(line)
    ax_angles.legend()

    # 位置图
    ax_pos = fig.add_subplot(223)
    ax_pos.set_title('位置')
    ax_pos.set_ylim([-0.5, 1.5])
    ax_pos.set_ylabel('位置 (m)')
    lines_pos = []
    for color in ['r', 'g', 'b']:
        line, = ax_pos.plot([], [], color=color, label=['X', 'Y', 'Z'][len(lines_pos)])
        lines_pos.append(line)
    ax_pos.legend()

    # 控制动作图
    ax_actions = fig.add_subplot(224)
    ax_actions.set_title('螺旋桨推力')
    ax_actions.set_ylim([0, 20])
    ax_actions.set_ylabel('推力 (N)')
    lines_actions = []
    for i in range(4):
        line, = ax_actions.plot([], [], label=f'螺旋桨 {i + 1}')
        lines_actions.append(line)
    ax_actions.legend()

    # 数据缓存
    history = {
        'angles': np.zeros((3, 0)),
        'pos': np.zeros((3, 0)),
        'actions': np.zeros((4, 0)),
        'time': np.array([])
    }
    max_history = 100
    time = 0

    def update(frame):
        nonlocal state, time, history

        # 使用深度学习模型计算控制动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action = model(state_tensor).numpy()

        # 执行动作
        state, reward, done = drone.step(action)

        # 更新历史数据
        time += drone.dt
        history['time'] = np.append(history['time'], time)
        history['angles'] = np.hstack((history['angles'], state[3:6].reshape(-1, 1)))
        history['pos'] = np.hstack((history['pos'], state[0:3].reshape(-1, 1)))
        history['actions'] = np.hstack((history['actions'], action.reshape(-1, 1)))

        # 保持历史数据长度
        if len(history['time']) > max_history:
            history['time'] = history['time'][-max_history:]
            history['angles'] = history['angles'][:, -max_history:]
            history['pos'] = history['pos'][:, -max_history:]
            history['actions'] = history['actions'][:, -max_history:]

        # 更新3D无人机图
        draw_drone(ax3d, state)

        # 更新姿态角度图
        for i, line in enumerate(lines_angles):
            line.set_data(history['time'], history['angles'][i])

        # 更新位置图
        for i, line in enumerate(lines_pos):
            line.set_data(history['time'], history['pos'][i])

        # 更新控制动作图
        for i, line in enumerate(lines_actions):
            line.set_data(history['time'], history['actions'][i])

        # 自动调整X轴范围
        for ax in [ax_angles, ax_pos, ax_actions]:
            ax.set_xlim([history['time'][0], history['time'][-1]])

        if done:
            print("无人机失去平衡!")
            ani.event_source.stop()

        return ax3d, *lines_angles, *lines_pos, *lines_actions

    ani = FuncAnimation(fig, update, interval=50, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("正在训练无人机平衡控制模型...")
    model = train_model(epochs=50)
    print("训练完成，开始可视化控制过程...")
    visualize_control(model)