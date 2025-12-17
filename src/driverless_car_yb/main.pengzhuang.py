import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import random
import imageio
import matplotlib

matplotlib.use('TkAgg')  # 解决动画显示问题


# --------------------------
# 1. 模拟环境与数据集生成（支持动态障碍物）
# --------------------------
class CollisionAvoidanceDataset(Dataset):
    """无人车避撞数据集（模拟生成，含动态障碍物逻辑）"""

    def __init__(self, num_samples=5000, env_size=(100, 100), car_size=(5, 3), sensor_range=20):
        self.env_size = env_size  # 环境尺寸 (宽, 高)
        self.car_size = car_size  # 车辆尺寸 (长, 宽)
        self.sensor_range = sensor_range  # 传感器探测范围
        self.data = []
        self.labels = []

        # 生成样本：每个样本是传感器观测图 + 避撞动作标签
        for _ in range(num_samples):
            # 随机生成车辆位置和朝向
            car_x = random.uniform(car_size[0], env_size[0] - car_size[0])
            car_y = random.uniform(car_size[1], env_size[1] - car_size[1])
            car_theta = random.uniform(0, np.pi * 2)  # 朝向角（弧度）

            # 随机生成1-3个动态障碍物（含移动方向和速度）
            num_obstacles = random.randint(1, 3)
            obstacles = []
            for _ in range(num_obstacles):
                obs_x = random.uniform(0, env_size[0])
                obs_y = random.uniform(0, env_size[1])
                obs_radius = random.uniform(2, 5)
                # 随机移动方向（0-2π）和速度（0.3-1.0单位/步）
                obs_dir = random.uniform(0, np.pi * 2)
                obs_speed = random.uniform(0.3, 1.0)
                obstacles.append((obs_x, obs_y, obs_radius, obs_dir, obs_speed))

            # 生成传感器观测图（模拟激光雷达/摄像头观测）
            sensor_img = self.generate_sensor_image(car_x, car_y, car_theta, obstacles)

            # 计算安全动作标签（0=直行，1=左转，2=右转）
            label = self.calculate_safe_action(car_x, car_y, car_theta, obstacles)

            self.data.append(sensor_img)
            self.labels.append(label)

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32).unsqueeze(1)  # (N, 1, 64, 64)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)  # (N,)

    def generate_sensor_image(self, car_x, car_y, car_theta, obstacles):
        """生成传感器观测图（64x64灰度图）：白色=障碍物，黑色=安全区域"""
        img = np.zeros((64, 64), dtype=np.float32)
        # 传感器坐标系转换（车辆为中心，朝向为正前方）
        for i in range(64):
            for j in range(64):
                # 像素坐标转传感器坐标
                sensor_x = (j - 32) * (self.sensor_range / 32)  # 左右范围：-20~20
                sensor_y = (32 - i) * (self.sensor_range / 32)  # 前后范围：-20~20（前方为正）

                # 过滤超出传感器范围的点
                if np.sqrt(sensor_x ** 2 + sensor_y ** 2) > self.sensor_range:
                    continue

                # 转换到全局坐标系
                global_x = car_x + sensor_x * np.cos(car_theta) - sensor_y * np.sin(car_theta)
                global_y = car_y + sensor_x * np.sin(car_theta) + sensor_y * np.cos(car_theta)

                # 检查是否碰撞障碍物（只关注位置信息，忽略移动参数）
                for (obs_x, obs_y, obs_r, _, _) in obstacles:
                    if np.sqrt((global_x - obs_x) ** 2 + (global_y - obs_y) ** 2) < obs_r + self.car_size[1] / 2:
                        img[i, j] = 1.0  # 标记障碍物
                        break
        return img

    def calculate_safe_action(self, car_x, car_y, car_theta, obstacles):
        """计算安全动作：检查直行、左转、右转是否安全（考虑障碍物移动趋势）"""
        # 动作对应的转向角变化（弧度）
        actions = [0, -np.pi / 6, np.pi / 6]  # 0=直行，-30°=左转，30°=右转
        safe_actions = []

        for d_theta in actions:
            # 计算动作后的车辆位置和朝向
            new_theta = car_theta + d_theta
            new_x = car_x + self.car_size[0] * np.cos(new_theta)
            new_y = car_y + self.car_size[0] * np.sin(new_theta)

            # 检查是否超出边界
            if (new_x < self.car_size[0] / 2 or new_x > self.env_size[0] - self.car_size[0] / 2 or
                    new_y < self.car_size[1] / 2 or new_y > self.env_size[1] - self.car_size[1] / 2):
                continue

            # 检查是否与障碍物（含未来位置）碰撞
            collision = False
            for (obs_x, obs_y, obs_r, obs_dir, obs_speed) in obstacles:
                # 预测障碍物下一步位置（考虑移动趋势）
                future_obs_x = obs_x + obs_speed * np.cos(obs_dir)
                future_obs_y = obs_y + obs_speed * np.sin(obs_dir)

                # 检查车辆与障碍物当前位置或未来位置的距离
                dist_current = np.sqrt((new_x - obs_x) ** 2 + (new_y - obs_y) ** 2)
                dist_future = np.sqrt((new_x - future_obs_x) ** 2 + (new_y - future_obs_y) ** 2)

                if dist_current < obs_r + self.car_size[1] / 2 or dist_future < obs_r + self.car_size[1] / 2:
                    collision = True
                    break

            if not collision:
                safe_actions.append(d_theta)

        # 选择安全动作（优先直行，否则随机选一个安全动作）
        if 0 in safe_actions:
            return 0
        elif len(safe_actions) > 0:
            return 1 if -np.pi / 6 in safe_actions else 2
        else:
            return random.choice([0, 1, 2])  # 无安全动作时随机选择

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 生成数据集（减少样本数加快训练）
dataset = CollisionAvoidanceDataset(num_samples=2000)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# --------------------------
# 2. 构建避撞CNN模型
# --------------------------
class CollisionAvoidanceCNN(nn.Module):
    """基于CNN的避撞决策模型：输入传感器图像，输出动作（直行/左转/右转）"""

    def __init__(self, num_actions=3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CollisionAvoidanceCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --------------------------
# 3. 训练模型
# --------------------------
def train_model(model, train_loader, criterion, optimizer, epochs=15):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # 前向传播
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_data.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # 绘制训练损失曲线
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.title('Model Training Loss')
    plt.show()


# 训练模型
train_model(model, train_loader, criterion, optimizer, epochs=15)


# --------------------------
# 4. 动态障碍物避撞模拟与动图生成
# --------------------------
def simulate_collision_avoidance(model, env_size=(100, 100), car_size=(5, 3), sensor_range=20, num_steps=80):
    """模拟无人车避撞过程（含动态障碍物），生成动图"""
    # 初始化车辆状态
    car_x = car_size[0] * 2
    car_y = env_size[1] / 2
    car_theta = 0  # 初始朝向：右（x轴正方向）

    # 生成动态障碍物（3个，含移动参数）
    obstacles = [
        (40, 30, 4, np.pi / 4, 0.6),  # 右上方向移动，速度0.6
        (60, 70, 5, -np.pi / 3, 0.4),  # 左下方向移动，速度0.4
        (80, 50, 3, np.pi * 3 / 4, 0.8)  # 左上方向移动，速度0.8
    ]

    # 记录每一步的状态（用于生成动图）
    states = []
    model.eval()

    with torch.no_grad():
        for step in range(num_steps):
            # 生成传感器图像
            sensor_img = dataset.generate_sensor_image(car_x, car_y, car_theta, obstacles)
            sensor_tensor = torch.tensor(sensor_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # 模型预测动作
            output = model(sensor_tensor)
            action = torch.argmax(output).item()  # 0=直行，1=左转，2=右转

            # 更新车辆状态
            d_theta = 0.0
            if action == 1:
                d_theta = -np.pi / 6  # 左转30°
            elif action == 2:
                d_theta = np.pi / 6  # 右转30°

            car_theta += d_theta
            car_x += car_size[0] * np.cos(car_theta)
            car_y += car_size[0] * np.sin(car_theta)

            # 边界约束（防止车辆出界）
            car_x = max(car_size[0] / 2, min(env_size[0] - car_size[0] / 2, car_x))
            car_y = max(car_size[1] / 2, min(env_size[1] - car_size[1] / 2, car_y))

            # 更新动态障碍物位置（边界反弹）
            updated_obstacles = []
            for (x, y, r, dir_, speed) in obstacles:
                new_x = x + speed * np.cos(dir_)
                new_y = y + speed * np.sin(dir_)

                # 边界反弹逻辑
                if new_x < r or new_x > env_size[0] - r:
                    dir_ = np.pi - dir_  # x方向反弹
                if new_y < r or new_y > env_size[1] - r:
                    dir_ = -dir_  # y方向反弹

                # 重新计算位置（确保不越界）
                new_x = x + speed * np.cos(dir_)
                new_y = y + speed * np.sin(dir_)
                updated_obstacles.append((new_x, new_y, r, dir_, speed))

            obstacles = updated_obstacles

            # 记录状态
            states.append({
                'car_x': car_x,
                'car_y': car_y,
                'car_theta': car_theta,
                'obstacles': obstacles,
                'action': action
            })

    # 生成动图
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, env_size[0])
    ax.set_ylim(0, env_size[1])
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Autonomous Vehicle Collision Avoidance (Dynamic Obstacles)')

    # 绘制障碍物（初始位置）
    obstacle_patches = []
    for (x, y, r, _, _) in obstacles:
        patch = Circle((x, y), r, color='red', alpha=0.7)
        ax.add_patch(patch)
        obstacle_patches.append(patch)

    # 绘制车辆（矩形）
    car_patch = Rectangle((0, 0), car_size[0], car_size[1], color='blue', alpha=0.7)
    ax.add_patch(car_patch)

    # 绘制传感器范围（半圆）
    sensor_arc = plt.Circle((0, 0), sensor_range, color='green', alpha=0.1)
    ax.add_patch(sensor_arc)

    # 动作标签和步骤显示
    action_text = ax.text(10, env_size[1] - 10, '', fontsize=12, color='black', weight='bold')
    step_text = ax.text(env_size[0] - 80, env_size[1] - 10, '', fontsize=12, color='black', weight='bold')

    def update(frame):
        state = states[frame]
        cx, cy, ctheta = state['car_x'], state['car_y'], state['car_theta']
        action = state['action']
        current_obstacles = state['obstacles']

        # 更新车辆位置和朝向
        car_patch.set_xy((cx - car_size[0] / 2, cy - car_size[1] / 2))
        car_patch.set_angle(np.degrees(ctheta))

        # 更新传感器范围
        sensor_arc.center = (cx, cy)

        # 更新障碍物位置
        for i, (x, y, r, _, _) in enumerate(current_obstacles):
            obstacle_patches[i].center = (x, y)

        # 更新文本标签
        action_labels = ['Straight', 'Left Turn', 'Right Turn']
        action_text.set_text(f"Action: {action_labels[action]}")
        step_text.set_text(f"Step: {frame + 1}/{num_steps}")

        return [car_patch, sensor_arc, action_text, step_text] + obstacle_patches

    # 生成动画（提升帧率和清晰度）
    ani = animation.FuncAnimation(
        fig, update, frames=num_steps, interval=150, blit=True, repeat=False
    )

    # 保存为GIF（高帧率+高分辨率）
    ani.save(
        'dynamic_collision_avoidance.gif',
        writer='pillow',
        fps=10,
        dpi=150,
        savefig_kwargs={'facecolor': 'white'}
    )
    print("动态障碍物避撞模拟动图已保存为 dynamic_collision_avoidance.gif")

    # 显示动画
    plt.show()


# 运行避撞模拟并生成动图
simulate_collision_avoidance(model)