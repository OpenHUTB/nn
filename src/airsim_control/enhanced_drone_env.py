import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import os

from agents.dqn.config import DQNConfig


class EnhancedDroneEnv:
    """增强版无人机环境，包含智能奖励函数和可视化功能"""

    def __init__(self, client, config=None, enable_visualization=True):
        self.client = client
        self.config = config or DQNConfig()
        self.action_space = list(range(self.config.ACTION_DIM))
        self.prev_position = None
        self.enable_visualization = enable_visualization

        # 初始化可视化
        if self.enable_visualization:
            self.setup_visualization()

        # 目标设置
        self.target_position = np.array([0.0, 0.0, -10.0])  # 目标位置（下方10米）
        self.target_radius = 5.0  # 目标半径

        # 历史记录
        self.episode_history = {
            'rewards': [],
            'distances': [],
            'heights': [],
            'velocities': [],
            'collisions': []
        }

        # 参数权重（可调整）
        self.weights = {
            'distance': -0.5,      # 距离惩罚（越接近目标越好）
            'height': -0.1,       # 高度惩罚（保持在目标高度）
            'velocity_smooth': -0.05,  # 速度平滑惩罚
            'action_change': -0.02,    # 动作变化惩罚
            'target_bonus': 10.0,      # 到达目标奖励
            'survival': 0.5,      # 存活奖励
            'collision': -50.0,   # 碰撞惩罚
            'boundary': -20.0,    # 边界惩罚
            'energy': -0.01       # 能量消耗惩罚
        }

        # 动作历史（用于计算动作变化惩罚）
        self.prev_action = None
        self.velocity_history = deque(maxlen=5)

    def setup_visualization(self):
        """设置可视化"""
        plt.ioff()  # 关闭交互模式，避免打断训练

        # 创建图表
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle('无人机训练可视化', fontsize=16)

        # 奖励曲线
        self.ax1.set_title('每轮奖励', fontsize=14)
        self.ax1.set_xlabel('轮次')
        self.ax1.set_ylabel('奖励值')
        self.reward_line, = self.ax1.plot([], [], 'b-', linewidth=2)
        self.ax1.grid(True, alpha=0.3)

        # 距离曲线
        self.ax2.set_title('到目标距离', fontsize=14)
        self.ax2.set_xlabel('轮次')
        self.ax2.set_ylabel('距离 (米)')
        self.distance_line, = self.ax2.plot([], [], 'r-', linewidth=2)
        self.ax2.grid(True, alpha=0.3)

        # 高度曲线
        self.ax3.set_title('飞行高度 vs 目标高度', fontsize=14)
        self.ax3.set_xlabel('轮次')
        self.ax3.set_ylabel('高度 (米)')
        self.height_line, = self.ax3.plot([], [], 'g-', linewidth=2)
        self.ax3.axhline(y=-10, color='r', linestyle='--', linewidth=2, label='目标高度')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)

        # 速度分布
        self.ax4.set_title('速度分布', fontsize=14)
        self.ax4.set_xlabel('时间步')
        self.ax4.set_ylabel('速度大小')
        self.velocity_line, = self.ax4.plot([], [], 'm-', linewidth=2)
        self.ax4.grid(True, alpha=0.3)

        plt.tight_layout()

    def reset(self):
        """重置环境，返回初始状态"""
        self.client.destroy()
        self.client.client.confirmConnection()
        self.client.client.enableApiControl(True)
        self.client.client.armDisarm(True)
        self.client.start()

        state = self._get_state()
        self.prev_position = self._get_position()

        # 重置历史记录
        self.episode_history = {
            'rewards': [],
            'distances': [],
            'heights': [],
            'velocities': [],
            'collisions': []
        }
        self.velocity_history.clear()
        self.prev_action = None

        return state

    def step(self, action):
        """执行动作，返回 (next_state, reward, done, info)"""
        # 记录动作变化
        action_change_penalty = 0
        if self.prev_action is not None and action != self.prev_action:
            action_change_penalty = self.weights['action_change']
        self.prev_action = action

        # 执行动作
        vx, vy, vz = self.config.ACTION_VELOCITIES[action]
        self.client.move('velocity', vx, vy, vz)

        next_state = self._get_state()
        reward, done = self._compute_enhanced_reward(action, action_change_penalty)
        info = {
            'collision': self._check_collision(),
            'distance_to_target': self._get_distance_to_target(),
            'height': self._get_position()[2]
        }

        self.prev_position = self._get_position()

        # 记录历史
        self.episode_history['rewards'].append(reward)
        self.episode_history['distances'].append(info['distance_to_target'])
        self.episode_history['heights'].append(info['height'])
        velocity_mag = np.linalg.norm([vx, vy, vz])
        self.episode_history['velocities'].append(velocity_mag)
        self.episode_history['collisions'].append(info['collision'])

        return next_state, reward, done, info

    def _get_state(self):
        """获取当前状态向量"""
        state = self.client.get_state()
        kinematics = state.kinematics_estimated

        position = kinematics.position
        velocity = kinematics.linear_velocity

        collision_info = self.client.get_collision_info()
        has_collision = 1.0 if collision_info.has_collided else 0.0

        # 计算到目标的距离和方向
        distance_to_target = self._get_distance_to_target()
        direction_to_target = self._get_direction_to_target()

        state_vec = np.array([
            position.x_val / 50.0,      # 归一化位置
            position.y_val / 50.0,
            position.z_val / 50.0,
            velocity.x_val / 5.0,       # 归一化速度
            velocity.y_val / 5.0,
            velocity.z_val / 5.0,
            has_collision,
            distance_to_target / 50.0,  # 归一化距离到目标
            direction_to_target[0],      # 目标方向x
            direction_to_target[1],      # 目标方向y
            direction_to_target[2]       # 目标方向z
        ], dtype=np.float32)

        return state_vec

    def _get_position(self):
        """获取当前位置"""
        state = self.client.get_state()
        kinematics = state.kinematics_estimated
        pos = kinematics.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def _check_collision(self):
        """检查是否碰撞"""
        return self.client.get_collision_info().has_collided

    def _get_distance_to_target(self):
        """计算到目标的距离"""
        current_pos = self._get_position()
        return np.linalg.norm(current_pos - self.target_position)

    def _get_direction_to_target(self):
        """获取到目标的方向向量（归一化）"""
        current_pos = self._get_position()
        direction = self.target_position - current_pos
        norm = np.linalg.norm(direction)
        return direction / norm if norm > 0 else np.zeros(3)

    def _compute_enhanced_reward(self, action, action_change_penalty):
        """计算增强奖励"""
        reward = 0
        done = False
        current_pos = self._get_position()

        # 1. 距离奖励
        distance_to_target = self._get_distance_to_target()
        reward += self.weights['distance'] * distance_to_target / 50.0

        # 2. 高度奖励（保持在目标高度附近）
        height_diff = abs(current_pos[2] - self.target_position[2])
        reward += self.weights['height'] * height_diff / 10.0

        # 3. 速度平滑奖励
        if len(self.velocity_history) > 0:
            avg_velocity = np.mean(self.velocity_history)
            reward += self.weights['velocity_smooth'] * avg_velocity
        self.velocity_history.append(np.linalg.norm([
            self.config.ACTION_VELOCITIES[action][0],
            self.config.ACTION_VELOCITIES[action][1],
            self.config.ACTION_VELOCITIES[action][2]
        ]))

        # 4. 目标到达奖励
        if distance_to_target < self.target_radius:
            reward += self.weights['target_bonus']
            done = True
            return reward, done

        # 5. 边界惩罚
        boundary_limit = 50.0
        if (abs(current_pos[0]) > boundary_limit or
            abs(current_pos[1]) > boundary_limit or
            current_pos[2] > 0 or current_pos[2] < -50):
            reward += self.weights['boundary']
            done = True
            return reward, done

        # 6. 碰撞检测
        if self._check_collision():
            reward += self.weights['collision']
            done = True
            return reward, done

        # 7. 存活奖励
        reward += self.weights['survival']

        # 8. 动作变化惩罚
        reward += action_change_penalty

        # 9. 能量消耗惩罚（与速度相关）
        velocity_cost = np.linalg.norm([
            self.config.ACTION_VELOCITIES[action][0],
            self.config.ACTION_VELOCITIES[action][1],
            self.config.ACTION_VELOCITIES[action][2]
        ])
        reward += self.weights['energy'] * velocity_cost

        return reward, done

    def update_visualization(self, episode, total_rewards):
        """更新可视化图表"""
        if not self.enable_visualization:
            return

        # 更新奖励曲线
        episodes = list(range(1, len(total_rewards) + 1))
        self.reward_line.set_data(episodes, total_rewards)
        self.ax1.relim()
        self.ax1.autoscale_view()

        # 更新距离曲线
        if len(self.episode_history['distances']) > 0:
            avg_distances = [np.mean(self.episode_history['distances'][:i+1])
                           for i in range(len(self.episode_history['distances']))]
            self.distance_line.set_data(episodes[-len(avg_distances):], avg_distances)
            self.ax2.relim()
            self.ax2.autoscale_view()

        # 更新高度曲线
        if len(self.episode_history['heights']) > 0:
            self.height_line.set_data(episodes[-len(self.episode_history['heights']):],
                                   self.episode_history['heights'])
            self.ax3.relim()
            self.ax3.autoscale_view()

        # 更新速度曲线
        if len(self.episode_history['velocities']) > 0:
            self.velocity_line.set_data(range(len(self.episode_history['velocities'])),
                                     self.episode_history['velocities'])
            self.ax4.relim()
            self.ax4.autoscale_view()

        # 只在最后一轮更新图表并显示
        if episode == len(total_rewards):
            self.fig.suptitle(f'无人机训练可视化 - 训练完成 ({episode}轮)', fontsize=16)
            plt.tight_layout()
            plt.show()  # 显示图表并阻塞等待用户关闭

    def save_training_plot(self, save_path='./training_plots'):
        """保存训练图表"""
        if not self.enable_visualization:
            return

        os.makedirs(save_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.fig.savefig(os.path.join(save_path, f'training_plot_{timestamp}.png'),
                        dpi=300, bbox_inches='tight')
        print(f"训练图表已保存到 {save_path}")

    def close(self):
        """关闭环境"""
        self.client.destroy()
        if self.enable_visualization:
            plt.close('all')  # 关闭所有图形窗口