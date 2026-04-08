import numpy as np
import mujoco
import matplotlib
# 强制使用TkAgg后端，避免与MuJoCo冲突
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os
from collections import deque
import random
from datetime import datetime
import sys

print("=" * 60)
print("机械臂训练可视化系统 - 修复版")
print("Python版本:", sys.version)
print("MuJoCo版本:", mujoco.__version__)
print("=" * 60)

# ============================ 1. 完全修复的仿真环境 ============================

class RobustArmEnv:
    """鲁棒的机械臂环境"""

    def __init__(self, model_path=None, render=False):
        # 设置默认模型路径
        if model_path is None:
            # 尝试多个可能的模型路径
            possible_paths = [
                "arm_with_gripper.xml",
                "./arm_with_gripper.xml",
                "../arm_with_gripper.xml"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

        if model_path is None or not os.path.exists(model_path):
            print("错误: 找不到模型文件!")
            print("请将 'arm_with_gripper.xml' 放在以下位置之一:")
            for path in possible_paths:
                print(f"  - {path}")
            sys.exit(1)

        print(f"加载模型: {model_path}")

        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
            print("✓ 模型加载成功")
        except Exception as e:
            print(f"✗ 加载模型失败: {e}")
            sys.exit(1)

        # 渲染设置
        self.render_enabled = False  # 初始不渲染
        self.viewer = None

        # 状态维度
        self.state_dim = 18  # 简化的状态维度
        self.action_dim = 6  # 6个关节控制

        # 训练参数
        self.max_steps = 100

        # 数据记录
        self.states_history = []
        self.actions_history = []

        # 执行器映射
        self.actuator_map = {}
        self._setup_actuators()

        print(f"状态维度: {self.state_dim}, 动作维度: {self.action_dim}")

    def _setup_actuators(self):
        """设置执行器映射"""
        print("设置执行器映射...")
        for i in range(1, 7):
            motor_name = f"motor_joint{i}"
            try:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
                if actuator_id != -1:
                    self.actuator_map[i-1] = actuator_id
                    print(f"  ✓ {motor_name} -> 执行器 {actuator_id}")
                else:
                    self.actuator_map[i-1] = i-1
                    print(f"  ⚠ {motor_name} 未找到，使用索引 {i-1}")
            except:
                self.actuator_map[i-1] = i-1

        print("执行器映射完成")

    def _get_safe_state(self):
        """安全地获取状态"""
        state = []

        try:
            # 1. 关节位置 (6个关节)
            for i in range(min(6, self.model.nq)):
                state.append(float(self.data.qpos[i]))

            # 如果不足6个，用0填充
            for i in range(len(state), 6):
                state.append(0.0)

            # 2. 关节速度 (6个关节)
            for i in range(min(6, self.model.nv)):
                state.append(float(self.data.qvel[i]))

            for i in range(len(state)-6, 6):
                state.append(0.0)

            # 3. 末端位置 (估计)
            # 使用简单估计：假设机械臂完全伸展时末端在 [0.5, 0, 0.5]
            ee_pos = [0.5, 0, 0.5]
            state.extend(ee_pos)

            # 4. 目标位置 (固定)
            target_pos = [0.4, 0.2, 0.1]  # test_sphere的位置
            state.extend(target_pos)

            return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"获取状态时出错: {e}")
            # 返回默认状态
            return np.zeros(self.state_dim, dtype=np.float32)

    def reset(self):
        """重置环境"""
        try:
            mujoco.mj_resetData(self.model, self.data)

            # 设置合理的初始位置
            initial_positions = [0.0, 0.0, -0.5, 0.0, 1.0, 0.0]
            for i in range(min(6, len(initial_positions))):
                self.data.qpos[i] = initial_positions[i]

            # 向前模拟
            mujoco.mj_forward(self.model, self.data)

            # 清空历史
            self.states_history = []
            self.actions_history = []

            # 获取初始状态
            state = self._get_safe_state()
            self.states_history.append(state)

            return state

        except Exception as e:
            print(f"重置环境失败: {e}")
            return np.zeros(self.state_dim, dtype=np.float32)

    def step(self, action):
        """执行一步"""
        try:
            # 确保动作在有效范围内
            action = np.clip(action, -1.0, 1.0) * 2.0  # 缩放到 [-2, 2]

            # 应用动作
            for i in range(min(6, len(action))):
                if i in self.actuator_map:
                    self.data.ctrl[self.actuator_map[i]] = action[i]

            # 向前模拟
            mujoco.mj_step(self.model, self.data)

            # 如果启用渲染且viewer存在，则同步
            if self.render_enabled and self.viewer is not None:
                try:
                    self.viewer.sync()
                    time.sleep(0.001)  # 小延迟避免过快
                except:
                    pass

            # 获取新状态
            state = self._get_safe_state()

            # 计算奖励（简化版）
            ee_pos = state[12:15]
            target_pos = state[15:18]
            distance = np.linalg.norm(ee_pos - target_pos)

            # 奖励函数
            reward = -distance * 10.0  # 距离惩罚

            if distance < 0.1:
                reward += 5.0
            if distance < 0.05:
                reward += 10.0
            if distance < 0.02:
                reward += 50.0

            # 动作平滑惩罚
            reward -= 0.01 * np.sum(np.square(action))

            # 检查是否结束
            done = len(self.states_history) >= self.max_steps or distance < 0.02

            # 记录
            self.states_history.append(state)
            self.actions_history.append(action)

            return state, reward, done, {"distance": distance}

        except Exception as e:
            print(f"执行步骤失败: {e}")
            state = self._get_safe_state()
            return state, -10.0, True, {"distance": 10.0}

    def start_viewer(self):
        """启动viewer"""
        if self.viewer is None:
            try:
                from mujoco import viewer
                print("启动MuJoCo viewer...")
                self.viewer = viewer.launch_passive(self.model, self.data)
                self.render_enabled = True
                print("✓ Viewer启动成功")
            except Exception as e:
                print(f"✗ 启动viewer失败: {e}")
                self.viewer = None
                self.render_enabled = False

    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass
            self.viewer = None
        self.render_enabled = False

# ============================ 2. 简化的RL代理 ============================

class SimpleAgent:
    """简化的代理"""

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 策略参数
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        # 简单线性策略
        self.weights = np.random.randn(state_dim, action_dim) * 0.01
        self.bias = np.zeros(action_dim)

        # 经验回放
        self.memory = deque(maxlen=2000)
        self.batch_size = 32

        # 学习参数
        self.lr = 0.001
        self.gamma = 0.95

        # 记录
        self.losses = []
        self.q_values = []

    def act(self, state, training=True):
        """选择动作"""
        if training and np.random.rand() < self.epsilon:
            # 随机动作
            action = np.random.uniform(-1, 1, self.action_dim)
            return np.clip(action, -1, 1)
        else:
            # 策略网络
            q_vals = np.dot(state, self.weights) + self.bias
            action = np.tanh(q_vals)  # 限制在[-1, 1]

            # 记录Q值
            self.q_values.append(np.max(q_vals))

            # 衰减探索率
            if training:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            return action

    def remember(self, state, action, reward, next_state, done):
        """记住经验"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        loss = 0.0

        for state, action, reward, next_state, done in batch:
            # 当前Q值
            current_q = np.dot(state, self.weights) + self.bias

            # 目标Q值
            if done:
                target = reward
            else:
                next_q = np.dot(next_state, self.weights) + self.bias
                target = reward + self.gamma * np.max(next_q)

            # 找到最大Q值的动作索引
            action_idx = np.argmax(action)

            # 计算损失
            td_error = target - current_q[action_idx]
            loss += td_error ** 2

            # 更新权重
            self.weights[:, action_idx] += self.lr * td_error * state
            self.bias[action_idx] += self.lr * td_error

        avg_loss = loss / self.batch_size
        self.losses.append(avg_loss)

        return avg_loss

    def save(self, filename):
        """保存模型"""
        np.savez(filename,
                weights=self.weights,
                bias=self.bias,
                epsilon=self.epsilon)
        print(f"模型已保存: {filename}")

    def load(self, filename):
        """加载模型"""
        if os.path.exists(filename):
            data = np.load(filename)
            self.weights = data['weights']
            self.bias = data['bias']
            self.epsilon = data['epsilon']
            print(f"模型已加载: {filename}")
            return True
        else:
            print(f"模型文件不存在: {filename}")
            return False

# ============================ 3. 修复的图表系统 ============================

class FixedChartSystem:
    """修复的图表系统"""

    def __init__(self):
        print("初始化图表系统...")

        # 创建图形窗口
        plt.close('all')  # 关闭所有现有图形

        # 设置图形大小
        self.fig = plt.figure(figsize=(14, 8))

        # 创建子图
        self.ax1 = plt.subplot(2, 3, 1)  # 奖励
        self.ax2 = plt.subplot(2, 3, 2)  # 距离
        self.ax3 = plt.subplot(2, 3, 3)  # 步数
        self.ax4 = plt.subplot(2, 3, 4)  # 动作分布
        self.ax5 = plt.subplot(2, 3, 5)  # 轨迹
        self.ax6 = plt.subplot(2, 3, 6)  # 统计信息

        # 数据存储
        self.episode_rewards = []
        self.episode_distances = []
        self.episode_lengths = []
        self.loss_history = []
        self.q_history = []

        # 初始化图表
        self._init_charts()

        # 显示图形
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)  # 给图形时间初始化

        print("✓ 图表系统初始化完成")

    def _init_charts(self):
        """初始化图表"""
        # 奖励图表
        self.ax1.set_title('Episode Rewards')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        self.ax1.grid(True, alpha=0.3)
        self.reward_line, = self.ax1.plot([], [], 'b-', linewidth=2)

        # 距离图表
        self.ax2.set_title('Final Distance')
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Distance')
        self.ax2.grid(True, alpha=0.3)
        self.distance_line, = self.ax2.plot([], [], 'r-', linewidth=2)
        self.ax2.axhline(y=0.02, color='g', linestyle='--', alpha=0.5, label='Success')
        self.ax2.legend()

        # 步数图表
        self.ax3.set_title('Episode Length')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Steps')
        self.ax3.grid(True, alpha=0.3)
        self.length_line, = self.ax3.plot([], [], 'g-', linewidth=2)

        # 动作分布
        self.ax4.set_title('Action Distribution')
        self.ax4.set_xlabel('Action Value')
        self.ax4.set_ylabel('Frequency')
        self.ax4.grid(True, alpha=0.3)

        # 轨迹
        self.ax5.set_title('End Effector Trajectory (XY)')
        self.ax5.set_xlabel('X Position')
        self.ax5.set_ylabel('Y Position')
        self.ax5.grid(True, alpha=0.3)

        # 统计信息
        self.ax6.set_title('Training Statistics')
        self.ax6.axis('off')

        plt.tight_layout()
        plt.draw()

    def update(self, episode, reward, distance, length, actions=None, states=None, loss=0, q_val=0):
        """更新图表"""
        try:
            # 记录数据
            self.episode_rewards.append(reward)
            self.episode_distances.append(distance)
            self.episode_lengths.append(length)

            if loss > 0:
                self.loss_history.append(loss)
            if q_val > 0:
                self.q_history.append(q_val)

            episodes = list(range(len(self.episode_rewards)))

            # 更新奖励图表
            self.reward_line.set_data(episodes, self.episode_rewards)
            self.ax1.relim()
            self.ax1.autoscale_view()

            # 更新距离图表
            self.distance_line.set_data(episodes, self.episode_distances)
            self.ax2.relim()
            self.ax2.autoscale_view()

            # 更新步数图表
            self.length_line.set_data(episodes, self.episode_lengths)
            self.ax3.relim()
            self.ax3.autoscale_view()

            # 更新动作分布
            self.ax4.clear()
            if actions is not None and len(actions) > 0:
                actions_flat = np.array(actions).flatten()
                if len(actions_flat) > 0:
                    self.ax4.hist(actions_flat, bins=20, alpha=0.7, color='blue', edgecolor='black')
                    self.ax4.set_title(f'Action Distribution (n={len(actions_flat)})')
                    self.ax4.grid(True, alpha=0.3)

            # 更新轨迹
            self.ax5.clear()
            if states is not None and len(states) > 0:
                states_arr = np.array(states)
                if states_arr.shape[1] >= 15:
                    # 提取XY位置（索引12-13是末端X,Y）
                    ee_x = states_arr[:, 12]
                    ee_y = states_arr[:, 13]

                    # 绘制轨迹
                    self.ax5.scatter(ee_x, ee_y, c=range(len(ee_x)), cmap='viridis', s=10, alpha=0.6)

                    # 标记起点和终点
                    if len(ee_x) > 0:
                        self.ax5.scatter(ee_x[0], ee_y[0], c='green', s=100, marker='o', label='Start')
                        self.ax5.scatter(ee_x[-1], ee_y[-1], c='red', s=100, marker='*', label='End')

                    # 标记目标位置（假设索引15-16是目标X,Y）
                    if states_arr.shape[1] >= 16:
                        target_x = states_arr[0, 15]
                        target_y = states_arr[0, 16]
                        self.ax5.scatter(target_x, target_y, c='orange', s=150, marker='X', label='Target')

                    self.ax5.legend()
                    self.ax5.set_title(f'Trajectory (n={len(ee_x)})')
                    self.ax5.grid(True, alpha=0.3)

            # 更新统计信息
            self.ax6.clear()
            self.ax6.axis('off')

            # 计算统计数据
            avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
            avg_distance = np.mean(self.episode_distances[-10:]) if len(self.episode_distances) >= 10 else np.mean(self.episode_distances)

            stats_text = (
                f"Episode: {episode}\n"
                f"Reward: {reward:.2f}\n"
                f"Distance: {distance:.3f}\n"
                f"Length: {length}\n"
                f"Avg Reward (last 10): {avg_reward:.2f}\n"
                f"Avg Distance (last 10): {avg_distance:.3f}\n"
                f"Exploration Rate: {1.0 * (0.995 ** episode):.3f}\n"
                f"Total Episodes: {len(self.episode_rewards)}"
            )

            self.ax6.text(0.05, 0.95, stats_text, transform=self.ax6.transAxes,
                         fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

            # 重绘
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)  # 短暂暂停

        except Exception as e:
            print(f"更新图表时出错: {e}")
            # 尝试重新初始化图表
            try:
                plt.close(self.fig)
                self.__init__()
            except:
                pass

    def save(self, filename):
        """保存图表"""
        try:
            self.fig.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"图表已保存: {filename}")
        except Exception as e:
            print(f"保存图表失败: {e}")

    def close(self):
        """关闭图表"""
        plt.close(self.fig)

# ============================ 4. 稳定的训练函数 ============================

def stable_training():
    """稳定的训练函数"""
    print("\n" + "=" * 60)
    print("开始稳定训练")
    print("=" * 60)

    # 训练参数
    total_episodes = 50
    render_episodes = [0, 10, 20, 30, 40, 49]  # 只在特定episode渲染

    # 创建环境（初始不渲染）
    print("创建环境...")
    env = RobustArmEnv(render=False)

    # 创建代理
    print("创建代理...")
    agent = SimpleAgent(env.state_dim, env.action_dim)

    # 创建图表系统
    print("创建图表系统...")
    charts = FixedChartSystem()

    # 开始训练
    print(f"\n开始训练 {total_episodes} 个episodes...")
    print("-" * 60)

    start_time = time.time()

    for episode in range(total_episodes):
        print(f"Episode {episode+1}/{total_episodes}", end="")

        # 决定是否渲染
        if episode in render_episodes:
            print(" [渲染]", end="")
            env.start_viewer()
        else:
            env.render_enabled = False

        # 重置环境
        state = env.reset()
        total_reward = 0
        episode_length = 0
        done = False

        # 运行一个episode
        while not done:
            # 选择动作
            action = agent.act(state, training=True)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 记住经验
            agent.remember(state, action, reward, next_state, done)

            # 经验回放
            loss = agent.replay()

            # 更新状态
            state = next_state
            total_reward += reward
            episode_length += 1

        # 获取最终距离
        final_distance = info.get("distance", 1.0)

        # 更新图表
        charts.update(
            episode=episode+1,
            reward=total_reward,
            distance=final_distance,
            length=episode_length,
            actions=env.actions_history,
            states=env.states_history,
            loss=loss if 'loss' in locals() else 0
        )

        # 打印进度
        print(f" | 奖励: {total_reward:6.2f} | 距离: {final_distance:.3f} | 步数: {episode_length:3d} | 探索率: {agent.epsilon:.3f}")

        # 定期保存
        if (episode + 1) % 10 == 0:
            charts.save(f"training_progress_ep{episode+1}.png")
            agent.save(f"agent_ep{episode+1}.npz")

        # 关闭渲染以节省资源
        if episode in render_episodes:
            env.close()

    # 训练结束
    training_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"训练完成!")
    print(f"总时间: {training_time:.1f}秒")
    print(f"平均奖励: {np.mean(charts.episode_rewards):.2f}")
    print(f"平均距离: {np.mean(charts.episode_distances):.3f}")
    print(f"平均步数: {np.mean(charts.episode_lengths):.2f}")
    print("=" * 60)

    # 保存最终结果
    charts.save("training_final.png")
    agent.save("agent_final.npz")

    # 询问是否测试
    print("\n是否测试训练好的模型? (y/n): ", end="")
    choice = input().strip().lower()

    if choice == 'y':
        test_agent(env, agent, charts)

    # 关闭所有资源
    env.close()
    charts.close()

    return agent

def test_agent(env, agent, charts):
    """测试代理"""
    print("\n" + "=" * 60)
    print("开始测试")
    print("=" * 60)

    # 启用渲染
    env.start_viewer()

    # 设置代理为评估模式
    agent.epsilon = 0.0

    test_episodes = 3

    for test_ep in range(test_episodes):
        print(f"\n测试 Episode {test_ep+1}/{test_episodes}")

        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done and step < 100:
            # 选择动作（无探索）
            action = agent.act(state, training=False)

            # 执行动作
            state, reward, done, info = env.step(action)

            total_reward += reward
            step += 1

            # 慢速演示
            time.sleep(0.02)

            # 显示进度
            if step % 20 == 0:
                distance = info.get("distance", 0)
                print(f"  步数: {step:3d} | 距离: {distance:.3f} | 累计奖励: {total_reward:.2f}")

        # 最终结果
        final_distance = info.get("distance", 1.0)
        print(f"  完成! 最终距离: {final_distance:.3f}, 总奖励: {total_reward:.2f}")

        time.sleep(1)

    print("\n测试完成!")
    env.close()

# ============================ 5. 主程序 ============================

def main():
    """主程序"""
    print("机械臂强化学习可视化系统")
    print("\n选项:")
    print("1. 训练新模型")
    print("2. 测试现有模型")
    print("3. 仅查看图表演示")

    choice = input("请选择 (1/2/3): ").strip()

    if choice == "1":
        # 训练模式
        print("\n开始训练模式...")
        agent = stable_training()

    elif choice == "2":
        # 测试模式
        print("\n开始测试模式...")

        # 检查模型文件
        model_file = "agent_final.npz"
        if not os.path.exists(model_file):
            print(f"模型文件 '{model_file}' 不存在!")
            print("请先训练模型或提供正确的模型文件路径")
            return

        # 创建环境
        env = RobustArmEnv(render=True)

        # 创建代理
        agent = SimpleAgent(env.state_dim, env.action_dim)

        # 加载模型
        if agent.load(model_file):
            # 创建图表系统
            charts = FixedChartSystem()

            # 测试代理
            test_agent(env, agent, charts)

            # 关闭
            env.close()
            charts.close()

    elif choice == "3":
        # 图表演示模式
        print("\n图表演示模式...")

        # 创建图表系统
        charts = FixedChartSystem()

        # 生成示例数据
        print("生成示例数据...")
        for i in range(50):
            # 模拟数据
            reward = np.random.uniform(-10, 20) + i * 0.5
            distance = max(0.01, np.random.uniform(0, 0.5) - i * 0.01)
            length = np.random.randint(50, 100)

            # 模拟动作和状态
            actions = np.random.uniform(-1, 1, (length, 6))
            states = np.random.uniform(-1, 1, (length, 18))

            # 更新图表
            charts.update(
                episode=i+1,
                reward=reward,
                distance=distance,
                length=length,
                actions=actions,
                states=states
            )

            print(f"演示 Episode {i+1}/50 | 奖励: {reward:.2f} | 距离: {distance:.3f}")
            time.sleep(0.05)

        # 保存演示图表
        charts.save("chart_demo.png")

        print("\n演示完成! 图表已保存为 'chart_demo.png'")
        print("关闭图表窗口以退出...")

        # 保持图表显示
        plt.ioff()
        plt.show()

        charts.close()

    else:
        print("无效选择!")

# ============================ 6. 程序入口 ============================

if __name__ == "__main__":
    try:
        main()
        print("\n程序执行完成!")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保清理所有资源
        plt.close('all')
        print("程序结束")