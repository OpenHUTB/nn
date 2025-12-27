#!/usr/bin/env python3
"""
双机械臂协同操作仿真系统主程序
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
from pathlib import Path
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from dual_arm_bm_model import DualArmBMModel
from cooperative_task import CooperativeTransportTask
from perception_module import DualEndEffectorPerception
from visualization import DualArmVisualizer

class DualArmSimulator:
    """双机械臂仿真器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化仿真器"""
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 60)
        print("双机械臂协同操作仿真系统")
        print("=" * 60)
        
        # 初始化各模块
        self.bm_model = DualArmBMModel(self.config['simulation']['bm_model']['kwargs'])
        self.task = CooperativeTransportTask(self.config['simulation']['task']['kwargs'])
        self.perception = DualEndEffectorPerception(self.config['simulation']['perception_modules'][0]['kwargs'])
        
        # 仿真参数
        self.dt = self.config['simulation']['run_parameters']['dt']
        self.max_steps = self.config['simulation']['task']['kwargs']['max_steps']
        
        # 数据记录
        self.states = []
        self.actions = []
        self.rewards = []
        
        # 结果目录
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "frames").mkdir(exist_ok=True)
        (self.results_dir / "videos").mkdir(exist_ok=True)
        
        print("✓ 系统初始化完成")
        print(f"✓ 时间步长: {self.dt}秒")
        print(f"✓ 最大步数: {self.max_steps}")
        print(f"✓ 结果目录: {self.results_dir.absolute()}")
    
    def reset(self):
        """重置仿真"""
        self.bm_model.reset()
        self.task.reset()
        self.perception.reset()
        
        self.states = []
        self.actions = []
        self.rewards = []
        
        # 记录初始状态
        initial_state = {
            'left_arm': self.bm_model.left_arm,
            'right_arm': self.bm_model.right_arm,
            'object': self.task.state.object_position
        }
        self.states.append(initial_state)
        
        print("✓ 仿真已重置")
    
    def step(self, left_action: np.ndarray, right_action: np.ndarray):
        """执行一步仿真"""
        # 限制动作范围
        left_action = np.clip(left_action, -1.0, 1.0)
        right_action = np.clip(right_action, -1.0, 1.0)
        
        # 更新生物力学模型
        self.bm_model.update(left_action, right_action, self.dt)
        
        # 获取当前末端位置
        left_pos = self.bm_model.left_arm.end_effector_pos
        right_pos = self.bm_model.right_arm.end_effector_pos
        
        # 更新任务状态
        reward, terminated, info = self.task.update(
            left_pos, right_pos, left_action, right_action, self.dt
        )
        
        # 获取感知观测
        observation = self.perception.get_observation(
            left_pos, right_pos, self.task.state.object_position
        )
        
        # 记录数据
        self.actions.append((left_action.copy(), right_action.copy()))
        self.rewards.append(reward)
        
        current_state = {
            'left_arm': self.bm_model.left_arm,
            'right_arm': self.bm_model.right_arm,
            'object': self.task.state.object_position.copy(),
            'observation': observation,
            'info': info
        }
        self.states.append(current_state)
        
        return observation, reward, terminated, info
    
    def run_simulation(self, policy_type: str = "sinusoidal"):
        """运行仿真"""
        print("\n" + "=" * 60)
        print("开始双机械臂协同操作仿真")
        print("=" * 60)
        
        self.reset()
        
        # 定义控制策略
        def sinusoidal_policy(step: int) -> Tuple[np.ndarray, np.ndarray]:
            """正弦波控制策略（用于演示）"""
            t = step * self.dt
            freq = 1.0  # 频率
            
            # 左臂动作
            left_action = np.array([
                0.5 * np.sin(2 * np.pi * freq * t),      # 关节1
                0.3 * np.sin(2 * np.pi * freq * t + np.pi/3),  # 关节2
                0.2 * np.sin(2 * np.pi * freq * t + 2*np.pi/3)  # 关节3
            ])
            
            # 右臂动作（相位相反）
            right_action = np.array([
                0.5 * np.sin(2 * np.pi * freq * t + np.pi),    # 关节1
                0.3 * np.sin(2 * np.pi * freq * t + np.pi + np.pi/3),  # 关节2
                0.2 * np.sin(2 * np.pi * freq * t + np.pi + 2*np.pi/3)  # 关节3
            ])
            
            return left_action, right_action
        
        def target_tracking_policy(step: int) -> Tuple[np.ndarray, np.ndarray]:
            """目标跟踪策略"""
            t = step * self.dt
            
            # 计算目标位置（随时间移动）
            target_left = self.task.target_left + np.array([
                0.1 * np.sin(2 * np.pi * 0.2 * t),
                0.0,
                0.05 * np.sin(2 * np.pi * 0.3 * t)
            ])
            
            target_right = self.task.target_right + np.array([
                -0.1 * np.sin(2 * np.pi * 0.2 * t),
                0.0,
                0.05 * np.sin(2 * np.pi * 0.3 * t)
            ])
            
            # 计算当前末端位置误差
            current_left = self.bm_model.left_arm.end_effector_pos
            current_right = self.bm_model.right_arm.end_effector_pos
            
            # PD控制器
            left_error = target_left - current_left
            right_error = target_right - current_right
            
            # 简单比例控制
            kp = 2.0
            left_action = kp * left_error[:3]  # 只取前三个分量（位置）
            right_action = kp * right_error[:3]
            
            return left_action, right_action
        
        # 选择策略
        if policy_type == "sinusoidal":
            policy = sinusoidal_policy
        elif policy_type == "tracking":
            policy = target_tracking_policy
        else:
            raise ValueError(f"未知策略类型: {policy_type}")
        
        # 运行仿真循环
        terminated = False
        step = 0
        total_reward = 0.0
        
        print(f"\n使用策略: {policy_type}")
        print(f"{'Step':>6} {'Left Pos':>20} {'Right Pos':>20} {'Reward':>10} {'Terminated':>10}")
        print("-" * 80)
        
        while not terminated and step < self.max_steps:
            # 生成动作
            left_action, right_action = policy(step)
            
            # 执行一步
            observation, reward, terminated, info = self.step(left_action, right_action)
            
            total_reward += reward
            
            # 每100步打印一次状态
            if step % 100 == 0:
                left_pos = self.bm_model.left_arm.end_effector_pos
                right_pos = self.bm_model.right_arm.end_effector_pos
                print(f"{step:6d} {str(left_pos.round(2)):>20} {str(right_pos.round(2)):>20} "
                      f"{reward:10.3f} {str(terminated):>10}")
            
            step += 1
        
        print("-" * 80)
        print(f"仿真完成!")
        print(f"总步数: {step}")
        print(f"总奖励: {total_reward:.3f}")
        print(f"是否抓取成功: {self.task.state.is_grasped}")
        print(f"物体最终位置: {self.task.state.object_position.round(3)}")
        
        return step, total_reward
    
    def analyze_results(self):
        """分析仿真结果"""
        print("\n" + "=" * 60)
        print("仿真结果分析")
        print("=" * 60)
        
        # 提取数据
        left_positions = np.array([s['left_arm'].end_effector_pos for s in self.states])
        right_positions = np.array([s['right_arm'].end_effector_pos for s in self.states])
        object_positions = np.array([s['object'] for s in self.states])
        rewards = np.array(self.rewards)
        
        # 计算统计量
        left_path_length = np.sum(np.linalg.norm(np.diff(left_positions, axis=0), axis=1))
        right_path_length = np.sum(np.linalg.norm(np.diff(right_positions, axis=0), axis=1))
        
        left_max_speed = np.max(np.linalg.norm(np.diff(left_positions, axis=0) / self.dt, axis=1))
        right_max_speed = np.max(np.linalg.norm(np.diff(right_positions, axis=0) / self.dt, axis=1))
        
        # 协同度指标
        coordination_index = self._calculate_coordination_index(left_positions, right_positions)
        
        print(f"左机械臂路径长度: {left_path_length:.3f} m")
        print(f"右机械臂路径长度: {right_path_length:.3f} m")
        print(f"左机械臂最大速度: {left_max_speed:.3f} m/s")
        print(f"右机械臂最大速度: {right_max_speed:.3f} m/s")
        print(f"协同度指标: {coordination_index:.3f}")
        print(f"平均奖励: {np.mean(rewards):.3f}")
        print(f"总奖励: {np.sum(rewards):.3f}")
        
        # 保存统计数据
        stats = {
            'left_path_length': float(left_path_length),
            'right_path_length': float(right_path_length),
            'left_max_speed': float(left_max_speed),
            'right_max_speed': float(right_max_speed),
            'coordination_index': float(coordination_index),
            'mean_reward': float(np.mean(rewards)),
            'total_reward': float(np.sum(rewards)),
            'total_steps': len(self.states),
            'success': bool(self.task.state.is_grasped),
            'final_object_position': self.task.state.object_position.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        stats_path = self.results_dir / "simulation_stats.yaml"
        with open(stats_path, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)
        
        print(f"✓ 统计数据已保存至: {stats_path}")
        
        return stats
    
    def _calculate_coordination_index(self, left_pos: np.ndarray, right_pos: np.ndarray) -> float:
        """计算协同度指标"""
        # 计算双手距离的稳定性
        distances = np.linalg.norm(left_pos - right_pos, axis=1)
        distance_std = np.std(distances)
        
        # 计算运动方向的相似性
        left_vel = np.diff(left_pos, axis=0)
        right_vel = np.diff(right_pos, axis=0)
        
        if len(left_vel) > 0:
            # 计算速度方向余弦相似度
            cos_similarities = []
            for lv, rv in zip(left_vel, right_vel):
                if np.linalg.norm(lv) > 0.001 and np.linalg.norm(rv) > 0.001:
                    cos_sim = np.dot(lv, rv) / (np.linalg.norm(lv) * np.linalg.norm(rv))
                    cos_similarities.append(cos_sim)
            
            if cos_similarities:
                mean_cos_sim = np.mean(cos_similarities)
            else:
                mean_cos_sim = 0
        else:
            mean_cos_sim = 0
        
        # 综合协同度指标（距离稳定性 + 运动相似性）
        coordination = 0.5 * (1.0 / (1.0 + distance_std)) + 0.5 * (mean_cos_sim + 1) / 2
        
        return coordination
    
    def visualize_all(self):
        """生成所有可视化结果"""
        print("\n" + "=" * 60)
        print("生成可视化结果")
        print("=" * 60)
        
        # 1. 轨迹图
        trajectory_plot_path = self.results_dir / "trajectory_plot.png"
        self.bm_model.plot_trajectory(str(trajectory_plot_path))
        
        # 2. 任务可视化
        task_viz_path = self.results_dir / "task_visualization.png"
        self.task.visualize(self.bm_model.trajectory, str(task_viz_path))
        
        # 3. 创建动画
        animation_path = self.results_dir / "videos" / "dual_arm_animation.mp4"
        self.task.create_animation(self.bm_model.trajectory, str(animation_path))
        
        # 4. 性能图表
        self._plot_performance()
        
        print(f"✓ 轨迹图: {trajectory_plot_path}")
        print(f"✓ 任务可视化: {task_viz_path}")
        print(f"✓ 动画视频: {animation_path}")
    
    def _plot_performance(self):
        """绘制性能图表"""
        fig = plt.figure(figsize=(15, 10))
        
        # 奖励曲线
        ax1 = fig.add_subplot(221)
        rewards = np.array(self.rewards)
        cumulative_rewards = np.cumsum(rewards)
        
        ax1.plot(rewards, 'b-', alpha=0.7, label='Instant Reward')
        ax1.plot(cumulative_rewards, 'r-', linewidth=2, label='Cumulative Reward')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Reward')
        ax1.set_title('Reward Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # 末端位置误差
        ax2 = fig.add_subplot(222)
        left_errors = self.task.history['left_errors']
        right_errors = self.task.history['right_errors']
        
        ax2.plot(left_errors, 'r-', label='Left Hand Error')
        ax2.plot(right_errors, 'b-', label='Right Hand Error')
        ax2.axhline(y=self.task.grasp_distance, color='g', linestyle='--', 
                   label='Grasp Threshold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Distance to Object (m)')
        ax2.set_title('Position Errors')
        ax2.legend()
        ax2.grid(True)
        
        # 关节角度
        ax3 = fig.add_subplot(223)
        left_joints = np.array([s['left_arm'].joint_positions for s in self.states])
        right_joints = np.array([s['right_arm'].joint_positions for s in self.states])
        
        for i in range(3):
            ax3.plot(left_joints[:, i], f'C{i}-', alpha=0.7, label=f'Left Joint {i+1}')
            ax3.plot(right_joints[:, i], f'C{i}--', alpha=0.7, label=f'Right Joint {i+1}')
        
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Joint Angle (rad)')
        ax3.set_title('Joint Angles Over Time')
        ax3.legend(ncol=2)
        ax3.grid(True)
        
        # 协同度分析
        ax4 = fig.add_subplot(224)
        
        # 计算双手距离
        left_pos = np.array([s['left_arm'].end_effector_pos for s in self.states])
        right_pos = np.array([s['right_arm'].end_effector_pos for s in self.states])
        hand_distances = np.linalg.norm(left_pos - right_pos, axis=1)
        
        ax4.plot(hand_distances, 'purple', linewidth=2)
        ax4.axhline(y=0.2, color='g', linestyle='--', label='Ideal Distance (0.2m)')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Distance Between Hands (m)')
        ax4.set_title('Inter-Hand Distance (Coordination Metric)')
        ax4.legend()
        ax4.grid(True)
        
        plt.suptitle('Dual-Arm Cooperative Transport Performance Analysis', fontsize=14)
        plt.tight_layout()
        
        performance_path = self.results_dir / "performance_analysis.png"
        plt.savefig(str(performance_path), dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ 性能分析图: {performance_path}")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("双机械臂协同操作仿真系统")
    print("=" * 60)
    
    try:
        # 创建仿真器
        simulator = DualArmSimulator()
        
        # 运行仿真（可选择不同策略）
        print("\n请选择控制策略:")
        print("1. 正弦波控制 (演示)")
        print("2. 目标跟踪控制")
        
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            policy_type = "sinusoidal"
        elif choice == "2":
            policy_type = "tracking"
        else:
            print("使用默认策略: 正弦波控制")
            policy_type = "sinusoidal"
        
        # 运行仿真
        steps, total_reward = simulator.run_simulation(policy_type)
        
        # 分析结果
        stats = simulator.analyze_results()
        
        # 生成可视化
        simulator.visualize_all()
        
        # 保存仿真数据
        data_path = simulator.results_dir / "simulation_data.npz"
        np.savez_compressed(
            str(data_path),
            states=simulator.states,
            actions=simulator.actions,
            rewards=simulator.rewards,
            config=simulator.config
        )
        
        print(f"\n✓ 仿真数据已保存至: {data_path}")
        
        print("\n" + "=" * 60)
        print("仿真完成!")
        print(f"总步数: {steps}")
        print(f"总奖励: {total_reward:.3f}")
        print(f"协同度: {stats['coordination_index']:.3f}")
        print(f"任务成功: {'是' if stats['success'] else '否'}")
        print("=" * 60)
        
        # 显示关键结果文件
        print("\n生成的结果文件:")
        for file in simulator.results_dir.rglob("*"):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.relative_to(simulator.results_dir)} ({size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())