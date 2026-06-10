#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双机械臂协同操作仿真系统主程序

模块名称: dual_arm_simulator.main
功能描述: 实现双机械臂协同搬运任务的完整仿真环境
核心功能:
    - 双机械臂生物力学模型仿真
    - 协同搬运任务状态管理
    - 多策略控制（正弦波/目标跟踪）
    - 结果分析与可视化

依赖库:
    - numpy: 数值计算和矩阵运算
    - matplotlib: 可视化绘图
    - pyyaml: 配置文件解析
    - pathlib: 跨平台路径处理

运行环境:
    - Python 3.7+
    - 需要自定义模块: dual_arm_bm_model, cooperative_task, 
                       perception_module, visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
from pathlib import Path
import time
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List  # 类型注解

# 添加当前目录到Python路径，确保能导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from dual_arm_bm_model import DualArmBMModel      # 双机械臂生物力学模型
from cooperative_task import CooperativeTransportTask  # 协同搬运任务
from perception_module import DualEndEffectorPerception  # 末端感知模块
from visualization import DualArmVisualizer       # 可视化工具


class DualArmSimulator:
    """
    双机械臂协同操作仿真器
    
    功能说明:
        整合机械臂模型、任务环境和感知模块，提供完整的仿真环境。
        支持多种控制策略，自动记录仿真数据并生成可视化结果。
    
    属性:
        config (dict): 仿真配置参数
        bm_model (DualArmBMModel): 双机械臂生物力学模型
        task (CooperativeTransportTask): 协同搬运任务
        perception (DualEndEffectorPerception): 末端感知模块
        dt (float): 仿真时间步长（秒）
        max_steps (int): 最大仿真步数
        states (list): 状态历史记录
        actions (list): 动作历史记录
        rewards (list): 奖励历史记录
        results_dir (Path): 结果保存目录
    
    使用示例:
        >>> simulator = DualArmSimulator('config.yaml')
        >>> simulator.reset()
        >>> for step in range(max_steps):
        ...     left_action, right_action = policy(step)
        ...     obs, reward, terminated, info = simulator.step(left_action, right_action)
        ...     if terminated: break
        >>> stats = simulator.analyze_results()
        >>> simulator.visualize_all()
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化双机械臂仿真器
        
        参数:
            config_path (str): 配置文件路径，默认为'config.yaml'
            
        初始化流程:
            1. 加载YAML配置文件
            2. 打印系统信息
            3. 初始化各功能模块
            4. 创建结果保存目录
            5. 验证配置完整性
        """
        # ----- 1. 加载配置文件 -----
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        
        # ----- 2. 打印系统标题 -----
        print("=" * 60)
        print("双机械臂协同操作仿真系统")
        print("=" * 60)
        
        # ----- 3. 初始化各模块 -----
        # 3.1 生物力学模型
        bm_kwargs = self.config['simulation']['bm_model']['kwargs']
        self.bm_model = DualArmBMModel(bm_kwargs)
        
        # 3.2 协同搬运任务
        task_kwargs = self.config['simulation']['task']['kwargs']
        self.task = CooperativeTransportTask(task_kwargs)
        
        # 3.3 末端感知模块（取第一个感知模块配置）
        perception_kwargs = self.config['simulation']['perception_modules'][0]['kwargs']
        self.perception = DualEndEffectorPerception(perception_kwargs)
        
        # ----- 4. 提取仿真参数 -----
        self.dt = self.config['simulation']['run_parameters']['dt']
        self.max_steps = self.config['simulation']['task']['kwargs']['max_steps']
        
        # ----- 5. 初始化数据记录容器 -----
        self.states = []    # 状态历史: 每个元素包含机械臂状态、物体位置等
        self.actions = []   # 动作历史: 左右臂的动作序列
        self.rewards = []   # 奖励历史: 每步的即时奖励
        
        # ----- 6. 创建结果保存目录 -----
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)                      # 主结果目录
        (self.results_dir / "frames").mkdir(exist_ok=True)        # 帧图像目录
        (self.results_dir / "videos").mkdir(exist_ok=True)        # 视频目录
        
        # ----- 7. 打印初始化完成信息 -----
        print("✓ 系统初始化完成")
        print(f"✓ 时间步长: {self.dt}秒")
        print(f"✓ 最大步数: {self.max_steps}")
        print(f"✓ 结果目录: {self.results_dir.absolute()}")
    
    def reset(self) -> None:
        """
        重置仿真环境到初始状态
        
        功能:
            重置所有模块（机械臂、任务、感知）到初始状态，
            清空历史记录，记录初始状态快照。
        
        重置内容:
            - 机械臂关节角度归零
            - 物体位置重置到初始位置
            - 清空所有历史记录
            - 记录初始状态作为第一帧
        
        使用场景:
            - 开始新的仿真实验前
            - 训练强化学习智能体的episode开始时
            - 测试不同策略前
        
        示例:
            >>> simulator.reset()
            >>> print(len(simulator.states))  # 输出: 1（初始状态）
        """
        # 重置各模块
        self.bm_model.reset()      # 重置机械臂姿态
        self.task.reset()          # 重置任务状态（物体位置、抓取标志等）
        self.perception.reset()    # 重置感知模块
        
        # 清空历史记录
        self.states = []
        self.actions = []
        self.rewards = []
        
        # 记录初始状态快照
        initial_state = {
            'left_arm': self.bm_model.left_arm,          # 左臂状态
            'right_arm': self.bm_model.right_arm,        # 右臂状态
            'object': self.task.state.object_position    # 物体位置
        }
        self.states.append(initial_state)
        
        print("✓ 仿真已重置")
    
    def step(self, left_action: np.ndarray, right_action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步仿真推进
        
        功能:
            根据输入的动作指令，更新机械臂状态、任务状态，
            计算奖励并判断是否终止。
        
        参数:
            left_action (np.ndarray): 左臂动作指令，形状(3,)，范围[-1, 1]
            right_action (np.ndarray): 右臂动作指令，形状(3,)，范围[-1, 1]
        
        返回:
            Tuple包含4个元素:
                observation (np.ndarray): 观测数据（末端位置、物体位置等）
                reward (float): 即时奖励值
                terminated (bool): 是否终止（抓取成功或失败）
                info (dict): 额外信息（如抓取状态、距离误差等）
        
        执行流程:
            1. 动作限幅（确保在安全范围内）
            2. 更新生物力学模型
            3. 获取当前末端位置
            4. 更新任务状态
            5. 获取感知观测
            6. 记录数据
            7. 返回结果
        
        示例:
            >>> left_action = np.array([0.5, 0.3, 0.2])
            >>> right_action = np.array([-0.5, -0.3, -0.2])
            >>> obs, reward, terminated, info = simulator.step(left_action, right_action)
            >>> print(f"Reward: {reward:.3f}, Terminated: {terminated}")
        """
        # ----- 1. 动作限幅（防止控制量过大）-----
        # 限制动作值在[-1.0, 1.0]范围内，对应关节角度/力度的限制
        left_action = np.clip(left_action, -1.0, 1.0)
        right_action = np.clip(right_action, -1.0, 1.0)
        
        # ----- 2. 更新生物力学模型 -----
        # 根据动作指令和时间步长更新机械臂状态
        self.bm_model.update(left_action, right_action, self.dt)
        
        # ----- 3. 获取当前末端位置 -----
        # 末端位置用于任务更新和感知
        left_pos = self.bm_model.left_arm.end_effector_pos
        right_pos = self.bm_model.right_arm.end_effector_pos
        
        # ----- 4. 更新任务状态 -----
        # 计算奖励、检查终止条件、获取任务信息
        reward, terminated, info = self.task.update(
            left_pos, right_pos, left_action, right_action, self.dt
        )
        
        # ----- 5. 获取感知观测 -----
        # 综合末端位置和物体位置，生成观测向量
        observation = self.perception.get_observation(
            left_pos, right_pos, self.task.state.object_position
        )
        
        # ----- 6. 记录数据 -----
        # 保存动作和奖励历史
        self.actions.append((left_action.copy(), right_action.copy()))
        self.rewards.append(reward)
        
        # 保存当前状态快照
        current_state = {
            'left_arm': self.bm_model.left_arm,
            'right_arm': self.bm_model.right_arm,
            'object': self.task.state.object_position.copy(),
            'observation': observation,
            'info': info
        }
        self.states.append(current_state)
        
        return observation, reward, terminated, info
    
    def run_simulation(self, policy_type: str = "sinusoidal") -> Tuple[int, float]:
        """
        运行完整的仿真过程
        
        功能:
            根据指定的控制策略，运行从重置到终止的完整仿真。
            支持多种内置策略，可用于演示和测试。
        
        参数:
            policy_type (str): 控制策略类型
                - "sinusoidal": 正弦波控制（演示用）
                - "tracking": 目标跟踪控制（PD控制器）
        
        返回:
            Tuple[int, float]: (总步数, 总奖励)
        
        执行流程:
            1. 重置仿真环境
            2. 选择控制策略
            3. 循环执行step直到终止或达到最大步数
            4. 打印仿真结果
            5. 返回统计信息
        
        示例:
            >>> steps, total_reward = simulator.run_simulation("tracking")
            >>> print(f"完成 {steps} 步，总奖励 {total_reward:.3f}")
        """
        print("\n" + "=" * 60)
        print("开始双机械臂协同操作仿真")
        print("=" * 60)
        
        # 重置环境到初始状态
        self.reset()
        
        # ----- 定义控制策略函数 -----
        
        def sinusoidal_policy(step: int) -> Tuple[np.ndarray, np.ndarray]:
            """
            正弦波控制策略（演示用）
            
            功能:
                生成正弦波形式的周期性动作，用于测试机械臂运动范围。
                左右臂相位相反，模拟协同运动。
            
            参数:
                step (int): 当前步数
            
            返回:
                Tuple[np.ndarray, np.ndarray]: (左臂动作, 右臂动作)
            
            数学公式:
                左臂: A * sin(2πft + φ)
                右臂: A * sin(2πft + π + φ)
            """
            t = step * self.dt
            freq = 1.0  # 振荡频率 (Hz)
            
            # 左臂动作：三个关节的正弦波，相位递增
            left_action = np.array([
                0.5 * np.sin(2 * np.pi * freq * t),                           # 关节1
                0.3 * np.sin(2 * np.pi * freq * t + np.pi/3),                 # 关节2（相位偏移60°）
                0.2 * np.sin(2 * np.pi * freq * t + 2*np.pi/3)                # 关节3（相位偏移120°）
            ])
            
            # 右臂动作：整体相位偏移π（180°），与左臂形成对称运动
            right_action = np.array([
                0.5 * np.sin(2 * np.pi * freq * t + np.pi),                   # 关节1
                0.3 * np.sin(2 * np.pi * freq * t + np.pi + np.pi/3),        # 关节2
                0.2 * np.sin(2 * np.pi * freq * t + np.pi + 2*np.pi/3)       # 关节3
            ])
            
            return left_action, right_action
        
        def target_tracking_policy(step: int) -> Tuple[np.ndarray, np.ndarray]:
            """
            目标跟踪控制策略（PD控制器）
            
            功能:
                根据目标位置与当前末端位置的误差，使用比例控制
                生成动作指令，使机械臂跟踪移动目标。
            
            参数:
                step (int): 当前步数
            
            返回:
                Tuple[np.ndarray, np.ndarray]: (左臂动作, 右臂动作)
            
            控制原理:
                动作 = Kp * (目标位置 - 当前位置)
                其中 Kp 为比例增益系数
            
            目标轨迹:
                目标位置在初始位置附近做正弦运动
            """
            t = step * self.dt
            
            # 生成左臂移动目标（围绕初始位置做正弦运动）
            target_left = self.task.target_left + np.array([
                0.1 * np.sin(2 * np.pi * 0.2 * t),    # X轴方向移动 ±0.1m
                0.0,                                   # Y轴方向不移动
                0.05 * np.sin(2 * np.pi * 0.3 * t)     # Z轴方向移动 ±0.05m
            ])
            
            # 生成右臂移动目标（与左臂对称运动）
            target_right = self.task.target_right + np.array([
                -0.1 * np.sin(2 * np.pi * 0.2 * t),   # X轴方向相反相位
                0.0,
                0.05 * np.sin(2 * np.pi * 0.3 * t)    # Z轴方向相同相位
            ])
            
            # 获取当前末端位置
            current_left = self.bm_model.left_arm.end_effector_pos
            current_right = self.bm_model.right_arm.end_effector_pos
            
            # 计算位置误差
            left_error = target_left - current_left
            right_error = target_right - current_right
            
            # 比例控制器（P Controller）
            kp = 2.0  # 比例增益，控制响应速度
            
            # 只取前三个分量（位置控制），转换为动作指令
            left_action = kp * left_error[:3]
            right_action = kp * right_error[:3]
            
            return left_action, right_action
        
        # ----- 选择策略 -----
        if policy_type == "sinusoidal":
            policy = sinusoidal_policy
            print(f"\n使用策略: 正弦波控制（演示模式）")
        elif policy_type == "tracking":
            policy = target_tracking_policy
            print(f"\n使用策略: 目标跟踪控制（PD控制器）")
        else:
            raise ValueError(f"未知策略类型: {policy_type}. 支持的类型: 'sinusoidal', 'tracking'")
        
        # ----- 仿真主循环 -----
        terminated = False
        step = 0
        total_reward = 0.0
        
        # 打印表头
        print(f"{'Step':>6} {'Left Pos':>20} {'Right Pos':>20} {'Reward':>10} {'Terminated':>10}")
        print("-" * 80)
        
        # 循环执行，直到终止或达到最大步数
        while not terminated and step < self.max_steps:
            # 根据策略生成动作
            left_action, right_action = policy(step)
            
            # 执行一步仿真
            observation, reward, terminated, info = self.step(left_action, right_action)
            
            # 累加总奖励
            total_reward += reward
            
            # 每100步打印一次状态（避免输出过多）
            if step % 100 == 0:
                left_pos = self.bm_model.left_arm.end_effector_pos
                right_pos = self.bm_model.right_arm.end_effector_pos
                print(f"{step:6d} {str(left_pos.round(2)):>20} {str(right_pos.round(2)):>20} "
                      f"{reward:10.3f} {str(terminated):>10}")
            
            step += 1
        
        # ----- 打印仿真结果 -----
        print("-" * 80)
        print(f"仿真完成!")
        print(f"总步数: {step}")
        print(f"总奖励: {total_reward:.3f}")
        print(f"是否抓取成功: {self.task.state.is_grasped}")
        print(f"物体最终位置: {self.task.state.object_position.round(3)}")
        
        return step, total_reward
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        分析仿真结果并计算统计指标
        
        功能:
            从记录的状态历史中提取数据，计算各种性能指标：
            路径长度、最大速度、协同度、奖励统计等。
        
        返回:
            Dict[str, Any]: 包含所有统计指标的字典
            
        统计指标:
            - left_path_length: 左臂运动路径总长度（米）
            - right_path_length: 右臂运动路径总长度（米）
            - left_max_speed: 左臂最大速度（米/秒）
            - right_max_speed: 右臂最大速度（米/秒）
            - coordination_index: 协同度指标 [0,1]，越高表示协同越好
            - mean_reward: 平均奖励
            - total_reward: 总奖励
            - total_steps: 总步数
            - success: 任务是否成功
            - final_object_position: 物体最终位置
            - timestamp: 分析时间戳
        
        协同度指标算法:
            coordination = 0.5 * (1/(1+距离标准差)) + 0.5 * (平均余弦相似度+1)/2
            其中:
                - 距离标准差: 双手间距的稳定性
                - 余弦相似度: 双手运动方向的一致性
        
        示例:
            >>> stats = simulator.analyze_results()
            >>> print(f"协同度: {stats['coordination_index']:.3f}")
            >>> print(f"成功率: {'成功' if stats['success'] else '失败'}")
        """
        print("\n" + "=" * 60)
        print("仿真结果分析")
        print("=" * 60)
        
        # ----- 提取轨迹数据 -----
        # 从状态历史中提取末端位置和物体位置
        left_positions = np.array([s['left_arm'].end_effector_pos for s in self.states])
        right_positions = np.array([s['right_arm'].end_effector_pos for s in self.states])
        object_positions = np.array([s['object'] for s in self.states])
        rewards = np.array(self.rewards)
        
        # ----- 计算路径长度 -----
        # 路径长度 = 各段位移的累加和
        left_path_length = np.sum(np.linalg.norm(np.diff(left_positions, axis=0), axis=1))
        right_path_length = np.sum(np.linalg.norm(np.diff(right_positions, axis=0), axis=1))
        
        # ----- 计算最大速度 -----
        # 速度 = 位移 / 时间步长
        left_velocities = np.diff(left_positions, axis=0) / self.dt
        right_velocities = np.diff(right_positions, axis=0) / self.dt
        
        left_max_speed = np.max(np.linalg.norm(left_velocities, axis=1))
        right_max_speed = np.max(np.linalg.norm(right_velocities, axis=1))
        
        # ----- 计算协同度指标 -----
        coordination_index = self._calculate_coordination_index(left_positions, right_positions)
        
        # ----- 打印分析结果 -----
        print(f"左机械臂路径长度: {left_path_length:.3f} m")
        print(f"右机械臂路径长度: {right_path_length:.3f} m")
        print(f"左机械臂最大速度: {left_max_speed:.3f} m/s")
        print(f"右机械臂最大速度: {right_max_speed:.3f} m/s")
        print(f"协同度指标: {coordination_index:.3f}")
        print(f"平均奖励: {np.mean(rewards):.3f}")
        print(f"总奖励: {np.sum(rewards):.3f}")
        
        # ----- 保存统计数据到YAML文件 -----
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
        
        # 保存为YAML格式，便于阅读和后续分析
        stats_path = self.results_dir / "simulation_stats.yaml"
        with open(stats_path, 'w', encoding='utf-8') as f:
            yaml.dump(stats, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ 统计数据已保存至: {stats_path}")
        
        return stats
    
    def _calculate_coordination_index(self, left_pos: np.ndarray, right_pos: np.ndarray) -> float:
        """
        计算双机械臂协同度指标
        
        功能:
            评估左右机械臂运动的协同程度，综合考虑：
            1. 双手距离的稳定性（标准差）
            2. 运动方向的一致性（余弦相似度）
        
        参数:
            left_pos (np.ndarray): 左臂末端位置历史，形状(N, 3)
            right_pos (np.ndarray): 右臂末端位置历史，形状(N, 3)
        
        返回:
            float: 协同度指标，范围[0, 1]
                - 1.0: 完美协同（理想情况）
                - 0.5: 中等协同
                - 0.0: 完全无协同
        
        算法细节:
            1. 距离稳定性 = 1 / (1 + 距离标准差)
               距离标准差越小，稳定性越高
            
            2. 方向一致性 = (平均余弦相似度 + 1) / 2
               余弦相似度范围[-1,1]，归一化到[0,1]
            
            3. 最终协同度 = 0.5 * 距离稳定性 + 0.5 * 方向一致性
        
        注意:
            需要至少2个时间步才能计算速度方向，
            如果轨迹长度不足，返回默认值0.5
        """
        # 计算双手欧氏距离序列
        distances = np.linalg.norm(left_pos - right_pos, axis=1)
        distance_std = np.std(distances)  # 标准差越小，距离越稳定
        
        # 计算运动速度方向
        left_vel = np.diff(left_pos, axis=0)   # 左臂速度向量
        right_vel = np.diff(right_pos, axis=0) # 右臂速度向量
        
        # 计算速度方向余弦相似度
        if len(left_vel) > 0:
            cos_similarities = []
            for lv, rv in zip(left_vel, right_vel):
                # 避免除以零
                if np.linalg.norm(lv) > 0.001 and np.linalg.norm(rv) > 0.001:
                    # 余弦相似度 = (A·B) / (|A|·|B|)
                    cos_sim = np.dot(lv, rv) / (np.linalg.norm(lv) * np.linalg.norm(rv))
                    cos_similarities.append(cos_sim)
            
            if cos_similarities:
                mean_cos_sim = np.mean(cos_similarities)
            else:
                mean_cos_sim = 0
        else:
            mean_cos_sim = 0
        
        # 归一化余弦相似度到[0,1]范围
        # 原始范围[-1,1] → 映射到[0,1]
        normalized_cos_sim = (mean_cos_sim + 1) / 2
        
        # 距离稳定性指标：标准差越小，稳定性越高
        # 使用 1/(1+σ) 确保结果在(0,1]范围内
        distance_stability = 1.0 / (1.0 + distance_std)
        
        # 综合协同度：距离稳定性 + 方向一致性（各占50%）
        coordination = 0.5 * distance_stability + 0.5 * normalized_cos_sim
        
        return coordination
    
    def visualize_all(self) -> None:
        """
        生成所有可视化结果
        
        功能:
            调用各模块的可视化方法，生成：
            1. 轨迹图（机械臂运动轨迹）
            2. 任务可视化（物体位置、抓取状态）
            3. 动画视频（完整运动过程）
            4. 性能图表（奖励曲线、误差曲线等）
        
        输出文件:
            - results/trajectory_plot.png
            - results/task_visualization.png
            - results/videos/dual_arm_animation.mp4
            - results/performance_analysis.png
        
        使用场景:
            - 仿真完成后自动生成报告
            - 分析机械臂运动规律
            - 制作演示视频
        """
        print("\n" + "=" * 60)
        print("生成可视化结果")
        print("=" * 60)
        
        # 1. 生成机械臂轨迹图
        trajectory_plot_path = self.results_dir / "trajectory_plot.png"
        self.bm_model.plot_trajectory(str(trajectory_plot_path))
        
        # 2. 生成任务可视化（物体、抓取点等）
        task_viz_path = self.results_dir / "task_visualization.png"
        self.task.visualize(self.bm_model.trajectory, str(task_viz_path))
        
        # 3. 创建动画视频（需要ffmpeg支持）
        animation_path = self.results_dir / "videos" / "dual_arm_animation.mp4"
        self.task.create_animation(self.bm_model.trajectory, str(animation_path))
        
        # 4. 生成性能分析图表
        self._plot_performance()
        
        # 打印输出文件位置
        print(f"✓ 轨迹图: {trajectory_plot_path}")
        print(f"✓ 任务可视化: {task_viz_path}")
        print(f"✓ 动画视频: {animation_path}")
    
    def _plot_performance(self) -> None:
        """
        绘制性能分析图表（内部方法）
        
        功能:
            创建4个子图，分别显示：
            1. 奖励曲线（即时奖励和累积奖励）
            2. 位置误差（左右手到物体的距离）
            3. 关节角度变化（6个关节的时间序列）
            4. 双手间距（协同度指标）
        
        图表特点:
            - 4个子图布局：2x2网格
            - 包含图例、网格、阈值线
            - 高分辨率输出（150 DPI）
        
        输出文件:
            results/performance_analysis.png
        """
        # 创建2x2子图布局，总尺寸15x10英寸
        fig = plt.figure(figsize=(15, 10))
        
        # ----- 子图1: 奖励曲线 -----
        ax1 = fig.add_subplot(221)
        rewards = np.array(self.rewards)
        cumulative_rewards = np.cumsum(rewards)  # 累积奖励
        
        ax1.plot(rewards, 'b-', alpha=0.7, linewidth=1, label='即时奖励')
        ax1.plot(cumulative_rewards, 'r-', linewidth=2, label='累积奖励')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # 零线
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('奖励值')
        ax1.set_title('奖励随时间变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ----- 子图2: 位置误差曲线 -----
        ax2 = fig.add_subplot(222)
        left_errors = self.task.history['left_errors']   # 左臂到物体的距离
        right_errors = self.task.history['right_errors'] # 右臂到物体的距离
        
        ax2.plot(left_errors, 'r-', linewidth=1.5, label='左臂误差')
        ax2.plot(right_errors, 'b-', linewidth=1.5, label='右臂误差')
        ax2.axhline(y=self.task.grasp_distance, color='g', linestyle='--', 
                   linewidth=1.5, label='抓取阈值')
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('到物体的距离 (m)')
        ax2.set_title('位置误差')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ----- 子图3: 关节角度曲线 -----
        ax3 = fig.add_subplot(223)
        # 提取左右臂的所有关节角度
        left_joints = np.array([s['left_arm'].joint_positions for s in self.states])
        right_joints = np.array([s['right_arm'].joint_positions for s in self.states])
        
        # 绘制左臂关节（实线）和右臂关节（虚线）
        for i in range(3):  # 假设每个臂有3个关节
            ax3.plot(left_joints[:, i], f'C{i}-', alpha=0.7, linewidth=1.5, 
                    label=f'左臂关节{i+1}')
            ax3.plot(right_joints[:, i], f'C{i}--', alpha=0.7, linewidth=1.5, 
                    label=f'右臂关节{i+1}')
        
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('关节角度 (rad)')
        ax3.set_title('关节角度变化')
        ax3.legend(ncol=2, fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # ----- 子图4: 双手间距（协同度指标）-----
        ax4 = fig.add_subplot(224)
        
        # 计算双手之间的欧氏距离
        left_pos = np.array([s['left_arm'].end_effector_pos for s in self.states])
        right_pos = np.array([s['right_arm'].end_effector_pos for s in self.states])
        hand_distances = np.linalg.norm(left_pos - right_pos, axis=1)
        
        ax4.plot(hand_distances, 'purple', linewidth=2, label='双手距离')
        ax4.axhline(y=0.2, color='g', linestyle='--', linewidth=1.5, 
                   label='理想距离 (0.2m)')
        ax4.set_xlabel('时间步')
        ax4.set_ylabel('双手间距 (m)')
        ax4.set_title('双手间距 - 协同度指标')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 添加总标题
        plt.suptitle('双机械臂协同搬运性能分析', fontsize=14, fontweight='bold')
        
        # 调整子图间距，避免重叠
        plt.tight_layout()
        
        # 保存图片
        performance_path = self.results_dir / "performance_analysis.png"
        plt.savefig(str(performance_path), dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ 性能分析图: {performance_path}")


def main() -> int:
    """
    主函数：程序入口
    
    功能:
        1. 解析命令行参数（可选）
        2. 创建仿真器实例
        3. 交互式选择控制策略
        4. 运行仿真
        5. 分析结果并生成可视化
        6. 保存仿真数据
    
    返回:
        int: 退出码
            - 0: 正常完成
            - 1: 运行出错
    
    使用示例:
        $ python main.py
        请选择控制策略:
        1. 正弦波控制 (演示)
        2. 目标跟踪控制
        请输入选择 (1 或 2): 2
    """
    print("\n" + "=" * 60)
    print("双机械臂协同操作仿真系统")
    print("=" * 60)
    
    try:
        # ----- 1. 创建仿真器实例 -----
        # 默认读取当前目录下的 config.yaml 文件
        simulator = DualArmSimulator()
        
        # ----- 2. 选择控制策略 -----
        print("\n请选择控制策略:")
        print("1. 正弦波控制 (演示模式)")
        print("2. 目标跟踪控制 (PD控制器)")
        
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            policy_type = "sinusoidal"
            print("\n已选择: 正弦波控制模式")
        elif choice == "2":
            policy_type = "tracking"
            print("\n已选择: 目标跟踪控制模式")
        else:
            print("输入无效，使用默认策略: 正弦波控制")
            policy_type = "sinusoidal"
        
        # ----- 3. 运行仿真 -----
        steps, total_reward = simulator.run_simulation(policy_type)
        
        # ----- 4. 分析结果 -----
        stats = simulator.analyze_results()
        
        # ----- 5. 生成可视化 -----
        simulator.visualize_all()
        
        # ----- 6. 保存仿真数据 -----
        # 使用压缩格式保存numpy数组，节省空间
        data_path = simulator.results_dir / "simulation_data.npz"
        np.savez_compressed(
            str(data_path),
            states=simulator.states,
            actions=simulator.actions,
            rewards=simulator.rewards,
            config=simulator.config
        )
        print(f"\n✓ 仿真数据已保存至: {data_path}")
        
        # ----- 7. 打印最终总结 -----
        print("\n" + "=" * 60)
        print("仿真完成!")
        print(f"总步数: {steps}")
        print(f"总奖励: {total_reward:.3f}")
        print(f"协同度: {stats['coordination_index']:.3f}")
        print(f"任务成功: {'✓ 是' if stats['success'] else '✗ 否'}")
        print("=" * 60)
        
        # 列出所有生成的结果文件
        print("\n生成的结果文件:")
        for file in sorted(simulator.results_dir.rglob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.relative_to(simulator.results_dir)} ({size_mb:.2f} MB)")
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        print("请确保配置文件 config.yaml 存在于当前目录")
        return 1
    except KeyError as e:
        print(f"\n❌ 配置错误: 缺少必要的配置项 {e}")
        print("请检查 config.yaml 文件的完整性")
        return 1
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    """
    程序入口点
    
    当脚本被直接运行时（而不是作为模块导入），执行 main() 函数
    """
    sys.exit(main())