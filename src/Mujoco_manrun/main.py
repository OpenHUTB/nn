# -*- coding: utf-8 -*-
"""
人形机器人行走控制器（集成深度强化学习）
功能：结合传统PID控制和PPO强化学习算法，实现人形机器人的自主行走学习
环境：MuJoCo物理仿真环境 + PyTorch深度学习框架
"""

# 导入必要的库
import mujoco  # MuJoCo物理仿真库
import numpy as np  # 数值计算库
from mujoco import viewer  # MuJoCo可视化器
import time  # 时间控制库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
from torch.distributions import Normal  # 正态分布模块
from collections import deque  # 双端队列，用于经验缓存
import os  # 操作系统接口
import matplotlib.pyplot as plt  # 绘图库


# 人形机器人行走控制器类（集成强化学习）
class HumanoidWalker:
    """
    集成深度强化学习的人形机器人行走控制器类。

    该类结合了传统PID控制和PPO（Proximal Policy Optimization）强化学习算法，
    实现人形机器人在MuJoCo环境中的自主行走学习。控制器首先使用PID实现基础
    步态跟踪，然后通过强化学习不断优化控制策略，最终实现稳定高效的行走。
    """

    def __init__(self, model_path):
        """
        类的初始化方法，在创建类的实例时自动调用。

        参数:
            model_path (str): 指向机器人模型XML文件的路径

        初始化内容:
            1. 加载MuJoCo模型和数据
            2. 设置仿真参数
            3. 初始化PID控制器参数
            4. 配置强化学习相关参数
            5. 构建神经网络模型
            6. 初始化经验缓存和训练记录
        """
        # 强制确认传入的路径是字符串类型，确保类型安全
        if not isinstance(model_path, str):
            raise TypeError(f"模型路径必须是字符串，当前是 {type(model_path)} 类型")

        # 尝试加载MuJoCo模型和数据
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)  # 加载模型
            self.data = mujoco.MjData(self.model)  # 创建数据实例
        except Exception as e:
            # 捕获模型加载异常并提供友好的错误信息
            raise RuntimeError(f"模型加载失败：{e}\n请检查：1.路径是否为字符串 2.文件是否存在 3.文件是否完整")

        # 仿真参数配置
        self.sim_duration = 20.0  # 每个episode的最大仿真时间（秒）
        self.dt = self.model.opt.timestep  # 仿真步长（秒），从模型配置中获取
        self.init_wait_time = 1.0  # 初始化等待时间，确保可视化窗口加载完成

        # PID控制器参数（用于基础步态跟踪）
        self.kp = 30.0  # 比例增益，控制当前误差的响应
        self.ki = 0.01  # 积分增益，控制累积误差的响应
        self.kd = 5.0  # 微分增益，控制误差变化率的响应
        self.joint_errors = np.zeros(self.model.nu)  # 关节误差缓存
        self.joint_integrals = np.zeros(self.model.nu)  # 关节积分项缓存

        # 强化学习参数配置
        self.use_drl = True  # 是否启用强化学习
        self.state_dim = self.model.nq + self.model.nv + 3  # 状态维度：关节位置+关节速度+质心位置
        self.action_dim = self.model.nu  # 动作维度：等于关节数量
        self.gamma = 0.99  # 折扣因子，用于计算累积奖励
        self.epsilon = 0.2  # PPO算法的裁剪系数，控制策略更新的幅度
        self.learning_rate = 3e-4  # 学习率，控制参数更新的步长

        # 构建强化学习网络模型
        self.actor = self._build_actor_network()  # 策略网络（Actor），输出动作分布
        self.critic = self._build_critic_network()  # 价值网络（Critic），评估状态价值
        self.optimizer = optim.Adam(  # Adam优化器
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate
        )

        # 经验缓存系统（使用字典结构优化数据管理）
        self.memory = {
            'states': [],  # 状态缓存
            'actions': [],  # 动作缓存
            'log_probs': [],  # 动作对数概率缓存
            'rewards': [],  # 奖励缓存
            'values': [],  # 状态价值缓存
            'dones': []  # 结束标志缓存
        }
        self.max_memory_size = 10000  # 最大缓存容量，防止内存溢出
        self.episode_rewards = []  # 记录每个episode的总奖励，用于监控训练进度
        self.best_reward = -float('inf')  # 记录最优奖励，用于保存最佳模型

        # 初始化仿真环境
        mujoco.mj_resetData(self.model, self.data)  # 重置仿真数据
        mujoco.mj_forward(self.model, self.data)  # 前向计算，获取初始状态

        # 记录初始状态信息
        self.last_x_pos = self.data.subtree_com[1][0]  # 初始质心x坐标，用于计算前进奖励

        # 打印初始化信息，便于调试和监控
        print(f"=== 人形机器人行走控制器初始化完成 ===")
        print(f"状态维度: {self.state_dim}")
        print(f"动作维度: {self.action_dim}")
        print(f"关节数量: {self.model.nu}")
        print(f"仿真步长: {self.dt:.6f}秒")
        print(f"强化学习模式: {'启用' if self.use_drl else '禁用'}")

    def _build_actor_network(self):
        """
        构建策略网络（Actor Network）。

        策略网络接收环境状态作为输入，输出动作的概率分布参数（均值和标准差）。
        采用两层全连接网络结构，使用Tanh激活函数引入非线性。

        返回:
            nn.Sequential: 构建好的策略网络
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),  # 输入层：状态维度 -> 64维隐藏层
            nn.Tanh(),  # 激活函数，引入非线性
            nn.Linear(64, 64),  # 隐藏层：64 -> 64
            nn.Tanh(),  # 激活函数
            nn.Linear(64, self.action_dim * 2)  # 输出层：64 -> 2*动作维度（均值+标准差）
        )

    def _build_critic_network(self):
        """
        构建价值网络（Critic Network）。

        价值网络接收环境状态作为输入，输出该状态的价值估计。
        采用与策略网络相同的网络结构，但输出为标量值。

        返回:
            nn.Sequential: 构建好的价值网络
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),  # 输入层：状态维度 -> 64维隐藏层
            nn.Tanh(),  # 激活函数
            nn.Linear(64, 64),  # 隐藏层：64 -> 64
            nn.Tanh(),  # 激活函数
            nn.Linear(64, 1)  # 输出层：64 -> 1（状态价值）
        )

    def get_state(self):
        """
        获取当前环境状态，用于强化学习。

        状态向量包含：
            1. 所有关节的位置（self.data.qpos）
            2. 所有关节的速度（self.data.qvel）
            3. 质心位置（self.data.subtree_com[1]）

        返回:
            np.ndarray: 归一化的状态向量，形状为(self.state_dim,)
        """
        # 获取原始状态数据
        qpos = self.data.qpos.copy()  # 关节位置
        qvel = self.data.qvel.copy()  # 关节速度
        com = self.data.subtree_com[1].copy()  # 质心位置（第二个刚体，通常是躯干）

        # 创建固定长度的状态向量，确保维度一致性
        state = np.zeros(self.state_dim)

        # 填充数据，防止数组越界
        qpos_dim = min(len(qpos), self.model.nq)  # 实际关节位置维度
        qvel_dim = min(len(qvel), self.model.nv)  # 实际关节速度维度

        # 填充关节位置
        state[:qpos_dim] = qpos[:qpos_dim]
        # 填充关节速度
        state[self.model.nq:self.model.nq + qvel_dim] = qvel[:qvel_dim]
        # 填充质心位置（x,y,z）
        state[self.model.nq + self.model.nv:self.model.nq + self.model.nv + 3] = com[:3]

        # 状态归一化，将值限制在[-5,5]并缩放到[-1,1]，提高训练稳定性
        state = np.clip(state, -5, 5) / 5.0

        # 转换为float32类型，节省内存并提高计算效率
        return state.astype(np.float32)

    def store_experience(self, state, action, log_prob, reward, value, done):
        """
        存储经验数据到缓存中，用于后续的策略更新。

        参数:
            state (np.ndarray): 当前状态
            action (np.ndarray): 执行的动作
            log_prob (float): 动作的对数概率
            reward (float): 获得的奖励
            value (float): 状态价值估计
            done (bool): 是否结束标志
        """
        # 检查缓存大小，超过最大值时移除最旧的数据
        if len(self.memory['states']) >= self.max_memory_size:
            for key in self.memory.keys():
                del self.memory[key][0]

        # 存储数据，确保动作维度正确
        self.memory['states'].append(state)
        self.memory['actions'].append(action[:self.action_dim] if len(action) > self.action_dim else action)
        self.memory['log_probs'].append(float(log_prob))
        self.memory['rewards'].append(float(reward))
        self.memory['values'].append(float(value))
        self.memory['dones'].append(bool(done))

    def get_reward(self):
        """
        计算当前步骤的奖励值，引导机器人学习行走策略。

        奖励函数设计包含多个部分：
            1. 前进奖励：鼓励机器人向前移动
            2. 直立奖励：鼓励机器人保持直立姿态
            3. 高度奖励：鼓励机器人保持适当的高度
            4. 能耗惩罚：惩罚过大的控制输出，鼓励节能
            5. 摔倒惩罚：惩罚摔倒行为

        返回:
            float: 综合奖励值
        """
        # 1. 前进奖励：基于质心x方向的位移速度
        current_x = self.data.subtree_com[1][0]
        forward_reward = (current_x - self.last_x_pos) / self.dt * 10  # 乘以系数放大奖励
        self.last_x_pos = current_x  # 更新最后位置

        # 2. 直立奖励：基于躯干的直立程度（xmat[1,2]是躯干z轴方向与世界z轴的点积）
        torso_up = self.data.xmat[1, 2]
        upright_reward = torso_up * 5

        # 3. 高度奖励：基于质心高度，鼓励保持适当高度
        height = self.data.subtree_com[1][2]
        height_reward = max(0, height - 0.8) * 5

        # 4. 能耗惩罚：惩罚过大的控制信号，避免动作过于剧烈
        control_cost = -np.sum(np.square(self.data.ctrl)) * 0.01

        # 5. 摔倒惩罚：当质心高度过低时给予大惩罚
        fall_penalty = -50 if height < 0.6 else 0

        # 综合奖励并限制范围，防止梯度爆炸
        total_reward = forward_reward + upright_reward + height_reward + control_cost + fall_penalty
        return np.clip(total_reward, -50, 50)

    def get_gae(self, rewards, values, dones):
        """
        计算广义优势估计（Generalized Advantage Estimation, GAE）。

        GAE能够有效平衡优势估计的偏差和方差，提高策略梯度估计的稳定性。

        参数:
            rewards (list): 奖励序列
            values (list): 价值估计序列
            dones (list): 结束标志序列

        返回:
            tuple: (advantages, returns)
                advantages: 优势估计序列
                returns: 折扣奖励序列
        """
        advantages = []
        advantage = 0
        next_value = 0

        # 确保所有输入长度相同，避免维度错误
        min_length = min(len(rewards), len(values), len(dones))
        rewards = rewards[:min_length]
        values = values[:min_length]
        dones = dones[:min_length]

        # 从后向前计算优势估计
        for t in reversed(range(len(rewards))):
            # 获取下一个状态的价值（如果是最后一步则为0）
            if t + 1 < len(values):
                next_value = values[t + 1]
            else:
                next_value = 0

            # 计算TD误差
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            # 累积优势
            advantage = delta + self.gamma * 0.95 * (1 - dones[t]) * advantage
            # 插入到优势列表的开头（保持时间顺序）
            advantages.insert(0, advantage)

        # 确保advantages和values长度匹配
        advantages = advantages[:len(values)]
        # 计算折扣奖励（目标价值）
        returns = np.array(advantages) + np.array(values)
        # 标准化优势估计，提高训练稳定性
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages, returns

    def update_policy(self):
        """
        使用PPO算法更新策略网络和价值网络。

        PPO算法通过裁剪目标函数来限制策略更新的幅度，保证训练的稳定性。
        主要步骤：
            1. 从缓存中提取经验数据
            2. 计算优势估计和折扣奖励
            3. 执行多轮策略更新
            4. 优化价值网络
        """
        # 检查缓存中是否有足够的数据
        n_samples = min(len(self.memory['states']), len(self.memory['actions']),
                        len(self.memory['log_probs']), len(self.memory['rewards']),
                        len(self.memory['values']), len(self.memory['dones']))

        if n_samples < 64:  # 数据量不足时跳过更新
            return

        # 将列表转换为numpy数组，提高计算效率并消除张量创建警告
        states = np.array(self.memory['states'][:n_samples], dtype=np.float32)
        actions = np.array(self.memory['actions'][:n_samples], dtype=np.float32)
        old_log_probs = np.array(self.memory['log_probs'][:n_samples], dtype=np.float32)
        rewards = np.array(self.memory['rewards'][:n_samples], dtype=np.float32)
        values = np.array(self.memory['values'][:n_samples], dtype=np.float32)
        dones = np.array(self.memory['dones'][:n_samples], dtype=np.bool_)

        # 确保actions维度正确
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
        if actions.shape[1] > self.action_dim:
            actions = actions[:, :self.action_dim]

        # 转换为PyTorch张量
        states_tensor = torch.from_numpy(states)
        actions_tensor = torch.from_numpy(actions)
        old_log_probs_tensor = torch.from_numpy(old_log_probs)

        # 计算优势估计和折扣奖励
        advantages, returns = self.get_gae(rewards.tolist(), values.tolist(), dones.tolist())

        # 确保形状匹配
        advantages = advantages[:len(returns)]
        advantages_tensor = torch.from_numpy(np.array(advantages, dtype=np.float32))
        returns_tensor = torch.from_numpy(np.array(returns, dtype=np.float32))

        # 执行多轮策略更新，提高收敛性
        for _ in range(10):
            # 前向传播：获取策略输出
            actor_output = self.actor(states_tensor)
            mean, log_std = actor_output.chunk(2, dim=-1)  # 将输出分为均值和对数标准差
            std = log_std.exp()  # 转换为标准差

            # 创建正态分布并计算新的动作对数概率
            dist = Normal(mean, std)

            # 确保动作维度匹配
            if actions_tensor.shape[1] > self.action_dim:
                actions_tensor = actions_tensor[:, :self.action_dim]

            new_log_probs = dist.log_prob(actions_tensor).sum(-1)  # 对所有动作维度求和

            # 确保形状匹配
            if len(new_log_probs) != len(old_log_probs_tensor):
                min_len = min(len(new_log_probs), len(old_log_probs_tensor))
                new_log_probs = new_log_probs[:min_len]
                old_log_probs_tensor = old_log_probs_tensor[:min_len]
                advantages_tensor = advantages_tensor[:min_len]

            # 计算概率比率：新策略概率 / 旧策略概率
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            # PPO裁剪损失计算
            surr1 = ratio * advantages_tensor  # 未裁剪的目标
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages_tensor  # 裁剪后的目标
            policy_loss = -torch.min(surr1, surr2).mean()  # 取最小值并取负（梯度上升）

            # 价值损失计算（均方误差）
            value_pred = self.critic(states_tensor).squeeze()  # 获取价值估计并压缩维度

            # 确保价值预测形状匹配
            if len(value_pred) != len(returns_tensor):
                min_len = min(len(value_pred), len(returns_tensor))
                value_pred = value_pred[:min_len]
                returns_tensor = returns_tensor[:min_len]

            value_loss = nn.MSELoss()(value_pred, returns_tensor)

            # 总损失 = 策略损失 + 0.5 * 价值损失
            total_loss = policy_loss + 0.5 * value_loss

            # 反向传播和优化
            self.optimizer.zero_grad()  # 清零梯度
            total_loss.backward()  # 反向传播计算梯度
            self.optimizer.step()  # 更新参数

        # 清空缓存，为下一轮收集经验做准备
        for key in self.memory.keys():
            self.memory[key].clear()

    def get_drl_action(self, state):
        """
        使用策略网络获取动作。

        参数:
            state (np.ndarray): 当前环境状态

        返回:
            tuple: (action, log_prob, value)
                action: 输出的动作
                log_prob: 动作的对数概率
                value: 状态的价值估计
        """
        # 确保状态维度正确
        if len(state) != self.state_dim:
            state = self.get_state()

        # 将状态转换为张量并添加批次维度
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # 前向传播获取策略输出
        actor_output = self.actor(state_tensor)
        mean, log_std = actor_output.chunk(2, dim=-1)
        std = log_std.exp()

        # 创建正态分布并采样动作
        dist = Normal(mean, std)
        action = dist.sample()

        # 裁剪动作范围，确保在合理范围内
        action = torch.clamp(action, -1, 1)

        # 确保动作维度正确
        if action.shape[-1] > self.action_dim:
            action = action[..., :self.action_dim]

        # 计算动作的对数概率和状态价值
        log_prob = dist.log_prob(action).sum().item()
        value = self.critic(state_tensor).item()

        # 返回numpy数组格式的动作
        return action.detach().cpu().numpy()[0], log_prob, value

    def get_gait_trajectory(self, t):
        """
        生成预设的步态轨迹，作为PID控制的目标。

        参数:
            t (float): 当前仿真时间

        返回:
            np.ndarray: 各关节的目标位置
        """
        # 初始阶段保持静止
        if t < 2.0:
            return np.zeros(self.model.nu)

        # 计算步态相位
        t_adjusted = t - 2.0  # 跳过初始静止阶段
        cycle = t_adjusted % 1.5  # 步态周期（1.5秒）
        phase = 2 * np.pi * cycle / 1.5  # 相位角（0-2π）

        # 步态参数
        leg_amp = 0.3  # 腿部关节振幅
        arm_amp = 0.2  # 手臂关节振幅
        torso_amp = 0.05  # 躯干关节振幅

        # 初始化目标位置数组
        target = np.zeros(self.model.nu)

        # 设置腿部关节目标位置（基于正弦函数的周期性运动）
        leg_joint_offset = 5  # 腿部关节起始索引
        if len(target) > leg_joint_offset + 5:
            # 右腿关节
            target[leg_joint_offset] = -leg_amp * np.sin(phase)  # 髋关节俯仰
            target[leg_joint_offset + 1] = leg_amp * 1.5 * np.sin(phase + np.pi)  # 膝关节
            target[leg_joint_offset + 2] = leg_amp * 0.5 * np.sin(phase)  # 踝关节

            # 左腿关节（与右腿反相）
            target[leg_joint_offset + 3] = -leg_amp * np.sin(phase + np.pi)
            target[leg_joint_offset + 4] = leg_amp * 1.5 * np.sin(phase)
            target[leg_joint_offset + 5] = leg_amp * 0.5 * np.sin(phase + np.pi)

        # 设置躯干关节目标位置
        if len(target) > 0:
            target[0] = torso_amp * np.sin(phase + np.pi / 2)

        # 设置手臂关节目标位置（与腿部协调运动）
        if len(target) > 16:
            target[16] = arm_amp * np.sin(phase + np.pi)  # 右臂
        if len(target) > 20:
            target[20] = arm_amp * np.sin(phase)  # 左臂

        return target

    def pid_controller(self, target_pos):
        """
        PID控制器，用于跟踪预设的步态轨迹。

        参数:
            target_pos (np.ndarray): 关节目标位置

        返回:
            np.ndarray: 输出的控制扭矩
        """
        # 获取当前关节位置（跳过前7个全局位置）
        current_pos = self.data.qpos[7:] if len(self.data.qpos) > 7 else self.data.qpos.copy()

        # 确保当前位置和目标位置维度匹配
        if len(current_pos) < len(target_pos):
            current_pos = np.pad(current_pos, (0, len(target_pos) - len(current_pos)))
        elif len(current_pos) > len(target_pos):
            current_pos = current_pos[:len(target_pos)]

        # 计算误差
        error = target_pos - current_pos

        # 积分项计算（带饱和防止积分爆炸）
        self.joint_integrals += error * self.dt
        self.joint_integrals = np.clip(self.joint_integrals, -2.0, 2.0)

        # 微分项计算
        derivative = (error - self.joint_errors) / self.dt if self.dt != 0 else 0
        self.joint_errors = error.copy()  # 更新误差缓存

        # PID输出计算
        torque = self.kp * error + self.ki * self.joint_integrals + self.kd * derivative

        # 限制输出范围，防止扭矩过大
        return np.clip(torque, -5.0, 5.0)

    def simulate_with_learning(self):
        """
        启动带强化学习的仿真主循环。

        主循环流程:
            1. 初始化可视化窗口
            2. 循环执行episodes:
                a. 重置环境
                b. 执行仿真步骤直到结束
                c. 收集经验数据
                d. 更新策略网络
                e. 记录和可视化训练进度
        """
        episode = 0  # 初始化episode计数器

        # 启动MuJoCo可视化器
        with viewer.launch_passive(self.model, self.data) as v:
            # 打印启动信息
            print("\n=== 人形机器人行走仿真启动 ===")
            print("可视化窗口已启动（集成强化学习的行走控制器）")
            print(f"强化学习模式：{'开启' if self.use_drl else '关闭'}")
            print("操作说明：")
            print("  - 鼠标拖动：旋转视角")
            print("  - 滚轮缩放：缩放视角")
            print("  - W/A/S/D：平移视角")
            print("  - 关闭窗口：结束仿真")
            print("=" * 50)

            # 初始等待，确保可视化窗口完全加载
            start_time = time.time()
            while time.time() - start_time < self.init_wait_time:
                v.sync()  # 更新可视化
                time.sleep(0.01)

            # 主循环
            while True:
                # 重置环境状态
                mujoco.mj_resetData(self.model, self.data)  # 重置仿真数据
                mujoco.mj_forward(self.model, self.data)  # 前向计算获取初始状态
                self.last_x_pos = self.data.subtree_com[1][0]  # 重置质心位置记录
                self.joint_errors = np.zeros(self.model.nu)  # 重置PID误差
                self.joint_integrals = np.zeros(self.model.nu)  # 重置PID积分项

                episode_reward = 0  # 初始化episode奖励
                done = False  # 结束标志
                step = 0  # 步数计数器

                # 单episode内的仿真循环
                while not done and self.data.time < self.sim_duration:
                    # 获取当前状态
                    state = self.get_state()

                    # 获取目标步态轨迹
                    target_pos = self.get_gait_trajectory(self.data.time)

                    # 混合控制策略
                    if self.use_drl and np.random.random() > 0.1:  # 90%使用策略，10%探索
                        try:
                            # 使用强化学习获取动作
                            drl_action, log_prob, value = self.get_drl_action(state)

                            # 获取PID控制输出
                            pid_torque = self.pid_controller(target_pos)

                            # 确保动作维度匹配
                            if len(drl_action) != len(pid_torque):
                                drl_action = np.zeros(len(pid_torque))

                            # 缩放DRL输出并与PID输出融合（70% PID + 30% DRL）
                            drl_torque = drl_action * 5.0  # 缩放DRL输出到合适范围
                            control_torques = pid_torque * 0.7 + drl_torque * 0.3
                        except Exception as e:
                            # 出错时降级到纯PID控制
                            print(f"\nDRL动作获取错误：{e}")
                            control_torques = self.pid_controller(target_pos)
                            log_prob = value = 0
                    else:
                        # 纯PID控制（探索或禁用DRL时）
                        control_torques = self.pid_controller(target_pos)
                        log_prob = value = 0

                    # 执行控制动作
                    self.data.ctrl[:] = control_torques[:len(self.data.ctrl)]
                    mujoco.mj_step(self.model, self.data)  # 执行仿真步

                    # 计算奖励
                    reward = self.get_reward()
                    episode_reward += reward

                    # 检查是否摔倒（结束条件）
                    done = self.data.subtree_com[1][2] < 0.6  # 质心高度过低

                    # 存储经验数据（如果启用DRL）
                    if self.use_drl:
                        self.store_experience(state, control_torques, log_prob, reward, value, done)

                    # 更新可视化
                    v.sync()
                    time.sleep(0.001)  # 微小延迟，避免过快
                    step += 1

                # 更新episode计数器
                episode += 1
                self.episode_rewards.append(episode_reward)  # 记录总奖励

                # 定期更新策略网络
                if self.use_drl and episode % 5 == 0:
                    try:
                        self.update_policy()

                        # 保存最佳模型
                        avg_reward = np.mean(self.episode_rewards[-100:]) if len(
                            self.episode_rewards) > 100 else episode_reward
                        if avg_reward > self.best_reward:
                            self.best_reward = avg_reward
                            torch.save({
                                'actor_state_dict': self.actor.state_dict(),
                                'critic_state_dict': self.critic.state_dict(),
                                'episode': episode,
                                'best_reward': self.best_reward,
                            }, 'best_walker_model.pth')
                            print(f"\n保存最佳模型！Episode: {episode}, 平均奖励: {avg_reward:.2f}")
                    except Exception as e:
                        print(f"\n策略更新错误：{e}")

                # 打印训练进度
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) > 10 else episode_reward
                print(
                    f"Episode {episode:3d} | 总奖励: {episode_reward:6.2f} | 最近平均: {avg_reward:6.2f} | 步数: {step:4d} | 时间: {self.data.time:.2f}s")

                # 定期绘制奖励曲线
                if episode % 20 == 0 and self.use_drl and len(self.episode_rewards) > 0:
                    plt.figure(figsize=(10, 5))
                    plt.plot(self.episode_rewards, alpha=0.5, label='Episode Reward')

                    # 绘制移动平均线，更清晰地显示趋势
                    if len(self.episode_rewards) > 20:
                        moving_avg = np.convolve(self.episode_rewards, np.ones(20) / 20, mode='valid')
                        plt.plot(range(10, len(self.episode_rewards) - 9), moving_avg, 'r-', label='20-Episode Average')

                    plt.xlabel('Episode')
                    plt.ylabel('Reward')
                    plt.title('Training Progress - Humanoid Walker')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig('training_progress.png')
                    plt.close()


# 主程序入口
if __name__ == "__main__":
    """
    主程序执行流程：
        1. 获取当前脚本目录
        2. 构造模型文件路径
        3. 检查文件是否存在
        4. 创建控制器实例并启动仿真
    """
    # 获取当前脚本所在目录
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # 构造模型文件路径（假设模型文件名为"humanoid.xml"）
    model_file_path = os.path.join(current_directory, "humanoid.xml")

    # 打印路径信息，便于调试
    print(f"当前脚本所在目录：{current_directory}")
    print(f"模型文件完整路径：{model_file_path}")

    # 检查模型文件是否存在
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(
            f"模型文件不存在！\n查找路径：{model_file_path}\n"
            f"请确认 'humanoid.xml' 文件放在以下目录中：{current_directory}"
        )

    # 创建控制器实例并启动仿真
    try:
        # 创建人形机器人行走控制器实例
        walker = HumanoidWalker(model_file_path)

        # 启动仿真和学习
        walker.simulate_with_learning()

    except Exception as e:
        # 捕获并打印异常信息
        print(f"\n仿真过程中发生错误：{e}")
        import traceback

        traceback.print_exc()  # 打印详细的堆栈跟踪信息