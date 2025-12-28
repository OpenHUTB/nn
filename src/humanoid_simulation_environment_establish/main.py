import numpy as np
import mujoco
from mujoco import viewer
import time
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import deque
import pickle
import os


class DeepLearningController:
    """深度学习控制器：使用神经网络学习最优步态和姿态控制"""
    
    def __init__(self, action_dim, state_dim, actuator_indices=None, learning_rate=0.001):
        """
        Args:
            action_dim: 动作维度（执行器数量）
            state_dim: 状态维度（观测空间大小）
            actuator_indices: 执行器名称到索引的映射
            learning_rate: 学习率
        """
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.actuator_indices = actuator_indices or {}
        self.learning_rate = learning_rate
        
        # 策略网络：根据状态预测动作
        self.policy_network = self._build_policy_network()
        
        # 价值网络：评估状态价值（用于强化学习）
        self.value_network = self._build_value_network()
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        
        # 训练相关
        self.training_enabled = True
        self.update_frequency = 10  # 每10步更新一次
        self.step_count = 0
        
        # 历史状态和动作（用于时序学习）
        self.state_history = deque(maxlen=10)
        self.action_history = deque(maxlen=10)
        
        # 步态学习参数
        self.gait_phase = 0.0  # 步态相位
        self.gait_frequency = 1.2  # 步频
        
        print(f"[深度学习控制器] 初始化完成: 动作维度={action_dim}, 状态维度={state_dim}")
    
    def _build_policy_network(self):
        """构建策略网络（MLP + LSTM混合）"""
        # 输入：状态 + 步态相位编码（sin, cos）+ 历史动作
        input_dim = self.state_dim + 2 + self.action_dim  # 状态 + 相位编码(2维) + 上次动作
        hidden1_dim = 128
        hidden2_dim = 64
        lstm_dim = 32
        output_dim = self.action_dim
        
        # 初始化权重（使用Xavier初始化）
        np.random.seed(42)
        
        # 第一层MLP
        self.policy_w1 = np.random.randn(input_dim, hidden1_dim) * np.sqrt(2.0 / input_dim)
        self.policy_b1 = np.zeros(hidden1_dim)
        
        # 第二层MLP
        self.policy_w2 = np.random.randn(hidden1_dim, hidden2_dim) * np.sqrt(2.0 / hidden1_dim)
        self.policy_b2 = np.zeros(hidden2_dim)
        
        # LSTM层（简化版：只保留隐藏状态）
        self.policy_lstm_h = np.zeros(lstm_dim)
        self.policy_lstm_c = np.zeros(lstm_dim)
        # w_lstm需要分成两部分：forget_gate和input_gate，所以需要2*lstm_dim列
        self.policy_w_lstm = np.random.randn(hidden2_dim + lstm_dim, 2 * lstm_dim) * 0.1
        self.policy_w_lstm_out = np.random.randn(hidden2_dim + lstm_dim, lstm_dim) * 0.1
        
        # 输出层
        self.policy_w3 = np.random.randn(lstm_dim, output_dim) * np.sqrt(2.0 / lstm_dim)
        self.policy_b3 = np.zeros(output_dim)
        
        return {
            'w1': self.policy_w1, 'b1': self.policy_b1,
            'w2': self.policy_w2, 'b2': self.policy_b2,
            'w3': self.policy_w3, 'b3': self.policy_b3,
            'lstm_h': self.policy_lstm_h, 'lstm_c': self.policy_lstm_c,
            'w_lstm': self.policy_w_lstm, 'w_lstm_out': self.policy_w_lstm_out
        }
    
    def _build_value_network(self):
        """构建价值网络（评估状态价值）"""
        input_dim = self.state_dim
        hidden1_dim = 64
        hidden2_dim = 32
        output_dim = 1
        
        np.random.seed(43)
        
        self.value_w1 = np.random.randn(input_dim, hidden1_dim) * np.sqrt(2.0 / input_dim)
        self.value_b1 = np.zeros(hidden1_dim)
        self.value_w2 = np.random.randn(hidden1_dim, hidden2_dim) * np.sqrt(2.0 / hidden1_dim)
        self.value_b2 = np.zeros(hidden2_dim)
        self.value_w3 = np.random.randn(hidden2_dim, output_dim) * np.sqrt(2.0 / hidden2_dim)
        self.value_b3 = np.zeros(output_dim)
        
        return {
            'w1': self.value_w1, 'b1': self.value_b1,
            'w2': self.value_w2, 'b2': self.value_b2,
            'w3': self.value_w3, 'b3': self.value_b3
        }
    
    def _relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _tanh(self, x):
        """Tanh激活函数"""
        return np.tanh(x)
    
    def _sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def predict_action(self, state, gait_phase, last_action, command=None):
        """
        预测动作
        
        Args:
            state: 当前状态（观测）
            gait_phase: 步态相位 [0, 2π]
            last_action: 上次动作
            command: 用户命令 (forward, backward, turn_left, turn_right)
        
        Returns:
            预测的动作
        """
        # 构建输入
        if last_action is None:
            last_action = np.zeros(self.action_dim)
        
        # 归一化状态（防止数值过大）
        state_normalized = np.tanh(state / 10.0)  # 简单归一化
        
        # 确保状态维度匹配
        if len(state_normalized) > self.state_dim:
            state_normalized = state_normalized[:self.state_dim]
        elif len(state_normalized) < self.state_dim:
            state_normalized = np.pad(state_normalized, (0, self.state_dim - len(state_normalized)))
        
        # 确保动作维度匹配
        if len(last_action) > self.action_dim:
            last_action = last_action[:self.action_dim]
        elif len(last_action) < self.action_dim:
            last_action = np.pad(last_action, (0, self.action_dim - len(last_action)))
        
        # 构建输入向量：状态 + 相位编码 + 上次动作
        input_vec = np.concatenate([
            state_normalized,
            [np.sin(gait_phase), np.cos(gait_phase)],  # 相位编码（2维）
            last_action
        ])
        
        # 目标维度：state_dim + 2 + action_dim
        target_dim = self.state_dim + 2 + self.action_dim
        if len(input_vec) != target_dim:
            # 如果维度不匹配，调整
            if len(input_vec) < target_dim:
                input_vec = np.pad(input_vec, (0, target_dim - len(input_vec)))
            else:
                input_vec = input_vec[:target_dim]
        
        # 前向传播
        # 第一层
        h1 = self._relu(input_vec @ self.policy_network['w1'] + self.policy_network['b1'])
        
        # 第二层
        h2 = self._relu(h1 @ self.policy_network['w2'] + self.policy_network['b2'])
        
        # 简化的LSTM更新
        lstm_input = np.concatenate([h2, self.policy_network['lstm_h']])
        forget_gate = self._sigmoid(lstm_input @ self.policy_network['w_lstm'][:, :self.policy_network['lstm_h'].shape[0]])
        input_gate = self._sigmoid(lstm_input @ self.policy_network['w_lstm'][:, self.policy_network['lstm_h'].shape[0]:])
        
        # 更新LSTM状态
        new_c = forget_gate * self.policy_network['lstm_c'] + input_gate * np.tanh(lstm_input @ self.policy_network['w_lstm_out'])
        new_h = self._tanh(new_c)
        
        self.policy_network['lstm_h'] = new_h
        self.policy_network['lstm_c'] = new_c
        
        # 输出层
        output = self._tanh(new_h @ self.policy_network['w3'] + self.policy_network['b3'])
        
        # 根据用户命令调整动作
        if command is not None:
            output = self._apply_command(output, command)
        
        return np.clip(output, -1.0, 1.0)
    
    def _apply_command(self, action, command):
        """根据用户命令调整动作"""
        if not self.actuator_indices:
            return action
        
        # 解析命令
        forward = command.get('forward', False)
        backward = command.get('backward', False)
        turn_left = command.get('turn_left', False)
        turn_right = command.get('turn_right', False)
        
        # 调整髋关节前后摆动（前进/后退）
        if forward or backward:
            direction = 1.0 if forward else -1.0
            hip_x_right_idx = self.actuator_indices.get("hip_x_right")
            hip_x_left_idx = self.actuator_indices.get("hip_x_left")
            
            if hip_x_right_idx is not None:
                # 增强前进/后退动作
                action[hip_x_right_idx] = np.clip(action[hip_x_right_idx] + 0.3 * direction, -1.0, 1.0)
            if hip_x_left_idx is not None:
                # 左腿相反方向
                action[hip_x_left_idx] = np.clip(action[hip_x_left_idx] - 0.3 * direction, -1.0, 1.0)
        
        # 调整转向
        if turn_left or turn_right:
            turn_dir = -1.0 if turn_left else 1.0
            hip_z_right_idx = self.actuator_indices.get("hip_z_right")
            hip_z_left_idx = self.actuator_indices.get("hip_z_left")
            abdomen_z_idx = self.actuator_indices.get("abdomen_z")
            
            if hip_z_right_idx is not None:
                action[hip_z_right_idx] = np.clip(action[hip_z_right_idx] + 0.2 * turn_dir, -1.0, 1.0)
            if hip_z_left_idx is not None:
                action[hip_z_left_idx] = np.clip(action[hip_z_left_idx] - 0.2 * turn_dir, -1.0, 1.0)
            if abdomen_z_idx is not None:
                action[abdomen_z_idx] = np.clip(action[abdomen_z_idx] + 0.3 * turn_dir, -1.0, 1.0)
        
        return action
    
    def predict_value(self, state):
        """预测状态价值"""
        state_normalized = np.tanh(state / 10.0)
        
        # 确保维度匹配
        if len(state_normalized) > self.state_dim:
            state_normalized = state_normalized[:self.state_dim]
        elif len(state_normalized) < self.state_dim:
            state_normalized = np.pad(state_normalized, (0, self.state_dim - len(state_normalized)))
        
        h1 = self._relu(state_normalized @ self.value_network['w1'] + self.value_network['b1'])
        h2 = self._relu(h1 @ self.value_network['w2'] + self.value_network['b2'])
        value = h2 @ self.value_network['w3'] + self.value_network['b3']
        
        return value[0]
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.replay_buffer.append({
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_state': next_state.copy() if next_state is not None else None,
            'done': done
        })
    
    def train_step(self):
        """执行一步训练（使用经验回放和策略梯度）"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 采样批次
        batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        # 简化的策略梯度更新（使用REINFORCE算法）
        for experience in batch:
            state = experience['state']
            action = experience['action']
            reward = experience['reward']
            
            # 计算策略梯度（简化版）
            predicted_action = self.predict_action(state, self.gait_phase, None)
            action_error = action - predicted_action
            
            # 更新策略网络（使用奖励加权）
            learning_rate = self.learning_rate * reward  # 奖励越大，学习越快
            
            # 反向传播（简化版，只更新输出层）
            if abs(learning_rate) > 1e-6:
                grad = action_error * learning_rate
                self.policy_network['w3'] += np.outer(self.policy_network['lstm_h'], grad) * 0.01
                self.policy_network['b3'] += grad * 0.01
    
    def update_gait_phase(self, dt):
        """更新步态相位"""
        self.gait_phase += 2 * np.pi * self.gait_frequency * dt
        if self.gait_phase > 2 * np.pi:
            self.gait_phase -= 2 * np.pi
    
    def reset_lstm_state(self):
        """重置LSTM状态"""
        self.policy_network['lstm_h'] = np.zeros_like(self.policy_network['lstm_h'])
        self.policy_network['lstm_c'] = np.zeros_like(self.policy_network['lstm_c'])
    
    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'policy_network': self.policy_network,
            'value_network': self.value_network,
            'action_dim': self.action_dim,
            'state_dim': self.state_dim
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"[深度学习控制器] 模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.policy_network = model_data['policy_network']
            self.value_network = model_data['value_network']
            print(f"[深度学习控制器] 模型已从 {filepath} 加载")
        else:
            print(f"[深度学习控制器] 模型文件不存在: {filepath}")


class KeyboardController:
    """键盘控制节点：使用MuJoCo viewer的key_callback处理键盘输入"""
    def __init__(self, action_dim, actuator_indices=None):
        """
        Args:
            action_dim: 动作维度（执行器数量）
            actuator_indices: 执行器名称到索引的映射
        """
        self.action_dim = action_dim
        self.actuator_indices = actuator_indices or {}
        self.current_action = np.zeros(action_dim)
        
        self.exit_flag = False
        self.paused = False
        self.reset_flag = False
        
        # 移动控制状态
        self.move_forward = False
        self.move_backward = False
        self.turn_left = False
        self.turn_right = False
        
        # 步行动作时间计数器（改为基于键盘输入的脉冲式控制）
        self.step_time = 0.0
        self.step_frequency = 0.9  # 步频 (Hz) - 降低步频，让动作更自然
        self.step_duration = 0.5  # 每次按键的移动持续时间（秒）
        self.last_action_time = 0.0  # 上次执行动作的时间
        
        # 动作平滑：使用低通滤波和滑动平均
        self.action_smoothing_factor = 0.7  # 动作平滑系数（减小以更快停止）
        self.smoothed_action = np.zeros(action_dim)
        self.action_history = deque(maxlen=3)  # 减少历史长度，更快响应
        
        # PID控制器参数（用于速度控制）
        self.velocity_pid = {
            'kp': 2.0,  # 比例增益
            'ki': 0.1,  # 积分增益
            'kd': 0.5,  # 微分增益
            'integral': np.array([0.0, 0.0]),  # 积分项
            'last_error': np.array([0.0, 0.0])  # 上次误差
        }
        
        # 目标速度（根据键盘输入设置）
        self.target_velocity = np.array([0.0, 0.0])  # [vx, vy]
        self.current_velocity = np.array([0.0, 0.0])
        
        # 转向控制：累积转向角度，每次转向约45度
        self.target_turn_angle = 0.0  # 目标转向角度（弧度）
        self.current_turn_angle = 0.0  # 当前转向角度（弧度）
        self.turn_angle_per_step = np.pi / 4.0  # 每次转向目标角度：45度（π/4弧度）
        self.turn_speed = 2.0  # 转向速度（弧度/秒）
        
        # 键盘输入防抖：避免重复触发
        self.key_debounce_time = 0.15  # 防抖时间（秒）
        self.last_key_time = {}  # 记录每个按键的最后触发时间
        
        # 简单的神经网络控制器（用于动作平滑）
        self.use_neural_smoothing = True
        self._init_neural_smoother()
        
        # 深度学习控制器（用于学习最优步态）
        self.use_deep_learning = True
        self.deep_controller = None  # 将在get_action中初始化（需要state_dim）
        self.last_state = None
        self.last_reward = 0.0

        self._print_help()
    
    def _init_neural_smoother(self):
        """初始化简单的神经网络平滑器（单层感知机）"""
        # 简单的单层神经网络，用于学习动作平滑映射
        # 输入：当前动作 + 历史动作（最近3个）
        # 输出：平滑后的动作
        input_dim = self.action_dim * 4  # 当前 + 3个历史
        hidden_dim = self.action_dim * 2
        output_dim = self.action_dim
        
        # 使用简单的权重矩阵（可以后续用训练数据优化）
        np.random.seed(42)
        self.neural_weights1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.neural_weights2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.neural_bias1 = np.zeros(hidden_dim)
        self.neural_bias2 = np.zeros(output_dim)
        
        # 激活函数：ReLU + Tanh
        self.neural_history = deque(maxlen=3)
    
    def _neural_smooth_action(self, action):
        """使用神经网络平滑动作"""
        if not self.use_neural_smoothing or len(self.neural_history) < 2:
            # 历史不足时，使用简单平滑
            return self._simple_smooth_action(action)
        
        # 构建输入：当前动作 + 历史动作
        history_actions = list(self.neural_history)
        while len(history_actions) < 3:
            history_actions.insert(0, np.zeros(self.action_dim))
        
        input_vec = np.concatenate([
            action,
            history_actions[0],
            history_actions[1] if len(history_actions) > 1 else np.zeros(self.action_dim),
            history_actions[2] if len(history_actions) > 2 else np.zeros(self.action_dim)
        ])
        
        # 前向传播
        hidden = np.maximum(0, input_vec @ self.neural_weights1 + self.neural_bias1)  # ReLU
        output = np.tanh(hidden @ self.neural_weights2 + self.neural_bias2)  # Tanh
        
        # 混合原始动作和平滑动作
        smoothed = 0.7 * action + 0.3 * output
        return np.clip(smoothed, -1.0, 1.0)
    
    def _simple_smooth_action(self, action):
        """简单的动作平滑（低通滤波 + 滑动平均）"""
        # 检查动作是否为零（停止指令）
        if np.max(np.abs(action)) < 0.01:
            # 停止时，快速衰减
            self.smoothed_action = self.smoothed_action * 0.6
            if np.max(np.abs(self.smoothed_action)) < 0.01:
                self.smoothed_action = np.zeros(self.action_dim)
        else:
            # 有动作时，使用低通滤波
            self.smoothed_action = (
                self.action_smoothing_factor * self.smoothed_action +
                (1 - self.action_smoothing_factor) * action
            )
            
            # 滑动平均（只在有动作时）
            self.action_history.append(action.copy())
            if len(self.action_history) > 1:
                avg_action = np.mean(list(self.action_history), axis=0)
                # 混合低通滤波和滑动平均
                self.smoothed_action = 0.7 * self.smoothed_action + 0.3 * avg_action
        
        return np.clip(self.smoothed_action, -1.0, 1.0)
    
    def _update_pid_controller(self, target_vel, current_vel, dt):
        """更新PID控制器，计算速度修正"""
        error = target_vel - current_vel
        
        # 比例项
        p_term = self.velocity_pid['kp'] * error
        
        # 积分项（带抗饱和）
        self.velocity_pid['integral'] += error * dt
        self.velocity_pid['integral'] = np.clip(
            self.velocity_pid['integral'],
            -2.0, 2.0  # 限制积分项，防止积分饱和
        )
        i_term = self.velocity_pid['ki'] * self.velocity_pid['integral']
        
        # 微分项
        d_error = (error - self.velocity_pid['last_error']) / dt
        d_term = self.velocity_pid['kd'] * d_error
        
        # 更新上次误差
        self.velocity_pid['last_error'] = error.copy()
        
        # PID输出
        pid_output = p_term + i_term + d_term
        return pid_output
    
    def _update_target_velocity(self):
        """根据键盘输入更新目标速度"""
        # 重置目标速度
        self.target_velocity = np.array([0.0, 0.0])
        
        # 根据移动状态设置目标速度
        if self.move_forward:
            self.target_velocity[0] = 1.0  # 前进速度
        elif self.move_backward:
            self.target_velocity[0] = -0.8  # 后退速度
        
        # 转向速度（通过旋转实现，这里先设为0，由转向动作控制）
        if self.turn_left:
            self.target_velocity[1] = -0.3  # 左转
        elif self.turn_right:
            self.target_velocity[1] = 0.3  # 右转
    
    def _print_help(self):
        """打印键盘控制指令说明"""
        print("\n===== 键盘控制指令 =====")
        print("  w/↑: 前进")
        print("  s/↓: 后退")
        print("  a/←: 左转")
        print("  d/→: 右转")
        print("  空格: 暂停/继续")
        print("  r: 重置环境")
        print("  q: 退出程序")
        print("=======================")
        print("注意：请在查看器窗口内按键盘（窗口需要有焦点）\n")
    
    def key_callback(self, keycode):
        """MuJoCo viewer的键盘回调函数"""
        try:
            arrow_keys = {
                265: '\x1b[A',  # 上箭头 (Up)
                264: '\x1b[B',  # 下箭头 (Down)
                263: '\x1b[D',  # 左箭头 (Left)
                262: '\x1b[C',  # 右箭头 (Right)
            }
            
            if keycode in arrow_keys:
                key = arrow_keys[keycode]
            elif keycode == 32:  # 空格键 (Space)
                key = ' '
            elif 32 <= keycode <= 126:  # 可打印ASCII字符
                key = chr(keycode).lower()
            else:
                return
            
            self._process_key(key)
        except Exception as e:
            print(f"[错误] 处理按键时出错 (keycode={keycode}): {e}")
    
    def _set_action(self, action, name, value):
        """根据执行器名称写入动作，自动忽略缺失的执行器"""
        idx = self.actuator_indices.get(name)
        if idx is not None and 0 <= idx < self.action_dim:
            action[idx] = value
    
    def _create_walking_action(self, forward=True, turn_direction=0):
        """创建步行动作：更自然的人类步态，包含支撑相和摆动相的协调"""
        action = np.zeros(self.action_dim)
        
        if not self.actuator_indices:
            return action
        
        # 根据方向调节步频与强度：后退更慢、更谨慎
        step_freq = self.step_frequency if forward else self.step_frequency * 0.8
        
        # 计算步行动作相位（保持连续性）
        phase = 2 * np.pi * self.step_time * step_freq
        direction = 1 if forward else -1
        
        # 计算步态强度（基于step_time，用于平滑停止）
        # 当step_time衰减时，动作幅度也平滑减小
        gait_strength = min(1.0, self.step_time * step_freq * 2.0)  # 在第一个周期内从0到1
        # 如果step_time很小，进一步减小强度，实现平滑停止
        if self.step_time < 0.1:
            gait_strength *= self.step_time / 0.1  # 在最后0.1秒内平滑衰减到0
        
        # 后退或转向时整体动作幅度更柔和
        if not forward:
            gait_strength *= 0.75
        if turn_direction != 0:
            gait_strength *= 0.9
        
        # 人类步态特点：
        # 1. 支撑相约占60%，摆动相约占40%
        # 2. 摆动相时：抬腿、膝关节弯曲、踝关节背屈
        # 3. 支撑相时：腿伸直、踝关节跖屈、推进身体
        
        # 右腿相位
        right_phase = phase
        # 左腿相位（相差180度，形成交替步态）
        left_phase = phase + np.pi
        
        # 定义摆动相和支撑相的平滑过渡函数
        # 摆动相：0到π（约40%的时间），支撑相：π到2π（约60%的时间）
        def swing_phase_weight(phi):
            """计算摆动相权重：在0到π之间为1，在π到2π之间平滑过渡到0"""
            phi_norm = phi % (2 * np.pi)
            if phi_norm < np.pi:
                # 摆动相：使用平滑的上升和下降
                return 0.5 * (1 - np.cos(phi_norm))  # 0到1的平滑上升
            else:
                # 支撑相：快速下降到0
                support_phase = phi_norm - np.pi
                return max(0, 0.5 * (1 + np.cos(support_phase)))  # 1到0的平滑下降
        
        def support_phase_weight(phi):
            """计算支撑相权重：与摆动相相反"""
            return 1.0 - swing_phase_weight(phi)
        
        # 右腿动作
        right_swing = swing_phase_weight(right_phase)
        right_support = support_phase_weight(right_phase)
        
        # 髋关节前后摆动（主要推进力）- 更自然的协调
        # 使用更平滑的正弦波，在摆动相向前，支撑相向后推
        # 添加轻微的相位偏移，让动作更自然
        base_hip_amp = 0.45 if forward else 0.32
        right_hip_swing = base_hip_amp * direction * np.sin(right_phase + 0.1) * gait_strength
        self._set_action(action, "hip_x_right", right_hip_swing)
        
        # 髋关节上下（抬腿）- 更自然的抬腿动作
        # 在摆动相早期开始抬腿，中期达到最高，后期下降
        swing_phase_norm = (right_phase % (2 * np.pi)) / (2 * np.pi)
        if swing_phase_norm < 0.5:  # 摆动相（前50%）
            # 抬腿：使用平滑的曲线，在摆动相中期（25%）达到最高
            lift_curve = np.sin(swing_phase_norm * 2 * np.pi)  # 0到1再到0
            # 后退时减少抬腿幅度，保持脚部更接近地面
            lift_amplitude = 0.2 if forward else 0.1  # 后退时抬腿幅度减半
            right_hip_lift = lift_amplitude * lift_curve * gait_strength
        else:  # 支撑相（后50%）
            right_hip_lift = 0.0
        self._set_action(action, "hip_y_right", -right_hip_lift)
        
        # 膝关节 - 更自然的协调，与髋关节配合
        # 摆动相：早期快速弯曲（配合抬腿），中期保持弯曲，后期开始伸直准备落地
        # 支撑相：完全伸直
        if swing_phase_norm < 0.5:  # 摆动相
            # 膝关节弯曲曲线：早期快速弯曲，中期保持，后期开始伸直
            if swing_phase_norm < 0.3:
                # 早期：快速弯曲到最大
                knee_curve = swing_phase_norm / 0.3  # 0到1
            elif swing_phase_norm < 0.4:
                # 中期：保持弯曲
                knee_curve = 1.0
            else:
                # 后期：开始伸直
                knee_curve = 1.0 - (swing_phase_norm - 0.4) / 0.1  # 1到0
            # 后退时减少膝关节弯曲幅度，保持腿部更直，脚部更接近地面
            knee_amplitude = 0.6 if forward else 0.3
            right_knee_angle = knee_amplitude * knee_curve * gait_strength
        else:  # 支撑相
            right_knee_angle = 0.0
        self._set_action(action, "knee_right", right_knee_angle)
        
        # 踝关节 - 更自然的协调，与膝关节配合
        # 摆动相：早期背屈（脚尖向上，配合抬腿），中期保持，后期开始跖屈准备落地
        # 支撑相：跖屈（脚尖向下，推进）
        if swing_phase_norm < 0.5:  # 摆动相
            # 背屈：在摆动相早期和中期
            if swing_phase_norm < 0.35:
                # 后退时减少背屈幅度，保持脚部更平
                dorsiflex_amplitude = -0.15 if forward else -0.08
                ankle_dorsiflex = dorsiflex_amplitude * (1 - swing_phase_norm / 0.35) * gait_strength
            else:
                ankle_dorsiflex = 0.0
            ankle_plantarflex = 0.0
        else:  # 支撑相
            # 跖屈：在支撑相早期和中期推进
            support_phase_norm = (swing_phase_norm - 0.5) * 2  # 0到1
            if support_phase_norm < 0.6:
                ankle_plantarflex = 0.12 * np.sin(support_phase_norm * np.pi) * gait_strength
            else:
                ankle_plantarflex = 0.0
            ankle_dorsiflex = 0.0
        self._set_action(action, "ankle_y_right", ankle_dorsiflex + ankle_plantarflex)
        # 踝关节内外翻（配合步态，轻微）
        self._set_action(action, "ankle_x_right", 0.08 * np.sin(right_phase) * gait_strength)
        
        # 左腿动作（相位相反，与右腿完全对称）
        left_phase_norm = (left_phase % (2 * np.pi)) / (2 * np.pi)
        
        # 左腿髋关节前后摆动（与右腿相反）
        left_hip_swing = -0.45 * direction * np.sin(left_phase + 0.1) * gait_strength
        self._set_action(action, "hip_x_left", left_hip_swing)
        
        # 左腿髋关节上下（抬腿）
        if left_phase_norm < 0.5:  # 摆动相
            lift_curve = np.sin(left_phase_norm * 2 * np.pi)
            # 后退时减少抬腿幅度，保持脚部更接近地面
            lift_amplitude = 0.2 if forward else 0.1  # 后退时抬腿幅度减半
            left_hip_lift = lift_amplitude * lift_curve * gait_strength
        else:  # 支撑相
            left_hip_lift = 0.0
        self._set_action(action, "hip_y_left", -left_hip_lift)
        
        # 左腿膝关节
        if left_phase_norm < 0.5:  # 摆动相
            if left_phase_norm < 0.3:
                knee_curve = left_phase_norm / 0.3
            elif left_phase_norm < 0.4:
                knee_curve = 1.0
            else:
                knee_curve = 1.0 - (left_phase_norm - 0.4) / 0.1
            # 后退时减少膝关节弯曲幅度，保持腿部更直，脚部更接近地面
            knee_amplitude = 0.6 if forward else 0.3
            left_knee_angle = knee_amplitude * knee_curve * gait_strength
        else:  # 支撑相
            left_knee_angle = 0.0
        self._set_action(action, "knee_left", left_knee_angle)
        
        # 左腿踝关节
        if left_phase_norm < 0.5:  # 摆动相
            if left_phase_norm < 0.35:
                # 后退时减少背屈幅度，保持脚部更平
                dorsiflex_amplitude = -0.15 if forward else -0.08
                ankle_dorsiflex = dorsiflex_amplitude * (1 - left_phase_norm / 0.35) * gait_strength
            else:
                ankle_dorsiflex = 0.0
            ankle_plantarflex = 0.0
        else:  # 支撑相
            support_phase_norm = (left_phase_norm - 0.5) * 2
            if support_phase_norm < 0.6:
                ankle_plantarflex = 0.12 * np.sin(support_phase_norm * np.pi) * gait_strength
            else:
                ankle_plantarflex = 0.0
            ankle_dorsiflex = 0.0
        self._set_action(action, "ankle_y_left", ankle_dorsiflex + ankle_plantarflex)
        self._set_action(action, "ankle_x_left", -0.08 * np.sin(left_phase) * gait_strength)
        
        # 侧向平衡控制
        if turn_direction == 0:
            # 直行时，保持髋关节外展对称
            hip_z_balance = 0.0
            self._set_action(action, "hip_z_right", hip_z_balance)
            self._set_action(action, "hip_z_left", -hip_z_balance)
        else:
            # 转向时，外侧腿稍微外展，内侧腿稍微内收
            turn_strength = 0.25 * turn_direction * gait_strength
            self._set_action(action, "hip_z_right", turn_strength)
            self._set_action(action, "hip_z_left", -turn_strength)
            # 添加躯干旋转辅助转向
            self._set_action(action, "abdomen_z", 0.4 * turn_direction)  # 添加躯干旋转
        
        return action
    
    def _create_turning_only_action(self, turn_direction, dt=0.03):
        """创建仅转向动作（不产生腿部摆动，只在原地转向，目标转向45度）"""
        action = np.zeros(self.action_dim)
        
        if not self.actuator_indices:
            return action
        
        # 更新目标转向角度（每次按键设置目标为45度）
        turn_velocity = 0.0
        if turn_direction != 0:
            # 计算转向误差
            turn_error = self.target_turn_angle - self.current_turn_angle
            
            # 如果接近目标角度，重置目标（允许连续转向）
            if abs(turn_error) < 0.1:  # 接近目标时，设置新的目标
                self.target_turn_angle += turn_direction * self.turn_angle_per_step
            
            # 计算转向速度（基于误差）
            turn_velocity = np.clip(turn_error * 3.0, -self.turn_speed, self.turn_speed)
            
            # 更新当前转向角度（模拟）
            self.current_turn_angle += turn_velocity * dt
        else:
            # 没有转向指令时，逐渐减小转向角度
            self.current_turn_angle *= 0.95
            self.target_turn_angle = self.current_turn_angle  # 同步目标角度
        
        # 根据转向速度计算转向强度（归一化到-1到1）
        if abs(turn_velocity) > 0.01:
            normalized_turn = np.clip(turn_velocity / self.turn_speed, -1.0, 1.0)
        else:
            # 如果没有转向速度，直接使用方向（简化控制）
            normalized_turn = turn_direction * 0.8  # 直接使用方向，强度0.8
        
        # 原地转向：通过髋关节外展和躯干旋转实现
        # 增大转向强度，使转向更明显
        hip_turn_strength = 0.6 * normalized_turn  # 从0.25增大到0.6
        self._set_action(action, "hip_z_right", hip_turn_strength)
        self._set_action(action, "hip_z_left", -hip_turn_strength)
        
        # 躯干旋转辅助转向（主要转向来源，范围±45度）
        abdomen_turn_strength = 0.8 * normalized_turn  # 从0.15增大到0.8，充分利用±45度范围
        self._set_action(action, "abdomen_z", abdomen_turn_strength)
        self._set_action(action, "abdomen_x", 0.1 * normalized_turn)
        
        # 躯干控制 - 更自然的轻微摆动
        # 轻微前倾以辅助前进（减小前倾幅度，更自然）
        abdomen_pitch = 0.08 * direction * gait_strength
        # 添加轻微的上下摆动（配合步态，与腿部动作协调）
        # 在支撑相时稍微下沉，在摆动相时稍微上升
        abdomen_pitch += 0.02 * np.sin(phase + np.pi/4) * gait_strength
        self._set_action(action, "abdomen_y", abdomen_pitch)
        
        # 转向时允许侧倾（减小侧倾幅度）
        self._set_action(action, "abdomen_x", 0.05 * turn_direction * gait_strength)
        
        # 转向控制（减小转向幅度，更自然）
        if turn_direction != 0:
            self._set_action(action, "abdomen_z", 0.25 * turn_direction * gait_strength)
        else:
            self._set_action(action, "abdomen_z", 0.0)
        
        return action
    
    def _create_turning_only_action(self, turn_direction, dt=0.03):
        """创建仅转向动作（不产生腿部摆动，只在原地转向，目标转向45度）"""
        action = np.zeros(self.action_dim)
        
        if not self.actuator_indices:
            return action
        
        # 更新目标转向角度（每次按键设置目标为45度）
        turn_velocity = 0.0
        if turn_direction != 0:
            # 计算转向误差
            turn_error = self.target_turn_angle - self.current_turn_angle
            
            # 如果接近目标角度，重置目标（允许连续转向）
            if abs(turn_error) < 0.1:  # 接近目标时，设置新的目标
                self.target_turn_angle += turn_direction * self.turn_angle_per_step
            
            # 计算转向速度（基于误差）
            turn_velocity = np.clip(turn_error * 3.0, -self.turn_speed, self.turn_speed)
            
            # 更新当前转向角度（模拟）
            self.current_turn_angle += turn_velocity * dt
        else:
            # 没有转向指令时，逐渐减小转向角度
            self.current_turn_angle *= 0.95
            self.target_turn_angle = self.current_turn_angle  # 同步目标角度
        
        # 根据转向速度计算转向强度（归一化到-1到1）
        if abs(turn_velocity) > 0.01:
            normalized_turn = np.clip(turn_velocity / self.turn_speed, -1.0, 1.0)
        else:
            # 如果没有转向速度，直接使用方向（简化控制）
            normalized_turn = turn_direction * 0.8  # 直接使用方向，强度0.8
        
        # 原地转向：通过髋关节外展和躯干旋转实现
        # 增大转向强度，使转向更明显
        hip_turn_strength = 0.6 * normalized_turn  # 从0.25增大到0.6
        self._set_action(action, "hip_z_right", hip_turn_strength)
        self._set_action(action, "hip_z_left", -hip_turn_strength)
        
        # 躯干旋转辅助转向（主要转向来源，范围±45度）
        abdomen_turn_strength = 0.8 * normalized_turn  # 从0.15增大到0.8，充分利用±45度范围
        self._set_action(action, "abdomen_z", abdomen_turn_strength)
        self._set_action(action, "abdomen_x", 0.1 * normalized_turn)
        
        # 躯干控制 - 更自然的轻微摆动
        # 轻微前倾以辅助前进（减小前倾幅度，更自然）
        abdomen_pitch = 0.08 * direction * gait_strength
        # 添加轻微的上下摆动（配合步态，与腿部动作协调）
        # 在支撑相时稍微下沉，在摆动相时稍微上升
        abdomen_pitch += 0.02 * np.sin(phase + np.pi/4) * gait_strength
        self._set_action(action, "abdomen_y", abdomen_pitch)
        
        # 转向时允许侧倾（减小侧倾幅度）
        self._set_action(action, "abdomen_x", 0.05 * turn_direction * gait_strength)
        
        # 转向控制（减小转向幅度，更自然）
        if turn_direction != 0:
            self._set_action(action, "abdomen_z", 0.25 * turn_direction * gait_strength)
        else:
            self._set_action(action, "abdomen_z", 0.0)
        
        return action
    
    def _create_turning_only_action(self, turn_direction, dt=0.03):
        """创建仅转向动作（不产生腿部摆动，只在原地转向，目标转向45度）"""
        action = np.zeros(self.action_dim)
        
        if not self.actuator_indices:
            return action
        
        # 更新目标转向角度（每次按键设置目标为45度）
        turn_velocity = 0.0
        if turn_direction != 0:
            # 计算转向误差
            turn_error = self.target_turn_angle - self.current_turn_angle
            
            # 如果接近目标角度，重置目标（允许连续转向）
            if abs(turn_error) < 0.1:  # 接近目标时，设置新的目标
                self.target_turn_angle += turn_direction * self.turn_angle_per_step
            
            # 计算转向速度（基于误差）
            turn_velocity = np.clip(turn_error * 3.0, -self.turn_speed, self.turn_speed)
            
            # 更新当前转向角度（模拟）
            self.current_turn_angle += turn_velocity * dt
        else:
            # 没有转向指令时，逐渐减小转向角度
            self.current_turn_angle *= 0.95
            self.target_turn_angle = self.current_turn_angle  # 同步目标角度
        
        # 根据转向速度计算转向强度（归一化到-1到1）
        if abs(turn_velocity) > 0.01:
            normalized_turn = np.clip(turn_velocity / self.turn_speed, -1.0, 1.0)
        else:
            # 如果没有转向速度，直接使用方向（简化控制）
            normalized_turn = turn_direction * 0.8  # 直接使用方向，强度0.8
        
        # 原地转向：通过髋关节外展和躯干旋转实现
        # 增大转向强度，使转向更明显
        hip_turn_strength = 0.6 * normalized_turn  # 从0.25增大到0.6
        self._set_action(action, "hip_z_right", hip_turn_strength)
        self._set_action(action, "hip_z_left", -hip_turn_strength)
        
        # 躯干旋转辅助转向（主要转向来源，范围±45度）
        abdomen_turn_strength = 0.8 * normalized_turn  # 从0.15增大到0.8，充分利用±45度范围
        self._set_action(action, "abdomen_z", abdomen_turn_strength)
        self._set_action(action, "abdomen_x", 0.1 * normalized_turn)
        
        # 躯干控制 - 更自然的轻微摆动
        # 轻微前倾以辅助前进（减小前倾幅度，更自然）
        abdomen_pitch = 0.07 * direction * gait_strength
        # 添加轻微的上下摆动（配合步态，与腿部动作协调）
        # 在支撑相时稍微下沉，在摆动相时稍微上升
        abdomen_pitch += 0.02 * np.sin(phase + np.pi/4) * gait_strength
        self._set_action(action, "abdomen_y", abdomen_pitch)
        
        # 转向时允许侧倾（减小侧倾幅度）
        self._set_action(action, "abdomen_x", 0.04 * turn_direction * gait_strength)
        
        # 转向控制（减小转向幅度，更自然）
        if turn_direction != 0:
            self._set_action(action, "abdomen_z", 0.22 * turn_direction * gait_strength)
        else:
            self._set_action(action, "abdomen_z", 0.0)
        
        return action
    
    def _create_turning_only_action(self, turn_direction, dt=0.03):
        """创建仅转向动作：更平滑的原地转身"""
        action = np.zeros(self.action_dim)
        
        if not self.actuator_indices:
            return action
        
        # 更新目标转向角度（每次按键设置目标为45度）
        turn_velocity = 0.0
        if turn_direction != 0:
            # 计算转向误差
            turn_error = self.target_turn_angle - self.current_turn_angle
            
            # 如果接近目标角度，重置目标（允许连续转向）
            if abs(turn_error) < 0.1:  # 接近目标时，设置新的目标
                self.target_turn_angle += turn_direction * self.turn_angle_per_step
            
            # 计算转向速度（基于误差），限制更小避免生硬
            turn_velocity = np.clip(turn_error * 2.0, -self.turn_speed * 0.7, self.turn_speed * 0.7)
            
            # 更新当前转向角度（模拟）
            self.current_turn_angle += turn_velocity * dt
        else:
            # 没有转向指令时，逐渐减小转向角度
            self.current_turn_angle *= 0.95
            self.target_turn_angle = self.current_turn_angle  # 同步目标角度
        
        # 根据转向速度计算转向强度（归一化到-1到1）
        if abs(turn_velocity) > 0.01:
            normalized_turn = np.clip(turn_velocity / self.turn_speed, -1.0, 1.0)
        else:
            # 如果没有转向速度，直接使用方向（简化控制）
            normalized_turn = turn_direction * 0.8  # 直接使用方向，强度0.8
        
        # 原地转向：通过髋关节外展和躯干旋转实现
        # 略微减小强度并加入轻微屈膝，让转身更稳
        hip_turn_strength = 0.45 * normalized_turn
        self._set_action(action, "hip_z_right", hip_turn_strength)
        self._set_action(action, "hip_z_left", -hip_turn_strength)
        
        # 躯干旋转辅助转向（主要转向来源，范围±45度）
        abdomen_turn_strength = 0.65 * normalized_turn
        self._set_action(action, "abdomen_z", abdomen_turn_strength)
        self._set_action(action, "abdomen_x", 0.1 * normalized_turn)
        
        # 轻微屈膝降低质心
        self._set_action(action, "knee_right", 0.12 * abs(normalized_turn))
        self._set_action(action, "knee_left", 0.12 * abs(normalized_turn))
        
        return action
    
    def _process_key(self, key):
        """处理按键输入（带防抖机制）"""
        import time
        current_time = time.time()
        
        if isinstance(key, str) and key.startswith('\x1b['):
            key_char = None  # 方向键用特殊序列表示
            key_id = key  # 使用特殊序列作为ID
        else:
            key_char = key if isinstance(key, str) and len(key) == 1 else None
            key_id = key_char if key_char else key
        
        # 防抖检查：如果距离上次按键时间太短，忽略此次按键
        if key_id in self.last_key_time:
            time_since_last = current_time - self.last_key_time[key_id]
            if time_since_last < self.key_debounce_time:
                return  # 忽略重复按键
        
        # 更新按键时间
        self.last_key_time[key_id] = current_time
        
        # 处理移动指令（切换模式：每次按键切换状态）
        move_commands = {
            ('w', '\x1b[A'): ('move_forward', 'move_backward', '前进', '停止前进'),
            ('s', '\x1b[B'): ('move_backward', 'move_forward', '后退', '停止后退'),
            ('a', '\x1b[D'): ('turn_left', 'turn_right', '左转', '停止左转'),
            ('d', '\x1b[C'): ('turn_right', 'turn_left', '右转', '停止右转'),
        }
        
        for (key1, key2), (attr, opposite_attr, start_msg, stop_msg) in move_commands.items():
            if (key_char == key1) or (key == key2):
                current_state = getattr(self, attr)
                if current_state:
                    # 停止移动时，不立即重置step_time，让当前步态周期平滑完成
                    setattr(self, attr, False)
                    # 注意：不重置step_time，让它自然衰减，保持步态连续性
                    # 快速清零平滑动作
                    if not (self.move_forward or self.move_backward or self.turn_left or self.turn_right):
                        self.smoothed_action = np.zeros(self.action_dim)
                    print(f"[键盘] {stop_msg}")
                else:
                    setattr(self, attr, True)
                    if hasattr(self, opposite_attr):
                        setattr(self, opposite_attr, False)
                    # 开始移动时，不重置step_time，保持步态相位连续性
                    # 如果step_time为0（首次启动），保持为0；否则继续累积
                    print(f"[键盘] {start_msg}")
                return
        
        if key == ' ':
            self.paused = not self.paused
            if self.paused:
                self.current_action = np.zeros(self.action_dim)
                self.move_forward = False
                self.move_backward = False
                self.turn_left = False
                self.turn_right = False
            print(f"[键盘] {'⏸️ 已暂停' if self.paused else '▶️ 继续'}")
        elif key_char == 'r':
            self.reset_flag = True
            print("[键盘] 🔄 重置环境")
        elif key_char == 'q':
            self.exit_flag = True
            print("[键盘] ❌ 准备退出程序...")
    
    def update_step_time(self, dt):
        """更新步行动作时间（保持步态连续性）"""
        if not self.paused and (self.move_forward or self.move_backward or self.turn_left or self.turn_right):
            # 有键盘输入时，持续累积时间，保持步态连续性
            self.step_time += dt
        else:
            # 没有键盘输入时，平滑衰减step_time，让步态平滑停止
            # 使用指数衰减，而不是立即重置，保持动作连续性
            decay_rate = 0.95  # 每步衰减5%
            self.step_time *= decay_rate
            # 当step_time很小时，重置为0，避免无限小的值
            if self.step_time < 0.01:
                self.step_time = 0.0
    
    def get_action(self, dt=0.03, current_velocity=None, state=None, reward=None):
        """获取当前控制动作（基于键盘输入的离散控制 + 深度学习增强）"""
        if self.paused:
            self.smoothed_action = np.zeros(self.action_dim)
            self.target_velocity = np.array([0.0, 0.0])
            self.step_time = 0.0
            return np.zeros(self.action_dim)
        
        # 更新当前速度（如果提供）
        if current_velocity is not None:
            self.current_velocity = current_velocity.copy()
        
        # 更新目标速度
        self._update_target_velocity()
        
        # 检查是否有任何移动指令
        has_movement = self.move_forward or self.move_backward or self.turn_left or self.turn_right
        
        # 初始化深度学习控制器（如果启用且未初始化）
        if self.use_deep_learning and self.deep_controller is None and state is not None:
            state_dim = len(state)
            self.deep_controller = DeepLearningController(
                self.action_dim, 
                state_dim, 
                self.actuator_indices
            )
            print("[键盘控制器] 深度学习控制器已初始化")
        
        # 使用深度学习控制器生成动作（如果启用）
        if self.use_deep_learning and self.deep_controller is not None and state is not None:
            # 更新步态相位
            self.deep_controller.update_gait_phase(dt)
            gait_phase = self.deep_controller.gait_phase
            
            # 构建用户命令
            command = {
                'forward': self.move_forward,
                'backward': self.move_backward,
                'turn_left': self.turn_left,
                'turn_right': self.turn_right
            }
            
            # 使用深度学习控制器预测动作
            last_action = self.current_action if hasattr(self, 'current_action') else None
            dl_action = self.deep_controller.predict_action(state, gait_phase, last_action, command)
            
            # 如果用户有移动指令，混合传统动作和深度学习动作
            if has_movement:
                # 更新步行动作时间
                self.update_step_time(dt)
                
                # 生成传统动作
                if self.move_forward:
                    turn_dir = 0
                    if self.turn_left:
                        turn_dir = -1
                    elif self.turn_right:
                        turn_dir = 1
                    traditional_action = self._create_walking_action(forward=True, turn_direction=turn_dir)
                elif self.move_backward:
                    turn_dir = 0
                    if self.turn_left:
                        turn_dir = 1
                    elif self.turn_right:
                        turn_dir = -1
                    traditional_action = self._create_walking_action(forward=False, turn_direction=turn_dir)
                elif self.turn_left or self.turn_right:
                    turn_dir = -1 if self.turn_left else 1
                    traditional_action = self._create_turning_only_action(turn_dir, dt=dt)
                else:
                    traditional_action = np.zeros(self.action_dim)
                
                # 混合传统动作和深度学习动作（70%传统，30%深度学习）
                raw_action = 0.7 * traditional_action + 0.3 * dl_action
            else:
                # 没有移动指令时，返回零动作，不生成任何移动
                raw_action = np.zeros(self.action_dim)
                self.step_time = 0.0
        else:
            # 传统方法：没有深度学习控制器时使用原有逻辑
            if not has_movement:
                self.step_time = 0.0
                self.smoothed_action = self.smoothed_action * 0.5
                if np.max(np.abs(self.smoothed_action)) < 0.01:
                    self.smoothed_action = np.zeros(self.action_dim)
                self.current_action = self.smoothed_action.copy()
                return self.current_action.copy()
            
            self.update_step_time(dt)
            
            if self.move_forward:
                turn_dir = 0
                if self.turn_left:
                    turn_dir = -1
                elif self.turn_right:
                    turn_dir = 1
                raw_action = self._create_walking_action(forward=True, turn_direction=turn_dir)
            elif self.move_backward:
                turn_dir = 0
                if self.turn_left:
                    turn_dir = 1
                elif self.turn_right:
                    turn_dir = -1
                raw_action = self._create_walking_action(forward=False, turn_direction=turn_dir)
            elif self.turn_left or self.turn_right:
                turn_dir = -1 if self.turn_left else 1
                raw_action = self._create_turning_only_action(turn_dir, dt=dt)
            else:
                raw_action = np.zeros(self.action_dim)
        
        # 应用动作平滑
        if self.use_neural_smoothing and len(self.neural_history) >= 2:
            smoothed = self._neural_smooth_action(raw_action)
            self.neural_history.append(raw_action.copy())
        else:
            smoothed = self._simple_smooth_action(raw_action)
        
        self.current_action = smoothed
        return self.current_action.copy()
    
    def should_exit(self):
        """检查是否应该退出"""
        return self.exit_flag
    
    def should_reset(self):
        """检查是否应该重置"""
        return self.reset_flag
    
    def clear_reset_flag(self):
        """清除重置标志"""
        self.reset_flag = False


class GapCorridorEnvironment:
    """基于mujoco的带空隙走廊环境（使用自定义人形机器人模型）"""
    def __init__(self, corridor_length=100, corridor_width=10, robot_xml_path=None, use_gravity=True):
        """
        Args:
            corridor_length: 走廊总长度
            corridor_width: 走廊宽度
            robot_xml_path: 自定义人形机器人XML文件路径
            use_gravity: 是否启用重力（False 表示无重力）
        """
        self.corridor_length = corridor_length
        self.corridor_width = corridor_width
        self.use_gravity = use_gravity
        # if robot_xml_path is None:
        #     default_path = Path(__file__).resolve().parent / "model" / "humanoid" / "humanoid.xml"
        # else:
        #     default_path = Path(robot_xml_path)
        # if not default_path.is_file():
        #     raise FileNotFoundError(f"无法找到机器人XML文件: {default_path}")
        # self.robot_xml_path = default_path
        self.robot_xml_path = "humanoid.xml"
        xml_string = self._build_model()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        # 保险起见，在模型创建后再次根据标志位设置重力（即使 XML 中已经设置）
        if not self.use_gravity:
            self.model.opt.gravity[:] = 0.0
        self.data = mujoco.MjData(self.model)
        self.timestep = self.model.opt.timestep
        self.control_timestep = 0.03
        self.control_steps = int(self.control_timestep / self.timestep)
        self._max_episode_steps = 30 / self.control_timestep
        self.current_step = 0
        self._actuator_indices = self._build_actuator_indices()
        
        # 无重力模式：只固定Z高度，允许XY平移和姿态变化
        if not self.use_gravity:
            self._initial_z_height = None
            self._root_joint_qpos_start = None
            self._root_joint_qvel_start = None
            self._root_body_id = None
            self._max_xy_velocity = 2.0  # 最大XY速度 (m/s)
            self._xy_damping = 0.99  # XY速度阻尼系数（减小阻尼，保持速度）
            self._forward_velocity_gain = 2.5  # 前进速度增益（增大增益，产生明显移动）
            self._turn_velocity_gain = 0.5  # 转向速度增益
            
            # 姿态稳定控制参数
            self._initial_head_height = None  # 初始头部高度
            self._head_stability_gain = 5.0  # 头部高度稳定增益
            self._torso_pitch_target = 0.0  # 目标躯干俯仰角（前倾角度）
            self._torso_roll_target = 0.0  # 目标躯干侧倾角
            self._torso_stability_gain = 2.0  # 躯干姿态稳定增益
            
            self._find_root_joint_indices()

    def _parse_robot_xml(self):
        """解析自定义机器人XML，提取需要的节点（身体、执行器、肌腱等）"""
        tree = ET.parse(self.robot_xml_path)
        root = tree.getroot()
        
        robot_body = root.find("worldbody").find("body[@name='torso']")
        robot_body.set("pos", "1.0 0.5 1.5")
        
        # 提取XML节点并转换为字符串
        single_nodes = ["actuator", "tendon", "contact", "asset", "visual", "keyframe", "statistic"]
        parts = {"robot_body": ET.tostring(robot_body, encoding="unicode")}
        for node_name in single_nodes:
            node = root.find(node_name)
            parts[node_name] = ET.tostring(node, encoding="unicode") if node is not None else ""
        default_nodes = root.findall("default")
        parts["default"] = "".join(ET.tostring(node, encoding="unicode") for node in default_nodes)
        
        return parts

    def _build_model(self):
        """构建带空隙的走廊环境，并整合自定义人形机器人模型"""
        # 解析自定义机器人XML
        robot_parts = self._parse_robot_xml()

        # 根据是否使用重力设置 gravity 参数
        gravity_z = -9.81 if self.use_gravity else 0.0

        # 基础XML结构（走廊环境+机器人）
        xml = f"""
        <mujoco model="gap_corridor_with_custom_humanoid">
            <!-- 物理参数 -->
            <option timestep="0.005" gravity="0 0 {gravity_z}"/>
            
            <!-- 整合机器人的材质和可视化配置 -->
            {robot_parts['visual']}
            {robot_parts['asset']}
            {robot_parts['statistic']}
            
            <!-- 走廊环境的默认参数 -->
            <default>
                <joint armature="0.1" damping="1" limited="true"/>
                <geom conaffinity="0" condim="3" friction="1 0.1 0.1" 
                      solimp="0.99 0.99 0.003" solref="0.02 1"/>
            </default>
            {robot_parts['default']}
            
            <worldbody>
                <!-- 走廊地面（半透明，方便观察空隙） -->
                <geom name="floor" type="plane" size="{self.corridor_length/2} {self.corridor_width/2} 0.1" 
                      pos="{self.corridor_length/2} 0 0" rgba="0.9 0.9 0.9 0.3"/>
                
                <!-- 带空隙的走廊平台 -->
                {self._build_gaps_corridor()}
                
                <!-- 整合自定义人形机器人 -->
                {robot_parts['robot_body']}
            </worldbody>
            
            <!-- 机器人的接触排除配置 -->
            {robot_parts['contact']}
            
            <!-- 机器人的肌腱定义 -->
            {robot_parts['tendon']}
            
            <!-- 机器人的执行器（电机） -->
            {robot_parts['actuator']}
            
            <!-- 机器人的关键帧（可选） -->
            {robot_parts['keyframe']}
        </mujoco>
        """
        return xml

    def _build_gaps_corridor(self):
        """构建带空隙的走廊（平台+空隙交替）"""
        platform_length, gap_length, platform_thickness = 2.0, 1.0, 0.2
        platform_width = self.corridor_width / 4 - 0.1
        gaps = []
        
        current_pos = 0.0
        while current_pos < self.corridor_length:
            x_pos = current_pos + platform_length / 2
            z_pos = platform_thickness / 2
            size_str = f"{platform_length/2} {platform_width} {platform_thickness/2}"
            
            for side, y_pos in [("left", -self.corridor_width/4), ("right", self.corridor_width/4)]:
                gaps.append(f"""
            <geom name="platform_{side}_{current_pos}" type="box" 
                  size="{size_str}" 
                  pos="{x_pos} {y_pos} {z_pos}" 
                  rgba="0.4 0.4 0.8 1"/>
            """)
            current_pos += platform_length + gap_length
        
        return ''.join(gaps)
    
    def _build_actuator_indices(self):
        """建立执行器名称到索引的映射，方便控制器按名称写入动作"""
        indices = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                indices[name] = i
        return indices
    
    def get_actuator_indices(self):
        return self._actuator_indices.copy()
    
    def _find_root_joint_indices(self):
        """找到根关节（freejoint）的位置和速度在qpos/qvel中的索引"""
        try:
            root_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
            if root_joint_id >= 0:
                self._root_joint_qpos_start = self.model.jnt_qposadr[root_joint_id]
                self._root_joint_qvel_start = self.model.jnt_dofadr[root_joint_id]
                self._root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
                print(f"[无重力模式] 找到根关节: qpos={self._root_joint_qpos_start}, qvel={self._root_joint_qvel_start}")
                return
        except Exception as e:
            print(f"[警告] 查找根关节时出错: {e}")
        
        # 使用默认值（通常freejoint是第一个关节）
        self._root_joint_qpos_start = 0
        self._root_joint_qvel_start = 0
        self._root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso") if self.model else None
        print(f"[无重力模式] 使用默认根关节索引")

    def reset(self):
        """重置环境到初始状态"""
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # 无重力模式：记录根关节的初始Z高度、Y位置和姿态（保持身体直立）
        if not self.use_gravity and self._root_joint_qpos_start is not None:
            self._initial_z_height = float(self.data.qpos[self._root_joint_qpos_start + 2])
            self._initial_y_position = float(self.data.qpos[self._root_joint_qpos_start + 1])  # 记录初始Y位置
            # 记录初始姿态（四元数），用于保持身体直立
            if (self._root_joint_qpos_start + 6) < len(self.data.qpos):
                self._initial_quat = self.data.qpos[self._root_joint_qpos_start + 3:self._root_joint_qpos_start + 7].copy()
            else:
                self._initial_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 默认单位四元数（无旋转）
            # 记录初始头部高度（用于姿态稳定控制）
            head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
            if head_id >= 0:
                self._initial_head_height = float(self.data.xpos[head_id][2])
            else:
                self._initial_head_height = None
            
            # 记录初始根关节位置（用于计算脚部相对位置）
            if hasattr(self, '_root_body_id') and self._root_body_id is not None:
                self._initial_root_pos = self.data.xpos[self._root_body_id].copy()
            else:
                self._initial_root_pos = None
            
            # 记录脚部初始位置（用于保持脚部着地）
            self._initial_foot_positions = {}
            foot_names = ["foot_right", "foot_left", "right_foot", "left_foot"]
            for foot_name in foot_names:
                foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_name)
                if foot_id >= 0:
                    self._initial_foot_positions[foot_name] = self.data.xpos[foot_id].copy()
            
            print(f"[无重力模式] 记录初始Z高度: {self._initial_z_height:.4f}，初始Y位置: {self._initial_y_position:.4f}，保持身体直立")
        
        return self._get_observation()

    def _get_observation(self):
        """获取观测（关节位置、速度、躯干位置）"""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        torso_pos = self.data.xpos[torso_id].copy()
        return np.concatenate([qpos, qvel, torso_pos])

    def _get_reward(self):
        """计算奖励：前进速度（沿走廊X轴）+ 稳定性奖励 + 空隙掉落惩罚"""
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        
        geom_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, 
            self.data, 
            mujoco.mjtObj.mjOBJ_BODY, 
            torso_id, 
            geom_vel, 
            0
        )
        reward = geom_vel[0] * 0.1
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            geom_names = [geom1_name, geom2_name]
            if not any(name and "platform" in name for name in geom_names):
                reward -= 0.3
                break
        return reward

    def _apply_zero_gravity_constraints(self, action, before_step=True):
        """应用无重力模式的约束：固定Z高度和Y位置（保持在走廊中心），允许X方向移动，并根据动作主动施加速度"""
        if self.use_gravity or self._initial_z_height is None:
            return
        
        pos_start = self._root_joint_qpos_start
        vel_start = self._root_joint_qvel_start
        
        if pos_start is None or vel_start is None:
            return
        
        if before_step:
            # mj_step前：固定Z位置、Y位置和姿态（保持身体直立），不干扰其他物理量
            if (pos_start + 2) < len(self.data.qpos):
                self.data.qpos[pos_start + 2] = self._initial_z_height
            if (pos_start + 1) < len(self.data.qpos) and hasattr(self, '_initial_y_position'):
                self.data.qpos[pos_start + 1] = self._initial_y_position
            
            # 稳定姿态：保持身体直立，只允许绕Z轴旋转（yaw）
            if (pos_start + 6) < len(self.data.qpos) and hasattr(self, '_initial_quat'):
                # 获取当前四元数
                current_quat = self.data.qpos[pos_start + 3:pos_start + 7]
                
                # 从初始四元数提取yaw角（绕Z轴旋转）
                qw0, qx0, qy0, qz0 = self._initial_quat
                initial_yaw = np.arctan2(2.0 * (qw0 * qz0 + qx0 * qy0), 1.0 - 2.0 * (qy0 * qy0 + qz0 * qz0))
                
                # 从当前四元数提取yaw角
                qw, qx, qy, qz = current_quat
                current_yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                
                # 保持roll和pitch为0，但保留yaw角（允许转向）
                # 从yaw角重建四元数（只绕Z轴旋转）
                target_yaw = current_yaw  # 保留当前yaw角，允许转向
                yaw_quat = np.array([
                    np.cos(target_yaw / 2),  # w
                    0.0,  # x (roll = 0)
                    0.0,  # y (pitch = 0)
                    np.sin(target_yaw / 2)   # z (yaw)
                ])
                
                # 平滑应用姿态修正（防止突然变化）
                correction_strength = 0.3  # 姿态修正强度
                self.data.qpos[pos_start + 3:pos_start + 7] = (
                    current_quat * (1 - correction_strength) + yaw_quat * correction_strength
                )
                # 归一化四元数
                quat_norm = np.linalg.norm(self.data.qpos[pos_start + 3:pos_start + 7])
                if quat_norm > 1e-6:
                    self.data.qpos[pos_start + 3:pos_start + 7] /= quat_norm
            
            # 清零Z方向和Y方向速度，以及roll和pitch角速度，防止飘起、左右移动和倾斜
            if (vel_start + 2) < len(self.data.qvel):
                self.data.qvel[vel_start + 2] = 0.0  # Z方向速度
            if (vel_start + 1) < len(self.data.qvel):
                self.data.qvel[vel_start + 1] = 0.0  # Y方向速度
            if (vel_start + 3) < len(self.data.qvel):
                self.data.qvel[vel_start + 3] = 0.0  # 绕X轴角速度（roll）
            if (vel_start + 4) < len(self.data.qvel):
                self.data.qvel[vel_start + 4] = 0.0  # 绕Y轴角速度（pitch）
            
            # 固定脚部位置（保持脚部着地）- 在mj_step前应用
            if (hasattr(self, '_initial_foot_positions') and hasattr(self, '_root_body_id') and 
                self._root_body_id is not None and hasattr(self, '_initial_root_pos') and 
                self._initial_root_pos is not None):
                root_pos = self.data.xpos[self._root_body_id]
                for foot_name, initial_pos in self._initial_foot_positions.items():
                    foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_name)
                    if foot_id >= 0:
                        # 计算初始时脚部相对于根关节的偏移
                        foot_offset = initial_pos - self._initial_root_pos
                        # 计算期望的脚部位置（相对于当前根关节位置）
                        expected_foot_pos = root_pos + foot_offset
                        # 平滑修正脚部位置（特别是Z位置）
                        current_foot_pos = self.data.xpos[foot_id].copy()
                        # 只修正Z位置，保持X和Y相对位置
                        self.data.xpos[foot_id][2] = current_foot_pos[2] * 0.5 + expected_foot_pos[2] * 0.5
        else:
            # mj_step后：固定Z位置、Y位置和姿态（保持身体直立），应用X方向速度控制
            if (pos_start + 2) < len(self.data.qpos):
                self.data.qpos[pos_start + 2] = self._initial_z_height
            if (pos_start + 1) < len(self.data.qpos) and hasattr(self, '_initial_y_position'):
                self.data.qpos[pos_start + 1] = self._initial_y_position
            
            # 稳定姿态：保持身体直立，只允许绕Z轴旋转（yaw）
            if (pos_start + 6) < len(self.data.qpos) and hasattr(self, '_initial_quat'):
                # 获取当前四元数
                current_quat = self.data.qpos[pos_start + 3:pos_start + 7]
                
                # 从当前四元数提取yaw角
                qw, qx, qy, qz = current_quat
                current_yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                
                # 保持roll和pitch为0，但保留yaw角（允许转向）
                yaw_quat = np.array([
                    np.cos(current_yaw / 2),  # w
                    0.0,  # x (roll = 0)
                    0.0,  # y (pitch = 0)
                    np.sin(current_yaw / 2)   # z (yaw)
                ])
                
                # 平滑应用姿态修正
                correction_strength = 0.3
                self.data.qpos[pos_start + 3:pos_start + 7] = (
                    current_quat * (1 - correction_strength) + yaw_quat * correction_strength
                )
                # 归一化四元数
                quat_norm = np.linalg.norm(self.data.qpos[pos_start + 3:pos_start + 7])
                if quat_norm > 1e-6:
                    self.data.qpos[pos_start + 3:pos_start + 7] /= quat_norm
            
            if (vel_start + 2) < len(self.data.qvel):
                self.data.qvel[vel_start + 2] = 0.0  # Z方向速度
            if (vel_start + 1) < len(self.data.qvel):
                self.data.qvel[vel_start + 1] = 0.0  # Y方向速度
            # 清零roll和pitch角速度，防止倾斜
            if (vel_start + 3) < len(self.data.qvel):
                self.data.qvel[vel_start + 3] = 0.0  # 绕X轴角速度（roll）
            if (vel_start + 4) < len(self.data.qvel):
                self.data.qvel[vel_start + 4] = 0.0  # 绕Y轴角速度（pitch）
            
            # X方向速度控制（只在mj_step后，Y方向已固定）
            if (vel_start + 2) <= len(self.data.qvel):
                vx = self.data.qvel[vel_start]
                vy = 0.0  # Y方向速度固定为0
                
                # 根据动作计算期望速度
                desired_vx = 0.0
                desired_vy = 0.0
                
                # 获取躯干朝向（从根关节的四元数）
                yaw = 0.0
                if pos_start + 6 < len(self.data.qpos):
                    # 提取四元数（w, x, y, z）
                    qw = self.data.qpos[pos_start + 3]
                    qx = self.data.qpos[pos_start + 4]
                    qy = self.data.qpos[pos_start + 5]
                    qz = self.data.qpos[pos_start + 6]
                    # 计算绕Z轴的旋转角度（yaw）
                    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                
                # 检测前进/后退动作（通过髋关节前后摆动判断）
                if self._actuator_indices:
                    hip_x_right_idx = self._actuator_indices.get("hip_x_right")
                    hip_x_left_idx = self._actuator_indices.get("hip_x_left")
                    
                    if hip_x_right_idx is not None and hip_x_left_idx is not None:
                        # 计算髋关节前后摆动的差异
                        # 当两腿摆动方向相反时，产生前进力
                        hip_x_right = action[hip_x_right_idx]
                        hip_x_left = action[hip_x_left_idx]
                        hip_x_diff = hip_x_right - hip_x_left
                        
                        # 直接使用差异来计算速度（差异已经包含了方向和强度信息）
                        # 当右腿向前、左腿向后时，差异为正，产生前进速度
                        # 当右腿向后、左腿向前时，差异为负，产生后退速度
                        local_forward_vel = hip_x_diff * self._forward_velocity_gain
                        
                        # 如果差异很小，也可以使用平均摆动幅度作为备用
                        if abs(local_forward_vel) < 0.1:
                            hip_x_avg_amplitude = (abs(hip_x_right) + abs(hip_x_left)) / 2.0
                            if hip_x_avg_amplitude > 0.1:
                                # 根据右腿的摆动方向确定前进方向
                                direction_sign = 1.0 if hip_x_right > 0 else -1.0
                                local_forward_vel = hip_x_avg_amplitude * direction_sign * self._forward_velocity_gain * 0.8
                        
                        # 根据躯干朝向，将局部前进速度转换到世界坐标系（只计算X方向，Y方向已固定）
                        desired_vx = local_forward_vel * np.cos(yaw)
                        # desired_vy = 0.0  # Y方向已固定，不需要计算
                
                # 应用X方向速度平滑过渡（Y方向已固定，不需要控制）
                if abs(desired_vx) > 0.01:
                    # 有主动移动时，使用更平滑的过渡
                    alpha = 0.4  # 平滑系数
                    vx = vx * (1 - alpha) + desired_vx * alpha
                    # 应用轻微阻尼（几乎不衰减，保持速度）
                    vx *= self._xy_damping
                else:
                    # 没有主动移动时，快速停止
                    damping = 0.85  # 增大阻尼，使停止更快
                    vx *= damping
                    
                    # 如果速度很小，直接清零以避免微小震荡
                    if abs(vx) < 0.05:
                        vx = 0.0
                
                # 限制最大速度（只限制X方向，Y方向已固定）
                if abs(vx) > self._max_xy_velocity:
                    vx = np.sign(vx) * self._max_xy_velocity
                
                self.data.qvel[vel_start] = vx
                self.data.qvel[vel_start + 1] = 0.0  # Y方向速度固定为0，保持在走廊中心
            
            # 转向角速度控制：检测转向动作并应用绕Z轴的角速度
            if (vel_start + 5) < len(self.data.qvel) and self._actuator_indices:
                # 检测转向动作（通过hip_z或abdomen_z关节）
                hip_z_right_idx = self._actuator_indices.get("hip_z_right")
                hip_z_left_idx = self._actuator_indices.get("hip_z_left")
                abdomen_z_idx = self._actuator_indices.get("abdomen_z")
                
                turn_angular_vel = 0.0
                if hip_z_right_idx is not None and hip_z_left_idx is not None:
                    # 计算转向强度（通过髋关节外展差异）
                    hip_z_right = action[hip_z_right_idx] if hip_z_right_idx < len(action) else 0.0
                    hip_z_left = action[hip_z_left_idx] if hip_z_left_idx < len(action) else 0.0
                    hip_z_diff = hip_z_right - hip_z_left
                    turn_angular_vel += hip_z_diff * 0.5  # 转向角速度增益
                
                if abdomen_z_idx is not None and abdomen_z_idx < len(action):
                    # 躯干旋转也贡献转向角速度
                    abdomen_z = action[abdomen_z_idx]
                    turn_angular_vel += abdomen_z * 0.8  # 躯干旋转的转向增益更大
                
                # 应用转向角速度（绕Z轴旋转，索引vel_start+5是绕Z轴的角速度）
                current_angular_vel_z = self.data.qvel[vel_start + 5]
                # 平滑过渡转向角速度
                if abs(turn_angular_vel) > 0.01:
                    alpha = 0.5  # 转向角速度平滑系数
                    new_angular_vel_z = current_angular_vel_z * (1 - alpha) + turn_angular_vel * alpha
                    # 限制最大转向角速度
                    max_turn_angular_vel = 2.0  # 最大转向角速度（弧度/秒）
                    new_angular_vel_z = np.clip(new_angular_vel_z, -max_turn_angular_vel, max_turn_angular_vel)
                    self.data.qvel[vel_start + 5] = new_angular_vel_z
                else:
                    # 没有转向指令时，逐渐减小转向角速度
                    self.data.qvel[vel_start + 5] *= 0.9
            
            # 姿态稳定控制：防止头部高度持续下降
            if not before_step and self._initial_head_height is not None:
                head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
                if head_id >= 0:
                    current_head_height = self.data.xpos[head_id][2]
                    head_height_error = self._initial_head_height - current_head_height
                    
                    # 如果头部高度下降超过阈值，应用姿态稳定控制
                    if head_height_error > 0.05:  # 下降超过5cm
                        # 通过调整躯干俯仰角来恢复姿态
                        # 计算需要的俯仰角修正（前倾以恢复高度）
                        pitch_correction = min(head_height_error * self._head_stability_gain, 0.3)  # 限制最大修正
                        
                        # 获取当前躯干俯仰角（从四元数计算）
                        if pos_start + 6 < len(self.data.qpos):
                            qw = self.data.qpos[pos_start + 3]
                            qx = self.data.qpos[pos_start + 4]
                            qy = self.data.qpos[pos_start + 5]
                            qz = self.data.qpos[pos_start + 6]
                            
                            # 计算当前俯仰角（pitch）
                            sin_pitch = 2.0 * (qw * qy - qz * qx)
                            current_pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
                            
                            # 计算目标俯仰角（稍微前倾以恢复高度）
                            target_pitch = current_pitch + pitch_correction
                            
                            # 通过调整abdomen_y执行器来修正姿态（在动作中应用）
                            # 注意：这里只是记录修正值，实际应用在动作生成时
                            # 由于动作已经生成，这里通过直接调整根关节姿态来快速响应
                            # 但为了不影响动作生成，我们只在严重偏差时应用
                            if head_height_error > 0.15:  # 下降超过15cm时，直接修正姿态
                                # 计算新的四元数（绕X轴旋转）
                                pitch_quat = np.array([
                                    np.cos(target_pitch / 2),
                                    np.sin(target_pitch / 2),
                                    0.0,
                                    0.0
                                ])
                                # 简化处理：只在小范围内修正
                                correction_factor = 0.1  # 每次只修正10%
                                self.data.qpos[pos_start + 3] = self.data.qpos[pos_start + 3] * (1 - correction_factor) + pitch_quat[0] * correction_factor
                                self.data.qpos[pos_start + 4] = self.data.qpos[pos_start + 4] * (1 - correction_factor) + pitch_quat[1] * correction_factor
            
            # 固定脚部位置（保持脚部着地）- 在mj_step后应用
            if (not before_step and hasattr(self, '_initial_foot_positions') and 
                hasattr(self, '_root_body_id') and self._root_body_id is not None and 
                hasattr(self, '_initial_root_pos') and self._initial_root_pos is not None):
                # 需要先更新物理状态以获取最新的xpos
                mujoco.mj_forward(self.model, self.data)
                root_pos = self.data.xpos[self._root_body_id]
                for foot_name, initial_pos in self._initial_foot_positions.items():
                    foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_name)
                    if foot_id >= 0:
                        # 计算初始时脚部相对于根关节的偏移
                        foot_offset = initial_pos - self._initial_root_pos
                        # 计算期望的脚部位置（相对于当前根关节位置）
                        expected_foot_pos = root_pos + foot_offset
                        # 平滑修正脚部位置（特别是Z位置）
                        current_foot_pos = self.data.xpos[foot_id].copy()
                        # 只修正Z位置，保持X和Y相对位置
                        self.data.xpos[foot_id][2] = current_foot_pos[2] * 0.5 + expected_foot_pos[2] * 0.5
    
    def step(self, action):
        """执行动作并推进环境"""
        self.current_step += 1
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        
        for _ in range(self.control_steps):
            # mj_step前应用约束
            self._apply_zero_gravity_constraints(action, before_step=True)
            
            mujoco.mj_step(self.model, self.data)
            
            # mj_step后应用约束
            self._apply_zero_gravity_constraints(action, before_step=False)
            
            # 更新物理状态
            if not self.use_gravity:
                mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_observation()
        reward = self._get_reward()
        done = self.current_step >= self._max_episode_steps
        
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        torso_z = self.data.xpos[torso_id][2]
        if torso_z < 0.5:
            done = True
            reward -= 1.0
        return obs, reward, done

    def render(self, viewer_handle=None):
        """渲染画面"""
        if viewer_handle is not None:
            with viewer_handle.lock():
                viewer_handle.sync()


def main():
    # 将环境切换为“无重力”模式
    env = GapCorridorEnvironment(corridor_length=100, corridor_width=10, use_gravity=False)
    
    controller = KeyboardController(env.model.nu, env.get_actuator_indices())
    obs = env.reset()
    total_reward = 0.0
    
    print("\n" + "="*80)
    print("🚀 环境初始化完成")
    print("-"*80)
    print(f"   执行器数量: {env.model.nu}")
    print(f"   关节数量: {env.model.nq}")
    print(f"   观测维度: {len(obs)}")
    print(f"   重力模式: {'启用' if env.use_gravity else '禁用（无重力模式）'}")
    print(f"   控制时间步: {env.control_timestep:.3f}s")
    print(f"   物理时间步: {env.timestep:.3f}s")
    print(f"   最大Episode步数: {env._max_episode_steps}")
    print("="*80)
    
    print("\n📺 启动MuJoCo交互式查看器...")
    print("   提示: 在查看器窗口中按键盘进行控制")
    print("   提示: 按 ESC 或关闭窗口退出程序")
    
    try:
        viewer_handle = mujoco.viewer.launch_passive(
            env.model, 
            env.data,
            key_callback=controller.key_callback,
            show_left_ui=True,
            show_right_ui=True
        )
        
        print("\n✅ 查看器已启动，开始仿真循环...")
        print(f"   状态报告将每100步输出一次")
        print("")
        
        step = 0
        last_move_state = None  # 记录上次移动状态，用于检测状态变化
        
        while viewer_handle.is_running() and not controller.should_exit():
            if controller.should_reset():
                obs = env.reset()
                total_reward = 0.0
                step = 0
                # 重置移动状态
                controller.move_forward = False
                controller.move_backward = False
                controller.turn_left = False
                controller.turn_right = False
                controller.step_time = 0.0
                # 重置PID控制器
                controller.velocity_pid['integral'] = np.array([0.0, 0.0])
                controller.velocity_pid['last_error'] = np.array([0.0, 0.0])
                controller.target_velocity = np.array([0.0, 0.0])
                controller.smoothed_action = np.zeros(controller.action_dim)
                controller.action_history.clear()
                controller.neural_history.clear()
                # 重置深度学习控制器状态
                if controller.deep_controller is not None:
                    controller.deep_controller.reset_lstm_state()
                controller.last_state = None
                last_move_state = None
                controller.clear_reset_flag()
            
            # 检测移动状态变化，重置PID控制器以避免震荡
            current_move_state = (
                controller.move_forward,
                controller.move_backward,
                controller.turn_left,
                controller.turn_right
            )
            if current_move_state != last_move_state:
                # 状态改变时，重置PID积分项，避免累积误差导致震荡
                controller.velocity_pid['integral'] = np.array([0.0, 0.0])
                controller.velocity_pid['last_error'] = np.array([0.0, 0.0])
                last_move_state = current_move_state
            
            # 获取当前速度（用于PID控制）
            if not env.use_gravity and env._root_joint_qvel_start is not None:
                vel_start = env._root_joint_qvel_start
                if (vel_start + 2) <= len(env.data.qvel):
                    current_vel = np.array([
                        env.data.qvel[vel_start],
                        env.data.qvel[vel_start + 1]
                    ])
                else:
                    current_vel = np.array([0.0, 0.0])
            else:
                current_vel = np.array([0.0, 0.0])
            
            # 获取动作（传入控制步长、当前速度、状态和奖励）
            action = controller.get_action(
                dt=env.control_timestep, 
                current_velocity=current_vel,
                state=obs,
                reward=total_reward
            )
            obs, reward, done = env.step(action)
            total_reward += reward
            
            # 存储经验并训练深度学习控制器
            if controller.use_deep_learning and controller.deep_controller is not None:
                # 存储经验（使用上一个状态）
                if controller.last_state is not None:
                    next_obs = obs if not done else None
                    controller.deep_controller.store_experience(
                        controller.last_state, action, reward, next_obs, done
                    )
                
                # 更新上一个状态
                controller.last_state = obs.copy()
                
                # 定期训练
                controller.deep_controller.step_count += 1
                if controller.deep_controller.step_count % controller.deep_controller.update_frequency == 0:
                    controller.deep_controller.train_step()
                
                # 重置时清空LSTM状态
                if done:
                    controller.deep_controller.reset_lstm_state()
            
            env.render(viewer_handle)
            
            if step % 200 == 0:
                # 获取身体位置
                torso_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
                torso_pos = env.data.xpos[torso_id].copy() if torso_id >= 0 else None
                                
                # 获取速度信息
                linear_vel = np.zeros(6)
                angular_vel = np.zeros(6)
                if torso_id >= 0:
                    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, torso_id, linear_vel, 0)
                    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, torso_id, angular_vel, 1)
                vx, vy, vz = linear_vel[0], linear_vel[1], linear_vel[2]
                angular_vz = angular_vel[5]  # 绕Z轴角速度（转向）
                speed = np.sqrt(vx**2 + vy**2)
                
                # 获取步态相位信息（如果有深度学习控制器）
                gait_info = ""
                if controller.deep_controller is not None:
                    gait_phase_deg = np.degrees(controller.deep_controller.gait_phase) % 360
                    gait_info = f"步态相位: {gait_phase_deg:.1f}°, 步频: {controller.deep_controller.gait_frequency:.2f}Hz"
                
                # 获取动作统计信息
                action_magnitude = np.max(np.abs(action))
                action_mean = np.mean(np.abs(action))
                action_std = np.std(action)
                
                # 获取键盘控制状态
                control_state = []
                if controller.move_forward:
                    control_state.append("前进")
                if controller.move_backward:
                    control_state.append("后退")
                if controller.turn_left:
                    control_state.append("左转")
                if controller.turn_right:
                    control_state.append("右转")
                if not control_state:
                    control_state.append("静止")
                control_str = "+".join(control_state) if control_state else "静止"
                
                # 获取步态时间信息
                step_time_info = f"步态时间: {controller.step_time:.2f}s"
                
                # 计算运行时间（模拟）
                sim_time = step * env.control_timestep
                
                # 获取奖励信息（当前步奖励和累计奖励）
                recent_reward = reward  # 当前步的奖励
                avg_reward_per_step = total_reward / max(step, 1)
                
                # 打印分隔线
                print("\n" + "="*80)
                print(f"📊 状态报告 [Step {step} | 模拟时间: {sim_time:.2f}s]")
                print("-"*80)
                
                # 控制状态
                print(f"🎮 控制状态: {control_str:20s} | {step_time_info}")
                if gait_info:
                    print(f"🚶 {gait_info}")
                
                # 速度和运动信息（精简）
                print(f"\n⚡ 速度: |V|={speed:.3f} m/s, vx={vx:+.3f}, vy={vy:+.3f}, yaw_rate={np.degrees(angular_vz):+.2f} °/s")
                
                # 关键位置
                if torso_pos is not None:
                    print(f"📍 位置: X={torso_pos[0]:+.3f}, Y={torso_pos[1]:+.3f}, Z={torso_pos[2]:+.3f} m")
                
                # 动作信息
                print(f"🎯 动作: max={action_magnitude:.3f}, mean={action_mean:.3f}, std={action_std:.3f}")
                
                # 奖励信息
                print(f"🏆 奖励: step={recent_reward:+.4f}, total={total_reward:+.4f}, avg/step={avg_reward_per_step:+.4f}")
                
                # 深度学习信息（如果启用）
                if controller.use_deep_learning and controller.deep_controller is not None:
                    buffer_size = len(controller.deep_controller.replay_buffer)
                    max_buffer = controller.deep_controller.replay_buffer.maxlen
                    print(f"🧠 训练: buffer={buffer_size}/{max_buffer}, steps={controller.deep_controller.step_count}")
                
                print("="*80 + "\n")
            
            if done:
                # 计算Episode统计信息
                episode_duration = step * env.control_timestep
                avg_reward_per_step = total_reward / max(step, 1)
                
                # 获取最终位置信息
                final_torso_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
                
                print("\n" + "="*80)
                print("🎯 Episode 结束")
                print("-"*80)
                print(f"   总步数: {step}")
                print(f"   持续时间: {episode_duration:.2f}s")
                print(f"   累计奖励: {total_reward:+.4f}")
                print(f"   平均奖励/步: {avg_reward_per_step:+.4f}")
                
                # 获取最终位置信息
                if final_torso_id >= 0:
                    final_torso_pos = env.data.xpos[final_torso_id]
                    print(f"   最终位置: X={final_torso_pos[0]:+.3f}, Y={final_torso_pos[1]:+.3f}, Z={final_torso_pos[2]:+.3f} m")
                    # 计算前进距离（从初始位置）
                    initial_pos = env._root_joint_qpos_start
                    if initial_pos is not None and (initial_pos + 2) < len(env.data.qpos):
                        initial_x = env.data.qpos[initial_pos]
                        distance_traveled = final_torso_pos[0] - initial_x
                        print(f"   前进距离: {distance_traveled:+.3f} m")
                
                print("="*80 + "\n")
                
                obs = env.reset()
                total_reward = 0.0
                step = 0
                # 重置深度学习控制器状态
                if controller.deep_controller is not None:
                    controller.deep_controller.reset_lstm_state()
                controller.last_state = None
            
            step += 1
            time.sleep(0.01)
        
        viewer_handle.close()
        print("\n查看器已关闭")
        
    except Exception as e:
        print(f"无法启动查看器: {e}")
        import traceback
        traceback.print_exc()
    
    print("程序已退出")

if __name__ == "__main__":
    main()
