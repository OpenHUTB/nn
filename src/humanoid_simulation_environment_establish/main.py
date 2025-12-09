import numpy as np
import mujoco
from mujoco import viewer
import time
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import deque


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
        self.step_frequency = 1.2  # 步频 (Hz)
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
        
        # 简单的神经网络控制器（用于动作平滑）
        self.use_neural_smoothing = True
        self._init_neural_smoother()

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
        """创建步行动作：基于周期的左右腿交替摆动，更自然的步态"""
        action = np.zeros(self.action_dim)
        
        if not self.actuator_indices:
            return action
        
        # 计算步行动作相位
        phase = 2 * np.pi * self.step_time * self.step_frequency
        direction = 1 if forward else -1
        
        # 使用更自然的步态模式：区分支撑相和摆动相
        # 右腿相位
        right_phase = phase
        # 左腿相位（相差180度）
        left_phase = phase + np.pi
        
        # 计算摆动相和支撑相（使用平滑的过渡）
        # 摆动相：0到π，支撑相：π到2π
        right_swing_phase = (right_phase % (2 * np.pi)) / np.pi  # 归一化到0-2
        left_swing_phase = (left_phase % (2 * np.pi)) / np.pi
        
        # 右腿：更自然的步态
        # 髋关节前后摆动（主要推进力）
        # 右腿向前摆动时产生推进力
        right_hip_swing = 0.6 * direction * np.sin(right_phase)
        self._set_action(action, "hip_x_right", right_hip_swing)
        
        # 髋关节上下（抬腿）
        right_hip_lift = 0.2 * max(0, np.sin(right_phase))  # 只在摆动相抬腿
        self._set_action(action, "hip_y_right", -right_hip_lift)
        
        # 膝关节（在摆动相弯曲，支撑相伸直）
        right_knee_angle = 0.5 * (1 - np.cos(right_phase))  # 0到1的平滑变化
        self._set_action(action, "knee_right", 0.6 * right_knee_angle)
        
        # 踝关节（配合抬腿）
        self._set_action(action, "ankle_y_right", -0.15 * max(0, np.sin(right_phase)))
        self._set_action(action, "ankle_x_right", 0.15 * np.sin(right_phase))
        
        # 左腿（相位相反，摆动方向也相反以产生推进力）
        # 关键：左腿的摆动与右腿相反（当右腿向前时，左腿向后）
        # 右腿：0.6 * direction * sin(phase)
        # 左腿：-0.6 * direction * sin(phase)  （直接使用负号，确保与右腿相反）
        # 这样差异 = 0.6*direction*sin(phase) - (-0.6*direction*sin(phase)) = 1.2*direction*sin(phase)
        # 能产生有效的推进力！
        left_hip_swing = -0.6 * direction * np.sin(right_phase)  # 直接使用右腿相位的负值
        self._set_action(action, "hip_x_left", left_hip_swing)
        
        left_hip_lift = 0.2 * max(0, np.sin(left_phase))
        self._set_action(action, "hip_y_left", -left_hip_lift)
        
        left_knee_angle = 0.5 * (1 - np.cos(left_phase))
        self._set_action(action, "knee_left", 0.6 * left_knee_angle)
        
        self._set_action(action, "ankle_y_left", -0.15 * max(0, np.sin(left_phase)))
        self._set_action(action, "ankle_x_left", -0.15 * np.sin(left_phase))
        
        # 躯干控制（轻微前倾以辅助前进）
        self._set_action(action, "abdomen_y", 0.2 * direction)
        self._set_action(action, "abdomen_x", 0.1 * turn_direction)
        
        # 转向控制（更平滑，增大转向幅度）
        if turn_direction != 0:
            turn_strength = 0.5 * turn_direction  # 从0.3增大到0.5
            # 转向时，外侧腿稍微外展，内侧腿稍微内收
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
        
        return action
    
    def _create_turning_only_action(self, turn_direction):
        """创建仅转向动作（不产生腿部摆动，只在原地转向）"""
        action = np.zeros(self.action_dim)
        
        if not self.actuator_indices:
            return action
        
        # 只设置转向相关的动作，不产生腿部摆动
        # 转向控制通过髋关节外展实现
        turn_strength = 0.3 * turn_direction  # 减小转向强度
        self._set_action(action, "hip_z_right", turn_strength)
        self._set_action(action, "hip_z_left", -turn_strength)
        
        # 可以添加轻微的躯干倾斜来辅助转向
        self._set_action(action, "abdomen_x", 0.1 * turn_direction)
        
        return action
    
    def _process_key(self, key):
        """处理按键输入"""
        if isinstance(key, str) and key.startswith('\x1b['):
            key_char = None  # 方向键用特殊序列表示
        else:
            key_char = key if isinstance(key, str) and len(key) == 1 else None
        
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
                    # 停止移动时，立即重置相关状态
                    setattr(self, attr, False)
                    self.step_time = 0.0  # 重置步行动作时间
                    # 快速清零平滑动作
                    if not (self.move_forward or self.move_backward or self.turn_left or self.turn_right):
                        self.smoothed_action = np.zeros(self.action_dim)
                    print(f"[键盘] {stop_msg}")
                else:
                    setattr(self, attr, True)
                    if hasattr(self, opposite_attr):
                        setattr(self, opposite_attr, False)
                    # 开始移动时，重置步行动作时间
                    self.step_time = 0.0
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
        """更新步行动作时间（只在有键盘输入时更新）"""
        if not self.paused and (self.move_forward or self.move_backward or self.turn_left or self.turn_right):
            self.step_time += dt
        else:
            # 没有键盘输入时，立即重置时间，停止动作
            self.step_time = 0.0
    
    def get_action(self, dt=0.03, current_velocity=None):
        """获取当前控制动作（基于键盘输入的离散控制）"""
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
        
        if not has_movement:
            # 没有键盘输入时，立即停止并清零动作
            self.step_time = 0.0
            # 快速衰减到零
            self.smoothed_action = self.smoothed_action * 0.5
            if np.max(np.abs(self.smoothed_action)) < 0.01:
                self.smoothed_action = np.zeros(self.action_dim)
            self.current_action = self.smoothed_action.copy()
            return self.current_action.copy()
        
        # 有键盘输入时，更新步行动作时间
        self.update_step_time(dt)
        
        # 根据移动状态创建动作
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
            # 只转向时，不产生腿部摆动，只在原地转向
            turn_dir = -1 if self.turn_left else 1
            raw_action = self._create_turning_only_action(turn_dir, dt=dt)
        else:
            # 不应该到这里，但以防万一
            raw_action = np.zeros(self.action_dim)
        
        # 应用动作平滑（但使用更小的平滑系数，使停止更快）
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
        
        # 无重力模式：记录根关节的初始Z高度和姿态
        if not self.use_gravity and self._root_joint_qpos_start is not None:
            self._initial_z_height = float(self.data.qpos[self._root_joint_qpos_start + 2])
            print(f"[无重力模式] 记录初始Z高度: {self._initial_z_height:.4f}，允许上身自由移动")
        
        return self._get_observation()

    def _get_observation(self):
        """获取观测（关节位置、速度、躯干位置）"""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        torso_pos = self.data.xpos[torso_id].copy()
        return np.concatenate([qpos, qvel, torso_pos])

    def _get_reward(self):
        """计算奖励：前进速度（沿走廊X轴）+ 空隙掉落惩罚"""
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
        """应用无重力模式的约束：只固定Z高度，允许上身自由移动，并根据动作主动施加速度"""
        if self.use_gravity or self._initial_z_height is None:
            return
        
        pos_start = self._root_joint_qpos_start
        vel_start = self._root_joint_qvel_start
        
        if pos_start is None or vel_start is None:
            return
        
        if before_step:
            # mj_step前：只固定Z位置，不干扰其他物理量
            if (pos_start + 2) < len(self.data.qpos):
                self.data.qpos[pos_start + 2] = self._initial_z_height
            # 清零Z方向速度，防止飘起
            if (vel_start + 2) < len(self.data.qvel):
                self.data.qvel[vel_start + 2] = 0.0
        else:
            # mj_step后：固定Z位置，应用XY速度控制
            if (pos_start + 2) < len(self.data.qpos):
                self.data.qpos[pos_start + 2] = self._initial_z_height
            if (vel_start + 2) < len(self.data.qvel):
                self.data.qvel[vel_start + 2] = 0.0
            
            # XY速度控制（只在mj_step后）
            if (vel_start + 2) <= len(self.data.qvel):
                vx, vy = self.data.qvel[vel_start], self.data.qvel[vel_start + 1]
                
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
                        
                        # 根据躯干朝向，将局部前进速度转换到世界坐标系
                        desired_vx = local_forward_vel * np.cos(yaw)
                        desired_vy = local_forward_vel * np.sin(yaw)
                
                # 应用速度平滑过渡（使用更平滑的混合策略，减少震荡）
                if abs(desired_vx) > 0.01 or abs(desired_vy) > 0.01:
                    # 有主动移动时，使用更平滑的过渡
                    # 使用更小的平滑系数，减少震荡
                    alpha = 0.4  # 平滑系数
                    vx = vx * (1 - alpha) + desired_vx * alpha
                    vy = vy * (1 - alpha) + desired_vy * alpha
                    # 应用轻微阻尼（几乎不衰减，保持速度）
                    vx *= self._xy_damping
                    vy *= self._xy_damping
                else:
                    # 没有主动移动时，快速停止
                    damping = 0.85  # 增大阻尼，使停止更快
                    vx *= damping
                    vy *= damping
                    
                    # 如果速度很小，直接清零以避免微小震荡
                    if abs(vx) < 0.05:
                        vx = 0.0
                    if abs(vy) < 0.05:
                        vy = 0.0
                
                # 限制最大速度
                speed = np.sqrt(vx * vx + vy * vy)
                if speed > self._max_xy_velocity:
                    scale = self._max_xy_velocity / speed
                    vx *= scale
                    vy *= scale
                
                self.data.qvel[vel_start] = vx
                self.data.qvel[vel_start + 1] = vy
    
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
    
    print("\n环境已初始化")
    print(f"执行器数量: {env.model.nu}")
    print(f"关节数量: {env.model.nq}")
    
    controller = KeyboardController(env.model.nu, env.get_actuator_indices())
    obs = env.reset()
    total_reward = 0.0
    
    print("\n启动MuJoCo交互式查看器...")
    print("按 ESC 或关闭窗口退出程序")
    
    try:
        viewer_handle = mujoco.viewer.launch_passive(
            env.model, 
            env.data,
            key_callback=controller.key_callback,
            show_left_ui=True,
            show_right_ui=True
        )
        
        print("\n查看器已启动，开始仿真循环...")
        
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
            
            # 获取动作（传入控制步长和当前速度）
            action = controller.get_action(dt=env.control_timestep, current_velocity=current_vel)
            obs, reward, done = env.step(action)
            total_reward += reward
            
            env.render(viewer_handle)
            
            if step % 100 == 0:
                torso_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
                torso_pos = env.data.xpos[torso_id]
                head_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "head")
                head_pos = env.data.xpos[head_id] if head_id >= 0 else None
                if head_pos is not None:
                    print(f"Step {step}: 躯干位置 = {torso_pos}, 头部位置 = {head_pos}, 累计奖励 = {total_reward:.2f}")
                else:
                    print(f"Step {step}: 躯干位置 = {torso_pos}, 累计奖励 = {total_reward:.2f}")
            
            if done:
                print(f"\nEpisode finished. Total reward: {total_reward:.2f}")
                obs = env.reset()
                total_reward = 0.0
                step = 0
            
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
