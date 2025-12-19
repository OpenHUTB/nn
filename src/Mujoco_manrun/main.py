import mujoco
import numpy as np
from mujoco import viewer
import time
import os
import sys
import threading

# ===================== 新增：ROS话题接收模块（可选兼容，无ROS不报错） =====================
class ROSCmdVelHandler(threading.Thread):
    """ROS /cmd_vel话题接收线程，映射linear.x→速度，angular.z→转向"""
    def __init__(self, stabilizer):
        super().__init__(daemon=True)
        self.stabilizer = stabilizer
        self.running = True
        self.has_ros = False
        self.twist_msg = None

        # 尝试导入ROS库，无ROS则跳过
        try:
            import rospy
            from geometry_msgs.msg import Twist
            self.rospy = rospy
            self.Twist = Twist
            self.has_ros = True
        except ImportError:
            print("[ROS提示] 未检测到ROS环境，跳过/cmd_vel话题监听（仅保留键盘控制）")
            return

        # 初始化ROS节点
        try:
            if not self.rospy.core.is_initialized():
                self.rospy.init_node('humanoid_cmd_vel_listener', anonymous=True)
            # 订阅/cmd_vel话题（队列大小1，避免延迟）
            self.sub = self.rospy.Subscriber(
                "/cmd_vel", self.Twist, self._cmd_vel_callback, queue_size=1
            )
            print("[ROS提示] 已启动/cmd_vel话题监听：")
            print("  - linear.x (0.1~1.0) → 行走速度（0.1=最慢，1.0=最快）")
            print("  - angular.z (-1.0~1.0) → 转向角度（正=左转，负=右转，映射±0.3rad）")
        except Exception as e:
            print(f"[ROS提示] ROS节点初始化失败：{e}")
            self.has_ros = False

    def _cmd_vel_callback(self, msg):
        """回调函数：解析/cmd_vel并映射到速度/转向"""
        self.twist_msg = msg
        # 1. linear.x → 行走速度（限幅0.1~1.0）
        target_speed = np.clip(msg.linear.x, 0.1, 1.0)
        # 2. angular.z → 转向角度（-1.0~1.0映射到-0.3~0.3rad）
        target_turn = np.clip(msg.angular.z, -1.0, 1.0) * 0.3  # 缩放系数0.3

        # 更新到控制器（优先级高于键盘，避免冲突）
        self.stabilizer.set_walk_speed(target_speed)
        self.stabilizer.set_turn_angle(target_turn)

        # 自动触发行走（如果接收到速度指令且当前是停止状态）
        if target_speed > 0.1 and self.stabilizer.state == "STAND":
            self.stabilizer.set_state("WALK")

        # 调试输出
        if self.stabilizer.data.time % 0.5 < 0.1:  # 避免刷屏，每0.5秒输出一次
            print(f"[ROS指令] 速度={target_speed:.2f} | 转向={target_turn:.2f}rad")

    def run(self):
        """线程主循环（ROS自旋）"""
        if not self.has_ros:
            return
        while self.running and not self.rospy.is_shutdown():
            self.rospy.spin_once()
            time.sleep(0.01)

    def stop(self):
        self.running = False

# ===================== 原始代码：键盘输入监听线程（完全保留） =====================
class KeyboardInputHandler(threading.Thread):
    def __init__(self, stabilizer):
        super().__init__(daemon=True)
        self.stabilizer = stabilizer
        self.running = True

    def run(self):
        print("\n===== 控制指令说明 =====")
        print("w: 开始行走 | s: 停止行走 | e: 紧急停止 | r: 恢复站立")
        print("a: 左转 | d: 右转 | 空格: 原地转向 | z: 减速 | x: 加速")
        print("========================\n")
        while self.running:
            try:
                # 非阻塞键盘输入（兼容不同系统）
                if sys.platform == "win32":
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8').lower()
                        self._handle_key(key)
                else:
                    import select
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1).lower()
                        self._handle_key(key)
                time.sleep(0.01)
            except:
                continue

    def _handle_key(self, key):
        if key == 'w':
            self.stabilizer.set_state("WALK")
            print(f"[指令] 切换为行走状态 | 当前速度: {self.stabilizer.walk_speed:.2f} | 转向: {self.stabilizer.turn_angle:.2f}")
        elif key == 's':
            self.stabilizer.set_state("STOP")
            print("[指令] 切换为停止状态")
        elif key == 'e':
            self.stabilizer.set_state("EMERGENCY")
            print("[指令] 触发紧急停止")
        elif key == 'r':
            self.stabilizer.set_state("STAND")
            print("[指令] 恢复站立姿态")
        elif key == 'a':
            self.stabilizer.set_turn_angle(self.stabilizer.turn_angle + 0.05)
            print(f"[指令] 左转 | 当前转向角度: {self.stabilizer.turn_angle:.2f}rad")
        elif key == 'd':
            self.stabilizer.set_turn_angle(self.stabilizer.turn_angle - 0.05)
            print(f"[指令] 右转 | 当前转向角度: {self.stabilizer.turn_angle:.2f}rad")
        elif key == ' ':
            self.stabilizer.set_turn_angle(0.2 if self.stabilizer.turn_angle <= 0 else -0.2)
            print(f"[指令] 原地转向 | 当前转向角度: {self.stabilizer.turn_angle:.2f}rad")
        elif key == 'z':
            self.stabilizer.set_walk_speed(self.stabilizer.walk_speed - 0.1)
            print(f"[指令] 减速 | 当前速度: {self.stabilizer.walk_speed:.2f}")
        elif key == 'x':
            self.stabilizer.set_walk_speed(self.stabilizer.walk_speed + 0.1)
            print(f"[指令] 加速 | 当前速度: {self.stabilizer.walk_speed:.2f}")

# ===================== 原始代码：CPG中枢模式发生器（完全保留） =====================
class CPGOscillator:
    def __init__(self, freq=0.5, amp=0.4, phase=0.0, coupling_strength=0.2):
        self.base_freq = freq  # 基础频率（对应原始步态周期2s）
        self.base_amp = amp    # 基础振幅（对应原始步长）
        self.freq = freq
        self.amp = amp
        self.phase = phase     # 初始相位（左右腿差π）
        self.coupling = coupling_strength  # 腿间耦合强度
        self.state = np.array([np.sin(phase), np.cos(phase)])  # 振荡器状态(x,y)

    def update(self, dt, target_phase=0.0):
        """更新CPG状态，返回关节目标偏移量"""
        # 范德波尔振荡器方程（生物节律更自然，抗干扰）
        mu = 1.0  # 非线性系数，控制振荡收敛性
        x, y = self.state
        dx = 2 * np.pi * self.freq * y + self.coupling * np.sin(target_phase - self.phase)
        dy = 2 * np.pi * self.freq * (mu * (1 - x**2) * y - x)
        # 更新状态（积分）
        self.state += np.array([dx, dy]) * dt
        self.phase = np.arctan2(self.state[0], self.state[1])  # 更新相位
        # 返回当前输出（关节目标偏移量）
        return self.amp * self.state[0]

    def reset(self):
        """重置CPG状态"""
        self.freq = self.base_freq
        self.amp = self.base_amp
        self.phase = 0.0 if self.base_phase == 0.0 else np.pi
        self.state = np.array([np.sin(self.phase), np.cos(self.phase)])

    @property
    def base_phase(self):
        return 0.0 if self.phase < np.pi else np.pi

# ===================== 原始代码：人形机器人控制器（完全保留，仅新增ROS线程启动） =====================
class HumanoidStabilizer:
    """适配humanoid.xml模型的稳定站立与行走控制器（新增转向/变速/状态机）"""

    def __init__(self, model_path):
        # 类型检查与模型加载（原始逻辑完全保留）
        if not isinstance(model_path, str):
            raise TypeError(f"模型路径必须是字符串，当前是 {type(model_path)} 类型")

        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}\n请检查路径和文件完整性")

        # 仿真核心参数（原始逻辑保留）
        self.sim_duration = 120.0
        self.dt = self.model.opt.timestep
        self.init_wait_time = 4.0
        self.model.opt.gravity[2] = -9.81
        self.model.opt.iterations = 200
        self.model.opt.tolerance = 1e-8

        # 关节名称映射（原始逻辑完全保留）
        self.joint_names = [
            "abdomen_z", "abdomen_y", "abdomen_x",
            "hip_x_right", "hip_z_right", "hip_y_right",
            "knee_right", "ankle_y_right", "ankle_x_right",
            "hip_x_left", "hip_z_left", "hip_y_left",
            "knee_left", "ankle_y_left", "ankle_x_left",
            "shoulder1_right", "shoulder2_right", "elbow_right",
            "shoulder1_left", "shoulder2_left", "elbow_left"
        ]
        self.joint_name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
        self.num_joints = len(self.joint_names)

        # PD控制增益（原始逻辑完全保留）
        self.kp_roll = 120.0
        self.kd_roll = 40.0
        self.kp_pitch = 100.0
        self.kd_pitch = 35.0
        self.kp_yaw = 30.0
        self.kd_yaw = 15.0
        self.kp_hip = 250.0
        self.kd_hip = 45.0
        self.kp_knee = 280.0
        self.kd_knee = 50.0
        self.kp_ankle = 200.0
        self.kd_ankle = 60.0
        self.kp_waist = 40.0
        self.kd_waist = 20.0
        self.kp_arm = 20.0
        self.kd_arm = 20.0

        # 重心目标（原始逻辑保留）
        self.com_target = np.array([0.05, 0.0, 0.78])
        self.kp_com = 50.0
        self.foot_contact_threshold = 1.5

        # 状态变量（原始逻辑保留，新增状态机/CPG/转向/变速参数）
        self.joint_targets = np.zeros(self.num_joints)
        self.prev_com = np.zeros(3)
        self.foot_contact = np.zeros(2)
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.integral_limit = 0.15

        # 新增：运动状态机（核心）
        self.state = "STAND"  # 初始状态：STAND/WALK/STOP/EMERGENCY
        self.state_map = {
            "STAND": self._state_stand,    # 站立（初始稳定）
            "WALK": self._state_walk,      # 行走（CPG驱动）
            "STOP": self._state_stop,      # 停止（关节归零）
            "EMERGENCY": self._state_emergency  # 急停（力矩清零）
        }

        # 新增：CPG振荡器（替代原始固定正弦步态，兼容变速）
        self.right_leg_cpg = CPGOscillator(freq=0.5, amp=0.4, phase=0.0)  # 右腿初始相位0
        self.left_leg_cpg = CPGOscillator(freq=0.5, amp=0.4, phase=np.pi)  # 左腿初始相位π（交替）
        self.gait_phase = 0.0  # 保留原始相位变量，兼容调试输出

        # 新增：转向参数
        self.turn_angle = 0.0  # 转向角度（左正右负，范围±0.3rad）
        self.turn_gain = 0.1   # 躯干偏航增益

        # 新增：变速参数
        self.walk_speed = 0.5  # 行走速度（0.1~1.0，映射到CPG频率/振幅）
        self.speed_freq_gain = 0.4  # 速度→频率增益（0.3~0.7Hz）
        self.speed_amp_gain = 0.2   # 速度→振幅增益（0.3~0.5）

        # 行走功能参数（原始逻辑保留，CPG替代后不再使用，仅做兼容）
        self.gait_cycle = 2.0
        self.step_offset_hip = 0.4
        self.step_offset_knee = 0.6
        self.step_offset_ankle = 0.3
        self.walk_start_time = None

        # 初始化稳定姿态（原始逻辑保留）
        self._init_stable_pose()

    def _init_stable_pose(self):
        """初始化稳定姿态（原始逻辑完全保留）"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 1.282
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

        # 腰部关节
        self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["abdomen_y"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["abdomen_x"]] = 0.0

        # 右腿关节
        self.joint_targets[self.joint_name_to_idx["hip_x_right"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_z_right"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_y_right"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["knee_right"]] = -0.4
        self.joint_targets[self.joint_name_to_idx["ankle_y_right"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["ankle_x_right"]] = 0.0

        # 左腿关节
        self.joint_targets[self.joint_name_to_idx["hip_x_left"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_z_left"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_y_left"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["knee_left"]] = -0.4
        self.joint_targets[self.joint_name_to_idx["ankle_y_left"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["ankle_x_left"]] = 0.0

        # 手臂关节
        self.joint_targets[self.joint_name_to_idx["shoulder1_right"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["shoulder2_right"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["elbow_right"]] = 1.5
        self.joint_targets[self.joint_name_to_idx["shoulder1_left"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["shoulder2_left"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["elbow_left"]] = 1.5

        mujoco.mj_forward(self.model, self.data)

    # 新增：状态机核心方法
    def _state_stand(self):
        """站立状态：维持初始稳定姿态，CPG重置"""
        self.right_leg_cpg.reset()
        self.left_leg_cpg.reset()
        self._init_stable_pose()  # 恢复站立姿态

    def _state_walk(self):
        """行走状态：CPG驱动+转向+变速"""
        if self.walk_start_time is None:
            self.walk_start_time = self.data.time
        # 1. 变速联动：速度→CPG频率+振幅
        self.right_leg_cpg.freq = 0.3 + self.walk_speed * self.speed_freq_gain
        self.left_leg_cpg.freq = 0.3 + self.walk_speed * self.speed_freq_gain
        self.right_leg_cpg.amp = 0.3 + self.walk_speed * self.speed_amp_gain
        self.left_leg_cpg.amp = 0.3 + self.walk_speed * self.speed_amp_gain

        # 2. 转向联动：躯干偏航 + 左右腿步长差
        # 躯干偏航（abdomen_z）由转向角度控制
        self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = self.turn_angle * self.turn_gain
        # 左右腿步长差（CPG振幅差）：左转→左腿振幅减小/右腿增大，右转反之
        if self.turn_angle > 0:  # 左转
            self.right_leg_cpg.amp *= 1.1
            self.left_leg_cpg.amp *= 0.9
        elif self.turn_angle < 0:  # 右转
            self.right_leg_cpg.amp *= 0.9
            self.left_leg_cpg.amp *= 1.1

        # 3. CPG更新（替代原始固定正弦步态）
        right_hip_offset = self.right_leg_cpg.update(self.dt, target_phase=self.left_leg_cpg.phase)
        left_hip_offset = self.left_leg_cpg.update(self.dt, target_phase=self.right_leg_cpg.phase)

        # 4. 更新关节目标（兼容原始关节映射）
        self.joint_targets[self.joint_name_to_idx["hip_y_right"]] = 0.1 + right_hip_offset
        self.joint_targets[self.joint_name_to_idx["knee_right"]] = -0.4 - right_hip_offset * 1.2  # 膝髋联动
        self.joint_targets[self.joint_name_to_idx["ankle_y_right"]] = 0.0 + right_hip_offset * 0.5  # 踝髋联动

        self.joint_targets[self.joint_name_to_idx["hip_y_left"]] = 0.1 + left_hip_offset
        self.joint_targets[self.joint_name_to_idx["knee_left"]] = -0.4 - left_hip_offset * 1.2
        self.joint_targets[self.joint_name_to_idx["ankle_y_left"]] = 0.0 + left_hip_offset * 0.5

        # 更新原始步态相位（兼容调试输出）
        self.gait_phase = self.right_leg_cpg.phase / (2 * np.pi) % 1.0

    def _state_stop(self):
        """停止状态：关节目标归零，缓慢减速"""
        self.joint_targets *= 0.95  # 渐进归零，避免突变
        self.data.qvel[:] *= 0.9    # 速度阻尼

    def _state_emergency(self):
        """急停状态：力矩清零，速度归零"""
        self.data.ctrl[:] = 0.0
        self.data.qvel[:] = 0.0
        self.joint_targets[:] = 0.0

    # 新增：外部控制接口（设置状态/转向/速度）
    def set_state(self, state):
        if state in self.state_map.keys():
            self.state = state
            # 状态切换时重置行走开始时间
            if state == "WALK":
                self.walk_start_time = None
            elif state == "STAND":
                self._init_stable_pose()

    def set_turn_angle(self, angle):
        self.turn_angle = np.clip(angle, -0.3, 0.3)  # 限制转向角度，避免失衡

    def set_walk_speed(self, speed):
        self.walk_speed = np.clip(speed, 0.1, 1.0)  # 限制速度范围

    # 原始方法：四元数转欧拉角（完全保留）
    def _quat_to_euler_xyz(self, quat):
        w, x, y, z = quat
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(sinp))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    # 原始方法：提取躯干欧拉角（完全保留）
    def _get_root_euler(self):
        quat = self.data.qpos[3:7].astype(np.float64).copy()
        euler = self._quat_to_euler_xyz(quat)
        euler = np.mod(euler + np.pi, 2 * np.pi) - np.pi
        return np.clip(euler, -0.3, 0.3)

    # 原始方法：检测脚部接触（完全保留）
    def _detect_foot_contact(self):
        try:
            left_foot_geoms = ["foot1_left", "foot2_left"]
            right_foot_geoms = ["foot1_right", "foot2_right"]

            left_force = 0.0
            for geom_name in left_foot_geoms:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, geom_id, force)
                left_force += np.linalg.norm(force[:3])

            right_force = 0.0
            for geom_name in right_foot_geoms:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, geom_id, force)
                right_force += np.linalg.norm(force[:3])

            self.foot_contact[1] = 1 if left_force > self.foot_contact_threshold else 0
            self.foot_contact[0] = 1 if right_force > self.foot_contact_threshold else 0

        except Exception as e:
            print(f"接触检测警告: {e}")
            self.foot_contact = np.ones(2)

    # 原始方法：计算稳定力矩（修改：加入状态机逻辑，保留所有原始PD/补偿逻辑）
    def _calculate_stabilizing_torques(self):
        # 状态机驱动：根据当前状态更新关节目标
        self.state_map[self.state]()

        # 原始力矩计算逻辑（完全保留）
        torques = np.zeros(self.num_joints, dtype=np.float64)

        # 躯干姿态控制
        root_euler = self._get_root_euler()
        root_vel = self.data.qvel[3:6].astype(np.float64).copy()
        root_vel = np.clip(root_vel, -3.0, 3.0)

        roll_error = -root_euler[0]
        self.integral_roll += roll_error * self.dt
        self.integral_roll = np.clip(self.integral_roll, -self.integral_limit, self.integral_limit)
        roll_torque = self.kp_roll * roll_error + self.kd_roll * (-root_vel[0]) + 5.0 * self.integral_roll

        pitch_error = -root_euler[1]
        self.integral_pitch += pitch_error * self.dt
        self.integral_pitch = np.clip(self.integral_pitch, -self.integral_limit, self.integral_limit)
        pitch_torque = self.kp_pitch * pitch_error + self.kd_pitch * (-root_vel[1]) + 4.0 * self.integral_pitch

        yaw_error = -root_euler[2]
        yaw_torque = self.kp_yaw * yaw_error + self.kd_yaw * (-root_vel[2])

        torso_torque = np.array([roll_torque, pitch_torque, yaw_torque])
        torso_torque = np.clip(torso_torque, -30.0, 30.0)

        # 重心补偿
        com = self.data.subtree_com[0].astype(np.float64).copy()
        com_error = self.com_target - com
        com_error = np.clip(com_error, -0.03, 0.03)
        com_compensation = self.kp_com * com_error

        # 关节控制
        self._detect_foot_contact()
        current_joints = self.data.qpos[7:7 + self.num_joints].astype(np.float64)
        current_vel = self.data.qvel[6:6 + self.num_joints].astype(np.float64)
        current_vel = np.clip(current_vel, -8.0, 8.0)

        # 腰部关节控制
        waist_joints = ["abdomen_z", "abdomen_y", "abdomen_x"]
        for joint_name in waist_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            joint_error = np.clip(joint_error, -0.3, 0.3)
            torques[idx] = self.kp_waist * joint_error - self.kd_waist * current_vel[idx]

        # 腿部关节控制
        leg_joints = [
            "hip_x_right", "hip_z_right", "hip_y_right",
            "knee_right", "ankle_y_right", "ankle_x_right",
            "hip_x_left", "hip_z_left", "hip_y_left",
            "knee_left", "ankle_y_left", "ankle_x_left"
        ]

        for joint_name in leg_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            joint_error = np.clip(joint_error, -0.3, 0.3)

            if "hip" in joint_name:
                kp = self.kp_hip
                kd = self.kd_hip
                if "y" in joint_name:
                    if "right" in joint_name:
                        joint_error += torso_torque[1] * 0.02
                    else:
                        joint_error += torso_torque[1] * 0.02

            elif "knee" in joint_name:
                kp = self.kp_knee
                kd = self.kd_knee
                joint_error += com_compensation[2] * 0.05

            elif "ankle" in joint_name:
                kp = self.kp_ankle
                kd = self.kd_ankle
                if "y" in joint_name:
                    joint_error += torso_torque[1] * 0.01

            if ("left" in joint_name and self.foot_contact[1] == 0) or \
               ("right" in joint_name and self.foot_contact[0] == 0):
                kp *= 0.8
                kd *= 0.9

            torques[idx] = kp * joint_error - kd * current_vel[idx]

        # 手臂关节控制
        arm_joints = [
            "shoulder1_right", "shoulder2_right", "elbow_right",
            "shoulder1_left", "shoulder2_left", "elbow_left"
        ]
        for joint_name in arm_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            torques[idx] = self.kp_arm * joint_error - self.kd_arm * current_vel[idx]

        # 力矩限幅
        torque_limits = {
            "abdomen_z": 50, "abdomen_y": 50, "abdomen_x": 50,
            "hip_x_right": 150, "hip_z_right": 150, "hip_y_right": 150,
            "knee_right": 200, "ankle_y_right": 120, "ankle_x_right": 100,
            "hip_x_left": 150, "hip_z_left": 150, "hip_y_left": 150,
            "knee_left": 200, "ankle_y_left": 120, "ankle_x_left": 100,
            "shoulder1_right": 20, "shoulder2_right": 20, "elbow_right": 20,
            "shoulder1_left": 20, "shoulder2_left": 20, "elbow_left": 20
        }
        for joint_name, limit in torque_limits.items():
            idx = self.joint_name_to_idx[joint_name]
            torques[idx] = np.clip(torques[idx], -limit, limit)

        # 调试输出（保留原始逻辑，新增状态/速度/转向输出）
        if self.data.time % 1 < 0.1 and self.data.time > self.init_wait_time:
            print(f"=== 行走调试 ===")
            print(f"状态: {self.state} | 步态相位: {self.gait_phase:.2f} | 速度: {self.walk_speed:.2f} | 转向: {self.turn_angle:.2f}")
            print(f"右腿髋目标: {self.joint_targets[self.joint_name_to_idx['hip_y_right']]:.2f}")
            print(f"左腿髋目标: {self.joint_targets[self.joint_name_to_idx['hip_y_left']]:.2f}")
            print(f"右脚接触: {self.foot_contact[0]}, 左脚接触: {self.foot_contact[1]}")

        self.prev_com = com
        return torques

    # 原始方法：仿真循环（修改：新增ROS线程启动，保留所有原始逻辑）
    def simulate_stable_standing(self):
        # 新增：启动ROS /cmd_vel监听线程
        self.ros_handler = ROSCmdVelHandler(self)
        self.ros_handler.start()

        # 原始：启动键盘监听线程
        keyboard_handler = KeyboardInputHandler(self)
        keyboard_handler.start()

        with viewer.launch_passive(self.model, self.data) as v:
            # 优化相机视角（原始逻辑保留）
            v.cam.distance = 3.0
            v.cam.azimuth = 90
            v.cam.elevation = -25
            v.cam.lookat = [0, 0, 0.6]

            print("人形机器人稳定站立+行走仿真启动...")
            print(f"初始稳定{self.init_wait_time}秒后，按W开始行走 | 支持转向/变速/启停/急停")

            # 初始落地阶段（原始逻辑保留）
            start_time = time.time()
            while time.time() - start_time < self.init_wait_time:
                alpha = min(1.0, (time.time() - start_time) / self.init_wait_time)
                torques = self._calculate_stabilizing_torques() * alpha
                self.data.ctrl[:] = torques
                mujoco.mj_step(self.model, self.data)
                self.data.qvel[:] *= 0.97
                v.sync()
                time.sleep(self.dt)

            # 主仿真循环（原始逻辑保留，状态机驱动）
            print("=== 初始稳定完成，可输入控制指令 ===")
            while self.data.time < self.sim_duration:
                torques = self._calculate_stabilizing_torques()
                self.data.ctrl[:] = torques
                mujoco.mj_step(self.model, self.data)

                # 状态监测（原始逻辑保留）
                if self.data.time % 2 < 0.1:
                    com = self.data.subtree_com[0]
                    euler = self._get_root_euler()
                    print(
                        f"时间:{self.data.time:.1f}s | 重心(x/z):{com[0]:.3f}/{com[2]:.3f}m | "
                        f"姿态(roll/pitch):{euler[0]:.3f}/{euler[1]:.3f}rad | 脚接触:{self.foot_contact}"
                    )

                v.sync()
                time.sleep(self.dt * 0.5)

                # 跌倒判定（原始逻辑保留）
                com = self.data.subtree_com[0]
                euler = self._get_root_euler()
                if com[2] < 0.4 or abs(euler[0]) > 0.6 or abs(euler[1]) > 0.6:
                    print(
                        f"跌倒！时间:{self.data.time:.1f}s | 重心(z):{com[2]:.3f}m | "
                        f"最大倾角:{max(abs(euler[0]), abs(euler[1])):.3f}rad"
                    )
                    self.set_state("STAND")  # 跌倒后自动恢复站立
                    # break  # 注释掉break，跌倒后可继续控制

        # 停止ROS线程
        self.ros_handler.stop()
        print("仿真完成！")


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(current_directory, "humanoid.xml")

    print(f"模型路径：{model_file_path}")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"模型文件不存在：{model_file_path}")

    try:
        stabilizer = HumanoidStabilizer(model_file_path)
        stabilizer.simulate_stable_standing()
    except Exception as e:
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()