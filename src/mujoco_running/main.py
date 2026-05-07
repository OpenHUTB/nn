import numpy as np
import mujoco
from mujoco import viewer
import time
import os
import sys
import threading
from collections import deque


# ===================== ROS话题接收模块 =====================
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
                "/cmd_vel", self.Twist, self._cmd_vel_callback, queue_size=1, tcp_nodelay=True
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
        raw_speed = float(msg.linear.x)
        if raw_speed <= 0.05:
            self.stabilizer.set_turn_angle(0.0)
            self.stabilizer.set_state("STOP")
            if hasattr(self.stabilizer, "_should_log") and self.stabilizer._should_log("ros_cmd_vel", 0.5):
                print("[ROS指令] 停止")
            return

        # 1. linear.x → 行走速度（限幅0.1~1.0）
        target_speed = float(np.clip(raw_speed, 0.1, 1.0))
        # 2. angular.z → 转向角度（-1.0~1.0映射到-0.3~0.3rad）
        target_turn = float(np.clip(msg.angular.z, -1.0, 1.0) * 0.3)  # 缩放系数0.3

        # 更新到控制器（优先级高于键盘，避免冲突）
        self.stabilizer.set_walk_speed(target_speed)
        self.stabilizer.set_turn_angle(target_turn)

        # 自动触发行走（如果接收到速度指令且当前是停止状态）
        if target_speed > 0.1 and self.stabilizer.state == "STAND":
            self.stabilizer.set_state("WALK")  # 默认切换到正常走

        # 调试输出
        if hasattr(self.stabilizer, "_should_log") and self.stabilizer._should_log("ros_cmd_vel", 0.5):
            print(
                f"[ROS指令] 速度={target_speed:.2f} | 转向={target_turn:.2f}rad | 当前步态: {self.stabilizer.gait_mode}")

    def run(self):
        """线程主循环（ROS自旋）"""
        if not self.has_ros:
            return
        if hasattr(self.rospy, "spin_once"):
            while self.running and not self.rospy.is_shutdown():
                try:
                    self.rospy.spin_once()
                except Exception:
                    pass
                time.sleep(0.01)
            return

        rate = self.rospy.Rate(100)
        while self.running and not self.rospy.is_shutdown():
            try:
                rate.sleep()
            except Exception:
                time.sleep(0.01)

    def stop(self):
        """停止线程（不强制关闭ROS进程，仅停止本线程循环）"""
        self.running = False


# ===================== 终端键盘控制（pycham） =====================
class KeyboardInputHandler(threading.Thread):
    """终端键盘控制线程（Windows使用msvcrt，Linux/macOS使用termios/tty）"""

    def __init__(self, stabilizer):
        super().__init__(daemon=True)
        self.stabilizer = stabilizer
        self.running = True

    def run(self):
        print("\n===== PyCharm 终端控制已启动 =====")
        print("w: 行走 | s: 停止 | e: 急停 | r: 复位")
        print("a: 左转 | d: 右转 | 空格: 原地转")
        print("z: 减速 | x: 加速")
        print("1: 慢走 | 2: 正常 | 3: 小跑 | 4: 踏步")
        print("m: 传感器开关 | p: 打印数据")
        print("===================================\n")

        if sys.platform == "win32":
            import msvcrt

            while self.running:
                try:
                    if msvcrt.kbhit():
                        key = msvcrt.getwch()
                        if key in ("\x00", "\xe0"):
                            msvcrt.getwch()
                            continue
                        self._handle_key(key)
                    time.sleep(0.01)
                except Exception:
                    time.sleep(0.01)
            return

        while self.running:
            old = None
            try:
                import tty
                import termios

                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                tty.setraw(fd)

                if sys.stdin.read(1) == '\x1b':
                    sys.stdin.read(2)
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    continue

                key = sys.stdin.read(1)
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                old = None
                self._handle_key(key)
            except Exception:
                if old is not None:
                    try:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    except Exception:
                        pass
            time.sleep(0.05)

    def _handle_key(self, key):
        key = key.lower()
        if key == 'w':
            current_gait = self.stabilizer.gait_mode
            if self.stabilizer.state != "WALK":
                self.stabilizer.set_state("WALK")
                self.stabilizer.set_gait_mode(current_gait)
            if hasattr(self.stabilizer, "_should_log") and self.stabilizer._should_log("key_walk", 0.5):
                print(f"[行走] 速度:{self.stabilizer.walk_speed:.2f} | 转向:{self.stabilizer.turn_angle:.2f}")
        elif key == 's':
            self.stabilizer.set_state("STOP")
            print("[停止]")
        elif key == 'e':
            self.stabilizer.set_state("EMERGENCY")
            print("[紧急停止]")
        elif key == 'r':
            self.stabilizer.set_state("STAND")
            print("[恢复站立]")
        elif key == 'a':
            self.stabilizer.set_turn_angle(self.stabilizer.turn_angle + 0.05)
            print(f"[左转] {self.stabilizer.turn_angle:.2f} rad")
        elif key == 'd':
            self.stabilizer.set_turn_angle(self.stabilizer.turn_angle - 0.05)
            print(f"[右转] {self.stabilizer.turn_angle:.2f} rad")
        elif key == ' ':
            self.stabilizer.set_turn_angle(0.2 if self.stabilizer.turn_angle <= 0 else -0.2)
            print(f"[原地转向] {self.stabilizer.turn_angle:.2f} rad")
        elif key == 'z':
            self.stabilizer.set_walk_speed(self.stabilizer.walk_speed - 0.1)
            print(f"[减速] 速度:{self.stabilizer.walk_speed:.2f}")
        elif key == 'x':
            self.stabilizer.set_walk_speed(self.stabilizer.walk_speed + 0.1)
            print(f"[加速] 速度:{self.stabilizer.walk_speed:.2f}")
        elif key == 'm':
            self.stabilizer.enable_sensor_simulation = not self.stabilizer.enable_sensor_simulation
            print(f"[传感器] {'开启' if self.stabilizer.enable_sensor_simulation else '关闭'}")
        elif key == 'p':
            self.stabilizer.print_sensor_data()
        elif key == '1':
            self.stabilizer.set_gait_mode("SLOW")
            print("[模式] 慢走")
        elif key == '2':
            self.stabilizer.set_gait_mode("NORMAL")
            print("[模式] 正常")
        elif key == '3':
            self.stabilizer.set_gait_mode("TROT")
            print("[模式] 小跑")
        elif key == '4':
            self.stabilizer.set_gait_mode("STEP_IN_PLACE")
            print("[模式] 原地踏步")


# ===================== CPG中枢模式发生器 =====================
class CPGOscillator:
    def __init__(self, freq=0.5, amp=0.4, phase=0.0, coupling_strength=0.2):
        self.base_freq = freq  # 基础频率（对应原始步态周期2s）
        self.base_amp = amp  # 基础振幅（对应原始步长）
        self.freq = freq
        self.amp = amp
        self.phase = phase  # 初始相位（左右腿差π）
        self.base_coupling = coupling_strength  # 基础耦合强度
        self.coupling = coupling_strength  # 动态耦合强度
        self.state = np.array([np.sin(phase), np.cos(phase)])  # 振荡器状态(x,y)

    def update(self, dt, target_phase=0.0, speed_factor=1.0, turn_factor=0.0):
        """
        更新CPG状态，返回关节目标偏移量
        新增：speed_factor（速度系数）、turn_factor（转向系数）动态调整耦合强度
        """
        # 动态调整耦合强度：速度越快/转向越大，耦合越强（提升步态稳定性）
        self.coupling = self.base_coupling * (1.0 + 0.5 * speed_factor + 0.8 * abs(turn_factor))
        self.coupling = np.clip(self.coupling, 0.1, 0.5)  # 限幅避免耦合过强/过弱

        # 范德波尔振荡器方程（生物节律更自然，抗干扰）
        mu = 1.0  # 非线性系数，控制振荡收敛性
        x, y = self.state
        dx = 2 * np.pi * self.freq * y + self.coupling * np.sin(target_phase - self.phase)
        dy = 2 * np.pi * self.freq * (mu * (1 - x ** 2) * y - x)
        # 更新状态（积分）
        self.state += np.array([dx, dy]) * dt
        self.phase = np.arctan2(self.state[0], self.state[1])  # 更新相位
        # 返回当前输出（关节目标偏移量）
        return self.amp * self.state[0]

    def reset(self):
        """重置CPG状态"""
        self.freq = self.base_freq
        self.amp = self.base_amp
        self.coupling = self.base_coupling
        self.phase = 0.0 if self.phase < np.pi else np.pi
        self.state = np.array([np.sin(self.phase), np.cos(self.phase)])


# ===================== 人形机器人控制器 =====================
class HumanoidStabilizer:
    """适配humanoid.xml模型的稳定站立与行走控制器（新增多步态+传感器模拟+鲁棒性优化）"""

    def __init__(self, model_path):
        # 类型检查与模型加载（原有逻辑完全保留）
        if not isinstance(model_path, str):
            raise TypeError(f"模型路径必须是字符串，当前是 {type(model_path)} 类型")

        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}\n请检查路径和文件完整性")

        # 仿真核心参数（原有逻辑保留）
        self.sim_duration = 120.0
        self.dt = self.model.opt.timestep
        self.init_wait_time = 4.0
        self.model.opt.gravity[2] = -9.81
        self.model.opt.iterations = 200
        self.model.opt.tolerance = 1e-8

        self._log_last = {}
        self._fall_cooldown_until = 0.0
        self._fall_count = 0
        self._recovery_until = 0.0
        self._imu_euler_filt = np.zeros(3, dtype=np.float64)
        self._imu_angvel_filt = np.zeros(3, dtype=np.float64)

        # 关节名称映射（原有逻辑完全保留）
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

        self._actuator_id_by_joint = {}
        self._actuator_gear_by_joint = {}
        self._actuator_ctrlrange_by_joint = {}
        for joint_name in self.joint_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id < 0:
                raise RuntimeError(f"未找到与关节同名的执行器：{joint_name}")
            self._actuator_id_by_joint[joint_name] = int(actuator_id)
            self._actuator_gear_by_joint[joint_name] = float(self.model.actuator_gear[actuator_id, 0])
            self._actuator_ctrlrange_by_joint[joint_name] = self.model.actuator_ctrlrange[actuator_id].astype(
                np.float64
            )

        self._qpos_adr = np.empty(self.num_joints, dtype=np.int32)
        self._qvel_adr = np.empty(self.num_joints, dtype=np.int32)
        for joint_name in self.joint_names:
            joint_idx = self.joint_name_to_idx[joint_name]
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"未找到关节：{joint_name}")
            self._qpos_adr[joint_idx] = int(self.model.jnt_qposadr[joint_id])
            self._qvel_adr[joint_idx] = int(self.model.jnt_dofadr[joint_id])

        # ===================== 【仅修改：超强站立平衡增益】 =====================
        self.kp_roll = 380.0
        self.kd_roll = 100.0
        self.kp_pitch = 340.0
        self.kd_pitch = 90.0
        self.kp_yaw = 60.0
        self.kd_yaw = 25.0
        self.base_kp_hip = 420.0
        self.base_kd_hip = 90.0
        self.base_kp_knee = 460.0
        self.base_kd_knee = 100.0
        self.base_kp_ankle = 380.0
        self.base_kd_ankle = 110.0
        self.kp_waist = 80.0
        self.kd_waist = 35.0
        self.kp_arm = 20.0
        self.kd_arm = 20.0

        # 重心目标绝对居中
        self.com_target = np.array([0.0, 0.0, 0.80])
        self.kp_com = 120.0
        # ======================================================================

        self.total_mass = float(np.sum(self.model.body_mass))
        self.weight = float(self.total_mass * abs(float(self.model.opt.gravity[2])))
        self.foot_contact_threshold = float(max(20.0, 0.12 * self.weight))
        self._force_factor_norm = float(max(1.0, 0.5 * self.weight))
        self.com_safety_threshold = 0.6
        self.speed_reduction_factor = 0.5
        self._support_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
        if self._support_body_id < 0:
            self._support_body_id = 0
        self._support_until = 0.0

        self._left_foot_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot1_left"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot2_left"),
        }
        self._right_foot_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot1_right"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot2_right"),
        }

        # 状态变量
        self.joint_targets = np.zeros(self.num_joints)
        self.prev_joint_targets = np.zeros(self.num_joints)
        self.prev_com = np.zeros(3)
        self.foot_contact = np.zeros(2)
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.integral_limit = 0.15
        self.filter_alpha = 0.1
        self.enable_robust_optim = True

        # 多步态模式配置
        self.gait_config = {
            "SLOW": {
                "freq": 0.3,
                "amp": 0.3,
                "coupling": 0.3,
                "speed_freq_gain": 0.2,
                "speed_amp_gain": 0.1,
                "com_z_offset": 0.02
            },
            "NORMAL": {
                "freq": 0.5,
                "amp": 0.4,
                "coupling": 0.2,
                "speed_freq_gain": 0.4,
                "speed_amp_gain": 0.2,
                "com_z_offset": 0.0
            },
            "TROT": {
                "freq": 0.8,
                "amp": 0.5,
                "coupling": 0.25,
                "speed_freq_gain": 0.5,
                "speed_amp_gain": 0.3,
                "com_z_offset": -0.01
            },
            "STEP_IN_PLACE": {
                "freq": 0.4,
                "amp": 0.2,
                "coupling": 0.3,
                "speed_freq_gain": 0.0,
                "speed_amp_gain": 0.0,
                "com_z_offset": 0.01,
                "lock_torso": True
            }
        }
        self.gait_mode = "NORMAL"
        self.current_gait_params = self.gait_config[self.gait_mode]

        self.state = "STAND"
        self.state_map = {
            "STAND": self._state_stand,
            "WALK": self._state_walk,
            "STOP": self._state_stop,
            "EMERGENCY": self._state_emergency
        }

        self.right_leg_cpg = CPGOscillator(
            freq=self.current_gait_params["freq"],
            amp=self.current_gait_params["amp"],
            phase=0.0,
            coupling_strength=self.current_gait_params["coupling"]
        )
        self.left_leg_cpg = CPGOscillator(
            freq=self.current_gait_params["freq"],
            amp=self.current_gait_params["amp"],
            phase=np.pi,
            coupling_strength=self.current_gait_params["coupling"]
        )
        self.gait_phase = 0.0

        self.turn_angle = 0.0
        self.turn_gain = 0.1
        self.walk_speed = 0.5
        self.speed_freq_gain = self.current_gait_params["speed_freq_gain"]
        self.speed_amp_gain = self.current_gait_params["speed_amp_gain"]
        self.gait_cycle = 2.0
        self.step_offset_hip = 0.4
        self.step_offset_knee = 0.6
        self.step_offset_ankle = 0.3
        self.walk_start_time = None

        self.enable_sensor_simulation = True
        self.imu_angle_noise = 0.01
        self.imu_vel_noise = 0.05
        self.imu_delay_frames = 2
        self.foot_force_noise = 0.3
        self.foot_force_offset = 0.1
        self.imu_data_buffer = deque(maxlen=self.imu_delay_frames)
        self.foot_data_buffer = deque(maxlen=self.imu_delay_frames)
        self.current_sensor_data = {}

        self._init_stable_pose()

    def _should_log(self, key, interval_s):
        now = float(self.data.time)
        last = float(self._log_last.get(key, -1e9))
        if (now - last) >= float(interval_s):
            self._log_last[key] = now
            return True
        return False

    def _get_joint_positions(self):
        return self.data.qpos[self._qpos_adr].astype(np.float64, copy=True)

    def _get_joint_velocities(self):
        return self.data.qvel[self._qvel_adr].astype(np.float64, copy=True)

    def _torques_to_ctrl(self, joint_torques):
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        for joint_name in self.joint_names:
            joint_idx = self.joint_name_to_idx[joint_name]
            actuator_id = self._actuator_id_by_joint[joint_name]
            gear = float(self._actuator_gear_by_joint[joint_name])
            ctrl_min, ctrl_max = self._actuator_ctrlrange_by_joint[joint_name]
            max_torque = max(abs(ctrl_min), abs(ctrl_max)) * max(gear, 1e-9)
            torque = float(np.clip(joint_torques[joint_idx], -max_torque, max_torque))
            ctrl_val = torque / max(gear, 1e-9)
            ctrl[actuator_id] = float(np.clip(ctrl_val, ctrl_min, ctrl_max))
        return ctrl

    def set_gait_mode(self, mode):
        if mode not in self.gait_config.keys():
            print(f"[警告] 无效的步态模式：{mode}，默认使用NORMAL")
            mode = "NORMAL"

        self.gait_mode = mode
        self.current_gait_params = self.gait_config[mode]

        self.right_leg_cpg.base_freq = self.current_gait_params["freq"]
        self.right_leg_cpg.base_amp = self.current_gait_params["amp"]
        self.right_leg_cpg.base_coupling = self.current_gait_params["coupling"]

        self.left_leg_cpg.base_freq = self.current_gait_params["freq"]
        self.left_leg_cpg.base_amp = self.current_gait_params["amp"]
        self.left_leg_cpg.base_coupling = self.current_gait_params["coupling"]

        self.speed_freq_gain = self.current_gait_params["speed_freq_gain"]
        self.speed_amp_gain = self.current_gait_params["speed_amp_gain"]
        self.com_target[2] = 0.78 + self.current_gait_params["com_z_offset"]
        self.right_leg_cpg.reset()
        self.left_leg_cpg.reset()

    def _init_stable_pose(self):
        keep_time = float(self.data.time)
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = keep_time
        self.data.qpos[2] = 1.282
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self._imu_euler_filt[:] = 0.0
        self._imu_angvel_filt[:] = 0.0
        self.current_sensor_data = {}
        self.imu_data_buffer.clear()
        self.foot_data_buffer.clear()

        self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["abdomen_y"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["abdomen_x"]] = 0.0

        self.joint_targets[self.joint_name_to_idx["hip_x_right"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_z_right"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_y_right"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["knee_right"]] = -0.4
        self.joint_targets[self.joint_name_to_idx["ankle_y_right"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["ankle_x_right"]] = 0.0

        self.joint_targets[self.joint_name_to_idx["hip_x_left"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_z_left"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_y_left"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["knee_left"]] = -0.4
        self.joint_targets[self.joint_name_to_idx["ankle_y_left"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["ankle_x_left"]] = 0.0

        self.joint_targets[self.joint_name_to_idx["shoulder1_right"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["shoulder2_right"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["elbow_right"]] = 1.5
        self.joint_targets[self.joint_name_to_idx["shoulder1_left"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["shoulder2_left"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["elbow_left"]] = 1.5
        self.prev_joint_targets = self.joint_targets.copy()

        self.data.qpos[self._qpos_adr] = self.joint_targets.astype(np.float64)
        mujoco.mj_forward(self.model, self.data)
        target_clearance = 0.002
        min_bottom_z = None
        foot_geom_ids = list(self._left_foot_geom_ids | self._right_foot_geom_ids)
        for gid in foot_geom_ids:
            if gid < 0:
                continue
            radius = float(self.model.geom_size[gid, 0])
            z = float(self.data.geom_xpos[gid, 2]) - radius
            if min_bottom_z is None or z < min_bottom_z:
                min_bottom_z = z
        if min_bottom_z is not None:
            dz = (min_bottom_z - target_clearance)
            if abs(dz) > 1e-6:
                self.data.qpos[2] -= dz
                mujoco.mj_forward(self.model, self.data)
        self._support_until = float(self.data.time) + 3.0

    def _simulate_imu_data(self):
        true_quat = self.data.qpos[3:7].astype(np.float64).copy()
        true_euler = self._quat_to_euler_xyz(true_quat)
        true_ang_vel = self.data.qvel[3:6].astype(np.float64).copy()

        noisy_euler = true_euler + np.random.normal(0, self.imu_angle_noise, 3)
        noisy_ang_vel = true_ang_vel + np.random.normal(0, self.imu_vel_noise, 3)

        noisy_euler = np.clip(noisy_euler, -np.pi / 2, np.pi / 2)
        noisy_ang_vel = np.clip(noisy_ang_vel, -5.0, 5.0)

        self.imu_data_buffer.append({
            "euler": noisy_euler,
            "ang_vel": noisy_ang_vel,
            "true_euler": true_euler,
            "true_ang_vel": true_ang_vel
        })

        if len(self.imu_data_buffer) < self.imu_delay_frames:
            return {
                "euler": true_euler,
                "ang_vel": true_ang_vel,
                "true_euler": true_euler,
                "true_ang_vel": true_ang_vel
            }
        else:
            return self.imu_data_buffer[0]

    def _simulate_foot_force_data(self):
        true_left_force, true_right_force = self._compute_foot_forces()

        noisy_left_force = true_left_force + np.random.normal(0, self.foot_force_noise) + self.foot_force_offset
        noisy_right_force = true_right_force + np.random.normal(0, self.foot_force_noise) + self.foot_force_offset

        noisy_left_force = max(0.0, noisy_left_force)
        noisy_right_force = max(0.0, noisy_right_force)

        left_contact = 1 if noisy_left_force > self.foot_contact_threshold else 0
        right_contact = 1 if noisy_right_force > self.foot_contact_threshold else 0

        self.foot_data_buffer.append({
            "left_force": noisy_left_force,
            "right_force": noisy_right_force,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "true_left_force": true_left_force,
            "true_right_force": true_right_force
        })

        if len(self.foot_data_buffer) < self.imu_delay_frames:
            return {
                "left_force": true_left_force,
                "right_force": true_right_force,
                "left_contact": 1 if true_left_force > self.foot_contact_threshold else 0,
                "right_contact": 1 if true_right_force > self.foot_contact_threshold else 0,
                "true_left_force": true_left_force,
                "true_right_force": true_right_force
            }
        else:
            return self.foot_data_buffer[0]

    def _get_sensor_data(self):
        if not self.enable_sensor_simulation:
            imu_data = {
                "euler": self._get_root_euler(),
                "ang_vel": self.data.qvel[3:6].astype(np.float64).copy(),
                "true_euler": self._get_root_euler(),
                "true_ang_vel": self.data.qvel[3:6].astype(np.float64).copy()
            }

            self._detect_foot_contact()
            foot_data = {
                "left_force": self.left_foot_force,
                "right_force": self.right_foot_force,
                "left_contact": self.foot_contact[1],
                "right_contact": self.foot_contact[0],
                "true_left_force": self.left_foot_force,
                "true_right_force": self.right_foot_force
            }
        else:
            imu_data = self._simulate_imu_data()
            foot_data = self._simulate_foot_force_data()

        self.current_sensor_data = {
            "imu": imu_data,
            "foot": foot_data,
            "time": self.data.time,
            "gait_mode": self.gait_mode
        }

        return self.current_sensor_data

    def print_sensor_data(self):
        if not self.current_sensor_data:
            print("[传感器数据] 暂无数据")
            return

        imu = self.current_sensor_data["imu"]
        foot = self.current_sensor_data["foot"]
        print("\n=== 传感器数据 ===")
        print(
            f"仿真时间: {self.current_sensor_data['time']:.2f}s | 模拟状态: {'开启' if self.enable_sensor_simulation else '关闭'} | 当前步态: {self.gait_mode}")
        print(f"IMU欧拉角(roll/pitch/yaw): {imu['euler'][0]:.3f}/{imu['euler'][1]:.3f}/{imu['euler'][2]:.3f}rad")
        print(f"IMU真实值: {imu['true_euler'][0]:.3f}/{imu['true_euler'][1]:.3f}/{imu['true_euler'][2]:.3f}rad")
        print(
            f"左脚力: {foot['left_force']:.2f}N (真实: {foot['true_left_force']:.2f}N) | 接触: {foot['left_contact']}")
        print(
            f"右脚力: {foot['right_force']:.2f}N (真实: {foot['true_right_force']:.2f}N) | 接触: {foot['right_contact']}")
        print("==================\n")

    def _state_stand(self):
        self.right_leg_cpg.reset()
        self.left_leg_cpg.reset()

    def _state_walk(self):
        if self.walk_start_time is None:
            self.walk_start_time = self.data.time

        current_com_z = self.data.subtree_com[0][2]
        if current_com_z < self.com_safety_threshold and self.enable_robust_optim:
            current_speed = self.walk_speed * self.speed_reduction_factor
            self.walk_speed = np.clip(current_speed, 0.1, self.walk_speed)
            if self._should_log("com_low_speed_reduce", 1.0):
                print(
                    f"[鲁棒优化] 重心过低({current_com_z:.2f}m)，自动降速到{self.walk_speed:.2f} | 当前步态: {self.gait_mode}")

        gait_params = self.current_gait_params

        self.right_leg_cpg.freq = gait_params["freq"] + self.walk_speed * gait_params["speed_freq_gain"]
        self.left_leg_cpg.freq = gait_params["freq"] + self.walk_speed * gait_params["speed_freq_gain"]
        self.right_leg_cpg.amp = gait_params["amp"] + self.walk_speed * gait_params["speed_amp_gain"]
        self.left_leg_cpg.amp = gait_params["amp"] + self.walk_speed * gait_params["speed_amp_gain"]

        if self.gait_mode == "STEP_IN_PLACE":
            self.turn_angle = 0.0
            self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = 0.0
            self.right_leg_cpg.amp = gait_params["amp"]
            self.left_leg_cpg.amp = gait_params["amp"]
        else:
            self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = self.turn_angle * self.turn_gain
            if self.turn_angle > 0:
                self.right_leg_cpg.amp *= 1.1
                self.left_leg_cpg.amp *= 0.9
            elif self.turn_angle < 0:
                self.right_leg_cpg.amp *= 0.9
                self.left_leg_cpg.amp *= 1.1

        speed_factor = self.walk_speed / 1.0
        turn_factor = self.turn_angle / 0.3
        right_hip_offset = self.right_leg_cpg.update(
            self.dt, target_phase=self.left_leg_cpg.phase,
            speed_factor=speed_factor, turn_factor=turn_factor
        )
        left_hip_offset = self.left_leg_cpg.update(
            self.dt, target_phase=self.right_leg_cpg.phase,
            speed_factor=speed_factor, turn_factor=turn_factor
        )

        ramp = float(np.clip((float(self.data.time) - float(self.walk_start_time)) / 1.0, 0.0, 1.0))
        right_hip_offset *= ramp
        left_hip_offset *= ramp

        if self.gait_mode != "STEP_IN_PLACE":
            self.joint_targets[self.joint_name_to_idx["abdomen_y"]] = float(np.clip(-0.06 * speed_factor, -0.12, 0.0))
        self.joint_targets[self.joint_name_to_idx["hip_y_right"]] = 0.1 + right_hip_offset
        self.joint_targets[self.joint_name_to_idx["knee_right"]] = -0.4 - right_hip_offset * 1.2
        self.joint_targets[self.joint_name_to_idx["ankle_y_right"]] = 0.0 + right_hip_offset * 0.5
        self.joint_targets[self.joint_name_to_idx["hip_y_left"]] = 0.1 + left_hip_offset
        self.joint_targets[self.joint_name_to_idx["knee_left"]] = -0.4 - left_hip_offset * 1.2
        self.joint_targets[self.joint_name_to_idx["ankle_y_left"]] = 0.0 + left_hip_offset * 0.5

        if self.enable_robust_optim:
            self.joint_targets = (
                    1 - self.filter_alpha) * self.prev_joint_targets + self.filter_alpha * self.joint_targets
            self.prev_joint_targets = self.joint_targets.copy()

        self.gait_phase = self.right_leg_cpg.phase / (2 * np.pi) % 1.0

    def _state_stop(self):
        self.joint_targets *= 0.95
        self.data.qvel[:] *= 0.9

    def _state_emergency(self):
        self.data.ctrl[:] = 0.0
        self.data.qvel[:] = 0.0
        self.joint_targets[:] = 0.0

    def set_state(self, state):
        if state in self.state_map.keys():
            if state == "WALK" and float(self.data.time) < float(self._recovery_until):
                return
            self.state = state
            if state == "WALK":
                self.walk_start_time = None
            elif state == "STAND":
                self._init_stable_pose()

    def set_turn_angle(self, angle):
        if self.gait_mode != "STEP_IN_PLACE":
            self.turn_angle = np.clip(angle, -0.3, 0.3)

    def set_walk_speed(self, speed):
        if self.gait_mode != "STEP_IN_PLACE":
            self.walk_speed = np.clip(speed, 0.1, 1.0)
        else:
            self.walk_speed = 0.5

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

    def _get_root_euler(self):
        quat = self.data.qpos[3:7].astype(np.float64).copy()
        euler = self._quat_to_euler_xyz(quat)
        euler = np.mod(euler + np.pi, 2 * np.pi) - np.pi
        return euler

    def _detect_foot_contact(self):
        try:
            left_force, right_force = self._compute_foot_forces()

            self.foot_contact[1] = 1 if left_force > self.foot_contact_threshold else 0
            self.foot_contact[0] = 1 if right_force > self.foot_contact_threshold else 0
            self.left_foot_force = left_force
            self.right_foot_force = right_force

        except Exception as e:
            print(f"接触检测警告: {e}")
            self.foot_contact = np.ones(2)
            self.left_foot_force = self.foot_contact_threshold
            self.right_foot_force = self.foot_contact_threshold

    def _compute_foot_forces(self):
        left_force = 0.0
        right_force = 0.0
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            in_left = (geom1 in self._left_foot_geom_ids) or (geom2 in self._left_foot_geom_ids)
            in_right = (geom1 in self._right_foot_geom_ids) or (geom2 in self._right_foot_geom_ids)
            if not (in_left or in_right):
                continue

            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_id, force)
            f = float(np.linalg.norm(force[:3]))
            if in_left:
                left_force += f
            if in_right:
                right_force += f

        return left_force, right_force

    # ===================== 【仅修改：核心平衡计算】 =====================
    def _calculate_stabilizing_torques(self):
        self.state_map[self.state]()
        sensor_data = self._get_sensor_data()
        imu = sensor_data["imu"]
        foot = sensor_data["foot"]
        torques = np.zeros(self.num_joints, dtype=np.float64)

        if self.enable_sensor_simulation:
            root_euler = imu.get("true_euler", imu["euler"])
            root_vel = imu.get("true_ang_vel", imu["ang_vel"])
        else:
            root_euler = imu["euler"]
            root_vel = imu["ang_vel"]
        root_vel = np.clip(root_vel, -3.0, 3.0)

        # 更强滤波
        imu_alpha = 0.2
        self._imu_euler_filt = (1.0 - imu_alpha) * self._imu_euler_filt + imu_alpha * root_euler
        self._imu_angvel_filt = (1.0 - imu_alpha) * self._imu_angvel_filt + imu_alpha * root_vel

        # 超强PID + 积分
        roll_error = -self._imu_euler_filt[0]
        self.integral_roll = np.clip(self.integral_roll + roll_error * self.dt, -self.integral_limit, self.integral_limit)
        roll_torque = self.kp_roll * roll_error + self.kd_roll * (-self._imu_angvel_filt[0]) + 8.0 * self.integral_roll

        pitch_error = -self._imu_euler_filt[1]
        self.integral_pitch = np.clip(self.integral_pitch + pitch_error * self.dt, -self.integral_limit, self.integral_limit)
        pitch_torque = self.kp_pitch * pitch_error + self.kd_pitch * (-self._imu_angvel_filt[1]) + 6.0 * self.integral_pitch

        yaw_error = -self._imu_euler_filt[2]
        yaw_torque = self.kp_yaw * yaw_error + self.kd_yaw * (-self._imu_angvel_filt[2])

        torso_torque = np.array([roll_torque, pitch_torque, yaw_torque])
        torso_torque = np.clip(torso_torque, -35.0, 35.0)

        self.data.xfrc_applied[self._support_body_id, :] = 0.0
        now_t = float(self.data.time)
        if now_t < float(self._support_until):
            scale = float(np.clip((float(self._support_until) - now_t) / 3.0, 0.0, 1.0))
            self.data.xfrc_applied[self._support_body_id, 2] = self.weight * 0.95 * scale
            self.data.xfrc_applied[self._support_body_id, 3] = (-100.0 * self._imu_euler_filt[0] - 25.0 * self._imu_angvel_filt[0]) * scale
            self.data.xfrc_applied[self._support_body_id, 4] = (-100.0 * self._imu_euler_filt[1] - 25.0 * self._imu_angvel_filt[1]) * scale

        com = self.data.subtree_com[0].astype(np.float64).copy()
        com_error = self.com_target - com
        com_error = np.clip(com_error, -0.03, 0.03)
        com_compensation = self.kp_com * com_error

        current_joints = self._get_joint_positions()
        current_vel = self._get_joint_velocities()
        current_vel = np.clip(current_vel, -8.0, 8.0)

        self.foot_contact = np.array([foot["right_contact"], foot["left_contact"]])
        self.left_foot_force = foot["left_force"]
        self.right_foot_force = foot["right_force"]

        # 腰部
        waist_joints = ["abdomen_z", "abdomen_y", "abdomen_x"]
        for joint_name in waist_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = float(self.joint_targets[idx] - current_joints[idx])
            joint_error = max(-0.3, min(0.3, joint_error))
            if joint_name == "abdomen_y":
                joint_error -= float(np.clip(torso_torque[1] * 0.008, -0.15, 0.15))
            torques[idx] = self.kp_waist * joint_error - self.kd_waist * current_vel[idx]

        # 腿部
        leg_joints = [
            "hip_x_right", "hip_z_right", "hip_y_right",
            "knee_right", "ankle_y_right", "ankle_x_right",
            "hip_x_left", "hip_z_left", "hip_y_left",
            "knee_left", "ankle_y_left", "ankle_x_left"
        ]

        for joint_name in leg_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = float(self.joint_targets[idx] - current_joints[idx])
            joint_error = max(-0.3, min(0.3, joint_error))

            if self.enable_robust_optim and self.state == "WALK":
                if "right" in joint_name:
                    force_factor = np.clip(self.right_foot_force / self._force_factor_norm, 0.4, 1.1)
                else:
                    force_factor = np.clip(self.left_foot_force / self._force_factor_norm, 0.4, 1.1)
            else:
                force_factor = 1.0

            if "hip" in joint_name:
                kp = self.base_kp_hip * force_factor
                kd = self.base_kd_hip * force_factor
                if "y" in joint_name:
                    joint_error -= torso_torque[1] * 0.02

            elif "knee" in joint_name:
                kp = self.base_kp_knee * force_factor
                kd = self.base_kd_knee * force_factor
                joint_error += com_compensation[2] * 0.05
                joint_error += torso_torque[1] * 0.01

            elif "ankle" in joint_name:
                kp = self.base_kp_ankle * force_factor
                kd = self.base_kd_ankle * force_factor
                if "y" in joint_name:
                    joint_error += torso_torque[1] * 0.015

            if self.state == "WALK":
                if ("left" in joint_name and self.foot_contact[1] == 0) or \
                        ("right" in joint_name and self.foot_contact[0] == 0):
                    kp *= 0.8
                    kd *= 0.9

            torques[idx] = kp * joint_error - kd * current_vel[idx]

        # 手臂
        arm_joints = [
            "shoulder1_right", "shoulder2_right", "elbow_right",
            "shoulder1_left", "shoulder2_left", "elbow_left"
        ]
        for joint_name in arm_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            torques[idx] = self.kp_arm * joint_error - self.kd_arm * current_vel[idx]

        # 力矩限制
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

        if self.data.time > self.init_wait_time and self._should_log("walk_debug", 2.0):
            print(f"=== 行走调试 ===")
            print(f"状态: {self.state} | 步态模式: {self.gait_mode}")
            print(f"右脚接触: {self.foot_contact[0]}, 左脚接触: {self.foot_contact[1]}")

        self.prev_com = com
        return torques
    # ==================================================================

    def simulate_stable_standing(self):
        self.ros_handler = ROSCmdVelHandler(self)
        self.ros_handler.start()

        keyboard_handler = KeyboardInputHandler(self)
        keyboard_handler.start()

        try:
            with viewer.launch_passive(self.model, self.data) as v:
                v.cam.distance = 3.0
                v.cam.azimuth = 90
                v.cam.elevation = -25
                v.cam.lookat = [0, 0, 0.6]

                print("人形机器人稳定站立+行走仿真启动（已启用多步态+传感器模拟+步态鲁棒性优化）...")
                print(f"初始稳定{self.init_wait_time}秒后，按W开始行走 | 支持多步态切换（1=慢走/2=正常/3=小跑/4=原地踏步）")
                print(f"默认步态模式：{self.gait_mode}\n")

                self._support_until = max(float(self._support_until), float(self.data.time) + float(self.init_wait_time))
                start_time = time.time()
                while time.time() - start_time < self.init_wait_time:
                    elapsed = time.time() - start_time
                    alpha = min(1.0, elapsed / 1.0)
                    torque_scale = 0.5 + 0.5 * alpha
                    torques = self._calculate_stabilizing_torques() * torque_scale
                    self.data.ctrl[:] = self._torques_to_ctrl(torques)
                    mujoco.mj_step(self.model, self.data)
                    self.data.qvel[:] *= 0.97
                    v.sync()
                    time.sleep(self.dt)

                print("=== 初始稳定完成，可输入控制指令 ===")
                while self.data.time < self.sim_duration:
                    torques = self._calculate_stabilizing_torques()
                    self.data.ctrl[:] = self._torques_to_ctrl(torques)
                    mujoco.mj_step(self.model, self.data)

                    if self._should_log("status", 2.0):
                        com = self.data.subtree_com[0]
                        if self.current_sensor_data:
                            imu_data = self.current_sensor_data["imu"]
                            euler = imu_data.get("true_euler", imu_data.get("euler"))
                        else:
                            euler = self._get_root_euler()
                        print(
                            f"时间:{self.data.time:.1f}s | 重心(x/z):{com[0]:.3f}/{com[2]:.3f}m | "
                            f"姿态(roll/pitch):{euler[0]:.3f}/{euler[1]:.3f}rad | 脚接触:{self.foot_contact} | "
                            f"当前步态:{self.gait_mode}"
                        )

                    v.sync()
                    time.sleep(self.dt)

                    if self.data.time < self._fall_cooldown_until:
                        continue
                    com = self.data.subtree_com[0]
                    if self.current_sensor_data:
                        imu_data = self.current_sensor_data["imu"]
                        euler_for_fall = imu_data.get("true_euler", imu_data.get("euler"))
                    else:
                        euler_for_fall = self._get_root_euler()
                    if com[2] < 0.4 or abs(euler_for_fall[0]) > 0.6 or abs(euler_for_fall[1]) > 0.6:
                        self._fall_count += 1
                        print(
                            f"跌倒！#{self._fall_count} 时间:{self.data.time:.1f}s | 重心(z):{com[2]:.3f}m | "
                            f"最大倾角:{max(abs(euler_for_fall[0]), abs(euler_for_fall[1])):.3f}rad | 当前步态:{self.gait_mode}"
                        )
                        self.set_state("STAND")
                        self._fall_cooldown_until = float(self.data.time) + 2.0
                        self._recovery_until = float(self.data.time) + 2.0
                        self.set_turn_angle(0.0)
        finally:
            keyboard_handler.running = False
            self.ros_handler.stop()
            print("仿真完成！")


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(current_directory, "models","humanoid.xml")

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