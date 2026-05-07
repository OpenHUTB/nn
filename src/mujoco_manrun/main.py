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

        # PD控制增益（原有逻辑保留，新增动态增益系数）
        # 关节控制增益（根据生物学原理调整，提升步态自然性和稳定性）
        self.kp_roll = 350.0
        self.kd_roll = 90.0
        self.kp_pitch = 300.0
        self.kd_pitch = 80.0
        self.kp_yaw = 60.0
        self.kd_yaw = 30.0

        # 髋、膝盖、脚踝大幅加固
        self.base_kp_hip = 320.0
        self.base_kd_hip = 65.0
        self.base_kp_knee = 300.0
        self.base_kd_knee = 60.0
        # 脚踝拉满，专治左右摇晃
        self.base_kp_ankle = 400.0
        self.base_kd_ankle = 100.0

        self.kp_waist = 60.0
        self.kd_waist = 30.0
        self.kp_arm = 20.0
        self.kd_arm = 20.0

        # 重心目标（原有逻辑保留，新增重心保护参数）
        self.com_target = np.array([0.08, 0.0, 0.78])
        self.kp_com = 50.0
        self.total_mass = float(np.sum(self.model.body_mass))
        self.weight = float(self.total_mass * abs(float(self.model.opt.gravity[2])))
        self.foot_contact_threshold = float(max(20.0, 0.12 * self.weight))
        self._force_factor_norm = float(max(1.0, 0.5 * self.weight))
        self.com_safety_threshold = 0.6  # 重心z轴安全阈值（新增）
        self.speed_reduction_factor = 0.5  # 重心过低时的降速系数（新增）
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

        # 状态变量（原有逻辑保留，新增鲁棒优化参数）
        self.joint_targets = np.zeros(self.num_joints)
        self.prev_joint_targets = np.zeros(self.num_joints)  # 低通滤波用（新增）
        self.prev_com = np.zeros(3)
        self.foot_contact = np.zeros(2)
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.integral_limit = 0.15
        self.filter_alpha = 0.06  # 低通滤波系数（新增，0.1=更平滑）
        self.enable_robust_optim = True  # 鲁棒优化开关（新增，默认启用）

        # ===================== 新增：多步态模式配置 =====================
        # 步态参数配置（可根据需求调整）
        self.gait_config = {
            "SLOW": {  # 慢走：低频、小步幅、高耦合（稳定）
                "freq": 0.3,  # CPG基础频率
                "amp": 0.3,  # CPG基础振幅
                "coupling": 0.3,  # CPG耦合强度
                "speed_freq_gain": 0.2,  # 速度→频率增益
                "speed_amp_gain": 0.1,  # 速度→振幅增益
                "com_z_offset": 0.02  # 重心z轴抬高（更稳定）
            },
            "NORMAL": {  # 正常走：原有参数
                "freq": 0.5,
                "amp": 0.4,
                "coupling": 0.2,
                "speed_freq_gain": 0.4,
                "speed_amp_gain": 0.2,
                "com_z_offset": 0.0
            },
            "TROT": {  # 小跑：高频、大步幅、中耦合（快速）
                "freq": 0.8,
                "amp": 0.5,
                "coupling": 0.25,
                "speed_freq_gain": 0.5,
                "speed_amp_gain": 0.3,
                "com_z_offset": -0.01  # 重心略降低（提升步幅）
            },
            "STEP_IN_PLACE": {  # 原地踏步：步幅减半、躯干锁定
                "freq": 0.4,
                "amp": 0.2,  # 步幅减半
                "coupling": 0.3,
                "speed_freq_gain": 0.0,  # 速度不影响频率
                "speed_amp_gain": 0.0,  # 速度不影响振幅
                "com_z_offset": 0.01,
                "lock_torso": True  # 锁定躯干偏航/俯仰
            }
        }
        self.gait_mode = "NORMAL"  # 默认步态：正常走
        self.current_gait_params = self.gait_config[self.gait_mode]

        # 运动状态机（扩展步态相关状态，原有状态保留）
        self.state = "STAND"  # 初始状态：STAND/WALK/STOP/EMERGENCY
        self.state_map = {
            "STAND": self._state_stand,  # 站立（初始稳定）
            "WALK": self._state_walk,  # 行走（根据当前步态模式调整）
            "STOP": self._state_stop,  # 停止（关节归零）
            "EMERGENCY": self._state_emergency  # 急停（力矩清零）
        }

        # CPG振荡器（初始化时加载默认步态参数）
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
        self.gait_phase = 0.0  # 保留原始相位变量，兼容调试输出

        # 转向/变速参数（原有逻辑保留，步态切换时动态更新）
        self.turn_angle = 0.0  # 转向角度（左正右负，范围±0.3rad）
        self.turn_gain = 0.1  # 躯干偏航增益
        self.walk_speed = 0.5  # 行走速度（0.1~1.0）
        self.speed_freq_gain = self.current_gait_params["speed_freq_gain"]
        self.speed_amp_gain = self.current_gait_params["speed_amp_gain"]

        # 行走功能参数（原有逻辑保留）
        self.gait_cycle = 2.0
        self.step_offset_hip = 0.4
        self.step_offset_knee = 0.6
        self.step_offset_ankle = 0.3
        self.walk_start_time = None

        # ===================== 传感器模拟相关参数（原有新增逻辑保留） =====================
        self.enable_sensor_simulation = True  # 传感器模拟开关（默认开启）
        # IMU传感器噪声参数（可调整）
        self.imu_angle_noise = 0.01  # 欧拉角噪声标准差（rad）
        self.imu_vel_noise = 0.05  # 角速度噪声标准差（rad/s）
        self.imu_delay_frames = 2  # IMU延迟帧数（模拟硬件传输延迟）
        # 足底力传感器噪声参数
        self.foot_force_noise = 0.3  # 足底力噪声标准差（N）
        self.foot_force_offset = 0.1  # 足底力偏移误差（模拟零漂）
        # 传感器数据缓存（用于模拟延迟）
        self.imu_data_buffer = deque(maxlen=self.imu_delay_frames)
        self.foot_data_buffer = deque(maxlen=self.imu_delay_frames)
        # 存储当前传感器数据（用于调试打印）
        self.current_sensor_data = {}

        # 初始化稳定姿态（原有逻辑完全保留）
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

    # ===================== 新增：步态模式切换接口 =====================
    def set_gait_mode(self, mode):
        """切换步态模式"""
        if mode not in self.gait_config.keys():
            print(f"[警告] 无效的步态模式：{mode}，默认使用NORMAL")
            mode = "NORMAL"

        self.gait_mode = mode
        self.current_gait_params = self.gait_config[mode]

        # 更新CPG基础参数
        self.right_leg_cpg.base_freq = self.current_gait_params["freq"]
        self.right_leg_cpg.base_amp = self.current_gait_params["amp"]
        self.right_leg_cpg.base_coupling = self.current_gait_params["coupling"]

        self.left_leg_cpg.base_freq = self.current_gait_params["freq"]
        self.left_leg_cpg.base_amp = self.current_gait_params["amp"]
        self.left_leg_cpg.base_coupling = self.current_gait_params["coupling"]

        # 更新速度增益
        self.speed_freq_gain = self.current_gait_params["speed_freq_gain"]
        self.speed_amp_gain = self.current_gait_params["speed_amp_gain"]

        # 更新重心目标（步态适配）
        self.com_target[2] = 0.78 + self.current_gait_params["com_z_offset"]

        # 重置CPG状态
        self.right_leg_cpg.reset()
        self.left_leg_cpg.reset()

    def _init_stable_pose(self):
        # """初始化稳定姿态（原有逻辑完全保留）"""
        keep_time = float(self.data.time)
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = keep_time
        self.data.qpos[2] = 0.72
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

        # 腰部关节
        self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["abdomen_y"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["abdomen_x"]] = 0.0

        # 右腿关节
        self.joint_targets[self.joint_name_to_idx["hip_x_right"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_z_right"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_y_right"]] = 0.05
        self.joint_targets[self.joint_name_to_idx["knee_right"]] = -0.70
        self.joint_targets[self.joint_name_to_idx["ankle_y_right"]] = 0.10
        self.joint_targets[self.joint_name_to_idx["ankle_x_right"]] = 0.0

        # 左腿关节
        self.joint_targets[self.joint_name_to_idx["hip_x_left"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_z_left"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_y_left"]] = 0.05
        self.joint_targets[self.joint_name_to_idx["knee_left"]] = -0.70
        self.joint_targets[self.joint_name_to_idx["ankle_y_left"]] = 0.10
        self.joint_targets[self.joint_name_to_idx["ankle_x_left"]] = 0.0

        # 手臂关节
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

    # ===================== 传感器模拟相关方法（原有新增逻辑保留） =====================
    def _simulate_imu_data(self):
        """模拟带噪声+延迟的IMU数据（欧拉角+角速度）"""
        # 获取真实姿态和角速度
        true_quat = self.data.qpos[3:7].astype(np.float64).copy()
        true_euler = self._quat_to_euler_xyz(true_quat)
        true_ang_vel = self.data.qvel[3:6].astype(np.float64).copy()

        # 添加高斯噪声（模拟传感器精度）
        noisy_euler = true_euler + np.random.normal(0, self.imu_angle_noise, 3)
        noisy_ang_vel = true_ang_vel + np.random.normal(0, self.imu_vel_noise, 3)

        # 限幅（模拟传感器量程）
        noisy_euler = np.clip(noisy_euler, -np.pi / 2, np.pi / 2)
        noisy_ang_vel = np.clip(noisy_ang_vel, -5.0, 5.0)

        # 加入缓存（模拟传输延迟）
        self.imu_data_buffer.append({
            "euler": noisy_euler,
            "ang_vel": noisy_ang_vel,
            "true_euler": true_euler,
            "true_ang_vel": true_ang_vel
        })

        # 获取延迟后的传感器数据（缓存不足时用真实数据）
        if len(self.imu_data_buffer) < self.imu_delay_frames:
            return {
                "euler": true_euler,
                "ang_vel": true_ang_vel,
                "true_euler": true_euler,
                "true_ang_vel": true_ang_vel
            }
        else:
            return self.imu_data_buffer[0]  # 返回最早的缓存数据

    def _simulate_foot_force_data(self):
        """模拟带噪声+零漂的足底力传感器数据"""
        true_left_force, true_right_force = self._compute_foot_forces()

        # 添加噪声和零漂（模拟真实传感器）
        noisy_left_force = true_left_force + np.random.normal(0, self.foot_force_noise) + self.foot_force_offset
        noisy_right_force = true_right_force + np.random.normal(0, self.foot_force_noise) + self.foot_force_offset

        # 限幅（避免负力）
        noisy_left_force = max(0.0, noisy_left_force)
        noisy_right_force = max(0.0, noisy_right_force)

        # 接触判定（基于带噪声的力）
        left_contact = 1 if noisy_left_force > self.foot_contact_threshold else 0
        right_contact = 1 if noisy_right_force > self.foot_contact_threshold else 0

        # 加入缓存（模拟延迟）
        self.foot_data_buffer.append({
            "left_force": noisy_left_force,
            "right_force": noisy_right_force,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "true_left_force": true_left_force,
            "true_right_force": true_right_force
        })

        # 获取延迟后的传感器数据
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
        """获取传感器数据（模拟/真实可切换）"""
        if not self.enable_sensor_simulation:
            # 关闭模拟：返回原始数据（兼容原有逻辑）
            imu_data = {
                "euler": self._get_root_euler(),
                "ang_vel": self.data.qvel[3:6].astype(np.float64).copy(),
                "true_euler": self._get_root_euler(),
                "true_ang_vel": self.data.qvel[3:6].astype(np.float64).copy()
            }

            # 原始接触力
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
            # 开启模拟：返回带噪声+延迟的传感器数据
            imu_data = self._simulate_imu_data()
            foot_data = self._simulate_foot_force_data()

        # 整合传感器数据并存储（用于调试）
        self.current_sensor_data = {
            "imu": imu_data,
            "foot": foot_data,
            "time": self.data.time,
            "gait_mode": self.gait_mode  # 新增：记录当前步态
        }

        return self.current_sensor_data

    def print_sensor_data(self):
        """打印当前传感器数据（调试用，新增步态信息）"""
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

    # 状态机方法（WALK状态适配多步态模式）
    # ===================== 站立状态：维持初始稳定姿态，CPG重置 =====================
        #  【修正版】标准直立站立状态   躯干完全挺直，不弯腰不后仰；双腿对称伸直，脚掌放平贴地。
    def _state_stand(self):
        # 站立状态：完全固定姿态，只允许身体转向
        self.right_leg_cpg.reset()
        self.left_leg_cpg.reset()
        self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = self.turn_angle * 0.8
        # self.right_leg_cpg.reset()
        # self.left_leg_cpg.reset()

        # # 1. 躯干/腰部：强制挺直，不弯腰、不后仰
        # self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = 0.0
        # self.joint_targets[self.joint_name_to_idx["abdomen_y"]] = 0.0  # 关键：设为0，不弯腰
        # self.joint_targets[self.joint_name_to_idx["abdomen_x"]] = 0.0

        # # 2. 右腿：标准直立站姿（膝盖微屈、脚踝放平）
        # self.joint_targets[self.joint_name_to_idx["hip_x_right"]] = 0.0
        # self.joint_targets[self.joint_name_to_idx["hip_z_right"]] = 0.0
        # self.joint_targets[self.joint_name_to_idx["hip_y_right"]] = 0.0
        # self.joint_targets[self.joint_name_to_idx["knee_right"]] = -0.70
        # self.joint_targets[self.joint_name_to_idx["ankle_y_right"]] = 0.0 # 关键：设为0，脚掌放平，不上翘不下勾
        # self.joint_targets[self.joint_name_to_idx["ankle_x_right"]] = 0.0

        # # 3. 左腿：和右腿完全对称
        # self.joint_targets[self.joint_name_to_idx["hip_x_left"]] = 0.0
        # self.joint_targets[self.joint_name_to_idx["hip_z_left"]] = 0.0
        # self.joint_targets[self.joint_name_to_idx["hip_y_left"]] = 0.0
        # self.joint_targets[self.joint_name_to_idx["knee_left"]] = -0.70
        # self.joint_targets[self.joint_name_to_idx["ankle_y_left"]] = 0.0
        # self.joint_targets[self.joint_name_to_idx["ankle_x_left"]] = 0.0

        # # 4. 手臂：自然下垂，不影响平衡
        # self.joint_targets[self.joint_name_to_idx["shoulder1_right"]] = 0.0
        # self.joint_targets[self.joint_name_to_idx["shoulder2_right"]] = 0.0
        # self.joint_targets[self.joint_name_to_idx["elbow_right"]] = 1.2
        # self.joint_targets[self.joint_name_to_idx["shoulder1_left"]] = 0.0
        # self.joint_targets[self.joint_name_to_idx["shoulder2_left"]] = 0.0
        # self.joint_targets[self.joint_name_to_idx["elbow_left"]] = 1.2

        # # 5. 低通滤波，消除抖动
        # self.joint_targets = (1 - self.filter_alpha) * self.prev_joint_targets + self.filter_alpha * self.joint_targets
        # self.prev_joint_targets = self.joint_targets.copy()

    def _state_walk(self):
        # """行走状态：根据当前步态模式调整CPG参数 + 原有鲁棒优化"""
        if self.walk_start_time is None:
            self.walk_start_time = self.data.time

        # 重心保护 - 重心过低时自动降速（原有逻辑）
        current_com_z = self.data.subtree_com[0][2]
        if current_com_z < self.com_safety_threshold and self.enable_robust_optim:
            current_speed = self.walk_speed * self.speed_reduction_factor
            self.walk_speed = np.clip(current_speed, 0.1, self.walk_speed)
            if self._should_log("com_low_speed_reduce", 1.0):
                print(
                    f"[鲁棒优化] 重心过低({current_com_z:.2f}m)，自动降速到{self.walk_speed:.2f} | 当前步态: {self.gait_mode}")

        # ===================== 适配多步态的CPG参数更新 =====================
        # 1. 加载当前步态的基础参数
        gait_params = self.current_gait_params

        # 2. 变速联动：根据当前步态的增益调整CPG频率+振幅
        self.right_leg_cpg.freq = gait_params["freq"] + self.walk_speed * gait_params["speed_freq_gain"]
        self.left_leg_cpg.freq = gait_params["freq"] + self.walk_speed * gait_params["speed_freq_gain"]
        self.right_leg_cpg.amp = gait_params["amp"] + self.walk_speed * gait_params["speed_amp_gain"]
        self.left_leg_cpg.amp = gait_params["amp"] + self.walk_speed * gait_params["speed_amp_gain"]

        # 3. 原地踏步模式：锁定躯干偏航，步幅固定
        if self.gait_mode == "STEP_IN_PLACE":
            self.turn_angle = 0.0  # 强制归零转向
            self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = 0.0  # 锁定躯干偏航
            # 步幅固定，不受速度影响
            self.right_leg_cpg.amp = gait_params["amp"]
            self.left_leg_cpg.amp = gait_params["amp"]
        else:
            # 转向联动：躯干偏航 + 左右腿步长差（原有逻辑）
            self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = self.turn_angle * self.turn_gain
            if self.turn_angle > 0:  # 左转
                self.right_leg_cpg.amp *= 1.1
                self.left_leg_cpg.amp *= 0.9
            elif self.turn_angle < 0:  # 右转
                self.right_leg_cpg.amp *= 0.9
                self.left_leg_cpg.amp *= 1.1

        # 4. CPG更新（传入速度/转向系数，动态调整耦合）
        speed_factor = self.walk_speed / 1.0  # 速度归一化（0~1）
        turn_factor = self.turn_angle / 0.3  # 转向归一化（-1~1）
        right_hip_offset = self.right_leg_cpg.update(
            self.dt, target_phase=self.left_leg_cpg.phase,
            speed_factor=speed_factor, turn_factor=turn_factor
        )
        left_hip_offset = self.left_leg_cpg.update(
            self.dt, target_phase=self.right_leg_cpg.phase,
            speed_factor=speed_factor, turn_factor=turn_factor
        )

        # 5. 更新关节目标（原有逻辑保留）
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

        # 关节目标低通滤波（原有逻辑）
        if self.enable_robust_optim:
            self.joint_targets = (
                    1 - self.filter_alpha) * self.prev_joint_targets + self.filter_alpha * self.joint_targets
            self.prev_joint_targets = self.joint_targets.copy()

        # 更新原始步态相位（原有逻辑）
        self.gait_phase = self.right_leg_cpg.phase / (2 * np.pi) % 1.0

    def _state_stop(self):
        """停止状态：关节目标归零，缓慢减速"""
        self.joint_targets *= 0.95  # 渐进归零，避免突变
        self.data.qvel[:] *= 0.9  # 速度阻尼

    def _state_emergency(self):
        """急停状态：力矩清零，速度归零"""
        self.data.ctrl[:] = 0.0
        self.data.qvel[:] = 0.0
        self.joint_targets[:] = 0.0

    # 外部控制接口（原有逻辑完全保留）
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
        # 原地踏步模式下禁止转向
        if self.gait_mode != "STEP_IN_PLACE":
            self.turn_angle = np.clip(angle, -0.3, 0.3)

    def set_walk_speed(self, speed):
        # 原地踏步模式下速度固定
        if self.gait_mode != "STEP_IN_PLACE":
            self.walk_speed = np.clip(speed, 0.1, 1.0)
        else:
            self.walk_speed = 0.5  # 原地踏步固定速度

    # 四元数转欧拉角（原有逻辑完全保留）
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

    # 提取躯干欧拉角（原有逻辑，现在仅用于传感器模拟的真实值参考）
    def _get_root_euler(self):
        quat = self.data.qpos[3:7].astype(np.float64).copy()
        euler = self._quat_to_euler_xyz(quat)
        euler = np.mod(euler + np.pi, 2 * np.pi) - np.pi
        return euler

    # 原有接触检测方法（保留，用于关闭传感器模拟时的 fallback）
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

    # 计算稳定力矩（核心：基于传感器数据的反馈控制，新增步态信息）
    def _calculate_stabilizing_torques(self):
        # 状态机驱动（原有逻辑保留）
        self.state_map[self.state]()

        # 获取传感器数据（模拟/真实可切换）
        sensor_data = self._get_sensor_data()
        imu = sensor_data["imu"]
        foot = sensor_data["foot"]

        # 原始力矩计算逻辑（替换为传感器数据）
        torques = np.zeros(self.num_joints, dtype=np.float64)

        # 躯干姿态控制（改用传感器IMU数据）
        if self.enable_sensor_simulation:
            root_euler = imu.get("true_euler", imu["euler"])
            root_vel = imu.get("true_ang_vel", imu["ang_vel"])
        else:
            root_euler = imu["euler"]
            root_vel = imu["ang_vel"]
        root_vel = np.clip(root_vel, -3.0, 3.0)

        imu_alpha = 0.2
        self._imu_euler_filt = (1.0 - imu_alpha) * self._imu_euler_filt + imu_alpha * root_euler
        self._imu_angvel_filt = (1.0 - imu_alpha) * self._imu_angvel_filt + imu_alpha * root_vel

        roll_error = -self._imu_euler_filt[0]
        self.integral_roll += roll_error * self.dt
        self.integral_roll = np.clip(self.integral_roll, -self.integral_limit, self.integral_limit)
        roll_torque = self.kp_roll * roll_error + self.kd_roll * (-self._imu_angvel_filt[0]) + 5.0 * self.integral_roll

        pitch_error = -self._imu_euler_filt[1]
        self.integral_pitch += pitch_error * self.dt
        self.integral_pitch = np.clip(self.integral_pitch, -self.integral_limit, self.integral_limit)
        pitch_torque = self.kp_pitch * pitch_error + self.kd_pitch * (-self._imu_angvel_filt[1]) + 4.0 * self.integral_pitch

        yaw_error = -self._imu_euler_filt[2]
        yaw_torque = self.kp_yaw * yaw_error + self.kd_yaw * (-self._imu_angvel_filt[2])

        torso_torque = np.array([roll_torque, pitch_torque, yaw_torque])
        torso_torque = np.clip(torso_torque, -30.0, 30.0)

        self.data.xfrc_applied[self._support_body_id, :] = 0.0
        # now_t = float(self.data.time)
        # if now_t < float(self._support_until):
        #     scale = float(np.clip((float(self._support_until) - now_t) / 3.0, 0.0, 1.0))
        #     self.data.xfrc_applied[self._support_body_id, 2] = self.weight * 0.9 * scale
        #     self.data.xfrc_applied[self._support_body_id, 3] = (-80.0 * self._imu_euler_filt[0] - 20.0 * self._imu_angvel_filt[0]) * scale
        #     self.data.xfrc_applied[self._support_body_id, 4] = (-80.0 * self._imu_euler_filt[1] - 20.0 * self._imu_angvel_filt[1]) * scale

        # 重心补偿（原有逻辑完全保留）
        com = self.data.subtree_com[0].astype(np.float64).copy()
        com_error = self.com_target - com
        com_error = np.clip(com_error, -0.03, 0.03)
        com_compensation = self.kp_com * com_error

        # 关节控制（改用传感器足底力数据）
        current_joints = self._get_joint_positions()
        current_vel = self._get_joint_velocities()
        current_vel = np.clip(current_vel, -8.0, 8.0)

        # 更新接触状态（来自传感器）
        self.foot_contact = np.array([foot["right_contact"], foot["left_contact"]])
        self.left_foot_force = foot["left_force"]
        self.right_foot_force = foot["right_force"]

        # 腰部关节控制（原有逻辑完全保留）
        waist_joints = ["abdomen_z", "abdomen_y", "abdomen_x"]
        for joint_name in waist_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = float(self.joint_targets[idx] - current_joints[idx])
            joint_error = max(-0.3, min(0.3, joint_error))
            if joint_name == "abdomen_y":
                joint_error -= float(np.clip(torso_torque[1] * 0.008, -0.15, 0.15))
            torques[idx] = self.kp_waist * joint_error - self.kd_waist * current_vel[idx]

        # 腿部关节控制（新增：基于接触力的动态PD增益）
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

            # 动态PD增益 - 接触力越小，增益越低（避免打滑）
            if self.enable_robust_optim and self.state == "WALK":
                # 计算接触力归一化系数（0~1）
                if "right" in joint_name:
                    force_factor = np.clip(self.right_foot_force / self._force_factor_norm, 0.4, 1.1)
                else:
                    force_factor = np.clip(self.left_foot_force / self._force_factor_norm, 0.4, 1.1)
            else:
                force_factor = 1.0  # 关闭优化则使用原始增益

            # 原有PD增益逻辑，替换为动态增益
            if "hip" in joint_name:
                kp = self.base_kp_hip * force_factor
                kd = self.base_kd_hip * force_factor
                if "y" in joint_name:
                    if "right" in joint_name:
                        joint_error -= torso_torque[1] * 0.02
                    else:
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

            # 原有接触判断逻辑（保留，与动态增益叠加）
            if self.state == "WALK":
                if ("left" in joint_name and self.foot_contact[1] == 0) or \
                        ("right" in joint_name and self.foot_contact[0] == 0):
                    kp *= 0.8
                    kd *= 0.9

            torques[idx] = kp * joint_error - kd * current_vel[idx]

        # 手臂关节控制（原有逻辑完全保留）
        arm_joints = [
            "shoulder1_right", "shoulder2_right", "elbow_right",
            "shoulder1_left", "shoulder2_left", "elbow_left"
        ]
        for joint_name in arm_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            torques[idx] = self.kp_arm * joint_error - self.kd_arm * current_vel[idx]

        # 力矩限幅（原有逻辑完全保留）
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

        # 调试输出（新增步态模式信息）
        if self.data.time > self.init_wait_time and self._should_log("walk_debug", 2.0):
            print(f"=== 行走调试 ===")
            print(
                f"状态: {self.state} | 步态模式: {self.gait_mode} | 步态相位: {self.gait_phase:.2f} | 速度: {self.walk_speed:.2f} | 转向: {self.turn_angle:.2f}")
            print(f"右腿髋目标: {self.joint_targets[self.joint_name_to_idx['hip_y_right']]:.2f}")
            print(f"左腿髋目标: {self.joint_targets[self.joint_name_to_idx['hip_y_left']]:.2f}")
            print(
                f"右脚接触: {self.foot_contact[0]}, 左脚接触: {self.foot_contact[1]} | 鲁棒优化: {'开启' if self.enable_robust_optim else '关闭'} | 传感器模拟: {'开启' if self.enable_sensor_simulation else '关闭'}")

        self.prev_com = com
        return torques

    # 仿真循环（原有逻辑完全保留，新增步态信息输出）
    def simulate_stable_standing(self):
        # 启动ROS /cmd_vel监听线程
        self.ros_handler = ROSCmdVelHandler(self)
        self.ros_handler.start()

        # 启动键盘监听线程
        keyboard_handler = KeyboardInputHandler(self)
        keyboard_handler.start()

        try:
            with viewer.launch_passive(self.model, self.data) as v:
                # 优化相机视角（原有逻辑保留）
                v.cam.distance = 3.0
                v.cam.azimuth = 90
                v.cam.elevation = -25
                v.cam.lookat = [0, 0, 0.6]

                print("人形机器人稳定站立+行走仿真启动（已启用多步态+传感器模拟+步态鲁棒性优化）...")
                print(f"初始稳定{self.init_wait_time}秒后，按W开始行走 | 支持多步态切换（1=慢走/2=正常/3=小跑/4=原地踏步）")
                print(f"默认步态模式：{self.gait_mode}\n")

                # 初始落地阶段（原有逻辑保留）
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

                # 主仿真循环（原有逻辑保留）
                print("=== 初始稳定完成，可输入控制指令 ===")
                while self.data.time < self.sim_duration:
                    torques = self._calculate_stabilizing_torques()
                    self.data.ctrl[:] = self._torques_to_ctrl(torques)
                    mujoco.mj_step(self.model, self.data)

                    # 状态监测（新增步态信息）
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

                    # 跌倒判定（原有逻辑保留）
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
                        self.set_state("STAND")  # 跌倒后自动恢复站立
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
