import mujoco
import numpy as np
from mujoco import viewer
import time
import os
import sys
import threading
from collections import deque


class ROSCmdVelHandler(threading.Thread):
    def __init__(self, stabilizer):
        super().__init__(daemon=True)
        self.stabilizer = stabilizer
        self.running = True
        self.has_ros = False
        self.twist_msg = None

        try:
            import rospy
            from geometry_msgs.msg import Twist
            self.rospy = rospy
            self.Twist = Twist
            self.has_ros = True
        except ImportError:
            print("[ROS提示] 未检测到ROS环境，跳过/cmd_vel话题监听（仅保留键盘控制）")
            return

        try:
            if not self.rospy.core.is_initialized():
                self.rospy.init_node('humanoid_cmd_vel_listener', anonymous=True)
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
        self.twist_msg = msg
        target_speed = np.clip(msg.linear.x, 0.1, 1.0)
        target_turn = np.clip(msg.angular.z, -1.0, 1.0) * 0.3

        self.stabilizer.set_walk_speed(target_speed)
        self.stabilizer.set_turn_angle(target_turn)

        if target_speed > 0.1 and self.stabilizer.state == "STAND":
            self.stabilizer.set_state("WALK")

        if self.stabilizer.data.time % 0.5 < 0.1:
            print(
                f"[ROS指令] 速度={target_speed:.2f} | 转向={target_turn:.2f}rad | 当前步态: {self.stabilizer.gait_mode}")

    def run(self):
        if not self.has_ros:
            return
        while self.running and not self.rospy.is_shutdown():
            self.rospy.spin_once()
            time.sleep(0.01)

    def stop(self):
        self.running = False


class KeyboardInputHandler(threading.Thread):
    def __init__(self, stabilizer):
        super().__init__(daemon=True)
        self.stabilizer = stabilizer
        self.running = True

    def run(self):
        print("\n===== 控制指令说明 =====")
        print("w: 开始行走 | s: 停止行走 | e: 紧急停止 | r: 恢复站立")
        print("a: 左转 | d: 右转 | 空格: 原地转向 | z: 减速 | x: 加速")
        print("m: 传感器模拟开关 | p: 打印传感器数据")
        print("1: 慢走 | 2: 正常走 | 3: 小跑 | 4: 原地踏步")
        print("========================\n")
        while self.running:
            try:
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
        key_actions = {
            'w': lambda: (self.stabilizer.set_state("WALK"), self.stabilizer.set_gait_mode(self.stabilizer.gait_mode),
                          print(
                              f"[指令] 切换为行走状态 | 当前步态: {self.stabilizer.gait_mode} | 速度: {self.stabilizer.walk_speed:.2f} | 转向: {self.stabilizer.turn_angle:.2f}")),
            's': lambda: (self.stabilizer.set_state("STOP"), print("[指令] 切换为停止状态")),
            'e': lambda: (self.stabilizer.set_state("EMERGENCY"), print("[指令] 触发紧急停止")),
            'r': lambda: (self.stabilizer.set_state("STAND"), print("[指令] 恢复站立姿态")),
            'a': lambda: (self.stabilizer.set_turn_angle(self.stabilizer.turn_angle + 0.05),
                          print(f"[指令] 左转 | 当前转向角度: {self.stabilizer.turn_angle:.2f}rad")),
            'd': lambda: (self.stabilizer.set_turn_angle(self.stabilizer.turn_angle - 0.05),
                          print(f"[指令] 右转 | 当前转向角度: {self.stabilizer.turn_angle:.2f}rad")),
            ' ': lambda: (self.stabilizer.set_turn_angle(0.2 if self.stabilizer.turn_angle <= 0 else -0.2),
                          print(f"[指令] 原地转向 | 当前转向角度: {self.stabilizer.turn_angle:.2f}rad")),
            'z': lambda: (self.stabilizer.set_walk_speed(self.stabilizer.walk_speed - 0.1),
                          print(
                              f"[指令] 减速 | 当前速度: {self.stabilizer.walk_speed:.2f} | 当前步态: {self.stabilizer.gait_mode}")),
            'x': lambda: (self.stabilizer.set_walk_speed(self.stabilizer.walk_speed + 0.1),
                          print(
                              f"[指令] 加速 | 当前速度: {self.stabilizer.walk_speed:.2f} | 当前步态: {self.stabilizer.gait_mode}")),
            'm': lambda: (
            setattr(self.stabilizer, 'enable_sensor_simulation', not self.stabilizer.enable_sensor_simulation),
            print(f"[指令] 传感器模拟{'开启' if self.stabilizer.enable_sensor_simulation else '关闭'}")),
            'p': lambda: self.stabilizer.print_sensor_data(),
            '1': lambda: (self.stabilizer.set_gait_mode("SLOW"),
                          print(
                              f"[指令] 切换为慢走模式 | CPG频率: {self.stabilizer.gait_config['SLOW']['freq']:.2f}Hz | 振幅: {self.stabilizer.gait_config['SLOW']['amp']:.2f}")),
            '2': lambda: (self.stabilizer.set_gait_mode("NORMAL"),
                          print(
                              f"[指令] 切换为正常走模式 | CPG频率: {self.stabilizer.gait_config['NORMAL']['freq']:.2f}Hz | 振幅: {self.stabilizer.gait_config['NORMAL']['amp']:.2f}")),
            '3': lambda: (self.stabilizer.set_gait_mode("TROT"),
                          print(
                              f"[指令] 切换为小跑模式 | CPG频率: {self.stabilizer.gait_config['TROT']['freq']:.2f}Hz | 振幅: {self.stabilizer.gait_config['TROT']['amp']:.2f}")),
            '4': lambda: (self.stabilizer.set_gait_mode("STEP_IN_PLACE"),
                          print(f"[指令] 切换为原地踏步模式 | 步幅减半，躯干锁定"))
        }
        if key in key_actions:
            key_actions[key]()


class CPGOscillator:
    def __init__(self, freq=0.5, amp=0.4, phase=0.0, coupling_strength=0.2):
        self.base_freq = freq
        self.base_amp = amp
        self.freq = freq
        self.amp = amp
        self.phase = phase
        self.base_coupling = coupling_strength
        self.coupling = coupling_strength
        self.state = np.array([np.sin(phase), np.cos(phase)])

    def update(self, dt, target_phase=0.0, speed_factor=1.0, turn_factor=0.0):
        self.coupling = self.base_coupling * (1.0 + 0.5 * speed_factor + 0.8 * abs(turn_factor))
        self.coupling = np.clip(self.coupling, 0.1, 0.5)

        mu = 1.0
        x, y = self.state
        dx = 2 * np.pi * self.freq * y + self.coupling * np.sin(target_phase - self.phase)
        dy = 2 * np.pi * self.freq * (mu * (1 - x ** 2) * y - x)
        self.state += np.array([dx, dy]) * dt
        self.phase = np.arctan2(self.state[0], self.state[1])
        return self.amp * self.state[0]

    def reset(self):
        self.freq = self.base_freq
        self.amp = self.base_amp
        self.coupling = self.base_coupling
        self.phase = 0.0 if self.base_phase == 0.0 else np.pi
        self.state = np.array([np.sin(self.phase), np.cos(self.phase)])

    @property
    def base_phase(self):
        return 0.0 if self.phase < np.pi else np.pi


class HumanoidStabilizer:
    def __init__(self, model_path):
        if not isinstance(model_path, str):
            raise TypeError(f"模型路径必须是字符串，当前是 {type(model_path)} 类型")

        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}\n请检查路径和文件完整性")

        self.sim_duration = 120.0
        self.dt = self.model.opt.timestep
        self.init_wait_time = 4.0
        self.model.opt.gravity[2] = -9.81
        self.model.opt.iterations = 200
        self.model.opt.tolerance = 1e-8

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

        self.kp_roll = 120.0
        self.kd_roll = 40.0
        self.kp_pitch = 100.0
        self.kd_pitch = 35.0
        self.kp_yaw = 30.0
        self.kd_yaw = 15.0
        self.base_kp_hip = 250.0
        self.base_kd_hip = 45.0
        self.base_kp_knee = 280.0
        self.base_kd_knee = 50.0
        self.base_kp_ankle = 200.0
        self.base_kd_ankle = 60.0
        self.kp_waist = 40.0
        self.kd_waist = 20.0
        self.kp_arm = 20.0
        self.kd_arm = 20.0

        self.com_target = np.array([0.05, 0.0, 0.78])
        self.kp_com = 50.0
        self.foot_contact_threshold = 1.5
        self.com_safety_threshold = 0.6
        self.speed_reduction_factor = 0.5

        self.joint_targets = np.zeros(self.num_joints)
        self.prev_joint_targets = np.zeros(self.num_joints)
        self.prev_com = np.zeros(3)
        self.foot_contact = np.zeros(2)
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.integral_limit = 0.15
        self.filter_alpha = 0.1
        self.enable_robust_optim = True

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
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 1.282
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

        joint_init_vals = {
            "abdomen_z": 0.0, "abdomen_y": 0.0, "abdomen_x": 0.0,
            "hip_x_right": 0.0, "hip_z_right": 0.0, "hip_y_right": 0.1,
            "knee_right": -0.4, "ankle_y_right": 0.0, "ankle_x_right": 0.0,
            "hip_x_left": 0.0, "hip_z_left": 0.0, "hip_y_left": 0.1,
            "knee_left": -0.4, "ankle_y_left": 0.0, "ankle_x_left": 0.0,
            "shoulder1_right": 0.1, "shoulder2_right": 0.1, "elbow_right": 1.5,
            "shoulder1_left": 0.1, "shoulder2_left": 0.1, "elbow_left": 1.5
        }
        for name, val in joint_init_vals.items():
            self.joint_targets[self.joint_name_to_idx[name]] = val

        mujoco.mj_forward(self.model, self.data)

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
        def get_foot_force(foot_geoms):
            force = 0.0
            for geom_name in foot_geoms:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                f = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, geom_id, f)
                force += np.linalg.norm(f[:3])
            return force

        left_foot_geoms = ["foot1_left", "foot2_left"]
        right_foot_geoms = ["foot1_right", "foot2_right"]

        true_left_force = get_foot_force(left_foot_geoms)
        true_right_force = get_foot_force(right_foot_geoms)

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
        self._init_stable_pose()

    def _state_walk(self):
        if self.walk_start_time is None:
            self.walk_start_time = self.data.time

        current_com_z = self.data.subtree_com[0][2]
        if current_com_z < self.com_safety_threshold and self.enable_robust_optim:
            current_speed = self.walk_speed * self.speed_reduction_factor
            self.walk_speed = np.clip(current_speed, 0.1, self.walk_speed)
            if self.data.time % 1 < 0.1:
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

        leg_joint_offsets = {
            "hip_y_right": 0.1 + right_hip_offset,
            "knee_right": -0.4 - right_hip_offset * 1.2,
            "ankle_y_right": 0.0 + right_hip_offset * 0.5,
            "hip_y_left": 0.1 + left_hip_offset,
            "knee_left": -0.4 - left_hip_offset * 1.2,
            "ankle_y_left": 0.0 + left_hip_offset * 0.5
        }
        for name, val in leg_joint_offsets.items():
            self.joint_targets[self.joint_name_to_idx[name]] = val

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
        return np.clip(euler, -0.3, 0.3)

    def _detect_foot_contact(self):
        def get_foot_force(foot_geoms):
            force = 0.0
            for geom_name in foot_geoms:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                f = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, geom_id, f)
                force += np.linalg.norm(f[:3])
            return force

        left_foot_geoms = ["foot1_left", "foot2_left"]
        right_foot_geoms = ["foot1_right", "foot2_right"]

        self.left_foot_force = get_foot_force(left_foot_geoms)
        self.right_foot_force = get_foot_force(right_foot_geoms)

        self.foot_contact[1] = 1 if self.left_foot_force > self.foot_contact_threshold else 0
        self.foot_contact[0] = 1 if self.right_foot_force > self.foot_contact_threshold else 0

    def _calculate_stabilizing_torques(self):
        self.state_map[self.state]()

        sensor_data = self._get_sensor_data()
        imu = sensor_data["imu"]
        foot = sensor_data["foot"]

        torques = np.zeros(self.num_joints, dtype=np.float64)

        root_euler = imu["euler"]
        root_vel = imu["ang_vel"]
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

        com = self.data.subtree_com[0].astype(np.float64).copy()
        com_error = self.com_target - com
        com_error = np.clip(com_error, -0.03, 0.03)
        com_compensation = self.kp_com * com_error

        current_joints = self.data.qpos[7:7 + self.num_joints].astype(np.float64)
        current_vel = self.data.qvel[6:6 + self.num_joints].astype(np.float64)
        current_vel = np.clip(current_vel, -8.0, 8.0)

        self.foot_contact = np.array([foot["right_contact"], foot["left_contact"]])
        self.left_foot_force = foot["left_force"]
        self.right_foot_force = foot["right_force"]

        waist_joints = ["abdomen_z", "abdomen_y", "abdomen_x"]
        for joint_name in waist_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            joint_error = np.clip(joint_error, -0.3, 0.3)
            torques[idx] = self.kp_waist * joint_error - self.kd_waist * current_vel[idx]

        leg_joints = [
            "hip_x_right", "hip_z_right", "hip_y_right",
            "knee_right", "ankle_y_right", "ankle_x_right",
            "hip_x_left", "hip_z_left", "hip_y_left",
            "knee_left", "ankle_y_left", "ankle_x_left"
        ]

        torque_config = {
            "hip": {"base_kp": self.base_kp_hip, "base_kd": self.base_kd_hip},
            "knee": {"base_kp": self.base_kp_knee, "base_kd": self.base_kd_knee},
            "ankle": {"base_kp": self.base_kp_ankle, "base_kd": self.base_kd_ankle}
        }

        for joint_name in leg_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            joint_error = np.clip(joint_error, -0.3, 0.3)

            force_factor = 1.0
            if self.enable_robust_optim:
                if "right" in joint_name:
                    force_factor = np.clip(self.right_foot_force / (self.foot_contact_threshold * 2), 0.5, 1.0)
                else:
                    force_factor = np.clip(self.left_foot_force / (self.foot_contact_threshold * 2), 0.5, 1.0)

            joint_type = next(key for key in torque_config.keys() if key in joint_name)
            kp = torque_config[joint_type]["base_kp"] * force_factor
            kd = torque_config[joint_type]["base_kd"] * force_factor

            if joint_type == "hip" and "y" in joint_name:
                joint_error += torso_torque[1] * 0.02
            elif joint_type == "knee":
                joint_error += com_compensation[2] * 0.05
            elif joint_type == "ankle" and "y" in joint_name:
                joint_error += torso_torque[1] * 0.01

            if ("left" in joint_name and self.foot_contact[1] == 0) or \
                    ("right" in joint_name and self.foot_contact[0] == 0):
                kp *= 0.8
                kd *= 0.9

            torques[idx] = kp * joint_error - kd * current_vel[idx]

        arm_joints = [
            "shoulder1_right", "shoulder2_right", "elbow_right",
            "shoulder1_left", "shoulder2_left", "elbow_left"
        ]
        for joint_name in arm_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            torques[idx] = self.kp_arm * joint_error - self.kd_arm * current_vel[idx]

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

        if self.data.time % 1 < 0.1 and self.data.time > self.init_wait_time:
            print(f"=== 行走调试 ===")
            print(
                f"状态: {self.state} | 步态模式: {self.gait_mode} | 步态相位: {self.gait_phase:.2f} | 速度: {self.walk_speed:.2f} | 转向: {self.turn_angle:.2f}")
            print(f"右腿髋目标: {self.joint_targets[self.joint_name_to_idx['hip_y_right']]:.2f}")
            print(f"左腿髋目标: {self.joint_targets[self.joint_name_to_idx['hip_y_left']]:.2f}")
            print(
                f"右脚接触: {self.foot_contact[0]}, 左脚接触: {self.foot_contact[1]} | 鲁棒优化: {'开启' if self.enable_robust_optim else '关闭'} | 传感器模拟: {'开启' if self.enable_sensor_simulation else '关闭'}")

        self.prev_com = com
        return torques

    def simulate_stable_standing(self):
        self.ros_handler = ROSCmdVelHandler(self)
        self.ros_handler.start()

        keyboard_handler = KeyboardInputHandler(self)
        keyboard_handler.start()

        with viewer.launch_passive(self.model, self.data) as v:
            v.cam.distance = 3.0
            v.cam.azimuth = 90
            v.cam.elevation = -25
            v.cam.lookat = [0, 0, 0.6]

            print("人形机器人稳定站立+行走仿真启动（已启用多步态+传感器模拟+步态鲁棒性优化）...")
            print(f"初始稳定{self.init_wait_time}秒后，按W开始行走 | 支持多步态切换（1=慢走/2=正常/3=小跑/4=原地踏步）")
            print(f"默认步态模式：{self.gait_mode}\n")

            start_time = time.time()
            while time.time() - start_time < self.init_wait_time:
                alpha = min(1.0, (time.time() - start_time) / self.init_wait_time)
                torques = self._calculate_stabilizing_torques() * alpha
                self.data.ctrl[:] = torques
                mujoco.mj_step(self.model, self.data)
                self.data.qvel[:] *= 0.97
                v.sync()
                time.sleep(self.dt)

            print("=== 初始稳定完成，可输入控制指令 ===")
            while self.data.time < self.sim_duration:
                torques = self._calculate_stabilizing_torques()
                self.data.ctrl[:] = torques
                mujoco.mj_step(self.model, self.data)

                if self.data.time % 2 < 0.1:
                    com = self.data.subtree_com[0]
                    euler = self.current_sensor_data["imu"][
                        "euler"] if self.current_sensor_data else self._get_root_euler()
                    print(
                        f"时间:{self.data.time:.1f}s | 重心(x/z):{com[0]:.3f}/{com[2]:.3f}m | "
                        f"姿态(roll/pitch):{euler[0]:.3f}/{euler[1]:.3f}rad | 脚接触:{self.foot_contact} | "
                        f"当前步态:{self.gait_mode}"
                    )

                v.sync()
                time.sleep(self.dt * 0.5)

                com = self.data.subtree_com[0]
                euler = self.current_sensor_data["imu"]["euler"] if self.current_sensor_data else self._get_root_euler()
                if com[2] < 0.4 or abs(euler[0]) > 0.6 or abs(euler[1]) > 0.6:
                    print(
                        f"跌倒！时间:{self.data.time:.1f}s | 重心(z):{com[2]:.3f}m | "
                        f"最大倾角:{max(abs(euler[0]), abs(euler[1])):.3f}rad | 当前步态:{self.gait_mode}"
                    )
                    self.set_state("STAND")

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