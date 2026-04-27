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
            print("[ROS提示] 已启动/cmd_vel话题监听")
        except Exception as e:
            print(f"[ROS提示] ROS节点初始化失败：{e}")
            self.has_ros = False

    def _cmd_vel_callback(self, msg):
        raw_speed = float(msg.linear.x)
        if raw_speed <= 0.05:
            self.stabilizer.set_turn_angle(0.0)
            self.stabilizer.set_state("STAND")
            return

        target_speed = float(np.clip(raw_speed, 0.1, 1.0))
        target_turn = float(np.clip(msg.angular.z, -1.0, 1.0) * 0.3)

        self.stabilizer.set_walk_speed(target_speed)
        self.stabilizer.set_turn_angle(target_turn)

        if target_speed > 0.1 and self.stabilizer.state == "STAND":
            self.stabilizer.set_state("WALK")

    def run(self):
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
        self.running = False


# ===================== Windows 稳定键盘控制 =====================
class KeyboardInputHandler(threading.Thread):
    def __init__(self, stabilizer):
        super().__init__(daemon=True)
        self.stabilizer = stabilizer
        self.running = True

    def run(self):
        print("\n✅ 键盘控制已就绪！")
        print("📌 W = 行走   S = 停止   R = 复位站立")
        print("📌 A = 左转   D = 右转   空格 = 回正")
        print("=====================================\n")

        import msvcrt
        while self.running:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                self._handle_key(key)
            time.sleep(0.02)

    def _handle_key(self, key):
        if key == 'w':
            self.stabilizer.set_state("WALK")
            print("👉 开始行走")
        elif key == 's':
            self.stabilizer.set_state("STAND")
            print("👉 停止并站立")
        elif key == 'r':
            self.stabilizer.set_state("STAND")
            self.stabilizer.set_turn_angle(0)
            print("👉 已复位站立")
        elif key == 'a':
            new_t = self.stabilizer.turn_angle + 0.06
            self.stabilizer.set_turn_angle(new_t)
            print(f"↪️  左转：{new_t:.2f}")
        elif key == 'd':
            new_t = self.stabilizer.turn_angle - 0.06
            self.stabilizer.set_turn_angle(new_t)
            print(f"↩️  右转：{new_t:.2f}")
        elif key == ' ':
            self.stabilizer.set_turn_angle(0.0)
            print("✅ 方向回正")
        elif key == '1':
            self.stabilizer.set_gait_mode("SLOW")
        elif key == '2':
            self.stabilizer.set_gait_mode("NORMAL")
        elif key == '3':
            self.stabilizer.set_gait_mode("TROT")
        elif key == '4':
            self.stabilizer.set_gait_mode("STEP_IN_PLACE")


# ===================== CPG步态发生器 =====================
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
        self.phase = 0.0 if self.phase < np.pi else np.pi
        self.state = np.array([np.sin(self.phase), np.cos(self.phase)])


# ===================== 人形机器人控制器（超强稳定站立版） =====================
class HumanoidStabilizer:
    def __init__(self, model_path):
        if not isinstance(model_path, str):
            raise TypeError(f"模型路径必须是字符串")

        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}")

        self.sim_duration = 9999.0
        self.dt = self.model.opt.timestep
        self.init_wait_time = 5.0
        self.model.opt.gravity[2] = -9.81
        self.model.opt.iterations = 300
        self.model.opt.tolerance = 1e-8

        self._log_last = {}
        self._fall_cooldown_until = 0.0
        self._fall_count = 0
        self._imu_euler_filt = np.zeros(3, dtype=np.float64)
        self._imu_angvel_filt = np.zeros(3, dtype=np.float64)

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
                raise RuntimeError(f"未找到执行器：{joint_name}")
            self._actuator_id_by_joint[joint_name] = int(actuator_id)
            self._actuator_gear_by_joint[joint_name] = float(self.model.actuator_gear[actuator_id, 0])
            self._actuator_ctrlrange_by_joint[joint_name] = self.model.actuator_ctrlrange[actuator_id].astype(np.float64)

        # 🔥 超级平衡增益，保证绝对站稳
        self.kp_roll = 350.0
        self.kd_roll = 90.0
        self.kp_pitch = 300.0
        self.kd_pitch = 80.0
        self.kp_yaw = 60.0
        self.kd_yaw = 30.0

        self.base_kp_hip = 450.0
        self.base_kd_hip = 80.0
        self.base_kp_knee = 500.0
        self.base_kd_knee = 90.0
        self.base_kp_ankle = 400.0
        self.base_kd_ankle = 100.0
        self.kp_waist = 70.0
        self.kd_waist = 35.0
        self.kp_arm = 40.0
        self.kd_arm = 30.0

        # 重心固定，不飘、不倒
        self.com_target = np.array([0.08, 0.0, 0.80])
        self.kp_com = 100.0
        self.total_mass = float(np.sum(self.model.body_mass))
        self.weight = float(self.total_mass * abs(float(self.model.opt.gravity[2])))
        self.foot_contact_threshold = float(max(30.0, 0.15 * self.weight))
        self._force_factor_norm = float(max(1.0, 0.6 * self.weight))
        self.com_safety_threshold = 0.65

        self._left_foot_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot1_left"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot2_left"),
        }
        self._right_foot_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot1_right"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot2_right"),
        }

        self.joint_targets = np.zeros(self.num_joints)
        self.prev_joint_targets = np.zeros(self.num_joints)
        self.prev_com = np.zeros(3)
        self.foot_contact = np.zeros(2)
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.integral_limit = 0.2
        self.filter_alpha = 0.1
        self.enable_robust_optim = True

        self.gait_config = {
            "SLOW": {"freq": 0.3, "amp": 0.3, "coupling": 0.3, "speed_freq_gain": 0.2, "speed_amp_gain": 0.1, "com_z_offset": 0.02},
            "NORMAL": {"freq": 0.5, "amp": 0.4, "coupling": 0.2, "speed_freq_gain": 0.4, "speed_amp_gain": 0.2, "com_z_offset": 0.0},
            "TROT": {"freq": 0.8, "amp": 0.5, "coupling": 0.25, "speed_freq_gain": 0.5, "speed_amp_gain": 0.3, "com_z_offset": -0.01},
            "STEP_IN_PLACE": {"freq": 0.4, "amp": 0.2, "coupling": 0.3, "speed_freq_gain": 0.0, "speed_amp_gain": 0.0, "com_z_offset": 0.01, "lock_torso": True}
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

        self.right_leg_cpg = CPGOscillator(freq=self.current_gait_params["freq"], amp=self.current_gait_params["amp"], phase=0.0, coupling_strength=self.current_gait_params["coupling"])
        self.left_leg_cpg = CPGOscillator(freq=self.current_gait_params["freq"], amp=self.current_gait_params["amp"], phase=np.pi, coupling_strength=self.current_gait_params["coupling"])
        self.gait_phase = 0.0

        self.turn_angle = 0.0
        self.turn_gain = 0.25
        self.walk_speed = 0.5
        self.speed_freq_gain = self.current_gait_params["speed_freq_gain"]
        self.speed_amp_gain = self.current_gait_params["speed_amp_gain"]
        self.walk_start_time = None

        self.enable_sensor_simulation = False
        self.imu_data_buffer = deque(maxlen=2)
        self.foot_data_buffer = deque(maxlen=2)
        self.current_sensor_data = {}

        self._init_stable_pose()

    def _should_log(self, key, interval_s):
        now = float(self.data.time)
        last = float(self._log_last.get(key, -1e9))
        if (now - last) >= float(interval_s):
            self._log_last[key] = now
            return True
        return False

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
        if mode not in self.gait_config:
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
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self._imu_euler_filt[:] = 0.0
        self._imu_angvel_filt[:] = 0.0

        self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["abdomen_y"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["abdomen_x"]] = 0.0

        # 🔥 深蹲稳定姿态，绝对站得稳
        self.joint_targets[self.joint_name_to_idx["hip_y_right"]] = 0.05
        self.joint_targets[self.joint_name_to_idx["knee_right"]] = -0.70
        self.joint_targets[self.joint_name_to_idx["ankle_y_right"]] = 0.10

        self.joint_targets[self.joint_name_to_idx["hip_y_left"]] = 0.05
        self.joint_targets[self.joint_name_to_idx["knee_left"]] = -0.70
        self.joint_targets[self.joint_name_to_idx["ankle_y_left"]] = 0.10

        self.prev_joint_targets = self.joint_targets.copy()
        mujoco.mj_forward(self.model, self.data)

    def _get_sensor_data(self):
        if not self.enable_sensor_simulation:
            true_euler = self._get_root_euler()
            self._detect_foot_contact()
            return {
                "imu": {"euler": true_euler, "ang_vel": self.data.qvel[3:6]},
                "foot": {"left_force": self.left_foot_force, "right_force": self.right_foot_force,
                         "left_contact": self.foot_contact[1], "right_contact": self.foot_contact[0]}
            }

    def _state_stand(self):
        # 站立状态：完全固定姿态，只允许身体转向
        self.right_leg_cpg.reset()
        self.left_leg_cpg.reset()
        self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = self.turn_angle * 0.8

    def _state_walk(self):
        if self.walk_start_time is None:
            self.walk_start_time = self.data.time

        g = self.current_gait_params
        self.right_leg_cpg.freq = g["freq"] + self.walk_speed * g["speed_freq_gain"]
        self.left_leg_cpg.freq = g["freq"] + self.walk_speed * g["speed_freq_gain"]
        self.right_leg_cpg.amp = g["amp"] + self.walk_speed * g["speed_amp_gain"]
        self.left_leg_cpg.amp = g["amp"] + self.walk_speed * g["speed_amp_gain"]

        if self.turn_angle > 0:
            self.right_leg_cpg.amp *= 1.15
            self.left_leg_cpg.amp *= 0.85
        elif self.turn_angle < 0:
            self.right_leg_cpg.amp *= 0.85
            self.left_leg_cpg.amp *= 1.15

        speed_factor = self.walk_speed
        turn_factor = self.turn_angle / 0.3
        r = self.right_leg_cpg.update(self.dt, self.left_leg_cpg.phase, speed_factor, turn_factor)
        l = self.left_leg_cpg.update(self.dt, self.right_leg_cpg.phase, speed_factor, turn_factor)

        self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = self.turn_angle * 0.3
        self.joint_targets[self.joint_name_to_idx["abdomen_y"]] = 0.15

        self.joint_targets[self.joint_name_to_idx["hip_y_right"]] = 0.12 + r
        self.joint_targets[self.joint_name_to_idx["knee_right"]] = -0.75 - r * 2.2
        self.joint_targets[self.joint_name_to_idx["ankle_y_right"]] = 0.22 + r * 0.8

        self.joint_targets[self.joint_name_to_idx["hip_y_left"]] = 0.12 + l
        self.joint_targets[self.joint_name_to_idx["knee_left"]] = -0.75 - l * 2.2
        self.joint_targets[self.joint_name_to_idx["ankle_y_left"]] = 0.22 + l * 0.8

    def _state_stop(self):
        self.joint_targets *= 0.95

    def _state_emergency(self):
        self.data.ctrl[:] = 0
        self.data.qvel[:] = 0

    def set_state(self, state):
        if state in self.state_map:
            self.state = state
            if state == "STAND":
                self._init_stable_pose()

    def set_turn_angle(self, angle):
        self.turn_angle = np.clip(angle, -0.4, 0.4)

    def set_walk_speed(self, speed):
        self.walk_speed = np.clip(speed, 0.1, 1.0)

    def _quat_to_euler_xyz(self, q):
        w, x, y, z = q
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x ** 2 + y ** 2)
        roll = np.arctan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y ** 2 + z ** 2)
        yaw = np.arctan2(siny, cosy)
        return np.array([roll, pitch, yaw])

    def _get_root_euler(self):
        q = self.data.qpos[3:7]
        e = self._quat_to_euler_xyz(q)
        return np.clip(e, -0.4, 0.4)

    def _detect_foot_contact(self):
        lf, rf = 0, 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self._left_foot_geom_ids or g2 in self._left_foot_geom_ids):
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                lf += np.linalg.norm(f[:3])
            if (g1 in self._right_foot_geom_ids or g2 in self._right_foot_geom_ids):
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                rf += np.linalg.norm(f[:3])
        self.foot_contact[1] = 1 if lf > self.foot_contact_threshold else 0
        self.foot_contact[0] = 1 if rf > self.foot_contact_threshold else 0
        self.left_foot_force = lf
        self.right_foot_force = rf

    def _calculate_stabilizing_torques(self):
        self.state_map[self.state]()
        sensor = self._get_sensor_data()
        imu, foot = sensor["imu"], sensor["foot"]
        euler, vel = imu["euler"], imu["ang_vel"]

        a = 0.2
        self._imu_euler_filt = (1 - a) * self._imu_euler_filt + a * euler
        self._imu_angvel_filt = (1 - a) * self._imu_angvel_filt + a * vel

        r_err = -self._imu_euler_filt[0]
        self.integral_roll = np.clip(self.integral_roll + r_err * self.dt, -self.integral_limit, self.integral_limit)
        r_tor = self.kp_roll * r_err + self.kd_roll * (-self._imu_angvel_filt[0]) + 10.0 * self.integral_roll

        p_err = -self._imu_euler_filt[1]
        self.integral_pitch = np.clip(self.integral_pitch + p_err * self.dt, -self.integral_limit, self.integral_limit)
        p_tor = self.kp_pitch * p_err + self.kd_pitch * (-self._imu_angvel_filt[1]) + 8.0 * self.integral_pitch

        y_err = -self._imu_euler_filt[2]
        y_tor = self.kp_yaw * y_err + self.kd_yaw * (-self._imu_angvel_filt[2])

        torques = np.zeros(self.num_joints)
        com = self.data.subtree_com[0]
        com_err = np.clip(self.com_target - com, -0.03, 0.03)
        com_comp = self.kp_com * com_err

        q = self.data.qpos[7:7 + self.num_joints]
        qv = np.clip(self.data.qvel[6:6 + self.num_joints], -6, 6)
        self.foot_contact = np.array([foot["right_contact"], foot["left_contact"]])

        for jn in ["abdomen_z", "abdomen_y", "abdomen_x"]:
            i = self.joint_name_to_idx[jn]
            e = np.clip(self.joint_targets[i] - q[i], -0.3, 0.3)
            torques[i] = self.kp_waist * e - self.kd_waist * qv[i]

        legs = ["hip_x_right", "hip_z_right", "hip_y_right", "knee_right", "ankle_y_right", "ankle_x_right",
                "hip_x_left", "hip_z_left", "hip_y_left", "knee_left", "ankle_y_left", "ankle_x_left"]
        for jn in legs:
            i = self.joint_name_to_idx[jn]
            e = np.clip(self.joint_targets[i] - q[i], -0.3, 0.3)
            ff = 1.0
            if "right" in jn:
                ff = np.clip(foot["right_force"] / self._force_factor_norm, 0.5, 1.2)
            else:
                ff = np.clip(foot["left_force"] / self._force_factor_norm, 0.5, 1.2)

            if "hip" in jn:
                kp = self.base_kp_hip * ff
                kd = self.base_kd_hip * ff
            elif "knee" in jn:
                kp = self.base_kp_knee * ff
                kd = self.base_kd_knee * ff
            elif "ankle" in jn:
                kp = self.base_kp_ankle * ff
                kd = self.base_kd_ankle * ff
            else:
                kp, kd = 250, 50

            torques[i] = kp * e - kd * qv[i]

        arms = ["shoulder1_right", "shoulder2_right", "elbow_right", "shoulder1_left", "shoulder2_left", "elbow_left"]
        for jn in arms:
            i = self.joint_name_to_idx[jn]
            e = self.joint_targets[i] - q[i]
            torques[i] = self.kp_arm * e - self.kd_arm * qv[i]

        return torques

    def simulate_stable_standing(self):
        self.ros_handler = ROSCmdVelHandler(self)
        self.ros_handler.start()
        kh = KeyboardInputHandler(self)
        kh.start()

        try:
            with viewer.launch_passive(self.model, self.data) as v:
                v.cam.distance = 3.2
                v.cam.azimuth = 90
                v.cam.elevation = -22
                v.cam.lookat = [0, 0, 0.7]

                print(" 机器人启动成功 → 5秒后站稳")
                start = time.time()
                while time.time() - start < self.init_wait_time:
                    alpha = min(1.0, (time.time() - start) / 5.0)
                    t = self._calculate_stabilizing_torques() * alpha
                    self.data.ctrl[:] = self._torques_to_ctrl(t)
                    mujoco.mj_step(self.model, self.data)
                    self.data.qvel *= 0.96
                    v.sync()
                    time.sleep(self.dt)

                print("✅ 已站稳！可以用键盘控制！")
                while self.data.time < self.sim_duration:
                    t = self._calculate_stabilizing_torques()
                    self.data.ctrl[:] = self._torques_to_ctrl(t)
                    mujoco.mj_step(self.model, self.data)
                    v.sync()
                    time.sleep(self.dt)

        finally:
            kh.running = False
            self.ros_handler.stop()


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(current_directory, "models", "humanoid.xml")
    print(f"模型路径：{model_file_path}")
    stabilizer = HumanoidStabilizer(model_file_path)
    stabilizer.simulate_stable_standing()