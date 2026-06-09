import numpy as np
import mujoco
from mujoco import viewer
import time
import os
import sys
import threading
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


# ===================== 梯度裁剪回调 =====================
class GradientClipCallback(BaseCallback):
    def __init__(self, clip_value: float = 1.0, verbose: int = 0):
        super().__init__(verbose)
        self.clip_value = clip_value

    def _on_step(self) -> bool:
        for param in self.model.policy.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-self.clip_value, self.clip_value)
        return True


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
            print("[ROS提示] 未检测到ROS环境，跳过/cmd_vel话题监听")
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
        target_turn = float(np.clip(msg.angular.z, -1.0, 1.0) * 0.2)
        self.stabilizer.set_turn_angle(target_turn)

        if abs(raw_speed) < 0.05:
            self.stabilizer.set_state("STAND")
            return

        if raw_speed > 0:
            target_speed = float(np.clip(raw_speed, 0.05, 0.30))
            self.stabilizer.set_walk_speed(target_speed)
            if self.stabilizer.state == "STAND":
                self.stabilizer.set_state("PREPARE")
        else:
            target_speed = float(np.clip(raw_speed, -0.25, -0.05))
            self.stabilizer.set_walk_speed(target_speed)
            if self.stabilizer.state == "STAND":
                self.stabilizer.set_state("PREPARE")

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


# ===================== 键盘控制 =====================
class KeyboardInputHandler(threading.Thread):
    def __init__(self, stabilizer):
        super().__init__(daemon=True)
        self.stabilizer = stabilizer
        self.running = True

    def run(self):
        print("\n 键盘控制已就绪！")
        print(" W = 缓步前进   S = 停止   R = 复位")
        print(" X = 缓步后退   A = 左转   D = 右转")
        print(" 空格 = 回正   1=慢走 2=正常 3=小跑 4=原地踏步")
        print(" P = 加载SAC智能步态")
        print("=====================================\n")

        import msvcrt
        while self.running:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\x03':
                    break
                key = key.decode('utf-8', errors='ignore').lower()
                if key:
                    self._handle_key(key)
            time.sleep(0.02)

    def _handle_key(self, key):
        if key == 'w':
            self.stabilizer.set_walk_speed(0.22)
            self.stabilizer.set_state("PREPARE")
            print("缓步前进")
        elif key == 'x':
            self.stabilizer.set_walk_speed(-0.15)
            self.stabilizer.set_state("PREPARE")
            print("缓步后退")
        elif key == 's':
            self.stabilizer.set_state("STOP")
            print("缓停站立")
        elif key == 'r':
            self.stabilizer._init_stable_pose()
            self.stabilizer.set_state("STAND")
            self.stabilizer.set_turn_angle(0)
            print("已复位")
        elif key == 'a':
            new_t = self.stabilizer.turn_angle + 0.03
            self.stabilizer.set_turn_angle(new_t)
            print(f"左转：{new_t:.2f}")
        elif key == 'd':
            new_t = self.stabilizer.turn_angle - 0.03
            self.stabilizer.set_turn_angle(new_t)
            print(f"右转：{new_t:.2f}")
        elif key == ' ':
            self.stabilizer.set_turn_angle(0.0)
            print("方向回正")
        elif key == '1':
            self.stabilizer.set_gait_mode("SLOW")
        elif key == '2':
            self.stabilizer.set_gait_mode("NORMAL")
        elif key == '3':
            self.stabilizer.set_gait_mode("TROT")
        elif key == '4':
            self.stabilizer.set_gait_mode("STEP_IN_PLACE")
        elif key == 'p':
            self.stabilizer.load_sac_policy()
            print("已加载SAC步态")


# ===================== CPG振荡器 =====================
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
        self.smooth_alpha = 0.08
        self.tar_freq = freq
        self.tar_amp = amp

    def set_target(self, freq, amp):
        self.tar_freq = freq
        self.tar_amp = amp

    def update_smooth(self):
        self.freq = (1 - self.smooth_alpha) * self.freq + self.smooth_alpha * self.tar_freq
        self.amp = (1 - self.smooth_alpha) * self.amp + self.smooth_alpha * self.tar_amp

    def update(self, dt, target_phase=0.0, speed_factor=1.0, turn_factor=0.0, foot_contact=1.0):
        self.update_smooth()
        amp_scale = 0.65 if foot_contact > 0.5 else 1.0
        self.coupling = self.base_coupling * (1.0 + 0.3 * abs(speed_factor) + 0.5 * abs(turn_factor))
        self.coupling = np.clip(self.coupling, 0.1, 0.5)
        mu = 1.0
        x, y = self.state
        dx = 2 * np.pi * self.freq * y + self.coupling * np.sin(target_phase - self.phase)
        dy = 2 * np.pi * self.freq * (mu * (1 - x ** 2) * y - x)
        self.state += np.array([dx, dy]) * dt
        self.phase = np.arctan2(self.state[0], self.state[1])
        return self.amp * amp_scale * self.state[0]

    def reset(self):
        self.freq = self.base_freq
        self.amp = self.base_amp
        self.tar_freq = self.base_freq
        self.tar_amp = self.base_amp
        self.coupling = self.base_coupling
        self.phase = 0.0 if self.phase < np.pi else np.pi
        self.state = np.array([np.sin(self.phase), np.cos(self.phase)])


# ===================== 人形稳定控制器 =====================
class HumanoidStabilizer:
    def __init__(self, model_path, train_mode=False):
        self.train_mode = train_mode
        if not isinstance(model_path, str):
            raise TypeError("模型路径必须是字符串")

        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}")

        self.sim_duration = 9999.0
        self.dt = 0.001
        self.model.opt.timestep = self.dt
        self.model.opt.gravity[2] = -9.81
        self.model.opt.iterations = 500
        self.model.opt.tolerance = 1e-8

        self.init_wait_time = 7.0
        self._imu_euler_filt = np.zeros(3, dtype=np.float64)
        self._imu_angvel_filt = np.zeros(3, dtype=np.float64)

        self.joint_names = [
            "abdomen_z", "abdomen_y", "abdomen_x",
            "hip_x_right", "hip_z_right", "hip_y_right", "knee_right", "ankle_y_right", "ankle_x_right",
            "hip_x_left", "hip_z_left", "hip_y_left", "knee_left", "ankle_y_left", "ankle_x_left",
            "shoulder1_right", "shoulder2_right", "elbow_right",
            "shoulder1_left", "shoulder2_left", "elbow_left"
        ]
        self.joint_name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
        self.num_joints = len(self.joint_names)

        self._actuator_id_by_joint = {}
        self._actuator_gear_by_joint = {}
        self._actuator_ctrlrange_by_joint = {}
        for joint_name in self.joint_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if aid < 0:
                raise RuntimeError(f"未找到执行器：{joint_name}")
            self._actuator_id_by_joint[joint_name] = aid
            self._actuator_gear_by_joint[joint_name] = self.model.actuator_gear[aid, 0]
            self._actuator_ctrlrange_by_joint[joint_name] = self.model.actuator_ctrlrange[aid]

        # PID参数
        self.kp_roll = 420.0
        self.kd_roll = 110.0
        self.kp_pitch = 400.0
        self.kd_pitch = 100.0
        self.kp_yaw = 70.0
        self.kd_yaw = 40.0

        self.base_kp_hip = 480
        self.base_kd_hip = 95
        self.base_kp_knee = 520
        self.base_kd_knee = 105
        self.base_kp_ankle = 450
        self.base_kd_ankle = 110
        self.kp_waist = 350
        self.kd_waist = 80
        self.kp_arm = 40
        self.kd_arm = 25

        self.integral_limit = 0.12
        self.integral_yaw_limit = 0.10
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.integral_yaw = 0.0

        self.lipm_height = 0.70
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / self.lipm_height)

        self.com_target = np.array([0.05, 0.0, 0.70])
        self.kp_com = 85.0
        self.total_mass = np.sum(self.model.body_mass)
        self.weight = self.total_mass * abs(self.model.opt.gravity[2])
        self.foot_contact_threshold = max(45.0, 0.18 * self.weight)
        self._force_factor_norm = max(1.0, 0.6 * self.weight)

        self._left_foot_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot1_left"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot2_left"),
        }
        self._right_foot_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot1_right"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot2_right"),
        }

        self.joint_targets = np.zeros(self.num_joints)
        self.rl_joint_delta = np.zeros(14)
        self.foot_contact = np.zeros(2)
        self.left_foot_force = 0.0
        self.right_foot_force = 0.0

        self.gait_config = {
            "SLOW":     {"freq": 0.28, "amp": 0.20, "coupling": 0.3, "sf": 0.15, "sa": 0.05, "cz": 0.02},
            "NORMAL":   {"freq": 0.40, "amp": 0.25, "coupling": 0.25, "sf": 0.25, "sa": 0.10, "cz": 0.0},
            "TROT":     {"freq": 0.50, "amp": 0.30, "coupling": 0.25, "sf": 0.32, "sa": 0.15, "cz": -0.01},
            "STEP_IN_PLACE": {"freq": 0.35, "amp": 0.15, "coupling": 0.3, "sf": 0.0, "sa": 0.0, "cz": 0.01},
        }
        self.gait_mode = "NORMAL"
        self.g = self.gait_config[self.gait_mode]

        self.state = "STAND"
        self.state_map = {
            "STAND": self._state_stand,
            "PREPARE": self._state_prepare,
            "WALK": self._state_walk,
            "STOP": self._state_stop,
            "EMERGENCY": self._state_emergency
        }

        self.right_leg_cpg = CPGOscillator(self.g["freq"], self.g["amp"], 0.0, self.g["coupling"])
        self.left_leg_cpg = CPGOscillator(self.g["freq"], self.g["amp"], np.pi, self.g["coupling"])

        self.turn_angle = 0.0
        self.walk_speed = 0.22
        self.walk_start_time = None
        self.stop_start_time = None

        self.sac_model = None
        self.sac_model_path = "humanoid_sac_gait.zip"

        self._init_stable_pose()

    def load_sac_policy(self):
        if os.path.exists(self.sac_model_path):
            self.sac_model = SAC.load(self.sac_model_path)
            return True
        print("未找到SAC训练模型，请先执行训练！")
        return False

    def _torques_to_ctrl(self, tqs):
        ctrl = np.zeros(self.model.nu)
        for jn in self.joint_names:
            i = self.joint_name_to_idx[jn]
            aid = self._actuator_id_by_joint[jn]
            gear = self._actuator_gear_by_joint[jn]
            mn, mx = self._actuator_ctrlrange_by_joint[jn]
            mt = max(abs(mn), abs(mx)) * max(gear, 1e-9)
            tq = np.clip(tqs[i], -mt, mt)
            ctrl[aid] = np.clip(tq / gear, mn, mx)
        return ctrl

    def set_gait_mode(self, mode):
        if mode not in self.gait_config:
            mode = "NORMAL"
        self.gait_mode = mode
        self.g = self.gait_config[mode]
        self.right_leg_cpg.set_target(self.g["freq"], self.g["amp"])
        self.left_leg_cpg.set_target(self.g["freq"], self.g["amp"])
        self.com_target[2] = 0.70 + self.g["cz"]

    def _init_stable_pose(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.70
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qvel[:] = 0

        i = self.joint_name_to_idx
        self.joint_targets[:] = 0
        self.joint_targets[i["abdomen_y"]] = 0.02
        self.joint_targets[i["abdomen_x"]] = 0.0
        self.joint_targets[i["hip_y_right"]] = 0.12
        self.joint_targets[i["knee_right"]] = -0.72
        self.joint_targets[i["ankle_y_right"]] = 0.10
        self.joint_targets[i["hip_y_left"]] = 0.12
        self.joint_targets[i["knee_left"]] = -0.72
        self.joint_targets[i["ankle_y_left"]] = 0.10

    def _get_sensor_data(self):
        q = self.data.qpos[3:7]
        w, x, y, z = q
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x ** 2 + y ** 2)
        roll = np.arctan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y ** 2 + z ** 2)
        yaw = np.arctan2(siny, cosy)
        euler = np.clip([roll, pitch, yaw], -0.4, 0.4)

        lf, rf = 0.0, 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, f)
            vf = abs(f[2])
            if c.geom1 in self._left_foot_geom_ids or c.geom2 in self._left_foot_geom_ids:
                lf += vf
            if c.geom1 in self._right_foot_geom_ids or c.geom2 in self._right_foot_geom_ids:
                rf += vf

        self.left_foot_force = lf
        self.right_foot_force = rf
        lc = 1 if lf > self.foot_contact_threshold else 0
        rc = 1 if rf > self.foot_contact_threshold else 0
        return {"euler": euler, "vel": self.data.qvel[3:6], "lf": lf, "rf": rf,
                "lc": lc, "rc": rc, "com_z": self.data.subtree_com[0][2]}

    def _state_stand(self):
        self.right_leg_cpg.reset()
        self.left_leg_cpg.reset()
        self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = self.turn_angle * 0.2

    def _state_prepare(self):
        if self.walk_start_time is None:
            self.walk_start_time = self.data.time
        k = self.joint_name_to_idx["knee_right"]
        self.joint_targets[k] *= 0.98
        k = self.joint_name_to_idx["knee_left"]
        self.joint_targets[k] *= 0.98
        if self.data.time - self.walk_start_time > 0.8:
            self.set_state("WALK")

    def _state_walk(self):
        s = self.walk_speed
        g = self.g
        self.right_leg_cpg.set_target(g["freq"] + abs(s) * g["sf"], g["amp"] + abs(s) * g["sa"])
        self.left_leg_cpg.set_target(g["freq"] + abs(s) * g["sf"], g["amp"] + abs(s) * g["sa"])

        po = 0.05 * self.turn_angle
        rs, ls = (1.01, 0.99) if self.turn_angle > 0 else (0.99, 1.01) if self.turn_angle < 0 else (1, 1)

        r = self.right_leg_cpg.update(self.dt, self.left_leg_cpg.phase + po, s, self.turn_angle,
                                      self.foot_contact[0]) * rs
        l = self.left_leg_cpg.update(self.dt, self.right_leg_cpg.phase - po, s, self.turn_angle,
                                     self.foot_contact[1]) * ls

        i = self.joint_name_to_idx

        self.joint_targets[i["hip_y_right"]] = 0.10 + r
        self.joint_targets[i["knee_right"]] = -0.65 - r * 1.8
        self.joint_targets[i["ankle_y_right"]] = 0.10 + r * 0.6
        self.joint_targets[i["hip_y_left"]] = 0.10 + l
        self.joint_targets[i["knee_left"]] = -0.65 - l * 1.8
        self.joint_targets[i["ankle_y_left"]] = 0.10 + l * 0.6

        # 重心横移（基底）
        lf = self.left_foot_force
        rf = self.right_foot_force
        force_ratio = (rf - lf) / (rf + lf + 1e-6)
        hip_shift = -0.35 * force_ratio + (l - r) * 0.2
        hip_shift = np.clip(hip_shift, -0.18, 0.18)
        ankle_x_comp = -0.6 * self._imu_euler_filt[0] - 0.15 * self._imu_angvel_filt[0]
        ankle_x_comp = np.clip(ankle_x_comp, -0.12, 0.12)

        lateral_lean = 0.0
        fwd_lean = 0.03
        twist = self.turn_angle * 0.2

        arm_swing = 0.03
        self.joint_targets[i["shoulder2_right"]] = arm_swing * l
        self.joint_targets[i["shoulder2_left"]]  = -arm_swing * r
        self.joint_targets[i["shoulder1_right"]] = -0.02 * abs(r)
        self.joint_targets[i["shoulder1_left"]]  = -0.02 * abs(l)
        self.joint_targets[i["elbow_right"]]     = 0.12
        self.joint_targets[i["elbow_left"]]      = 0.12

        self.joint_targets[i["hip_x_right"]]   =  hip_shift
        self.joint_targets[i["hip_x_left"]]    = -hip_shift
        self.joint_targets[i["ankle_x_right"]] =  ankle_x_comp
        self.joint_targets[i["ankle_x_left"]]  = -ankle_x_comp
        self.joint_targets[i["abdomen_y"]]     = lateral_lean
        self.joint_targets[i["abdomen_x"]]     = fwd_lean
        self.joint_targets[i["abdomen_z"]]     = twist

        # RL增量叠加（动作范围±0.08）
        leg_joints = ["hip_x_right", "hip_z_right", "hip_y_right", "knee_right", "ankle_y_right", "ankle_x_right",
                      "hip_x_left", "hip_z_left", "hip_y_left", "knee_left", "ankle_y_left", "ankle_x_left"]
        for idx, jn in enumerate(leg_joints):
            self.joint_targets[i[jn]] += self.rl_joint_delta[idx]
        self.joint_targets[i["abdomen_y"]] += self.rl_joint_delta[12]
        self.joint_targets[i["abdomen_z"]] += self.rl_joint_delta[13]

    def _state_stop(self):
        if self.stop_start_time is None:
            self.stop_start_time = self.data.time
        self.joint_targets *= 0.93
        if self.data.time - self.stop_start_time > 0.5:
            self.set_state("STAND")
            self.stop_start_time = None

    def _state_emergency(self):
        self.data.ctrl[:] = 0

    def set_state(self, state):
        if state in self.state_map:
            self.state = state
            if state == "STAND":
                self._init_stable_pose()
            if state == "PREPARE":
                self.walk_start_time = None
            if state == "STOP":
                self.stop_start_time = None

    def set_turn_angle(self, angle):
        self.turn_angle = np.clip(angle, -0.25, 0.25)

    def set_walk_speed(self, speed):
        self.walk_speed = np.clip(speed, -0.22, 0.30)

    def _calculate_stabilizing_torques(self):
        if self.sac_model is not None and not self.train_mode:
            obs = self._get_sac_obs()
            action, _ = self.sac_model.predict(obs, deterministic=True)
            self.rl_joint_delta = action

        self.state_map[self.state]()
        sens = self._get_sensor_data()
        euler, vel, lf, rf, lc, rc = sens["euler"], sens["vel"], sens["lf"], sens["rf"], sens["lc"], sens["rc"]
        self.foot_contact = np.array([rc, lc])

        a = 0.15
        self._imu_euler_filt = (1 - a) * self._imu_euler_filt + a * euler
        self._imu_angvel_filt = (1 - a) * self._imu_angvel_filt + a * vel

        r_err = -self._imu_euler_filt[0]
        self.integral_roll = np.clip(self.integral_roll + r_err * self.dt, -self.integral_limit, self.integral_limit)
        r_tor = self.kp_roll * r_err + self.kd_roll * (-self._imu_angvel_filt[0]) + 8 * self.integral_roll

        p_err = -self._imu_euler_filt[1]
        self.integral_pitch = np.clip(self.integral_pitch + p_err * self.dt, -self.integral_limit, self.integral_limit)
        p_tor = self.kp_pitch * p_err + self.kd_pitch * (-self._imu_angvel_filt[1]) + 6 * self.integral_pitch

        y_err = -self._imu_euler_filt[2]
        self.integral_yaw = np.clip(self.integral_yaw + y_err * self.dt, -self.integral_yaw_limit,
                                    self.integral_yaw_limit)
        y_tor = self.kp_yaw * y_err + self.kd_yaw * (-self._imu_angvel_filt[2]) + 5 * self.integral_yaw

        if abs(self._imu_euler_filt[1]) > 0.7 or abs(self._imu_euler_filt[0]) > 0.7:
            self.set_state("EMERGENCY")

        tq = np.zeros(self.num_joints)
        q = self.data.qpos[7:7 + self.num_joints]
        qv = np.clip(self.data.qvel[6:6 + self.num_joints], -6, 6)

        for jn in ["abdomen_z", "abdomen_y", "abdomen_x"]:
            i = self.joint_name_to_idx[jn]
            e = np.clip(self.joint_targets[i] - q[i], -0.3, 0.3)
            base_torque = self.kp_waist * e - self.kd_waist * qv[i]
            if jn == "abdomen_x":
                extra = r_tor
            elif jn == "abdomen_y":
                extra = p_tor
            elif jn == "abdomen_z":
                extra = y_tor
            else:
                extra = 0.0
            tq[i] = base_torque + extra

        legs = ["hip_x_right", "hip_z_right", "hip_y_right", "knee_right", "ankle_y_right", "ankle_x_right",
                "hip_x_left", "hip_z_left", "hip_y_left", "knee_left", "ankle_y_left", "ankle_x_left"]
        for jn in legs:
            i = self.joint_name_to_idx[jn]
            e = np.clip(self.joint_targets[i] - q[i], -0.3, 0.3)
            ff = np.clip(rf / self._force_factor_norm, 0.5, 1.3) if "right" in jn else np.clip(
                lf / self._force_factor_norm, 0.5, 1.3)

            if "hip" in jn:
                kp, kd = self.base_kp_hip * ff, self.base_kd_hip * ff
            elif "knee" in jn:
                kp, kd = self.base_kp_knee * ff, self.base_kd_knee * ff
            elif "ankle" in jn:
                kp, kd = self.base_kp_ankle * ff, self.base_kd_ankle * ff
            else:
                kp, kd = 250, 50
            tq[i] = kp * e - kd * qv[i]

        arms = ["shoulder1_right", "shoulder2_right", "elbow_right", "shoulder1_left", "shoulder2_left", "elbow_left"]
        for jn in arms:
            i = self.joint_name_to_idx[jn]
            e = self.joint_targets[i] - q[i]
            tq[i] = self.kp_arm * e - self.kd_arm * qv[i]

        return tq

    def _get_sac_obs(self):
        sens = self._get_sensor_data()
        euler = sens["euler"]
        vel = sens["vel"]
        lf, rf = sens["lf"], sens["rf"]
        com_z = sens["com_z"]

        norm_euler = np.clip(euler / 0.5, -1.0, 1.0)
        norm_vel = np.clip(vel / 2.0, -1.0, 1.0)
        norm_lf = np.clip(lf / 200.0, 0.0, 1.0)
        norm_rf = np.clip(rf / 200.0, 0.0, 1.0)
        norm_comz = np.clip((com_z - 0.70) / 0.05, -1.0, 1.0)

        included_joints = ["hip_x_right", "hip_z_right", "hip_y_right", "knee_right", "ankle_y_right", "ankle_x_right",
                           "hip_x_left", "hip_z_left", "hip_y_left", "knee_left", "ankle_y_left", "ankle_x_left",
                           "abdomen_y", "abdomen_z"]
        q = self.data.qpos[7:7 + self.num_joints]
        joint_angles = []
        for jn in included_joints:
            idx = self.joint_name_to_idx[jn]
            if "knee" in jn:
                val = q[idx] / 1.2
            elif "abdomen" in jn:
                val = q[idx] / 0.5
            else:
                val = q[idx] / 0.8
            joint_angles.append(np.clip(val, -1.0, 1.0))

        obs = np.concatenate([norm_euler, norm_vel, [norm_lf, norm_rf, norm_comz],
                              joint_angles, self.rl_joint_delta])
        return obs.astype(np.float32)

    def simulate_stable_standing(self):
        ros = ROSCmdVelHandler(self)
        ros.start()
        kb = KeyboardInputHandler(self)
        kb.start()

        try:
            with viewer.launch_passive(self.model, self.data) as v:
                v.cam.distance = 3.2
                v.cam.azimuth = 90
                v.cam.elevation = -22
                print(" 启动成功 ")
                start = time.time()
                while time.time() - start < self.init_wait_time:
                    alpha = min(1.0, (time.time() - start) / 7)
                    t = self._calculate_stabilizing_torques() * alpha
                    self.data.ctrl[:] = self._torques_to_ctrl(t)
                    mujoco.mj_step(self.model, self.data)
                    self.data.qvel *= 0.94
                    v.sync()
                    time.sleep(self.dt)

                print("就绪！W前进 X后退 A/D转弯")
                while self.data.time < self.sim_duration:
                    t = self._calculate_stabilizing_torques()
                    self.data.ctrl[:] = self._torques_to_ctrl(t)
                    mujoco.mj_step(self.model, self.data)
                    v.sync()
                    time.sleep(self.dt)
        finally:
            kb.running = False
            ros.stop()


# ===================== 强化学习环境 =====================
class HumanoidGaitEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, model_path, target_speed=0.22, curriculum_stage=0):
        super().__init__()
        self.stabilizer = HumanoidStabilizer(model_path, train_mode=True)
        self.target_speed = target_speed
        self.curriculum_stage = curriculum_stage

        obs_dim = 3 + 3 + 2 + 1 + 14 + 14   # 37维
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.08, high=0.08, shape=(14,), dtype=np.float32)

        self.max_step = 300
        self.current_step = 0
        self.prev_action = np.zeros(14)

    def _get_obs(self):
        sens = self.stabilizer._get_sensor_data()
        euler = sens["euler"]
        vel = sens["vel"]
        lf, rf = sens["lf"], sens["rf"]
        com_z = sens["com_z"]

        norm_euler = np.clip(euler / 0.5, -1.0, 1.0)
        norm_vel = np.clip(vel / 2.0, -1.0, 1.0)
        norm_lf = np.clip(lf / 200.0, 0.0, 1.0)
        norm_rf = np.clip(rf / 200.0, 0.0, 1.0)
        norm_comz = np.clip((com_z - 0.70) / 0.05, -1.0, 1.0)

        included_joints = ["hip_x_right", "hip_z_right", "hip_y_right", "knee_right", "ankle_y_right", "ankle_x_right",
                           "hip_x_left", "hip_z_left", "hip_y_left", "knee_left", "ankle_y_left", "ankle_x_left",
                           "abdomen_y", "abdomen_z"]
        q = self.stabilizer.data.qpos[7:7 + self.stabilizer.num_joints]
        joint_angles = []
        for jn in included_joints:
            idx = self.stabilizer.joint_name_to_idx[jn]
            if "knee" in jn:
                val = q[idx] / 1.2
            elif "abdomen" in jn:
                val = q[idx] / 0.5
            else:
                val = q[idx] / 0.8
            joint_angles.append(np.clip(val, -1.0, 1.0))

        obs = np.concatenate([norm_euler, norm_vel, [norm_lf, norm_rf, norm_comz],
                              joint_angles, self.prev_action])
        return obs.astype(np.float32)

    def _domain_randomization(self, strong=False):
        friction = np.random.uniform(0.7, 1.3)
        for i in range(self.stabilizer.model.ngeom):
            self.stabilizer.model.geom_friction[i, 0] = friction
        damping = np.random.uniform(0.85, 1.15)
        self.stabilizer.model.dof_damping[:] = damping
        for i in range(self.stabilizer.model.nu):
            gear = self.stabilizer.model.actuator_gear[i, 0]
            self.stabilizer.model.actuator_gear[i, 0] = gear * np.random.uniform(0.9, 1.1)

        if strong and np.random.rand() < 0.3:
            force_mag = np.random.uniform(20, 60)
            direction = np.random.uniform(-1, 1, size=3)
            direction[2] = 0.0
            direction /= np.linalg.norm(direction) + 1e-6
            self.stabilizer.data.xfrc_applied[0, :3] = force_mag * direction

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._domain_randomization(strong=(self.curriculum_stage > 0))
        self.stabilizer._init_stable_pose()
        self.current_step = 0
        self.prev_action = np.zeros(14)
        self.stabilizer.rl_joint_delta = np.zeros(14)

        if self.curriculum_stage == 0:
            self.stabilizer.set_walk_speed(0.0)
            self.stabilizer.set_state("STAND")
        else:
            speed = self.target_speed if self.curriculum_stage == 2 else 0.15
            self.stabilizer.set_walk_speed(speed)
            self.stabilizer.set_state("PREPARE")

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, -0.08, 0.08)
        self.prev_action = action
        self.stabilizer.rl_joint_delta = action

        torques = self.stabilizer._calculate_stabilizing_torques()
        self.stabilizer.data.ctrl[:] = self.stabilizer._torques_to_ctrl(torques)
        mujoco.mj_step(self.stabilizer.model, self.stabilizer.data)

        obs = self._get_obs()
        sens = self.stabilizer._get_sensor_data()
        roll, pitch = sens["euler"][0], sens["euler"][1]
        ang_vel = sens["vel"]
        forward_vel = self.stabilizer.data.qvel[0]
        lateral_vel = self.stabilizer.data.qvel[1]
        yaw_error = sens["euler"][2]
        com_z = sens["com_z"]

        reward = 2.0
        reward -= 20.0 * (roll ** 2 + pitch ** 2)
        reward -= 3.0 * (ang_vel[0] ** 2 + ang_vel[1] ** 2)
        reward += 2.0 * np.exp(-20.0 * (com_z - 0.70) ** 2)
        reward += 0.3 * (sens["lc"] + sens["rc"])

        if self.curriculum_stage > 0:
            target_v = self.target_speed if self.curriculum_stage == 2 else 0.15
            reward += 3.0 * np.exp(-10.0 * (forward_vel - target_v) ** 2)
            reward -= 2.0 * lateral_vel ** 2
            reward -= 0.8 * abs(yaw_error)

        reward -= 0.08 * np.sum(action ** 2)

        lf = sens["lf"]
        rf = sens["rf"]
        if (lf + rf) > 1:
            force_asym = (rf - lf) / (rf + lf + 1e-6)
            reward -= 1.5 * force_asym ** 2

        terminated = False
        if abs(roll) > 0.5 or abs(pitch) > 0.5 or com_z < 0.4 or com_z > 1.0:
            reward -= 30
            terminated = True

        truncated = self.current_step >= self.max_step
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass


class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.curriculum_stage = 0
        self.target_speed = 0.22

    def advance_stage(self):
        if self.curriculum_stage < 2:
            self.curriculum_stage += 1
            self.unwrapped.curriculum_stage = self.curriculum_stage
            self.unwrapped.target_speed = self.target_speed
            print(f"升级至阶段 {self.curriculum_stage}")


# ===================== 训练入口 =====================
def train_sac():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(current_directory, "models", "humanoid.xml")

    base_env = HumanoidGaitEnv(model_file_path, curriculum_stage=0)
    curriculum_env = CurriculumWrapper(base_env)  # 保存包装器引用，用于课程升级
    env = Monitor(curriculum_env)                 # Monitor 用于记录

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        gamma=0.999,
        tau=0.005,
        buffer_size=20000,
        learning_starts=500,
        batch_size=128,
        train_freq=4,
        gradient_steps=2,
        policy_kwargs=dict(
            net_arch=[256, 128],
            activation_fn=nn.ReLU,
        ),
        target_entropy=-4.0,
        device="cpu"
    )

    clip_callback = GradientClipCallback(clip_value=1.0)

    class CurriculumCallback(BaseCallback):
        def __init__(self, env_wrapper, upgrade_interval=30000, verbose=0):
            super().__init__(verbose)
            self.env_wrapper = env_wrapper  # 保存的是 CurriculumWrapper
            self.upgrade_interval = upgrade_interval
            self.last_upgrade = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self.last_upgrade >= self.upgrade_interval:
                self.env_wrapper.advance_stage()
                self.last_upgrade = self.num_timesteps
            return True

    curriculum_cb = CurriculumCallback(curriculum_env, upgrade_interval=30000)

    print("开始快速SAC训练(10万步)")
    model.learn(
        total_timesteps=100000,
        callback=[clip_callback, curriculum_cb],
        progress_bar=True
    )
    model.save("humanoid_sac_gait")
    print("训练完成，模型已保存为 humanoid_sac_gait.zip")


if __name__ == "__main__":
    # 训练模式（默认启用，完成后可注释掉并开启仿真）
    #train_sac()

    # 仿真交互模式（取消下面注释，并注释 train_sac() 即可）
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(current_directory, "models", "humanoid.xml")
    stabilizer = HumanoidStabilizer(model_file_path)
    stabilizer.simulate_stable_standing()