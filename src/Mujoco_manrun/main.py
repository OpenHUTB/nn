import mujoco
import numpy as np
from mujoco import viewer
import time
import os


class HumanoidStabilizer:
    """适配humanoid.xml的高稳定性行走控制器（修复com_target属性错误）"""

    def __init__(self, model_path):
        # 基础初始化（保留原有核心逻辑）
        if not isinstance(model_path, str):
            raise TypeError(f"模型路径必须是字符串，当前是 {type(model_path)} 类型")

        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}\n请检查路径和文件完整性")

        # 1. 仿真参数优化：固定时间步+更保守的初始稳定
        self.sim_duration = 120.0
        self.dt = 0.005  # 固定0.005s步长，提升仿真稳定性
        self.model.opt.timestep = self.dt
        self.init_wait_time = 5.0  # 延长初始稳定时间至5s
        self.model.opt.gravity[2] = -9.81
        self.model.opt.iterations = 300  # 增加迭代次数提升接触求解精度
        self.model.opt.tolerance = 1e-9  # 降低容差，提升计算精度

        # 2. 关节名称映射（保留适配humanoid.xml的命名）
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

        # 3. PD增益优化：分阶段动态增益（核心）
        # 基础增益（支撑腿）
        self.kp_roll = 150.0  # 提高侧倾增益，增强横向稳定
        self.kd_roll = 50.0
        self.kp_pitch = 130.0  # 提高俯仰增益，抑制后仰
        self.kd_pitch = 45.0
        self.kp_yaw = 40.0
        self.kd_yaw = 20.0

        # 腿部增益（分支撑/摆动）
        self.kp_hip_support = 300.0  # 支撑腿髋关节高增益
        self.kd_hip_support = 60.0
        self.kp_hip_swing = 150.0   # 摆动腿髋关节低增益，避免过度摆动
        self.kd_hip_swing = 30.0

        self.kp_knee_support = 320.0  # 支撑腿膝关节高增益
        self.kd_knee_support = 70.0
        self.kp_knee_swing = 180.0   # 摆动腿膝关节低增益
        self.kd_knee_swing = 40.0

        self.kp_ankle_support = 250.0  # 支撑腿踝关节高增益（核心防踮脚）
        self.kd_ankle_support = 80.0
        self.kp_ankle_swing = 120.0   # 摆动腿踝关节低增益
        self.kd_ankle_swing = 40.0

        # 腰部/手臂增益（小幅提升）
        self.kp_waist = 50.0
        self.kd_waist = 25.0
        self.kp_arm = 25.0
        self.kd_arm = 25.0

        # 4. 重心控制优化：动态目标+速度反馈
        self.com_base_target = np.array([0.0, 0.0, 0.85])  # 基础重心高度提升，降低跌倒风险
        # ========== 关键修复1：初始化com_target，避免属性不存在 ==========
        self.com_target = self.com_base_target.copy()  # 初始赋值，与基础重心一致
        self.com_gait_offset = np.array([0.08, 0.0, 0.02])  # 步态相位对应的重心偏移
        self.kp_com = 80.0  # 提高重心比例增益
        self.kd_com = 30.0  # 新增重心速度阻尼
        self.foot_contact_threshold = 2.0  # 提高接触阈值，减少误判

        # 5. 状态变量优化：新增滤波和平滑参数
        self.joint_targets = np.zeros(self.num_joints)
        self.joint_targets_prev = np.zeros(self.num_joints)  # 上一帧关节目标（滤波用）
        self.prev_com = np.zeros(3)
        self.com_vel = np.zeros(3)  # 重心速度
        self.foot_contact = np.zeros(2)  # [right, left]
        self.foot_contact_history = np.zeros((2, 5))  # 接触状态历史（5帧滤波）
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.integral_limit = 0.1  # 进一步收紧积分限幅
        self.integral_deadband = 0.01  # 积分死区，小误差不积分

        # 6. 步态规划优化：平滑插值+慢周期
        self.gait_phase = 0.0
        self.gait_cycle = 2.5  # 延长步态周期至2.5s，动作更平缓
        self.step_offset_hip = 0.35  # 降低髋关节偏移，减少重心波动
        self.step_offset_knee = 0.5  # 降低膝关节偏移
        self.step_offset_ankle = 0.25
        self.walk_start_time = None
        self.filter_alpha = 0.2  # 关节目标低通滤波系数

        # 7. 力矩输出滤波：减少力矩突变
        self.torque_prev = np.zeros(self.num_joints)
        self.torque_filter_alpha = 0.15

        # 初始化稳定姿态
        self._init_stable_pose()

    def _init_stable_pose(self):
        """优化初始姿态：更贴近自然站立，降低初始冲击"""
        mujoco.mj_resetData(self.model, self.data)

        # 躯干初始位置：略高+零偏航，减少初始调整
        self.data.qpos[2] = 1.3  # 初始高度略高，落地更平稳
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

        # 腰部关节：微前倾，匹配自然站立
        self.joint_targets[self.joint_name_to_idx["abdomen_z"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["abdomen_y"]] = 0.05  # 微前倾，防止后仰
        self.joint_targets[self.joint_name_to_idx["abdomen_x"]] = 0.0

        # 右腿关节：微弯，增加缓冲
        self.joint_targets[self.joint_name_to_idx["hip_x_right"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_z_right"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_y_right"]] = 0.05
        self.joint_targets[self.joint_name_to_idx["knee_right"]] = -0.35  # 膝盖微弯，减少落地冲击
        self.joint_targets[self.joint_name_to_idx["ankle_y_right"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["ankle_x_right"]] = 0.0

        # 左腿关节：镜像右腿，保证对称
        self.joint_targets[self.joint_name_to_idx["hip_x_left"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_z_left"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["hip_y_left"]] = 0.05
        self.joint_targets[self.joint_name_to_idx["knee_left"]] = -0.35
        self.joint_targets[self.joint_name_to_idx["ankle_y_left"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["ankle_x_left"]] = 0.0

        # 手臂关节：自然下垂，减少干扰
        self.joint_targets[self.joint_name_to_idx["shoulder1_right"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["shoulder2_right"]] = 0.2
        self.joint_targets[self.joint_name_to_idx["elbow_right"]] = 1.6
        self.joint_targets[self.joint_name_to_idx["shoulder1_left"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["shoulder2_left"]] = 0.2
        self.joint_targets[self.joint_name_to_idx["elbow_left"]] = 1.6

        mujoco.mj_forward(self.model, self.data)

    def _quat_to_euler_xyz(self, quat):
        """四元数转欧拉角（保留）"""
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
        """提取躯干欧拉角（保留，增加限幅）"""
        quat = self.data.qpos[3:7].astype(np.float64).copy()
        euler = self._quat_to_euler_xyz(quat)
        euler = np.mod(euler + np.pi, 2 * np.pi) - np.pi
        return np.clip(euler, -0.2, 0.2)  # 更严格的倾角限制，防止过度倾斜

    def _detect_foot_contact(self):
        """优化接触检测：5帧历史滤波+迟滞，避免频繁切换"""
        try:
            left_foot_geoms = ["foot1_left", "foot2_left"]
            right_foot_geoms = ["foot1_right", "foot2_right"]

            # 计算当前接触力
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

            # 1. 更新接触历史（滑动窗口）
            self.foot_contact_history[0] = np.roll(self.foot_contact_history[0], -1)
            self.foot_contact_history[1] = np.roll(self.foot_contact_history[1], -1)
            self.foot_contact_history[0, -1] = 1 if right_force > self.foot_contact_threshold else 0
            self.foot_contact_history[1, -1] = 1 if left_force > self.foot_contact_threshold else 0

            # 2. 多数表决：5帧中≥3帧接触则判定为支撑
            self.foot_contact[0] = 1 if np.sum(self.foot_contact_history[0]) >= 3 else 0
            self.foot_contact[1] = 1 if np.sum(self.foot_contact_history[1]) >= 3 else 0

        except Exception as e:
            print(f"接触检测警告: {e}")
            self.foot_contact = np.ones(2)
            self.foot_contact_history = np.ones((2, 5))

    def _update_gait_phase(self):
        """步态相位更新（保留）"""
        if self.walk_start_time is None:
            self.walk_start_time = self.data.time
        self.gait_phase = (self.data.time - self.walk_start_time) % self.gait_cycle / self.gait_cycle

    def _update_walk_joint_targets(self):
        """优化步态规划：余弦插值+动态重心，动作更平滑"""
        if self.data.time < self.init_wait_time:
            # 初始稳定阶段：渐进式提升目标，避免突变
            alpha = min(1.0, (self.data.time) / self.init_wait_time)
            self.joint_targets = self.joint_targets * alpha + self.joint_targets_prev * (1 - alpha)
            # ========== 关键修复2：初始阶段保持com_target为基础值 ==========
            self.com_target = self.com_base_target.copy()
            return

        self._update_gait_phase()
        phase = self.gait_phase

        # 核心优化：使用余弦插值代替正弦，加速度更平滑（减少冲击）
        phase_rad = phase * 2 * np.pi
        cos_phase = (1 - np.cos(phase_rad)) / 2  # 0→1→0，无加速度突变

        # 动态重心目标：随步态相位前移/后移，匹配迈步节奏
        if phase < 0.5:
            # 右腿摆动：重心小幅右移+前移
            self.com_target = self.com_base_target + np.array([0.05, 0.03, 0.0])
            # 右腿关节目标：余弦插值，动作更平滑
            right_hip_yaw = 0.05 + self.step_offset_hip * cos_phase
            right_knee = -0.35 - self.step_offset_knee * cos_phase
            right_ankle = 0.0 + self.step_offset_ankle * cos_phase

            self.joint_targets[self.joint_name_to_idx["hip_y_right"]] = right_hip_yaw
            self.joint_targets[self.joint_name_to_idx["knee_right"]] = right_knee
            self.joint_targets[self.joint_name_to_idx["ankle_y_right"]] = right_ankle

            # 左腿支撑：微调整，保持平衡
            self.joint_targets[self.joint_name_to_idx["hip_y_left"]] = 0.05 - self.step_offset_hip * 0.1
            self.joint_targets[self.joint_name_to_idx["knee_left"]] = -0.35 + self.step_offset_knee * 0.05
        else:
            # 左腿摆动：重心小幅左移+前移
            self.com_target = self.com_base_target + np.array([0.05, -0.03, 0.0])
            # 左腿关节目标：余弦插值
            left_phase_rad = (phase - 0.5) * 2 * np.pi
            left_cos_phase = (1 - np.cos(left_phase_rad)) / 2
            left_hip_yaw = 0.05 + self.step_offset_hip * left_cos_phase
            left_knee = -0.35 - self.step_offset_knee * left_cos_phase
            left_ankle = 0.0 + self.step_offset_ankle * left_cos_phase

            self.joint_targets[self.joint_name_to_idx["hip_y_left"]] = left_hip_yaw
            self.joint_targets[self.joint_name_to_idx["knee_left"]] = left_knee
            self.joint_targets[self.joint_name_to_idx["ankle_y_left"]] = left_ankle

            # 右腿支撑：微调整
            self.joint_targets[self.joint_name_to_idx["hip_y_right"]] = 0.05 - self.step_offset_hip * 0.1
            self.joint_targets[self.joint_name_to_idx["knee_right"]] = -0.35 + self.step_offset_knee * 0.05

        # 关节目标低通滤波：平滑目标值，减少抖动
        self.joint_targets = self.filter_alpha * self.joint_targets + (1 - self.filter_alpha) * self.joint_targets_prev
        self.joint_targets_prev = self.joint_targets.copy()

    def _calculate_stabilizing_torques(self):
        """优化力矩计算：动态增益+积分分离+速度反馈"""
        self._update_walk_joint_targets()
        torques = np.zeros(self.num_joints, dtype=np.float64)

        # 1. 躯干姿态控制：积分分离（核心防积分饱和）
        root_euler = self._get_root_euler()
        root_vel = self.data.qvel[3:6].astype(np.float64).copy()
        root_vel = np.clip(root_vel, -2.0, 2.0)  # 更严格的速度限制

        # 积分分离：误差超过死区时不积分
        roll_error = -root_euler[0]
        if abs(roll_error) > self.integral_deadband:
            self.integral_roll = 0.0
        else:
            self.integral_roll += roll_error * self.dt
        self.integral_roll = np.clip(self.integral_roll, -self.integral_limit, self.integral_limit)
        roll_torque = self.kp_roll * roll_error + self.kd_roll * (-root_vel[0]) + 3.0 * self.integral_roll

        pitch_error = -root_euler[1]
        if abs(pitch_error) > self.integral_deadband:
            self.integral_pitch = 0.0
        else:
            self.integral_pitch += pitch_error * self.dt
        self.integral_pitch = np.clip(self.integral_pitch, -self.integral_limit, self.integral_limit)
        pitch_torque = self.kp_pitch * pitch_error + self.kd_pitch * (-root_vel[1]) + 2.0 * self.integral_pitch

        yaw_error = -root_euler[2]
        yaw_torque = self.kp_yaw * yaw_error + self.kd_yaw * (-root_vel[2])

        torso_torque = np.array([roll_torque, pitch_torque, yaw_torque])
        torso_torque = np.clip(torso_torque, -25.0, 25.0)  # 降低躯干力矩上限，减少抖动

        # 2. 重心补偿：新增速度反馈，抑制重心波动
        com = self.data.subtree_com[0].astype(np.float64).copy()
        self.com_vel = (com - self.prev_com) / self.dt  # 计算重心速度
        self.com_vel = np.clip(self.com_vel, -0.5, 0.5)  # 速度限幅
        com_error = self.com_target - com
        com_error = np.clip(com_error, -0.05, 0.05)
        # 比例+微分补偿，减少重心震荡
        com_compensation = self.kp_com * com_error + self.kd_com * (-self.com_vel)

        # 3. 关节控制：动态增益（支撑/摆动腿区分）
        self._detect_foot_contact()
        current_joints = self.data.qpos[7:7 + self.num_joints].astype(np.float64)
        current_vel = self.data.qvel[6:6 + self.num_joints].astype(np.float64)
        current_vel = np.clip(current_vel, -5.0, 5.0)  # 降低关节速度上限

        # 腰部关节控制（小幅调整）
        waist_joints = ["abdomen_z", "abdomen_y", "abdomen_x"]
        for joint_name in waist_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            joint_error = np.clip(joint_error, -0.2, 0.2)
            torques[idx] = self.kp_waist * joint_error - self.kd_waist * current_vel[idx]

        # 腿部关节控制：动态增益（核心优化）
        leg_joints = [
            "hip_x_right", "hip_z_right", "hip_y_right",
            "knee_right", "ankle_y_right", "ankle_x_right",
            "hip_x_left", "hip_z_left", "hip_y_left",
            "knee_left", "ankle_y_left", "ankle_x_left"
        ]

        for joint_name in leg_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            joint_error = np.clip(joint_error, -0.2, 0.2)

            # 判断是支撑腿还是摆动腿，分配不同增益
            is_support = False
            if "right" in joint_name and self.foot_contact[0] == 1:
                is_support = True
            elif "left" in joint_name and self.foot_contact[1] == 1:
                is_support = True

            # 动态分配增益
            if "hip" in joint_name:
                kp = self.kp_hip_support if is_support else self.kp_hip_swing
                kd = self.kd_hip_support if is_support else self.kd_hip_swing
                if "y" in joint_name:
                    joint_error += torso_torque[1] * 0.01  # 降低姿态补偿，减少后仰
            elif "knee" in joint_name:
                kp = self.kp_knee_support if is_support else self.kp_knee_swing
                kd = self.kd_knee_support if is_support else self.kd_knee_swing
                joint_error += com_compensation[2] * 0.03  # 降低重心补偿，减少抖动
            elif "ankle" in joint_name:
                kp = self.kp_ankle_support if is_support else self.kp_ankle_swing
                kd = self.kd_ankle_support if is_support else self.kd_ankle_swing
                if "y" in joint_name:
                    joint_error += torso_torque[1] * 0.005  # 极小的姿态补偿，防踮脚

            torques[idx] = kp * joint_error - kd * current_vel[idx]

        # 手臂关节控制（保留，小幅调整）
        arm_joints = [
            "shoulder1_right", "shoulder2_right", "elbow_right",
            "shoulder1_left", "shoulder2_left", "elbow_left"
        ]
        for joint_name in arm_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            torques[idx] = self.kp_arm * joint_error - self.kd_arm * current_vel[idx]

        # 力矩限幅：根据支撑/摆动调整，支撑腿更高
        torque_limits = {
            "abdomen_z": 60, "abdomen_y": 60, "abdomen_x": 60,
            "hip_x_right": 180 if self.foot_contact[0] == 1 else 100,
            "hip_z_right": 180 if self.foot_contact[0] == 1 else 100,
            "hip_y_right": 180 if self.foot_contact[0] == 1 else 100,
            "knee_right": 220 if self.foot_contact[0] == 1 else 120,
            "ankle_y_right": 150 if self.foot_contact[0] == 1 else 80,
            "ankle_x_right": 120 if self.foot_contact[0] == 1 else 70,
            "hip_x_left": 180 if self.foot_contact[1] == 1 else 100,
            "hip_z_left": 180 if self.foot_contact[1] == 1 else 100,
            "hip_y_left": 180 if self.foot_contact[1] == 1 else 100,
            "knee_left": 220 if self.foot_contact[1] == 1 else 120,
            "ankle_y_left": 150 if self.foot_contact[1] == 1 else 80,
            "ankle_x_left": 120 if self.foot_contact[1] == 1 else 70,
            "shoulder1_right": 25, "shoulder2_right": 25, "elbow_right": 25,
            "shoulder1_left": 25, "shoulder2_left": 25, "elbow_left": 25
        }
        for joint_name, limit in torque_limits.items():
            idx = self.joint_name_to_idx[joint_name]
            torques[idx] = np.clip(torques[idx], -limit, limit)

        # 力矩输出低通滤波：减少力矩突变，提升平滑性
        torques = self.torque_filter_alpha * torques + (1 - self.torque_filter_alpha) * self.torque_prev
        self.torque_prev = torques.copy()

        # 调试输出（保留）
        if self.data.time % 2 < 0.1 and self.data.time > self.init_wait_time:
            print(f"=== 行走调试 ===")
            print(f"步态相位: {self.gait_phase:.2f} | 重心: {com[0]:.3f}/{com[2]:.3f}m")
            print(f"右脚接触: {self.foot_contact[0]}, 左脚接触: {self.foot_contact[1]}")
            print(f"躯干倾角: roll={root_euler[0]:.3f}, pitch={root_euler[1]:.3f}")

        self.prev_com = com
        return torques

    def simulate_stable_standing(self):
        """优化仿真循环：渐进式启动+更严格的跌倒判定"""
        with viewer.launch_passive(self.model, self.data) as v:
            # 优化相机视角：更清晰观察行走
            v.cam.distance = 4.0
            v.cam.azimuth = 75  # 调整视角，避免遮挡
            v.cam.elevation = -20
            v.cam.lookat = [0, 0, 0.8]

            print("人形机器人高稳定性行走仿真启动...")
            print(f"初始稳定{self.init_wait_time}秒后自动开始行走")

            # 初始稳定阶段：渐进式提升控制力度，避免冲击
            start_time = time.time()
            while time.time() - start_time < self.init_wait_time:
                alpha = min(1.0, (time.time() - start_time) / self.init_wait_time)
                torques = self._calculate_stabilizing_torques() * alpha
                self.data.ctrl[:] = torques
                mujoco.mj_step(self.model, self.data)
                self.data.qvel[:] *= 0.98  # 稍弱的速度衰减，避免过度制动
                v.sync()
                time.sleep(self.dt * 0.8)  # 稍快的仿真，提升流畅度

            # 主仿真循环：更严格的跌倒判定
            print("=== 开始稳定行走 ===")
            while self.data.time < self.sim_duration:
                torques = self._calculate_stabilizing_torques()
                self.data.ctrl[:] = torques
                mujoco.mj_step(self.model, self.data)

                # 状态监测：增加重心速度输出
                if self.data.time % 2 < 0.1:
                    com = self.data.subtree_com[0]
                    euler = self._get_root_euler()
                    print(
                        f"时间:{self.data.time:.1f}s | 重心(x/z/vel):{com[0]:.3f}/{com[2]:.3f}/{self.com_vel[0]:.3f}m/s | "
                        f"姿态(roll/pitch):{euler[0]:.3f}/{euler[1]:.3f}rad | 脚接触:{self.foot_contact}"
                    )

                v.sync()
                time.sleep(self.dt * 0.6)

                # 优化跌倒判定：更严格的阈值，提前预警
                com = self.data.subtree_com[0]
                euler = self._get_root_euler()
                if com[2] < 0.6 or abs(euler[0]) > 0.4 or abs(euler[1]) > 0.4:
                    print(
                        f"⚠️  跌倒预警/跌倒！时间:{self.data.time:.1f}s | 重心(z):{com[2]:.3f}m | "
                        f"最大倾角:{max(abs(euler[0]), abs(euler[1])):.3f}rad"
                    )
                    break

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