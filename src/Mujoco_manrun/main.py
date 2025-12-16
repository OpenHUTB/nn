import mujoco
import numpy as np
from mujoco import viewer
import time
import os


class G1Stabilizer:
    """修复踮脚后仰的G1机器人稳定站立控制器（新增行走功能）"""

    def __init__(self, model_path):
        # 类型检查与模型加载
        if not isinstance(model_path, str):
            raise TypeError(f"模型路径必须是字符串，当前是 {type(model_path)} 类型")

        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}\n请检查路径和文件完整性")

        # 仿真核心参数（强化稳定性）
        self.sim_duration = 120.0
        self.dt = 0.001
        self.model.opt.timestep = self.dt
        self.init_wait_time = 4.0  # 延长初始稳定时间
        self.model.opt.gravity[2] = -9.81
        self.model.opt.iterations = 200
        self.model.opt.tolerance = 1e-8

        # 关节名称映射
        self.joint_names = [
            # 左腿关节
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            # 右腿关节
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            # 腰部关节
            "waist_yaw_joint",
            # 左臂关节
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
            # 右臂关节
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"
        ]

        self.joint_name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
        self.num_joints = len(self.joint_names)

        # 关键修复：PD增益（抑制踮脚/后仰）
        self.kp_roll = 120.0  # 降低侧倾增益，避免过度修正
        self.kd_roll = 40.0  # 提高侧倾阻尼
        self.kp_pitch = 100.0  # 大幅降低俯仰增益（核心：防止后仰）
        self.kd_pitch = 35.0  # 提高俯仰阻尼
        self.kp_yaw = 30.0
        self.kd_yaw = 15.0

        # 腿部关节增益（重点优化踝关节）
        self.kp_hip = 250.0  # 【增大】髋关节增益，确保有足够力矩迈步
        self.kd_hip = 45.0  # 提高髋关节阻尼
        self.kp_knee = 280.0  # 【增大】膝关节增益
        self.kd_knee = 50.0
        self.kp_ankle = 200.0  # 提高踝关节比例增益（防止踮脚）
        self.kd_ankle = 60.0  # 大幅提高踝关节阻尼（核心：抑制踮脚）

        # 腰部/手臂增益（几乎固定）
        self.kp_waist = 40.0
        self.kd_waist = 20.0
        self.kp_arm = 20.0
        self.kd_arm = 20.0

        # 重心目标（核心修复：前移+降低重心）
        self.com_target = np.array([0.05, 0.0, 0.78])  # 重心前移5cm，高度降至0.78m
        self.kp_com = 50.0  # 降低重心补偿，避免过度修正
        self.foot_contact_threshold = 1.5  # 降低接触阈值，提高检测灵敏度

        # 状态变量
        self.joint_targets = np.zeros(self.num_joints)
        self.prev_com = np.zeros(3)
        self.foot_contact = np.zeros(2)
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.integral_limit = 0.15  # 进一步收紧积分限幅

        # 【行走功能新增】步态参数（核心：增大偏移量确保动作明显）
        self.gait_phase = 0.0  # 步态相位（0-1循环）
        self.gait_cycle = 2.0  # 步态周期（秒，放慢周期让动作更明显）
        self.step_offset_hip = 0.8  # 髋关节俯仰偏移量（【大幅增大】原0.1→0.8）
        self.step_offset_knee = 1.2  # 膝关节偏移量（【大幅增大】原0.4→1.2）
        self.step_offset_ankle = 0.5  # 踝关节偏移量
        self.walk_start_time = None  # 行走开始时间标记

        # 初始化稳定姿态（核心修复：关节角度）
        self._init_stable_pose()

    def _init_stable_pose(self):
        """修复：调整关节角度，让脚掌完全落地，重心前移"""
        mujoco.mj_resetData(self.model, self.data)

        # 躯干初始位置（更低+更稳）
        self.data.qpos[2] = 0.78  # 初始高度匹配重心目标
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

        # 核心修复：腿部关节角度（避免踮脚/后仰）
        # 左腿关节（脚掌完全落地，无背屈）
        self.joint_targets[self.joint_name_to_idx["left_hip_pitch_joint"]] = 0.1  # 降低髋俯仰，避免重心后移
        self.joint_targets[self.joint_name_to_idx["left_hip_roll_joint"]] = 0.02
        self.joint_targets[self.joint_name_to_idx["left_hip_yaw_joint"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["left_knee_joint"]] = -0.4  # 膝盖稍弯，增加缓冲
        self.joint_targets[self.joint_name_to_idx["left_ankle_pitch_joint"]] = 0.0  # 踝关节0度，脚掌完全落地（核心）
        self.joint_targets[self.joint_name_to_idx["left_ankle_roll_joint"]] = 0.0

        # 右腿关节（镜像左腿）
        self.joint_targets[self.joint_name_to_idx["right_hip_pitch_joint"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["right_hip_roll_joint"]] = -0.02
        self.joint_targets[self.joint_name_to_idx["right_hip_yaw_joint"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["right_knee_joint"]] = -0.4
        self.joint_targets[self.joint_name_to_idx["right_ankle_pitch_joint"]] = 0.0  # 踝关节0度（核心）
        self.joint_targets[self.joint_name_to_idx["right_ankle_roll_joint"]] = 0.0

        # 腰部/手臂（固定）
        self.joint_targets[self.joint_name_to_idx["waist_yaw_joint"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["left_shoulder_pitch_joint"]] = 0.6
        self.joint_targets[self.joint_name_to_idx["left_elbow_joint"]] = 1.8
        self.joint_targets[self.joint_name_to_idx["right_shoulder_pitch_joint"]] = 0.6
        self.joint_targets[self.joint_name_to_idx["right_elbow_joint"]] = 1.8

        mujoco.mj_forward(self.model, self.data)

    def _quat_to_euler_xyz(self, quat):
        """纯Numpy四元数转欧拉角"""
        w, x, y, z = quat
        # Roll (X)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch (Y)
        sinp = 2 * (w * y - z * x)
        pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(sinp))
        # Yaw (Z)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    def _get_root_euler(self):
        """提取躯干欧拉角"""
        quat = self.data.qpos[3:7].astype(np.float64).copy()
        euler = self._quat_to_euler_xyz(quat)
        euler = np.mod(euler + np.pi, 2 * np.pi) - np.pi
        return np.clip(euler, -0.3, 0.3)  # 进一步限制最大倾角

    def _detect_foot_contact(self):
        """优化接触检测，确保脚掌落地判定准确"""
        try:
            left_foot_geoms = ["left_foot_1_col", "left_foot_2_col", "left_foot_3_col", "left_foot_4_col"]
            right_foot_geoms = ["right_foot_1_col", "right_foot_2_col", "right_foot_3_col", "right_foot_4_col"]

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

            # 迟滞优化：防止接触状态频繁切换
            self.foot_contact[1] = 1 if left_force > self.foot_contact_threshold else 0
            self.foot_contact[0] = 1 if right_force > self.foot_contact_threshold else 0

        except Exception as e:
            print(f"接触检测警告: {e}")
            self.foot_contact = np.ones(2)

    # 【行走功能新增】更新步态相位
    def _update_gait_phase(self):
        """更新步态相位，确保初始稳定后触发行走"""
        if self.walk_start_time is None:
            self.walk_start_time = self.data.time  # 标记行走开始时间
        # 计算归一化步态相位（0-1循环）
        self.gait_phase = (self.data.time - self.walk_start_time) % self.gait_cycle / self.gait_cycle

    # 【行走功能新增】生成行走的关节目标角度
    def _update_walk_joint_targets(self):
        """基于步态相位更新关节目标角度（核心行走逻辑）"""
        # 仅在初始稳定期结束后启动行走
        if self.data.time < self.init_wait_time:
            return  # 初始稳定期不修改目标角度

        self._update_gait_phase()
        phase = self.gait_phase  # 0-1

        # ========== 左腿/右腿交替摆动逻辑 ==========
        if phase < 0.5:
            # 0-0.5周期：右腿摆动（向前迈），左腿支撑
            # 右腿髋关节：向前摆（正弦曲线，0→最大值→0）
            right_hip_pitch = 0.1 + self.step_offset_hip * np.sin(phase * 2 * np.pi)
            # 右腿膝关节：弯曲抬腿（正弦曲线，0→最小值→0）
            right_knee = -0.4 - self.step_offset_knee * np.sin(phase * 2 * np.pi)
            # 右腿踝关节：协调（随膝关节联动）
            right_ankle = 0.0 + self.step_offset_ankle * np.sin(phase * 2 * np.pi)

            # 更新右腿目标角度
            self.joint_targets[self.joint_name_to_idx["right_hip_pitch_joint"]] = right_hip_pitch
            self.joint_targets[self.joint_name_to_idx["right_knee_joint"]] = right_knee
            self.joint_targets[self.joint_name_to_idx["right_ankle_pitch_joint"]] = right_ankle

            # 左腿保持支撑姿态（小幅调整平衡）
            self.joint_targets[self.joint_name_to_idx["left_hip_pitch_joint"]] = 0.1 - self.step_offset_hip * 0.2
            self.joint_targets[self.joint_name_to_idx["left_knee_joint"]] = -0.4 + self.step_offset_knee * 0.1
        else:
            # 0.5-1.0周期：左腿摆动（向前迈），右腿支撑
            # 左腿髋关节：向前摆
            left_hip_pitch = 0.1 + self.step_offset_hip * np.sin((phase - 0.5) * 2 * np.pi)
            # 左腿膝关节：弯曲抬腿
            left_knee = -0.4 - self.step_offset_knee * np.sin((phase - 0.5) * 2 * np.pi)
            # 左腿踝关节：协调
            left_ankle = 0.0 + self.step_offset_ankle * np.sin((phase - 0.5) * 2 * np.pi)

            # 更新左腿目标角度
            self.joint_targets[self.joint_name_to_idx["left_hip_pitch_joint"]] = left_hip_pitch
            self.joint_targets[self.joint_name_to_idx["left_knee_joint"]] = left_knee
            self.joint_targets[self.joint_name_to_idx["left_ankle_pitch_joint"]] = left_ankle

            # 右腿保持支撑姿态
            self.joint_targets[self.joint_name_to_idx["right_hip_pitch_joint"]] = 0.1 - self.step_offset_hip * 0.2
            self.joint_targets[self.joint_name_to_idx["right_knee_joint"]] = -0.4 + self.step_offset_knee * 0.1

    def _calculate_stabilizing_torques(self):
        """优化力矩计算，抑制踮脚/后仰 + 集成行走控制"""
        # 【新增】先更新行走的关节目标角度
        self._update_walk_joint_targets()

        torques = np.zeros(self.num_joints, dtype=np.float64)

        # 1. 躯干姿态控制（更保守）
        root_euler = self._get_root_euler()
        root_vel = self.data.qvel[3:6].astype(np.float64).copy()
        root_vel = np.clip(root_vel, -3.0, 3.0)  # 严格限制躯干角速度

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
        torso_torque = np.clip(torso_torque, -30.0, 30.0)  # 降低躯干修正力矩上限

        # 2. 重心补偿（更柔和）
        com = self.data.subtree_com[0].astype(np.float64).copy()
        com_error = self.com_target - com
        com_error = np.clip(com_error, -0.03, 0.03)
        com_compensation = self.kp_com * com_error

        # 3. 关节控制（重点优化踝关节）
        self._detect_foot_contact()
        current_joints = self.data.qpos[7:7 + self.num_joints].astype(np.float64)
        current_vel = self.data.qvel[6:6 + self.num_joints].astype(np.float64)
        current_vel = np.clip(current_vel, -8.0, 8.0)  # 限制关节速度

        # 腿部关节控制
        leg_joints = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
        ]

        for joint_name in leg_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            joint_error = np.clip(joint_error, -0.3, 0.3)  # 【放宽】关节误差限制，允许更大动作

            # 关节增益匹配
            if "hip" in joint_name:
                kp = self.kp_hip
                kd = self.kd_hip
                # 核心：降低髋关节俯仰的姿态补偿，避免后仰
                if "pitch" in joint_name:
                    if "left" in joint_name:
                        joint_error += torso_torque[1] * 0.02  # 大幅降低补偿系数
                    else:
                        joint_error += torso_torque[1] * 0.02

            elif "knee" in joint_name:
                kp = self.kp_knee
                kd = self.kd_knee
                joint_error += com_compensation[2] * 0.05

            elif "ankle" in joint_name:
                kp = self.kp_ankle
                kd = self.kd_ankle
                # 核心：踝关节仅保留极小的姿态补偿，防止踮脚
                if "pitch" in joint_name:
                    joint_error += torso_torque[1] * 0.01  # 几乎无补偿

            # 非支撑腿增益调整（【修改】摆动腿增益仅小幅降低，确保有力气摆动）
            if ("left" in joint_name and self.foot_contact[1] == 0) or \
                    ("right" in joint_name and self.foot_contact[0] == 0):
                kp *= 0.8  # 原0.1→0.8，大幅提高摆动腿增益
                kd *= 0.9

            torques[idx] = kp * joint_error - kd * current_vel[idx]

        # 腰部/手臂控制（几乎无动作）
        waist_joint = "waist_yaw_joint"
        idx = self.joint_name_to_idx[waist_joint]
        joint_error = self.joint_targets[idx] - current_joints[idx]
        torques[idx] = self.kp_waist * joint_error - self.kd_waist * current_vel[idx]

        arm_joints = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint"
        ]
        for joint_name in arm_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            torques[idx] = self.kp_arm * joint_error - self.kd_arm * current_vel[idx]

        # 力矩限幅（【放宽】腿部力矩限制，确保能驱动行走）
        torque_limits = {
            "left_hip_pitch_joint": 150, "left_hip_roll_joint": 150, "left_hip_yaw_joint": 150,  # 原70→150
            "left_knee_joint": 200, "left_ankle_pitch_joint": 120, "left_ankle_roll_joint": 100,  # 原100→200
            "right_hip_pitch_joint": 150, "right_hip_roll_joint": 150, "right_hip_yaw_joint": 150,
            "right_knee_joint": 200, "right_ankle_pitch_joint": 120, "right_ankle_roll_joint": 100,
            "waist_yaw_joint": 50,
            "left_shoulder_pitch_joint": 10, "left_shoulder_roll_joint": 10,
            "left_shoulder_yaw_joint": 10, "left_elbow_joint": 10, "left_wrist_roll_joint": 10,
            "right_shoulder_pitch_joint": 10, "right_shoulder_roll_joint": 10,
            "right_shoulder_yaw_joint": 10, "right_elbow_joint": 10, "right_wrist_roll_joint": 10
        }
        for joint_name, limit in torque_limits.items():
            idx = self.joint_name_to_idx[joint_name]
            torques[idx] = np.clip(torques[idx], -limit, limit)

        # 【新增】调试输出：查看关键参数（确认行走逻辑在运行）
        if self.data.time % 1 < 0.1 and self.data.time > self.init_wait_time:
            print(f"=== 行走调试 ===")
            print(f"步态相位: {self.gait_phase:.2f}")
            print(f"右腿髋俯仰目标: {self.joint_targets[self.joint_name_to_idx['right_hip_pitch_joint']]:.2f}")
            print(f"左腿髋俯仰目标: {self.joint_targets[self.joint_name_to_idx['left_hip_pitch_joint']]:.2f}")
            print(f"右脚接触: {self.foot_contact[0]}, 左脚接触: {self.foot_contact[1]}")

        self.prev_com = com
        return torques

    def simulate_stable_standing(self):
        """运行稳定站立+行走仿真（原方法保留，新增行走逻辑）"""
        with viewer.launch_passive(self.model, self.data) as v:
            # 优化相机视角
            v.cam.distance = 3.0  # 拉远相机，方便看行走
            v.cam.azimuth = 90
            v.cam.elevation = -25
            v.cam.lookat = [0, 0, 0.6]

            print("G1机器人稳定站立+行走仿真启动...")
            print(f"初始稳定{self.init_wait_time}秒后自动开始行走")

            # 初始落地阶段（更强阻尼）
            start_time = time.time()
            while time.time() - start_time < self.init_wait_time:
                alpha = min(1.0, (time.time() - start_time) / self.init_wait_time)
                torques = self._calculate_stabilizing_torques() * alpha
                self.data.ctrl[:] = torques
                mujoco.mj_step(self.model, self.data)
                self.data.qvel[:] *= 0.97  # 更强的速度衰减，防止初始后仰
                v.sync()
                time.sleep(self.dt)

            # 主仿真循环（行走阶段）
            print("=== 开始行走 ===")
            while self.data.time < self.sim_duration:
                torques = self._calculate_stabilizing_torques()
                self.data.ctrl[:] = torques
                mujoco.mj_step(self.model, self.data)

                # 状态监测
                if self.data.time % 2 < 0.1:
                    com = self.data.subtree_com[0]
                    euler = self._get_root_euler()
                    print(
                        f"时间:{self.data.time:.1f}s | 重心(x/z):{com[0]:.3f}/{com[2]:.3f}m | "
                        f"姿态(roll/pitch):{euler[0]:.3f}/{euler[1]:.3f}rad | 脚接触:{self.foot_contact}"
                    )

                v.sync()
                time.sleep(self.dt * 0.5)

                # 跌倒判定
                com = self.data.subtree_com[0]
                euler = self._get_root_euler()
                if com[2] < 0.4 or abs(euler[0]) > 0.6 or abs(euler[1]) > 0.6:
                    print(
                        f"跌倒！时间:{self.data.time:.1f}s | 重心(z):{com[2]:.3f}m | "
                        f"最大倾角:{max(abs(euler[0]), abs(euler[1])):.3f}rad"
                    )
                    break

        print("仿真完成！")


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(current_directory, "g1_23dof.xml")

    print(f"模型路径：{model_file_path}")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"模型文件不存在：{model_file_path}")

    try:
        stabilizer = G1Stabilizer(model_file_path)
        stabilizer.simulate_stable_standing()  # 原方法不变，内部已集成行走
    except Exception as e:
        print(f"错误：{e}")
        import traceback

        traceback.print_exc()