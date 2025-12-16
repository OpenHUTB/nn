import mujoco
import numpy as np
from mujoco import viewer
import time
import os


class G1Stabilizer:

    def __init__(self, model_path):
        # 类型检查
        if not isinstance(model_path, str):
            raise TypeError(f"模型路径必须是字符串，当前是 {type(model_path)} 类型")

        # 模型加载
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}\n请检查：1.路径是否为字符串 2.文件是否存在 3.文件是否完整")

        # 仿真参数
        self.sim_duration = 120.0
        self.dt = self.model.opt.timestep
        self.init_wait_time = 2.0

        # 关节名称映射表，与g1_23dof.xml对应
        self.joint_names = [
            # 左腿关节
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            # 右腿关节
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            # 腰部关节
            "waist_yaw_joint",
            # 左臂关节
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            # 右臂关节
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint"
        ]

        # 创建关节名称到索引的映射
        self.joint_name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
        self.num_joints = len(self.joint_names)

        # 站立控制参数（针对G1模型优化）
        self.kp_roll = 200.0
        self.kd_roll = 28.0
        self.kp_pitch = 180.0
        self.kd_pitch = 22.0
        self.kp_yaw = 60.0
        self.kd_yaw = 12.0

        # 腿部关节增益（针对G1的关节结构优化）
        self.kp_hip = 300.0
        self.kd_hip = 35.0
        self.kp_knee = 350.0
        self.kd_knee = 40.0
        self.kp_ankle = 250.0
        self.kd_ankle = 30.0

        # 腰部和手臂关节增益
        self.kp_waist = 100.0
        self.kd_waist = 15.0
        self.kp_arm = 80.0
        self.kd_arm = 10.0

        # 重心补偿参数
        self.com_target = np.array([0.0, 0.0, 0.85])  # G1的目标重心位置
        self.kp_com = 90.0
        self.foot_contact_threshold = 3.0  # 适应G1的足部结构

        # 状态变量
        self.joint_targets = np.zeros(self.num_joints)  # 所有关节的目标角度
        self.prev_com = np.zeros(3)
        self.foot_contact = np.zeros(2)  # [右脚, 左脚]
        self.integral_roll = 0.0
        self.integral_pitch = 0.0

        # 初始化稳定姿态
        self._init_stable_pose()

    def _init_stable_pose(self):
        """初始化G1机器人的稳定站立姿态"""
        mujoco.mj_resetData(self.model, self.data)

        # 设置初始位置（根据G1模型调整）
        self.data.qpos[2] = 0.85  # 躯干初始高度
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 躯干直立
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

        # 设置各关节目标角度（针对G1的23自由度优化）
        # 左腿关节目标角度
        self.joint_targets[self.joint_name_to_idx["left_hip_pitch_joint"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["left_hip_roll_joint"]] = 0.05
        self.joint_targets[self.joint_name_to_idx["left_hip_yaw_joint"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["left_knee_joint"]] = -0.4
        self.joint_targets[self.joint_name_to_idx["left_ankle_pitch_joint"]] = 0.2
        self.joint_targets[self.joint_name_to_idx["left_ankle_roll_joint"]] = 0.0

        # 右腿关节目标角度
        self.joint_targets[self.joint_name_to_idx["right_hip_pitch_joint"]] = 0.1
        self.joint_targets[self.joint_name_to_idx["right_hip_roll_joint"]] = -0.05
        self.joint_targets[self.joint_name_to_idx["right_hip_yaw_joint"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["right_knee_joint"]] = -0.4
        self.joint_targets[self.joint_name_to_idx["right_ankle_pitch_joint"]] = 0.2
        self.joint_targets[self.joint_name_to_idx["right_ankle_roll_joint"]] = 0.0

        # 腰部关节
        self.joint_targets[self.joint_name_to_idx["waist_yaw_joint"]] = 0.0

        # 左臂关节（自然下垂）
        self.joint_targets[self.joint_name_to_idx["left_shoulder_pitch_joint"]] = 0.5
        self.joint_targets[self.joint_name_to_idx["left_shoulder_roll_joint"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["left_shoulder_yaw_joint"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["left_elbow_joint"]] = 1.5
        self.joint_targets[self.joint_name_to_idx["left_wrist_roll_joint"]] = 0.0

        # 右臂关节（自然下垂）
        self.joint_targets[self.joint_name_to_idx["right_shoulder_pitch_joint"]] = 0.5
        self.joint_targets[self.joint_name_to_idx["right_shoulder_roll_joint"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["right_shoulder_yaw_joint"]] = 0.0
        self.joint_targets[self.joint_name_to_idx["right_elbow_joint"]] = 1.5
        self.joint_targets[self.joint_name_to_idx["right_wrist_roll_joint"]] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def _get_root_euler(self):
        """提取躯干欧拉角（roll, pitch, yaw）"""
        rot_mat = np.zeros(9, dtype=np.float64)
        quat = self.data.qpos[3:7].astype(np.float64).copy()
        mujoco.mju_quat2Mat(rot_mat, quat)

        euler = np.zeros(3, dtype=np.float64)
        mujoco.mju_mat2Euler(euler, rot_mat, 1)  # XYZ顺序

        # 角度限幅（-π~π）
        euler = np.mod(euler + np.pi, 2 * np.pi) - np.pi
        return euler

    def _detect_foot_contact(self):
        """检测左右脚与地面的接触力（适配G1的足部结构）"""
        try:
            # 检测左脚接触（使用g1_23dof.xml中定义的四个足部碰撞体）
            left_foot_geoms = [
                "left_foot_1_col",
                "left_foot_2_col",
                "left_foot_3_col",
                "left_foot_4_col"
            ]

            # 检测右脚接触
            right_foot_geoms = [
                "right_foot_1_col",
                "right_foot_2_col",
                "right_foot_3_col",
                "right_foot_4_col"
            ]

            # 计算左脚总接触力
            left_force = 0.0
            for geom_name in left_foot_geoms:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, geom_id, force)
                left_force += force[2]  # z轴分量

            # 计算右脚总接触力
            right_force = 0.0
            for geom_name in right_foot_geoms:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, geom_id, force)
                right_force += force[2]  # z轴分量

            # 更新接触状态
            self.foot_contact[0] = 1 if right_force > self.foot_contact_threshold else 0  # 右脚
            self.foot_contact[1] = 1 if left_force > self.foot_contact_threshold else 0  # 左脚

        except Exception as e:
            print(f"接触检测警告: {e}")
            self.foot_contact = np.ones(2)  # 出错时默认双脚接触

    def _calculate_stabilizing_torques(self):
        """计算稳定站立所需的关节力矩"""
        torques = np.zeros(self.num_joints, dtype=np.float64)

        # 1. 躯干姿态控制
        root_euler = self._get_root_euler()
        root_vel = self.data.qvel[3:6].astype(np.float64).copy()
        root_vel = np.clip(root_vel, -10.0, 10.0)

        # 侧倾控制（带积分补偿）
        roll_error = -root_euler[0]
        self.integral_roll += roll_error * self.dt
        self.integral_roll = np.clip(self.integral_roll, -0.5, 0.5)
        roll_torque = self.kp_roll * roll_error + self.kd_roll * (-root_vel[0]) + 12.0 * self.integral_roll

        # 俯仰控制（带积分补偿）
        pitch_error = -root_euler[1]
        self.integral_pitch += pitch_error * self.dt
        self.integral_pitch = np.clip(self.integral_pitch, -0.5, 0.5)
        pitch_torque = self.kp_pitch * pitch_error + self.kd_pitch * (-root_vel[1]) + 10.0 * self.integral_pitch

        # 偏航控制
        yaw_error = -root_euler[2]
        yaw_torque = self.kp_yaw * yaw_error + self.kd_yaw * (-root_vel[2])

        torso_torque = np.array([roll_torque, pitch_torque, yaw_torque])
        torso_torque = np.clip(torso_torque, -60.0, 60.0)

        # 2. 重心位置补偿
        com = self.data.subtree_com[0].astype(np.float64).copy()
        com_error = self.com_target - com
        com_error[2] = np.clip(com_error[2], -0.1, 0.1)
        com_compensation = self.kp_com * com_error

        # 3. 关节控制
        self._detect_foot_contact()
        current_joints = self.data.qpos[7:7 + self.num_joints].astype(np.float64)
        current_vel = self.data.qvel[6:6 + self.num_joints].astype(np.float64)
        current_vel = np.clip(current_vel, -15.0, 15.0)

        # 处理腿部关节
        leg_joints = [
            # 左腿
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            # 右腿
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
        ]

        for joint_name in leg_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]

            # 根据关节类型设置增益
            if "hip" in joint_name:
                kp = self.kp_hip
                kd = self.kd_hip
                # 髋关节加入姿态补偿
                if "left" in joint_name:
                    if "roll" in joint_name:
                        joint_error -= torso_torque[0] * 0.07
                    elif "pitch" in joint_name:
                        joint_error += torso_torque[1] * 0.06
                else:  # 右腿
                    if "roll" in joint_name:
                        joint_error += torso_torque[0] * 0.07
                    elif "pitch" in joint_name:
                        joint_error += torso_torque[1] * 0.06

            elif "knee" in joint_name:
                kp = self.kp_knee
                kd = self.kd_knee
                # 膝关节加入重心补偿
                joint_error += com_compensation[2] * 0.12

            elif "ankle" in joint_name:
                kp = self.kp_ankle
                kd = self.kd_ankle
                # 踝关节加入姿态补偿
                if "pitch" in joint_name:
                    joint_error += torso_torque[1] * 0.08

            # 根据接触状态调整增益
            if ("left" in joint_name and self.foot_contact[1] == 0) or \
                    ("right" in joint_name and self.foot_contact[0] == 0):
                kp *= 0.4
                kd *= 0.6

            torques[idx] = kp * joint_error - kd * current_vel[idx]

        # 处理腰部关节
        waist_joint = "waist_yaw_joint"
        idx = self.joint_name_to_idx[waist_joint]
        joint_error = self.joint_targets[idx] - current_joints[idx]
        # 腰部加入偏航补偿
        joint_error += torso_torque[2] * 0.05
        torques[idx] = self.kp_waist * joint_error - self.kd_waist * current_vel[idx]

        # 处理手臂关节（保持自然下垂姿态）
        arm_joints = [
            # 左臂
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
            # 右臂
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"
        ]

        for joint_name in arm_joints:
            idx = self.joint_name_to_idx[joint_name]
            joint_error = self.joint_targets[idx] - current_joints[idx]
            torques[idx] = self.kp_arm * joint_error - self.kd_arm * current_vel[idx]

        # 4. 力矩限幅（根据g1_23dof.xml中的actuatorfrcrange设置）
        torque_limits = {
            # 腿部关节限幅
            "left_hip_pitch_joint": 88, "left_hip_roll_joint": 88, "left_hip_yaw_joint": 88,
            "left_knee_joint": 139, "left_ankle_pitch_joint": 50, "left_ankle_roll_joint": 50,
            "right_hip_pitch_joint": 88, "right_hip_roll_joint": 88, "right_hip_yaw_joint": 88,
            "right_knee_joint": 139, "right_ankle_pitch_joint": 50, "right_ankle_roll_joint": 50,
            # 腰部关节
            "waist_yaw_joint": 88,
            # 手臂关节
            "left_shoulder_pitch_joint": 25, "left_shoulder_roll_joint": 25,
            "left_shoulder_yaw_joint": 25, "left_elbow_joint": 25, "left_wrist_roll_joint": 25,
            "right_shoulder_pitch_joint": 25, "right_shoulder_roll_joint": 25,
            "right_shoulder_yaw_joint": 25, "right_elbow_joint": 25, "right_wrist_roll_joint": 25
        }

        for joint_name, limit in torque_limits.items():
            idx = self.joint_name_to_idx[joint_name]
            torques[idx] = np.clip(torques[idx], -limit, limit)

        self.prev_com = com
        return torques

    def simulate_stable_standing(self):
        """运行稳定站立仿真"""
        # 优化仿真参数
        self.model.opt.gravity[2] = -9.81
        self.model.opt.timestep = 0.002
        self.model.opt.iterations = 100
        self.model.opt.tolerance = 1e-6

        with viewer.launch_passive(self.model, self.data) as v:
            print("G1机器人稳定站立仿真启动...")
            print("适配g1_23dof模型 | 23自由度控制 | 足部接触检测")

            # 初始落地阶段
            start_time = time.time()
            while time.time() - start_time < self.init_wait_time:
                self.data.ctrl[:] = 0.0
                mujoco.mj_step(self.model, self.data)
                self.data.qvel[:] *= 0.9  # 速度衰减，减少冲击
                v.sync()
                time.sleep(0.01)

            # 主仿真循环
            while self.data.time < self.sim_duration:
                torques = self._calculate_stabilizing_torques()
                self.data.ctrl[:] = torques

                mujoco.mj_step(self.model, self.data)

                # 状态监测（每2秒打印）
                if self.data.time % 2 < 0.1:
                    com = self.data.subtree_com[0]
                    euler = self._get_root_euler()
                    print(
                        f"时间:{self.data.time:.1f}s | 重心(z):{com[2]:.3f}m | "
                        f"姿态(roll/pitch):{euler[0]:.3f}/{euler[1]:.3f}rad | 脚接触:{self.foot_contact}"
                    )

                v.sync()
                time.sleep(0.001)

                # 跌倒判定
                com = self.data.subtree_com[0]
                euler = self._get_root_euler()
                if com[2] < 0.5 or abs(euler[0]) > 0.6 or abs(euler[1]) > 0.6:
                    print(
                        f"跌倒！时间:{self.data.time:.1f}s | 重心(z):{com[2]:.3f}m | "
                        f"最大倾角:{max(abs(euler[0]), abs(euler[1])):.3f}rad"
                    )
                    break

        print("仿真完成！")


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(current_directory, "g1_23dof.xml")  # 适配G1模型文件名

    print(f"模型路径：{model_file_path}")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"模型文件不存在：{model_file_path}")

    try:
        stabilizer = G1Stabilizer(model_file_path)
        stabilizer.simulate_stable_standing()
    except Exception as e:
        print(f"错误：{e}")