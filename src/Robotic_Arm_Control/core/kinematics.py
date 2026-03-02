import numpy as np
import math
import yaml

class RoboticArmKinematics:
    def __init__(self, config_path="config/arm_config.yaml"):
        # 加载配置（兼容pyyaml>=6.0）
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.dh_params = self.config['DH_PARAMS']
        self.joint_limits = self.config['JOINT_LIMITS']
        self.joint_num = 6  # 固定六轴

    def dh_transform(self, a, alpha, d, theta):
        """单个D-H变换矩阵（单位：mm/度转弧度）"""
        alpha_rad = math.radians(alpha)
        theta_rad = math.radians(theta)
        return np.array([
            [math.cos(theta_rad), -math.sin(theta_rad)*math.cos(alpha_rad), math.sin(theta_rad)*math.sin(alpha_rad), a*math.cos(theta_rad)],
            [math.sin(theta_rad), math.cos(theta_rad)*math.cos(alpha_rad), -math.cos(theta_rad)*math.sin(alpha_rad), a*math.sin(theta_rad)],
            [0, math.sin(alpha_rad), math.cos(alpha_rad), d],
            [0, 0, 0, 1]
        ], dtype=np.float64)  # 显式指定类型，兼容numpy>=1.24.0

    def forward_kinematics(self, joint_angles):
        """正运动学：关节角→末端位姿"""
        if len(joint_angles) != 6:
            raise ValueError("需输入6个关节角")
        # 关节限位检查
        for i, limits in enumerate(self.joint_limits.values()):
            if not (limits[0] <= joint_angles[i] <= limits[1]):
                raise ValueError(f"关节{i+1}超出限位：{joint_angles[i]}度 (范围：{limits[0]}-{limits[1]}度)")

        T_total = np.eye(4, dtype=np.float64)
        for i, (_, params) in enumerate(self.dh_params.items()):
            T_i = self.dh_transform(params['a'], params['alpha'], params['d'], params['theta'] + joint_angles[i])
            T_total = np.dot(T_total, T_i)

        # 提取位置（转换为米，适配MuJoCo）和姿态
        x = round(T_total[0, 3] / 1000, 3)  # mm → m
        y = round(T_total[1, 3] / 1000, 3)
        z = round(T_total[2, 3] / 1000, 3)
        rx = round(math.degrees(math.atan2(T_total[2, 1], T_total[2, 2])), 2)
        ry = round(math.degrees(math.atan2(-T_total[2, 0], math.sqrt(T_total[2, 1]**2 + T_total[2, 2]**2))), 2)
        rz = round(math.degrees(math.atan2(T_total[1, 0], T_total[0, 0])), 2)
        return [x, y, z, rx, ry, rz]

    def inverse_kinematics(self, target_pose):
        """简化逆运动学（数值迭代，适配MuJoCo米单位）"""
        # 转换为mm（内部计算用）
        target_pose_mm = [p * 1000 for p in target_pose[:3]] + target_pose[3:]
        initial_joints = [0.0] * 6
        max_iter = 500
        tolerance = 1e-3  # mm
        current_joints = np.array(initial_joints, dtype=np.float64)
        target_pos = np.array(target_pose_mm[:3])

        for _ in range(max_iter):
            current_pose = self.forward_kinematics(current_joints)
            current_pos = np.array([p * 1000 for p in current_pose[:3]])  # m → mm
            error = target_pos - current_pos
            if np.linalg.norm(error) < tolerance:
                break

            # 简化雅克比矩阵
            J = np.zeros((3, 6), dtype=np.float64)
            delta = 1e-4
            for i in range(6):
                joints_plus = current_joints.copy()
                joints_plus[i] += delta
                pos_plus = np.array([p * 1000 for p in self.forward_kinematics(joints_plus)[:3]])
                joints_minus = current_joints.copy()
                joints_minus[i] -= delta
                pos_minus = np.array([p * 1000 for p in self.forward_kinematics(joints_minus)[:3]])
                J[:, i] = (pos_plus - pos_minus) / (2 * delta)

            # 更新关节角
            current_joints += 0.01 * np.dot(np.linalg.pinv(J), error)
            # 限位裁剪
            for i, limits in enumerate(self.joint_limits.values()):
                current_joints[i] = np.clip(current_joints[i], limits[0], limits[1])
        else:
            raise ValueError("逆运动学迭代未收敛，误差：{:.2f}mm".format(np.linalg.norm(error)))
        return [round(j, 2) for j in current_joints]