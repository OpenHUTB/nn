import numpy as np
import math
import yaml
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RoboticArmKinematics:
    """机械臂运动学解算类（防关节溢出版）"""

    def __init__(self, config_path="config/arm_config.yaml"):
        """初始化：加载D-H参数和关节限位"""
        self.config_path = config_path
        self.dh_params = {}
        self.joint_limits = {}
        self.joint_num = 6

        # 缓存配置
        self._jacobian_cache = {}
        self._cache_max = 1000

        # 加载配置并缓存
        self._load_config()
        # 预计算D-H参数的固定部分
        self._precompute_dh_fixed_params()

    def _load_config(self):
        """加载配置文件（带异常处理）"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.dh_params = config['DH_PARAMS']
            self.joint_limits = config['JOINT_LIMITS']
            logger.info("成功加载运动学配置文件")
        except FileNotFoundError:
            logger.error(f"配置文件不存在：{self.config_path}")
            raise
        except KeyError as e:
            logger.error(f"配置文件缺少关键参数：{e}")
            raise

    def _precompute_dh_fixed_params(self):
        """预计算D-H参数的固定部分（alpha转弧度）"""
        self.dh_fixed = {}
        for joint, params in self.dh_params.items():
            self.dh_fixed[joint] = {
                'a': params['a'] / 1000,  # mm → m
                'alpha_rad': math.radians(params['alpha']),
                'd': params['d'] / 1000,
                'theta_base': params['theta']
            }

    def _dh_transform(self, theta_joint):
        """生成D-H变换矩阵"""

        def _single_dh(a, alpha_rad, d, theta_total):
            theta_rad = math.radians(theta_total)
            cos_theta = math.cos(theta_rad)
            sin_theta = math.sin(theta_rad)
            cos_alpha = math.cos(alpha_rad)
            sin_alpha = math.sin(alpha_rad)

            return np.array([
                [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
                [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
                [0, sin_alpha, cos_alpha, d],
                [0, 0, 0, 1]
            ], dtype=np.float64)

        T_total = np.eye(4, dtype=np.float64)
        for i, (joint, fixed) in enumerate(self.dh_fixed.items()):
            theta_total = fixed['theta_base'] + theta_joint[i]
            T_i = _single_dh(
                fixed['a'], fixed['alpha_rad'], fixed['d'], theta_total
            )
            T_total = np.dot(T_total, T_i)
        return T_total

    def _clip_joint_angles(self, joint_angles):
        """双重限位裁剪：确保关节角绝对在限位内"""
        clipped_angles = []
        for i, (joint, limits) in enumerate(self.joint_limits.items()):
            angle = joint_angles[i]
            # 先将角度归一化到 [-360, 360]（处理数值溢出）
            angle = angle % 360
            if angle > 180:
                angle -= 360
            # 再裁剪到关节限位
            clipped_angle = np.clip(angle, limits[0], limits[1])
            # 记录裁剪情况
            if abs(clipped_angle - angle) > 0.1:
                logger.warning(f"关节{joint}角度{angle:.2f}度超出限位，已裁剪为{clipped_angle:.2f}度")
            clipped_angles.append(clipped_angle)
        return clipped_angles

    def forward_kinematics(self, joint_angles):
        """正运动学解算（防溢出版）"""
        # 参数校验
        if len(joint_angles) != self.joint_num:
            raise ValueError(f"需输入{self.joint_num}个关节角，当前输入{len(joint_angles)}个")

        # 第一步：强制裁剪关节角（即使超出也先裁剪，避免直接报错）
        joint_angles = self._clip_joint_angles(joint_angles)

        # 第二步：限位检查（冗余保护）
        for i, (joint, limits) in enumerate(self.joint_limits.items()):
            angle = joint_angles[i]
            if not (limits[0] <= angle <= limits[1]):
                raise ValueError(f"关节{joint}超出限位：{angle}度（范围：{limits[0]}-{limits[1]}度）")

        # 计算变换矩阵
        T_total = self._dh_transform(joint_angles)

        # 提取位置和姿态
        x, y, z = T_total[0, 3], T_total[1, 3], T_total[2, 3]
        r11, r12, r13 = T_total[0, 0], T_total[0, 1], T_total[0, 2]
        r21, r22, r23 = T_total[1, 0], T_total[1, 1], T_total[1, 2]
        r31, r32, r33 = T_total[2, 0], T_total[2, 1], T_total[2, 2]

        rx = math.degrees(math.atan2(r32, r33))
        ry = math.degrees(math.atan2(-r31, math.hypot(r32, r33)))
        rz = math.degrees(math.atan2(r21, r11))

        return [round(x, 3), round(y, 3), round(z, 3), round(rx, 2), round(ry, 2), round(rz, 2)]

    def inverse_kinematics(self, target_pose, initial_joints=None, max_iter=200, tolerance=1e-3):
        """逆运动学解算（缓存优化版）"""
        # 初始化关节角（默认更安全的初始值）
        initial_joints = initial_joints if initial_joints else [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
        current_joints = np.array(self._clip_joint_angles(initial_joints), dtype=np.float64)
        target_pos = np.array(target_pose[:3], dtype=np.float64)

        # 降低步长，增加阻尼，避免超调
        damping_factor = 0.8  # 阻尼系数
        base_step_size = 0.005  # 步长

        for iter_num in range(max_iter):
            # 计算当前末端位置
            current_pose = self.forward_kinematics(current_joints)
            current_pos = np.array(current_pose[:3], dtype=np.float64)

            # 计算误差
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            if error_norm < tolerance:
                logger.info(f"逆解收敛：迭代{iter_num}次，误差{error_norm:.4f}米")
                return self._clip_joint_angles(current_joints)

            # 雅克比矩阵计算（带缓存）
            J = self._compute_jacobian_fast(current_joints)

            # 伪逆求解（添加正则化）
            J_pinv = np.linalg.pinv(J, rcond=1e-6)

            # 自适应步长+阻尼
            step_size = base_step_size * min(1.0, error_norm / 0.01)
            delta_joints = step_size * np.dot(J_pinv, error)
            delta_joints *= damping_factor

            # 更新关节角并立即裁剪
            current_joints += delta_joints
            current_joints = np.array(self._clip_joint_angles(current_joints), dtype=np.float64)

        # 迭代未收敛
        logger.warning(f"逆解迭代{max_iter}次未收敛，误差{error_norm:.4f}米，返回合法关节角")
        return self._clip_joint_angles(current_joints)

    def _compute_jacobian_fast(self, joint_angles):
        """快速计算雅克比矩阵（缓存优化）"""
        # 缓存键（圆整减少缓存条目）
        cache_key = tuple(np.round(joint_angles, 2))
        if cache_key in self._jacobian_cache:
            return self._jacobian_cache[cache_key]

        J = np.zeros((3, self.joint_num), dtype=np.float64)
        delta = 1e-4

        # 批量扰动计算
        joints_plus = joint_angles.copy()
        joints_minus = joint_angles.copy()
        for i in range(self.joint_num):
            # 正向扰动
            joints_plus[i] += delta
            pos_plus = np.array(self.forward_kinematics(self._clip_joint_angles(joints_plus))[:3])
            joints_plus[i] = joint_angles[i]

            # 反向扰动
            joints_minus[i] -= delta
            pos_minus = np.array(self.forward_kinematics(self._clip_joint_angles(joints_minus))[:3])
            joints_minus[i] = joint_angles[i]

            J[:, i] = (pos_plus - pos_minus) / (2 * delta)

        # 缓存管理
        if len(self._jacobian_cache) < self._cache_max:
            self._jacobian_cache[cache_key] = J
        return J