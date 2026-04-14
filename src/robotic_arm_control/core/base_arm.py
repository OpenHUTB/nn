import numpy as np
import math
import yaml
import logging
import mujoco
import os
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseRoboticArm:
    """机械臂基础类：封装运动学和核心配置"""

    def __init__(self, config_path="config/arm_config.yaml"):
        self.config_path = os.path.abspath(config_path)
        self.dh_params = {}
        self.joint_limits = {}
        self.joint_num = 6
        self.joint_speed_limits = [10.0, 8.0, 8.0, 15.0, 15.0, 20.0]  # 关节最大速度（度/秒）

        # 加载配置
        self._load_config()
        # 预计算固定参数
        self._precompute_dh_fixed_params()
        # 缓存雅克比矩阵
        self.jacobian_cache = {}

    def _load_config(self):
        """加载配置文件（带异常处理）"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.dh_params = config['DH_PARAMS']
            self.joint_limits = config['JOINT_LIMITS']
            logger.info(f"成功加载配置文件：{self.config_path}")
        except FileNotFoundError:
            logger.error(f"配置文件不存在：{self.config_path}")
            raise
        except KeyError as e:
            logger.error(f"配置文件缺少参数：{e}")
            raise

    def _precompute_dh_fixed_params(self):
        """预计算D-H固定参数（减少重复计算）"""
        self.dh_fixed = {}
        for joint, params in self.dh_params.items():
            self.dh_fixed[joint] = {
                'a': params['a'] / 1000,
                'alpha_rad': math.radians(params['alpha']),
                'd': params['d'] / 1000,
                'theta_base': params['theta']
            }

    def _clip_joint_angles(self, joint_angles):
        """关节角裁剪（归一化+限位）"""
        clipped_angles = []
        for i, (joint, limits) in enumerate(self.joint_limits.items()):
            angle = joint_angles[i]
            # 归一化到[-180, 180]
            angle = angle % 360
            if angle > 180:
                angle -= 360
            # 限位裁剪
            clipped_angle = np.clip(angle, limits[0], limits[1])
            clipped_angles.append(clipped_angle)
        return clipped_angles

    def _clip_joint_speed(self, current_joints, target_joints, dt):
        """关节速度限制"""
        current = np.array(current_joints)
        target = np.array(target_joints)
        delta = target - current
        # 计算最大允许变化量
        max_delta = np.array(self.joint_speed_limits) * dt
        # 限制每个关节的变化量
        delta_clipped = np.clip(delta, -max_delta, max_delta)
        return (current + delta_clipped).tolist()

    def forward_kinematics(self, joint_angles):
        """正运动学解算（缓存优化）"""
        if len(joint_angles) != self.joint_num:
            raise ValueError(f"需输入{self.joint_num}个关节角")

        joint_angles = self._clip_joint_angles(joint_angles)

        # 计算D-H变换矩阵
        T_total = np.eye(4, dtype=np.float64)
        for i, (joint, fixed) in enumerate(self.dh_fixed.items()):
            theta_total = fixed['theta_base'] + joint_angles[i]
            theta_rad = math.radians(theta_total)
            cos_theta = math.cos(theta_rad)
            sin_theta = math.sin(theta_rad)
            cos_alpha = math.cos(fixed['alpha_rad'])
            sin_alpha = math.sin(fixed['alpha_rad'])

            T_i = np.array([
                [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, fixed['a'] * cos_theta],
                [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, fixed['a'] * sin_theta],
                [0, sin_alpha, cos_alpha, fixed['d']],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            T_total = np.dot(T_total, T_i)

        # 提取位姿
        x, y, z = T_total[0, 3], T_total[1, 3], T_total[2, 3]
        r11, r12, r13 = T_total[0, 0], T_total[0, 1], T_total[0, 2]
        r21, r22, r23 = T_total[1, 0], T_total[1, 1], T_total[1, 2]
        r31, r32, r33 = T_total[2, 0], T_total[2, 1], T_total[2, 2]

        rx = math.degrees(math.atan2(r32, r33))
        ry = math.degrees(math.atan2(-r31, math.hypot(r32, r33)))
        rz = math.degrees(math.atan2(r21, r11))

        return [round(x, 3), round(y, 3), round(z, 3), round(rx, 2), round(ry, 2), round(rz, 2)]

    def inverse_kinematics(self, target_pose, initial_joints=None, max_iter=200, tolerance=1e-3, dt=0.033):
        """逆运动学解算（速度限制+缓存优化）"""
        initial_joints = initial_joints if initial_joints else [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
        current_joints = np.array(self._clip_joint_angles(initial_joints), dtype=np.float64)
        target_pos = np.array(target_pose[:3], dtype=np.float64)

        damping_factor = 0.8
        base_step_size = 0.005

        for iter_num in range(max_iter):
            current_pose = self.forward_kinematics(current_joints)
            current_pos = np.array(current_pose[:3], dtype=np.float64)

            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            if error_norm < tolerance:
                logger.debug(f"逆解收敛：迭代{iter_num}次，误差{error_norm:.4f}米")
                return self._clip_joint_angles(current_joints)

            # 缓存雅克比矩阵（减少重复计算）
            cache_key = tuple(current_joints.round(2))
            if cache_key in self.jacobian_cache:
                J = self.jacobian_cache[cache_key]
            else:
                J = self._compute_jacobian(current_joints)
                self.jacobian_cache[cache_key] = J

            J_pinv = np.linalg.pinv(J, rcond=1e-6)
            step_size = base_step_size * min(1.0, error_norm / 0.01)
            delta_joints = step_size * np.dot(J_pinv, error)
            delta_joints *= damping_factor

            current_joints += delta_joints
            # 速度限制
            current_joints = np.array(self._clip_joint_speed(
                current_joints.tolist(), (current_joints + delta_joints).tolist(), dt
            ), dtype=np.float64)
            current_joints = np.array(self._clip_joint_angles(current_joints.tolist()), dtype=np.float64)

        logger.warning(f"逆解未收敛，误差{error_norm:.4f}米，返回合法关节角")
        return self._clip_joint_angles(current_joints)

    def _compute_jacobian(self, joint_angles):
        """计算雅克比矩阵（缓存核心）"""
        J = np.zeros((3, self.joint_num), dtype=np.float64)
        delta = 1e-4
        for i in range(self.joint_num):
            joints_plus = joint_angles.copy()
            joints_plus[i] += delta
            joints_plus = self._clip_joint_angles(joints_plus)
            pos_plus = np.array(self.forward_kinematics(joints_plus)[:3])

            joints_minus = joint_angles.copy()
            joints_minus[i] -= delta
            joints_minus = self._clip_joint_angles(joints_minus)
            pos_minus = np.array(self.forward_kinematics(joints_minus)[:3])

            J[:, i] = (pos_plus - pos_minus) / (2 * delta)
        return J


class BaseMuJoCoSim:
    """MuJoCo基础仿真类"""

    def __init__(self, model_path="model/six_axis_arm.xml"):
        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}")

        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            logger.info(f"成功加载MuJoCo模型：{self.model_path}")
            logger.info(f"MuJoCo版本：{mujoco.__version__}")
        except Exception as e:
            logger.error(f"加载模型失败：{e}")
            raise

        # 基础配置
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.actuator_names = [f"act{i + 1}" for i in range(6)]
        self.link_geom_names = [
            "base_geom", "link1_geom", "link2_geom", "link3_geom",
            "link4_geom", "link5_geom", "end_effector"
        ]

        # 获取ID
        self.joint_ids = self._get_ids(mujoco.mjtObj.mjOBJ_JOINT, self.joint_names)
        self.actuator_ids = self._get_ids(mujoco.mjtObj.mjOBJ_ACTUATOR, self.actuator_names)

        # 仿真参数
        self.fps = 30
        self.dt = 1.0 / self.fps

    def _get_ids(self, obj_type, names):
        """批量获取MuJoCo对象ID"""
        ids = []
        for name in names:
            obj_id = mujoco.mj_name2id(self.model, obj_type, name)
            if obj_id == -1:
                raise ValueError(f"未找到对象：{name}")
            ids.append(obj_id)
        return ids

    def get_joint_angles(self):
        """获取当前关节角（滤波）"""
        raw_radians = [self.data.joint(jid).qpos[0] for jid in self.joint_ids]
        raw_angles = np.degrees(raw_radians).tolist()

        # 简单滤波
        normalized_angles = []
        for angle in raw_angles:
            angle = angle % 360
            if angle > 180:
                angle -= 360
            normalized_angles.append(angle)

        return [round(angle, 2) for angle in normalized_angles]

    def set_joint_angles(self, joint_angles):
        """设置关节角"""
        if len(joint_angles) != 6:
            raise ValueError(f"需输入6个关节角")

        joint_radians = np.radians(joint_angles)
        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = joint_radians[i]

    def check_collision(self):
        """碰撞检测"""
        collision_pairs = []
        has_collision = False

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            if (geom1_name in self.link_geom_names and geom2_name in self.link_geom_names) or \
                    (geom1_name in self.link_geom_names and geom2_name == "floor") or \
                    (geom2_name in self.link_geom_names and geom1_name == "floor"):
                has_collision = True
                collision_pairs.append((geom1_name, geom2_name))

        if has_collision:
            logger.warning(f"碰撞检测：{collision_pairs}")
        return has_collision, collision_pairs