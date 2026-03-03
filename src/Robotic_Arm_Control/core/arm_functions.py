import numpy as np
import math
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArmFunctions:
    """机械臂扩展功能类"""

    def __init__(self, kinematics):
        self.kinematics = kinematics  # 关联运动学类
        self.joint_num = 6
        self.trajectory_points = []  # 轨迹点缓存

    # ========== 功能1：轨迹规划 ==========
    def generate_linear_trajectory(self, start_joints, target_joints, num_points=50):
        """
        生成关节空间线性插值轨迹
        :param start_joints: 起始关节角 [j1-j6]（度）
        :param target_joints: 目标关节角 [j1-j6]（度）
        :param num_points: 轨迹点数量
        :return: 轨迹点列表 [[j1,j2,...j6], ...]
        """
        # 校验输入
        if len(start_joints) != self.joint_num or len(target_joints) != self.joint_num:
            raise ValueError("起始/目标关节角需为6维数组")

        # 线性插值
        start_joints = np.array(start_joints, dtype=np.float64)
        target_joints = np.array(target_joints, dtype=np.float64)
        trajectory = np.linspace(start_joints, target_joints, num_points)

        # 每个轨迹点都做限位裁剪
        self.trajectory_points = []
        for point in trajectory:
            clipped_point = self.kinematics._clip_joint_angles(point.tolist())
            self.trajectory_points.append(clipped_point)

        logger.info(f"生成轨迹：{num_points}个点，从{start_joints.round(2)}到{target_joints.round(2)}")
        return self.trajectory_points

    def get_next_trajectory_point(self, current_index):
        """获取下一个轨迹点（循环/终止）"""
        if current_index >= len(self.trajectory_points) - 1:
            logger.info("轨迹执行完成")
            return None, current_index
        next_index = current_index + 1
        return self.trajectory_points[next_index], next_index

    # ========== 功能2：实时手动控制 ==========
    def manual_joint_control(self, current_joints, key, step=1.0):
        """
        键盘手动控制关节
        :param current_joints: 当前关节角 [j1-j6]（度）
        :param key: 按键（j1+/j1-/j2+/j2-.../stop）
        :param step: 单步角度（度）
        :return: 新关节角
        """
        new_joints = current_joints.copy()
        joint_map = {
            'j1+': 0, 'j1-': 0,
            'j2+': 1, 'j2-': 1,
            'j3+': 2, 'j3-': 2,
            'j4+': 3, 'j4-': 3,
            'j5+': 4, 'j5-': 4,
            'j6+': 5, 'j6-': 5
        }

        if key not in joint_map and key != 'stop':
            logger.warning(f"无效按键：{key}，支持的按键：j1+/j1-/j2+/j2-.../j6+/j6-/stop")
            return new_joints

        if key == 'stop':
            return new_joints

        # 调整关节角
        joint_idx = joint_map[key]
        if '+' in key:
            new_joints[joint_idx] += step
        else:
            new_joints[joint_idx] -= step

        # 裁剪限位
        new_joints = self.kinematics._clip_joint_angles(new_joints)
        logger.debug(
            f"手动控制：{key} → 关节{joint_idx + 1}从{current_joints[joint_idx]:.1f}→{new_joints[joint_idx]:.1f}度")
        return new_joints

    # ========== 功能3：目标点跟随 ==========
    def follow_moving_target(self, current_joints, target_pos, max_iter=100, tolerance=1e-3):
        """
        实时跟随移动的目标点
        :param current_joints: 当前关节角（度）
        :param target_pos: 实时目标位置 [x,y,z]（米）
        :return: 新关节角（度）
        """
        # 补全姿态为0（仅跟随位置）
        target_pose = target_pos + [0, 0, 0]
        try:
            new_joints = self.kinematics.inverse_kinematics(
                target_pose, initial_joints=current_joints, max_iter=max_iter, tolerance=tolerance
            )
            return new_joints
        except Exception as e:
            logger.warning(f"目标点跟随失败：{e}，使用当前关节角")
            return current_joints

    # ========== 功能4：碰撞检测 ==========
    def check_collision(self, model, data, link_geom_names):
        """
        检测机械臂碰撞
        :param model: MuJoCo模型
        :param data: MuJoCo数据
        :param link_geom_names: 连杆几何名称列表
        :return: 碰撞状态（bool）、碰撞对列表
        """
        collision_pairs = []
        has_collision = False

        # 遍历所有接触对
        for i in range(data.ncon):
            contact = data.contact[i]
            # 获取接触的两个几何名称
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)

            # 判断是否是机械臂连杆之间的碰撞，或连杆与地板的碰撞
            if (geom1_name in link_geom_names and geom2_name in link_geom_names) or \
                    (geom1_name in link_geom_names and geom2_name == "floor") or \
                    (geom2_name in link_geom_names and geom1_name == "floor"):
                has_collision = True
                collision_pairs.append((geom1_name, geom2_name))

        if has_collision:
            logger.warning(f"检测到碰撞：{collision_pairs}")
        return has_collision, collision_pairs


# 修复：导入mujoco（避免循环导入）
import mujoco