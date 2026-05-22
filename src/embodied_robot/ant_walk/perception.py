# -*- coding: utf-8 -*-
import numpy as np
import mujoco


class AntPerception:
    """
    Ant 感知模块。
    负责处理机器人与动态目标点、机器人与障碍物之间的空间关系。

    当前版本支持：
    1. 动态小球目标：目标坐标由 env_manager.py 按时间刷新后传入；
    2. box 墙体障碍：按矩形边界计算最近距离；
    3. y 方向任务绕墙：墙体横跨 x 方向，机器人沿 y 方向前进时，从 +X 或 -X 侧绕行。
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

        # 自动收集 XML 中所有以 obstacle_ 开头的 geom。
        self.obstacle_ids = []
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name and name.startswith("obstacle_"):
                self.obstacle_ids.append(geom_id)

        self.forward_axis = int(getattr(self.config, "ball_forward_axis", 1))
        self.side_axis = 1 - self.forward_axis

    def get_robot_pose(self, data):
        """
        获取机器人当前的位置和朝向（偏航角 Yaw）。
        """
        root_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self.config.root_joint_name
        )
        qpos_adr = self.model.jnt_qposadr[root_id]

        pos = data.qpos[qpos_adr: qpos_adr + 3]

        quat = data.qpos[qpos_adr + 3: qpos_adr + 7]
        siny_cosp = 2 * (quat[0] * quat[3] + quat[1] * quat[2])
        cosy_cosp = 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return pos, yaw

    def get_target_feedback(self, data, target_pos):
        """
        计算相对于动态目标点的距离和角度误差。

        注意：当前 Ant 的开环步态自然前进方向更接近机体局部 +Y。
        MuJoCo 四元数换算得到的 yaw 通常表示机体局部 +X 轴朝向，
        因此这里用 yaw + robot_forward_yaw_offset 作为实际前进方向。
        这样目标位于世界坐标 +Y 方向时，初始角度误差为 0，
        小球沿 x 轴移动或绕墙时才产生转向误差。
        """
        robot_pos, robot_yaw = self.get_robot_pose(data)

        error_vec = target_pos - robot_pos
        distance = np.linalg.norm(error_vec[:2])

        target_angle = np.arctan2(error_vec[1], error_vec[0])
        forward_yaw = robot_yaw + getattr(self.config, "robot_forward_yaw_offset", 0.0)
        angle_error = target_angle - forward_yaw
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        return distance, angle_error

    @staticmethod
    def _distance_vector_to_box(robot_xy, box_center_xy, box_half_size_xy):
        """
        计算机器人到轴对齐 box 的最近距离向量。
        返回 diff, dist，其中 diff 指向“从墙体最近点到机器人”的方向。
        """
        delta = robot_xy - box_center_xy
        closest_delta = np.clip(delta, -box_half_size_xy, box_half_size_xy)
        closest = box_center_xy + closest_delta
        diff = robot_xy - closest
        dist = np.linalg.norm(diff)

        # 如果机器人投影已经进入 box 内部，按最近边界推出。
        if dist < 1e-6:
            remain = box_half_size_xy - np.abs(delta)
            axis = int(np.argmin(remain))
            diff = np.zeros(2)
            diff[axis] = 1.0 if delta[axis] >= 0 else -1.0
            dist = 0.0

        return diff, dist

    def _wall_bypass_vector(self, robot_pos, target_pos, wall_pos, wall_half_size, dist_to_wall):
        """
        当远处动态小球在墙后方、机器人尚未越过墙体且仍处于墙体阻挡范围附近时，
        给机器人一个固定侧向引导，使其从墙体一侧绕过去。

        这里不再写死 x 方向任务，而是通过 config.ball_forward_axis 指定主前进轴：
        - forward_axis = 1：沿 y 方向追踪；
        - side_axis = 0：沿 x 方向绕墙。
        """
        if target_pos is None:
            return np.zeros(2)

        robot_xy = robot_pos[:2]
        target_xy = target_pos[:2]
        wall_xy = wall_pos[:2]
        half_xy = wall_half_size[:2]

        f = self.forward_axis
        s = self.side_axis

        target_behind_wall = target_xy[f] > wall_xy[f] + half_xy[f]
        robot_not_past_wall = robot_xy[f] < wall_xy[f] + half_xy[f] + self.config.wall_exit_margin
        close_in_forward_axis = abs(robot_xy[f] - wall_xy[f]) < self.config.wall_detect_margin

        desired_side_clearance = half_xy[s] + self.config.wall_bypass_clearance
        still_in_blocking_band = abs(robot_xy[s] - wall_xy[s]) < desired_side_clearance

        if not (target_behind_wall and robot_not_past_wall and close_in_forward_axis and still_in_blocking_band):
            return np.zeros(2)

        # 越靠近墙、越处于墙体正前方，侧向引导越强。
        clearance_error = desired_side_clearance - abs(robot_xy[s] - wall_xy[s])
        band_strength = np.clip(clearance_error / max(desired_side_clearance, 1e-6), 0.0, 1.0)
        dist_strength = np.clip(
            (self.config.obstacle_margin - dist_to_wall) / max(self.config.obstacle_margin, 1e-6),
            0.0,
            1.0,
        )
        strength = max(band_strength, dist_strength)

        side = float(np.sign(self.config.wall_bypass_side) or 1.0)
        vec = np.zeros(2, dtype=float)
        vec[s] = side * self.config.wall_side_force * strength

        # 轻微保留前进方向，引导机器人绕过墙角后继续向远球靠近，避免纯侧向横移。
        vec[f] = 0.25 * self.config.wall_side_force * strength
        return vec

    def get_obstacle_avoidance_vector(self, data, current_target=None):
        """
        感知周围障碍物并生成避障向量。

        对墙体 box：
        - 根据机器人到 box 边界的最近距离生成斥力；
        - 当墙挡在机器人和远球之间时，额外加入侧向绕行向量。
        """
        robot_pos, _ = self.get_robot_pose(data)
        robot_xy = robot_pos[:2]

        avoid_vec = np.zeros(2, dtype=float)
        min_dist = float("inf")

        for geom_id in self.obstacle_ids:
            obs_pos = data.geom_xpos[geom_id]
            geom_type = self.model.geom_type[geom_id]

            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                half_xy = self.model.geom_size[geom_id][:2]
                diff, dist = self._distance_vector_to_box(robot_xy, obs_pos[:2], half_xy)
            else:
                diff = robot_xy - obs_pos[:2]
                dist = np.linalg.norm(diff)

            min_dist = min(min_dist, dist)

            if dist < self.config.obstacle_margin:
                strength = (self.config.obstacle_margin - dist) / self.config.obstacle_margin
                avoid_vec += (diff / (dist + 1e-5)) * strength

            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                wall_half_size = self.model.geom_size[geom_id]
                avoid_vec += self._wall_bypass_vector(robot_pos, current_target, obs_pos, wall_half_size, dist)

        return avoid_vec, min_dist

    def sense_environment(self, data, current_target):
        """
        综合感知接口：提供给 Planner 调用的最终数据。
        """
        robot_pos, robot_yaw = self.get_robot_pose(data)
        robot_forward_yaw = robot_yaw + getattr(self.config, "robot_forward_yaw_offset", 0.0)
        dist_to_target, angle_to_target = self.get_target_feedback(data, current_target)
        avoid_vector, min_obs_dist = self.get_obstacle_avoidance_vector(data, current_target)

        return {
            "robot_pos": robot_pos.copy(),
            "robot_yaw": robot_yaw,
            "robot_forward_yaw": robot_forward_yaw,
            "dist_target": dist_to_target,
            "angle_target": angle_to_target,
            "avoid_vector": avoid_vector,
            "min_obstacle_distance": min_obs_dist,
            "danger_level": 1.0 if min_obs_dist < self.config.safe_distance else 0.0,
        }
