import airsim
import numpy as np
import math
from gym import spaces


class MultirotorDynamicsAirsim:
    """
    一种用于基于视觉导航的简化多旋翼动力学模型
    控制器为 AirSim Simple Flight (https://microsoft.github.io/AirSim/simple_flight/)
    控制接口: 使用 client.moveByVelocityZAsync(v_x, v_y, v_z, yaw_rate)
    """

    def __init__(self, cfg) -> None:

        # 配置
        self.navigation_3d = cfg.getboolean("options", "navigation_3d")
        self.using_velocity_state = cfg.getboolean("options", "using_velocity_state")
        self.dt = cfg.getfloat("multirotor", "dt")

        # AirSim客户端
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # 起点与目标位置
        self.start_position = [0, 0, 0]
        self.start_random_angle = None
        self.goal_position = [0, 0, 0]
        self.goal_distance = None
        self.goal_random_angle = None
        self.goal_rect = None

        # 状态变量
        self.x = 0
        self.y = 0
        self.z = 0
        self.v_xy = 0
        self.v_z = 0
        self.yaw = 0
        self.yaw_rate = 0

        # 控制指令
        self.v_xy_sp = 0
        self.v_z_sp = 0
        self.yaw_rate_sp = 0
        self.yaw_sp = 0

        # 动作空间
        self.acc_xy_max = cfg.getfloat("multirotor", "acc_xy_max")
        self.v_xy_max = cfg.getfloat("multirotor", "v_xy_max")
        self.v_xy_min = cfg.getfloat("multirotor", "v_xy_min")
        self.v_z_max = cfg.getfloat("multirotor", "v_z_max")
        self.yaw_rate_max_deg = cfg.getfloat("multirotor", "yaw_rate_max_deg")
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)
        self.max_vertical_difference = 5
        self.action_smoothing_alpha = cfg.getfloat(
            "multirotor", "action_smoothing_alpha", fallback=0.25
        )
        self.action_smoothing_alpha = float(
            np.clip(self.action_smoothing_alpha, 0.0, 1.0)
        )
        self.yaw_rate_change_max_rad = math.radians(
            cfg.getfloat("multirotor", "yaw_rate_change_max_deg", fallback=5.0)
        )

        if self.navigation_3d:
            if self.using_velocity_state:
                self.state_feature_length = 6
            else:
                self.state_feature_length = 3
            self.action_space = spaces.Box(
                low=np.array([self.v_xy_min, -self.v_z_max, -self.yaw_rate_max_rad]),
                high=np.array([self.v_xy_max, self.v_z_max, self.yaw_rate_max_rad]),
                dtype=np.float32,
            )
        else:
            if self.using_velocity_state:
                self.state_feature_length = 4
            else:
                self.state_feature_length = 2
            self.action_space = spaces.Box(
                low=np.array([self.v_xy_min, -self.yaw_rate_max_rad]),
                high=np.array([self.v_xy_max, self.yaw_rate_max_rad]),
                dtype=np.float32,
            )

    def reset(self):
        self.client.reset()
        self.v_xy_sp = 0
        self.v_z_sp = 0
        self.yaw_rate_sp = 0
        # 重置目标
        self.update_goal_pose()

        # 重置起始状态
        yaw_noise = self.start_random_angle * np.random.random()
        # 设置AirSim位姿
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.start_position[0]
        pose.position.y_val = self.start_position[1]
        pose.position.z_val = -self.start_position[2]
        pose.orientation = airsim.to_quaternion(0, 0, yaw_noise)
        self.client.simSetVehiclePose(pose, True)

        self.client.simPause(False)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # 起飞
        self.client.moveToZAsync(-self.start_position[2], 2).join()

        self.client.simPause(True)

    def set_action(self, action):

        v_xy_target = float(np.clip(action[0], self.v_xy_min, self.v_xy_max))
        yaw_rate_target = float(
            np.clip(action[-1], -self.yaw_rate_max_rad, self.yaw_rate_max_rad)
        )
        if self.navigation_3d:
            v_z_target = float(np.clip(action[1], -self.v_z_max, self.v_z_max))
        else:
            v_z_target = 0

        alpha = self.action_smoothing_alpha
        v_xy_smoothed = self.v_xy_sp + alpha * (v_xy_target - self.v_xy_sp)
        v_z_smoothed = self.v_z_sp + alpha * (v_z_target - self.v_z_sp)
        yaw_rate_smoothed = self.yaw_rate_sp + alpha * (
            yaw_rate_target - self.yaw_rate_sp
        )

        v_xy_delta_max = self.acc_xy_max * self.dt
        self.v_xy_sp += float(
            np.clip(v_xy_smoothed - self.v_xy_sp, -v_xy_delta_max, v_xy_delta_max)
        )
        self.v_z_sp = float(np.clip(v_z_smoothed, -self.v_z_max, self.v_z_max))
        self.yaw_rate_sp += float(
            np.clip(
                yaw_rate_smoothed - self.yaw_rate_sp,
                -self.yaw_rate_change_max_rad,
                self.yaw_rate_change_max_rad,
            )
        )

        self.yaw = self.get_attitude()[2]
        self.yaw_sp = self.yaw + self.yaw_rate_sp * self.dt

        if self.yaw_sp > math.radians(180):
            self.yaw_sp -= math.pi * 2
        elif self.yaw_sp < math.radians(-180):
            self.yaw_sp += math.pi * 2

        vx_local_sp = self.v_xy_sp * math.cos(self.yaw_sp)
        vy_local_sp = self.v_xy_sp * math.sin(self.yaw_sp)

        self.client.simPause(False)
        if len(action) == 2:
            self.client.moveByVelocityZAsync(
                vx_local_sp,
                vy_local_sp,
                -self.start_position[2],
                self.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(
                    is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp)
                ),
            ).join()
        elif len(action) == 3:
            self.client.moveByVelocityAsync(
                vx_local_sp,
                vy_local_sp,
                -self.v_z_sp,
                self.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(
                    is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp)
                ),
            ).join()

        self.client.simPause(True)

    def update_goal_pose(self):
        # 若目标采用矩形区域模式
        if self.goal_rect is None:
            distance = self.goal_distance
            noise = np.random.random()
            angle = noise * self.goal_random_angle  # (0~2pi)
            goal_x = distance * math.cos(angle) + self.start_position[0]
            goal_y = distance * math.sin(angle) + self.start_position[1]
        else:
            goal_x, goal_y = self.get_goal_from_rect(
                self.goal_rect, self.goal_random_angle
            )
            self.goal_distance = math.sqrt(goal_x * goal_x + goal_y * goal_y)
        self.goal_position[0] = goal_x
        self.goal_position[1] = goal_y
        self.goal_position[2] = self.start_position[2]

    def set_start(self, position, random_angle):
        self.start_position = position
        self.start_random_angle = random_angle

    def set_goal(self, distance=None, random_angle=0, rect=None):
        if distance is not None:
            self.goal_distance = distance
        self.goal_random_angle = random_angle
        if rect is not None:
            self.goal_rect = rect

    def get_goal_from_rect(self, rect_set, random_angle_set):
        rect = rect_set
        random_angle = random_angle_set
        noise = np.random.random()
        angle = random_angle * noise - math.pi
        rect = [-128, -128, 128, 128]

        if abs(angle) == math.pi / 2:
            goal_x = 0
            if angle > 0:
                goal_y = rect[3]
            else:
                goal_y = rect[1]
        if abs(angle) <= math.pi / 4:
            goal_x = rect[2]
            goal_y = goal_x * math.tan(angle)
        elif abs(angle) > math.pi / 4 and abs(angle) <= math.pi / 4 * 3:
            if angle > 0:
                goal_y = rect[3]
                goal_x = goal_y / math.tan(angle)
            else:
                goal_y = rect[1]
                goal_x = goal_y / math.tan(angle)
        else:
            goal_x = rect[0]
            goal_y = goal_x * math.tan(angle)

        return goal_x, goal_y

    def _get_state_feature(self):
        """
        @description: 更新并获取当前无人机状态及state_norm
        @param {type}
        @return: state_norm
        归一化状态范围 0-255
        """

        distance = self.get_distance_to_goal_2d()
        relative_yaw = self._get_relative_yaw()
        relative_pose_z = (
            self.get_position()[2] - self.goal_position[2]
        )
        vertical_distance_norm = (
            relative_pose_z / self.max_vertical_difference / 2 + 0.5
        ) * 255

        distance_norm = distance / self.goal_distance * 255
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5) * 255

        # 当前线速度与角速度
        velocity = self.get_velocity()
        linear_velocity_xy = velocity[0]
        linear_velocity_norm = (
            (linear_velocity_xy - self.v_xy_min) / (self.v_xy_max - self.v_xy_min) * 255
        )
        linear_velocity_z = velocity[1]
        linear_velocity_z_norm = (linear_velocity_z / self.v_z_max / 2 + 0.5) * 255
        angular_velocity_norm = (velocity[2] / self.yaw_rate_max_rad / 2 + 0.5) * 255
        # 状态: 水平距离、垂直距离、相对偏航、水平速度、垂直速度、偏航角速度
        self.state_raw = np.array(
            [
                distance,
                relative_pose_z,
                math.degrees(relative_yaw),
                linear_velocity_xy,
                linear_velocity_z,
                math.degrees(velocity[2]),
            ]
        )
        state_norm = np.array(
            [
                distance_norm,
                vertical_distance_norm,
                relative_yaw_norm,
                linear_velocity_norm,
                linear_velocity_z_norm,
                angular_velocity_norm,
            ]
        )
        state_norm = np.clip(state_norm, 0, 255)

        if self.navigation_3d:
            if self.using_velocity_state == False:
                state_norm = state_norm[:3]
        else:
            state_norm = np.array(
                [state_norm[0], state_norm[2], state_norm[3], state_norm[5]]
            )
            if self.using_velocity_state == False:
                state_norm = state_norm[:2]

        self.state_norm = state_norm

        return state_norm

    def _get_relative_yaw(self):
        """
        @description: 获取从当前位置指向目标的相对偏航角（弧度）
        @param {type}
        @return:
        """
        current_position = self.get_position()
        # 获取相对方向角
        relative_pose_x = self.goal_position[0] - current_position[0]
        relative_pose_y = self.goal_position[1] - current_position[1]
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # 获取当前偏航角
        yaw_current = self.get_attitude()[2]

        # 获取偏航误差
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        return yaw_error

    def get_position(self):
        position = self.client.simGetVehiclePose().position
        return [position.x_val, position.y_val, -position.z_val]

    def get_velocity(self):
        states = self.client.getMultirotorState()
        linear_velocity = states.kinematics_estimated.linear_velocity
        angular_velocity = states.kinematics_estimated.angular_velocity

        velocity_xy = math.sqrt(
            pow(linear_velocity.x_val, 2) + pow(linear_velocity.y_val, 2)
        )
        velocity_z = linear_velocity.z_val
        yaw_rate = angular_velocity.z_val

        return [velocity_xy, -velocity_z, yaw_rate]

    def get_attitude(self):
        self.state_current_attitude = self.client.simGetVehiclePose().orientation
        return airsim.to_eularian_angles(self.state_current_attitude)

    def get_attitude_cmd(self):
        return [0.0, 0.0, self.yaw_sp]

    def get_distance_to_goal_2d(self):
        return math.sqrt(
            pow(self.get_position()[0] - self.goal_position[0], 2)
            + pow(self.get_position()[1] - self.goal_position[1], 2)
        )

    def close(self):
        """Release AirSim control so the vehicle does not stay frozen after script exit."""
        try:
            # Ensure simulation is running before releasing control.
            self.client.simPause(False)
        except Exception:
            pass

        try:
            self.client.armDisarm(False)
        except Exception:
            pass

        try:
            self.client.enableApiControl(False)
        except Exception:
            pass
