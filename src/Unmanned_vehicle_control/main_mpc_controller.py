import casadi as ca
import numpy as np
from src.main_help_functions import calculate_lateral_deviation
from src.config import MAX_CONTROL_WHEEL_ANGLE_RAD, MAX_CONTROL_ACCELERATION_M_S_2, MAX_CONTROL_BRAKING_M_S_2, \
    PATH_TOLERANCE_M, FINE_X_COEF, FINE_Y_COEF, FINE_V_COEF, FINE_STEER_COEF, FINE_ACC_COEF, FINE_STEER_DOT_COEF, \
    FINE_ACC_DOT_COEF, FINE_THETA_COEF, FINE_LATERAL_COEF
from src.main_vehicle_model import vehicle_model


class MpcController:
    def __init__(self, horizon=10, dt=0.1):
        self.horizon = horizon
        self.opti = None
        self.dt = dt
        self.sol = None
        self.cost = None
        self.is_success = False
        self.X = None
        self.Y = None
        self.control_buffer = {"acceleration": [0] * self.horizon, "wheel_angle": [0] * self.horizon}
        self.buffer_index = 0

        # 避障相关参数
        self.obstacles = []  # 障碍物列表，每个元素是字典包含位置和半径
        self.obstacle_avoidance_enabled = True
        self.obstacle_safety_distance = 3.0  # 安全距离
        self.obstacle_penalty_weight = 500.0  # 增加障碍物惩罚权重，确保有效避障

    def reset_solver(self):
        self.opti = ca.Opti()
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_cpu_time': self.dt,
            'ipopt.max_iter': 100,  # 增加最大迭代次数，确保找到可行解
            'ipopt.tol': 1e-3,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_slack_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.acceptable_tol': 1e-2,  # 可接受容差
            'ipopt.acceptable_iter': 5,
        }
        self.opti.solver('ipopt', opts)

        # 定义状态和控制变量
        self.X = self.opti.variable(4, self.horizon + 1)
        self.U = self.opti.variable(2, self.horizon)

        # 热启动
        if hasattr(self, 'prev_X') and hasattr(self, 'prev_U'):
            self.opti.set_initial(self.X, self.prev_X)
            self.opti.set_initial(self.U, self.prev_U)

        # 动力学约束
        for k in range(self.horizon):
            self.opti.subject_to(self.X[:, k + 1] == vehicle_model(self.X[:, k], self.U[:, k], self.dt))
            self.opti.subject_to(self.U[0, k] <= MAX_CONTROL_WHEEL_ANGLE_RAD)
            self.opti.subject_to(self.U[0, k] >= -MAX_CONTROL_WHEEL_ANGLE_RAD)
            self.opti.subject_to(self.U[1, k] <= MAX_CONTROL_ACCELERATION_M_S_2)
            self.opti.subject_to(self.U[1, k] >= MAX_CONTROL_BRAKING_M_S_2)

    def set_init_vehicle_state(self, x, y, theta, v):
        self.opti.subject_to(self.X[:, 0] == ca.vertcat(x, y, theta, v))

    def set_obstacles(self, obstacles):
        """设置障碍物信息"""
        self.obstacles = obstacles

    def enable_obstacle_avoidance(self, enable=True):
        """启用/禁用避障功能"""
        self.obstacle_avoidance_enabled = enable

    def update_cost_function(self, x_ref, y_ref, theta_ref, v_ref):
        self.cost = 0

        # 原始跟踪代价
        for k in range(self.horizon):
            fine_x = ca.if_else(ca.fabs(self.X[0, k] - x_ref[k]) > PATH_TOLERANCE_M,
                                FINE_X_COEF * (self.X[0, k] - x_ref[k]) ** 2, 0)
            fine_y = ca.if_else(ca.fabs(self.X[1, k] - y_ref[k]) > PATH_TOLERANCE_M,
                                FINE_Y_COEF * (self.X[1, k] - y_ref[k]) ** 2, 0)
            fine_v = FINE_V_COEF * (self.X[3, k] - v_ref[k]) ** 2
            fine_steer = FINE_STEER_COEF * self.U[0, k] ** 2
            fine_acc = FINE_ACC_COEF * self.U[1, k] ** 2
            fine_theta = FINE_THETA_COEF * (self.X[2, k] - theta_ref[k]) ** 2

            if k < self.horizon - 1:
                lateral_deviation = calculate_lateral_deviation(self.X[0, k], self.X[1, k],
                                                                x_ref[k], y_ref[k],
                                                                x_ref[k + 1], y_ref[k + 1])
            else:
                lateral_deviation = 0

            fine_lateral = FINE_LATERAL_COEF * lateral_deviation ** 2

            fine_steer_dot = 0
            fine_acc_dot = 0
            if k > 0:
                fine_steer_dot += FINE_STEER_DOT_COEF * (self.U[0, k] - self.U[0, k - 1]) ** 2
                fine_acc_dot += FINE_ACC_DOT_COEF * (self.U[1, k] - self.U[1, k - 1]) ** 2

            self.cost += fine_x + fine_y + fine_v + fine_steer + fine_acc + fine_steer_dot + fine_acc_dot + fine_theta + fine_lateral

        # 避障代价
        if self.obstacle_avoidance_enabled and self.obstacles:
            obstacle_cost = self._calculate_obstacle_cost()
            self.cost += obstacle_cost

        self.opti.minimize(self.cost)

    def _calculate_obstacle_cost(self):
        """计算障碍物惩罚项 - 更有效的避障代价函数"""
        obstacle_cost = 0

        for k in range(self.horizon):
            vehicle_x = self.X[0, k]
            vehicle_y = self.X[1, k]

            for obstacle in self.obstacles:
                obs_x = obstacle.get('x', 0)
                obs_y = obstacle.get('y', 0)
                radius = obstacle.get('radius', 1.0)
                safe_distance = obstacle.get('safe_distance', self.obstacle_safety_distance)

                # 计算车辆到障碍物的距离
                dx = vehicle_x - obs_x
                dy = vehicle_y - obs_y
                distance_sq = dx ** 2 + dy ** 2
                distance = ca.sqrt(distance_sq)

                # 障碍物半径加上车辆半径（假设车辆半径约2米）
                total_radius = radius + 2.0

                # 使用强排斥力场：当距离小于安全距离时，施加极大的惩罚
                # 使用反比例函数，距离越近惩罚越大
                penalty_strength = self.obstacle_penalty_weight

                # 计算惩罚项
                if total_radius > 0:
                    # 使用障碍物的实际半径计算
                    normalized_distance = distance / (total_radius + 0.1)  # 加0.1避免除零
                else:
                    normalized_distance = distance

                # 当距离很近时，使用更大的惩罚
                obstacle_penalty = penalty_strength * ca.exp(-normalized_distance / 2.0)

                # 额外的近距离惩罚（当距离小于总半径+安全距离时）
                collision_distance = total_radius + 1.0  # 碰撞距离
                if collision_distance > 0:
                    collision_factor = ca.fmax(0, collision_distance - distance) / collision_distance
                    collision_penalty = 1000.0 * collision_factor ** 2
                    obstacle_penalty += collision_penalty

                # 添加方向引导：如果车辆朝向障碍物运动，增加惩罚
                if k > 0:
                    prev_x = self.X[0, k - 1]
                    prev_y = self.X[1, k - 1]
                    vel_x = vehicle_x - prev_x
                    vel_y = vehicle_y - prev_y

                    # 计算速度方向与到障碍物方向的内积（如果为正表示朝向障碍物运动）
                    vel_mag = ca.sqrt(vel_x ** 2 + vel_y ** 2 + 0.01)
                    dir_to_obs_x = dx / (distance + 0.01)
                    dir_to_obs_y = dy / (distance + 0.01)

                    dot_product = (vel_x / vel_mag) * dir_to_obs_x + (vel_y / vel_mag) * dir_to_obs_y

                    # 如果朝向障碍物运动，增加惩罚
                    approach_penalty = 200.0 * ca.fmax(0, dot_product) * ca.exp(-distance / 5.0)
                    obstacle_penalty += approach_penalty

                obstacle_cost += obstacle_penalty

        return obstacle_cost

    def solve(self):
        try:
            self.sol = self.opti.solve()
            self.is_success = True

            # 保存当前解作为下次的初始猜测
            self.prev_X = self.sol.value(self.X)
            self.prev_U = self.sol.value(self.U)

            wheel_angle_rad, acceleration_m_s_2 = self.get_controls_value()
            self.control_buffer["acceleration"] = self.control_buffer["acceleration"][1:] + [acceleration_m_s_2]
            self.control_buffer["wheel_angle"] = self.control_buffer["wheel_angle"][1:] + [wheel_angle_rad]
            self.buffer_index = 0

        except Exception as e:
            print(f"Error or delay upon MPC solution calculation: {e}")
            print("Previous calculated control value will be used")
            self.is_success = False

    def get_controls_value(self):
        if self.is_success:
            wheel_angle_rad = self.sol.value(self.U[0, 0])
            acceleration_m_s_2 = self.sol.value(self.U[1, 0])

            # 控制量饱和
            wheel_angle_rad = np.clip(wheel_angle_rad,
                                      -MAX_CONTROL_WHEEL_ANGLE_RAD,
                                      MAX_CONTROL_WHEEL_ANGLE_RAD)
            acceleration_m_s_2 = np.clip(acceleration_m_s_2,
                                         MAX_CONTROL_BRAKING_M_S_2,
                                         MAX_CONTROL_ACCELERATION_M_S_2)
        else:
            if self.buffer_index < self.horizon:
                decay = 0.9
                acceleration_m_s_2 = self.control_buffer["acceleration"][self.buffer_index] * decay
                wheel_angle_rad = self.control_buffer["wheel_angle"][self.buffer_index] * decay
                self.buffer_index += 1
            else:
                # 如果没有历史控制量，使用零控制
                acceleration_m_s_2 = 0
                wheel_angle_rad = 0

        return wheel_angle_rad, acceleration_m_s_2

    def get_optimized_cost(self):
        if self.sol:
            return self.sol.value(self.cost)
        return float('inf')