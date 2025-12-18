import casadi as ca

from src.main_help_functions import calculate_lateral_deviation
from src.config import MAX_CONTROL_WHEEL_ANGLE_RAD, MAX_CONTROL_ACCELERATION_M_S_2, MAX_CONTROL_BRAKING_M_S_2, \
    PATH_TOLERANCE_M, FINE_X_COEF, FINE_Y_COEF, FINE_V_COEF, FINE_STEER_COEF, FINE_ACC_COEF, FINE_STEER_DOT_COEF, \
    FINE_ACC_DOT_COEF, FINE_THETA_COEF, FINE_LATERAL_COEF
from src.main_vehicle_model import vehicle_model

class MpcController:
    def __init__(self, horizon = 10, dt = 0.1):
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

    def reset_solver(self):
        self.opti = ca.Opti()
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_cpu_time': self.dt,  # 严格限制求解时间不超过控制周期
            'ipopt.max_iter': 50,  # 减少最大迭代次数（默认300，可根据精度需求降低）
            'ipopt.tol': 1e-3,  # 放宽收敛精度（默认1e-8，实时控制可放宽）
            'ipopt.warm_start_init_point': 'yes',  # 启用热启动，加速收敛
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_slack_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6
        }
        self.opti.solver('ipopt', opts)

        self.X = self.opti.variable(4, self.horizon + 1)  # State: (x, y, theta, v)
        self.U = self.opti.variable(2, self.horizon)  # Control (delta, a)

        for k in range(self.horizon):
            self.opti.subject_to(self.X[:, k + 1] == vehicle_model(self.X[:, k], self.U[:, k], self.dt))
            self.opti.subject_to(self.U[0, k] <= MAX_CONTROL_WHEEL_ANGLE_RAD)
            self.opti.subject_to(self.U[0, k] >= -MAX_CONTROL_WHEEL_ANGLE_RAD)
            self.opti.subject_to(self.U[1, k] <= MAX_CONTROL_ACCELERATION_M_S_2)
            self.opti.subject_to(self.U[1, k] >= MAX_CONTROL_BRAKING_M_S_2)

    def set_init_vehicle_state(self, x, y, theta, v):
        self.opti.subject_to(self.X[:, 0] == ca.vertcat(x, y, theta, v))

    def update_cost_function(self, x_ref, y_ref, theta_ref, v_ref):
        self.cost = 0
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
                lateral_deviation = calculate_lateral_deviation(self.X[0, k], self.X[1, k], x_ref[k], y_ref[k], x_ref[k + 1], y_ref[k + 1])
            else:
                lateral_deviation = 0

            fine_lateral = FINE_LATERAL_COEF * lateral_deviation ** 2

            fine_steer_dot = 0
            fine_acc_dot = 0
            if k > 0:
                fine_steer_dot += FINE_STEER_DOT_COEF * (self.U[0, k] - self.U[0, k - 1]) ** 2
                fine_acc_dot += FINE_ACC_DOT_COEF * (self.U[1, k] - self.U[1, k - 1]) ** 2

            self.cost += fine_x + fine_y + fine_v + fine_steer + fine_acc + fine_steer_dot + fine_acc_dot + fine_theta + fine_lateral

        self.opti.minimize(self.cost)

    def solve(self):
        try:
            self.sol = self.opti.solve()
            self.is_success = True

            wheel_angle_rad, acceleration_m_s_2 = self.get_controls_value()
            self.control_buffer["acceleration"] = self.control_buffer["acceleration"][1:] + [acceleration_m_s_2]
            self.control_buffer["wheel_angle"] = self.control_buffer["wheel_angle"][1:] + [wheel_angle_rad]
            self.buffer_index = 0
        except Exception:
            print("Error or delay upon MPC solution calculation. Previous calculated control value will be used")
            self.is_success = False

    def get_controls_value(self):
        if self.is_success:
            wheel_angle_rad = self.sol.value(self.U[0, 0])
            acceleration_m_s_2 = self.sol.value(self.U[1, 0])
        else:
            if self.buffer_index < self.horizon:
                acceleration_m_s_2 = self.control_buffer["acceleration"][self.buffer_index]
                wheel_angle_rad = self.control_buffer["wheel_angle"][self.buffer_index]
                self.buffer_index += 1
            else:
                raise Exception("Control buffer is empty.")

        return wheel_angle_rad, acceleration_m_s_2

    def get_optimized_cost(self):
        return self.sol.value(self.cost)