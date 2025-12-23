import numpy as np

X_INIT_M = 210
Y_INIT_M = 362
A = 25
LAPS = 1

MAX_ACCELERATION_M_S_2 = 10
MAX_BRAKING_M_S_2 = -4.1
MAX_WHEEL_ANGLE_RAD = 70 / 180 * np.pi
L = 2.875

### MPC settings
N = 5
dt = 0.1
MAX_CONTROL_WHEEL_ANGLE_RAD = 70 / 180 * np.pi
MAX_CONTROL_ACCELERATION_M_S_2 = 10
MAX_CONTROL_BRAKING_M_S_2 = -4.1
PATH_TOLERANCE_M = 0.5
V_REF = 5

"""
原始代码被注释掉：
FINE_X_COEF = 10
FINE_Y_COEF = 10
FINE_STEER_COEF = 0
FINE_ACC_COEF = 0
FINE_STEER_DOT_COEF = 100
FINE_ACC_DOT_COEF = 1
FINE_V_COEF = 20
FINE_THETA_COEF = 0
FINE_LATERAL_COEF = 40
"""

FINE_X_COEF = 10
FINE_Y_COEF = 10
FINE_STEER_COEF = 0
FINE_ACC_COEF = 0
FINE_STEER_DOT_COEF = 100
FINE_ACC_DOT_COEF = 1
FINE_V_COEF = 20
FINE_THETA_COEF = 0
FINE_LATERAL_COEF = 40

# 避障相关参数
OBSTACLE_PENALTY_WEIGHT = 100.0
OBSTACLE_SAFETY_DISTANCE = 3.0
OBSTACLE_AVOIDANCE_ENABLED = True

# 圆形轨迹参数
CIRCLE_CENTER_X = X_INIT_M
CIRCLE_CENTER_Y = Y_INIT_M
CIRCLE_RADIUS = 50

VISUALIZATION_CONFIG = {
    'show_full_trajectory': False,
    'show_perception_planning': True,
    'show_frenet_frame': False,
    'look_ahead_points': N,
    'target_point_color': (255, 0, 0),
    'trajectory_color': (0, 255, 0),
    'obstacle_color': (255, 0, 0),
    'safety_zone_color': (255, 165, 0),
}
"""
X_INIT_M = 210 # Start X coordinate for vehicle
Y_INIT_M = 362 # Start Y coordinate for vehicle
A = 25 # Size of 8-trajectory
LAPS = 1 #Laps of simulation

MAX_ACCELERATION_M_S_2 = 10 # Vehicle characteristics (from CARLA simulator, see get_physics_control())
MAX_BRAKING_M_S_2 = -4.1 # Vehicle characteristics (from CARLA simulator, see get_physics_control())
MAX_WHEEL_ANGLE_RAD = 70 / 180 * np.pi # Vehicle characteristics (from CARLA simulator, see get_physics_control())
L = 2.875  #The wheelbase length of the vehicle (meters)

### MPC settings
N = 5 # Horizon of planning for MPC controller
dt = 0.1 #Sample time of control
MAX_CONTROL_WHEEL_ANGLE_RAD = 70 / 180 * np.pi # Maximum value which can use MPC controller in terms of steering
MAX_CONTROL_ACCELERATION_M_S_2 = 10 # Maximum value which can use MPC controller in terms of acceleration
MAX_CONTROL_BRAKING_M_S_2 = -4.1 # Maximum value which can use MPC controller in terms of braking
PATH_TOLERANCE_M = 0.5 # Acceptance radius of each waypoint
V_REF = 5 #Speed of vehicle

FINE_X_COEF = 10 #Fine for deviation from reference X coordinate
FINE_Y_COEF = 10 #Fine for deviation from reference Y coordinate
FINE_STEER_COEF = 0 #Fine for use high value of steering
FINE_ACC_COEF = 0 #Fine for use high values of acceleration
FINE_STEER_DOT_COEF = 100 #Fine for rapid steering
FINE_ACC_DOT_COEF = 1 #Fine for rapid changing of acceleration
FINE_V_COEF = 20 #Fine for deviation from reference speed
FINE_THETA_COEF = 0 #Fine for deviation from reference theta
FINE_LATERAL_COEF = 40 #Fine for deviation from path line
# 圆形轨迹参数
CIRCLE_CENTER_X = X_INIT_M  # 圆心X坐标
CIRCLE_CENTER_Y = Y_INIT_M  # 圆心Y坐标
CIRCLE_RADIUS = 50  # 圆半径

#添加
VISUALIZATION_CONFIG = {
    'show_full_trajectory': False,  # 是否显示完整轨迹
    'show_perception_planning': True,  # 是否显示感知规划信息
    'show_frenet_frame': False,  # 是否显示Frenet坐标系
    'look_ahead_points': N,  # 向前看的点数
    'target_point_color': (255, 0, 0),  # 目标点颜色 (R, G, B)
    'trajectory_color': (0, 255, 0),  # 轨迹线颜色
}
"""