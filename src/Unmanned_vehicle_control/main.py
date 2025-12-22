import time
import numpy as np
from src.main_carla_simulator import CarlaSimulator
from src.config import N, dt, V_REF, LAPS
from src.main_help_functions import get_ref_trajectory, update_reference_point
from src.main_logger import Logger
from src.main_mpc_controller import MpcController

"""
# 注释掉的"8"字形轨迹场景
def draw_trajectory_in_thread(carla, x_traj, y_traj, dt):
    while True:
        carla.draw_trajectory(x_traj, y_traj, height=0.2, green=255, life_time=dt * 2)
        time.sleep(dt)

carla = CarlaSimulator()
carla.load_world('Town02_Opt')
carla.spawn_ego_vehicle('vehicle.tesla.model3', x=X_INIT_M, y=Y_INIT_M, z=0.1)  # "8"字形
carla.print_ego_vehicle_characteristics()
carla.set_spectator(X_INIT_M, Y_INIT_M, z=50, pitch=-90)

logger = Logger()
x_traj, y_traj, v_ref, theta_traj = get_eight_trajectory(X_INIT_M, Y_INIT_M)  # "8"形状轨迹
current_idx = 0
laps = 0

trajectory_thread = threading.Thread(target=draw_trajectory_in_thread, args=(carla, x_traj, y_traj, dt))
trajectory_thread.daemon = True
trajectory_thread.start()
"""

# ================ 新的道路场景 - 使用CARLA Waypoint系统 ================

# 初始化CARLA模拟器
carla = CarlaSimulator()

# 加载一个有道路的地图
carla.load_world('Town01')  # 使用Town01地图，它有清晰的道路网络

# 获取地图的生成点
spawn_points = carla.world.get_map().get_spawn_points()
if not spawn_points:
    print("Warning: No spawn points found in the map!")
    raise RuntimeError("No spawn points available")

# 尝试多个生成点直到成功
vehicle_spawned = False
for i, spawn_point in enumerate(spawn_points):
    try:
        print(f"Attempting to spawn vehicle at spawn point {i + 1}/{len(spawn_points)}...")

        carla.spawn_ego_vehicle(
            'vehicle.tesla.model3',
            x=spawn_point.location.x,
            y=spawn_point.location.y,
            z=spawn_point.location.z + 0.1,
            pitch=spawn_point.rotation.pitch,
            yaw=spawn_point.rotation.yaw,
            roll=spawn_point.rotation.roll
        )

        vehicle_spawned = True
        print(f"Successfully spawned vehicle at spawn point {i + 1}")
        break
    except Exception as e:
        print(f"Failed to spawn at spawn point {i + 1}: {e}")
        continue

if not vehicle_spawned:
    raise RuntimeError("Failed to spawn vehicle at any spawn point")

carla.print_ego_vehicle_characteristics()

# 设置spectator位置
vehicle_location = carla.ego_vehicle.get_location()
carla.set_spectator(
    x=vehicle_location.x - 15,  # 在车辆后方15米
    y=vehicle_location.y,
    z=12,  # 高度12米
    pitch=-25,  # 稍微向下看
    yaw=0  # 看向车辆前方
)

logger = Logger()


# ================ 使用CARLA Waypoint生成轨迹 ================

def generate_road_trajectory(carla_simulator, distance_ahead=100.0, waypoint_interval=2.0):
    """
    使用CARLA的Waypoint系统生成沿着道路的轨迹

    参数:
        carla_simulator: CarlaSimulator实例
        distance_ahead: 向前看的距离（米）
        waypoint_interval: waypoint之间的间隔（米）
    """
    if not carla_simulator.ego_vehicle:
        raise ValueError("Vehicle not spawned yet")

    # 获取当前车辆的waypoint
    vehicle_location = carla_simulator.ego_vehicle.get_location()
    carla_map = carla_simulator.world.get_map()
    current_waypoint = carla_map.get_waypoint(vehicle_location)

    if not current_waypoint:
        raise ValueError("Cannot get waypoint at vehicle location")

    # 收集前方的waypoints
    waypoints = [current_waypoint]
    distance_traveled = 0.0

    while distance_traveled < distance_ahead:
        # 获取下一个waypoint
        next_waypoints = current_waypoint.next(waypoint_interval)

        if not next_waypoints:
            print(f"Warning: No more waypoints after {distance_traveled} meters")
            break

        # 选择第一个（最直接的道路方向）
        current_waypoint = next_waypoints[0]
        waypoints.append(current_waypoint)
        distance_traveled += waypoint_interval

    print(f"Generated {len(waypoints)} waypoints for trajectory")

    # 提取轨迹点
    x_traj = []
    y_traj = []
    v_ref = []
    theta_traj = []

    for i, wp in enumerate(waypoints):
        transform = wp.transform
        location = transform.location

        x_traj.append(location.x)
        y_traj.append(location.y)
        v_ref.append(V_REF)  # 使用参考速度

        # 计算航向角（从yaw角度转换）
        yaw_deg = transform.rotation.yaw
        theta_traj.append(np.deg2rad(yaw_deg))

    return np.array(x_traj), np.array(y_traj), v_ref, theta_traj


# 生成沿着道路的轨迹
try:
    x_traj, y_traj, v_ref, theta_traj = generate_road_trajectory(
        carla_simulator=carla,
        distance_ahead=200.0,  # 生成200米长的轨迹
        waypoint_interval=2.0  # 每2米一个waypoint
    )
    print(f"Trajectory generated with {len(x_traj)} points")
except Exception as e:
    print(f"Failed to generate road trajectory: {e}")
    # 如果生成失败，使用简单的直线轨迹作为后备
    print("Using straight trajectory as fallback...")
    vehicle_location = carla.ego_vehicle.get_location()
    # 生成简单的直线轨迹
    x_traj = np.array([vehicle_location.x + i * 2.0 for i in range(100)])
    y_traj = np.array([vehicle_location.y for i in range(100)])  # 直线
    v_ref = [V_REF] * 100
    theta_traj = [0.0] * 100  # 朝向正东方向

current_idx = 0
laps = 0


# ================ 添加碰撞检测功能 ================

def check_collision(carla_simulator):
    """
    检查车辆是否发生碰撞

    返回:
        bool: 是否发生碰撞
        str: 碰撞对象类型（如果发生碰撞）
    """
    collision_sensor = None

    try:
        # 创建碰撞传感器
        blueprint_library = carla_simulator.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')

        # 将传感器附加到车辆
        collision_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        collision_sensor = carla_simulator.world.spawn_actor(
            collision_bp,
            collision_transform,
            attach_to=carla_simulator.ego_vehicle
        )

        # 设置碰撞事件处理器
        collision_detected = [False]
        collision_actor = [None]

        def on_collision(event):
            collision_detected[0] = True
            collision_actor[0] = event.other_actor
            print(f"Collision detected with {event.other_actor.type_id}")

        collision_sensor.listen(on_collision)

        # 等待一小段时间检查碰撞
        time.sleep(0.1)

        # 销毁传感器
        if collision_sensor:
            collision_sensor.destroy()

        return collision_detected[0], collision_actor[0].type_id if collision_actor[0] else "unknown"

    except Exception as e:
        print(f"Error in collision detection: {e}")
        if collision_sensor:
            collision_sensor.destroy()
        return False, "error"


# ================ MPC控制器初始化 ================

mpc_controller = MpcController(horizon=N, dt=dt)

# 降低速度以减少碰撞风险
SAFE_V_REF = 3.0  # 降低到3 m/s (约10.8 km/h)

try:
    consecutive_collisions = 0
    max_consecutive_collisions = 3

    while True:
        start_time = time.time()

        # 检查碰撞
        collision_detected, collision_type = check_collision(carla)
        if collision_detected:
            consecutive_collisions += 1
            print(f"COLLISION WARNING #{consecutive_collisions}: Collision with {collision_type}")

            if consecutive_collisions >= max_consecutive_collisions:
                print("Too many consecutive collisions. Stopping simulation.")
                break

            # 碰撞后短暂停止
            carla.apply_control(0.0, 0.0, 1.0)  # 刹车
            time.sleep(0.5)
            continue
        else:
            consecutive_collisions = 0

        mpc_controller.reset_solver()

        x0, y0, theta0, v0 = carla.get_main_ego_vehicle_state()
        mpc_controller.set_init_vehicle_state(x0, y0, theta0, v0)

        x_ref, y_ref, theta_ref = get_ref_trajectory(x_traj, y_traj, theta_traj, current_idx)

        # 使用感知规划绘制，只显示前方轨迹
        carla.draw_perception_planning(x_ref, y_ref, current_idx=0, look_ahead_points=N)

        logger.log_controller_input(x0, y0, v0, theta0, x_ref[0], y_ref[0], SAFE_V_REF, theta_ref[0])
        mpc_controller.update_cost_function(x_ref, y_ref, theta_ref, [SAFE_V_REF] * len(x_ref))

        mpc_controller.solve()

        wheel_angle_rad, acceleration_m_s_2 = mpc_controller.get_controls_value()

        # 限制控制量范围，避免极端值导致错误
        wheel_angle_rad = max(min(wheel_angle_rad, 0.5), -0.5)
        acceleration_m_s_2 = max(min(acceleration_m_s_2, 2.0), -2.0)  # 进一步限制加速度

        throttle, brake, steer = CarlaSimulator.process_control_inputs(wheel_angle_rad, acceleration_m_s_2)
        logger.log_controller_output(steer, throttle, brake)
        carla.apply_control(steer, throttle, brake)

        prev_current_idx = current_idx
        current_idx = update_reference_point(x0, y0, current_idx, x_traj, y_traj, min_distance=3.0)  # 减小最小距离

        # 如果到达轨迹终点，重新生成轨迹
        if current_idx >= len(x_traj) - N:
            print("Reached end of trajectory. Regenerating...")
            try:
                x_traj, y_traj, v_ref, theta_traj = generate_road_trajectory(
                    carla_simulator=carla,
                    distance_ahead=200.0,
                    waypoint_interval=2.0
                )
                current_idx = 0
                print(f"New trajectory generated with {len(x_traj)} points")
            except Exception as e:
                print(f"Failed to regenerate trajectory: {e}")
                break

        if prev_current_idx == len(x_traj) - 1 and current_idx == 0:
            laps += 1

        end_time = time.time()
        mpc_calculation_time = end_time - start_time

        # 显示更多调试信息
        print(f"MPC calculation: {mpc_calculation_time:.3f}s | "
              f"Position: ({x0:.1f}, {y0:.1f}) | "
              f"Speed: {v0:.1f}m/s | "
              f"Controls: steer={steer:.2f}, throttle={throttle:.2f}, brake={brake:.2f}")

        time.sleep(max(dt - mpc_calculation_time, 0))

        # 退出条件
        if laps == LAPS:
            print(f"Completed {laps} lap(s). Stopping simulation.")
            break

        # 安全停止条件：如果车辆偏离轨迹太远
        if len(x_traj) > current_idx:
            distance_to_ref = np.sqrt((x0 - x_traj[current_idx]) ** 2 + (y0 - y_traj[current_idx]) ** 2)
            if distance_to_ref > 10.0:  # 如果偏离超过10米
                print(f"Vehicle deviated too far from reference ({distance_to_ref:.1f}m). Stopping.")
                break

finally:
    carla.clean()
    logger.show_plots()

"""
# 注释掉的其他轨迹选项
# 圆形轨迹参数：圆心（X_INIT_M, Y_INIT_M），半径20米，200个点
x_traj, y_traj, v_ref, theta_traj = get_circle_trajectory()  # 一行生成轨迹
init_x, init_y = x_traj[0], y_traj[0]                       # 2行获取初始位置
init_yaw = np.rad2deg(theta_traj[0])                        # 1行获取初始角度
carla.spawn_ego_vehicle(
    'vehicle.tesla.model3',
    x=init_x,
    y=init_y,
    z=0.1,
    yaw=init_yaw  # 初始方向与轨迹一致
)

carla.print_ego_vehicle_characteristics()
# 调整 spectator 位置以便更好观察圆形轨迹
carla.set_spectator(X_INIT_M, Y_INIT_M, z=80, pitch=-90)  # 从圆心正上方俯视
"""
"""
#螺旋轨迹
x_traj, y_traj, v_ref, theta_traj = get_spiral_trajectory(
    x_init=X_INIT_M,
    y_init=Y_INIT_M,
    turns=2,  # 螺旋圈数
    scale=2   # 螺旋缩放因子
)  # 一行生成螺旋轨迹

init_x, init_y = x_traj[0], y_traj[0]                       # 获取初始位置
init_yaw = np.rad2deg(theta_traj[0])                        # 获取初始角度
carla.spawn_ego_vehicle(
    'vehicle.tesla.model3',
    x=init_x,
    y=init_y,
    z=0.1,
    yaw=init_yaw  # 初始方向与轨迹一致
)

# 调整 spectator 位置以便观察螺旋轨迹
carla.set_spectator(X_INIT_M, Y_INIT_M, z=50, pitch=-90)  # 降低高度，适应缩小的轨迹
"""
"""
# 使用新的方形轨迹
x_traj, y_traj, v_ref, theta_traj = get_square_trajectory(
    x_init=X_INIT_M,
    y_init=Y_INIT_M,
    side_length=23,  # 方形边长
    total_points=200  # 轨迹点数量
)

# 设置初始位置和方向
init_x, init_y = x_traj[0], y_traj[0]
init_yaw = np.rad2deg(theta_traj[0])
carla.spawn_ego_vehicle(
    'vehicle.tesla.model3',
    x=init_x,
    y=init_y,
    z=0.1,
    yaw=init_yaw  # 初始方向与轨迹一致
)

# 调整 spectator 位置以便观察方形轨迹
carla.set_spectator(X_INIT_M + 20, Y_INIT_M + 20, z=60, pitch=-90)  # 从方形中心上方俯视
"""