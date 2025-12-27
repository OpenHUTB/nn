import time
import numpy as np
from src.main_carla_simulator import CarlaSimulator
from src.config import N, dt, V_REF, LAPS
from src.main_help_functions import get_ref_trajectory, update_reference_point
from src.main_logger import Logger
from src.main_mpc_controller import MpcController

carla = CarlaSimulator()
carla.load_world('Town01')

# 获取地图的生成点
spawn_points = carla.world.get_map().get_spawn_points()
if not spawn_points:
    print("Warning: No spawn points found in the map!")
    raise RuntimeError("No spawn points available")

# 生成主车辆
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

# ================ 生成间距合适的障碍物 ================
print("\n" + "=" * 50)
print("Generating spaced obstacles...")
print("=" * 50)

# 生成间距合适的障碍物
blocking_obstacles = carla.spawn_spaced_obstacles()

# 设置spectator位置 - 调整视角以更好观察障碍物
if carla.ego_vehicle:
    vehicle_location = carla.ego_vehicle.get_location()
    vehicle_transform = carla.ego_vehicle.get_transform()

    # 设置spectator在车辆后上方，以便观察前方障碍物
    camera_distance = 25.0  # 相机距离
    camera_height = 15.0  # 相机高度
    camera_pitch = -25.0  # 相机俯仰角

    # 计算相机位置（在车辆后方）
    yaw_rad = np.radians(vehicle_transform.rotation.yaw)
    camera_x = vehicle_location.x - camera_distance * np.cos(yaw_rad)
    camera_y = vehicle_location.y - camera_distance * np.sin(yaw_rad)
    camera_z = vehicle_location.z + camera_height

    carla.set_spectator(
        x=camera_x,
        y=camera_y,
        z=camera_z,
        pitch=camera_pitch,
        yaw=vehicle_transform.rotation.yaw,
        roll=0
    )

    print(f"Spectator set to: ({camera_x:.1f}, {camera_y:.1f}, {camera_z:.1f})")

logger = Logger()


# ================ 使用CARLA Waypoint生成轨迹 ================
def generate_road_trajectory(carla_simulator, distance_ahead=100.0, waypoint_interval=2.0):
    """使用CARLA的Waypoint系统生成沿着道路的轨迹"""
    if not carla_simulator.ego_vehicle:
        raise ValueError("Vehicle not spawned yet")

    vehicle_location = carla_simulator.ego_vehicle.get_location()
    carla_map = carla_simulator.world.get_map()
    current_waypoint = carla_map.get_waypoint(vehicle_location)

    if not current_waypoint:
        raise ValueError("Cannot get waypoint at vehicle location")

    # 收集前方的waypoints
    waypoints = [current_waypoint]
    distance_traveled = 0.0

    while distance_traveled < distance_ahead:
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
        v_ref.append(V_REF)

        yaw_deg = transform.rotation.yaw
        theta_traj.append(np.deg2rad(yaw_deg))

    return np.array(x_traj), np.array(y_traj), v_ref, theta_traj


# 生成沿着道路的轨迹
print("\n" + "=" * 50)
print("Generating road trajectory...")
print("=" * 50)

try:
    x_traj, y_traj, v_ref, theta_traj = generate_road_trajectory(
        carla_simulator=carla,
        distance_ahead=100.0,  # 减小距离，以便更集中观察避障行为
        waypoint_interval=2.0
    )
    print(f"Trajectory generated with {len(x_traj)} points")

    # 打印轨迹点信息
    print(f"First 5 trajectory points:")
    for i in range(min(5, len(x_traj))):
        print(f"  Point {i}: ({x_traj[i]:.1f}, {y_traj[i]:.1f}), theta={np.degrees(theta_traj[i]):.1f}°")

except Exception as e:
    print(f"Failed to generate road trajectory: {e}")
    # 如果生成失败，使用简单的直线轨迹作为后备
    print("Using straight trajectory as fallback...")
    if carla.ego_vehicle:
        vehicle_location = carla.ego_vehicle.get_location()
        vehicle_transform = carla.ego_vehicle.get_transform()
        yaw_rad = np.radians(vehicle_transform.rotation.yaw)

        # 生成直线轨迹
        x_traj = np.array([vehicle_location.x + i * 2.0 * np.cos(yaw_rad) for i in range(100)])
        y_traj = np.array([vehicle_location.y + i * 2.0 * np.sin(yaw_rad) for i in range(100)])
    else:
        x_traj = np.array([i * 2.0 for i in range(100)])
        y_traj = np.array([0.0 for i in range(100)])
    v_ref = [V_REF] * 100
    theta_traj = [yaw_rad] * 100

current_idx = 0
laps = 0


# ================ 碰撞检测功能 ================
def check_collision(carla_simulator):
    """检查车辆是否发生碰撞"""
    if not carla_simulator.ego_vehicle:
        return False, "no_vehicle"

    collision_sensor = None

    try:
        blueprint_library = carla_simulator.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        collision_sensor = carla_simulator.world.spawn_actor(
            collision_bp,
            collision_transform,
            attach_to=carla_simulator.ego_vehicle
        )

        collision_detected = [False]
        collision_actor = [None]

        def on_collision(event):
            collision_detected[0] = True
            collision_actor[0] = event.other_actor
            print(f"Collision detected with {event.other_actor.type_id}")

        collision_sensor.listen(on_collision)
        time.sleep(0.05)  # 减小等待时间

        if collision_sensor:
            collision_sensor.destroy()

        return collision_detected[0], collision_actor[0].type_id if collision_actor[0] else "unknown"

    except Exception as e:
        print(f"Error in collision detection: {e}")
        if collision_sensor:
            collision_sensor.destroy()
        return False, "error"


# ================ MPC控制器初始化 ================
print("\n" + "=" * 50)
print("Initializing MPC controller...")
print("=" * 50)

mpc_controller = MpcController(horizon=N, dt=dt)

# 启用避障功能
mpc_controller.enable_obstacle_avoidance(True)

# 降低速度以减少碰撞风险，同时确保能够有效避障
SAFE_V_REF = 3.0  # 稍微提高速度，因为障碍物间距增大了

print(f"MPC Horizon: {N}")
print(f"Control dt: {dt}")
print(f"Safe speed: {SAFE_V_REF} m/s")
print(f"Obstacle avoidance: Enabled")

try:
    consecutive_collisions = 0
    max_consecutive_collisions = 3
    simulation_steps = 0
    max_simulation_steps = 500  # 限制模拟步数

    print("\n" + "=" * 50)
    print("Starting simulation with obstacle avoidance...")
    print("=" * 50)

    while simulation_steps < max_simulation_steps:
        simulation_steps += 1
        start_time = time.time()

        # 检查碰撞
        try:
            collision_detected, collision_type = check_collision(carla)
            if collision_detected:
                consecutive_collisions += 1
                print(f"COLLISION WARNING #{consecutive_collisions}: Collision with {collision_type}")

                if consecutive_collisions >= max_consecutive_collisions:
                    print("Too many consecutive collisions. Stopping simulation.")
                    break

                # 碰撞后短暂停止
                carla.apply_control(0.0, 0.0, 1.0)
                time.sleep(0.5)
                continue
            else:
                consecutive_collisions = 0
        except Exception as e:
            print(f"Error checking collision: {e}")

        # 获取障碍物信息并传递给MPC控制器
        try:
            obstacles_info = carla.get_obstacle_positions()
            mpc_controller.set_obstacles(obstacles_info)

            # 打印障碍物信息
            if obstacles_info and simulation_steps % 30 == 0:
                print(f"\nCurrent obstacles ({len(obstacles_info)} total):")
                for i, obs in enumerate(obstacles_info):
                    obs_type = obs['type'].split('.')[-1]
                    distance_to_vehicle = np.sqrt((obs['x'] - x0) ** 2 + (obs['y'] - y0) ** 2)
                    print(f"  Obstacle {i + 1}: {obs_type} at ({obs['x']:.1f}, {obs['y']:.1f}), "
                          f"distance: {distance_to_vehicle:.1f}m, safe_dist: {obs.get('safe_distance', 3.0):.1f}m")

        except Exception as e:
            print(f"Error getting obstacle info: {e}")
            obstacles_info = []

        # 重新初始化求解器
        mpc_controller.reset_solver()

        # 获取车辆状态
        x0, y0, theta0, v0 = carla.get_main_ego_vehicle_state()

        # 打印车辆状态
        if simulation_steps % 15 == 0:
            print(f"\nVehicle state: Position=({x0:.1f}, {y0:.1f}), "
                  f"Theta={np.degrees(theta0):.1f}°, Speed={v0:.1f}m/s")

        mpc_controller.set_init_vehicle_state(x0, y0, theta0, v0)

        # 获取参考轨迹
        try:
            x_ref, y_ref, theta_ref = get_ref_trajectory(x_traj, y_traj, theta_traj, current_idx)
        except Exception as e:
            print(f"Error getting reference trajectory: {e}")
            break

        # 使用感知规划绘制，包括障碍物
        try:
            carla.draw_perception_planning(x_ref, y_ref, current_idx=0,
                                           look_ahead_points=N,
                                           obstacles_info=obstacles_info)
        except Exception as e:
            print(f"Error drawing perception planning: {e}")

        # 记录日志
        logger.log_controller_input(x0, y0, v0, theta0, x_ref[0], y_ref[0], SAFE_V_REF, theta_ref[0])

        try:
            # 更新代价函数
            mpc_controller.update_cost_function(x_ref, y_ref, theta_ref, [SAFE_V_REF] * len(x_ref))

            # 求解MPC
            mpc_controller.solve()

            # 获取控制量
            wheel_angle_rad, acceleration_m_s_2 = mpc_controller.get_controls_value()

            # 限制控制量范围
            wheel_angle_rad = max(min(wheel_angle_rad, 0.8), -0.8)  # 增大转向限制，确保能避开障碍物
            acceleration_m_s_2 = max(min(acceleration_m_s_2, 3.0), -3.0)  # 增大加速度限制

            # 处理控制输入
            throttle, brake, steer = CarlaSimulator.process_control_inputs(wheel_angle_rad, acceleration_m_s_2)

            # 记录并应用控制
            logger.log_controller_output(steer, throttle, brake)
            carla.apply_control(steer, throttle, brake)

        except Exception as e:
            print(f"Error in MPC control: {e}")
            # 发生错误时应用零控制
            carla.apply_control(0.0, 0.0, 0.5)  # 轻微刹车

        # 更新参考点
        try:
            prev_current_idx = current_idx
            current_idx = update_reference_point(x0, y0, current_idx, x_traj, y_traj, min_distance=2.0)
        except Exception as e:
            print(f"Error updating reference point: {e}")
            current_idx = (current_idx + 1) % len(x_traj)

        # 如果到达轨迹终点，重新生成轨迹
        if current_idx >= len(x_traj) - N:
            print("Reached end of trajectory. Regenerating...")
            try:
                x_traj, y_traj, v_ref, theta_traj = generate_road_trajectory(
                    carla_simulator=carla,
                    distance_ahead=100.0,
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

        # 显示调试信息
        if simulation_steps % 8 == 0:
            print(f"Step {simulation_steps}: MPC={mpc_calculation_time:.3f}s | "
                  f"Position=({x0:.1f}, {y0:.1f}) | "
                  f"Speed={v0:.1f}m/s | "
                  f"Controls: steer={steer:.2f}, throttle={throttle:.2f}, brake={brake:.2f} | "
                  f"Obstacles={len(obstacles_info)} | "
                  f"Ref idx={current_idx}/{len(x_traj)}")

        time.sleep(max(dt - mpc_calculation_time, 0))

        # 退出条件
        if laps == LAPS:
            print(f"Completed {laps} lap(s). Stopping simulation.")
            break

        # 安全停止条件
        if len(x_traj) > current_idx:
            distance_to_ref = np.sqrt((x0 - x_traj[current_idx]) ** 2 + (y0 - y_traj[current_idx]) ** 2)
            if distance_to_ref > 30.0:  # 增大偏离阈值，因为车辆可能需要绕行
                print(f"Vehicle deviated too far from reference ({distance_to_ref:.1f}m). Stopping.")
                break

except KeyboardInterrupt:
    print("\nSimulation interrupted by user")
except Exception as e:
    print(f"\nUnexpected error: {e}")
    import traceback

    traceback.print_exc()
finally:
    print("\n" + "=" * 50)
    print("Simulation completed")
    print("=" * 50)

    # 显示最终统计信息
    if carla.ego_vehicle:
        final_x, final_y, final_theta, final_v = carla.get_main_ego_vehicle_state()
        print(f"Final position: ({final_x:.1f}, {final_y:.1f})")
        print(f"Final speed: {final_v:.1f} m/s")
        print(f"Total steps: {simulation_steps}")
        print(f"Collisions: {consecutive_collisions}")

    carla.clean()
    logger.show_plots()

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
#后面
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