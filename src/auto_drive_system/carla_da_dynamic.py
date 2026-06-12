import carla
import time
import random
import numpy as np  # 引入numpy处理向量运算，比math库更高效

# --- 全局配置 ---
TARGET_FPS = 60
DT = 1.0 / TARGET_FPS
AVOIDANCE_SPEED = 15.0

# --- 全局状态变量 ---
actor_list = []
collision_flag = False
is_avoiding = False
avoidance_stage = 0  # 0:正常, 1:变道, 2:直行超车, 3:回归
target_waypoint = None
original_lane_id = None
stuck_timer = 0
super_timer = 0  # 需要在外部定义以便在函数间共享

# --- 传感器回调 ---
def on_collision(event):
    global collision_flag
    if not collision_flag:
        # 格式化输出，保留原始逻辑
        impulse = np.linalg.norm(event.normal_impulse)
        print(f"⚠️ 碰撞! 强度: {impulse:.2f}")
        collision_flag = True
        if vehicle: vehicle.apply_control(carla.VehicleControl(brake=1.0))

def on_lane_invasion(event):
    if not is_avoiding:
        lane_types = [x.lane_type for x in event.crossed_lane_markings]
        print(f"⚠️ 意外偏离车道! 类型: {lane_types}")

# --- 核心算法 ---
def pure_pursuit(target_loc, vehicle_transform, lookahead_distance=10.0):
    """
    优化的纯追踪算法 (使用Numpy)
    """
    L = 2.875  # 轴距
    v_loc = vehicle_transform.location
    v_rot = vehicle_transform.rotation
    yaw = np.radians(v_rot.yaw)

    # 1. 计算后轴中心
    rear_x = v_loc.x - (L / 2) * np.cos(yaw)
    rear_y = v_loc.y - (L / 2) * np.sin(yaw)

    # 2. 计算目标向量
    dx = target_loc.x - rear_x
    dy = target_loc.y - rear_y
    ld = np.hypot(dx, dy) # 等效于 sqrt(x^2 + y^2)

    if ld < 0.5:
        return 0.0

    # 3. 计算夹角 (使用numpy的arctan2)
    alpha = np.arctan2(dy, dx) - yaw
    # 归一化角度
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

    # 4. 计算曲率
    delta = np.arctan2(2 * L * np.sin(alpha), ld)
    steer = np.degrees(delta) / 90.0

    # 死区处理
    if abs(steer) < 0.05:
        steer = 0.0

    return np.clip(steer, -1.0, 1.0)

def spawn_obstacles_ahead(world, blueprint_library, vehicle, count=2, distance=25.0):
    """
    强制在主车正前方生成障碍物
    """
    obstacles = []
    v_trans = vehicle.get_transform()
    v_loc = v_trans.location
    v_forward = v_trans.get_forward_vector()

    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    map = world.get_map()

    for i in range(count):
        # 计算生成位置
        offset_dist = distance + (i * 6.0)
        spawn_x = v_loc.x + v_forward.x * offset_dist
        spawn_y = v_loc.y + v_forward.y * offset_dist
        spawn_z = v_loc.z + 0.5

        spawn_location = carla.Location(spawn_x, spawn_y, spawn_z)

        # 吸附到车道
        waypoint = map.get_waypoint(spawn_location, project_to_road=True)
        final_transform = carla.Transform(spawn_location, v_trans.rotation)
        
        if waypoint:
            final_transform = carla.Transform(waypoint.transform.location, waypoint.transform.rotation)

        try:
            obs = world.try_spawn_actor(vehicle_bp, final_transform)
            if obs:
                obs.set_autopilot(False)
                obs.apply_control(carla.VehicleControl(brake=1.0))
                actor_list.append(obs)
                obstacles.append(obs)
                print(f"✅ 障碍物 {i+1} 已生成在正前方 {offset_dist:.1f}米处")
        except Exception as e:
            print(f"❌ 障碍物生成失败: {e}")

    return obstacles

def update_camera(world, vehicle):
    """
    封装视角控制逻辑
    """
    v_trans = vehicle.get_transform()
    v_loc = v_trans.location
    v_rot = v_rot = v_trans.rotation

    # 目标位置
    target_cam_loc = v_loc + v_trans.get_forward_vector() * -8.0
    target_cam_loc.z += 4.0

    # 获取当前状态
    current_spec = world.get_spectator().get_transform()
    
    # 线性插值 (Lerp)
    smooth_x = current_spec.location.x + (target_cam_loc.x - current_spec.location.x) * 0.1
    smooth_y = current_spec.location.y + (target_cam_loc.y - current_spec.location.y) * 0.1
    smooth_z = current_spec.location.z + (target_cam_loc.z - current_spec.location.z) * 0.1

    world.get_spectator().set_transform(carla.Transform(
        carla.Location(smooth_x, smooth_y, smooth_z),
        carla.Rotation(pitch=-15, yaw=v_rot.yaw)
    ))

def handle_avoidance_state(vehicle, map, current_waypoint, v_trans):
    """
    状态机逻辑封装
    """
    global is_avoiding, avoidance_stage, target_waypoint, original_lane_id, stuck_timer, super_timer

    velocity = vehicle.get_velocity()
    speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

    if not is_avoiding:
        # --- 阶段 0: 正常行驶 ---
        if speed < 1.5:
            stuck_timer += 1
        else:
            stuck_timer = 0

        if stuck_timer > 15:
            print("🚧 检测到前方障碍，准备变道...")
            original_lane_id = current_waypoint.lane_id

            # 寻找相邻车道
            target_waypoint = current_waypoint.get_left_lane()
            direction = "左"
            if not target_waypoint or target_waypoint.lane_type != carla.LaneType.Driving:
                target_waypoint = current_waypoint.get_right_lane()
                direction = "右"

            if target_waypoint and target_waypoint.lane_type == carla.LaneType.Driving:
                is_avoiding = True
                avoidance_stage = 1
                vehicle.set_autopilot(False)
                print(f"👉 向{direction}变道")
            else:
                print("❌ 无路可走，紧急停车")
                vehicle.apply_control(carla.VehicleControl(brake=1.0))
                stuck_timer = 0

    else:
        # --- 阶段 1, 2, 3: 手动控制 ---
        if avoidance_stage == 1:
            # 变道阶段
            if target_waypoint:
                next_wp = target_waypoint.next(15.0)
                if next_wp:
                    steer = pure_pursuit(next_wp[0].transform.location, v_trans)
                    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=steer))

                    if current_waypoint.lane_id == target_waypoint.lane_id:
                        print("✅ 变道完成，加速超车")
                        avoidance_stage = 2
                        super_timer = time.time() # 记录超车开始时间

        elif avoidance_stage == 2:
            # 超车直行阶段
            vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
            if time.time() - super_timer > 3.0:
                print("🔙 准备回归")
                avoidance_stage = 3

        elif avoidance_stage == 3:
            # 回归阶段
            if original_lane_id:
                return_wp = None
                # 判断回归方向
                if original_lane_id > current_waypoint.lane_id:
                    return_wp = current_waypoint.get_right_lane()
                else:
                    return_wp = current_waypoint.get_left_lane()

                if return_wp and return_wp.lane_id == original_lane_id:
                    next_wp = return_wp.next(10.0)
                    if next_wp:
                        steer = pure_pursuit(next_wp[0].transform.location, v_trans)
                        vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=steer))

                        if current_waypoint.lane_id == original_lane_id:
                            print("🏁 回归完成")
                            is_avoiding = False
                            vehicle.set_autopilot(True)
                            original_lane_id = None

# --- 主程序 ---
vehicle = None # 提前声明变量以便在finally中使用
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    map = world.get_map()

    # 1. 生成主车
    spawn_points = map.get_spawn_points()
    start_spawn = random.choice(spawn_points)
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, start_spawn)
    actor_list.append(vehicle)
    print(f"🚗 主车生成: {start_spawn.location}")

    # 2. 传感器
    collision_bp = blueprint_library.find('sensor.other.collision')
    sensor_col = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    sensor_col.listen(on_collision)
    actor_list.append(sensor_col)

    lane_inv_bp = blueprint_library.find('sensor.other.lane_invasion')
    sensor_lane = world.spawn_actor(lane_inv_bp, carla.Transform(), attach_to=vehicle)
    sensor_lane.listen(on_lane_invasion)
    actor_list.append(sensor_lane)

    # 3. 生成障碍物
    obstacles = spawn_obstacles_ahead(world, blueprint_library, vehicle, count=2, distance=20.0)

    # 4. 启动
    vehicle.set_autopilot(True)
    time.sleep(1)

    print("🚀 模拟开始...")

    while True:
        # 视角控制
        update_camera(world, vehicle)

        # 获取当前状态
        current_waypoint = map.get_waypoint(vehicle.get_location())
        v_trans = vehicle.get_transform()

        # 执行避障状态机
        handle_avoidance_state(vehicle, map, current_waypoint, v_trans)

        time.sleep(DT)

except Exception as e:
    print(f"❌ 错误: {e}")

finally:
    print("\n🧹 清理环境...")
    # 安全销毁传感器和Actor
    for actor in actor_list:
        if actor is not None:
            try:
                # 如果是传感器，先停止监听
                if hasattr(actor, 'stop'):
                    actor.stop()
                actor.destroy()
            except:
                pass
    print("✅ 结束")