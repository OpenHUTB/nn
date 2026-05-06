"""
动态避障超车脚本 (最终修复版)
1. 修复障碍物生成：确保生成在主车正前方
2. 修复视角抖动：引入平滑插值
3. 修复车道偏离：优化纯追踪算法与边界检查
"""

import carla
import time
import math
import random

# --- 全局配置 ---
TARGET_FPS = 60
DT = 1.0 / TARGET_FPS
AVOIDANCE_SPEED = 15.0  # 避障时的速度 (m/s)

# --- 全局变量 ---
actor_list = []
collision_flag = False
is_avoiding = False
avoidance_stage = 0  # 0:正常, 1:变道, 2:直行超车, 3:回归
target_waypoint = None
original_lane_id = None
stuck_timer = 0

# --- 传感器回调 ---
def on_collision(event):
    global collision_flag
    if not collision_flag:
        print(f"⚠️ 碰撞! 强度: {event.normal_impulse}")
        collision_flag = True
        vehicle.apply_control(carla.VehicleControl(brake=1.0))

def on_lane_invasion(event):
    # 仅在非避障阶段报警
    if not is_avoiding:
        lane_types = [x.lane_type for x in event.crossed_lane_markings]
        print(f"⚠️ 意外偏离车道! 类型: {lane_types}")

# --- 核心算法 ---
def pure_pursuit(target_loc, vehicle_transform, lookahead_distance=10.0):
    """
    优化的纯追踪算法
    """
    L = 2.875  # 轴距
    v_loc = vehicle_transform.location
    v_rot = vehicle_transform.rotation
    yaw = math.radians(v_rot.yaw)

    # 1. 计算后轴中心
    rear_x = v_loc.x - (L / 2) * math.cos(yaw)
    rear_y = v_loc.y - (L / 2) * math.sin(yaw)

    # 2. 计算目标向量
    dx = target_loc.x - rear_x
    dy = target_loc.y - rear_y
    ld = math.sqrt(dx**2 + dy**2)

    if ld < 0.5: # 距离太近，停止转向
        return 0.0

    # 3. 计算夹角
    alpha = math.atan2(dy, dx) - yaw
    # 归一化角度到 -pi ~ pi
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))

    # 4. 计算曲率 (引入死区防止抖动)
    delta = math.atan2(2 * L * math.sin(alpha), ld)
    steer = math.degrees(delta) / 90.0

    # 死区处理：如果转角很小，直接回正，防止画龙
    if abs(steer) < 0.05:
        steer = 0.0

    return max(-1.0, min(1.0, steer))

def spawn_obstacles_ahead(world, blueprint_library, vehicle, count=2, distance=25.0):
    """
    强制在主车正前方生成障碍物
    """
    obstacles = []
    v_trans = vehicle.get_transform()
    v_loc = v_trans.location
    v_forward = v_trans.get_forward_vector() # 获取车头朝向向量

    # 获取车辆蓝图
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

    for i in range(count):
        # 计算生成位置：当前位置 + 朝向向量 * 距离
        # 稍微增加一点侧向偏移，避免完全重叠导致物理爆炸
        offset_dist = distance + (i * 6.0)
        spawn_x = v_loc.x + v_forward.x * offset_dist
        spawn_y = v_loc.y + v_forward.y * offset_dist
        spawn_z = v_loc.z + 0.5 # 保持Z轴一致

        spawn_location = carla.Location(spawn_x, spawn_y, spawn_z)

        # 尝试将障碍物吸附到最近的车道中心（防止生成在路肩）
        map = world.get_map()
        waypoint = map.get_waypoint(spawn_location, project_to_road=True)

        final_transform = carla.Transform(spawn_location, v_trans.rotation)
        if waypoint:
            final_transform = carla.Transform(waypoint.transform.location, waypoint.transform.rotation)

        try:
            obs = world.try_spawn_actor(vehicle_bp, final_transform)
            if obs:
                # 关键：强制障碍物刹车或极低速，模拟路障
                obs.set_autopilot(False)
                obs.apply_control(carla.VehicleControl(brake=1.0))
                actor_list.append(obs)
                obstacles.append(obs)
                print(f"✅ 障碍物 {i+1} 已生成在正前方 {offset_dist:.1f}米处")
        except Exception as e:
            print(f"❌ 障碍物生成失败: {e}")

    return obstacles

# --- 主程序 ---
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

    # 3. 生成障碍物 (关键修复)
    obstacles = spawn_obstacles_ahead(world, blueprint_library, vehicle, count=2, distance=20.0)

    # 4. 启动
    vehicle.set_autopilot(True)
    time.sleep(1) # 等待车辆启动

    print("🚀 模拟开始...")

    while True:
        # --- 视角控制 (平滑插值) ---
        v_trans = vehicle.get_transform()
        v_loc = v_trans.location
        v_rot = v_trans.rotation

        # 目标摄像机位置：车后8米，高4米
        target_cam_loc = v_loc + v_trans.get_forward_vector() * -8.0
        target_cam_loc.z += 4.0

        # 目标旋转：始终看向车辆
        target_cam_rot = carla.Rotation(pitch=-15, yaw=v_rot.yaw)

        # 获取当前摄像机状态并插值 (0.1是平滑系数)
        current_spec = world.get_spectator().get_transform()

        # 简单的线性插值
        smooth_x = current_spec.location.x + (target_cam_loc.x - current_spec.location.x) * 0.1
        smooth_y = current_spec.location.y + (target_cam_loc.y - current_spec.location.y) * 0.1
        smooth_z = current_spec.location.z + (target_cam_loc.z - current_spec.location.z) * 0.1

        # yaw角插值需要处理360度跳变，这里简化处理
        smooth_yaw = v_rot.yaw

        world.get_spectator().set_transform(carla.Transform(
            carla.Location(smooth_x, smooth_y, smooth_z),
            carla.Rotation(pitch=-15, yaw=smooth_yaw)
        ))

        # --- 避障逻辑 ---
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        current_waypoint = map.get_waypoint(vehicle.get_location())

        if not is_avoiding:
            # 阶段 0: 正常行驶
            if speed < 1.5: # 阈值调低，防止误判
                stuck_timer += 1
            else:
                stuck_timer = 0

            if stuck_timer > 15: # 持续低速约0.3秒
                print("🚧 检测到前方障碍，准备变道...")
                original_lane_id = current_waypoint.lane_id

                # 寻找相邻车道
                target_waypoint = current_waypoint.get_left_lane()
                direction = "左"
                if not target_waypoint or target_waypoint.lane_type != carla.LaneType.Driving:
                    target_waypoint = current_waypoint.get_right_lane()
                    direction = "右"

                # 边界检查：确保目标车道存在且是行车道
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
            # 阶段 1, 2, 3: 手动控制
            if avoidance_stage == 1:
                # 变道阶段
                if target_waypoint:
                    # 目标点：目标车道前方15米
                    next_wp = target_waypoint.next(15.0)
                    if next_wp:
                        steer = pure_pursuit(next_wp[0].transform.location, v_trans)
                        vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=steer))

                        # 判断变道完成：进入目标车道
                        if current_waypoint.lane_id == target_waypoint.lane_id:
                            print("✅ 变道完成，加速超车")
                            avoidance_stage = 2
                            super_timer = time.time()

            elif avoidance_stage == 2:
                # 超车直行阶段
                vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
                # 超车持续3秒
                if time.time() - super_timer > 3.0:
                    print("🔙 准备回归")
                    avoidance_stage = 3

            elif avoidance_stage == 3:
                # 回归阶段
                if original_lane_id:
                    # 寻找回原车道的路点
                    return_wp = None
                    if original_lane_id > current_waypoint.lane_id:
                        return_wp = current_waypoint.get_right_lane()
                    else:
                        return_wp = current_waypoint.get_left_lane()

                    if return_wp and return_wp.lane_id == original_lane_id:
                        next_wp = return_wp.next(10.0)
                        if next_wp:
                            steer = pure_pursuit(next_wp[0].transform.location, v_trans)
                            vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=steer))

                            # 回归完成
                            if current_waypoint.lane_id == original_lane_id:
                                print("🏁 回归完成")
                                is_avoiding = False
                                vehicle.set_autopilot(True)
                                original_lane_id = None

        time.sleep(DT)

except Exception as e:
    print(f"❌ 错误: {e}")

finally:
    print("\n🧹 清理环境...")
    for actor in actor_list:
        if actor is not None:
            actor.destroy()
    print("✅ 结束")