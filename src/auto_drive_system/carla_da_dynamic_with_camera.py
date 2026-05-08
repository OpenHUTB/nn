"""
简单实现动态障碍车超车（配置车周摄像头）
"""

import carla
import time
import math
import numpy as np
import pygame
import util.camera as cs

actor_list = []
collision_flag = False  # 添加全局变量
is_avoiding = False  # 全局状态标记
target_lane = None   # 目标车道点

# 在碰撞回调中强制刹车
def callback(event):
    global collision_flag
    if not collision_flag:
        vehicle.apply_control(carla.VehicleControl(brake=1.0))  # 紧急制动
        collision_flag = True
        print("碰撞！")

# 车道偏离回调
def callback2(event):
    print("穿越车道!")

# 摄像头调用回调
def pygame_callback(image, side):
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    if side == 'Front':
        global Front
        Front = img
    elif side == 'Rear':
        global Rear
        Rear = img
    elif side == 'Left':
        global Left
        Left = img
    elif side == 'Right':
        global Right
        Right = img


    if ('Front' in globals() and 'Rear' in globals()
        and "Left" in globals()and 'Right' in globals()):
        # 横向拼接（前后）（左右）摄像头的画面
        img_combined_front = np.concatenate((Front, Rear), axis=1)
        img_combined_rear = np.concatenate((Left, Right), axis=1)
        # 纵向拼接（前后）（左右）摄像头的画面
        img_combined = np.concatenate((img_combined_front, img_combined_rear), axis=0)
        renderObject.surface = pygame.surfarray.make_surface(img_combined.swapaxes(0, 1))


def spawn_obstacles(world, blueprint_library, vehicle, num_obstacles=2, distance=30.0):
    """
    在主循环开始前，生成障碍物，确保障碍物在主车前方
    """
    obstacles = []
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_yaw = vehicle_transform.rotation.yaw  # 车辆的朝向（Yaw角）
    offset_ys = [3.0, 7] * num_obstacles
    # 障碍物的位置偏移：根据车辆的朝向（Yaw角）计算障碍物的生成位置
    for i in range(num_obstacles):
        offset_x = distance + i * 10.0  # 障碍物距离主车的偏移
        offset_y = offset_ys[i] # 障碍物的侧向偏移量
        # 计算障碍物的实际位置
        obstacle_x = vehicle_location.x + offset_x
        obstacle_y = vehicle_location.y + offset_y
        print(math.radians(vehicle_yaw))
        # 生成障碍物的spawn point
        spawn_point = carla.Transform(carla.Location(x=obstacle_x, y=obstacle_y, z=0.5))  # 设置Z轴为0.3，避免地面生成
        obstacle_bp = blueprint_library.filter('vehicle.*')[0]  # 障碍物类型为车辆
        print(spawn_point.location)
        obstacle = world.spawn_actor(obstacle_bp, spawn_point)  # 在指定位置生成障碍物
        obstacle.apply_control(carla.VehicleControl(throttle=0.15 * (i + 1), steer=0.0))
        actor_list.append(obstacle)  # 将障碍物添加到actor_list
        obstacles.append(obstacle)  # 将障碍物添加到障碍物列表

    return obstacles


def pure_pursuit(tar_location, v_transform):
    L = 2.875  # 汽车轴距
    yaw = v_transform.rotation.yaw * (math.pi / 180)  # 汽车航向角
    # 计算汽车后轮位置
    x = v_transform.location.x - L / 2 * math.cos(yaw)
    y = v_transform.location.y - L / 2 * math.sin(yaw)
    # 计算x, y方向上的距离
    dx = tar_location.x - x
    dy = tar_location.y - y
    ld = math.sqrt(dx ** 2 + dy ** 2)
    alpha = math.atan2(dy, dx) - yaw
    # 最优转角公式：tan(delta) = (2L sin(alpha)) / ld
    delta = math.atan(2 * math.sin(alpha) * L / ld) * 180 / math.pi
    steer = delta / 90
    if steer > 1:
        steer = 1
    elif steer < -1:
        steer = -1
    return steer

def destroy_actor(world, actor):
    """
    安全销毁Carla Actor，处理控制器
    """
    # 处理车辆
    if 'vehicle' in actor.type_id:
        actor.set_autopilot(False)  # 禁用autopilot
    # 处理行人
    if 'walker' in actor.type_id and hasattr(actor, 'controller'):
        actor.controller.stop()  # 停止控制器
        world.try_destroy_actor(actor.controller)  # 销毁控制器
    # 销毁Actor
    actor.destroy()

def get_new_lane(current_waypoint):
    right_lane = current_waypoint.get_right_lane()
    if right_lane and (right_lane.lane_type & carla.LaneType.Driving):
        return right_lane
    left_lane = current_waypoint.get_left_lane()
    if left_lane and (left_lane.lane_type & carla.LaneType.Driving):
        return left_lane
    return None


try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    map = world.get_map()
    blueprint_library = world.get_blueprint_library()

    # 生成主车
    v_bp = blueprint_library.filter("model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    location = carla.Location(x=-48, y=-34, z=0.275307)
    rotation = carla.Rotation(yaw=0)
    spawn_point = carla.Transform(location, rotation)
    vehicle = world.spawn_actor(v_bp, spawn_point)
    actor_list.append(vehicle)

    # 车周摄像头cameras设置
    pygame_size = {
        "image_x": 1152,
        "image_y": 600
    }
    cameras = cs.cameraManage(world, vehicle, pygame_size).camaraGenarate()
    actor_list.append(cameras.get("Front"))
    actor_list.append(cameras.get("Rear"))
    actor_list.append(cameras.get("Left"))
    actor_list.append(cameras.get("Right"))
    cameras.get("Front").listen(lambda image: pygame_callback(image, 'Front'))
    cameras.get("Rear").listen(lambda image: pygame_callback(image, 'Rear'))
    cameras.get("Left").listen(lambda image: pygame_callback(image, 'Left'))
    cameras.get("Right").listen(lambda image: pygame_callback(image, 'Right'))
    renderObject = cs.RenderObject(pygame_size.get("image_x"), pygame_size.get("image_y"))
    pygame.init()
    gameDisplay = pygame.display.set_mode((pygame_size.get("image_x"), pygame_size.get("image_y")),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)
    time.sleep(3) # 等待pygame窗口启动

    # 传感器collision_detector设置
    blueprint_collisionDetector = blueprint_library.find('sensor.other.collision')
    transform = carla.Transform(carla.Location(x=0.8, z=1.7))
    sensor_collision = world.spawn_actor(blueprint_collisionDetector, transform, attach_to=vehicle)
    actor_list.append(sensor_collision)
    sensor_collision.listen(callback)

    # 传感器lane_invasion设置
    blueprint_lane_invasion = blueprint_library.find('sensor.other.lane_invasion')
    transform = carla.Transform(carla.Location(x=0.8, z=1.7))
    sensor_lane_invasion = world.spawn_actor(blueprint_lane_invasion, transform, attach_to=vehicle)
    actor_list.append(sensor_lane_invasion)
    sensor_lane_invasion.listen(callback2)

    # 在主循环开始之前生成3-4辆障碍车，并确保它们位于主车前方
    obstacles = spawn_obstacles(world, blueprint_library, vehicle, num_obstacles=2, distance=20.0)

    vehicle.set_autopilot(True)
    time.sleep(2)  # 等autopilot启动车辆

    stuck_timer = 0  # 停滞计时器
    crashed = False
    while not crashed:
        # 摄像头画面设置
        world.tick()
        gameDisplay.blit(renderObject.surface, (0, 0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

        # 视角设置
        vehicle_transform = vehicle.get_transform()
        spectator = world.get_spectator()
        offset = carla.Location(x=-6.0, z=2.5)  # 车辆后方 6 米，高度 2.5 米
        spectator.set_transform(carla.Transform(
            vehicle_transform.location + vehicle_transform.get_forward_vector() * offset.x + offset,
            vehicle_transform.rotation
        ))

        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)  # 计算速度大小
        if not is_avoiding:
            if speed < 2.5:
                stuck_timer += 1
                if stuck_timer > 5:  # 车速过慢超过 0.02 * 5 = 0.1 秒，触发避障
                    print("前方障碍！")
                    current_waypoint = map.get_waypoint(vehicle.get_transform().location)
                    target_lane = get_new_lane(current_waypoint)
                    if target_lane:
                        print("开始绕行")
                        vehicle.set_autopilot(False)
                        is_avoiding = True
                        stuck_timer = 0
                    else:
                        print("无可用车道，紧急停车！")
                        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        else:
            # 持续跟踪目标车道点
            steer = pure_pursuit(target_lane.transform.location, vehicle.get_transform())
            vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=steer))
            # 检测是否到达目标车道
            current_loc = vehicle.get_location()
            target_distance = current_loc.distance(target_lane.transform.location)
            current_waypoint = map.get_waypoint(current_loc)
            # 条件1：距离目标点<1米 或 条件2：已进入右侧车道
            if target_distance < 1.0 or current_waypoint.lane_id == target_lane.lane_id:
                print("绕行完成，恢复正常行驶")
                vehicle.set_autopilot(True)
                is_avoiding = False
                target_lane = None

        time.sleep(0.02) # 控制循环频率

finally:
    # 安全销毁所有Actor
    for actor in actor_list:
        destroy_actor(world, actor)
    print("程序结束，所有Actor已销毁")
