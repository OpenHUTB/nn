import carla
import random
import time
import pygame
import numpy as np
import math

# 核心配置
CONFIG = {
    "CARLA_HOST": "localhost",
    "CARLA_PORT": 2000,
    "CAMERA_WIDTH": 800,
    "CAMERA_HEIGHT": 600,
    "CRUISE_SPEED": 40,
    "INTERSECTION_SPEED": 25,
    "SAFE_STOP_DISTANCE": 15,
    "MIN_STOP_DISTANCE": 3
}

# ================== V4.0 碰撞状态全局标记 ==================
need_vehicle_reset = False


# 初始化Pygame显示
def init_pygame(width, height):
    pygame.init()
    display = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CARLA V4.0 第三人称视角版")
    return display


# 转换CARLA图像用于显示
def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    return array


# 获取当前车速
def get_speed(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6


# 计算转向角
def get_steer(vehicle_transform, waypoint_transform):
    v_loc = vehicle_transform.location
    v_forward = vehicle_transform.get_forward_vector()
    wp_loc = waypoint_transform.location

    direction = carla.Vector3D(wp_loc.x - v_loc.x, wp_loc.y - v_loc.y, 0.0)
    v_forward = carla.Vector3D(v_forward.x, v_forward.y, 0.0)

    dir_norm = math.hypot(direction.x, direction.y)
    fwd_norm = math.hypot(v_forward.x, v_forward.y)
    if dir_norm < 1e-5 or fwd_norm < 1e-5:
        return 0.0

    dot = (v_forward.x * direction.x + v_forward.y * direction.y) / (dir_norm * fwd_norm)
    dot = max(-1.0, min(1.0, dot))
    angle = math.acos(dot)
    cross = v_forward.x * direction.y - v_forward.y * direction.x
    if cross < 0:
        angle *= -1
    return max(-1.0, min(1.0, angle * 2.0))


# 计算到路口距离
def get_distance_to_intersection(vehicle, map):
    vehicle_loc = vehicle.get_transform().location
    waypoint = map.get_waypoint(vehicle_loc, project_to_road=True)
    check_distance = 0
    current_wp = waypoint
    for _ in range(50):
        next_wps = current_wp.next(2.0)
        if not next_wps:
            break
        current_wp = next_wps[0]
        check_distance += 2.0
        if current_wp.is_junction or len(current_wp.next(2.0)) > 1:
            return check_distance
    return 999


# ================== V4.0 碰撞事件回调函数 ==================
def on_collision(event):
    global need_vehicle_reset
    need_vehicle_reset = True
    collision_force = event.normal_impulse.length()
    print(f"【V4.0 碰撞保护】检测到碰撞！强度：{collision_force:.1f}，准备重置车辆")


def main():
    global need_vehicle_reset
    actor_list = []
    try:
        # 连接CARLA
        client = carla.Client(CONFIG["CARLA_HOST"], CONFIG["CARLA_PORT"])
        client.set_timeout(10.0)
        world = client.get_world()
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        # 生成主车
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_point = random.choice(map.get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print("主车生成成功")

        # ================== V4.0 挂载碰撞传感器 ==================
        collision_bp = blueprint_library.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        collision_sensor.listen(on_collision)
        actor_list.append(collision_sensor)

        # 生成背景车辆
        traffic_count = random.randint(10, 15)
        spawned_traffic = 0
        for _ in range(traffic_count):
            traffic_bp = random.choice(blueprint_library.filter('vehicle.*'))
            traffic_spawn = random.choice(map.get_spawn_points())
            traffic_vehicle = world.try_spawn_actor(traffic_bp, traffic_spawn)
            if traffic_vehicle:
                traffic_vehicle.set_autopilot(True)
                actor_list.append(traffic_vehicle)
                spawned_traffic += 1
        print(f"生成背景车辆：{spawned_traffic}辆")

        # 生成摄像头
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(CONFIG["CAMERA_WIDTH"]))
        camera_bp.set_attribute("image_size_y", str(CONFIG["CAMERA_HEIGHT"]))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        # 摄像头回调
        image_surface = [None]

        def image_callback(image):
            image_surface[0] = process_image(image)

        camera.listen(image_callback)

        # 初始化显示
        display = init_pygame(CONFIG["CAMERA_WIDTH"], CONFIG["CAMERA_HEIGHT"])
        clock = pygame.time.Clock()

        # ================== 修正为：车辆后上方第三人称跟随视角 ==================
        spectator = world.get_spectator()

        def update_spectator():
            transform = vehicle.get_transform()
            # 视角位置：车辆正后方10米、上方8米
            # 视角角度：pitch向下15度，yaw与车辆完全同步
            spectator.set_transform(carla.Transform(
                transform.location + transform.get_forward_vector() * -10 + carla.Location(z=8),
                carla.Rotation(pitch=-15, yaw=transform.rotation.yaw, roll=0)
            ))

        # ========================================================================

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            update_spectator()
            control = carla.VehicleControl()
            current_speed = get_speed(vehicle)
            vehicle_transform = vehicle.get_transform()

            # ================== V4.0 碰撞后车辆重置逻辑 ==================
            if need_vehicle_reset:
                # 1. 先确保车辆停稳
                control.throttle = 0.0
                control.brake = 1.0
                control.steer = 0.0
                vehicle.apply_control(control)
                time.sleep(1)

                # 2. 随机选择新的生成点
                new_spawn_point = random.choice(map.get_spawn_points())

                # 3. 重置车辆位置和物理状态
                vehicle.set_transform(new_spawn_point)
                vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))

                # 4. 清除重置标记
                need_vehicle_reset = False
                print(f"【V4.0 碰撞保护】车辆已重置到新位置：{new_spawn_point.location}")
                continue

            # 原有红绿灯逻辑（完全不变）
            traffic_light = vehicle.get_traffic_light()
            light_state = vehicle.get_traffic_light_state()
            is_red_light = (light_state == carla.TrafficLightState.Red)
            distance_to_intersection = get_distance_to_intersection(vehicle, map)
            should_stop = False

            if is_red_light and traffic_light is not None:
                dynamic_stop_distance = CONFIG["SAFE_STOP_DISTANCE"] + (current_speed / 10)
                if distance_to_intersection < dynamic_stop_distance:
                    should_stop = True

            if should_stop:
                if distance_to_intersection < CONFIG["MIN_STOP_DISTANCE"] or current_speed < 5:
                    control.throttle = 0.0
                    control.brake = 1.0
                    control.steer = 0.0
                else:
                    brake_strength = 0.5 + (CONFIG["SAFE_STOP_DISTANCE"] - distance_to_intersection) / CONFIG[
                        "SAFE_STOP_DISTANCE"] * 0.5
                    control.throttle = 0.0
                    control.brake = min(brake_strength, 1.0)
                    control.steer = 0.0
            else:
                waypoint = map.get_waypoint(vehicle_transform.location, project_to_road=True)
                next_waypoints = waypoint.next(2.0)
                if next_waypoints:
                    next_waypoint = next_waypoints[0]
                    control.steer = get_steer(vehicle_transform, next_waypoint.transform)

                target_speed = CONFIG["CRUISE_SPEED"]
                if distance_to_intersection < 30:
                    target_speed = CONFIG["INTERSECTION_SPEED"]

                if current_speed < target_speed:
                    control.throttle = 0.5
                    control.brake = 0.0
                else:
                    control.throttle = 0.2
                    control.brake = 0.0


            vehicle.apply_control(control)

            # 画面显示
            if image_surface[0] is not None:
                surface = pygame.image.frombuffer(image_surface[0].tobytes(),
                                                  (CONFIG["CAMERA_WIDTH"], CONFIG["CAMERA_HEIGHT"]), "RGB")
                display.blit(surface, (0, 0))
                pygame.display.flip()


            clock.tick(30)

    finally:
        print("清理资源...")
        for actor in actor_list:
            try:
                actor.destroy()
            except:
                pass
        pygame.quit()
        print("结束")


if __name__ == "__main__":
    main()