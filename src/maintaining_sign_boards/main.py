import carla
import random
import time
import numpy as np
import math
import pygame

# 初始化Pygame显示窗口
def init_pygame(width, height):
    pygame.init()
    display = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Driver's View")
    return display

# 转换CARLA图像为RGB numpy数组
def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]  # 丢弃alpha通道
    return array

# 计算车辆到目标路点的转向角度
def get_steering_angle(vehicle_transform, waypoint_transform):
    v_loc = vehicle_transform.location
    v_forward = vehicle_transform.get_forward_vector()
    wp_loc = waypoint_transform.location
    direction = wp_loc - v_loc
    direction = carla.Vector3D(direction.x, direction.y, 0.0)

    v_forward = carla.Vector3D(v_forward.x, v_forward.y, 0.0)
    norm_dir = math.sqrt(direction.x ** 2 + direction.y ** 2)
    norm_fwd = math.sqrt(v_forward.x ** 2 + v_forward.y ** 2)

    dot = v_forward.x * direction.x + v_forward.y * direction.y
    angle = math.acos(dot / (norm_dir * norm_fwd + 1e-5))# 避免除零错误
    cross = v_forward.x * direction.y - v_forward.y * direction.x
    if cross < 0:
        angle *= -1
    return angle


# 主函数
def main():
    actor_list = []
    try:
        # 连接CARLA模拟器
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        print("连接CARLA模拟器成功")

        # 生成主车辆 特斯拉Model3
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_point = random.choice(map.get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print(f"车辆生成于: {spawn_point.location}")

        # 生成10辆随机交通车并开启自动行驶
        for _ in range(10):
            traffic_bp = random.choice(blueprint_library.filter('vehicle.*'))
            traffic_spawn = random.choice(map.get_spawn_points())
            traffic_vehicle = world.try_spawn_actor(traffic_bp, traffic_spawn)
            if traffic_vehicle:
                traffic_vehicle.set_autopilot(True)
                actor_list.append(traffic_vehicle)

        # 挂载RGB摄像头到主车辆
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        # 初始化Pygame窗口
        display = init_pygame(800, 600)

        # 摄像头回调：接收并转换图像
        image_surface = [None]

        def image_callback(image):
            image_surface[0] = process_image(image)

        camera.listen(image_callback)
        print("摄像头挂载完成")

        clock = pygame.time.Clock()

        # 实时显示
        while True:
            # 窗口退出逻辑
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            # 车辆自动转向控制
            transform = vehicle.get_transform()
            waypoint = map.get_waypoint(transform.location, project_to_road=True,lane_type=carla.LaneType.Driving)
            next_waypoint = waypoint.next(2.0)[0]  # 下一个2米的路点
            angle = get_steering_angle(transform, next_waypoint.transform)
            steer = max(-1.0, min(1.0, angle * 2.0))  # 限制转向范围

            # 应用车辆控制：默认油门0.5，自动转向
            control = carla.VehicleControl()
            control.throttle = 0.5
            control.steer = steer
            control.brake = 0.0
            vehicle.apply_control(control)

            # 渲染摄像头画面到Pygame窗口
            if image_surface[0] is not None:
                surface = pygame.image.frombuffer(image_surface[0].tobytes(), (800, 600), "RGB")
                display.blit(surface, (0, 0))
                pygame.display.flip()

            # 限制帧率30FPS
            clock.tick(30)

    finally:
        # 清理Actor
        print("清理actors")
        for actor in actor_list:
            actor.destroy()
        print("完成.")

if __name__ == "__main__":
    main()