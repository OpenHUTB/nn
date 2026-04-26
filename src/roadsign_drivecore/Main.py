import sys

sys.path.append(r'D:\CARLA_0.9.15\PythonAPI\carla\dist\carla-0.9.15-cp37-none-win_amd64.egg')

import carla
import random
import pygame
import numpy as np
import math
import time

# ====================== 全局配置 ======================
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FPS_TARGET = 30
TRAFFIC_VEHICLE_COUNT = 10
SAFE_FOLLOWING_DISTANCE = 10.0

# ====================== 全局状态变量 ======================
manual_mode = False
collision_flag = False
current_traffic_light = "GREEN"


# ====================== 工具函数 ======================
def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    return array


def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2) * 3.6


def get_distance_to_front_vehicle(vehicle, world):
    trans = vehicle.get_transform()
    loc = trans.location
    rot = trans.rotation

    # 向前发射射线检测前车
    ray_start = loc + carla.Location(z=1.0)
    ray_end = ray_start + carla.Location(
        x=math.cos(math.radians(rot.yaw)) * 50.0,
        y=math.sin(math.radians(rot.yaw)) * 50.0,
        z=0
    )

    # 简化版：直接获取附近车辆
    actors = world.get_actors().filter("vehicle.*")
    min_dist = float('inf')
    for actor in actors:
        if actor.id != vehicle.id:
            dist = actor.get_transform().location.distance(loc)
            if dist < min_dist and dist < 50.0:
                # 检查是否在正前方
                other_trans = actor.get_transform()
                other_loc = other_trans.location
                dx = other_loc.x - loc.x
                dy = other_loc.y - loc.y
                angle = math.degrees(math.atan2(dy, dx))
                angle_diff = abs(angle - rot.yaw)
                if angle_diff < 30 or angle_diff > 330:
                    min_dist = dist
    return min_dist if min_dist != float('inf') else 999.9


# ====================== 碰撞检测回调 ======================
def on_collision(event):
    global collision_flag
    collision_flag = True
    print("⚠️  碰撞检测触发！")


# ====================== 视角跟随（稳定版） ======================
def update_spectator(vehicle, world):
    trans = vehicle.get_transform()
    loc = trans.location
    rot = trans.rotation

    distance = 15.0
    height = 6.0
    pitch = -35.0

    angle = math.radians(rot.yaw)
    dx = -math.cos(angle) * distance
    dy = -math.sin(angle) * distance

    cam_loc = carla.Location(x=loc.x + dx, y=loc.y + dy, z=loc.z + height)
    cam_rot = carla.Rotation(pitch=pitch, yaw=rot.yaw)
    world.get_spectator().set_transform(carla.Transform(cam_loc, cam_rot))


# ====================== 红绿灯状态检测 ======================
def update_traffic_light_state(vehicle):
    global current_traffic_light
    if vehicle.is_at_traffic_light():
        light = vehicle.get_traffic_light()
        state = light.get_state()
        if state == carla.TrafficLightState.Red:
            current_traffic_light = "RED"
        elif state == carla.TrafficLightState.Yellow:
            current_traffic_light = "YELLOW"
        else:
            current_traffic_light = "GREEN"
    else:
        current_traffic_light = "GREEN"


# ====================== 车辆控制核心（自动+手动） ======================
def control_vehicle(vehicle, world):
    global collision_flag, manual_mode, current_traffic_light

    # 碰撞保护优先
    if collision_flag:
        vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0, steer=0))
        collision_flag = False
        return

    # 手动模式控制
    if manual_mode:
        vehicle.set_autopilot(False)
        keys = pygame.key.get_pressed()
        control = carla.VehicleControl()

        if keys[pygame.K_w]:
            control.throttle = 0.7
            control.reverse = False
        elif keys[pygame.K_s]:
            control.throttle = 0.5
            control.reverse = True
        else:
            control.throttle = 0.0

        if keys[pygame.K_SPACE]:
            control.brake = 1.0
        else:
            control.brake = 0.0

        if keys[pygame.K_a]:
            control.steer = -0.4
        elif keys[pygame.K_d]:
            control.steer = 0.4
        else:
            control.steer = 0.0

        vehicle.apply_control(control)
        return

    # 自动驾驶模式
    vehicle.set_autopilot(True)

    # 红绿灯控制
    if current_traffic_light == "RED":
        vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0, steer=0))
        return

    # 跟车距离控制
    front_dist = get_distance_to_front_vehicle(vehicle, world)
    if front_dist < SAFE_FOLLOWING_DISTANCE:
        vehicle.apply_control(carla.VehicleControl(throttle=0, brake=0.5, steer=0))
        return


# ====================== 生成交通车辆 ======================
def spawn_traffic_vehicles(world, bp_lib, count):
    vehicles = []
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for i in range(min(count, len(spawn_points))):
        bp = random.choice(bp_lib.filter("vehicle.*"))
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        veh = world.try_spawn_actor(bp, spawn_points[i])
        if veh:
            veh.set_autopilot(True)
            vehicles.append(veh)
            print(f"✅ 生成交通车辆 {i + 1}/{count}")

    return vehicles


# ====================== 界面绘制辅助函数 ======================
def draw_text(screen, text, position, color=(255, 255, 255), font_size=24):
    font = pygame.font.SysFont("Arial", font_size, bold=True)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)


def draw_traffic_light_indicator(screen, state, position):
    colors = {
        "RED": (255, 0, 0),
        "YELLOW": (255, 255, 0),
        "GREEN": (0, 255, 0)
    }
    color = colors.get(state, (100, 100, 100))
    pygame.draw.circle(screen, color, position, 25)
    pygame.draw.circle(screen, (255, 255, 255), position, 25, 3)


# ====================== 主程序入口 ======================
def main():
    global manual_mode, collision_flag, current_traffic_light

    # 初始化Pygame
    pygame.init()
    screen = pygame.display.set_mode((IMAGE_WIDTH, IMAGE_HEIGHT))
    pygame.display.set_caption("CARLA 智能驾驶系统 V3.0")
    clock = pygame.time.Clock()

    # 连接CARLA
    print("🔗 正在连接CARLA服务器...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    print("✅ CARLA连接成功！")

    # 生成主车
    print("🚗 正在生成主车...")
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    vehicle = None
    for sp in spawn_points:
        vehicle = world.try_spawn_actor(
            bp_lib.filter("vehicle.tesla.model3")[0],
            sp
        )
        if vehicle is not None:
            break

    if vehicle is None:
        print("❌ 错误：没有找到可用的生成点！")
        return
    print("✅ 主车生成成功！")

    # 生成碰撞传感器
    collision_sensor = world.spawn_actor(
        bp_lib.find("sensor.other.collision"),
        carla.Transform(),
        attach_to=vehicle
    )
    collision_sensor.listen(on_collision)

    # 生成摄像头
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    cam_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    cam_bp.set_attribute("fov", "110")
    cam_bp.set_attribute("sensor_tick", "0.033")
    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=1.5, z=1.8)),
        attach_to=vehicle
    )

    frame = None

    def camera_callback(image):
        nonlocal frame
        frame = process_image(image)

    camera.listen(camera_callback)

    # 生成交通车辆
    print("🚙 正在生成交通车辆...")
    traffic_vehicles = spawn_traffic_vehicles(world, bp_lib, TRAFFIC_VEHICLE_COUNT)
    print(f"✅ 共生成 {len(traffic_vehicles)} 辆交通车")

    print("\n" + "=" * 50)
    print("🚀 CARLA 智能驾驶系统 V3.0 启动成功！")
    print("=" * 50)
    print("操作说明：")
    print("  M键       - 切换手动/自动驾驶模式")
    print("  W/S/A/D   - 手动模式：前进/后退/转向")
    print("  空格键     - 手动模式：紧急刹车")
    print("  ESC键     - 退出程序")
    print("=" * 50 + "\n")

    try:
        while True:
            clock.tick(FPS_TARGET)

            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_m:
                        manual_mode = not manual_mode
                        mode_str = "手动驾驶" if manual_mode else "自动驾驶"
                        print(f"🔄 切换至：{mode_str}")

            # 更新系统状态
            update_spectator(vehicle, world)
            update_traffic_light_state(vehicle)
            control_vehicle(vehicle, world)

            # 绘制界面
            if frame is not None:
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(surf, (0, 0))
            else:
                screen.fill((0, 0, 0))

            # 绘制HUD信息
            hud_bg = pygame.Surface((350, 220), pygame.SRCALPHA)
            hud_bg.fill((0, 0, 0, 180))
            screen.blit(hud_bg, (10, 10))

            mode_color = (0, 255, 0) if not manual_mode else (255, 165, 0)
            mode_str = "自动驾驶 (AUTOPILOT)" if not manual_mode else "手动驾驶 (MANUAL)"
            draw_text(screen, mode_str, (25, 20), mode_color, 26)

            speed = get_speed(vehicle)
            draw_text(screen, f"车速: {speed:.1f} km/h", (25, 60), (255, 255, 255), 24)

            draw_text(screen, "红绿灯状态:", (25, 100), (255, 255, 255), 22)
            tl_color_map = {"RED": (255, 0, 0), "YELLOW": (255, 255, 0), "GREEN": (0, 255, 0)}
            draw_text(screen, current_traffic_light, (160, 100),
                      tl_color_map.get(current_traffic_light, (255, 255, 255)), 24)

            front_dist = get_distance_to_front_vehicle(vehicle, world)
            dist_color = (0, 255, 0) if front_dist > SAFE_FOLLOWING_DISTANCE else (255, 0, 0)
            draw_text(screen, f"前车距离: {front_dist:.1f} m", (25, 140), dist_color, 22)

            draw_text(screen, f"帧率: {int(clock.get_fps())} FPS", (25, 175), (200, 200, 200), 20)

            # 绘制操作提示
            tip_bg = pygame.Surface((400, 40), pygame.SRCALPHA)
            tip_bg.fill((0, 0, 0, 150))
            screen.blit(tip_bg, (IMAGE_WIDTH // 2 - 200, IMAGE_HEIGHT - 50))
            draw_text(screen, "按 M 切换模式 | ESC 退出", (IMAGE_WIDTH // 2 - 150, IMAGE_HEIGHT - 45), (200, 200, 200),
                      20)

            pygame.display.flip()

    finally:
        print("\n🛑 正在清理资源...")
        camera.destroy()
        collision_sensor.destroy()
        vehicle.destroy()
        for v in traffic_vehicles:
            v.destroy()
        pygame.quit()
        print("✅ 资源清理完成，程序退出。")


if __name__ == "__main__":
    main()