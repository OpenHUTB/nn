import carla
import time
import math

def main():
    # 连接Carla服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # 生成主车
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    spawn_point = world.get_map().get_spawn_points()[0]
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    ego_vehicle.set_autopilot(True)

    print("[INFO] Vehicle spawned successfully. Autopilot enabled.")

    # 视角跟随
    spectator = world.get_spectator()
    try:
        for _ in range(300):  # 运行15秒
            world.tick()
            spectator_transform = carla.Transform(
                ego_vehicle.get_transform().location + carla.Location(x=-10, z=5),
                carla.Rotation(pitch=-20)
            )
            spectator.set_transform(spectator_transform)

            # 状态日志
            if _ % 20 == 0:
                velocity = ego_vehicle.get_velocity()
                speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
                print(f"[STATUS] Speed: {speed:.1f} km/h, Location: {ego_vehicle.get_transform().location}")

    finally:
        ego_vehicle.destroy()
        print("[INFO] Vehicle destroyed. Program exit safely.")

if __name__ == '__main__':
    main()