import carla
import time
import math

def main():
    client = None
    world = None
    ego_vehicle = None
    collision_sensor = None

    try:
        # 连接服务器
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        print("[INFO] Connected to CARLA successfully")

        # 生成车辆
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_point = world.get_map().get_spawn_points()[0]
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        ego_vehicle.set_autopilot(True)
        print("[INFO] Ego vehicle is spawned and autopilot enabled")

        # 碰撞传感器（用于安全检测）
        collision_bp = bp_lib.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego_vehicle)
        
        def on_collision(event):
            print("[WARNING] Collision detected!")
        collision_sensor.listen(on_collision)

        # 主视角跟随
        spectator = world.get_spectator()
        for _ in range(600):
            world.tick()
            trans = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                trans.location + carla.Location(x=-12, z=5),
                carla.Rotation(pitch=-25)
            ))

            # 速度监控
            v = ego_vehicle.get_velocity()
            speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            print(f"[INFO] Current speed: {speed:.1f} km/h")
            time.sleep(0.02)

    finally:
        # 安全销毁
        if collision_sensor and collision_sensor.is_alive:
            collision_sensor.destroy()
        if ego_vehicle and ego_vehicle.is_alive:
            ego_vehicle.destroy()
        print("[INFO] All actors destroyed safely")

if __name__ == '__main__':
    main()