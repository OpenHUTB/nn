import carla
from module1_spawn import spawn_vehicle

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    vehicle = None
    try:
        vehicle = spawn_vehicle(world, carla_map)
        print("✅ 作业1运行成功：车辆生成")
        for _ in range(100):
            world.tick()
    finally:
        if vehicle:
            vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == "__main__":
    main()