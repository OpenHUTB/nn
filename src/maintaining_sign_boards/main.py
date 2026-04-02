import carla
import random
import time

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

    finally:
        # 清理Actor
        print("清理actors")
        for actor in actor_list:
            actor.destroy()
        print("完成.")

if __name__ == "__main__":
    main()