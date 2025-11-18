import carla
import time

# 连接CARLA服务（默认端口2000）
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# 示例：生成车辆并采集数据
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

try:
    # 模拟数据采集（可根据需求扩展）
    print("开始采集数据...")
    for _ in range(5):
        transform = vehicle.get_transform()
        print(f"车辆位置: {transform.location}")
        time.sleep(1)
finally:
    vehicle.destroy()
    print("数据采集结束")