import carla

print("尝试连接到 CARLA 模拟器...")
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print("连接成功！")
    
    # 获取地图信息
    carla_map = world.get_map()
    print(f"当前地图: {carla_map.name}")
    
    # 获取生成点数量
    spawn_points = carla_map.get_spawn_points()
    print(f"可用生成点: {len(spawn_points)}")
    
except Exception as e:
    print(f"连接失败: {e}")