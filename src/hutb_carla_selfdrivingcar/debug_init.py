import carla
import random

def debug_traffic_init():
    # 连接到模拟器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    bp_lib = world.get_blueprint_library()
    
    print("=== 调试交通场景初始化 ===\n")
    
    # 1. 检查车辆生成
    print("1. 检查车辆生成:")
    vehicle_bps = bp_lib.filter('vehicle.*')
    print(f"   可用车辆蓝图: {len(vehicle_bps)}")
    
    spawn_points = carla_map.get_spawn_points()
    print(f"   可用生成点: {len(spawn_points)}")
    
    if len(spawn_points) > 0:
        print(f"   第一个生成点: {spawn_points[0]}")
    
    # 2. 检查行人生成
    print("\n2. 检查行人生成:")
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    print(f"   可用行人蓝图: {len(walker_bps)}")
    
    # 测试获取随机导航位置
    print("\n3. 测试随机导航位置:")
    for i in range(3):
        loc = world.get_random_location_from_navigation()
        print(f"   位置 {i+1}: {loc}")
    
    # 4. 测试控制器蓝图
    print("\n4. 检查控制器蓝图:")
    try:
        controller_bp = bp_lib.find('controller.ai.walker')
        print(f"   找到行人控制器蓝图: {controller_bp.id}")
    except Exception as e:
        print(f"   找不到行人控制器蓝图: {e}")
    
    print("\n=== 调试完成 ===")

if __name__ == "__main__":
    debug_traffic_init()