import carla
import time
import sys

print("测试开始")

try:
    print("尝试连接到 localhost:2000...")
    start_time = time.time()
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    
    print("获取世界...")
    world = client.get_world()
    elapsed = time.time() - start_time
    print(f"连接成功！耗时 {elapsed:.2f} 秒")
    
    print("\n测试驾驶...")
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    
    print("车辆已生成")
    
    # 简单驾驶测试
    for i in range(50):
        world.tick()
        ctrl = carla.VehicleControl(throttle=0.6, brake=0.0)
        vehicle.apply_control(ctrl)
        
        if i % 10 == 0:
            speed = 3.6 * vehicle.get_velocity().length()
            print(f"速度: {speed:.1f} km/h")
        
        time.sleep(0.05)
    
    vehicle.destroy()
    print("\n测试完成")
    
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)