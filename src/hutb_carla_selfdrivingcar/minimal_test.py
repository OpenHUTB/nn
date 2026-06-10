import carla
import time

def main():
    print("1. 连接到模拟器...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print("连接成功")
    
    print("\n2. 获取地图...")
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    print(f"找到 {len(spawn_points)} 个生成点")
    
    print("\n3. 生成主车辆...")
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    print(f"车辆生成成功，ID: {vehicle.id}")
    
    print("\n4. 生成其他车辆...")
    vehicle_bps = bp_lib.filter('vehicle.*')
    for i in range(3):
        bp = vehicle_bps[i % len(vehicle_bps)]
        try:
            other_vehicle = world.spawn_actor(bp, spawn_points[i+1])
            print(f"生成车辆 {i+1}: {bp.id}")
        except:
            pass
    
    print("\n5. 开始驾驶...")
    for i in range(100):
        world.tick()
        ctrl = carla.VehicleControl(throttle=0.5, brake=0.0)
        vehicle.apply_control(ctrl)
        
        if i % 20 == 0:
            speed = 3.6 * vehicle.get_velocity().length()
            print(f"帧 {i}: 速度 {speed:.1f} km/h")
        
        time.sleep(0.05)
    
    print("\n6. 清理...")
    vehicle.destroy()
    print("完成")

if __name__ == "__main__":
    main()