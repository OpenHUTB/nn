import carla
import random
import time

def log_with_time(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def test_full_traffic():
    log_with_time("开始测试完整交通场景")
    
    # 连接到模拟器
    log_with_time("连接到模拟器...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    bp_lib = world.get_blueprint_library()
    log_with_time("模拟器连接成功")
    
    # 生成主车辆
    log_with_time("生成主车辆...")
    vehicle_bp = bp_lib.find('vehicle.tesla.model3')
    spawn_points = carla_map.get_spawn_points()
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    log_with_time(f"主车辆生成成功: {ego_vehicle.id}")
    
    # 生成其他车辆
    log_with_time("生成其他车辆...")
    vehicle_bps = bp_lib.filter('vehicle.*')
    vehicles = []
    
    for i in range(5):
        if i >= len(spawn_points):
            break
        bp = random.choice(vehicle_bps)
        try:
            vehicle = world.spawn_actor(bp, spawn_points[i+1])
            vehicle.set_autopilot(True)
            vehicles.append(vehicle)
            log_with_time(f"生成车辆 {i+1}/5: {bp.id}")
        except Exception as e:
            log_with_time(f"车辆生成失败: {e}")
    
    log_with_time(f"成功生成 {len(vehicles)} 辆车辆")
    
    # 生成行人
    log_with_time("生成行人...")
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    walkers = []
    walker_controllers = []
    
    for i in range(3):
        try:
            loc = world.get_random_location_from_navigation()
            if loc is None:
                log_with_time("获取位置失败")
                continue
            
            bp = walker_bps[i % len(walker_bps)]
            transform = carla.Transform(
                carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.5),
                carla.Rotation(yaw=random.uniform(-180, 180))
            )
            
            walker = world.spawn_actor(bp, transform)
            controller_bp = bp_lib.find('controller.ai.walker')
            controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
            
            controller.start()
            target_loc = world.get_random_location_from_navigation()
            if target_loc:
                controller.go_to_location(target_loc)
                controller.set_max_speed(1.5)
            
            walkers.append(walker)
            walker_controllers.append(controller)
            log_with_time(f"生成行人 {i+1}/3: {bp.id}")
            
        except Exception as e:
            log_with_time(f"行人生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    log_with_time(f"成功生成 {len(walkers)} 个行人")
    
    # 运行一段时间
    log_with_time("运行测试...")
    for i in range(100):
        world.tick()
        if i % 20 == 0:
            speed = 3.6 * ego_vehicle.get_velocity().length()
            log_with_time(f"帧 {i}: 速度 {speed:.1f} km/h, 附近车辆 {len(vehicles)}, 附近行人 {len(walkers)}")
        time.sleep(0.05)
    
    # 清理
    log_with_time("清理资源...")
    for controller in walker_controllers:
        controller.stop()
        controller.destroy()
    for walker in walkers:
        walker.destroy()
    for vehicle in vehicles:
        vehicle.destroy()
    ego_vehicle.destroy()
    
    log_with_time("测试完成")

if __name__ == "__main__":
    test_full_traffic()