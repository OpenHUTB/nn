import carla
import random
import time

def log_with_time(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def debug_detailed():
    log_with_time("开始详细调试")
    
    # 连接到模拟器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    bp_lib = world.get_blueprint_library()
    
    # 测试单个车辆生成（不设置自动驾驶）
    log_with_time("测试1: 单个车辆生成(不含自动驾驶)")
    vehicle_bps = bp_lib.filter('vehicle.*')
    spawn_points = carla_map.get_spawn_points()
    
    bp = vehicle_bps[0]
    spawn_point = spawn_points[1]  # 使用第二个生成点
    
    try:
        log_with_time("  开始生成车辆...")
        vehicle = world.spawn_actor(bp, spawn_point)
        log_with_time(f"  生成成功! ID: {vehicle.id}")
        
        log_with_time("  等待1秒...")
        time.sleep(1)
        
        log_with_time("  销毁车辆...")
        vehicle.destroy()
        log_with_time("  销毁完成")
    except Exception as e:
        log_with_time(f"  失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试设置自动驾驶
    log_with_time("\n测试2: 设置自动驾驶")
    try:
        bp = vehicle_bps[1]
        spawn_point = spawn_points[2]
        
        log_with_time("  生成车辆...")
        vehicle = world.spawn_actor(bp, spawn_point)
        log_with_time(f"  生成成功! ID: {vehicle.id}")
        
        log_with_time("  设置自动驾驶...")
        vehicle.set_autopilot(True)
        log_with_time("  自动驾驶设置完成")
        
        log_with_time("  等待2秒...")
        time.sleep(2)
        
        log_with_time("  销毁车辆...")
        vehicle.destroy()
        log_with_time("  销毁完成")
    except Exception as e:
        log_with_time(f"  失败: {e}")
        import traceback
        traceback.print_exc()
    
    log_with_time("\n调试完成")

if __name__ == "__main__":
    debug_detailed()