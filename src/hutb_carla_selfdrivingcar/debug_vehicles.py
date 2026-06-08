import carla
import random
import time

def log_with_time(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def debug_vehicle_spawn():
    log_with_time("开始调试车辆生成")
    
    # 连接到模拟器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    bp_lib = world.get_blueprint_library()
    
    # 获取车辆蓝图
    vehicle_bps = bp_lib.filter('vehicle.*')
    log_with_time(f"可用车辆蓝图: {len(vehicle_bps)}")
    
    # 获取生成点
    spawn_points = carla_map.get_spawn_points()
    log_with_time(f"可用生成点: {len(spawn_points)}")
    
    # 逐个测试生成车辆
    for i in range(min(5, len(spawn_points))):
        log_with_time(f"\n尝试生成车辆 {i+1}")
        try:
            # 选择蓝图
            bp = vehicle_bps[i % len(vehicle_bps)]
            log_with_time(f"  使用蓝图: {bp.id}")
            
            # 获取生成点
            spawn_point = spawn_points[i]
            log_with_time(f"  生成点位置: ({spawn_point.location.x:.2f}, {spawn_point.location.y:.2f})")
            
            # 尝试生成
            log_with_time("  开始生成...")
            vehicle = world.spawn_actor(bp, spawn_point)
            log_with_time(f"  生成成功! ID: {vehicle.id}")
            
            # 设置自动驾驶
            vehicle.set_autopilot(True)
            log_with_time("  设置自动驾驶完成")
            
            # 销毁车辆
            vehicle.destroy()
            log_with_time("  已销毁测试车辆")
            
        except Exception as e:
            log_with_time(f"  生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    log_with_time("\n调试完成")

if __name__ == "__main__":
    debug_vehicle_spawn()