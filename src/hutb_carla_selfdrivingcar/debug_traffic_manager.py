import carla
import random
import time

def log_with_time(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

class DebugTrafficManager:
    """调试版本的交通管理器"""

    def __init__(self, world, carla_map, num_vehicles=5, num_walkers=3):
        self.world = world
        self.carla_map = carla_map
        self.bp_lib = world.get_blueprint_library()
        self.vehicles = []
        self.walkers = []
        self.walker_controllers = []

        log_with_time("开始生成车辆...")
        self.spawn_vehicles(num_vehicles)
        log_with_time("车辆生成完成")
        
        log_with_time("开始生成行人...")
        self.spawn_walkers(num_walkers)
        log_with_time("行人生成完成")

    def spawn_vehicles(self, num_vehicles):
        """生成其他车辆"""
        vehicle_bps = self.bp_lib.filter('vehicle.*')
        log_with_time(f"  可用车辆蓝图: {len(vehicle_bps)}")
        
        spawn_points = self.carla_map.get_spawn_points()
        log_with_time(f"  可用生成点: {len(spawn_points)}")
        random.shuffle(spawn_points)

        count = 0
        for i, point in enumerate(spawn_points):
            if count >= num_vehicles:
                break

            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)

            try:
                log_with_time(f"  尝试生成车辆 {count+1}/{num_vehicles}...")
                vehicle = self.world.spawn_actor(bp, point)
                
                log_with_time(f"  车辆生成成功，设置自动驾驶...")
                try:
                    vehicle.set_autopilot(True)
                except Exception as e:
                    log_with_time(f"  自动驾驶设置失败: {e}")
                
                self.vehicles.append(vehicle)
                count += 1
                log_with_time(f"  成功生成车辆 {count}: {bp.id}")
                
            except Exception as e:
                log_with_time(f"  车辆生成失败: {e}")
                continue

        log_with_time(f"  总共生成 {len(self.vehicles)} 辆车辆")

    def spawn_walkers(self, num_walkers):
        """生成行人"""
        walker_bps = self.bp_lib.filter('walker.pedestrian.*')
        log_with_time(f"  可用行人蓝图: {len(walker_bps)}")
        
        if len(walker_bps) == 0:
            log_with_time("  警告：没有找到行人蓝图")
            return

        count = 0
        max_attempts = num_walkers * 5

        for attempt in range(max_attempts):
            if count >= num_walkers:
                break

            try:
                log_with_time(f"  尝试生成行人 {count+1}/{num_walkers}...")
                
                random_location = self.world.get_random_location_from_navigation()
                if random_location is None:
                    log_with_time("  获取位置失败")
                    continue

                bp = walker_bps[attempt % len(walker_bps)]

                spawn_transform = carla.Transform(
                    carla.Location(x=random_location.x, y=random_location.y, z=random_location.z + 0.5),
                    carla.Rotation(yaw=random.uniform(-180, 180))
                )

                walker = self.world.spawn_actor(bp, spawn_transform)
                log_with_time(f"  行人生成成功")

                controller_bp = self.bp_lib.find('controller.ai.walker')
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), walker)
                log_with_time(f"  控制器生成成功")

                controller.start()
                log_with_time(f"  控制器启动成功")
                
                target_location = self.world.get_random_location_from_navigation()
                if target_location is not None:
                    controller.go_to_location(target_location)
                    controller.set_max_speed(1.5 + random.random() * 1.0)
                    log_with_time(f"  设置行走目标")

                self.walkers.append(walker)
                self.walker_controllers.append(controller)
                count += 1
                log_with_time(f"  成功生成行人 {count}: {bp.id}")

            except Exception as e:
                log_with_time(f"  行人生成失败: {e}")
                continue

        log_with_time(f"  总共生成 {len(self.walkers)} 个行人")

def main():
    log_with_time("开始调试")
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    
    log_with_time("创建交通管理器...")
    tm = DebugTrafficManager(world, carla_map, num_vehicles=3, num_walkers=2)
    
    log_with_time("调试完成")

if __name__ == "__main__":
    main()