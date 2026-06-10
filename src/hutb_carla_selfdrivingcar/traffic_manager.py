import carla
import random
import math

class TrafficManager:
    """管理交通场景中的其他车辆和行人"""

    def __init__(self, world, carla_map, num_vehicles=10, num_walkers=5):
        self.world = world
        self.carla_map = carla_map
        self.bp_lib = world.get_blueprint_library()
        self.vehicles = []
        self.walkers = []
        self.walker_controllers = []

        # 生成其他车辆和行人
        self.spawn_vehicles(num_vehicles)
        self.spawn_walkers(num_walkers)

    def spawn_vehicles(self, num_vehicles):
        """生成其他车辆"""
        vehicle_bps = self.bp_lib.filter('vehicle.*')
        spawn_points = self.carla_map.get_spawn_points()
        random.shuffle(spawn_points)

        count = 0
        for point in spawn_points:
            if count >= num_vehicles:
                break

            # 随机选择车辆蓝图
            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)

            try:
                vehicle = self.world.spawn_actor(bp, point)
                # 不设置自动驾驶，让车辆静止或手动控制
                self.vehicles.append(vehicle)
                count += 1
                print(f"[Traffic] 生成车辆 {count}/{num_vehicles}: {bp.id}")
            except Exception as e:
                continue

        print(f"[Traffic] 成功生成 {len(self.vehicles)} 辆车辆")

    def spawn_walkers(self, num_walkers):
        """生成行人"""
        walker_bps = self.bp_lib.filter('walker.pedestrian.*')
        print(f"[Traffic] 找到 {len(walker_bps)} 种行人蓝图")
        
        if len(walker_bps) == 0:
            print("[Traffic] 警告：没有找到行人蓝图")
            return

        count = 0
        max_attempts = num_walkers * 5  # 最多尝试次数

        for attempt in range(max_attempts):
            if count >= num_walkers:
                break

            try:
                # 获取导航位置
                random_location = self.world.get_random_location_from_navigation()
                if random_location is None:
                    continue

                # 随机选择行人蓝图
                bp = walker_bps[attempt % len(walker_bps)]  # 依次选择蓝图

                # 设置生成变换
                spawn_transform = carla.Transform(
                    carla.Location(
                        x=random_location.x, 
                        y=random_location.y, 
                        z=random_location.z + 0.5
                    ),
                    carla.Rotation(yaw=random.uniform(-180, 180))
                )

                # 生成行人
                walker = self.world.spawn_actor(bp, spawn_transform)

                # 创建控制器
                controller_bp = self.bp_lib.find('controller.ai.walker')
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), walker)

                # 启动行人控制器
                controller.start()
                
                # 设置随机行走目标
                target_location = self.world.get_random_location_from_navigation()
                if target_location is not None:
                    controller.go_to_location(target_location)
                    controller.set_max_speed(1.5 + random.random() * 1.0)

                self.walkers.append(walker)
                self.walker_controllers.append(controller)
                count += 1
                print(f"[Traffic] 生成行人 {count}/{num_walkers}: {bp.id}")

            except Exception as e:
                # 生成失败，继续尝试
                continue

        print(f"[Traffic] 成功生成 {len(self.walkers)} 个行人")

    def get_nearby_vehicles(self, ego_vehicle, radius=50):
        """获取附近的车辆"""
        ego_location = ego_vehicle.get_location()
        nearby = []

        for vehicle in self.vehicles:
            if vehicle.is_alive:
                distance = ego_location.distance(vehicle.get_location())
                if distance < radius:
                    nearby.append({
                        'vehicle': vehicle,
                        'distance': distance,
                        'location': vehicle.get_location(),
                        'velocity': vehicle.get_velocity()
                    })

        return nearby

    def get_nearby_walkers(self, ego_vehicle, radius=30):
        """获取附近的行人"""
        ego_location = ego_vehicle.get_location()
        nearby = []

        for walker in self.walkers:
            if walker.is_alive:
                distance = ego_location.distance(walker.get_location())
                if distance < radius:
                    nearby.append({
                        'walker': walker,
                        'distance': distance,
                        'location': walker.get_location()
                    })

        return nearby

    def cleanup(self):
        """清理所有生成的交通参与者"""
        print("[Traffic] 正在清理交通参与者...")

        # 清理行人控制器
        for controller in self.walker_controllers:
            if controller.is_alive:
                controller.stop()
                controller.destroy()

        # 清理行人
        for walker in self.walkers:
            if walker.is_alive:
                walker.destroy()

        # 清理车辆
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.destroy()

        print(f"[Traffic] 已清理 {len(self.vehicles)} 辆车辆和 {len(self.walkers)} 个行人")