import carla
import math

class CollisionDetector:
    """碰撞检测和避免"""

    def __init__(self, world, ego_vehicle):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.collision_sensor = None
        self.collision_history = []
        self._setup_collision_sensor()

    def _setup_collision_sensor(self):
        """设置碰撞传感器"""
        bp_lib = self.world.get_blueprint_library()
        collision_bp = bp_lib.find('sensor.other.collision')

        # 将碰撞传感器附加到车辆上
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            self.ego_vehicle
        )

        # 注册碰撞回调函数
        self.collision_sensor.listen(self._on_collision)
        print("[Collision] 碰撞传感器已启动")

    def _on_collision(self, event):
        """碰撞事件回调"""
        # 记录碰撞信息
        collision_info = {
            'frame': event.frame,
            'time': event.timestamp,
            'actor_id': event.other_actor.id,
            'actor_type': event.other_actor.type_id,
            'impulse': event.normal_impulse,
            'location': event.transform.location
        }
        self.collision_history.append(collision_info)

        actor_type = event.other_actor.type_id
        impulse_magnitude = math.sqrt(
            event.normal_impulse.x ** 2 +
            event.normal_impulse.y ** 2 +
            event.normal_impulse.z ** 2
        )

        print(f"[Collision] 检测到碰撞! 对象: {actor_type}, 冲击力: {impulse_magnitude:.2f}")

    def check_front_collision(self, traffic_manager, safe_distance=10):
        """检查前方碰撞风险"""
        nearby_vehicles = traffic_manager.get_nearby_vehicles(self.ego_vehicle, radius=safe_distance * 2)
        nearby_walkers = traffic_manager.get_nearby_walkers(self.ego_vehicle, radius=safe_distance)

        ego_velocity = self.ego_vehicle.get_velocity()
        ego_speed = math.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2 + ego_velocity.z ** 2)
        ego_transform = self.ego_vehicle.get_transform()
        forward_vector = ego_transform.get_forward_vector()

        collision_risk = False
        closest_obstacle = None
        min_distance = float('inf')

        # 检查前方车辆
        for vehicle_info in nearby_vehicles:
            other_vehicle = vehicle_info['vehicle']
            other_location = vehicle_info['location']
            other_velocity = other_vehicle.get_velocity()
            distance = vehicle_info['distance']

            # 计算相对位置
            direction = carla.Vector3D(
                x=other_location.x - ego_transform.location.x,
                y=other_location.y - ego_transform.location.y,
                z=0
            )

            # 判断是否在前方
            dot_product = forward_vector.x * direction.x + forward_vector.y * direction.y

            if dot_product > 0:  # 在前方
                # 计算相对速度
                relative_speed = ego_speed - math.sqrt(
                    other_velocity.x ** 2 + other_velocity.y ** 2 + other_velocity.z ** 2
                )

                # 预测碰撞时间 (TTC)
                if relative_speed > 0:  # 正在接近
                    ttc = distance / relative_speed
                    if ttc < 3.0:  # 3秒内可能碰撞
                        collision_risk = True
                        if distance < min_distance:
                            min_distance = distance
                            closest_obstacle = {
                                'type': 'vehicle',
                                'distance': distance,
                                'ttc': ttc,
                                'object': other_vehicle
                            }

        # 检查前方行人
        for walker_info in nearby_walkers:
            walker_location = walker_info['location']
            distance = walker_info['distance']

            direction = carla.Vector3D(
                x=walker_location.x - ego_transform.location.x,
                y=walker_location.y - ego_transform.location.y,
                z=0
            )

            dot_product = forward_vector.x * direction.x + forward_vector.y * direction.y

            if dot_product > 0 and distance < safe_distance:
                collision_risk = True
                if distance < min_distance:
                    min_distance = distance
                    closest_obstacle = {
                        'type': 'walker',
                        'distance': distance,
                        'ttc': distance / ego_speed if ego_speed > 0 else 0,
                        'object': walker_info['walker']
                    }

        return collision_risk, closest_obstacle

    def get_avoidance_control(self, obstacle_info, ego_speed):
        """根据障碍物信息计算避撞控制"""
        if obstacle_info is None:
            return carla.VehicleControl(throttle=0.5, brake=0.0)

        distance = obstacle_info['distance']
        ttc = obstacle_info['ttc']

        # 根据距离和时间决定刹车力度
        if distance < 5 or ttc < 1.0:
            # 紧急刹车
            return carla.VehicleControl(throttle=0.0, brake=1.0)
        elif distance < 10 or ttc < 2.0:
            # 中等刹车
            brake_intensity = min(1.0, (10 - distance) / 5)
            return carla.VehicleControl(throttle=0.0, brake=brake_intensity)
        elif distance < 15 or ttc < 3.0:
            # 轻刹车
            brake_intensity = min(0.5, (15 - distance) / 10)
            return carla.VehicleControl(throttle=0.0, brake=brake_intensity)
        else:
            # 正常行驶
            return carla.VehicleControl(throttle=0.5, brake=0.0)

    def get_collision_count(self):
        """获取碰撞次数"""
        return len(self.collision_history)

    def get_collision_history(self):
        """获取碰撞历史"""
        return self.collision_history

    def cleanup(self):
        """清理碰撞传感器"""
        if self.collision_sensor and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
            print("[Collision] 碰撞传感器已关闭")