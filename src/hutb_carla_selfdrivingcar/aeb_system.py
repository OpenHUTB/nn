import carla
import math

class AEBSystem:
    """自动紧急刹车系统（AEB）"""
    
    def __init__(self, vehicle, world, warning_distance=15.0, emergency_distance=5.0):
        self.vehicle = vehicle
        self.world = world
        self.warning_distance = warning_distance  # 警告距离（米）
        self.emergency_distance = emergency_distance  # 紧急制动距离（米）
        self.aeb_triggered = False
        self.brake_intensity = 0.0
        
    def get_vehicle_speed(self):
        """获取当前速度（km/h）"""
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    
    def detect_front_obstacle(self):
        """检测前方障碍物"""
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        forward_vector = vehicle_transform.get_forward_vector()
        
        min_distance = float('inf')
        obstacle = None
        
        # 检测车辆
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.id == self.vehicle.id:
                continue
            
            actor_location = actor.get_location()
            dx = actor_location.x - vehicle_location.x
            dy = actor_location.y - vehicle_location.y
            
            distance = math.sqrt(dx**2 + dy**2)
            dot_product = dx * forward_vector.x + dy * forward_vector.y
            
            if dot_product > 0 and distance < 50:  # 只检测前方50米内
                if distance < min_distance:
                    min_distance = distance
                    obstacle = actor
        
        # 检测行人
        for actor in self.world.get_actors().filter('walker.*'):
            actor_location = actor.get_location()
            dx = actor_location.x - vehicle_location.x
            dy = actor_location.y - vehicle_location.y
            
            distance = math.sqrt(dx**2 + dy**2)
            dot_product = dx * forward_vector.x + dy * forward_vector.y
            
            if dot_product > 0 and distance < 30:  # 只检测前方30米内
                if distance < min_distance:
                    min_distance = distance
                    obstacle = actor
        
        return obstacle, min_distance
    
    def get_aeb_status(self, distance):
        """获取AEB状态"""
        if distance < self.emergency_distance:
            return "紧急制动", "red"
        elif distance < self.warning_distance:
            return "预制动", "yellow"
        else:
            return "正常", "green"
    
    def apply_brake(self, intensity):
        """应用刹车控制"""
        self.brake_intensity = max(0.0, min(1.0, intensity))
        control = carla.VehicleControl(
            throttle=0.0,
            brake=self.brake_intensity,
            steer=0.0
        )
        self.vehicle.apply_control(control)
    
    def update(self):
        """更新AEB系统"""
        obstacle, distance = self.detect_front_obstacle()
        
        if obstacle is None or distance > self.warning_distance:
            # 无障碍物或距离安全，不刹车
            self.aeb_triggered = False
            return 0.0, distance, "正常"
        
        # 计算刹车强度
        if distance < self.emergency_distance:
            # 紧急制动
            brake_intensity = 1.0
            self.aeb_triggered = True
            status = "紧急制动"
        elif distance < self.warning_distance:
            # 预制动
            ratio = (distance - self.emergency_distance) / (self.warning_distance - self.emergency_distance)
            brake_intensity = 1.0 - ratio * 0.5
            self.aeb_triggered = False
            status = "预制动"
        else:
            brake_intensity = 0.0
            status = "正常"
        
        self.apply_brake(brake_intensity)
        
        return brake_intensity, distance, status
