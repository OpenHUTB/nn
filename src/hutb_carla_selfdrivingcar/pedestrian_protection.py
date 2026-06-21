import carla
import math

class PedestrianProtectionSystem:
    """行人保护系统（PPS）"""
    
    def __init__(self, vehicle, world, safe_distance=25.0, warning_distance=15.0, danger_distance=8.0):
        self.vehicle = vehicle
        self.world = world
        self.safe_distance = safe_distance      # 安全距离（米）
        self.warning_distance = warning_distance  # 警告距离（米）
        self.danger_distance = danger_distance    # 危险距离（米）
        self.pedestrian_detected = False
    
    def get_vehicle_speed(self):
        """获取当前速度（km/h）"""
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    
    def detect_pedestrian(self):
        """检测前方行人"""
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        forward_vector = vehicle_transform.get_forward_vector()
        
        min_distance = float('inf')
        pedestrian = None
        
        # 检测行人
        for actor in self.world.get_actors().filter('walker.*'):
            actor_location = actor.get_location()
            dx = actor_location.x - vehicle_location.x
            dy = actor_location.y - vehicle_location.y
            
            distance = math.sqrt(dx**2 + dy**2)
            dot_product = dx * forward_vector.x + dy * forward_vector.y
            
            # 只检测前方35米内的行人
            if dot_product > 0 and distance < 35:
                if distance < min_distance:
                    min_distance = distance
                    pedestrian = actor
        
        return pedestrian, min_distance
    
    def get_status(self, distance):
        """获取PPS状态"""
        if distance < self.danger_distance:
            return "危险", "red", "danger"
        elif distance < self.warning_distance:
            return "警告", "yellow", "warning"
        elif distance < self.safe_distance:
            return "注意", "orange", "caution"
        else:
            return "安全", "green", "safe"
    
    def apply_brake(self, intensity):
        """应用刹车控制"""
        control = carla.VehicleControl(
            throttle=0.0,
            brake=max(0.0, min(1.0, intensity)),
            steer=0.0
        )
        self.vehicle.apply_control(control)
    
    def update(self):
        """更新行人保护系统"""
        pedestrian, distance = self.detect_pedestrian()
        
        if pedestrian is None:
            self.pedestrian_detected = False
            return 0.0, float('inf'), "安全", "safe"
        
        self.pedestrian_detected = True
        status, color, status_code = self.get_status(distance)
        
        # 根据距离计算刹车强度
        if distance < self.danger_distance:
            # 危险：全力刹车
            brake_intensity = 1.0
        elif distance < self.warning_distance:
            # 警告：适当减速
            ratio = (distance - self.danger_distance) / (self.warning_distance - self.danger_distance)
            brake_intensity = 0.7 - ratio * 0.3
        else:
            brake_intensity = 0.0
        
        if brake_intensity > 0:
            self.apply_brake(brake_intensity)
        
        return brake_intensity, distance, status, status_code
