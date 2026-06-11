import carla
import math

class AutoOvertake:
    """自动超车系统"""
    
    def __init__(self, vehicle, world, min_distance=15.0, speed_diff=10.0, overtake_speed=50.0):
        self.vehicle = vehicle
        self.world = world
        self.min_distance = min_distance
        self.speed_diff = speed_diff
        self.overtake_speed = overtake_speed
        
        self.overtake_state = "cruise"
        self.front_vehicle = None
        self.front_distance = float('inf')
        self.front_speed = 0.0
        self.change_progress = 0.0
    
    def get_vehicle_speed(self):
        """获取当前速度（km/h）"""
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    
    def detect_front_vehicle(self):
        """检测前方车辆"""
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        forward_vector = vehicle_transform.get_forward_vector()
        
        min_distance = float('inf')
        front_vehicle = None
        
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.id == self.vehicle.id:
                continue
            
            actor_location = actor.get_location()
            dx = actor_location.x - vehicle_location.x
            dy = actor_location.y - vehicle_location.y
            
            distance = math.sqrt(dx**2 + dy**2)
            dot_product = dx * forward_vector.x + dy * forward_vector.y
            
            if dot_product > 0 and distance < 50:
                if distance < min_distance:
                    min_distance = distance
                    front_vehicle = actor
        
        if front_vehicle:
            v = front_vehicle.get_velocity()
            front_speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        else:
            front_speed = 0.0
        
        return front_vehicle, min_distance, front_speed
    
    def check_lane_safety(self, direction):
        """检查相邻车道是否安全"""
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        
        if direction == 'left':
            check_angle = math.radians(vehicle_transform.rotation.yaw - 90)
        else:
            check_angle = math.radians(vehicle_transform.rotation.yaw + 90)
        
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.id == self.vehicle.id:
                continue
            
            actor_location = actor.get_location()
            dx = actor_location.x - vehicle_location.x
            dy = actor_location.y - vehicle_location.y
            
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < 30:
                rel_angle = math.atan2(dy, dx)
                angle_diff = abs(rel_angle - check_angle)
                
                if angle_diff < math.radians(45):
                    return False
        
        return True
    
    def update(self, current_speed):
        """更新超车系统"""
        self.front_vehicle, self.front_distance, self.front_speed = self.detect_front_vehicle()
        
        # 状态机
        if self.overtake_state == "cruise":
            if (self.front_vehicle and 
                self.front_distance < self.min_distance and 
                self.front_speed < current_speed - self.speed_diff):
                self.overtake_state = "detected"
        
        elif self.overtake_state == "detected":
            if self.check_lane_safety('left'):
                self.overtake_state = "changing_left"
                self.change_progress = 0.0
            else:
                self.overtake_state = "waiting"
        
        elif self.overtake_state == "changing_left":
            self.change_progress += 0.02
            if self.change_progress >= 1.0:
                self.overtake_state = "overtaking"
                self.change_progress = 0.0
        
        elif self.overtake_state == "overtaking":
            self.change_progress += 0.03
            if self.change_progress >= 1.0:
                self.overtake_state = "returning"
                self.change_progress = 0.0
        
        elif self.overtake_state == "returning":
            self.change_progress += 0.02
            if self.change_progress >= 1.0:
                self.overtake_state = "completed"
                self.change_progress = 0.0
        
        elif self.overtake_state == "completed":
            self.overtake_state = "cruise"
        
        elif self.overtake_state == "waiting":
            if self.check_lane_safety('left'):
                self.overtake_state = "changing_left"
                self.change_progress = 0.0
        
        # 计算控制
        steer = 0.0
        target_speed = self.overtake_speed
        
        if self.overtake_state == "changing_left":
            steer = -0.3 * (1 - self.change_progress)
            target_speed = 45
        elif self.overtake_state == "overtaking":
            target_speed = self.overtake_speed
        elif self.overtake_state == "returning":
            steer = 0.3 * (1 - self.change_progress)
            target_speed = 45
        
        return steer, target_speed, self.overtake_state
