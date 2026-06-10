import carla
import math

class LaneChangeAssist:
    """自动变道辅助系统（LCA）"""
    
    def __init__(self, vehicle, world, safe_distance=30.0):
        self.vehicle = vehicle
        self.world = world
        self.safe_distance = safe_distance  # 变道安全距离（米）
        self.lane_change_state = "cruise"  # cruise, left, right, completed
        self.change_progress = 0.0
        self.target_lane = 0  # 0: 当前车道, -1: 左车道, 1: 右车道
    
    def get_vehicle_speed(self):
        """获取当前速度（km/h）"""
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    
    def check_lane_safety(self, direction):
        """检查相邻车道是否安全"""
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        
        # 根据方向确定检查角度
        if direction == 'left':
            check_angle = math.radians(vehicle_transform.rotation.yaw - 90)
        else:  # right
            check_angle = math.radians(vehicle_transform.rotation.yaw + 90)
        
        # 检查相邻车道的车辆
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.id == self.vehicle.id:
                continue
            
            actor_location = actor.get_location()
            dx = actor_location.x - vehicle_location.x
            dy = actor_location.y - vehicle_location.y
            
            distance = math.sqrt(dx**2 + dy**2)
            
            # 检查是否在相邻车道方向
            if distance < self.safe_distance:
                # 计算相对角度
                rel_angle = math.atan2(dy, dx)
                angle_diff = abs(rel_angle - check_angle)
                
                if angle_diff < math.radians(45):
                    return False  # 相邻车道有车，不安全
        
        return True  # 相邻车道安全
    
    def check_all_lanes(self):
        """检查左右车道安全性"""
        left_safe = self.check_lane_safety('left')
        right_safe = self.check_lane_safety('right')
        return left_safe, right_safe
    
    def start_lane_change(self, direction):
        """开始变道"""
        if direction not in ['left', 'right']:
            return False
        
        if not self.check_lane_safety(direction):
            return False
        
        self.lane_change_state = direction
        self.change_progress = 0.0
        self.target_lane = -1 if direction == 'left' else 1
        return True
    
    def update(self):
        """更新变道系统"""
        if self.lane_change_state == "cruise":
            return 0.0, self.lane_change_state, self.change_progress
        
        # 执行变道
        self.change_progress += 0.02
        
        if self.change_progress >= 1.0:
            # 变道完成
            self.lane_change_state = "completed"
            self.change_progress = 1.0
            steer_angle = 0.0
        else:
            # 正在变道，计算转向角度
            if self.lane_change_state == 'left':
                steer_angle = -0.3 * (1 - self.change_progress)
            else:
                steer_angle = 0.3 * (1 - self.change_progress)
        
        return steer_angle, self.lane_change_state, self.change_progress
    
    def reset(self):
        """重置变道状态"""
        self.lane_change_state = "cruise"
        self.change_progress = 0.0
        self.target_lane = 0
