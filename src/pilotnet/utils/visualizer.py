import carla
import numpy as np
from collections import deque

class CarlaVisualizer:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.debug = world.debug
        self.trajectory_points = deque(maxlen=100)
        self.control_history = deque(maxlen=50)
        self.speed_history = deque(maxlen=50)
        
    def draw_trajectory(self, color=carla.Color(0, 255, 0), life_time=5.0):
        location = self.vehicle.get_location()
        self.trajectory_points.append(location)
        
        if len(self.trajectory_points) > 1:
            points = list(self.trajectory_points)
            for i in range(len(points) - 1):
                self.debug.draw_line(
                    points[i],
                    points[i + 1],
                    thickness=0.1,
                    color=color,
                    life_time=life_time
                )
    
    def draw_all(self, camera_location=None):
        control = self.vehicle.get_control()
        
        # 只保留轨迹显示，移除其他跟车显示
        self.draw_trajectory()
        
        # 记录控制和速度数据用于统计
        self.control_history.append(control)
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        self.speed_history.append(speed)
    
    def clear_trajectory(self):
        self.trajectory_points.clear()
    
    def get_statistics(self):
        if not self.control_history or not self.speed_history:
            return {}
        
        avg_speed = np.mean(list(self.speed_history))
        max_speed = np.max(list(self.speed_history))
        
        avg_steer = np.mean([c.steer for c in self.control_history])
        avg_throttle = np.mean([c.throttle for c in self.control_history])
        avg_brake = np.mean([c.brake for c in self.control_history])
        
        return {
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'avg_steer': avg_steer,
            'avg_throttle': avg_throttle,
            'avg_brake': avg_brake,
            'total_frames': len(self.control_history)
        }