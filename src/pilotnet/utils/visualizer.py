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
        
    def draw_vehicle_box(self, color=carla.Color(255, 0, 0), life_time=0.1):
        location = self.vehicle.get_location()
        rotation = self.vehicle.get_transform().rotation
        
        vehicle_transform = carla.Transform(location, rotation)
        extent = self.vehicle.bounding_box.extent
        
        corners = [
            carla.Location(x=extent.x, y=extent.y, z=extent.z),
            carla.Location(x=-extent.x, y=extent.y, z=extent.z),
            carla.Location(x=-extent.x, y=-extent.y, z=extent.z),
            carla.Location(x=extent.x, y=-extent.y, z=extent.z),
            carla.Location(x=extent.x, y=extent.y, z=-extent.z),
            carla.Location(x=-extent.x, y=extent.y, z=-extent.z),
            carla.Location(x=-extent.x, y=-extent.y, z=-extent.z),
            carla.Location(x=extent.x, y=-extent.y, z=-extent.z)
        ]
        
        transformed_corners = [vehicle_transform.transform(corner) for corner in corners]
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        for start, end in edges:
            self.debug.draw_arrow(
                transformed_corners[start],
                transformed_corners[end],
                thickness=0.1,
                arrow_size=0,
                color=color,
                life_time=life_time
            )
    
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
    
    def draw_control_info(self, control, location_offset=carla.Location(x=0, y=0, z=3)):
        location = self.vehicle.get_location() + location_offset
        
        steer_text = f"Steer: {control.steer:.3f}"
        throttle_text = f"Throttle: {control.throttle:.3f}"
        brake_text = f"Brake: {control.brake:.3f}"
        
        self.debug.draw_string(
            location,
            steer_text,
            color=carla.Color(255, 255, 255),
            life_time=0.1
        )
        
        self.debug.draw_string(
            location + carla.Location(z=0.3),
            throttle_text,
            color=carla.Color(0, 255, 0),
            life_time=0.1
        )
        
        self.debug.draw_string(
            location + carla.Location(z=0.6),
            brake_text,
            color=carla.Color(255, 0, 0),
            life_time=0.1
        )
        
        self.control_history.append(control)
    
    def draw_speed_info(self, location_offset=carla.Location(x=0, y=0, z=4)):
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        location = self.vehicle.get_location() + location_offset
        speed_text = f"Speed: {speed:.1f} km/h"
        
        color = carla.Color(0, 255, 255) if speed < 50 else carla.Color(255, 165, 0)
        self.debug.draw_string(
            location,
            speed_text,
            color=color,
            life_time=0.1
        )
        
        self.speed_history.append(speed)
    
    def draw_control_bars(self, location_offset=carla.Location(x=0, y=0, z=5)):
        control = self.vehicle.get_control()
        location = self.vehicle.get_location() + location_offset
        
        bar_length = 2.0
        bar_thickness = 0.1
        
        steer_start = location
        steer_end = location + carla.Location(x=control.steer * bar_length)
        steer_color = carla.Color(255, 255, 0)
        
        self.debug.draw_line(
            steer_start,
            steer_end,
            thickness=bar_thickness,
            color=steer_color,
            life_time=0.1
        )
        
        throttle_start = location + carla.Location(z=0.2)
        throttle_end = throttle_start + carla.Location(x=control.throttle * bar_length)
        throttle_color = carla.Color(0, 255, 0)
        
        self.debug.draw_line(
            throttle_start,
            throttle_end,
            thickness=bar_thickness,
            color=throttle_color,
            life_time=0.1
        )
        
        brake_start = location + carla.Location(z=0.4)
        brake_end = brake_start + carla.Location(x=control.brake * bar_length)
        brake_color = carla.Color(255, 0, 0)
        
        self.debug.draw_line(
            brake_start,
            brake_end,
            thickness=bar_thickness,
            color=brake_color,
            life_time=0.1
        )
    
    def draw_compass(self, location_offset=carla.Location(x=0, y=0, z=6)):
        rotation = self.vehicle.get_transform().rotation
        location = self.vehicle.get_location() + location_offset
        
        directions = ['N', 'E', 'S', 'W']
        for i, direction in enumerate(directions):
            angle = rotation.yaw + i * 90
            rad = np.radians(angle)
            
            offset = carla.Location(
                x=np.cos(rad) * 1.0,
                y=np.sin(rad) * 1.0,
                z=0
            )
            
            text_location = location + offset
            color = carla.Color(255, 255, 255) if direction == 'N' else carla.Color(150, 150, 150)
            
            self.debug.draw_string(
                text_location,
                direction,
                color=color,
                life_time=0.1
            )
        
        center_text = f"Yaw: {rotation.yaw:.1f}°"
        self.debug.draw_string(
            location + carla.Location(z=0.5),
            center_text,
            color=carla.Color(255, 255, 255),
            life_time=0.1
        )
    
    def draw_sensor_info(self, camera_location, location_offset=carla.Location(x=0, y=0, z=7)):
        location = self.vehicle.get_location() + location_offset
        
        self.debug.draw_string(
            location,
            "Camera Active",
            color=carla.Color(0, 255, 255),
            life_time=0.1
        )
        
        self.debug.draw_line(
            self.vehicle.get_location(),
            camera_location,
            thickness=0.05,
            color=carla.Color(0, 255, 255),
            life_time=0.1
        )
    
    def draw_all(self, camera_location=None):
        control = self.vehicle.get_control()
        
        self.draw_vehicle_box()
        self.draw_trajectory()
        self.draw_control_info(control)
        self.draw_speed_info()
        self.draw_control_bars()
        self.draw_compass()
        
        if camera_location:
            self.draw_sensor_info(camera_location)
    
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