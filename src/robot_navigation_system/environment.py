import numpy as np
import math
from config import Config

class Obstacle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

class RobotNavigationEnv:
    def __init__(self):
        self.config = Config()
        self.robot_x = self.config.START_POSITION[0]
        self.robot_y = self.config.START_POSITION[1]
        self.robot_theta = 0.0
        self.obstacles = self._generate_obstacles()
        self.target_x = self.config.TARGET_POSITION[0]
        self.target_y = self.config.TARGET_POSITION[1]
        self.step_count = 0
        
    def _generate_obstacles(self):
        obstacles = []
        np.random.seed(42)
        for _ in range(self.config.OBSTACLE_COUNT):
            while True:
                x = np.random.uniform(1, self.config.MAP_WIDTH - 1)
                y = np.random.uniform(1, self.config.MAP_HEIGHT - 1)
                radius = np.random.uniform(self.config.OBSTACLE_MIN_RADIUS, self.config.OBSTACLE_MAX_RADIUS)
                
                dist_to_start = math.hypot(x - self.config.START_POSITION[0], y - self.config.START_POSITION[1])
                dist_to_target = math.hypot(x - self.config.TARGET_POSITION[0], y - self.config.TARGET_POSITION[1])
                
                if dist_to_start > 2 and dist_to_target > 2:
                    obstacles.append(Obstacle(x, y, radius))
                    break
        return obstacles
    
    def reset(self):
        self.robot_x = self.config.START_POSITION[0]
        self.robot_y = self.config.START_POSITION[1]
        self.robot_theta = 0.0
        self.step_count = 0
        return self._get_state()
    
    def _get_lidar_data(self):
        lidar_data = []
        for angle_deg in range(self.config.LIDAR_ANGLES):
            angle_rad = math.radians(angle_deg) + self.robot_theta
            end_x = self.robot_x + self.config.LIDAR_RANGE * math.cos(angle_rad)
            end_y = self.robot_y + self.config.LIDAR_RANGE * math.sin(angle_rad)
            
            min_dist = self.config.LIDAR_RANGE
            
            for obs in self.obstacles:
                dist = self._line_circle_intersection(self.robot_x, self.robot_y, end_x, end_y, obs.x, obs.y, obs.radius)
                if dist < min_dist:
                    min_dist = dist
            
            if min_dist < self.config.LIDAR_RANGE:
                noise = np.random.normal(0, self.config.LIDAR_NOISE)
                lidar_data.append(min_dist + noise)
            else:
                lidar_data.append(self.config.LIDAR_RANGE)
        
        return np.array(lidar_data)
    
    def _line_circle_intersection(self, x1, y1, x2, y2, cx, cy, r):
        dx = x2 - x1
        dy = y2 - y1
        a = dx * dx + dy * dy
        b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
        c = (x1 - cx) ** 2 + (y1 - cy) ** 2 - r ** 2
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return float('inf')
        
        t = (-b - math.sqrt(discriminant)) / (2 * a)
        if 0 <= t <= 1:
            ix = x1 + t * dx
            iy = y1 + t * dy
            return math.hypot(ix - x1, iy - y1)
        
        return float('inf')
    
    def _get_state(self):
        lidar_data = self._get_lidar_data()
        dist_to_target = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        angle_to_target = math.atan2(self.target_y - self.robot_y, self.target_x - self.robot_x) - self.robot_theta
        angle_to_target = math.atan2(math.sin(angle_to_target), math.cos(angle_to_target))
        
        state = np.concatenate([
            lidar_data / self.config.LIDAR_RANGE,
            np.array([dist_to_target / self.config.MAP_WIDTH, 
                     angle_to_target / math.pi,
                     self.robot_x / self.config.MAP_WIDTH,
                     self.robot_y / self.config.MAP_HEIGHT])
        ])
        return state
    
    def step(self, action):
        actions = [
            (self.config.MAX_SPEED, 0),          
            (self.config.MAX_SPEED * 0.5, self.config.MAX_ANGULAR_SPEED),  
            (self.config.MAX_SPEED * 0.5, -self.config.MAX_ANGULAR_SPEED), 
            (0, self.config.MAX_ANGULAR_SPEED),  
            (0, -self.config.MAX_ANGULAR_SPEED)  
        ]
        
        linear_speed, angular_speed = actions[action]
        
        self.robot_theta += angular_speed * 0.1
        self.robot_x += linear_speed * math.cos(self.robot_theta) * 0.1
        self.robot_y += linear_speed * math.sin(self.robot_theta) * 0.1
        
        self.step_count += 1
        
        done = False
        reward = self.config.REWARD_STEP
        
        dist_to_target = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        
        if dist_to_target < 0.5:
            reward += self.config.REWARD_GOAL
            done = True
        elif self._check_collision():
            reward += self.config.REWARD_COLLISION
            done = True
        elif self.step_count >= self.config.MAX_STEPS:
            done = True
        
        return self._get_state(), reward, done, dist_to_target
    
    def _check_collision(self):
        for obs in self.obstacles:
            dist = math.hypot(self.robot_x - obs.x, self.robot_y - obs.y)
            if dist < obs.radius + self.config.ROBOT_RADIUS:
                return True
        
        if self.robot_x < 0 or self.robot_x > self.config.MAP_WIDTH or \
           self.robot_y < 0 or self.robot_y > self.config.MAP_HEIGHT:
            return True
        
        return False
    
    def get_robot_pose(self):
        return self.robot_x, self.robot_y, self.robot_theta
    
    def get_target_pose(self):
        return self.target_x, self.target_y
    
    def get_obstacles(self):
        return self.obstacles