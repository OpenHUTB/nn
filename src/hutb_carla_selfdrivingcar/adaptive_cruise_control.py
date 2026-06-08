import carla
import math

class AdaptiveCruiseControl:
    def __init__(self, vehicle, world, max_speed=40, safe_distance=20, follow_distance=40):
        self.vehicle = vehicle
        self.world = world
        self.max_speed = max_speed
        self.safe_distance = safe_distance
        self.follow_distance = follow_distance
    
    def get_vehicle_speed(self):
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    
    def detect_front_vehicle(self):
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        forward_vector = vehicle_transform.get_forward_vector()
        
        min_distance = float('inf')
        
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.id == self.vehicle.id:
                continue
            
            actor_location = actor.get_location()
            dx = actor_location.x - vehicle_location.x
            dy = actor_location.y - vehicle_location.y
            
            distance = math.sqrt(dx**2 + dy**2)
            dot_product = dx * forward_vector.x + dy * forward_vector.y
            
            if dot_product > 0 and distance < self.follow_distance * 2:
                if distance < min_distance:
                    min_distance = distance
        
        return min_distance if min_distance != float('inf') else None
    
    def get_target_speed(self):
        distance = self.detect_front_vehicle()
        
        if distance is None:
            return self.max_speed
        elif distance < self.safe_distance:
            return 0
        elif distance < self.follow_distance:
            ratio = (distance - self.safe_distance) / (self.follow_distance - self.safe_distance)
            return self.max_speed * ratio
        else:
            return self.max_speed