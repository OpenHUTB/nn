import carla
import sys
import time

class PlatoonController:
    def __init__(self, lead_vehicle, follow_vehicle):
        self.lead_vehicle = lead_vehicle
        self.follow_vehicle = follow_vehicle
        
        self.lead_vehicle.set_autopilot(True)
        
        self.target_distance = 8.0
        self.kp_speed = 0.3
        self.kp_distance = 0.1
        self.kp_steer = 0.5
        
        print("[INFO] Lead vehicle ready (ID: {})".format(lead_vehicle.id))
        print("[INFO] Follow vehicle ready (ID: {})".format(follow_vehicle.id))
    
    def get_lead_info(self):
        lead_transform = self.lead_vehicle.get_transform()
        lead_vel = self.lead_vehicle.get_velocity()
        lead_speed = ((lead_vel.x**2 + lead_vel.y**2 + lead_vel.z**2) ** 0.5) * 3.6
        return lead_transform, lead_speed
    
    def get_follow_info(self):
        follow_transform = self.follow_vehicle.get_transform()
        follow_vel = self.follow_vehicle.get_velocity()
        follow_speed = ((follow_vel.x**2 + follow_vel.y**2 + follow_vel.z**2) ** 0.5) * 3.6
        return follow_transform, follow_speed
    
    def calculate_distance(self):
        lead_loc = self.lead_vehicle.get_transform().location
        follow_loc = self.follow_vehicle.get_transform().location
        return lead_loc.distance(follow_loc)
    
    def control_follow(self):
        lead_transform, lead_speed = self.get_lead_info()
        follow_transform, follow_speed = self.get_follow_info()
        
        distance = self.calculate_distance()
        
        speed_error = lead_speed - follow_speed
        distance_error = distance - self.target_distance
        
        target_speed = lead_speed + distance_error * self.kp_distance
        
        throttle = max(0.0, min(1.0, (target_speed - follow_speed) * self.kp_speed))
        
        if distance_error < -2.0:
            brake = min(1.0, -distance_error * 0.2)
            throttle = 0.0
        else:
            brake = 0.0
        
        to_lead = lead_transform.location - follow_transform.location
        forward = follow_transform.get_forward_vector()
        right = carla.Vector3D(-forward.y, forward.x, 0)
        
        cross_track_error = right.dot(to_lead)
        steer = max(-1.0, min(1.0, cross_track_error * self.kp_steer * 0.1))
        
        control = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer
        )
        
        self.follow_vehicle.apply_control(control)
        
        return follow_speed, distance

def spawn_two_vehicles(world, blueprint_library):
    spawn_points = world.get_map().get_spawn_points()
    tesla_bp = blueprint_library.find("vehicle.tesla.model3")
    
    for spawn_point in spawn_points:
        try:
            lead_vehicle = world.spawn_actor(tesla_bp, spawn_point)
            print("[INFO] Lead vehicle spawned")
            
            follow_spawn = carla.Transform(
                carla.Location(
                    x=spawn_point.location.x - 8.0,
                    y=spawn_point.location.y,
                    z=spawn_point.location.z
                ),
                spawn_point.rotation
            )
            
            follow_bp = blueprint_library.find("vehicle.tesla.model3")
            follow_bp.set_attribute("color", "255, 100, 100")
            
            follow_vehicle = world.spawn_actor(follow_bp, follow_spawn)
            print("[INFO] Follow vehicle spawned 8m behind")
            
            return lead_vehicle, follow_vehicle
            
        except RuntimeError:
            continue
    
    return None, None

def main():
    print("=" * 60)
    print("CARLA - Platoon Control System")
    print("=" * 60)
    
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        print("[INFO] Connected to CARLA server")
        
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        print("[INFO] Spawning two vehicles...")
        lead_vehicle, follow_vehicle = spawn_two_vehicles(world, blueprint_library)
        
        if not lead_vehicle or not follow_vehicle:
            print("[ERROR] Failed to spawn vehicles")
            print("[INFO] Try restarting CARLA server")
            sys.exit(1)
        
        platoon = PlatoonController(lead_vehicle, follow_vehicle)
        
        print("[INFO] Platoon control activated")
        print("[INFO] Lead: White Tesla (autopilot)")
        print("[INFO] Follow: Red Tesla (follows lead)")
        print("[INFO] Target distance: 8m")
        print("[INFO] Press Ctrl+C to stop")
        
        try:
            while True:
                lead_vel = lead_vehicle.get_velocity()
                lead_speed = ((lead_vel.x**2 + lead_vel.y**2 + lead_vel.z**2) ** 0.5) * 3.6
                
                follow_speed, distance = platoon.control_follow()
                
                info = "[INFO] Lead: {:5.1f}km/h | Follow: {:5.1f}km/h | Distance: {:4.1f}m".format(
                    lead_speed, follow_speed, distance
                )
                print("\r" + info, end="")
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted")
        finally:
            print("\n[INFO] Cleaning up...")
            lead_vehicle.destroy()
            follow_vehicle.destroy()
            print("[INFO] Done")
            
    except RuntimeError as e:
        print("[ERROR] Runtime error: {}".format(e))
        print("[INFO] Make sure CARLA server is running")
        sys.exit(1)

if __name__ == "__main__":
    main()