import carla
import sys
import time

class PedestrianDetector:
    def __init__(self, vehicle, detection_range=50.0):
        self.vehicle = vehicle
        self.detection_range = detection_range
        self.warning_distance = 20.0
        self.danger_distance = 10.0
        self.is_braking = False
        self.warning_active = False
        print("[INFO] Pedestrian detection system initialized")
        print("[INFO] Detection range: {:.1f}m".format(detection_range))
    
    def detect_pedestrians(self, world):
        vehicle_loc = self.vehicle.get_transform().location
        vehicle_fwd = self.vehicle.get_transform().get_forward_vector()
        
        pedestrians = world.get_actors().filter("*pedestrian*")
        nearby_pedestrians = []
        
        for ped in pedestrians:
            ped_loc = ped.get_transform().location
            distance = vehicle_loc.distance(ped_loc)
            
            if distance < self.detection_range:
                to_ped = ped_loc - vehicle_loc
                dot = vehicle_fwd.x * to_ped.x + vehicle_fwd.y * to_ped.y
                
                if dot > 0:
                    nearby_pedestrians.append((ped, distance))
        
        nearby_pedestrians.sort(key=lambda x: x[1])
        return nearby_pedestrians
    
    def apply_braking(self, intensity=1.0):
        control = carla.VehicleControl()
        control.brake = intensity
        control.throttle = 0.0
        control.steer = 0.0
        self.vehicle.apply_control(control)
        self.is_braking = True
    
    def release_braking(self):
        control = carla.VehicleControl()
        control.throttle = 0.3
        control.brake = 0.0
        control.steer = 0.0
        self.vehicle.apply_control(control)
        self.is_braking = False
    
    def update(self, world):
        pedestrians = self.detect_pedestrians(world)
        
        if pedestrians:
            nearest_ped, distance = pedestrians[0]
            
            if distance < self.danger_distance:
                self.apply_braking(1.0)
                self.warning_active = True
                return "DANGER", distance, nearest_ped.id
            elif distance < self.warning_distance:
                self.apply_braking(0.5)
                self.warning_active = True
                return "WARNING", distance, nearest_ped.id
            else:
                if self.warning_active and self.is_braking:
                    self.release_braking()
                    self.warning_active = False
                return "DETECTED", distance, nearest_ped.id
        else:
            if self.warning_active and self.is_braking:
                self.release_braking()
                self.warning_active = False
            return "CLEAR", 0, None

def spawn_vehicle(world, blueprint_library):
    spawn_points = world.get_map().get_spawn_points()
    tesla_bp = blueprint_library.find("vehicle.tesla.model3")
    
    for spawn_point in spawn_points:
        try:
            vehicle = world.spawn_actor(tesla_bp, spawn_point)
            print("[INFO] Vehicle spawned at ({:.1f}, {:.1f})".format(
                spawn_point.location.x, spawn_point.location.y))
            return vehicle
        except RuntimeError:
            continue
    
    return None

def spawn_pedestrian_ahead(world, blueprint_library, vehicle_transform, offset=0):
    walker_bps = blueprint_library.filter("walker.pedestrian.*")
    if not walker_bps:
        print("[WARNING] No pedestrian blueprints found")
        return None
    
    walker_bp = walker_bps[0]
    
    waypoint = world.get_map().get_waypoint(
        vehicle_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    
    if waypoint:
        forward_loc = vehicle_transform.get_forward_vector()
        target_x = vehicle_transform.location.x + forward_loc.x * 25.0
        target_y = vehicle_transform.location.y + forward_loc.y * 25.0
        
        target_wp = world.get_map().get_waypoint(
            carla.Location(x=target_x, y=target_y, z=vehicle_transform.location.z),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if target_wp:
            spawn_loc = target_wp.transform.location
            spawn_loc.z += 1.0
            spawn_loc.y += offset
            
            spawn_rot = target_wp.transform.rotation
            
            ped_spawn = carla.Transform(spawn_loc, spawn_rot)
            
            try:
                ped = world.spawn_actor(walker_bp, ped_spawn)
                side = "LEFT" if offset < 0 else "RIGHT" if offset > 0 else "CENTER"
                print("[INFO] Pedestrian spawned at ({:.1f}, {:.1f}) - {} side".format(
                    spawn_loc.x, spawn_loc.y, side))
                return ped
            except RuntimeError as e:
                print("[ERROR] Failed to spawn pedestrian: {}".format(e))
                return None
    
    print("[ERROR] No valid waypoint found")
    return None

def main():
    print("=" * 60)
    print("CARLA - Pedestrian Detection System")
    print("=" * 60)
    
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        print("[INFO] Connected to CARLA server")
        
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        print("[INFO] Spawning vehicle...")
        vehicle = spawn_vehicle(world, blueprint_library)
        
        if not vehicle:
            print("[ERROR] Failed to spawn vehicle")
            sys.exit(1)
        
        time.sleep(0.5)
        
        print("[INFO] Spawning pedestrians ahead...")
        pedestrian1 = spawn_pedestrian_ahead(world, blueprint_library, vehicle.get_transform(), offset=-2.0)
        pedestrian2 = spawn_pedestrian_ahead(world, blueprint_library, vehicle.get_transform(), offset=2.0)
        pedestrian3 = spawn_pedestrian_ahead(world, blueprint_library, vehicle.get_transform(), offset=0.0)
        
        pedestrians = [p for p in [pedestrian1, pedestrian2, pedestrian3] if p is not None]
        
        if not pedestrians:
            print("[WARNING] Failed to spawn pedestrians")
        
        detector = PedestrianDetector(vehicle, detection_range=50.0)
        
        print("[INFO] Pedestrian detection system activated")
        print("[INFO] Vehicle will drive straight")
        print("[INFO] 3 pedestrians ahead: left, right, center")
        print("[INFO] Press Ctrl+C to stop")
        
        try:
            while True:
                status, distance, ped_id = detector.update(world)
                
                vel = vehicle.get_velocity()
                speed = ((vel.x**2 + vel.y**2 + vel.z**2) ** 0.5) * 3.6
                
                if not detector.is_braking:
                    control = carla.VehicleControl()
                    control.throttle = 0.4
                    control.brake = 0.0
                    control.steer = 0.0
                    vehicle.apply_control(control)
                
                if status == "DANGER":
                    info = "[DANGER] Pedestrian {} at {:.1f}m - BRAKING!".format(ped_id, distance)
                elif status == "WARNING":
                    info = "[WARNING] Pedestrian {} at {:.1f}m - Slowing down".format(ped_id, distance)
                elif status == "DETECTED":
                    info = "[INFO] Pedestrian {} at {:.1f}m".format(ped_id, distance)
                else:
                    info = "[INFO] Driving straight... | Speed: {:.1f} km/h".format(speed)
                
                print("\r" + info, end="")
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted")
        finally:
            print("\n[INFO] Cleaning up...")
            vehicle.destroy()
            for ped in pedestrians:
                ped.destroy()
            print("[INFO] Done")
            
    except RuntimeError as e:
        print("[ERROR] Runtime error: {}".format(e))
        print("[INFO] Make sure CARLA server is running")
        sys.exit(1)

if __name__ == "__main__":
    main()