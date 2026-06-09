"""
CARLA - Black Tesla with Weather Control and Auto Camera Tracking
"""
import carla
import sys
import time


class AutoCamera:
    """Automatically make your CARLA view follow the vehicle"""
    
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.spectator = world.get_spectator()
        
    def follow_vehicle(self, mode="third_person"):
        """Update your view to follow the vehicle
        
        Args:
            mode: "third_person" (default), "first_person", "top_down", or "chase"
        """
        vehicle_transform = self.vehicle.get_transform()
        
        if mode == "third_person":
            spectator_location = (
                vehicle_transform.location + 
                vehicle_transform.get_forward_vector() * -6.0 + 
                carla.Location(z=3.5)
            )
            spectator_rotation = carla.Rotation(
                pitch=vehicle_transform.rotation.pitch - 12.0,
                yaw=vehicle_transform.rotation.yaw,
                roll=vehicle_transform.rotation.roll
            )
        elif mode == "first_person":
            spectator_location = (
                vehicle_transform.location + 
                vehicle_transform.get_forward_vector() * 1.5 + 
                carla.Location(z=1.3)
            )
            spectator_rotation = carla.Rotation(
                pitch=vehicle_transform.rotation.pitch,
                yaw=vehicle_transform.rotation.yaw,
                roll=vehicle_transform.rotation.roll
            )
        elif mode == "top_down":
            spectator_location = vehicle_transform.location + carla.Location(z=25.0)
            spectator_rotation = carla.Rotation(pitch=-90.0, yaw=vehicle_transform.rotation.yaw)
        elif mode == "chase":
            spectator_location = (
                vehicle_transform.location + 
                vehicle_transform.get_forward_vector() * -3.5 + 
                carla.Location(z=2.0)
            )
            spectator_rotation = carla.Rotation(
                pitch=vehicle_transform.rotation.pitch - 8.0,
                yaw=vehicle_transform.rotation.yaw,
                roll=vehicle_transform.rotation.roll
            )
        else:
            spectator_location = (
                vehicle_transform.location + 
                vehicle_transform.get_forward_vector() * -6.0 + 
                carla.Location(z=3.5)
            )
            spectator_rotation = carla.Rotation(
                pitch=vehicle_transform.rotation.pitch - 12.0,
                yaw=vehicle_transform.rotation.yaw,
                roll=vehicle_transform.rotation.roll
            )
        
        spectator_transform = carla.Transform(spectator_location, spectator_rotation)
        self.spectator.set_transform(spectator_transform)


def set_weather(world, weather_type):
    weather = world.get_weather()
    
    if weather_type == "sunny":
        weather.sun_altitude_angle = 75.0
        weather.cloudiness = 0.0
        weather.precipitation = 0.0
        weather.fog_distance = 0.0
        print("[WEATHER] Set to Sunny")
    elif weather_type == "cloudy":
        weather.sun_altitude_angle = 60.0
        weather.cloudiness = 80.0
        weather.precipitation = 0.0
        weather.fog_distance = 0.0
        print("[WEATHER] Set to Cloudy")
    elif weather_type == "rainy":
        weather.sun_altitude_angle = 45.0
        weather.cloudiness = 100.0
        weather.precipitation = 80.0
        weather.fog_distance = 50.0
        print("[WEATHER] Set to Rainy")
    elif weather_type == "foggy":
        weather.sun_altitude_angle = 30.0
        weather.cloudiness = 90.0
        weather.precipitation = 10.0
        weather.fog_distance = 20.0
        print("[WEATHER] Set to Foggy")
    elif weather_type == "night":
        weather.sun_altitude_angle = -30.0
        weather.cloudiness = 30.0
        weather.precipitation = 0.0
        weather.fog_distance = 30.0
        print("[WEATHER] Set to Night")
    
    world.set_weather(weather)


def main():
    print("=" * 60)
    print("CARLA - Black Tesla with Weather Control")
    print("=" * 60)
    
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        print("[INFO] Connected to CARLA server successfully")
        
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        tesla_bp = blueprint_library.find("vehicle.tesla.model3")
        tesla_bp.set_attribute("color", "0, 0, 0")
        
        spawn_points = world.get_map().get_spawn_points()
        
        if len(spawn_points) == 0:
            print("[ERROR] No spawn points available on the map")
            return
        
        vehicle = None
        for i, spawn_point in enumerate(spawn_points[:5]):
            try:
                vehicle = world.spawn_actor(tesla_bp, spawn_point)
                print(f"[SUCCESS] Black Tesla Model 3 spawned at spawn point {i}!")
                print(f"[INFO] Vehicle ID: {vehicle.id}")
                print(f"[INFO] Location: ({spawn_point.location.x:.2f}, {spawn_point.location.y:.2f}, {spawn_point.location.z:.2f})")
                break
            except RuntimeError as e:
                if "collision" in str(e).lower():
                    print(f"[WARN] Spawn point {i} has collision, trying next...")
                    continue
                else:
                    raise
        
        if vehicle is None:
            print("[ERROR] Failed to spawn vehicle at all spawn points")
            return
        
        vehicle.set_autopilot(True)
        print("[INFO] Autopilot enabled - vehicle is driving")
        
        # Set up auto camera tracking
        auto_camera = AutoCamera(world, vehicle)
        print("[INFO] Auto camera tracking enabled - your view follows the car!")
        
        weather_types = ["sunny", "cloudy", "rainy", "foggy", "night"]
        current_weather = 0
        set_weather(world, weather_types[current_weather])
        
        print("\n[INFO] Press Ctrl+C to stop and cleanup")
        print("[INFO] Weather will change every 5 seconds...")
        
        try:
            last_weather_time = time.time()
            while True:
                # Update camera to follow vehicle
                auto_camera.follow_vehicle(mode="third_person")
                
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                speed = ((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5) * 3.6
                print(f"\r[INFO] Speed: {speed:.1f} km/h | Position: ({location.x:.1f}, {location.y:.1f})", end="")
                
                current_time = time.time()
                if current_time - last_weather_time >= 5.0:
                    current_weather = (current_weather + 1) % len(weather_types)
                    set_weather(world, weather_types[current_weather])
                    last_weather_time = current_time
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted the program")
        finally:
            print("[INFO] Cleaning up...")
            vehicle.destroy()
            print("[INFO] Vehicle destroyed successfully")
            
    except RuntimeError as e:
        print(f"[ERROR] Runtime error: {e}")
        print("[INFO] Make sure CARLA server (CarlaUE4.exe) is running")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
