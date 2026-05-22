import carla
import sys
import time

def main():
    print("=" * 60)
    print("CARLA - Spawn Black Tesla Model 3")
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
        
        print("\n[INFO] Press Ctrl+C to stop and cleanup")
        try:
            while True:
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                speed = ((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5) * 3.6
                print(f"[INFO] Speed: {speed:.1f} km/h | Position: ({location.x:.1f}, {location.y:.1f})")
                time.sleep(1)
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
