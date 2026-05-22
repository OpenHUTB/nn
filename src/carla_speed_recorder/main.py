import carla
import sys
import time
import csv
from datetime import datetime

def main():
    print("=" * 60)
    print("CARLA - Speed Recorder")
    print("=" * 60)

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        print("[INFO] Connected to CARLA server")

        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        tesla_bp = blueprint_library.find("vehicle.tesla.model3")
        tesla_bp.set_attribute("color", "0, 0, 0")

        spawn_points = world.get_map().get_spawn_points()

        vehicle = None
        for i, spawn_point in enumerate(spawn_points[:5]):
            try:
                vehicle = world.spawn_actor(tesla_bp, spawn_point)
                print(f"[SUCCESS] Black Tesla spawned at point {i}!")
                break
            except RuntimeError as e:
                if "collision" in str(e).lower():
                    continue
                else:
                    raise

        if vehicle is None:
            print("[ERROR] Failed to spawn vehicle")
            return

        vehicle.set_autopilot(True)
        print("[INFO] Autopilot enabled")
        print("[INFO] Speed recording started")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speed_log_{timestamp}.csv"

        print(f"\n[INFO] Saving to: {filename}")
        print("[INFO] Press Ctrl+C to stop")
        print("-" * 60)

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Speed (km/h)', 'X', 'Y', 'Z'])

            count = 0
            start_time = time.time()

            try:
                while True:
                    elapsed = time.time() - start_time
                    location = vehicle.get_location()
                    velocity = vehicle.get_velocity()
                    speed = ((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5)
                    speed_kmh = speed * 3.6

                    writer.writerow([f"{elapsed:.1f}", f"{speed_kmh:.1f}", f"{location.x:.2f}", f"{location.y:.2f}", f"{location.z:.2f}"])

                    count += 1
                    print(f"\r[INFO] Recording #{count} | Time: {elapsed:.1f}s | Speed: {speed_kmh:.1f} km/h", end="")
                    
                    time.sleep(0.5)

            except KeyboardInterrupt:
                print("\n[INFO] User interrupted")
            finally:
                if vehicle and vehicle.is_alive:
                    vehicle.destroy()
                    print("[INFO] Vehicle destroyed")
                print(f"[INFO] Saved {count} records to {filename}")

    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print("[INFO] Make sure CarlaUE4.exe is running")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
