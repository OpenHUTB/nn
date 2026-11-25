import carla
import sys
import time

def main():
    print("=" * 60)
    print("CARLA - Black Tesla Horn Control")
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
        print("[INFO] Horn control enabled")

        print("\n[INFO] Press Ctrl+C to stop")
        print("[INFO] Horn will beep automatically...")
        print("-" * 60)

        try:
            beep_count = 0
            while True:
                beep_count += 1
                control = vehicle.get_control()
                control.horn = True
                vehicle.apply_control(control)
                print(f"\r[INFO] Horn beep #{beep_count}", end="")
                time.sleep(0.5)

                control.horn = False
                vehicle.apply_control(control)
                time.sleep(2.5)

        except KeyboardInterrupt:
            print("\n[INFO] User interrupted")
        finally:
            if vehicle and vehicle.is_alive:
                vehicle.destroy()
                print("[INFO] Vehicle destroyed")

    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print("[INFO] Make sure CarlaUE4.exe is running")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
