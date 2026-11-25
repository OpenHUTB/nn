import carla
import sys
import time

class AutoDriveController:
    def __init__(self, client):
        self.client = client
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None

    def spawn_vehicle(self):
        vehicle_bp = self.blueprint_library.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("color", "0,255,255")
        spawn_points = self.map.get_spawn_points()
        if len(spawn_points) == 0:
            print("[AUTO] No spawn points available!")
            return False
        for i, spawn_point in enumerate(spawn_points[:10]):
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                print(f"[AUTO] Vehicle spawned at spawn point {i}! ID: {self.vehicle.id}")
                return True
            except RuntimeError as e:
                if "collision" in str(e).lower():
                    print(f"[AUTO] Spawn point {i} collision, trying next...")
                    continue
                else:
                    raise
        print("[AUTO] All spawn points failed!")
        return False

    def run(self):
        print("\n" + "=" * 60)
        print("[AUTO] CARLA Auto Drive Test")
        print("=" * 60)
        print("[AUTO] Step 1: Spawning vehicle...")
        if not self.spawn_vehicle():
            return False
        time.sleep(0.5)
        print("[AUTO] Step 2: Enabling autopilot...")
        self.vehicle.set_autopilot(True)
        print("[AUTO] Step 3: Vehicle is now driving!")
        print("[AUTO] Press Ctrl+C to stop and cleanup.\n")
        try:
            while True:
                location = self.vehicle.get_location()
                velocity = self.vehicle.get_velocity()
                speed = ((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5) * 3.6
                print(f"[AUTO] Speed: {speed:.1f} km/h | Location: ({location.x:.1f}, {location.y:.1f})")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[AUTO] Test interrupted by user.")
            return True
        except Exception as e:
            print(f"[AUTO] Error during test: {e}")
            return False
        finally:
            self.cleanup()
        return True

    def cleanup(self):
        print("\n[AUTO] Cleaning up...")
        if self.vehicle:
            self.vehicle.destroy()
            print("[AUTO] Vehicle destroyed.")
        print("[AUTO] Cleanup completed!")

def main():
    print("=" * 60)
    print("CARLA Auto Drive System Test")
    print("=" * 60)
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        print("[MAIN] Connected to CARLA server.")
        world = client.get_world()
        print(f"[MAIN] Current map: {world.get_map().name}")
        controller = AutoDriveController(client)
        success = controller.run()
        if success:
            print("\n[MAIN] Test completed successfully!")
        else:
            print("\n[MAIN] Test failed!")
    except RuntimeError as e:
        print(f"[MAIN] Runtime error: {e}")
        print("[MAIN] Make sure CARLA server (CarlaUE4.exe) is running!")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[MAIN] Script terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"[MAIN] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()