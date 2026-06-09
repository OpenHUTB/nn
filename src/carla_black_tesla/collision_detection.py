import carla
import sys
import time

class CollisionDetection:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.collision_sensor = None
        self.has_collided = False
        self.collision_time = 0
        self.brake_active = False
        self.collision_count = 0

    def setup_collision_sensor(self, world):
        blueprint = world.get_blueprint_library()
        collision_bp = blueprint.find('sensor.other.collision')
        
        self.collision_sensor = world.spawn_actor(
            collision_bp,
            carla.Transform(carla.Location(z=2.0)),
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        self.has_collided = True
        self.collision_time = time.time()
        self.brake_active = True
        self.collision_count += 1
        
        actor_type = event.other_actor.type_id
        impulse = event.normal_impulse
        print(f"\n[COLLISION #{self.collision_count}] Detected!")
        print(f"  - Object hit: {actor_type}")
        print(f"  - Impact impulse: ({impulse.x:.2f}, {impulse.y:.2f}, {impulse.z:.2f})")
        print("  - Activating emergency brake...")
        
        control = carla.VehicleControl(throttle=0, brake=1.0, steer=0, hand_brake=True)
        self.vehicle.apply_control(control)

    def update(self):
        if self.brake_active:
            velocity = self.vehicle.get_velocity()
            speed = ((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5) * 3.6
            
            if speed < 0.5:
                self.brake_active = False
                print("\n[COLLISION] Vehicle stopped, brake released")

    def is_colliding(self):
        return self.has_collided and self.brake_active

    def reset(self):
        self.has_collided = False
        self.brake_active = False

    def destroy(self):
        if self.collision_sensor:
            self.collision_sensor.destroy()

def main():
    print("=" * 60)
    print("CARLA - Collision Detection System")
    print("=" * 60)
    print("[TIP] Use arrow keys to drive and test collision!")
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
        vehicle = world.spawn_actor(tesla_bp, spawn_points[0])
        
        collision_detector = CollisionDetection(vehicle)
        collision_detector.setup_collision_sensor(world)
        
        print("[INFO] Vehicle spawned")
        print("[INFO] Collision sensor activated")
        print("[INFO] Manual control mode - use WASD/arrow keys")
        print("[INFO] Press Ctrl+C to stop")
        
        control = carla.VehicleControl()
        
        try:
            while True:
                collision_detector.update()
                
                control.throttle = 0.5
                control.steer = 0.0
                control.brake = 0.0
                
                vehicle.apply_control(control)
                
                velocity = vehicle.get_velocity()
                speed = ((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5) * 3.6
                
                status = "[EMERGENCY BRAKE]" if collision_detector.is_colliding() else "[DRIVING]"
                print(f"\r{status} Speed: {speed:.1f} km/h | Collisions: {collision_detector.collision_count}", end="")
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted")
        finally:
            print("\n[INFO] Cleaning up...")
            collision_detector.destroy()
            vehicle.destroy()
            print("[INFO] Done")
            
    except RuntimeError as e:
        print(f"[ERROR] Runtime error: {e}")
        print("[INFO] Make sure CARLA server is running")
        sys.exit(1)

if __name__ == "__main__":
    main()