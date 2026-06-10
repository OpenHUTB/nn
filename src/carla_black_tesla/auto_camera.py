import carla
import sys
import time

class AutoCamera:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.spectator = world.get_spectator()

    def follow(self, mode="third_person"):
        transform = self.vehicle.get_transform()
        
        if mode == "third_person":
            loc = transform.location + transform.get_forward_vector() * -6 + carla.Location(z=3)
            rot = carla.Rotation(pitch=-12, yaw=transform.rotation.yaw)
        elif mode == "first_person":
            loc = transform.location + transform.get_forward_vector() * 1.5 + carla.Location(z=1.3)
            rot = transform.rotation
        elif mode == "top_down":
            loc = transform.location + carla.Location(z=25)
            rot = carla.Rotation(pitch=-90, yaw=transform.rotation.yaw)
        else:
            loc = transform.location + transform.get_forward_vector() * -6 + carla.Location(z=3)
            rot = carla.Rotation(pitch=-12, yaw=transform.rotation.yaw)
        
        self.spectator.set_transform(carla.Transform(loc, rot))

def main():
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        bp_lib = world.get_blueprint_library()
        tesla_bp = bp_lib.find("vehicle.tesla.model3")
        tesla_bp.set_attribute("color", "0, 0, 0")
        
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(tesla_bp, spawn_points[0])
        vehicle.set_autopilot(True)
        
        camera = AutoCamera(world, vehicle)
        print("Auto camera tracking enabled!")
        
        while True:
            camera.follow(mode="third_person")
            time.sleep(0.05)
    except KeyboardInterrupt:
        vehicle.destroy()

if __name__ == "__main__":
    main()