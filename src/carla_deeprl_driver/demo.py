import sys
import os
import random
import time
sys.path.insert(0, r'c:\Users\12698\Desktop\carla_deeprl_driver')
os.chdir(r'c:\Users\12698\Desktop\carla_deeprl_driver')

import carla

print("=" * 60)
print("CARLA Demo - RED Tesla TOP-DOWN View")
print("=" * 60)

print("\n[1] Connecting...")
client = carla.Client('localhost', 2000)
client.set_timeout(10)
world = client.get_world()
print(f"    Map: {world.get_map().name}")

print("\n[2] Spawning RED Tesla...")
bp = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

vehicle_bp = bp.filter('vehicle.tesla.model3')[0]
vehicle_bp.set_attribute('color', '255, 0, 0')

spawn_point = random.choice(spawn_points)
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
print("    RED Tesla ready!")

print("\n[3] Setting TOP-DOWN camera (above the car)...")
spectator = world.get_spectator()
print("    Camera looking from above!")

print("\n[4] Running - You should see RED car from above!")
print("-" * 60)

for i in range(30):
    vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))
    time.sleep(0.2)

    v_loc = vehicle.get_transform().location
    v_rot = vehicle.get_transform().rotation

    offset = carla.Location(x=0, y=0, z=50)
    forward_vec = carla.Vector3D(
        x=v_loc.x + offset.x,
        y=v_loc.y + offset.y,
        z=v_loc.z + offset.z
    )

    new_transform = carla.Transform(
        forward_vec,
        carla.Rotation(pitch=-90, yaw=v_rot.yaw, roll=0)
    )
    spectator.set_transform(new_transform)

    if i % 5 == 0:
        print(f"    Step {i+1}/30: ({v_loc.x:.1f}, {v_loc.y:.1f})")

print("-" * 60)
print("\n[DONE] Screenshot!")

vehicle.destroy()