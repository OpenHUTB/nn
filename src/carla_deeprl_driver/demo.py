import sys
import os
import random
import time
import math
sys.path.insert(0, r'c:\Users\12698\Desktop\carla_deeprl_driver')
os.chdir(r'c:\Users\12698\Desktop\carla_deeprl_driver')

import carla

print("=" * 60)
print("CARLA Demo - Speed + Reward HUD")
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

print("\n[3] Driving with HUD (check behind car!)")
print("-" * 60)

reward = 0
for i in range(30):
    vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))
    time.sleep(0.15)
    
    velocity = vehicle.get_velocity()
    speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    speed_kmh = speed_ms * 3.6
    
    reward += 1
    
    v_transform = vehicle.get_transform()
    v_loc = v_transform.location
    v_rot = v_transform.rotation
    
    # Update spectator (behind the car - third person view)
    spectator = world.get_spectator()
    behind_offset = carla.Vector3D(
        x=-math.cos(math.radians(v_rot.yaw)) * 8,
        y=-math.sin(math.radians(v_rot.yaw)) * 8,
        z=5
    )
    camera_loc = carla.Location(
        x=v_loc.x + behind_offset.x,
        y=v_loc.y + behind_offset.y,
        z=v_loc.z + behind_offset.z
    )
    spectator.set_transform(carla.Transform(camera_loc, carla.Rotation(pitch=-20, yaw=v_rot.yaw)))
    
    # Draw HUD in front of car (visible from behind camera)
    hud_location = carla.Location(
        x=v_loc.x + math.cos(math.radians(v_rot.yaw)) * 12,
        y=v_loc.y + math.sin(math.radians(v_rot.yaw)) * 12,
        z=v_loc.z + 3
    )
    
    world.debug.draw_string(
        hud_location,
        f"===== SPEED: {speed_kmh:.1f} km/h =====",
        color=carla.Color(255, 255, 0),
        life_time=0.5,
        draw_shadow=True
    )
    
    world.debug.draw_string(
        hud_location + carla.Location(z=1.5),
        f"Reward: +{reward:.1f}",
        color=carla.Color(0, 255, 0),
        life_time=0.5,
        draw_shadow=True
    )

    if i % 5 == 0:
        print(f"    Step {i+1}/30: Speed = {speed_kmh:.1f} km/h, Reward = {reward}")

print("\n[DONE] Check in front of car (visible from behind)!")
vehicle.destroy()