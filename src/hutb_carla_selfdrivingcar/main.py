import carla
import time
from spawn_car import create_vehicle
from cruise_control import get_vehicle_speed, speed_cruise_control

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()

    vehicle = None
    try:
        vehicle = create_vehicle(world, carla_map)
        print("✅ 作业2：定速巡航功能启动")
        target_speed = 30

        for _ in range(500):
            world.tick()
            speed = get_vehicle_speed(vehicle)
            throttle, brake = speed_cruise_control(speed, target_speed)
            ctrl = carla.VehicleControl(throttle=throttle, brake=brake)
            vehicle.apply_control(ctrl)
            time.sleep(0.05)
    finally:
        if vehicle:
            vehicle.destroy()

if __name__ == "__main__":
    main()