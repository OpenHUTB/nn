import math

def get_vehicle_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def speed_cruise_control(current_speed, target_speed):
    if current_speed < target_speed:
        return 0.5, 0.0
    return 0.0, 0.0