import carla

def spawn_vehicle(world, carla_map):
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.tesla.model3')
    spawn_points = carla_map.get_spawn_points()

    for point in spawn_points:
        try:
            vehicle = world.spawn_actor(vehicle_bp, point)
            return vehicle
        except RuntimeError:
            continue
    raise RuntimeError("车辆生成失败")