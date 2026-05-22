import carla

def create_vehicle(world, carla_map):
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.tesla.model3')
    spawn_points = carla_map.get_spawn_points()
    for point in spawn_points:
        try:
            vehicle = world.spawn_actor(vehicle_bp, point)
            return vehicle
        except:
            continue
    raise Exception("车辆生成失败")