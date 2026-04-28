import carla

def setup_simulation(world, delta_seconds=0.05):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta_seconds
    world.apply_settings(settings)
    return settings

def cleanup_simulation(world, settings, ego_vehicle):
    settings.synchronous_mode = False
    world.apply_settings(settings)
    if ego_vehicle.is_alive:
        ego_vehicle.destroy()