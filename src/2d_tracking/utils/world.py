import carla


def _destroy_vehicles(world: carla.World) -> None:
    """Destroy all vehicle actors in the world.

    Args:
        world: CARLA world instance.
    """
    for npc in world.get_actors().filter('*vehicle*'):
        if npc:
            npc.destroy()


def clear_npc(world: carla.World) -> None:
    """Clear all NPC vehicles from the world.

    Args:
        world: CARLA world instance.
    """
    _destroy_vehicles(world)


def clear_static_vehicle(world: carla.World) -> None:
    """Disable all static vehicle environment objects.

    Args:
        world: CARLA world instance.
    """
    # Retrieve all the objects of the level
    car_objects = world.get_environment_objects(
        carla.CityObjectLabel.Car)
    truck_objects = world.get_environment_objects(
        carla.CityObjectLabel.Truck)
    bus_objects = world.get_environment_objects(
        carla.CityObjectLabel.Bus)

    # Disable all static vehicles
    env_object_ids = [obj.id for obj in (car_objects + truck_objects + bus_objects)]
    world.enable_environment_objects(env_object_ids, False)


def clear(world: carla.World, camera: carla.Sensor) -> None:
    """Clear world settings and destroy all vehicles.

    Args:
        world: CARLA world instance.
        camera: Camera sensor to stop.
    """
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    camera.stop()
    _destroy_vehicles(world)

    print("Vehicles Destroyed.")
