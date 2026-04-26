import carla


def clear_npc(world):
    # Clear existing NPC first
    for npc in world.get_actors().filter('*vehicle*'):
        if npc:
            npc.destroy()


def clear_static_vehicle(world):
    # Retrieve all the objects of the level
    car_objects = world.get_environment_objects(
        carla.CityObjectLabel.Car)  # doesn't have filter by type yet
    truck_objects = world.get_environment_objects(
        carla.CityObjectLabel.Truck)  # doesn't have filter by type yet
    bus_objects = world.get_environment_objects(
        carla.CityObjectLabel.Bus)  # doesn't have filter by type yet

    # Disable all static vehicles
    env_object_ids = []
    for obj in (car_objects + truck_objects + bus_objects):
        env_object_ids.append(obj.id)
    world.enable_environment_objects(env_object_ids, False)


def clear(world, camera):
    settings = world.get_settings()
    settings.synchronous_mode = False  # Disables synchronous mode
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    camera.stop()

    for npc in world.get_actors().filter('*vehicle*'):
        if npc:
            npc.destroy()

    print("Vehicles Destroyed.")
