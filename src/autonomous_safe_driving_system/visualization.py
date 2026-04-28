import carla

def draw_waypoint_path(world, ego_transform, waypoint):
    if waypoint:
        world.debug.draw_line(
            ego_transform.location + carla.Location(z=1.0),
            waypoint.transform.location + carla.Location(z=1.0),
            thickness=0.1,
            color=carla.Color(0, 255, 0),
            life_time=0.1
        )