import carla

def get_next_waypoint(map, ego_transform, distance=10.0):
    waypoint = map.get_waypoint(
        ego_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if waypoint:
        next_waypoints = waypoint.next(distance)
        if next_waypoints:
            return next_waypoints[0]
    return waypoint