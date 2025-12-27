import carla
import copy
import logging
from collections import deque
from shapely.geometry import Polygon
from gym_carla.multi_lane.agent.global_planner import RoadOption
from gym_carla.multi_lane.util.wrapper import WaypointWrapper, VehicleWrapper
from gym_carla.multi_lane.settings import ROADS, STRAIGHT, CURVE, JUNCTION, DOUBLE_DIRECTION, DISTURB_ROADS
from gym_carla.multi_lane.util.misc import get_lane_center, get_speed, vector, compute_magnitude_angle, \
    is_within_distance_ahead, is_within_distance_rear, draw_waypoints, compute_distance, is_within_distance, \
    test_waypoint, \
    get_trafficlight_trigger_location


#
class LocalPlanner:
    def __init__(self, vehicle,
                 opt_dict={'sampling_resolution': 4.0,
                           'buffer_size': 10,
                           'vehicle_proximity': 50}):
        """
            temporarily used to get front waypoints and vehicle
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._sampling_radius = opt_dict['sampling_resolution']
        self._base_min_distance = 3.0  # This value is tricky

        self._target_waypoint = None
        self._buffer_size = opt_dict['buffer_size']
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        self._waypoints_queue = deque(maxlen=600)
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._target_road_option = RoadOption.LANEFOLLOW
        self._stop_waypoint_creation = False

        self.vehicle_proximity = opt_dict['vehicle_proximity']
        self.traffic_light_proximity = opt_dict['traffic_light_proximity']

        self.waypoints_info = None
        self.lights_info = None
        self.vehicles_info = None

        self._waypoints_queue.append((self._current_waypoint, RoadOption.LANEFOLLOW))
        # self._waypoints_queue.append( (self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))
        # self._compute_next_waypoints(k=200)

    def run_step(self):
        self.waypoints_info = self._get_waypoints()
        self.lights_info = self._get_traffic_lights()
        self.vehicles_info = self._get_vehicles()

        return WaypointWrapper(self.waypoints_info), self.lights_info, VehicleWrapper(self.vehicles_info)

    # def _get_traffic_lights(self):
    #     lights_list = self._world.get_actors().filter("*traffic_light*")
    #     max_distance = self.traffic_light_proximity
    #     ego_vehicle_location = self._vehicle.get_location()
    #     ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

    #     sel_traffic_light = None

    #     if ego_vehicle_waypoint.is_junction:
    #         # It is too late. Do not block the intersection! Keep going!
    #         return sel_traffic_light

    #     # if self._last_traffic_light:
    #     #     if self._last_traffic_light.state != carla.TrafficLightState.Red:
    #     #         self._last_traffic_light = None
    #     #     else:
    #     #         return self._last_traffic_light

    #     for traffic_light in lights_list:
    #         object_location = get_trafficlight_trigger_location(traffic_light)
    #         object_waypoint = self._map.get_waypoint(object_location)

    #         if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
    #             continue

    #         ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
    #         wp_dir = object_waypoint.transform.get_forward_vector()
    #         dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

    #         if dot_ve_wp < 0:
    #             continue

    #         # if traffic_light.state != carla.TrafficLightState.Red:
    #         #     continue

    #         if is_within_distance(object_waypoint.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
    #             sel_traffic_light = traffic_light
    #             self._world.debug.draw_box(traffic_light.trigger_volume,traffic_light.trigger_volume.rotation,life_time=0)
    #             return sel_traffic_light

    #     return sel_traffic_light

    def _get_traffic_lights(self):
        """
        This method is specialized to check US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
            - bool_flag is True if there is a traffic light in RED
            affecting us and False otherwise
            - traffic_light is the object itself or None if there is no
            red traffic light affecting us
        """
        lights_list = self._world.get_actors().filter("*traffic_light*")
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        sel_traffic_light = None

        if ego_vehicle_waypoint.is_junction:
            # It is too late. Do not block the intersection! Keep going!
            return sel_traffic_light

        for traffic_light in lights_list:
            wps = traffic_light.get_stop_waypoints()
            for wp in wps:
                if wp.road_id == ego_vehicle_waypoint.road_id:
                    if wp.lane_id == ego_vehicle_waypoint.lane_id and \
                            wp.transform.location.distance(ego_vehicle_location) <= self.traffic_light_proximity:
                        sel_traffic_light = traffic_light
                        return sel_traffic_light

        return sel_traffic_light

    def _get_vehicles(self):
        # retrieve relevant elements for safe navigation, i.e.: other vehicles
        def caculate_dis(wps, veh):
            if len(wps) == 0:
                return 0
            else:
                if veh:
                    pre_wps = wps[0].previous(self._sampling_radius)
                    pre_wp = None
                    if len(pre_wps) == 1:
                        pre_wp = pre_wps[0]
                    elif len(pre_wps) != 0:
                        for i, wp in enumerate(pre_wps):
                            if wp.road_id in ROADS:
                                pre_wp = wp
                    vehicle_len = max(abs(self._vehicle.bounding_box.extent.x),
                                      abs(self._vehicle.bounding_box.extent.y)) + \
                                  max(abs(veh.bounding_box.extent.x),
                                      abs(veh.bounding_box.extent.y))

                    return max(pre_wp.transform.location.distance(veh.get_location()) - vehicle_len, 0.0001)
                else:
                    return self.vehicle_proximity

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        left_front_veh = self._get_vehicles_one_lane(vehicle_list, True, -1)
        left_rear_veh = self._get_vehicles_one_lane(vehicle_list, False, -1)
        center_front_veh = self._get_vehicles_one_lane(vehicle_list, True, 0)
        center_rear_veh = self._get_vehicles_one_lane(vehicle_list, False, 0)
        right_front_veh = self._get_vehicles_one_lane(vehicle_list, True, 1)
        right_rear_veh = self._get_vehicles_one_lane(vehicle_list, False, 1)

        distance_to_front_vehicles = []
        distance_to_rear_vehicles = []
        distance_to_front_vehicles.append(caculate_dis(self.waypoints_info['left_front_wps'], left_front_veh))
        distance_to_front_vehicles.append(caculate_dis(self.waypoints_info['center_front_wps'], center_front_veh))
        distance_to_front_vehicles.append(caculate_dis(self.waypoints_info['right_front_wps'], right_front_veh))
        distance_to_rear_vehicles.append(caculate_dis(self.waypoints_info['left_rear_wps'], left_rear_veh))
        distance_to_rear_vehicles.append(caculate_dis(self.waypoints_info['center_rear_wps'], center_rear_veh))
        distance_to_rear_vehicles.append(caculate_dis(self.waypoints_info['right_rear_wps'], right_rear_veh))

        return {'left_front_veh': left_front_veh,
                'left_rear_veh': left_rear_veh,
                'center_front_veh': center_front_veh,
                'center_rear_veh': center_rear_veh,
                'right_front_veh': right_front_veh,
                'right_rear_veh': right_rear_veh,
                'dis_to_front_vehs': distance_to_front_vehicles,
                'dis_to_rear_vehs': distance_to_rear_vehicles}

    def _get_vehicles_one_lane(self, vehicle_list, direction=True, lane_offset=0):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        behind our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
        vehicles, which center is actually on a different lane but their
        extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :param direction: True--detect vehicles in front of ego vehicle
                            False--detec vehicles at the back of ego vehicle
        :param lane_offset: the lane relative to current ego vehicle's lane,
            minus value means left, positive value means right
        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_transform = self._vehicle.get_transform()
        ego_vehicle_lane_center = get_lane_center(self._map, ego_vehicle_location)
        if not test_waypoint(ego_vehicle_lane_center):
            return None

        min_distance = self.vehicle_proximity
        vehicle = None
        lane_id = ego_vehicle_lane_center.lane_id - lane_offset
        if lane_id != -1 and lane_id != -2 and lane_id != -3:
            return vehicle

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            # check whether in the same road
            target_lane_center = get_lane_center(self._map, target_vehicle.get_location())
            if target_lane_center.transform.location.distance(
                    target_vehicle.get_location()) > target_lane_center.lane_width / 2 + 0.1:
                continue
            if not test_waypoint(target_vehicle_waypoint):
                continue
            # check whether in the specific lane
            if target_vehicle_waypoint.lane_id != lane_id:
                continue
            # if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
            #         target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
            #     continue

            loc = target_vehicle.get_location()
            if direction:
                if is_within_distance_ahead(loc, ego_vehicle_location, ego_vehicle_transform, self.vehicle_proximity):
                    if ego_vehicle_location.distance(loc) < min_distance:
                        # Return the most close vehicel in front of ego vehicle
                        vehicle = target_vehicle
                        min_distance = ego_vehicle_location.distance(loc)
            else:
                if is_within_distance_rear(loc, ego_vehicle_location, ego_vehicle_transform, self.vehicle_proximity):
                    if ego_vehicle_location.distance(loc) < min_distance:
                        # Return the most close vehicel in front of ego vehicle
                        vehicle = target_vehicle
                        min_distance = ego_vehicle_location.distance(loc)

        return vehicle

    def _get_waypoints(self):
        left_front_wps = None
        left_rear_wps = None
        center_front_wps = None
        center_rear_wps = None
        right_front_wps = None
        right_rear_wps = None

        lane_center = get_lane_center(self._map, self._vehicle.get_location())
        lane_id = lane_center.lane_id
        left = None
        center = lane_center
        right = None
        if lane_id == -1:
            right = center.get_right_lane()
        elif lane_id == -2:
            left = center.get_left_lane()
            right = center.get_right_lane()
        elif lane_id == -3:
            left = center.get_left_lane()
        else:
            lane_center = None
            # logging.error("WAYPOINTS GET BUG")

        left_front_wps = self._get_waypoints_one_lane(left, True)
        left_rear_wps = self._get_waypoints_one_lane(left, False)
        center_front_wps = self._get_waypoints_one_lane(center, True)
        center_rear_wps = self._get_waypoints_one_lane(center, False)
        right_front_wps = self._get_waypoints_one_lane(right, True)
        right_rear_wps = self._get_waypoints_one_lane(right, False)

        return {'left_front_wps': list(left_front_wps),
                'left_rear_wps': list(left_rear_wps),
                'center_front_wps': list(center_front_wps),
                'center_rear_wps': list(center_rear_wps),
                'right_front_wps': list(right_front_wps),
                'right_rear_wps': list(right_rear_wps)}

    def _get_waypoints_one_lane(self, waypoint=None, direction=True):
        """Get the  waypoint list according to ego vehicle's current location,
        direction = True: caculated waypoints in front of current location,
        direction = False: caculated waypoints at the back of current location"""
        _waypoints_queue = deque(maxlen=600)
        if waypoint is not None:
            _waypoints_queue.append(waypoint)
            available_entries = _waypoints_queue.maxlen - len(self._waypoints_queue)
            k = min(available_entries, self._buffer_size)
            for _ in range(k):
                last_waypoint = _waypoints_queue[-1]
                if direction:
                    next_waypoints = list(last_waypoint.next(self._sampling_radius))
                else:
                    next_waypoints = list(last_waypoint.previous(self._sampling_radius))

                if len(next_waypoints) == 0:
                    break
                elif len(next_waypoints) == 1:
                    # only one option available ==> lanefollowing
                    next_waypoint = next_waypoints[0]
                    # road_option = RoadOption.LANEFOLLOW
                else:
                    # road_options_list = self._retrieve_options(
                    #     next_waypoints, last_waypoint)

                    idx = None
                    for i, wp in enumerate(next_waypoints):
                        if wp.road_id in ROADS:
                            next_waypoint = wp
                            idx = i
                    # road_option = road_options_list[idx]

                _waypoints_queue.append(next_waypoint)
            # delete an element from the left
            _waypoints_queue.popleft()
        return list(_waypoints_queue)

    # def _get_waypoints(self):
    #     """
    #     Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
    #     follow the waypoints trajectory.

    #     :param debug: boolean flag to activate waypoints debugging
    #     :return:
    #     """

    #     # not enough waypoints in the horizon? => add more!
    #     if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5) and not self._stop_waypoint_creation:
    #         self._compute_next_waypoints(self._buffer_size * 2)

    #     #   Buffering the waypoints
    #     while len(self._waypoint_buffer) < self._buffer_size:
    #         if self._waypoints_queue:
    #             self._waypoint_buffer.append(
    #                 self._waypoints_queue.popleft())
    #         else:
    #             break

    #     waypoints = []

    #     for i, (waypoint, _) in enumerate(self._waypoint_buffer):
    #         waypoints.append(waypoint)
    #         # waypoints.append([waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.rotation.yaw])

    #     # current vehicle waypoint
    #     self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
    #     # target waypoint
    #     self._target_waypoint, self._target_road_option = self._waypoint_buffer[0]

    #     # purge the queue of obsolete waypoints
    #     # vehicle_transform = self._vehicle.get_transform()
    #     # max_index = -1

    #     # for i, (waypoint, _) in enumerate(self._waypoint_buffer):
    #     #     if distance_vehicle(waypoint, vehicle_transform) < self._min_distance:
    #     #         max_index = i
    #     # if max_index >= 0:
    #     #     for i in range(max_index - 1):
    #     #         self._waypoint_buffer.popleft()

    #     veh_location = self._vehicle.get_location()
    #     veh_speed = get_speed(self._vehicle, False)
    #     settings = self._world.get_settings()
    #     if settings.synchronous_mode:
    #         self._min_distance = self._base_min_distance + settings.fixed_delta_seconds * veh_speed
    #     else:
    #         self._min_distance = self._base_min_distance + 0.5 * veh_speed
    #     num_waypoint_removed = 0
    #     for waypoint, _ in self._waypoint_buffer:

    #         if len(self._waypoints_queue) - num_waypoint_removed == 1:
    #             min_distance = 1  # Don't remove the last waypoint until very close by
    #         else:
    #             min_distance = self._min_distance

    #         if veh_location.distance(waypoint.transform.location) < min_distance:
    #             num_waypoint_removed += 1
    #         else:
    #             break

    #     if num_waypoint_removed > 0:
    #         for _ in range(num_waypoint_removed):
    #             self._waypoint_buffer.popleft()

    #             # lane_center=get_lane_center(self._map,self._vehicle.get_location())
    #     # print(lane_center.road_id,lane_center.lane_id,lane_center.s,sep='\t',end='\n\n')
    #     # for wp,_ in self._waypoint_buffer:
    #     #     print(wp.road_id,wp.lane_id,wp.s,wp.transform.location.distance(lane_center.transform.location),sep='\t')

    #     return waypoints

    def _retrieve_options(self, list_waypoints, current_waypoint):
        """
        Compute the type of connection between the current active waypoint and the multiple waypoints present in
        list_waypoints. The result is encoded as a list of RoadOption enums.

        :param list_waypoints: list with the possible target waypoints in case of multiple options
        :param current_waypoint: current active waypoint
        :return: list of RoadOption enums representing the type of connection from the active waypoint to each
            candidate in list_waypoints
        """
        options = []
        for next_waypoint in list_waypoints:
            # this is needed because something we are linking to
            # the beggining of an intersection, therefore the
            # variation in angle is small
            next_next_waypoint = next_waypoint.next(3.0)[0]
            link = self._compute_connection(current_waypoint, next_next_waypoint)
            options.append(link)

        return options

    def _compute_connection(self, current_waypoint, next_waypoint):
        """
        Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
        (next_waypoint).

        :param current_waypoint: active waypoint
        :param next_waypoint: target waypoint
        :return: the type of topological connection encoded as a RoadOption enum:
            RoadOption.STRAIGHT
            RoadOption.LEFT
            RoadOption.RIGHT
        """
        n = next_waypoint.transform.rotation.yaw
        n = n % 360.0

        c = current_waypoint.transform.rotation.yaw
        c = c % 360.0

        diff_angle = (n - c) % 180.0
        if diff_angle < 1.0:
            return RoadOption.STRAIGHT
        elif diff_angle > 90.0:
            return RoadOption.LEFT
        else:
            return RoadOption.RIGHT

    def get_incoming_waypoint_and_direction(self, steps=3):
        """
        Returns direction and waypoint at a distance ahead defined by the user.

            :param steps: number of steps to get the incoming waypoint.
        """
        if len(self._waypoint_buffer) > steps:
            return self._waypoint_buffer[steps]
        else:
            try:
                wpt, direction = self._waypoint_buffer[-1]
                return wpt, direction
            except IndexError as i:
                return None, RoadOption.VOID

    def set_sampling_redius(self, sampling_resolution):
        self._sampling_radius = sampling_resolution

    def set_min_distance(self, min_distance):
        self._min_distance = min_distance

    def set_global_plan(self, current_plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a new plan to the local planner. A plan must be a list of [carla.Waypoint, RoadOption] pairs
        The 'clean_queue` parameter erases the previous plan if True, otherwise, it adds it to the old one
        The 'stop_waypoint_creation' flag stops the automatic creation of random waypoints

        :param current_plan: list of (carla.Waypoint, RoadOption)
        :param stop_waypoint_creation: bool
        :param clean_queue: bool
        :return:
        """
        if clean_queue:
            self._waypoints_queue.clear()

        # Remake the waypoints queue if the new plan has a higher length than the queue
        new_plan_length = len(current_plan) + len(self._waypoints_queue)
        if new_plan_length > self._waypoints_queue.maxlen:
            new_waypoint_queue = deque(maxlen=new_plan_length)
            for wp in self._waypoints_queue:
                new_waypoint_queue.append(wp)
            self._waypoints_queue = new_waypoint_queue

        for elem in current_plan:
            self._waypoints_queue.append((elem, RoadOption.LANEFOLLOW))

        self._stop_waypoint_creation = stop_waypoint_creation

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                road_options_list = self._retrieve_options(
                    next_waypoints, last_waypoint)

                # # random choice between the possible options
                # road_option = road_options_list[1]
                # #road_option = random.choice(road_options_list)
                # next_waypoint = next_waypoints[road_options_list.index(road_option)]

                idx = None
                for i, wp in enumerate(next_waypoints):
                    if wp.road_id in ROADS:
                        next_waypoint = wp
                        idx = i
                road_option = road_options_list[idx]

            self._waypoints_queue.append((next_waypoint, road_option))