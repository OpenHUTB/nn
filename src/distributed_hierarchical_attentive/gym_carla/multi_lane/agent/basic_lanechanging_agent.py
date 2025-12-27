# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""

import carla
import random
import numpy as np
from enum import Enum
from collections import deque
from shapely.geometry import Polygon
from gym_carla.multi_lane.util.wrapper import Action, ControlInfo
from gym_carla.multi_lane.agent.pid_controller import VehiclePIDController
from gym_carla.multi_lane.util.misc import get_speed, draw_waypoints, is_within_distance, \
    get_trafficlight_trigger_location, \
    compute_distance, get_lane_center


class Basic_Lanechanging_Agent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(self, vehicle, dt=1.0 / 20, opt_dict={}):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param target_speed: speed (in Km/h) at which the vehicle will move
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
        """
        self._vehicle = vehicle
        self._vehicle_location = self._vehicle.get_location()
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._last_traffic_light = None

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicle = False
        self._ignore_change_gap = False
        self.lanechanging_fps = 50

        # PID controller parameter
        self._dt = dt
        self._target_speed = 20.0  # Km/h
        self._args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': self._dt}
        self._args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': self._dt}
        self._max_throt = 0.75
        self._max_brake = 0.3
        self._max_steer = 0.8
        self._offset = 0
        self._base_min_distance = 3.0
        self._follow_speed_limits = False

        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = 10.0  # meters
        self.lane_change_mode = False
        self.last_lane = None
        self.autopilot_step = 0
        self.random_lane_change = True

        # set by carla_env.py
        self.left_wps = []
        self.center_wps = []
        self.right_wps = []
        self.left_rear_wps = []
        self.center_rear_wps = []
        self.right_rear_wps = []

        self.distance_to_left_front = None
        self.distance_to_center_front = None
        self.distance_to_right_front = None
        self.distance_to_left_rear = None
        self.distance_to_center_rear = None
        self.distance_to_right_rear = None

        self.left_next_wayppoint = None
        self.center_next_waypoint = None
        self.right_next_waypoint = None

        self.enable_left_change = True
        self.enable_right_change = True

        # Change parameters according to the dictionary
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'max_steering' in opt_dict:
            self._max_steer = opt_dict['max_steering']
        if 'max_throttle' in opt_dict:
            self._max_throt = opt_dict['max_throttle']
        if 'max_brake' in opt_dict:
            self._max_brake = opt_dict['max_brake']
        if 'buffer_size' in opt_dict:
            self._buffer_size = opt_dict['buffer_size']
        if 'ignore_front_vehicle' in opt_dict:
            self._ignore_vehicle = opt_dict['ignore_front_vehicle']
        if 'ignore_change_gap' in opt_dict:
            self._ignore_change_gap = opt_dict['ignore_change_gap']
        if 'lanechanging_fps' in opt_dict:
            self.lanechanging_fps = opt_dict['lanechanging_fps']
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'random_lane_change' in opt_dict:
            self.random_lane_change = opt_dict['random_lane_change']

        print('ignore_front_vehicle, ignore_change_gap: ', self._ignore_vehicle, self._ignore_change_gap)

        self.left_random_change = []
        self.center_random_change = []
        self.right_random_change = []
        self.init_random_change()

        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=self._args_lateral_dict,
                                                        args_longitudinal=self._args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer)

    def init_random_change(self):
        for i in range(self.lanechanging_fps):
            self.left_random_change.append(Action.LANE_FOLLOW)
            self.center_random_change.append(Action.LANE_FOLLOW)
            # center_random_change.append(0)
            self.right_random_change.append(Action.LANE_FOLLOW)
        self.left_random_change.append(Action.LANE_CHANGE_RIGHT)
        self.center_random_change.append(Action.LANE_CHANGE_RIGHT)
        self.center_random_change.append(Action.LANE_CHANGE_LEFT)
        self.right_random_change.append(Action.LANE_CHANGE_LEFT)

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_info(self, info_dict):
        """
        :param left_wps: waypoints in left-front lane
        :param center_wps: waypoints in center-front lane
        :param right_wps: waypoints in right-front lane
        :param vehicle_inlane: six vehicles in left-front, center-front, right-front, left-rear, center-rear, right-rear
        :return:
        """
        self.left_wps = info_dict['left_wps']
        self.center_wps = info_dict['center_wps']
        self.right_wps = info_dict['right_wps']
        self.left_rear_wps = info_dict['left_rear_wps']
        self.center_rear_wps = info_dict['center_rear_wps']
        self.right_rear_wps = info_dict['right_rear_wps']
        self.distance_to_left_front = info_dict['vehs_info'].distance_to_front_vehicles[0]
        self.distance_to_center_front = info_dict['vehs_info'].distance_to_front_vehicles[1]
        self.distance_to_right_front = info_dict['vehs_info'].distance_to_front_vehicles[2]
        self.distance_to_left_rear = info_dict['vehs_info'].distance_to_rear_vehicles[0]
        self.distance_to_center_rear = info_dict['vehs_info'].distance_to_rear_vehicles[1]
        self.distance_to_right_rear = info_dict['vehs_info'].distance_to_rear_vehicles[2]
        self._vehicle_location = self._vehicle.get_location()

        print('the length of six waypoint queues: ', len(self.left_wps), len(self.center_wps), len(self.right_wps),
              len(self.left_rear_wps),
              len(self.center_rear_wps), len(self.right_rear_wps))
        # For simplicity, we compute s for front vehicles, and compute Euler distance for rear vehicles.
        # set next waypoint that distance == 2m
        # if len(self.left_wps) != 0:
        #     self.left_next_wayppoint = self.left_wps[1]
        # if len(self.center_wps) != 0:
        #     self.center_next_waypoint = self.center_wps[1]
        # if len(self.right_wps) != 0:
        #     self.right_next_waypoint = self.right_wps[1]
        if self._ignore_change_gap:
            if len(self.left_wps) != 0:
                self.enable_left_change = True
            if len(self.right_wps) != 0:
                self.enable_right_change = True
        else:
            self.enable_left_change = False
            self.enable_right_change = False
            if len(self.left_wps) != 0:
                self.enable_left_change = True
            if len(self.right_wps) != 0:
                self.enable_right_change = True
        print("distance enable: ", self.distance_to_left_front, self.distance_to_center_front,
              self.distance_to_right_front, self.distance_to_left_rear, self.distance_to_center_rear,
              self.distance_to_right_rear, self.enable_left_change, self.enable_right_change)

    def run_step(self, current_lane, last_target_lane, last_action, modify_change_steer):
        self.autopilot_step = self.autopilot_step + 1
        """Execute one step of navigation."""
        affected_by_tlight, affected_by_vehicle = False, False
        # Retrieve all relevant actors
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + vehicle_speed
        affected_by_vehicle = self._vehicle_obstacle_detected(max_vehicle_distance)
        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(lights_list, max_tlight_distance)

        new_action, new_target_lane = self._lane_change_action(current_lane, last_target_lane, last_action)

        control = self._PID_run_step(new_action)

        if modify_change_steer:
            if new_action == Action.LANE_CHANGE_LEFT:
                control.steer = np.clip(control.steer, -1, 0)
            elif new_action == Action.LANE_CHANGE_RIGHT:
                control.steer = np.clip(control.steer, 0, 1)
        if affected_by_tlight or (new_action == Action.LANE_FOLLOW and affected_by_vehicle):
            control = self.add_emergency_stop(control)

        return control, new_target_lane, new_action

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def get_step(self):
        return self.autopilot_step

    def get_follow_action(self):
        pass

    def _lane_change_action(self, current_lane, last_target_lane, last_action):
        if self.random_lane_change:
            if current_lane == -2:
                lane_change = random.choice(self.center_random_change)
            elif current_lane == -1:
                lane_change = random.choice(self.left_random_change)
            elif current_lane == -3:
                lane_change = random.choice(self.right_random_change)
            else:
                # just to avoid error, dont work
                lane_change = Action.LANE_FOLLOW
        else:
            if current_lane == -2:
                change_choice = []
                if self.distance_to_left_front - self.distance_to_center_front > 5 and self.distance_to_left_rear > 1:
                    change_choice.append(Action.LANE_CHANGE_LEFT)
                if self.distance_to_right_front - self.distance_to_center_front > 5 and self.distance_to_right_rear > 1:
                    change_choice.append(Action.LANE_CHANGE_RIGHT)
                if len(change_choice) == 0:
                    lane_change = Action.LANE_FOLLOW
                else:
                    lane_change = random.choice(change_choice)
            elif current_lane == -1:
                if self.distance_to_right_front - self.distance_to_center_front > 5 and self.distance_to_right_rear > 1:
                    lane_change = Action.LANE_CHANGE_RIGHT
                else:
                    lane_change = Action.LANE_FOLLOW
            elif current_lane == -3:
                if self.distance_to_left_front - self.distance_to_center_front > 5 and self.distance_to_left_rear > 1:
                    lane_change = Action.LANE_CHANGE_LEFT
                else:
                    lane_change = Action.LANE_FOLLOW
            else:
                # just to avoid error, dont work
                lane_change = Action.LANE_FOLLOW

        if lane_change == Action.LANE_CHANGE_LEFT and not self.enable_left_change:
            lane_change = Action.LANE_FOLLOW
        if lane_change == Action.LANE_CHANGE_RIGHT and not self.enable_right_change:
            lane_change = Action.LANE_FOLLOW

        if self.last_lane:
            if current_lane == self.last_lane:
                if self.lane_change_mode:
                    # still on last lane, change lane behavior not finish
                    new_action = last_action
                    new_target_lane = last_target_lane
                else:
                    # lane follow mode
                    if lane_change != Action.LANE_FOLLOW:
                        self.lane_change_mode = True
                    new_action = lane_change
                    new_target_lane = current_lane - new_action.value
            else:
                # reach dest lane, change lane behavior finish
                new_action = Action.LANE_FOLLOW
                new_target_lane = current_lane
                self.lane_change_mode = False
        else:
            self.lane_change_mode = False
            new_action = Action.LANE_FOLLOW
            new_target_lane = current_lane
        self.last_lane = current_lane

        return new_action, new_target_lane

    def _PID_run_step(self, new_action):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return: control to be applied
        """

        # Purge the queue of obsolete waypoints
        veh_location = self._vehicle.get_location()
        veh_waypoint = get_lane_center(self._map, veh_location)

        vehicle_speed = get_speed(self._vehicle) / 3.6
        lane_center_ratio = 1 - veh_waypoint.transform.location.distance(veh_location) / 4
        self._min_distance = self._base_min_distance * lane_center_ratio
        # print('min_distance: ', self._min_distance)
        next_wp = 1
        if self._min_distance > 1:
            next_wp = 2
        elif self._min_distance > 2:
            next_wp = 3
        elif self._min_distance > 3:
            next_wp = 4
        if new_action == Action.LANE_CHANGE_LEFT:
            target_speed = 50
            self.target_waypoint = self.left_wps[next_wp + 20 - 1]
            # print('left target waypoint: ', self.target_waypoint)
        elif new_action == Action.LANE_FOLLOW:
            target_speed = self._target_speed
            self.target_waypoint = self.center_wps[next_wp + 5 - 1]
            # print('center target waypoint: ', self.target_waypoint)
        elif new_action == Action.LANE_CHANGE_RIGHT:
            target_speed = 50
            self.target_waypoint = self.right_wps[next_wp + 20 - 1]
            # print('right target waypoint: ', self.target_waypoint)

        # print("current location and target location: ", veh_location, self.target_waypoint.transform.location)
        control = self._vehicle_controller.run_step(target_speed, self.target_waypoint)

        return ControlInfo(throttle=control.throttle, brake=control.brake, steer=control.steer, gear=control.gear)

    def _vehicle_obstacle_detected(self, max_dis):
        have_dangerous_vehicle = False
        if self.distance_to_center_front < max_dis and not self._ignore_vehicle:
            have_dangerous_vehicle = True

        return have_dangerous_vehicle

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_location = get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(object_waypoint.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)