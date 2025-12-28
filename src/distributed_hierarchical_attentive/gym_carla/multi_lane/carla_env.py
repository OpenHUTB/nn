import time
import carla
import random
import logging
import pygame
import math, copy
import numpy as np
from enum import Enum
from queue import Queue
from collections import deque
from gym_carla.multi_lane.util.render import World, HUD
from gym_carla.multi_lane.agent.basic_agent import BasicAgent
from gym_carla.multi_lane.agent.local_planner import LocalPlanner
from gym_carla.multi_lane.agent.global_planner import GlobalPlanner, RoadOption
from gym_carla.multi_lane.agent.basic_lanechanging_agent import Basic_Lanechanging_Agent
# from gym_carla.single_lane.navigation.constant_velocity_agent import ConstantVelocityAgent
from gym_carla.multi_lane.util.sensor import CollisionSensor, LaneInvasionSensor, SemanticTags
from gym_carla.multi_lane.util.wrapper import WaypointWrapper, VehicleWrapper, Action, SpeedState, Truncated, \
    ControlInfo, process_veh, \
    process_steer, recover_steer, fill_action_param, ttc_reward, comfort, lane_center_reward, \
    calculate_guide_lane_center, process_lane_wp
from gym_carla.multi_lane.util.misc import draw_waypoints, get_speed, get_acceleration, test_waypoint, \
    compute_distance, get_actor_polygons, get_lane_center, remove_unnecessary_objects, get_yaw_diff, \
    get_trafficlight_trigger_location, is_within_distance, get_sign, is_within_distance_ahead, get_projection, \
    create_vehicle_blueprint


#
class CarlaEnv:
    def __init__(self, args) -> None:
        super().__init__()
        self.host = args.host
        self.port = args.port
        self.tm_port = args.tm_port
        self.sync = args.sync
        self.fps = args.fps
        self.no_rendering = args.no_rendering
        self.ego_filter = args.filter
        self.loop = args.loop
        self.agent = args.agent
        # arguments for debug
        self.debug = args.debug
        self.train = args.train  # argument indicating training agent
        self.seed = args.seed
        self.behavior = args.behavior
        self.num_of_vehicles = args.num_of_vehicles
        self.sampling_resolution = args.sampling_resolution
        self.min_distance = args.min_distance
        self.vehicle_proximity = args.vehicle_proximity
        self.traffic_light_proximity = args.traffic_light_proximity
        self.hybrid = args.hybrid
        self.auto_lanechange = args.auto_lane_change
        self.guide_change = args.guide_change
        self.stride = args.stride
        self.buffer_size = 160000
        if self.train:
            self.pre_train_steps = args.pre_train_steps
        else:
            self.pre_train_steps = 0
        self.speed_limit = args.speed_limit
        self.lane_change_reward = args.lane_change_reward
        # The RL agent acts only after ego vehicle speed reach speed threshold
        self.speed_threshold = args.speed_threshold
        self.speed_min = args.speed_min
        # controller action space
        self.steer_bound = args.steer_bound
        self.throttle_bound = args.throttle_bound
        self.brake_bound = args.brake_bound
        self.modify_change_steer = args.modify_change_steer
        self.ignore_traffic_light = args.ignore_traffic_light

        logging.info('listening to server %s:%s', args.host, args.port)
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.sim_world = self.client.load_world(args.map)
        remove_unnecessary_objects(self.sim_world)
        self.map = self.sim_world.get_map()
        self.origin_settings = self.sim_world.get_settings()
        self.traffic_manager = None
        self.speed_state = SpeedState.START
        # Set fixed simulation step for synchronous mode
        self._set_synchronous_mode()
        self._set_traffic_manager()
        logging.info('Carla server connected')

        # init pygame window
        self.pygame = args.pygame and not self.no_rendering
        self.width, self.height = [int(x) for x in args.res.split('x')]
        if self.pygame:
            self._init_renderer()

        # Record the time of total steps
        self.reset_step = 0
        self.total_step = 0
        self.time_step = 0
        self.rl_control_step = 0
        # RL_switch: True--currently RL in control, False--currently PID in control
        self.RL_switch = False
        self.SWITCH_THRESHOLD = args.switch_threshold
        self.switch_count = 0
        self.lights_info = None
        self.last_light_state = None
        self.wps_info = WaypointWrapper()
        self.vehs_info = VehicleWrapper()
        self.control = ControlInfo()
        # self.control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0,reverse=False, manual_gear_shift=False, gear=1)

        self.last_lane, self.current_lane = None, None
        self.last_action, self.current_action = Action.LANE_FOLLOW, Action.LANE_FOLLOW
        self.last_target_lane, self.current_target_lane = None, None

        self.calculate_impact = None

        # generate ego vehicle spawn points on chosen route
        self.global_planner = GlobalPlanner(self.map, self.sampling_resolution)
        self.local_planner = None
        self.spawn_points = self.global_planner.get_spawn_points()

        # arguments for caculating reward
        self.TTC_THRESHOLD = args.TTC_th
        self.penalty = args.penalty
        self.lane_penalty = args.lane_penalty
        self.last_acc = 0  # ego vehicle acceration along s in last step
        self.last_yaw = carla.Vector3D()
        self.vel_buffer = deque(maxlen=10)
        self.rear_vel_deque = deque(maxlen=2)
        self.step_info = None

        if self.debug:
            # draw_waypoints(self.sim_world,self.global_panner.get_route())
            random.seed(self.seed)

        # Set weather
        # self.sim_world.set_weather(carla.WeatherParamertes.ClearNoon)

        self.companion_vehicles = []
        self.vehicle_polygons = []
        self.ego_vehicle = None
        self.ego_spawn_point = None

        # Collision sensor
        self.collision_sensor = None
        self.lane_invasion_sensor = None

        # thread blocker
        self.sensor_queue = Queue(maxsize=10)
        self.camera = None

    def __del__(self):
        logging.info('\n Destroying all vehicles')
        self.sim_world.apply_settings(self.origin_settings)
        self._clear_actors(['vehicle.*', 'sensor.other.collison', 'sensor.camera.rgb', 'sensor.other.lane_invasion'])

    def reset(self):
        if self.ego_vehicle is not None:
            self._clear_actors(
                ['*vehicle.*', 'sensor.other.collision', 'sensor.camera.rgb', 'sensor.other.lane_invasion'])
            self.ego_vehicle = None
            self.vehicle_polygons.clear()
            self.companion_vehicles.clear()
            self.collision_sensor = None
            self.lane_invasion_sensor = None
            self.camera = None
            self.vel_buffer.clear()
            self.step_info.clear()
            while (self.sensor_queue.empty() is False):
                self.sensor_queue.get(block=False)
            if self.pygame:
                self.world.destroy()
                pygame.quit()
        else:
            self.step_info = {}

        # Spawn surrounding vehicles
        self._spawn_companion_vehicles()
        self.calculate_impact = 0
        self.rear_vel_deque.append(-1)
        self.rear_vel_deque.append(-1)
        # Get actors polygon list
        vehicle_poly_dict = get_actor_polygons(self.sim_world, 'vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)

        # try to spawn ego vehicle
        while self.ego_vehicle is None:
            self.ego_spawn_point = random.choice(self.spawn_points)
            self.ego_vehicle = self._try_spawn_ego_vehicle_at(self.ego_spawn_point)
        # self.ego_vehicle.set_simulate_physics(False)
        self.collision_sensor = CollisionSensor(self.ego_vehicle)
        self.lane_invasion_sensor = LaneInvasionSensor(self.ego_vehicle)
        # friction_bp=self.sim_world.get_blueprint_library().find('static.trigger.friction')
        # bb_extent=self.ego_vehicle.bounding_box.extent
        # friction_bp.set_attribute('friction',str(0.0))
        # friction_bp.set_attribute('extent_x',str(bb_extent.x))
        # friction_bp.set_attribute('extent_y',str(bb_extent.y))
        # friction_bp.set_attribute('extent_z',str(bb_extent.z))
        # self.sim_world.spawn_actor(friction_bp,self.ego_vehicle.get_transform())
        # self.sim_world.debug.draw_box()

        # let the client interact with server
        if self.sync:
            if self.pygame:
                self._init_renderer()
                self.world.restart(self.ego_vehicle)
            else:
                spectator = self.sim_world.get_spectator()
                transform = self.ego_vehicle.get_transform()
                spectator.set_transform(carla.Transform(transform.location + carla.Location(z=100),
                                                        carla.Rotation(pitch=-90)))
                self.sim_world.tick()
        else:
            self.sim_world.wait_for_tick()

        """Attention:
        get_location() Returns the actor's location the client recieved during last tick. The method does not call the simulator.
        Hence, upon initializing, the world should first tick before calling get_location, or it could cause fatal bug"""
        # self.ego_vehicle.get_location()

        # add route planner for ego vehicle
        self.local_planner = LocalPlanner(self.ego_vehicle, {'sampling_resolution': self.sampling_resolution,
                                                             'buffer_size': self.buffer_size,
                                                             'vehicle_proximity': self.vehicle_proximity,
                                                             'traffic_light_proximity': self.traffic_light_proximity})
        # self.local_planner.set_global_plan(self.global_planner.get_route(
        #      self.map.get_waypoint(self.ego_vehicle.get_location())))
        self.current_lane = get_lane_center(self.map, self.ego_vehicle.get_location()).lane_id
        self.last_lane = self.current_lane
        self.last_target_lane, self.current_target_lane = self.current_lane, self.current_lane
        self.last_action, self.current_action = Action.LANE_FOLLOW, Action.LANE_FOLLOW
        self.last_light_state = None

        self.wps_info, self.lights_info, self.vehs_info = self.local_planner.run_step()

        self._ego_autopilot(True)

        # Only use RL controller after ego vehicle speed reach speed_threshold
        self.speed_state = SpeedState.START
        self.control_sigma = {'Steer': random.choice([0]),
                              'Throttle_brake': random.choice([0])}
        # self.control_sigma = {'Steer': random.choice([0, 0.05,0.1, 0.1, 0.15]),
        #                     'Throttle_brake': random.choice([0, 0.05, 0.1, 0.1, 0.15])}
        self.autopilot_controller = Basic_Lanechanging_Agent(self.ego_vehicle, dt=1.0 / self.fps,
                                                             opt_dict={
                                                                 'ignore_traffic_lights': self.ignore_traffic_light,
                                                                 'ignore_stop_signs': True,
                                                                 'sampling_resolution': self.sampling_resolution,
                                                                 'max_steering': self.steer_bound,
                                                                 'max_throttle': self.throttle_bound,
                                                                 'max_brake': self.brake_bound,
                                                                 'buffer_size': self.buffer_size,
                                                                 'target_speed': self.speed_limit,
                                                                 'ignore_front_vehicle': False,
                                                                 'ignore_change_gap': False,
                                                                 #                    'ignore_front_vehicle': random.choice([False,True]),
                                                                 #                    'ignore_change_gap': random.choice([True, True, False]),
                                                                 'lanechanging_fps': random.choice([40, 50, 60]),
                                                                 'random_lane_change': False})
        #                    'random_lane_change':random.choice([False,True,True,True])})
        # self.controller = ConstantVelocityAgent(self.ego_vehicle,target_speed=self.speed_limit)
        # self.controller = BasicAgent(self.ego_vehicle, {'target_speed': self.speed_threshold, 'dt': 1 / self.fps,
        #                                                 'max_throttle': self.throttle_bound,
        #                                                 'max_brake': self.brake_bound})

        # code for synchronous mode
        # camera_bp = self.sim_world.get_blueprint_library().find('sensor.camera.rgb')
        # camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        # self.camera = self.sim_world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        # self.camera.listen(lambda image: self._sensor_callback(image, self.sensor_queue))

        # speed state switch
        if not self.debug:
            if self.total_step - self.rl_control_step < self.pre_train_steps:
                # During pre-train steps, let rl and pid alternatively take control
                if self.RL_switch:
                    if self.switch_count >= self.SWITCH_THRESHOLD:
                        self.RL_switch = False
                        self.switch_count = 0
                    else:
                        self.switch_count += 1
                else:
                    self.RL_switch = True
                    self.switch_count += 1
            else:
                self.RL_switch = True
        else:
            self.RL_switch = False
            # self.sim_world.debug.draw_point(self.ego_spawn_point.location,size=0.3,life_time=0)
            # while (True):
            #     spawn_point=random.choice(self.spawn_points).location
            #     if self.map.get_waypoint(spawn_point).lane_id==self.map.get_waypoint(self.ego_spawn_point.location).lane_id:
            #         break
            # self.controller.set_destination(spawn_point)

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # return state information
        return self._get_state()

    def step(self, a_index, action):
        self.autopilot_controller.set_info({'left_wps': self.wps_info.left_front_wps,
                                            'center_wps': self.wps_info.center_front_wps,
                                            'right_wps': self.wps_info.right_front_wps,
                                            'left_rear_wps': self.wps_info.left_rear_wps,
                                            'center_rear_wps': self.wps_info.center_rear_wps,
                                            'right_rear_wps': self.wps_info.right_rear_wps,
                                            'vehs_info': self.vehs_info})
        self.step_info.clear()
        self.lights_info = None
        self.control.steer, self.control.throttle, self.control.brake, self.control.gear = 0.0, 0.0, 0.0, 1
        self.wps_info = WaypointWrapper()
        self.vehs_info = VehicleWrapper()
        """throttle (float):A scalar value to control the vehicle throttle [0.0, 1.0]. Default is 0.0.
                steer (float):A scalar value to control the vehicle steering [-1.0, 1.0]. Default is 0.0.
                brake (float):A scalar value to control the vehicle brake [0.0, 1.0]. Default is 0.0."""
        if not self.modify_change_steer:
            self.control.steer = np.clip(action[0][0], -self.steer_bound, self.steer_bound)
        else:
            self.control.steer = float(process_steer(a_index, action[0][0]))
        if action[0][1] >= 0:
            self.control.brake = 0
            self.control.throttle = np.clip(action[0][1], 0, self.throttle_bound)
        else:
            self.control.throttle = 0
            self.control.brake = np.clip(abs(action[0][1]), 0, self.brake_bound)
        print(f"Steer--After Process:{self.control.steer}, After Recovery:{recover_steer(a_index, self.control.steer)}")
        # control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake),hand_brake=False,
        #                                reverse=False,manual_gear_shift=True,gear=1)

        # Only use RL controller after ego vehicle speed reach speed_threshold
        # Use DFA to calculate different speed state transition
        if not self.debug:
            self._speed_switch(a_index)
        else:
            self._speed_switch(a_index)
            # if self.controller.done() and self.loop:
            #     while (True):
            #         spawn_point=random.choice(self.spawn_points).location
            #         if self.map.get_waypoint(spawn_point).lane_id==self.map.get_waypoint(self.ego_spawn_point.location).lane_id:
            #             break
            #     self.controller.set_destination(spawn_point)
            # control = self.controller.run_step()
            # print("debug mode: last_lane, current lane, last target lane, current target lane, last action, current action: ",
            #       self.last_lane, self.current_lane, self.last_target_lane, self.current_target_lane, self.last_action.value,self.current_action.value)
            # self.control, self.current_target_lane, self.current_action= \
            #     self.autopilot_controller.run_step(self.last_lane, self.current_lane,self.current_target_lane, self.last_action,self.modify_change_steer)
            # print("debug mode: last_lane, current lane, last target lane, current target lane, last action, current action: ",
            #       self.last_lane, self.current_lane, self.last_target_lane, self.current_target_lane, self.last_action.value,self.current_action.value)
        if self.sync:
            if not self.debug:
                if not self.RL_switch:
                    # Add noise to autopilot controller's control command
                    # print(f"Basic Agent Control Before Noise:{control}")
                    if not self.modify_change_steer:
                        self.control.steer = np.clip(np.random.normal(self.control.steer, self.control_sigma['Steer']),
                                                     -self.steer_bound, self.steer_bound)
                    else:
                        if self.current_action == Action.LANE_CHANGE_LEFT:
                            self.control.steer = np.clip(
                                np.random.normal(self.control.steer, self.control_sigma['Steer']),
                                -self.steer_bound, 0)
                        elif self.current_action == Action.LANE_CHANGE_RIGHT:
                            self.control.steer = np.clip(
                                np.random.normal(self.control.steer, self.control_sigma['Steer']),
                                0, self.steer_bound)
                        else:
                            # LANE_FOLLOW and STOP mode
                            self.control.steer = np.clip(
                                np.random.normal(self.control.steer, self.control_sigma['Steer']),
                                -self.steer_bound, self.steer_bound)
                    if self.control.throttle > 0:
                        throttle_brake = self.control.throttle
                    else:
                        throttle_brake = -self.control.brake
                    throttle_brake = np.clip(np.random.normal(throttle_brake, self.control_sigma['Throttle_brake']),
                                             -self.brake_bound, self.throttle_bound)
                    if throttle_brake > 0:
                        self.control.throttle = throttle_brake
                        self.control.brake = 0
                    else:
                        self.control.throttle = 0
                        self.control.brake = abs(throttle_brake)
                if self.is_effective_action():
                    con = carla.VehicleControl(throttle=self.control.throttle, steer=self.control.steer,
                                               brake=self.control.brake, hand_brake=False, reverse=self.control.reverse,
                                               manual_gear_shift=self.control.manual_gear_shift, gear=self.control.gear)
                    self.ego_vehicle.apply_control(con)
            else:
                if self.is_effective_action():
                    con = carla.VehicleControl(throttle=self.control.throttle, steer=self.control.steer,
                                               brake=self.control.brake, hand_brake=False, reverse=self.control.reverse,
                                               manual_gear_shift=self.control.manual_gear_shift, gear=self.control.gear)
                    self.ego_vehicle.apply_control(con)

            # print(self.map.get_waypoint(self.ego_vehicle.get_location(),False),self.ego_vehicle.get_transform(),sep='\n')
            # print(self.sim_world.get_snapshot().timestamp)
            if self.pygame:
                self._tick()
            else:
                spectator = self.sim_world.get_spectator()
                transform = self.ego_vehicle.get_transform()
                spectator.set_transform(carla.Transform(transform.location + carla.Location(z=80),
                                                        carla.Rotation(pitch=-90)))
                self.sim_world.tick()
            # camera_data = self.sensor_queue.get(block=True)

            """Attention: the server's tick function only returns after it ran a fixed_delta_seconds, so the client need not to wait for
            the server, the world snapshot of tick returned already include the next state after the uploaded action."""
            # print(self.map.get_waypoint(self.ego_vehicle.get_location(),False),self.ego_vehicle.get_transform(),sep='\n')
            # print(self.sim_world.get_snapshot().timestamp)
            # print()
            cont = self.ego_vehicle.get_control()
            self.control.throttle, self.control.brake, self.control.steer = cont.throttle, cont.brake, cont.steer
            self.control.gear, self.control.manual_gear_shift = cont.gear, cont.manual_gear_shift
            lane_center = get_lane_center(self.map, self.ego_vehicle.get_location())
            self.current_lane = lane_center.lane_id
            # print(self.ego_vehicle.get_speed_limit(),get_speed(self.ego_vehicle,False),get_acceleration(self.ego_vehicle,False),sep='\t')
            # route planner
            self.wps_info, self.lights_info, self.vehs_info = self.local_planner.run_step()
            if self.last_light_state == carla.TrafficLightState.Red and self.lights_info and self.last_light_state != self.lights_info.state:
                # light state change during steps, from red to green
                self.vel_buffer.clear()
            # marks=lane_center.get_landmarks(self.traffic_light_proximity)
            # if marks:
            #     for mark in marks:
            #         print(f"Mark Road ID:{mark.road_id}, distance:{mark.distance}, name:{mark.distance}")
            print(
                "After Tick: last_lane, current_lane, last_target_lane, current_target_lane, last action, current action: ",
                self.last_lane, self.current_lane, self.last_target_lane, self.current_target_lane,
                self.last_action.value, self.current_action.value)
            print("Actual Control, change: ", cont, self.current_action.value)

            if self.debug:
                # draw_waypoints(self.sim_world, [self.next_wps[0]], 60, z=1)
                draw_waypoints(self.sim_world, self.wps_info.center_front_wps + self.wps_info.center_rear_wps + \
                               self.wps_info.left_front_wps + self.wps_info.left_rear_wps + self.wps_info.right_front_wps + self.wps_info.right_rear_wps,
                               1.0 / self.fps + 0.001, z=1)
            else:
                # draw_waypoints(self.sim_world, self.wps_info.center_front_wps+self.wps_info.center_rear_wps+\
                #     self.wps_info.left_front_wps+self.wps_info.left_rear_wps+self.wps_info.right_front_wps+self.wps_info.right_rear_wps,
                #     1.0 / self.fps + 0.001, z=1)
                pass

            temp = []
            if self.vehs_info.left_rear_veh is not None:
                temp.append(get_speed(self.vehs_info.left_rear_veh, False))
            else:
                temp.append(-1)
            if self.vehs_info.center_rear_veh is not None:
                temp.append(get_speed(self.vehs_info.center_rear_veh, False))
            else:
                temp.append(-1)
            if self.vehs_info.right_rear_veh is not None:
                temp.append(get_speed(self.vehs_info.right_rear_veh, False))
            else:
                temp.append(-1)
            self.rear_vel_deque.append(temp)

            """Attention: The sequence of following code is pivotal, do not recklessly change their execution order"""
            state = self._get_state()
            reward = self._get_reward()
            truncated = self._truncated()
            done = self._done(truncated)
            self.step_info.update({'Reward': reward})

            # update last step info
            yaw_forward = lane_center.transform.get_forward_vector().make_unit_vector()
            a_3d = self.ego_vehicle.get_acceleration()
            self.last_acc, a_t = get_projection(a_3d, yaw_forward)
            self.last_yaw = self.ego_vehicle.get_transform().get_forward_vector()
            self.last_action = self.current_action
            self.last_lane = self.current_lane
            self.last_target_lane = self.current_target_lane
            if self.lights_info:
                self.last_light_state = self.lights_info.state
            else:
                self.last_light_state = None
        else:
            temp = self.sim_world.wait_for_tick()
            self.sim_world.on_tick(lambda _: {})
            time.sleep(1.0 / self.fps)
            reward, state, truncated, done, control_info = None, None, Truncated.FALSE, None, None

        if self.debug:
            self.time_step += 1
            self.RL_switch = False
            print(
                f"Speed:{get_speed(self.ego_vehicle, False)}, Acc:{get_acceleration(self.ego_vehicle, False)}, Time_step:{self.time_step}")
            return state, reward, truncated != Truncated.FALSE, done, self._get_info()

        print(f"Current State:{self.speed_state}, RL In Control:{self.RL_switch}")
        if not self.RL_switch:
            print(
                f"Control Sigma -- Steer:{self.control_sigma['Steer']}, Throttle_brake:{self.control_sigma['Throttle_brake']}")
        if self.is_effective_action():
            # update timesteps
            self.time_step += 1
            self.total_step += 1
            self.vel_buffer.append(self.step_info['velocity'])
            if self.RL_switch == True:
                self.rl_control_step += 1
            # new_action \in [-1, 0, 1], but saved action is the index of max Q(s, a), and thus change \in [0, 1, 2]
            control_info = {'Steer': self.control.steer, 'Throttle': self.control.throttle, 'Brake': self.control.brake,
                            'Change': self.current_action, 'control_state': self.RL_switch}

            l_c = self.map.get_waypoint(self.ego_vehicle.get_location())
            print(
                f"Episode:{self.reset_step}, Total_step:{self.total_step}, Time_step:{self.time_step}, RL_control_step:{self.rl_control_step}\n"
                f"Vel: {self.step_info['velocity']}, Current Acc:{self.step_info['cur_acc']}, Last Acc:{self.step_info['last_acc']}\n"
                f"Light State: {self.lights_info.state if self.lights_info else None}, Light Distance:{state['light'][2] * self.traffic_light_proximity}, "
                f"Cur Road ID: {lane_center.road_id}, Cur Lane ID: {lane_center.lane_id}, Before Process Road ID: {l_c.road_id}, Lane ID: {l_c.lane_id}\n"
                f"Steer:{control_info['Steer']}, Throttle:{control_info['Throttle']}, Brake:{control_info['Brake']}\n"
                f"Reward:{self.step_info['Reward']}, Speed Limit:{self.ego_vehicle.get_speed_limit() * 3.6}, Abandon:{self.step_info['Abandon']}")
            if truncated == Truncated.FALSE:
                print(
                    f"TTC:{self.step_info['fTTC']}, Comfort:{self.step_info['Comfort']}, Efficiency:{self.step_info['Efficiency']}, "
                    f"Impact: {self.step_info['impact']}, Change_in_lane_follow:{self.step_info['change_in_lane_follow']}, \n"
                    f"Off-Lane:{self.step_info['offlane']}, fLcen:{self.step_info['Lane_center']}, "
                    f"Yaw_change:{self.step_info['yaw_change']}, Yaw_diff:{self.step_info['yaw_diff']}, fYaw:{self.step_info['Yaw']}")
            # print(f"Steer:{control_info['Steer']}, Throttle:{control_info['Throttle']}, Brake:{control_info['Brake']}\n")

            return state, reward, truncated != Truncated.FALSE, done, self._get_info(control_info)
        else:
            return state, reward, truncated != Truncated.FALSE, done, self._get_info()

    def get_observation_space(self):
        """
        :return:
        """
        """Get observation space of cureent environment"""
        return {'waypoints': 10, 'ego_vehicle': 6, 'companion_vehicle': 3, 'light': 3}

    def get_action_bound(self):
        """Return action bound of ego vehicle controller"""
        return {'steer': self.steer_bound, 'throttle': self.throttle_bound, 'brake': self.brake_bound}

    def is_effective_action(self):
        # testing if current ego vehcle's action should be put into replay buffer
        return self.speed_state == SpeedState.RUNNING

    def seed(self, seed=None):
        return

    def render(self, mode):
        pass

    def _get_state(self):
        """return a tuple: the first element is next waypoints, the second element is vehicle_front information"""

        left_wps = self.wps_info.left_front_wps
        center_wps = self.wps_info.center_front_wps
        right_wps = self.wps_info.right_front_wps

        lane_center = get_lane_center(self.map, self.ego_vehicle.get_location())
        right_lane_dis = lane_center.get_right_lane().transform.location.distance(self.ego_vehicle.get_location())
        ego_t = lane_center.lane_width / 2 + lane_center.get_right_lane().lane_width / 2 - right_lane_dis

        ego_vehicle_z = lane_center.transform.location.z
        ego_forward_vector = self.ego_vehicle.get_transform().get_forward_vector()
        my_sample_ratio = self.buffer_size // 10
        center_wps_processed = process_lane_wp(center_wps, ego_vehicle_z, ego_forward_vector, my_sample_ratio, 0)
        if len(left_wps) == 0:
            left_wps_processed = center_wps_processed.copy()
            for left_wp in left_wps_processed:
                left_wp[2] = -1
        else:
            left_wps_processed = process_lane_wp(left_wps, ego_vehicle_z, ego_forward_vector, my_sample_ratio, -1)
        if len(right_wps) == 0:
            right_wps_processed = center_wps_processed.copy()
            for right_wp in right_wps_processed:
                right_wp[2] = 1
        else:
            right_wps_processed = process_lane_wp(right_wps, ego_vehicle_z, ego_forward_vector, my_sample_ratio, 1)

        left_wall = False
        if len(left_wps) == 0:
            left_wall = True
        right_wall = False
        if len(right_wps) == 0:
            right_wall = True
        vehicle_inlane_processed = process_veh(self.ego_vehicle, self.vehs_info, left_wall, right_wall,
                                               self.vehicle_proximity)

        yaw_diff_ego = math.degrees(get_yaw_diff(lane_center.transform.get_forward_vector(),
                                                 self.ego_vehicle.get_transform().get_forward_vector()))

        yaw_forward = lane_center.transform.get_forward_vector()
        v_3d = self.ego_vehicle.get_velocity()
        v_s, v_t = get_projection(v_3d, yaw_forward)

        a_3d = self.ego_vehicle.get_acceleration()
        a_s, a_t = get_projection(a_3d, yaw_forward)

        if self.lights_info:
            wps = self.lights_info.get_stop_waypoints()
            stop_dis = 1.0
            for wp in wps:
                if wp.road_id == lane_center.road_id and wp.lane_id == lane_center.lane_id:
                    stop_dis = wp.transform.location.distance(
                        lane_center.transform.location) / self.traffic_light_proximity
                    break
            if (
                    self.lights_info.state == carla.TrafficLightState.Red or self.lights_info.state == carla.TrafficLightState.Yellow):
                light = [0, 1, stop_dis]
            else:
                light = [1, 0, stop_dis]
        else:
            stop_dis = 1.0
            light = [1, 0, stop_dis]

        """Attention:
        Upon initializing, there are some bugs in the theta_v and theta_a, which could be greater than 90,
        this might be caused by carla."""
        self.step_info.update({'velocity': v_s, 'last_acc': self.last_acc, 'cur_acc': a_s})
        # update informatino for rear vehicle
        if self.vehs_info.center_rear_veh is None or \
                (self.lights_info is not None and self.lights_info.state != carla.TrafficLightState.Green):
            self.step_info.update({'rear_id': -1, 'rear_v': 0, 'rear_a': 0, 'time_step': self.time_step + 1,
                                   'change_lane': self.current_lane != self.last_lane})
        else:
            lane_center = get_lane_center(self.map, self.vehs_info.center_rear_veh.get_location())
            yaw_forward = lane_center.transform.get_forward_vector()
            v_3d = self.vehs_info.center_rear_veh.get_velocity()
            v_s, v_t = get_projection(v_3d, yaw_forward)
            a_3d = self.vehs_info.center_rear_veh.get_acceleration()
            a_s, a_t = get_projection(a_3d, yaw_forward)
            self.step_info.update({'rear_id': self.vehs_info.center_rear_veh.id,
                                   'rear_v': v_s, 'rear_a': a_s, 'time_step': self.time_step + 1,
                                   'change_lane': self.current_lane != self.last_lane})

        return {'left_waypoints': left_wps_processed, 'center_waypoints': center_wps_processed,
                'right_waypoints': right_wps_processed, 'vehicle_info': vehicle_inlane_processed,
                'ego_vehicle': [v_s / 10, v_t / 10, a_s / 3, a_t / 3, ego_t, yaw_diff_ego / 90],
                'light': light}

    def _get_reward(self):
        """Calculate the step reward:
        TTC: Time to collide with front vehicle
        Eff: Ego vehicle efficiency, speed ralated
        Com: Ego vehicle comfort, ego vehicle acceration change rate
        Lcen: Distance between ego vehicle location and lane center
        """
        truncated = self._truncated()
        self.step_info['Abandon'] = False
        if truncated != Truncated.FALSE:
            if truncated == Truncated.CHANGE_LANE_IN_LANE_FOLLOW:
                return -self.lane_penalty
            elif truncated == Truncated.COLLISION:
                history, tags = self.collision_sensor.get_collision_history()
                if SemanticTags.Car in tags or SemanticTags.Truck in tags or SemanticTags.Bus in tags or SemanticTags.Motorcycle in tags \
                        or SemanticTags.Rider in tags or SemanticTags.Bicycle in tags:
                    return -self.penalty
                else:
                    # Abandon the experience that ego vehicle collide with other obstacle
                    self.step_info['Abandon'] = True
            else:
                return -self.penalty

        ttc, fTTC = ttc_reward(self.ego_vehicle, self.vehs_info.center_front_veh, self.min_distance, self.TTC_THRESHOLD)

        lane_center = get_lane_center(self.map, self.ego_vehicle.get_location())
        yaw_forward = lane_center.transform.get_forward_vector().make_unit_vector()

        v_3d = self.ego_vehicle.get_velocity()
        v_s, v_t = get_projection(v_3d, yaw_forward)
        speed_1, speed_2 = self.speed_limit, self.speed_limit
        # if self.lights_info and self.lights_info.state!=carla.TrafficLightState.Green:
        #     wps=self.lights_info.get_stop_waypoints()
        #     for wp in wps:
        #         if wp.lane_id==lane_center.lane_id:
        #             dis=self.ego_vehicle.get_location().distance(wp.transform.location)
        #             if dis<self.traffic_light_proximity:
        #                 speed_1=(dis+0.0001)/self.traffic_light_proximity*self.speed_limit
        max_speed = min(speed_1, speed_2)
        if v_s * 3.6 > max_speed:
            # fEff = 1
            fEff = math.exp(max_speed - v_s * 3.6)
        else:
            fEff = v_s * 3.6 / max_speed
        # if max_speed<self.speed_min:
        #     fEff=1

        a_3d = self.ego_vehicle.get_acceleration()
        cur_acc, a_t = get_projection(a_3d, yaw_forward)

        fCom, yaw_change = comfort(self.fps, self.last_acc, cur_acc, self.last_yaw,
                                   self.ego_vehicle.get_transform().get_forward_vector())
        # jerk = (cur_acc.x - self.last_acc.x) ** 2 / (1.0 / self.fps) + (cur_acc.y - self.last_acc.y) ** 2 / (
        #         1.0 / self.fps)
        # jerk = ((cur_acc.x - self.last_acc.x) * self.fps) ** 2 + ((cur_acc.y - self.last_acc.y) * self.fps) ** 2
        # # whick still requires further testing, longitudinal and lateral
        # fCom = -jerk / ((6 * self.fps) ** 2 + (12 * self.fps) ** 2)

        if self.guide_change:
            Lcen, fLcen = calculate_guide_lane_center(self.ego_vehicle.get_location(), lane_center,
                                                      self.ego_vehicle.get_location(),
                                                      self.vehs_info.distance_to_front_vehicles,
                                                      self.vehs_info.distance_to_rear_vehicles)
        else:
            Lcen, fLcen = lane_center_reward(lane_center, self.ego_vehicle.get_location())

        yaw_diff = math.degrees(get_yaw_diff(lane_center.transform.get_forward_vector(),
                                             self.ego_vehicle.get_transform().get_forward_vector()))
        fYaw = -abs(yaw_diff) / 90

        impact = 0
        if self.calculate_impact != 0:
            last_rear_vel = self.rear_vel_deque[0][1]
            current_rear_vel = self.rear_vel_deque[1][1]
            if last_rear_vel == -1 or current_rear_vel == -1:
                impact = 0
            else:
                if current_rear_vel < last_rear_vel:
                    impact = (current_rear_vel - last_rear_vel) * self.fps
            self.calculate_impact = 0

        # reward for lane_changing
        lane_changing_reward = self._lane_change_reward(self.last_action, self.last_lane, self.current_lane,
                                                        self.current_action,
                                                        self.vehs_info.distance_to_front_vehicles,
                                                        self.vehs_info.distance_to_rear_vehicles)
        # flag: In the lane follow mode, the ego vehicle pass the lane
        change_in_lane_follow = self.current_action == 0 and self.current_lane != self.last_lane

        self.step_info.update({'offlane': Lcen, 'yaw_diff': yaw_diff, 'TTC': ttc, 'fTTC': fTTC, 'Comfort': fCom,
                               'Efficiency': fEff, 'Lane_center': fLcen, 'Yaw': fYaw, 'yaw_change': yaw_change,
                               'lane_changing_reward': lane_changing_reward, 'impact': impact,
                               'change_in_lane_follow': change_in_lane_follow})

        return fTTC + fEff + fCom + fLcen + lane_changing_reward

    def _lane_change_reward(self, last_action, last_lane, current_lane, current_action, distance_to_front_vehicles,
                            distance_to_rear_vehicles):
        print('distance_to_front_vehicles, distance_to_rear_vehicles: ', distance_to_front_vehicles,
              distance_to_rear_vehicles)
        # still the distances of the last time step
        reward = 0
        if current_action == Action.LANE_FOLLOW:
            # if change lane in lane following mode, we set this reward=0, but will be truncated
            return reward
        if current_lane - last_lane == -1:
            # change right
            self.calculate_impact = 1
            center_front_dis = distance_to_front_vehicles[0]
            right_front_dis = distance_to_front_vehicles[1]
            dis = right_front_dis - center_front_dis
            reward = dis / self.vehicle_proximity * self.lane_change_reward
            # if right_front_dis > center_front_dis:
            #     reward = min((right_front_dis / center_front_dis - 1) * self.lane_change_reward, self.lane_change_reward)
            # else:
            #     reward = max((right_front_dis / center_front_dis - 1) * self.lane_change_reward, -self.lane_change_reward)
            # reward = 0
            ttc, rear_ttc_reward = ttc_reward(self.vehs_info.center_rear_veh, self.ego_vehicle, self.min_distance,
                                              self.TTC_THRESHOLD)
            # add rear_ttc_reward?
            print('lane change reward and rear ttc reward: ', reward, rear_ttc_reward)
        elif current_lane - last_lane == 1:
            # change left
            self.calculate_impact = -1
            center_front_dis = distance_to_front_vehicles[2]
            left_front_dis = distance_to_front_vehicles[1]
            dis = left_front_dis - center_front_dis
            reward = dis / self.vehicle_proximity * self.lane_change_reward
            # if left_front_dis > center_front_dis:
            #     reward = min((left_front_dis / center_front_dis - 1) * self.lane_change_reward, self.lane_change_reward)
            # else:
            #     reward = max((left_front_dis / center_front_dis - 1) * self.lane_change_reward, -self.lane_change_reward)
            # reward = 0
            ttc, rear_ttc_reward = ttc_reward(self.vehs_info.center_rear_veh, self.ego_vehicle, self.min_distance,
                                              self.TTC_THRESHOLD)
            print('lane change reward and rear ttc reward: ', reward, rear_ttc_reward)

        return reward

    def _truncated(self):
        """Calculate whether to terminate the current episode"""
        lane_center = get_lane_center(self.map, self.ego_vehicle.get_location())
        yaw_diff = math.degrees(get_yaw_diff(lane_center.transform.get_forward_vector(),
                                             self.ego_vehicle.get_transform().get_forward_vector()))

        if len(self.collision_sensor.get_collision_history()[0]) != 0:
            # Here we judge speed state because there might be collision event when spawning vehicles
            logging.warn('collison happend')
            return Truncated.COLLISION
        if not test_waypoint(lane_center, False):
            logging.warn('vehicle drive out of road')
            return Truncated.OUT_OF_ROAD
        if self.current_action == Action.LANE_FOLLOW and self.current_lane != self.last_lane:
            logging.warn('change lane in lane following mode')
            return Truncated.CHANGE_LANE_IN_LANE_FOLLOW
        if self.current_action == Action.LANE_CHANGE_LEFT and self.current_lane - self.last_lane < 0:
            logging.warn('vehicle change to wrong lane')
            return Truncated.CHANGE_TO_WRONG_LANE
        if self.current_action == Action.LANE_CHANGE_RIGHT and self.current_lane - self.last_lane > 0:
            logging.warn('vehicle change to wrong lane')
            return Truncated.CHANGE_TO_WRONG_LANE
        if self.speed_state != SpeedState.START and not self.vehs_info.center_front_veh:
            if not self.lights_info or self.lights_info.state != carla.TrafficLightState.Red:
                if len(self.vel_buffer) == self.vel_buffer.maxlen:
                    avg_vel = 0
                    for vel in self.vel_buffer:
                        avg_vel += vel / self.vel_buffer.maxlen
                    if avg_vel * 3.6 < self.speed_min:
                        logging.warn('vehicle speed too low')
                        return Truncated.SPEED_LOW

        # if self.lane_invasion_sensor.get_invasion_count()!=0:
        #     logging.warn('lane invasion occur')
        #     return True
        # if self.step_info['Lane_center'] <=-1.0:
        #     logging.warn('drive out of road, lane invasion occur')
        #     return True
        if abs(yaw_diff) > 90:
            logging.warn('moving in opposite direction')
            return Truncated.OPPOSITE_DIRECTION
        if self.lights_info and self.lights_info.state != carla.TrafficLightState.Green:
            self.sim_world.debug.draw_point(self.lights_info.get_location(), size=0.3, life_time=0)
            wps = self.lights_info.get_stop_waypoints()
            for wp in wps:
                self.sim_world.debug.draw_point(wp.transform.location, size=0.1, life_time=0)
                if is_within_distance_ahead(self.ego_vehicle.get_location(), wp.transform.location, wp.transform,
                                            self.min_distance):
                    logging.warn('break traffic light rule')
                    return Truncated.TRAFFIC_LIGHT_BREAK

        return Truncated.FALSE

    def _done(self, truncated):
        if truncated != Truncated.FALSE:
            return False
        if self.wps_info.center_front_wps[2].transform.location.distance(
                self.ego_spawn_point.location) < self.sampling_resolution:
            # The local planner's waypoint list has been depleted
            logging.info('vehicle reach destination, simulation terminate')
            return True
        if self.wps_info.left_front_wps and \
                self.wps_info.left_front_wps[2].transform.location.distance(
                    self.ego_spawn_point.location) < self.sampling_resolution:
            # The local planner's waypoint list has been depleted
            logging.info('vehicle reach destination, simulation terminate')
            return True
        if self.wps_info.right_front_wps and \
                self.wps_info.right_front_wps[2].transform.location.distance(
                    self.ego_spawn_point.location) < self.sampling_resolution:
            # The local planner's waypoint list has been depleted
            logging.info('vehicle reach destination, simulation terminate')
            return True
        if not self.RL_switch:
            if self.time_step > 5000:
                # Let the traffic manager only execute 5000 steps. or it can fill the replay buffer
                logging.info('5000 steps passed under traffic manager control')
                return True

        return False

    def _speed_switch(self, a_index):
        """cont: the control command of RL agent"""
        ego_speed = get_speed(self.ego_vehicle)
        if self.speed_state == SpeedState.START:
            # control = self.controller.run_step({'waypoints':self.next_wps,'vehicle_front':self.vehicle_front})
            if ego_speed >= self.speed_threshold:
                self.speed_state = SpeedState.RUNNING
                self._ego_autopilot(False)
                if not self.RL_switch:
                    # Under basic lanechange agent control
                    # self.autopilot_controller.set_destination(random.choice(self.spawn_points).location)
                    # if self.autopilot_controller.done() and self.loop:
                    #     self.autopilot_controller.set_destination(self.my_set_destination())
                    # control = self.autopilot_controller.run_step()
                    print(
                        "basic_lanechanging_agent before: last_lane, current_lane, last_target_lane, current_target_lane, last action, current action: ",
                        self.last_lane, self.current_lane, self.last_target_lane, self.current_target_lane,
                        self.last_action.value, self.current_action.value)
                    self.control, self.current_target_lane, self.current_action = \
                        self.autopilot_controller.run_step(self.current_lane, self.last_target_lane, self.last_action,
                                                           self.modify_change_steer)
                    print(
                        "basic_lanechanging_agent after: last_lane, current_lane, last_target_lane, current_target_lane, last action, current action: ",
                        self.last_lane, self.current_lane, self.last_target_lane, self.current_target_lane,
                        self.last_action.value, self.current_action.value)
                else:
                    if a_index == 0:
                        self.current_action = Action.LANE_CHANGE_LEFT
                        self.current_target_lane = self.current_lane + 1
                    elif a_index == 2:
                        self.current_action = Action.LANE_CHANGE_RIGHT
                        self.current_target_lane = self.current_lane - 1
                    elif a_index == 1:
                        self.current_action = Action.LANE_FOLLOW
                        self.current_target_lane = self.current_lane
                    else:
                        # a_index=4
                        self.current_action = Action.STOP
                        self.current_target_lane = self.current_lane
                    print(
                        "initial: last_lane, current_lane, last_target_lane, current_target_lane, last action, current action: ",
                        self.last_lane, self.current_lane, self.last_target_lane, self.current_target_lane,
                        self.last_action.value, self.current_action.value)
        elif self.speed_state == SpeedState.RUNNING:
            if self.RL_switch:
                # under rl control, used to set the self.new_action.
                print(
                    "RL_control before: last_lane, current_lane, last_target_lane, current_target_lane, last action, current action: ",
                    self.last_lane, self.current_lane, self.last_target_lane, self.current_target_lane,
                    self.last_action.value, self.current_action.value)
                if a_index == 0:
                    self.current_action = Action.LANE_CHANGE_LEFT
                    self.current_target_lane = self.current_lane + 1
                elif a_index == 2:
                    self.current_action = Action.LANE_CHANGE_RIGHT
                    self.current_target_lane = self.current_lane - 1
                elif a_index == 1:
                    self.current_action = Action.LANE_FOLLOW
                    self.current_target_lane = self.current_lane
                else:
                    # a_index=4
                    self.current_action = Action.STOP
                    self.current_target_lane = self.current_lane
                # _, _, _, self.distance_to_front_vehicles, self.distance_to_rear_vehicles = \
                #     self.autopilot_controller.run_step(self.last_lane, self.last_target_lane, self.last_action, True, a_index, self.modify_change_steer)
                print(
                    "RL_control after: last_lane, current_lane, last_target_lane, current_target_lane, last action, current action: ",
                    self.last_lane, self.current_lane, self.last_target_lane, self.current_target_lane,
                    self.last_action.value, self.current_action.value)
                if ego_speed < self.speed_min:
                    # Only add reboot state in the beginning 200 episodes
                    # self._ego_autopilot(True)
                    # self.speed_state = SpeedState.REBOOT
                    pass
            else:
                # Under basic lane change agent control
                # if self.autopilot_controller.done() and self.loop:
                #     # self.autopilot_controller.set_destination(random.choice(self.spawn_points).location)
                #     self.autopilot_controller.set_destination(self.my_set_destination())
                # control=self.autopilot_controller.run_step()
                print(
                    "basic_lanechanging_agent before: last_lane, current_lane, last_target_lane, current_target_lane, last action, current action: ",
                    self.last_lane, self.current_lane, self.last_target_lane, self.current_target_lane,
                    self.last_action.value, self.current_action.value)
                self.control, self.current_target_lane, self.current_action = \
                    self.autopilot_controller.run_step(self.current_lane, self.last_target_lane, self.last_action,
                                                       self.modify_change_steer)
                print(
                    "basic_lanechanging_agent after: last_lane, current_lane, last_target_lane, current_target_lane, last action, current action: ",
                    self.last_lane, self.current_lane, self.last_target_lane, self.current_target_lane,
                    self.last_action.value, self.current_action.value)
        else:
            logging.error('CODE LOGIC ERROR')

        return

    def _get_info(self, control_info=None):
        """Rerurn simulation running information,
            param: control_info, the current controller information
        """
        if control_info is None:
            return self.step_info
        else:
            self.step_info.update(control_info)
            return self.step_info

    def _ego_autopilot(self, setting=True):
        # Use traffic manager to control ego vehicle
        self.ego_vehicle.set_autopilot(setting, self.tm_port)
        if setting:
            speed_diff = (30 - self.speed_limit) / 30 * 100
            self.traffic_manager.distance_to_leading_vehicle(self.ego_vehicle, self.min_distance)
            if self.ignore_traffic_light:
                self.traffic_manager.ignore_lights_percentage(self.ego_vehicle, 100)
                self.traffic_manager.ignore_walkers_percentage(self.ego_vehicle, 100)
            self.traffic_manager.ignore_signs_percentage(self.ego_vehicle, 100)
            self.traffic_manager.ignore_vehicles_percentage(self.ego_vehicle, 0)
            self.traffic_manager.vehicle_percentage_speed_difference(self.ego_vehicle, speed_diff)
            # if self.auto_lanechange and self.speed_state == SpeedState.RUNNING:
            self.traffic_manager.auto_lane_change(self.ego_vehicle, True)
            self.traffic_manager.random_left_lanechange_percentage(self.ego_vehicle, 0)
            self.traffic_manager.random_right_lanechange_percentage(self.ego_vehicle, 0)

            # self.traffic_manager.set_desired_speed(self.ego_vehicle, 72)
            # ego_wp=self.map.get_waypoint(self.ego_vehicle.get_location())
            # self.traffic_manager.set_path(self.ego_vehicle,path)
            """set_route(self, actor, path):
                Sets a list of route instructions for a vehicle to follow while controlled by the Traffic Manager. 
                The possible route instructions are 'Left', 'Right', 'Straight'.
                The traffic manager only need this instruction when faces with a junction."""
            self.traffic_manager.set_route(self.ego_vehicle,
                                           ['Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight',
                                            'Straight', 'Straight', 'Straight', 'Straight'])

    def _sensor_callback(self, sensor_data, sensor_queue):
        array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('uint8'))
        # image is rgba format
        array = np.reshape(array, (sensor_data.height, sensor_data.width, 4))
        array = array[:, :, :3]
        sensor_queue.put((sensor_data.frame, array))

    def _tick(self):
        self.clock.tick()
        # self.sim_world.tick()
        if self.sync:
            self.world.world.tick()
        else:
            self.world.world.wait_for_tick()
        self.world.tick(self.clock)
        self.world.render(self.display)
        pygame.display.flip()

    def _init_renderer(self):
        """Initialize the birdeye view renderer."""
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.hud = HUD(self.width, self.height)
        self.world = World(self.sim_world, self.hud)
        self.clock = pygame.time.Clock()

    def _set_synchronous_mode(self):
        """Set whether to use the synchronous mode."""
        # Set fixed simulation step for synchronous mode
        if self.sync:
            settings = self.sim_world.get_settings()
            settings.no_rendering_mode = self.no_rendering
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1.0 / self.fps
                self.sim_world.apply_settings(settings)

    def _set_traffic_manager(self):
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        # every vehicle keeps a distance of 3.0 meter
        self.traffic_manager.set_global_distance_to_leading_vehicle(10)
        # Set physical mode only for cars around ego vehicle to save computation
        if self.hybrid:
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(200.0)

        """The default global speed limit is 30 m/s
        Vehicles' target speed is 70% of their current speed limit unless any other value is set."""
        speed_diff = (30 * 3.6 - (self.speed_limit + 1)) / (30 * 3.6) * 100
        # Let the companion vehicles drive a bit faster than ego speed limit
        self.traffic_manager.global_percentage_speed_difference(-100)
        self.traffic_manager.set_synchronous_mode(self.sync)
        # set traffic light elpse time
        lights_list = self.sim_world.get_actors().filter("*traffic_light*")
        for light in lights_list:
            light.set_green_time(15)
            light.set_red_time(0)
            light.set_yellow_time(0)

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn a  vehicle at specific transform
        Args:
            transform: the carla transform object.

        Returns:
            Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            ego_bp = create_vehicle_blueprint(self.sim_world, self.ego_filter, ego=True,
                                              color=random.choice(['255,0,0', '0,255,0', '0,0,255']))
            vehicle = self.sim_world.try_spawn_actor(ego_bp, transform)
            if vehicle is None:
                logging.warn("Ego vehicle generation fail")

        # if self.debug and vehicle:
        #      vehicle.show_debug_telemetry()

        return vehicle

    def _spawn_companion_vehicles(self):
        """
        Spawn surrounding vehcles of this simulation
        each vehicle is set to autopilot mode and controled by Traffic Maneger
        note: the ego vehicle trafficmanager and companion vehicle trafficmanager shouldn't be the same one
        """
        # spawn_points_ = self.map.get_spawn_points()
        spawn_points_ = self.spawn_points
        # make sure companion vehicles also spawn on chosen route
        # spawn_points_=[x.transform for x in self.ego_spawn_waypoints]

        num_of_spawn_points = len(spawn_points_)
        num_of_vehicles = random.choice(self.num_of_vehicles)

        if num_of_vehicles < num_of_spawn_points:
            random.shuffle(spawn_points_)
            spawn_points = random.sample(spawn_points_, num_of_vehicles)
        else:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, num_of_vehicles, num_of_spawn_points)
            num_of_vehicles = num_of_spawn_points - 1

        # Use command to apply actions on batch of data
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor  # FutureActor is eaqual to 0
        command_batch = []

        for i, transform in enumerate(spawn_points_):
            if i >= num_of_vehicles:
                break

            blueprint = create_vehicle_blueprint(self.sim_world, 'vehicle.audi.etron', ego=False, color='0,0,0',
                                                 number_of_wheels=[4])
            # Spawn the cars and their autopilot all together
            command_batch.append(SpawnActor(blueprint, transform).
                                 then(SetAutopilot(FutureActor, True, self.tm_port)))

        # execute the command batch
        for (i, response) in enumerate(self.client.apply_batch_sync(command_batch, self.sync)):
            if response.has_error():
                logging.warn(response.error)
            else:
                # print("Future Actor",response.actor_id)
                vehicle = self.sim_world.get_actor(response.actor_id)
                self.companion_vehicles.append(vehicle)

                if self.ignore_traffic_light:
                    self.traffic_manager.ignore_lights_percentage(vehicle, 100)
                    self.traffic_manager.ignore_walkers_percentage(vehicle, 100)
                else:
                    self.traffic_manager.ignore_lights_percentage(vehicle, 50)
                    self.traffic_manager.ignore_walkers_percentage(vehicle, 50)
                self.traffic_manager.ignore_signs_percentage(vehicle, 100)
                self.traffic_manager.auto_lane_change(vehicle, False)
                # modify change probability
                self.traffic_manager.random_left_lanechange_percentage(vehicle, 0)
                self.traffic_manager.random_right_lanechange_percentage(vehicle, 0)
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle,
                                                                         random.choice(
                                                                             [-100, -100, -100, -140, -160, -180]))
                self.traffic_manager.set_route(vehicle,
                                               ['Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight',
                                                'Straight', 'Straight', 'Straight', 'Straight'])
                self.traffic_manager.update_vehicle_lights(vehicle, True)
                # print(vehicle.attributes)

        msg = 'requested %d vehicles, generate %d vehicles, press Ctrl+C to exit.'
        logging.info(msg, num_of_vehicles, len(self.companion_vehicles))

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
            transform: the carla transform object.

        Returns:
            Bool indicating whether the spawn is successful.
        """
        pass

    def _clear_actors(self, actor_filters, filter=True):
        """Clear specific actors
        filter: True means filter actors by blueprint, Fals means fiter actors by carla.CityObjectLabel"""
        if filter:
            if self.camera is not None:
                self.camera.stop()
            if self.collision_sensor is not None:
                self.collision_sensor.sensor.stop()
            if self.lane_invasion_sensor is not None:
                self.lane_invasion_sensor.sensor.stop()
            for actor_filter in actor_filters:
                self.client.apply_batch([carla.command.DestroyActor(x)
                                         for x in self.sim_world.get_actors().filter(actor_filter)])

        # for actor_filter in actor_filters:
        #     for actor in self.sim_world.get_actors().filter(actor_filter):
        #         if actor.is_alive:
        #             if actor.type_id =='controller.ai.walker':
        #                 actor.stop()
        #             actor.destroy()
