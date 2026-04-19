# -*- coding: utf-8 -*-
"""
Author: SilverWings
GitHub: https://github.com/silverwingsbot
"""

from __future__ import division
import numpy as np
import random
import time
import gym
from gym import spaces
from gym.utils import seeding
import carla

class CarlaEnv(gym.Env):
    def __init__(self, params):
        self.collision_sensor = None
        self.lidar_sensor = None
        self._is_collision = False
        self._is_off_road = False
        self.off_road_counter = 0
        self.number_of_vehicles = params['number_of_vehicles']
        self.number_of_walkers = params['number_of_walkers']
        self.dt = params['dt']
        self.max_time_episode = params['max_time_episode']
        self.max_waypoints = params['max_waypoints']
        self.visualize_waypoints = params['visualize_waypoints']
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.view_mode = params['view_mode']
        self.traffic = params['traffic']
        self.lidar_max_range = params['lidar_max_range']
        self.max_nearby_vehicles = params['max_nearby_vehicles']
        self.surrounding_vehicle_spawned_randomly = params['surrounding_vehicle_spawned_randomly']
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(low=0.0, high=1.0, shape=(240,), dtype=np.float32),
            'ego_state': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            'nearby_vehicles': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_nearby_vehicles, 6), dtype=np.float32),
            'waypoints': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_waypoints, 4), dtype=np.float32),
            'lane_info': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32)
        )
    
        print('Connecting to Carla server...')
        client = carla.Client('localhost', params['port'])
        client.set_timeout(10.0)
        self.world = client.load_world(params['town'])
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        print('Connection established!')
    
        # Get all predefined vehicle spawn points from the map
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        # Prepare a list to hold spawn points for pedestrians (walkers)
        self.walker_spawn_points = []
        # Randomly generate spawn points for the specified number of pedestrians
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()  # Create an empty transform object
            # Try to get a random navigable location in the environment
            loc = self.world.get_random_location_from_navigation()
            # If a valid location is found, use it as a spawn point for a pedestrian
            if loc is not None:
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)
    
    
        self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='255,0,0')
    
        self.collision_hist = []
        self.collision_hist_l = 1
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
    
        self.lidar_data = None  # Placeholder to store incoming LiDAR data
        self.lidar_height = 0.8  # Height at which the LiDAR is mounted on the vehicle (in meters)
        # Set the position of the LiDAR sensor using a transform (translation only in Z direction)
        self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
        # Get the LiDAR blueprint from Carla's sensor library
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        # Set LiDAR attributes
        self.lidar_bp.set_attribute('channels', '1')  # Use 1 channel to perform a flat 360° horizontal scan
        self.lidar_bp.set_attribute('range', '50')  # Maximum LiDAR range in meters
        self.lidar_bp.set_attribute('rotation_frequency', '10')  # How many full 360° rotations per second
        self.lidar_bp.set_attribute('points_per_second', '10000')  # Total number of points generated per second
        self.lidar_bp.set_attribute('upper_fov', '0')  # upper and lower FOV are both 0 for a flat horizontal scan
        self.lidar_bp.set_attribute('lower_fov', '0')  
    
    
        self.settings = self.world.get_settings()  # Get the current world settings
        self.settings.fixed_delta_seconds = self.dt  # Set the physics simulation step size (in seconds)
                                                      # This ensures consistent time intervals for simulation updates
    
    
        self.reset_step = 0
        self.total_step = 0

    def reset(self):
        # Stop and destroy the collision sensor if it exists
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except:
                pass
            self.collision_sensor = None
    
        # Stop and destroy the LiDAR sensor if it exists
        if self.lidar_sensor is not None:
            try:
                self.lidar_sensor.stop()
                self.lidar_sensor.destroy()
            except:
                pass
            self.lidar_sensor = None
    
        # Reset collision and off-road status flags
        self._is_collision = False
        self._is_off_road = False
    
        self._set_synchronous_mode(False)  # Switch back to asynchronous mode
        self._clear_all_actors([
            'sensor.other.collision',
            'sensor.lidar.ray_cast',
            'sensor.camera.rgb',
            'vehicle.*',
            'controller.ai.walker',
            'walker.*'
        ])  # Remove all specified actors from the world

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles
        self.spawned_vehicles = []
        self.used_spawn_points = []
        
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                vehicle = self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4])
                if vehicle:
                    self.spawned_vehicles.append(vehicle)  # Record the spawned vehicle
                    self.used_spawn_points.append(spawn_point)  # Mark spawn point as used
                    count -= 1
                if count <= 0:
                    break
        # print(f"Surrounding vehicles number is {len(self.spawned_vehicles)}")

        # Spawn pedestrians
        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers
        
        if count > 0:
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                if count <= 0:
                    break
        
        # Try random spawn points until all pedestrians are spawned
        while count > 0:
            if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                count -= 1

        # Get actors' polygon list
        # Calculate and collect the bounding polygons (e.g., four corners) of surrounding vehicles and pedestrians
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)

        # Spawn the ego vehicle
        ego_spawn_times = 0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                self.reset()  # If failed too many times, reset the environment
        
            # Select a spawn point for the ego vehicle by excluding locations used by nearby vehicles
            available_spawn_points = [
                sp for sp in self.vehicle_spawn_points if sp not in self.used_spawn_points
            ]
            
            if len(available_spawn_points) > 0:
                transform = random.choice(available_spawn_points)  # Choose a spawn point not used by nearby vehicles
            else:
                transform = random.choice(self.vehicle_spawn_points)  # Fallback: use any spawn point
        
            # Try to spawn the ego vehicle at the selected location
            if self._try_spawn_ego_vehicle_at(transform):
                break  # Successfully spawned the ego vehicle
            else:
                ego_spawn_times += 1  # Retry counter
                time.sleep(0.1)  # Small delay before retrying

        if self.traffic == 'off':
            # Set all traffic lights to green and freeze them
            for actor in self.world.get_actors().filter('traffic.traffic_light*'):
                actor.set_state(carla.TrafficLightState.Green)
                actor.freeze(True)
        elif self.traffic == 'on':
            # Allow traffic lights to work normally
            for actor in self.world.get_actors().filter('traffic.traffic_light*'):
                actor.freeze(False)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp,
            carla.Transform(),  # Attach at the center of the ego vehicle (no offset)
            attach_to=self.ego
        )
        
        # Start listening for collision events
        self.collision_sensor.listen(
            lambda event: get_collision_hist(event)  # When a collision event happens, pass the event to get_collision_hist()
        )

        def get_collision_hist(event):
            impulse = event.normal_impulse  # Get the collision impulse (a 3D vector)
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)  # Calculate collision intensity (vector norm)
            self.collision_hist.append(intensity)  # Record the collision intensity
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)  # Keep only the latest collision records (FIFO)
        
        # Initialize collision history list
        # Clear collision history after each episode because in gym-carla setup,
        # a collision typically triggers episode termination and reset.
        self.collision_hist = []

        # Add lidar sensor
        self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
        self.lidar_sensor.listen(lambda data: get_lidar_data(data))
        def get_lidar_data(data):
            self.lidar_data = data

        # Update timesteps
        self.time_step = 1  # Indicates a new episode has started
        self.reset_step += 1  # Count how many resets have occurred
        
        # Enable autopilot for all surrounding vehicles
        for vehicle in self.spawned_vehicles:
            vehicle.set_autopilot()
        
        self._set_synchronous_mode(True)  # Switch to synchronous mode for simulation
        self.world.tick()  # Advance the simulation by one tick
        
        return self._get_obs()  # Return the initial observation after reset

    def step(self, action):
        throttle = float(np.clip(action[0], 0.0, 1.0))
        steer    = float(np.clip(action[1], -1.0, 1.0))
        brake    = float(np.clip(action[2], 0.0, 1.0))

        # Apply control
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.ego.apply_control(control)

        self.world.tick()

        # Set spectator (camera) view
        spectator = self.world.get_spectator()
        transform = self.ego.get_transform()
        if self.view_mode == 'top':
            # Top-down view (bird's eye)
            spectator.set_transform(
                carla.Transform(
                    transform.location + carla.Location(z=40),
                    carla.Rotation(pitch=-90)
                )
            )
        elif self.view_mode == 'follow':
            # Follow view (behind and above the ego vehicle)
            cam_location = transform.transform(carla.Location(x=-6.0, z=3.0))  # 6 meters behind, 3 meters above
            cam_rotation = carla.Rotation(pitch=-10, yaw=transform.rotation.yaw, roll=0)
            spectator.set_transform(carla.Transform(cam_location, cam_rotation))

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        obs = self._get_obs()
        done = self._terminal()
        reward = self._get_reward(obs, done)
        cost = self._get_cost(obs)

        # state information
        info = {
          'is_collision': self._is_collision,
          'is_off_road': self._is_off_road
        }
        return (obs, reward, cost, done, info)

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create a vehicle blueprint based on the given filter and wheel number.

        Args:
            actor_filter (str): Filter string to select vehicle types, e.g., 'vehicle.lincoln*' 
                                ('*' matches a series of models).
            color (str, optional): Desired vehicle color. Randomly chosen if None.
            number_of_wheels (list): A list of acceptable wheel numbers (default is [4]).

        Returns:
            bp (carla.ActorBlueprint): A randomly selected blueprint matching the criteria.
        """
        # Get all blueprints matching the actor filter
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []

        # Further filter blueprints based on the number of wheels
        # Keeping number_of_wheels as a list makes it flexible to match multiple types (e.g., cars, trucks)
        for nw in number_of_wheels:
            blueprint_library += [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]

        # Randomly select one blueprint from the filtered list
        bp = random.choice(blueprint_library)

        # Set the vehicle color
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        return bp

    def _set_synchronous_mode(self, synchronous=True):

        """Enable or disable synchronous mode for the simulation.
        Args:
            synchronous (bool):
                True to enable synchronous mode (server waits for client each frame),
                False to disable and run in asynchronous mode (default is True).
        """
        self.settings.synchronous_mode = synchronous  # Set the synchronous mode
        self.world.apply_settings(self.settings)  # Apply the updated settings to the world

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at a specific transform.
    
        Args:
            transform (carla.Transform): Location and orientation where the vehicle should be spawned.
            number_of_wheels (list): Acceptable number(s) of wheels for the vehicle blueprint.
            random_vehicle (bool): 
                False to use Tesla Model 3 with a blue color,
                True to randomly select a vehicle model and color (default).
    
        Returns:
            carla.Actor or None: Spawned vehicle actor if successful, otherwise None.
        """
        if self.surrounding_vehicle_spawned_randomly:
            # Randomly choose any vehicle blueprint
            blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
        else:
            # Fixed: Tesla Model 3 with blue color
            blueprint = self._create_vehicle_bluepprint('vehicle.tesla.model3', color='0,0,255', number_of_wheels=number_of_wheels)
        
        blueprint.set_attribute('role_name', 'autopilot')  # Set the vehicle to autopilot mode
    
        # Try to spawn the vehicle
        vehicle = self.world.try_spawn_actor(blueprint, transform)
    
        return vehicle if vehicle is not None else None

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at a specific transform with a random blueprint.
    
        Args:
            transform (carla.Transform): Location and orientation where the walker should be spawned.
    
        Returns:
            Bool: True if spawn is successful, False otherwise.
        """
        # Randomly select a walker blueprint
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
    
        # Make the walker vulnerable (can be affected by collisions)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
    
        # Try to spawn the walker actor
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)
    
        if walker_actor is not None:
            # Spawn a controller for the walker
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
    
            # Start the controller to control the walker
            walker_controller_actor.start()
    
            # Move the walker to a random location
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
    
            # Set a random walking speed between 1 m/s and 2 m/s (default is 1.4 m/s)
            walker_controller_actor.set_max_speed(1 + random.random())
    
            return True  # Spawn and initialization successful
    
        return False  # Failed to spawn

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at a specific transform.
    
        Args:
            transform (carla.Transform): Target location and orientation.
    
        Returns:
            Bool: True if spawn is successful, False otherwise.
        """
        vehicle = None
        overlap = False
    
        # Check if ego position overlaps with surrounding vehicles
        for idx, poly in self.vehicle_polygons[-1].items():  # Use .items() to iterate over dict keys and values
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
    
            if dis > 8:
                continue
            else:
                overlap = True
                break
    
        # If no overlap, try to spawn the ego vehicle
        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
    
        if vehicle is not None:
            self.ego = vehicle
            return True
    
        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.
    
        Args:
            filt: the filter indicating what type of actors we'll look at.
    
        Returns:
            actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt): 
            # Get all actors in the current world that meet the filt condition, such as vehicle.* or walker.*
            # Note that self.world.get_actors() retrieves all objects in the current simulation environment (vehicles, pedestrians, traffic lights, etc.).
    
            # Get x, y and yaw of the actor
            trans = actor.get_transform() 
            # Get the actor's global position (location) and heading angle (rotation).
    
            x = trans.location.x 
            # x, y are the actor's global coordinates.
    
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi 
            # yaw is the heading angle, whose unit is degrees, needs to be converted to radians (multiply by pi/180) to facilitate subsequent matrix calculations.
    
            # Get length and width
            bb = actor.bounding_box 
            # Get the "half-length" boundary.
    
            l = bb.extent.x 
            # "Half-length" in the x-direction (the distance from the center to the edge).
    
            w = bb.extent.y
            # "Half-width" in the y-direction (the distance from the center to the edge).
    
            # Get bounding box polygon in the actor's local coordinate
            # Take the vehicle center as the origin, build a local coordinate system, and list four corner points:
            # (l, w): front right corner, (l, -w): rear right corner, (-l, -w): rear left corner, (-l, w): front left corner
            poly_local = np.array([
                [l, w], [l, -w], [-l, -w], [-l, w]
            ]).transpose() 
            # Transpose() here is to facilitate subsequent matrix operations,
            # changing the matrix from (4,2) to (2,4) format.
    
            # Get rotation matrix to transform to global coordinate
            # This is a standard 2D rotation matrix: used to transform points from the local coordinate system to the global coordinate system.
            # Rotation matrix R = [cosθ  -sinθ]
            #                     [sinθ   cosθ]
            R = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
    
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0) 
            # np.matmul(R, poly_local):
            # Transform the four corners (in the local coordinate system) into the global direction through the rotation matrix.
            # After .transpose(), it becomes (4,2) format (one point per row).
            # + np.repeat([[x,y]],4,axis=0):
            # Add the global position offset of the vehicle/pedestrian to each point
            # to obtain the final polygon coordinates in the global coordinate system.
    
            actor_poly_dict[actor.id] = poly 
            # Store the calculated poly (a 4×2 array, four corner points in global coordinates)
            # with actor.id as the key into actor_poly_dict.
            # After returning, the entire dictionary structure:
            # {
            # actor_id_1: np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]),
            # actor_id_2: np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]),
            # ...
            # }
    
        return actor_poly_dict

    def _get_obs(self):
        obs = {}
        
# ========================== LIDAR feature extraction (240 dimensions) ==========================
        max_range = self.lidar_max_range  # Set a maximum perception distance
        lidar_features = np.full((240,), max_range, dtype=np.float32)  # Initialize all values to the maximum distance
    
        # Get ego pose
        ego_transform = self.ego.get_transform()
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
    
        # Traverse all point clouds
        for detection in self.lidar_data:
            x = detection.point.x
            y = detection.point.y
    
            # Rotate back to ego vehicle heading direction (make ego vehicle heading as 0 degrees)
            local_x = np.cos(-ego_yaw) * x - np.sin(-ego_yaw) * y
            local_y = np.sin(-ego_yaw) * x + np.cos(-ego_yaw) * y
    
            distance = np.sqrt(local_x**2 + local_y**2)
            angle = np.arctan2(local_y, local_x)  # Range [-π, π]
            angle_deg = (np.degrees(angle) + 360) % 360  # Map to [0, 360)
            index = int(angle_deg // 1.5)  # Each angular bin has a width of 1.5 degrees, 240 bins in total
        
            if index < 240:
                lidar_features[index] = min(lidar_features[index], distance)
    
        # Normalize to [0, 1]
        lidar_features /= max_range
    
        # Store into observation
        obs['lidar'] = lidar_features

# ========================== Ego vehicle state extraction =======================================
        velocity = self.ego.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        angular_velocity = self.ego.get_angular_velocity()
        acceleration = self.ego.get_acceleration()
        
        front_vehicle_distance = 0.0
        relative_speed = 0.0
        
        min_front_distance = 20.0  # Search range threshold
        vehicle_list = self.world.get_actors().filter('vehicle.*')
        
        for vehicle in vehicle_list:
            if vehicle.id == self.ego.id:
                continue
        
            transform = vehicle.get_transform()
            rel_x = transform.location.x - ego_x
            rel_y = transform.location.y - ego_y
        
            local_x = np.cos(-ego_yaw) * rel_x - np.sin(-ego_yaw) * rel_y
            local_y = np.sin(-ego_yaw) * rel_x + np.cos(-ego_yaw) * rel_y
        
            if 0 < local_x < min_front_distance and abs(local_y) < 2.5:
                d = np.sqrt(local_x**2 + local_y**2)
                if front_vehicle_distance == 0.0 or d < front_vehicle_distance:
                    front_vehicle_distance = d
                    front_speed = vehicle.get_velocity()
                    front_speed_mag = np.sqrt(front_speed.x**2 + front_speed.y**2 + front_speed.z**2)
                    relative_speed = speed - front_speed_mag
        
        ego_state = np.array([
            ego_x,
            ego_y,
            ego_yaw,
            speed,
            angular_velocity.z,
            acceleration.x,
            acceleration.y,
            front_vehicle_distance,
            relative_speed
        ], dtype=np.float32)
        
        obs['ego_state'] = ego_state

# ================ Nearby vehicles state extraction (up to 5 vehicles, within perception range) ===============
        max_vehicles = self.max_nearby_vehicles
        perception_range = self.lidar_max_range
        vehicle_list = self.world.get_actors().filter('vehicle.*')
        
        vehicle_data = []
        for vehicle in vehicle_list:
            if vehicle.id == self.ego.id:
                continue  # Skip the ego vehicle itself
        
            transform = vehicle.get_transform()
            x = transform.location.x
            y = transform.location.y
            yaw = np.deg2rad(transform.rotation.yaw)
        
            rel_x = x - ego_x
            rel_y = y - ego_y
        
            distance = np.sqrt(rel_x**2 + rel_y**2)
            if distance > perception_range:
                continue  # Ignore vehicles outside the perception range
        
            # Transform to ego-centric local coordinates
            local_x = np.cos(-ego_yaw) * rel_x - np.sin(-ego_yaw) * rel_y
            local_y = np.sin(-ego_yaw) * rel_x + np.cos(-ego_yaw) * rel_y
        
            v = vehicle.get_velocity()
            speed = np.sqrt(v.x**2 + v.y**2 + v.z**2)
        
            vehicle_data.append((distance, [local_x, local_y, yaw - ego_yaw, speed]))
        
        # Sort vehicles by distance and select the nearest max_vehicles
        vehicle_data.sort(key=lambda x: x[0])
        nearby_vehicles = [data for _, data in vehicle_data[:max_vehicles]]
        
        # Pad with zeros if fewer than max_vehicles are detected
        while len(nearby_vehicles) < max_vehicles:
            nearby_vehicles.append([0.0, 0.0, 0.0, 0.0])
        
        obs['nearby_vehicles'] = np.array(nearby_vehicles, dtype=np.float32).flatten()

# ========================== Current reference waypoints (up to N waypoints) ==========================
        max_waypoints = self.max_waypoints
        world_map = self.world.get_map()
        waypoint = world_map.get_waypoint(self.ego.get_location())
        waypoints_array = np.zeros((max_waypoints, 3), dtype=np.float32)
        
        for i in range(max_waypoints):
            if waypoint is None:
                break
        
            loc = waypoint.transform.location
            yaw = waypoint.transform.rotation.yaw
        
            # Transform waypoint location into ego-centric local coordinates
            local_x = np.cos(-ego_yaw) * (loc.x - ego_x) - np.sin(-ego_yaw) * (loc.y - ego_y)
            local_y = np.sin(-ego_yaw) * (loc.x - ego_x) + np.cos(-ego_yaw) * (loc.y - ego_y)
            yaw_relative = np.deg2rad(yaw) - ego_yaw  # Relative heading
        
            waypoints_array[i] = [local_x, local_y, yaw_relative]
        
            # Move to the next waypoint 2.0 meters ahead
            next_waypoints = waypoint.next(2.0)
            waypoint = next_waypoints[0] if next_waypoints else None
        
        obs['waypoints'] = waypoints_array.flatten()

# ============================= Lane boundary information =========================================
        waypoint_center = world_map.get_waypoint(
            self.ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
        
        if waypoint_center is None:
            # If no valid driving lane is found
            obs['lane_info'] = np.array([0.0, 0.0], dtype=np.float32)
        else:
            lane_width = waypoint_center.lane_width
            ego_location = self.ego.get_location()
            center_location = waypoint_center.transform.location
        
            # Calculate lateral offset between ego position and lane centerline
            lateral_offset = np.linalg.norm(
                np.array([
                    ego_location.x - center_location.x,
                    ego_location.y - center_location.y
                ])
            )
        
            obs['lane_info'] = np.array([lane_width, lateral_offset], dtype=np.float32)

# =============================== Visualize current reference waypoints ===============================
        if self.visualize_waypoints:
            for i in range(max_waypoints):
                wx, wy, _ = waypoints_array[i]
        
                # Transform from ego-centric local coordinates to global coordinates
                gx = np.cos(ego_yaw) * wx - np.sin(ego_yaw) * wy + ego_x
                gy = np.sin(ego_yaw) * wx + np.cos(ego_yaw) * wy + ego_y
        
                self.world.debug.draw_point(
                    carla.Location(x=gx, y=gy, z=ego_transform.location.z + 1.0),
                    size=0.1,
                    life_time=0.5,
                    color=carla.Color(0, 255, 0)  # Green points
                )

        return obs

    def _get_reward(self, obs, done):
        reward = 0.0
    
        # 1. Forward driving reward (within speed limit and along lane direction)
        speed = obs['ego_state'][3]
        if speed <= self.desired_speed:
            reward += 1.0 * speed
        else:
            reward += -1.0 * (speed - self.desired_speed)
    
        # 2. Lane deviation penalty (penalize offset from lane center)
        lane_width, lateral_offset = obs['lane_info']
        reward += -1.0 * lateral_offset
    
        # 3. Smooth driving penalty (lateral acceleration penalty)
        a_lat = obs['ego_state'][6]
        reward += -0.5 * abs(a_lat)
    
        # 4. Stationary penalty (if no vehicle ahead but ego is barely moving)
        front_distance = obs['ego_state'][7]
        if front_distance > 10.0 and speed < 0.1:
            reward += -1.0
    
        # 5. Collision penalty
        if self._is_collision:
            reward += -100.0
    
        # 6. Off-road penalty
        if self._is_off_road:
            reward += -100.0
    
        # # 7. Sparse terminal reward (for safely reaching the goal)
        # if done:
        #     if not self._is_collision and not self._is_off_road:
        #         reward += 200.0
    
        return reward

    def _get_cost(self, obs):
        """
        Calculate the constraint cost for safe reinforcement learning.
    
        This cost is only used in safe RL settings and does not affect the reward function.
        It penalizes collisions, off-road events, and overspeeding behavior.
        
        Args:
            obs: The current observation dictionary.
    
        Returns:
            cost (float): The accumulated constraint cost.
        """
        cost = 0.0
    
        # 1. Collision cost
        if self._is_collision:
            cost += 20.0
    
        # 2. Off-road cost
        if self._is_off_road:
            cost += 20.0
    
        # 3. Overspeed cost
        speed = obs['ego_state'][3]
        if speed > self.desired_speed:
            cost += (speed - self.desired_speed) / self.desired_speed  # Cost proportional to overspeed percentage
    
        return cost

    def _terminal(self):
        ego_transform = self.ego.get_transform()
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
    
        # 1. Collision termination
        if len(self.collision_hist) > 0:
            self._is_collision = True
            print('Collision occurred')
            return True
    
        # 2. Exceeding maximum allowed timesteps
        if self.time_step > self.max_time_episode:
            print('Exceeded maximum timesteps')
            return True
    
        # # 3. Goal reaching termination (optional)
        # if self.dests is not None:
        #     for dest in self.dests:
        #         if np.sqrt((ego_x - dest[0])**2 + (ego_y - dest[1])**2) < 4:
        #             return True
    
        # 4. Check if the current lane is a drivable lane
        waypoint = self.world.get_map().get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            self._is_off_road = True
            print('Non-drivable lane detected')
            return True
    
        # 5. Check if the vehicle's heading deviates too much from lane direction (> ±90°)
        ego_yaw = self.ego.get_transform().rotation.yaw
        lane_yaw = waypoint.transform.rotation.yaw
        yaw_diff = np.deg2rad(ego_yaw - lane_yaw)
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))  # Normalize to [-π, π]
        if not waypoint.is_intersection:
            if abs(yaw_diff) > np.pi / 2:  # More than 90 degrees deviation (wrong-way driving)
                self._is_off_road = True
                print('Wrong-way driving detected')
                return True
    
        # 6. Deviation too far from lane center
        lane_width, lateral_offset = self._get_obs()['lane_info']
        if not waypoint.is_intersection:
            if lateral_offset > lane_width / 2 + 1.0:
                self._is_off_road = True
                print('Deviated from lane')
                return True
    
        return False

    def _clear_all_actors(self, actor_filters):
        """Clear (destroy) all actors matching the given filter patterns.
    
        Args:
            actor_filters (list): A list of filter strings, e.g., ['vehicle.*', 'walker.*', 'sensor.*'].
        """
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                try:
                    # If the actor is a sensor, stop it before destroying
                    if 'sensor' in actor.type_id:
                        actor.stop()
                    actor.destroy()
                except:
                    pass  # Ignore any errors during destruction