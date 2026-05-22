import sys
import math
import time
import carla
import random
import pygame
import logging
from colorama import Fore, Back, Style, Cursor
from utils.utils import exit_game

class CarlaEnvironment:
    def __init__(self, name=None, world_map='Town10HD', timeout=10.0):
        self.client = None
        self.name = name
        self.world_map = world_map
        self.timeout = timeout
        self.server_fps = 0.0
        self.simulation_time = 0
        self.server_clock = pygame.time.Clock()
        
        # World
        self.world = None
        self.town_map = None
        self.actor_with_transform = []

    def setup_carla_environment(self):
        """Initialize CARLA client and world settings"""
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(self.timeout)
            # Load world
            # world = self.client.get_world()
            world = self.client.load_world(self.world_map, reset_settings=True, map_layers=carla.MapLayer.All)
        except RuntimeError as e:
            logging.error(e)
            exit_game()
        
        # Configure world settings
        setting = world.get_settings()
        setting.no_rendering_mode = True
        setting.synchronous_mode = True
        setting.fixed_delta_seconds = 0.033
        setting.substepping = True
        setting.max_substep_delta_time = 0.01
        setting.max_substeps = 10
        setting.spectator_as_ego = True
        setting.tile_stream_distance = 3000
        setting.actor_active_distance = 2000
        world.apply_settings(setting)
        
        # Configure traffic manager
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.synchronous_mode = True
        traffic_manager.global_percentage_speed_difference(-10)
        traffic_manager.set_respawn_dormant_vehicles(True)
        traffic_manager.set_boundaries_respawn_dormant_vehicles(50, 150)
        traffic_manager.set_random_device_seed(0)

        return world, traffic_manager
    
    def setup_spectator(self, world):
        """Configure spectator view"""
        spectator = world.get_spectator()
        transform = spectator.get_transform()
        location = transform.location + carla.Location(x=-30, z=20)
        rotation = carla.Rotation(pitch=-20, yaw=-20, roll=0)
        new_transform = carla.Transform(location, rotation)
        spectator.set_transform(new_transform)
    
    def spawn_actors(self, world, num_vehicle=70, num_walkers=10, ego_vehicle_type='vehicle.tesla.cybertruck', ego_vehicle_name='Lebron James'):
        """Spawn vehicles, pedestrians, and ego vehicle"""
        actors = Actors()
        # Spawn ego vehicle
        ego_vehicle = actors.spawn_ego_vehicle(world, world.get_map().get_spawn_points(), ego_vehicle_type, ego_vehicle_name, True)

        # Spawn other actors
        actors.spawn_vehicles(world, num_vehicle)
        actors.spawn_pedestrians(world, num_walkers, 0.25, 0.15)
        
        return ego_vehicle

    def vehicle_light_state(self, world):
        weather = world.get_weather()
        for vehicle in world.get_actors().filter('*vehicle*'):
            control = vehicle.get_control()
            try:
                current_light_state = vehicle.get_light_state()
                if weather.sun_altitude_angle < 0:
                    current_light_state |= carla.VehicleLightState.Position
                    current_light_state |= carla.VehicleLightState.HighBeam
                    current_light_state |= carla.VehicleLightState.Interior
                if weather.sun_altitude_angle > 0:
                    current_light_state &= 0b11011111011
                if weather.fog_density > 30 and weather.precipitation > 30:
                    current_light_state |= carla.VehicleLightState.Position
                    current_light_state |= carla.VehicleLightState.HighBeam
                    current_light_state |= carla.VehicleLightState.Interior
                if weather.fog_density < 30 and weather.precipitation < 30:
                    current_light_state &= 0b11101111101
                if control.brake > 0.1:
                    current_light_state |= carla.VehicleLightState.Brake
                if control.brake <= 0.1:
                    current_light_state &= 0b11111110111
                if control.steer < -0.1:
                    current_light_state |= carla.VehicleLightState.LeftBlinker
                if control.steer > 0.1:
                    current_light_state |= carla.VehicleLightState.RightBlinker
                if abs(control.steer) < 0.1:
                    current_light_state &= 0b11111001111
                if control.reverse:
                    current_light_state |= carla.VehicleLightState.Reverse
                if not control.reverse:
                    current_light_state &= 0b11110111111

                vehicle.set_light_state(carla.VehicleLightState(current_light_state))
            except RuntimeError:
                # print(f"Vehicle {vehicle.id} no longer exists.")
                pass    


class Actors(object):
    def __init__(self):
        pass

    def spawn_pedestrians(self, world, num_walkers, percentage_pedestrians_running, percentage_pedestrians_crossing):
        ped_blueprints = world.get_blueprint_library().filter('*pedestrian*')
        walker_ai_blueprint = world.get_blueprint_library().find('controller.ai.walker')
        ped_spawn_points = []

        for _ in range(num_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                ped_spawn_points.append(spawn_point)

        if not ped_spawn_points:
            logging.warning("No valid pedestrian spawn points were found.")
            return

        walker_speed_pairs = []
        for spawn_point in ped_spawn_points:
            walker_bp = random.choice(ped_blueprints)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            walker_speed = 0.0
            if walker_bp.has_attribute('speed'):
                if random.random() > percentage_pedestrians_running:
                    walker_speed = float(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    walker_speed = float(walker_bp.get_attribute('speed').recommended_values[2])
            walker = world.try_spawn_actor(walker_bp, spawn_point)
            if walker is not None:
                walker_speed_pairs.append((walker, walker_speed))

        if not walker_speed_pairs:
            logging.warning("Failed to spawn any pedestrians from the sampled navigation points.")
            return

        walker_ai_batch = []
        for walker, walker_speed in walker_speed_pairs:
            try:
                walker_ai = world.spawn_actor(walker_ai_blueprint, carla.Transform(), walker)
            except RuntimeError:
                logging.warning("Failed to spawn AI controller for pedestrian %s", walker.id)
                walker.destroy()
                continue
            walker_ai_batch.append((walker_ai, walker_speed))

        for walker_ai, walker_speed in walker_ai_batch:
            walker_ai.start()
            destination = world.get_random_location_from_navigation()
            if destination is not None:
                walker_ai.go_to_location(destination)
            walker_ai.set_max_speed(walker_speed)

        world.set_pedestrians_cross_factor(percentage_pedestrians_crossing)

    def spawn_vehicles(self, world, num_vehicle):
        """Spawn vehicles with autopilot enabled"""
        vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
        vehicle_spawn_points = world.get_map().get_spawn_points()
        random.shuffle(vehicle_spawn_points)
        spawned_vehicles = []

        # Try to spawn vehicles
        for spawn_point in vehicle_spawn_points[:num_vehicle]:
            blueprint = random.choice(vehicle_blueprints)

            # Try to spawn the vehicle
            vehicle = world.try_spawn_actor(blueprint, spawn_point)
            if vehicle is not None:
                spawned_vehicles.append(vehicle)

        # Enable autopilot for all spawned vehicles
        for vehicle in spawned_vehicles:
            vehicle.set_autopilot(True)
            # Set a random speed between 30 and 50 km/h
            vehicle.set_target_velocity(carla.Vector3D(
                x=random.uniform(8.33, 13.89),  # 30-50 km/h in m/s
                y=0.0,
                z=0.0
            ))

    def spawn_ego_vehicle(self, world, vehicle_spawn_points, vehicle_type, role_name='ego_vehicle', autopilot=False):
        """Spawn ego vehicle with autopilot disabled"""
        ego_spawn_point = random.choice(vehicle_spawn_points)
        ego_blueprint = world.get_blueprint_library().find(vehicle_type)
        ego_blueprint.set_attribute('role_name', role_name)
        
        ego_vehicle = world.spawn_actor(ego_blueprint, ego_spawn_point)
        ego_vehicle.set_autopilot(autopilot)
        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
        return ego_vehicle