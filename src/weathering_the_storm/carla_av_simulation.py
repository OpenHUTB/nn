import carla
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import json
import pygame
import time
import queue
import random
import cv2
import math
import logging
import sys
import traceback
from matplotlib import pyplot as plt
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class SimulationError(Exception):
    """Custom exception for simulation-specific errors"""
    pass

class SensorConfiguration:
    """
    Class to hold sensor configuration specifications.
    """
    def __init__(self, name, sensors_specs):
        """
        Initialize a sensor configuration.

        Args:
            name (str): Name of the configuration
            sensors_specs (list): List of tuples (blueprint_name, attributes, transform)
                where attributes is a dict of sensor attributes
        """
        self.name = name
        self.sensors_specs = sensors_specs

class AVSimulation:
    def __init__(self):
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(20.0)  # Increased timeout

            # Verify connection
            try:
                self.world = self.client.get_world()
                logging.info("Successfully connected to CARLA server")
            except RuntimeError as e:
                raise SimulationError(f"Failed to connect to CARLA server: {str(e)}")

            self.map = self.world.get_map()

            # Initialize Pygame
            if not pygame.get_init():
                pygame.init()
            try:
                self.display = pygame.display.set_mode((1920, 1080), pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
            except pygame.error as e:
                raise SimulationError(f"Failed to initialize Pygame display: {str(e)}")

            # Initialize queues with maximum size
            self.image_queue = queue.Queue(maxsize=100)
            self.lidar_queue = queue.Queue(maxsize=100)
            self.radar_queue = queue.Queue(maxsize=100)

            # Sensor data storage with capacity checks
            self.sensor_data = {
                'camera': [],
                'lidar': [],
                'radar': [],
                'semantic': [],
                'depth': [],
                'weather': []
            }

            self.active_sensors = []
            self.active_actors = []

            # Define sensor configurations (same as before)
            self.define_sensor_configurations()

            logging.info("AVSimulation initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize AVSimulation: {str(e)}")
            raise
    def setup_sensors(self, vehicle, config_name):
        """
        Set up sensors based on the selected configuration.

        Args:
            vehicle: CARLA vehicle actor to attach sensors to
            config_name: Name of sensor configuration to use ('minimal', 'standard', or 'advanced')
        """
        sensors = []
        try:
            if config_name not in self.sensor_configurations:
                raise SimulationError(f"Invalid sensor configuration: {config_name}")

            config = self.sensor_configurations[config_name]
            logging.info(f"Setting up {config_name} sensor configuration")

            for blueprint_name, attributes, transform in config.sensors_specs:
                try:
                    # Get the blueprint
                    blueprint = self.world.get_blueprint_library().find(blueprint_name)
                    if blueprint is None:
                        raise SimulationError(f"Sensor blueprint not found: {blueprint_name}")

                    # Set attributes
                    for attr_name, attr_value in attributes.items():
                        blueprint.set_attribute(attr_name, attr_value)

                    # Spawn the sensor
                    sensor = self.world.spawn_actor(
                        blueprint,
                        transform,
                        attach_to=vehicle
                    )

                    if sensor is None:
                        raise SimulationError(f"Failed to spawn sensor: {blueprint_name}")

                    # Set up the appropriate callback
                    if 'camera.rgb' in blueprint_name:
                        sensor.listen(lambda data: self.sensor_callback(data, 'camera'))
                        logging.info(f"RGB camera set up at location: {transform.location}")

                    elif 'camera.semantic_segmentation' in blueprint_name:
                        sensor.listen(lambda data: self.sensor_callback(data, 'semantic'))
                        logging.info(f"Semantic segmentation camera set up at location: {transform.location}")

                    elif 'camera.depth' in blueprint_name:
                        sensor.listen(lambda data: self.sensor_callback(data, 'depth'))
                        logging.info(f"Depth camera set up at location: {transform.location}")

                    elif 'lidar' in blueprint_name:
                        sensor.listen(lambda data: self.sensor_callback(data, 'lidar'))
                        logging.info(f"LiDAR set up at location: {transform.location}")

                    elif 'radar' in blueprint_name:
                        sensor.listen(lambda data: self.sensor_callback(data, 'radar'))
                        logging.info(f"Radar set up at location: {transform.location}")

                    sensors.append(sensor)
                    self.active_sensors.append(sensor)

                except Exception as e:
                    logging.error(f"Failed to set up sensor {blueprint_name}: {str(e)}")
                    # Continue with other sensors even if one fails
                    continue

            if not sensors:
                raise SimulationError("No sensors were successfully set up")

            logging.info(f"Successfully set up {len(sensors)} sensors")
            return sensors

        except Exception as e:
            logging.error(f"Error in setup_sensors: {str(e)}")
            # Clean up any sensors that were created before the error
            for sensor in sensors:
                try:
                    if sensor is not None and sensor.is_alive:
                        sensor.destroy()
                        logging.info(f"Cleaned up sensor after setup error: {sensor.type_id}")
                except:
                    pass
            raise SimulationError(f"Sensor setup failed: {str(e)}")

    def define_sensor_configurations(self):
        """
        Define the sensor configurations available in the simulation.
        """
        self.sensor_configurations = {
            'minimal': SensorConfiguration('minimal', [
                ('sensor.camera.rgb', {
                    'image_size_x': '1920',
                    'image_size_y': '1080',
                    'fov': '90'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.lidar.ray_cast', {
                    'channels': '32',
                    'points_per_second': '100000',
                    'rotation_frequency': '20',
                    'range': '50'
                }, carla.Transform(carla.Location(x=0, z=1.8)))
            ]),

            'standard': SensorConfiguration('standard', [
                ('sensor.camera.rgb', {
                    'image_size_x': '1920',
                    'image_size_y': '1080',
                    'fov': '90'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.lidar.ray_cast', {
                    'channels': '64',
                    'points_per_second': '200000',
                    'rotation_frequency': '20',
                    'range': '70'
                }, carla.Transform(carla.Location(x=0, z=1.8))),
                ('sensor.other.radar', {
                    'horizontal_fov': '30',
                    'vertical_fov': '30',
                    'points_per_second': '1500',
                    'range': '100'
                }, carla.Transform(carla.Location(x=2.0, z=1.0)))
            ]),

            'advanced': SensorConfiguration('advanced', [
                ('sensor.camera.rgb', {
                    'image_size_x': '1920',
                    'image_size_y': '1080',
                    'fov': '90'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.camera.semantic_segmentation', {
                    'image_size_x': '1920',
                    'image_size_y': '1080'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.camera.depth', {
                    'image_size_x': '1920',
                    'image_size_y': '1080'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.lidar.ray_cast', {
                    'channels': '128',
                    'points_per_second': '500000',
                    'rotation_frequency': '20',
                    'range': '100'
                }, carla.Transform(carla.Location(x=0, z=1.8))),
                ('sensor.other.radar', {
                    'horizontal_fov': '45',
                    'vertical_fov': '45',
                    'points_per_second': '2000',
                    'range': '100'
                }, carla.Transform(carla.Location(x=2.0, z=1.0)))
            ])
        }
    def setup_weather(self):
        try:
            # Set up rainy weather conditions
            weather = carla.WeatherParameters(
                cloudiness=100.0,
                precipitation=100.0,
                precipitation_deposits=100.0,
                wind_intensity=50.0,
                fog_density=20.0,
                wetness=100.0,
                sun_altitude_angle=45.0
            )

            # Apply weather settings
            self.world.set_weather(weather)

            # Store weather state
            self.sensor_data['weather'].append({
                'timestamp': datetime.now().isoformat(),
                'params': {
                    'cloudiness': weather.cloudiness,
                    'precipitation': weather.precipitation,
                    'precipitation_deposits': weather.precipitation_deposits,
                    'wind_intensity': weather.wind_intensity,
                    'fog_density': weather.fog_density,
                    'wetness': weather.wetness,
                    'sun_altitude_angle': weather.sun_altitude_angle
                }
            })

            logging.info("Weather configured: Heavy rain conditions")
            return weather

        except Exception as e:
            logging.error(f"Failed to setup weather: {str(e)}")
            raise SimulationError(f"Weather setup failed: {str(e)}")

    def setup_traffic(self, num_vehicles=50, num_pedestrians=30):
        vehicles = []
        pedestrians = []
        controllers = []
        
        try:
            # Set up traffic manager
            traffic_manager = self.client.get_trafficmanager(8000)  # Port 8000
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_random_device_seed(0)
            traffic_manager.global_percentage_speed_difference(0)
            
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                raise SimulationError("No spawn points found in map")
            
            # Shuffle spawn points
            random.shuffle(spawn_points)
            
            # Spawn vehicles with collision checking
            vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')
            for _ in range(num_vehicles):
                try:
                    blueprint = random.choice(vehicle_bps)
                    if blueprint.has_attribute('color'):
                        color = random.choice(blueprint.get_attribute('color').recommended_values)
                        blueprint.set_attribute('color', color)
                    
                    vehicle = self.world.try_spawn_actor(blueprint, random.choice(spawn_points))
                    if vehicle is not None:
                        vehicle.set_autopilot(True, traffic_manager.get_port())
                        # Set behavior parameters
                        traffic_manager.update_vehicle_lights(vehicle, True)
                        traffic_manager.distance_to_leading_vehicle(vehicle, 0.5)
                        traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(-20, 10))
                        
                        vehicles.append(vehicle)
                        self.active_actors.append(vehicle)
                except Exception as e:
                    logging.warning(f"Failed to spawn vehicle: {str(e)}")
                    continue
                
            logging.info(f"Successfully spawned {len(vehicles)} vehicles")
            
            # Spawn pedestrians with collision checking
            pedestrian_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')
            controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            
            for _ in range(num_pedestrians):
                try:
                    # Try multiple times to find a valid spawn point
                    for _ in range(5):  # Try 5 times per pedestrian
                        spawn_point = carla.Transform(
                            self.world.get_random_location_from_navigation(),
                            carla.Rotation()
                        )
                        
                        # Check if location is valid
                        if self.world.get_map().get_waypoint(spawn_point.location, project_to_road=False):
                            blueprint = random.choice(pedestrian_bps)
                            pedestrian = self.world.try_spawn_actor(blueprint, spawn_point)
                            
                            if pedestrian is not None:
                                # Spawn controller
                                controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=pedestrian)
                                controller.start()
                                controller.set_max_speed(1.4)
                                
                                pedestrians.append(pedestrian)
                                controllers.append(controller)
                                self.active_actors.extend([pedestrian, controller])
                                break
                
                except Exception as e:
                    logging.warning(f"Failed to spawn pedestrian: {str(e)}")
                    continue
            
            logging.info(f"Successfully spawned {len(pedestrians)} pedestrians")
            
            # Wait a moment for physics to settle
            time.sleep(0.5)
            
            # Start pedestrian movement
            for controller in controllers:
                try:
                    controller.go_to_location(self.world.get_random_location_from_navigation())
                    # Add some randomization to pedestrian behavior
                    controller.set_max_speed(random.uniform(0.8, 1.8))
                except Exception as e:
                    logging.warning(f"Failed to set pedestrian destination: {str(e)}")
            
            return vehicles, pedestrians
            
        except Exception as e:
            logging.error(f"Error in setup_traffic: {str(e)}")
            self.cleanup_actors()
            raise

    def sensor_callback(self, data, sensor_type):
        try:
            if sensor_type == 'camera':
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]

                try:
                    if not self.image_queue.full():
                        self.image_queue.put((data.frame, array), block=False)
                except queue.Full:
                    logging.warning("Image queue is full, dropping frame")

                self.sensor_data['camera'].append({
                    'timestamp': data.timestamp,
                    'frame': data.frame,
                    'data': array,
                    'transform': data.transform
                })

            elif sensor_type == 'lidar':
                points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0] / 4), 4))

                try:
                    if not self.lidar_queue.full():
                        self.lidar_queue.put((data.frame, points), block=False)
                except queue.Full:
                    logging.warning("LiDAR queue is full, dropping frame")

                self.sensor_data['lidar'].append({
                    'timestamp': data.timestamp,
                    'frame': data.frame,
                    'points': points,
                    'transform': data.transform
                })

            elif sensor_type == 'radar':
                points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0] / 4), 4))

                try:
                    if not self.radar_queue.full():
                        self.radar_queue.put((data.frame, points), block=False)
                except queue.Full:
                    logging.warning("Radar queue is full, dropping frame")

                self.sensor_data['radar'].append({
                    'timestamp': data.timestamp,
                    'frame': data.frame,
                    'points': points,
                    'transform': data.transform
                })

        except Exception as e:
            logging.error(f"Error in sensor callback ({sensor_type}): {str(e)}")

    def cleanup_actors(self):
        """Clean up all actors spawned during simulation"""
        try:
            logging.info("Cleaning up actors...")
            for actor in self.active_actors:
                try:
                    if actor is not None and actor.is_alive:
                        actor.destroy()
                except Exception as e:
                    logging.warning(f"Failed to destroy actor: {str(e)}")
            self.active_actors.clear()

            # Clean up sensors specifically
            for sensor in self.active_sensors:
                try:
                    if sensor is not None and sensor.is_alive:
                        sensor.destroy()
                except Exception as e:
                    logging.warning(f"Failed to destroy sensor: {str(e)}")
            self.active_sensors.clear()

        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

    def run_simulation(self, config_name, duration_seconds=600):
        try:
            # Enable synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.025  # 40 FPS
            self.world.apply_settings(settings)
            
            # Setup traffic manager
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            
            vehicle = None
            sensors = []
            traffic_vehicles = []
            pedestrians = []

            # Set up weather
            weather = self.setup_weather()
            logging.info("Weather configured successfully")

            # Spawn ego vehicle with collision checking
            # blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
            blueprint = self.world.get_blueprint_library().find('vehicle.yamaha.yzf')
            # blueprint = self.world.get_blueprint_library().find('vehicle.mercedes.sprinter')
            spawn_points = self.map.get_spawn_points()

            if not spawn_points:
                raise SimulationError("No spawn points available")

            # Try different spawn points until finding one without collision
            spawn_success = False
            random.shuffle(spawn_points)  # Randomize spawn points

            for spawn_point in spawn_points:
                try:
                    # Check if spawn point is clear
                    if self.world.get_spectator().get_transform().location.distance(spawn_point.location) > 2.0:
                        vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
                        if vehicle is not None:
                            spawn_success = True
                            break
                except Exception as e:
                    logging.warning(f"Failed to spawn at point {spawn_point}: {str(e)}")
                    continue

            if not spawn_success:
                raise SimulationError("Could not find a clear spawn point for ego vehicle")

            self.active_actors.append(vehicle)
            vehicle.set_autopilot(True)
            logging.info(f"Ego vehicle spawned successfully at {spawn_point}")

            # Wait a moment for physics to settle
            time.sleep(0.5)

            # Setup traffic
            traffic_vehicles, pedestrians = self.setup_traffic()
            logging.info("Traffic setup completed")

            # Setup sensors
            sensors = self.setup_sensors(vehicle, config_name)
            self.active_sensors.extend(sensors)
            logging.info(f"Sensors setup completed for configuration: {config_name}")

            # Main simulation loop
            start_time = time.time()
            frame_count = 0

            while time.time() - start_time < duration_seconds:
                try:
                    # Tick the world
                    self.world.tick()
                    
                    # Visualize data
                    self.visualize_data()
                    
                    frame_count += 1
                    
                    # Process Pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return
                
                except Exception as e:
                    logging.error(f"Error in simulation loop: {str(e)}")
                    break
                
        finally:
            self.export_data(".")

            # Disable synchronous mode when done
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

    def export_data(self, output_dir):
        """
        Export collected sensor data to the specified output directory.
        
        Args:
            output_dir (Path or str): Directory to save the exported data
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ensure output directory exists
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export metadata as JSON
            try:
                metadata = {
                    'timestamp': timestamp,
                    'weather_conditions': self.sensor_data['weather'],
                    'num_frames': {
                        'camera': len(self.sensor_data['camera']),
                        'lidar': len(self.sensor_data['lidar']),
                        'radar': len(self.sensor_data['radar']),
                        'semantic': len(self.sensor_data['semantic']),
                        'depth': len(self.sensor_data['depth'])
                    }
                }
                
                with open(output_dir / f'metadata_{timestamp}.json', 'w') as f:
                    json.dump(metadata, f, indent=4)
                logging.info("Metadata export completed")
                
            except Exception as e:
                logging.error(f"Failed to export metadata: {str(e)}")
            
            # Export camera data (images)
            if self.sensor_data['camera']:
                try:
                    camera_dir = output_dir / 'camera_data'
                    camera_dir.mkdir(exist_ok=True)
                    
                    for idx, frame in enumerate(self.sensor_data['camera']):
                        # Save as PNG to preserve quality
                        img_path = camera_dir / f'frame_{idx:06d}.png'
                        cv2.imwrite(str(img_path), cv2.cvtColor(frame['data'], cv2.COLOR_RGB2BGR))
                    
                    logging.info(f"Exported {len(self.sensor_data['camera'])} camera frames")
                except Exception as e:
                    logging.error(f"Failed to export camera data: {str(e)}")
            
            # Export LiDAR points with serializable transforms
            if self.sensor_data['lidar']:
                try:
                    lidar_data = {
                        'timestamps': [frame['timestamp'] for frame in self.sensor_data['lidar']],
                        'points': [frame['points'] for frame in self.sensor_data['lidar']],
                        # Convert Transform objects to dictionaries
                        'transforms': [{
                            'location': {
                                'x': frame['transform'].location.x,
                                'y': frame['transform'].location.y,
                                'z': frame['transform'].location.z
                            },
                            'rotation': {
                                'pitch': frame['transform'].rotation.pitch,
                                'yaw': frame['transform'].rotation.yaw,
                                'roll': frame['transform'].rotation.roll
                            }
                        } for frame in self.sensor_data['lidar']]
                    }
                    
                    with open(output_dir / f'lidar_data_{timestamp}.pkl', 'wb') as f:
                        pickle.dump(lidar_data, f)
                    logging.info(f"Exported {len(self.sensor_data['lidar'])} LiDAR frames")
                except Exception as e:
                    logging.error(f"Failed to export LiDAR data: {str(e)}")
            
            # Export radar data as CSV
            if self.sensor_data['radar']:
                try:
                    radar_frames = []
                    for frame in self.sensor_data['radar']:
                        df = pd.DataFrame(
                            frame['points'],
                            columns=['velocity', 'azimuth', 'altitude', 'depth']
                        )
                        df['timestamp'] = frame['timestamp']
                        df['frame'] = frame.get('frame', 0)
                        radar_frames.append(df)
                    
                    if radar_frames:
                        radar_df = pd.concat(radar_frames, ignore_index=True)
                        radar_df.to_csv(output_dir / f'radar_data_{timestamp}.csv', index=False)
                        logging.info(f"Exported {len(radar_frames)} radar frames")
                except Exception as e:
                    logging.error(f"Failed to export radar data: {str(e)}")
            
            # Export semantic segmentation data if available
            if self.sensor_data['semantic']:
                try:
                    semantic_dir = output_dir / 'semantic_data'
                    semantic_dir.mkdir(exist_ok=True)
                    
                    for idx, frame in enumerate(self.sensor_data['semantic']):
                        semantic_path = semantic_dir / f'frame_{idx:06d}.png'
                        cv2.imwrite(str(semantic_path), frame['data'])
                    
                    logging.info(f"Exported {len(self.sensor_data['semantic'])} semantic segmentation frames")
                except Exception as e:
                    logging.error(f"Failed to export semantic segmentation data: {str(e)}")
            
            # Export depth data if available
            if self.sensor_data['depth']:
                try:
                    depth_dir = output_dir / 'depth_data'
                    depth_dir.mkdir(exist_ok=True)
                    
                    for idx, frame in enumerate(self.sensor_data['depth']):
                        depth_path = depth_dir / f'frame_{idx:06d}.png'
                        # Normalize depth data for visualization
                        depth_data = frame['data'].astype(np.float32)
                        depth_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
                        cv2.imwrite(str(depth_path), depth_data.astype(np.uint8))
                    
                    logging.info(f"Exported {len(self.sensor_data['depth'])} depth frames")
                except Exception as e:
                    logging.error(f"Failed to export depth data: {str(e)}")
            
            logging.info(f"Data export completed successfully to {output_dir}")
            
        except Exception as e:
            logging.error(f"Error during data export: {str(e)}")
            raise SimulationError(f"Data export failed: {str(e)}")
    def visualize_data(self):
        try:
            # Process camera image
            if not self.image_queue.empty():
                frame, image = self.image_queue.get()
                if not np.isnan(image).any():  # Check for NaN values
                    image = np.rot90(image)
                    image = pygame.surfarray.make_surface(image)
                    self.display.blit(image, (0, 0))
            
            # Process LiDAR data
            if not self.lidar_queue.empty():
                frame, points = self.lidar_queue.get()
                lidar_surface = pygame.Surface((400, 400))
                lidar_surface.fill((0, 0, 0))
                
                if points.size > 0 and not np.isnan(points).any():  # Check for NaN values
                    points_2d = points[:, :2]
                    points_2d = points_2d * 10
                    points_2d += np.array([200, 200])
                    
                    # Remove any points outside the visible area
                    mask = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < 400) & 
                           (points_2d[:, 1] >= 0) & (points_2d[:, 1] < 400))
                    points_2d = points_2d[mask]
                    
                    if points_2d.size > 0:
                        heights = points[mask, 2]
                        heights_norm = np.clip((heights - np.min(heights)) / 
                                             (np.max(heights) - np.min(heights) + 1e-6), 0, 1)
                        
                        for i, (x, y) in enumerate(points_2d):
                            try:
                                color = (int(255 * heights_norm[i]), 0, 
                                       int(255 * (1 - heights_norm[i])))
                                pygame.draw.circle(lidar_surface, color, 
                                                (int(x), int(y)), 2)
                            except (ValueError, IndexError):
                                continue
                
                self.display.blit(lidar_surface, (0, self.display.get_height() - 400))
            
            # Process radar data
            if not self.radar_queue.empty():
                frame, radar_data = self.radar_queue.get()
                radar_surface = pygame.Surface((200, 200))
                radar_surface.fill((0, 0, 0))
                
                if radar_data.size > 0 and not np.isnan(radar_data).any():  # Check for NaN values
                    for detection in radar_data:
                        velocity, azimuth, altitude, depth = detection
                        if not any(np.isnan([velocity, azimuth, altitude, depth])):
                            x = depth * math.cos(azimuth) * 2
                            y = depth * math.sin(azimuth) * 2
                            
                            screen_x = int(100 + x)
                            screen_y = int(100 + y)
                            
                            if 0 <= screen_x < 200 and 0 <= screen_y < 200:
                                color = (255, 0, 0) if velocity < 0 else (0, 255, 0)
                                pygame.draw.circle(radar_surface, color, 
                                                (screen_x, screen_y), 3)
                
                # Display radar visualization in bottom-right corner
                self.display.blit(radar_surface, (self.display.get_width() - 200, 
                                                self.display.get_height() - 200))
            
            # Add text overlay with basic info
            try:
                font = pygame.font.Font(None, 36)
                text_color = (255, 255, 255)
                
                # Display frame counts
                camera_text = f"Camera Frames: {len(self.sensor_data['camera'])}"
                lidar_text = f"LiDAR Frames: {len(self.sensor_data['lidar'])}"
                radar_text = f"Radar Frames: {len(self.sensor_data['radar'])}"
                
                camera_surface = font.render(camera_text, True, text_color)
                lidar_surface = font.render(lidar_text, True, text_color)
                radar_surface = font.render(radar_text, True, text_color)
                
                # Position text in top-right corner
                self.display.blit(camera_surface, (self.display.get_width() - 300, 10))
                self.display.blit(lidar_surface, (self.display.get_width() - 300, 50))
                self.display.blit(radar_surface, (self.display.get_width() - 300, 90))
                
            except Exception as e:
                logging.warning(f"Failed to render text overlay: {str(e)}")
            
            # Update display
            pygame.display.flip()
            
        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")
            # Don't raise the error to prevent simulation from stopping
            pass
def main():
    simulation = None
    try:
        simulation = AVSimulation()
        configs = ['minimal', 'standard', 'advanced']

        print("\nAvailable sensor configurations:")
        for i, config in enumerate(configs):
            print(f"{i+1}. {config}")

        while True:
            try:
                choice = input("\nSelect configuration (1-3) or 0 to exit: ")
                if not choice.strip():
                    continue

                choice = int(choice)
                if choice == 0:
                    break
                if 1 <= choice <= len(configs):
                    logging.info(f"Starting simulation with {configs[choice-1]} configuration...")
                    simulation.run_simulation(configs[choice-1])
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                logging.info("Simulation interrupted by user")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                break

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        traceback.print_exc()

    finally:
        if simulation is not None:
            simulation.cleanup_actors()
        pygame.quit()
        logging.info("Simulation ended")

if __name__ == '__main__':
    main()
