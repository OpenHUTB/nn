import time
import carla
import numpy as np
import cv2
import queue
import pygame


from utils import eagle_eye_map, DisplayManager
from . import SensorHandler, CustomTimer, WINDOW_SIZE



class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_position):
        self.CustomTimer = CustomTimer()
        self.world = world
        self.display_man = display_man
        self.sensor_type = sensor_type
        self.display_position = display_position
        self.sensors_oks = []
        self.sensors = []
        
        self.sensor_options = sensor_options
        self.surface = None
        self.display_man.add_sensor(self)
        
        self.CustomTimer = CustomTimer()
        self.SensorHandler = SensorHandler(world, self.sensor_options)
        self.data_recorder = None  # Will be set externally
        
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
    
    def set_data_recorder(self, data_recorder):
        """Set the data recorder instance for all sensor handlers."""
        self.data_recorder = data_recorder
        self.SensorHandler.set_data_recorder(data_recorder)

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        print(f"Initializing sensor: {sensor_type}")

        # RGBCamera
        if sensor_type == 'RGBCamera':
            rgb_camera_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            display_size = self.display_man.get_display_size()
            rgb_camera_blueprint.set_attribute('image_size_x', str(display_size[0]))
            rgb_camera_blueprint.set_attribute('image_size_y', str(display_size[1]))

            for key in sensor_options:
                rgb_camera_blueprint.set_attribute(key, sensor_options[key])
            
            rgb_camera = self.world.spawn_actor(rgb_camera_blueprint, transform, attached)
            rgb_camera.listen(
                lambda image: (
                    self.SensorHandler.render_rgb_image(image, self),
                    self.CustomTimer.timer_callback(sensor_type, image.timestamp)
                )
            )

            return rgb_camera
        
        if sensor_type == 'RGBCamera_BEV':
            rgb_camera_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            display_size = self.display_man.get_display_size()
            rgb_camera_blueprint.set_attribute('image_size_x', str(display_size[0]))
            rgb_camera_blueprint.set_attribute('image_size_y', str(display_size[1]))

            for key in sensor_options:
                rgb_camera_blueprint.set_attribute(key, sensor_options[key])
            
            rgb_camera = self.world.spawn_actor(rgb_camera_blueprint, transform, attached)
            rgb_camera.listen(
                lambda image: (
                    self.SensorHandler.render_rgb_image(image, self),
                    self.CustomTimer.timer_callback(sensor_type, image.timestamp)
                )
            )

            return rgb_camera
        
        if sensor_type == 'RGBCamera_Lane':
            rgb_camera_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            display_size = self.display_man.get_display_size()
            rgb_camera_blueprint.set_attribute('image_size_x', str(display_size[0]))
            rgb_camera_blueprint.set_attribute('image_size_y', str(display_size[1]))

            for key in sensor_options:
                rgb_camera_blueprint.set_attribute(key, sensor_options[key])
            
            rgb_camera = self.world.spawn_actor(rgb_camera_blueprint, transform, attached)
            
            # rgb_camera.listen(lambda image: self.SensorHandler.rgbcamera_lane_callback(image, self))
            rgb_camera.listen(
                lambda image: (
                    self.SensorHandler.rgbcamera_yolopv2_callback(image, self),
                    self.CustomTimer.timer_callback(sensor_type, image.timestamp)
                )
            )
            return rgb_camera
        
        if sensor_type == 'RGBCamera_Lane_Edges':
            rgb_camera_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            display_size = self.display_man.get_display_size()
            rgb_camera_blueprint.set_attribute('image_size_x', str(display_size[0]))
            rgb_camera_blueprint.set_attribute('image_size_y', str(display_size[1]))

            for key in sensor_options:
                rgb_camera_blueprint.set_attribute(key, sensor_options[key])
            
            rgb_camera = self.world.spawn_actor(rgb_camera_blueprint, transform, attached)
            rgb_camera.listen(
                lambda image: (
                    self.SensorHandler.rgbcamera_lane_edges_callback(image, self),
                    self.CustomTimer.timer_callback(sensor_type, image.timestamp)
                )
            )
            return rgb_camera
        
        # Depth Camera
        elif sensor_type == 'DepthCamera':
            depth_camera_blueprint = self.world.get_blueprint_library().find('sensor.camera.depth')
            for key in sensor_options:
                depth_camera_blueprint.set_attribute(key, sensor_options[key])

            depth_camera = self.world.spawn_actor(depth_camera_blueprint, transform, attached)
            depth_camera.listen(
                lambda image: (
                    self.SensorHandler.render_depth_image(image, self),
                    self.CustomTimer.timer_callback(sensor_type, image.timestamp)
                )
            )
            return depth_camera

        # Lidar
        elif sensor_type == 'LiDAR':
            lidar_blueprint = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_blueprint.set_attribute('range', '100')
            lidar_blueprint.set_attribute('dropoff_general_rate', lidar_blueprint.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_blueprint.set_attribute('dropoff_intensity_limit', lidar_blueprint.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_blueprint.set_attribute('dropoff_zero_intensity', lidar_blueprint.get_attribute('dropoff_zero_intensity').recommended_values[0])
            
            for key in sensor_options:
                lidar_blueprint.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_blueprint, transform, attached)
            lidar.listen(
                lambda image: (
                    self.SensorHandler.render_lidar_image(image, self),
                    self.CustomTimer.timer_callback(sensor_type, image.timestamp)
                )
            )
            return lidar
        
        # Semantic Lidar
        elif sensor_type == 'SemanticLiDAR':
            semantic_Lidar_blueprint = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            semantic_Lidar_blueprint.set_attribute('range', '100')

            for key in sensor_options:
                semantic_Lidar_blueprint.set_attribute(key, sensor_options[key])

            semantic_Lidar = self.world.spawn_actor(semantic_Lidar_blueprint, transform, attached)
            semantic_Lidar.listen(
                lambda image: (
                    self.SensorHandler.render_semantic_image(image, self),
                    self.CustomTimer.timer_callback(sensor_type, image.timestamp)
                )
            )
            return semantic_Lidar
        
        # Radar
        elif sensor_type == 'Radar':
            radar_blueprint = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_blueprint.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_blueprint, transform, attached)
            radar.listen(
                lambda image: (
                    self.SensorHandler.render_radar_image(image, self),
                    self.CustomTimer.timer_callback(sensor_type, image.timestamp)
                )
            )
            return radar
        
        # Semantic Segmentation Camera
        elif sensor_type == 'SemanticSegmentationCamera':
            semantic_segmentation_camera_blueprint = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            display_size = self.display_man.get_display_size()
            semantic_segmentation_camera_blueprint.set_attribute('image_size_x', str(display_size[0]))
            semantic_segmentation_camera_blueprint.set_attribute('image_size_y', str(display_size[1]))
            
            for key in sensor_options:
                semantic_segmentation_camera_blueprint.set_attribute(key, sensor_options[key])

            semantic_segmentation_camera = self.world.spawn_actor(semantic_segmentation_camera_blueprint, transform, attached, attachment_type=carla.AttachmentType.Rigid)
            semantic_segmentation_camera.listen(
                lambda image: (
                    self.SensorHandler.render_semantic_segmentation_image(image, self),
                    self.CustomTimer.timer_callback(sensor_type, image.timestamp)
                )
            )
            return semantic_segmentation_camera
        
        # Instance Segmentation Camera
        elif sensor_type == 'InstanceSegmentationCamera':
            instance_segmentation_camera_blueprint = self.world.get_blueprint_library().find('sensor.camera.instance_segmentation')
            for key in sensor_options:
                instance_segmentation_camera_blueprint.set_attribute(key, sensor_options[key])

            instance_segmentation_camera = self.world.spawn_actor(instance_segmentation_camera_blueprint, transform, attached, attachment_type=carla.AttachmentType.Rigid)
            instance_segmentation_camera.listen(
                lambda image: (
                    self.SensorHandler.render_instance_segmentation_image(image, self),
                    self.CustomTimer.timer_callback(sensor_type, image.timestamp)
                )
            )
            return instance_segmentation_camera

        else:
            None
            
    def destroy(self):
        if self.sensor is not None:
            print(f"Destroy {self.sensor_type}")
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_position)
            self.display_man.display.blit(self.surface, offset)

def setup_sensors(world, ego_vehicle, displaymanager, sensors_dict):
        """Initialize and configure all sensors"""
        # Calculate cell dimensions for grid layout
        cell_width = WINDOW_SIZE[0] // 4
        cell_height = WINDOW_SIZE[1] // 2
        
        senser_mangers = []

        

        # Initialize sensors with specific configurations
        for sensor_name, ((x, y, z), position, enabled) in sensors_dict.items():
            if sensor_name == 'DepthCamera' and enabled:
                sm = SensorManager(
                    world, displaymanager, 'DepthCamera',
                    carla.Transform(carla.Location(x=x, z=z), carla.Rotation(yaw=0)),
                    ego_vehicle, {'fov': '90'}, position
                )
                senser_mangers.append(sm)
            
            if sensor_name == 'RGBCamera' and enabled:
                sm = SensorManager(
                    world, displaymanager, 'RGBCamera',
                    carla.Transform(carla.Location(x=x, z=z), carla.Rotation(yaw=0)),
                    ego_vehicle, {'fov': '90'}, position
                )
                senser_mangers.append(sm)
                
            if sensor_name == 'RGBCamera_BEV' and enabled:
                sm = SensorManager(
                    world, displaymanager, 'RGBCamera_BEV',
                    carla.Transform(carla.Location(x=x, z=z), carla.Rotation(yaw=0, pitch=-90.0)),
                    ego_vehicle, {'fov': '90'}, position
                )
                senser_mangers.append(sm)
            
            if sensor_name == 'RGBCamera_Lane' and enabled:
                sm = SensorManager(
                    world, displaymanager, 'RGBCamera_Lane',
                    carla.Transform(carla.Location(x=x, z=z), carla.Rotation(yaw=0)),
                    ego_vehicle, {'fov': '90'}, position
                )
                senser_mangers.append(sm)

            if sensor_name == 'RGBCamera_Lane_Edges' and enabled:
                sm = SensorManager(
                    world, displaymanager, 'RGBCamera_Lane_Edges',
                    carla.Transform(carla.Location(x=x, z=z), carla.Rotation(yaw=0)),
                    ego_vehicle, {'fov': '90'}, position
                )
                senser_mangers.append(sm)

            if sensor_name == 'SemanticSegmentationCamera' and enabled:
                sm = SensorManager(
                    world, displaymanager, 'SemanticSegmentationCamera',
                    carla.Transform(carla.Location(x=x, z=z), carla.Rotation(yaw=0)),
                    ego_vehicle, {'fov': '90'}, position
                )
                senser_mangers.append(sm)

            if sensor_name == 'InstanceSegmentationCamera' and enabled:
                sm = SensorManager(
                    world, displaymanager, 'InstanceSegmentationCamera',
                    carla.Transform(carla.Location(x=x, z=z), carla.Rotation(yaw=0)),
                    ego_vehicle, {'fov': '90'}, position
                )
                senser_mangers.append(sm)

            if sensor_name == 'LiDAR' and enabled:
                sm = SensorManager(
                    world, displaymanager, 'LiDAR',
                    carla.Transform(carla.Location(x=x, z=z)),
                    ego_vehicle,
                    {
                        'channels': '64',
                        'range': '100',
                        'points_per_second': '250000',
                        'rotation_frequency': '20'
                    }, position
                )
                senser_mangers.append(sm)

            if sensor_name == 'SemanticLiDAR' and enabled:
                sm = SensorManager(
                    world, displaymanager, 'SemanticLiDAR',
                    carla.Transform(carla.Location(x=x, z=z)),
                    ego_vehicle,
                    {
                        'channels': '64',
                        'range': '100',
                        'points_per_second': '100000',
                        'rotation_frequency': '20'
                    }, position
                )
                senser_mangers.append(sm)

        return senser_mangers