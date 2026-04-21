import cv2
import carla
import pygame
import numpy as np

from . import LaneDetector, CarDetector
from . import YOLOPv2Detector
from . import LidarManager


class SensorHandler:
    def __init__(self, world, sensor_options):

        self.time_processing = 0.0
        self.num_frames = 0
        self.fps = 0.0
        self.world = world
        self.sensor_options = sensor_options
        self.sensors = []
        self.data_recorder = None  # Will be set externally

        self.LaneDetector = LaneDetector()
        self.CarDetector = CarDetector()
        self.YOLOPv2Detector = YOLOPv2Detector()
        self.LidarManager = LidarManager()
    
    def set_data_recorder(self, data_recorder):
        """Set the data recorder instance for RGB image recording."""
        self.data_recorder = data_recorder
    
    def rgbcamera_lane_callback(self, image: carla.Image, sensor_manager=None) -> np.ndarray:
        try:
            # Convert CARLA image to numpy array
            img = np.frombuffer(image.raw_data, dtype=np.uint8)
            img = img.reshape((image.height, image.width, 4))[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Process detections
            output, lanes = self.LaneDetector.detect(img)
            
            if len(lanes) == 2:
                output = self.LaneDetector.draw_lane_area(output, lanes)
            
            output = self.LaneDetector.detect(output)
            
            # Ensure output is a valid numpy array
            if not isinstance(output, np.ndarray):
                output = img.copy()
            
            # Draw status text
            cv2.putText(output, "Status: Traditional Detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if sensor_manager and sensor_manager.display_man.render_enable():
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                sensor_manager.surface = pygame.surfarray.make_surface(output_rgb.swapaxes(0, 1))

        except Exception as e:
            print(f"Error in lane detection callback: {str(e)}")

    def rgbcamera_lane_edges_callback(self, image: carla.Image, sensor_manager=None) -> np.ndarray:
        try:
            # Convert CARLA image to numpy array
            img = np.frombuffer(image.raw_data, dtype=np.uint8)
            img = img.reshape((image.height, image.width, 4))[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Get lane edges
            output = self.LaneDetector.edges_frame(img)
            
            # Ensure output is a valid numpy array
            if not isinstance(output, np.ndarray):
                output = img.copy()

            if sensor_manager and sensor_manager.display_man.render_enable():
                sensor_manager.surface = pygame.surfarray.make_surface(output.swapaxes(0, 1))
            
        except Exception as e:
            print(f"Error in lane edges callback: {str(e)}")
            
    def rgbcamera_yolopv2_callback(self, image: carla.Image, sensor_manager=None) -> np.ndarray:
        try:
            # Convert CARLA image to numpy array
            img = np.frombuffer(image.raw_data, dtype=np.uint8)
            img = img.reshape((image.height, image.width, 4))[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    
            # Run YOLOPv2 detection
            output, detections = self.YOLOPv2Detector.detect(img)
            
            # Ensure output is a valid numpy array
            if not isinstance(output, np.ndarray):
                output = img.copy()
            
            if sensor_manager and sensor_manager.display_man.render_enable():
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                sensor_manager.surface = pygame.surfarray.make_surface(output_rgb.swapaxes(0, 1))
            
        except Exception as e:
            print(f"Error in YOLOPv2 callback: {str(e)}")
            
    def render_lidar_image(self, image: carla.Image, sensor_manager=None):
        try:
            display_size = sensor_manager.display_man.get_display_size()
            lidar_range = 2.0 * float(self.sensor_options['range'])

            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(display_size) / lidar_range
            lidar_data += (0.5 * display_size[0], 0.5 * display_size[1])
            lidar_data = np.fabs(lidar_data).astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_image_size = (display_size[0], display_size[1], 3)
            lidar_image = np.zeros((lidar_image_size), dtype=np.uint8)

            lidar_image[tuple(lidar_data.T)] = (255, 255, 255)

            if sensor_manager and sensor_manager.display_man.render_enable():
                sensor_manager.surface = pygame.surfarray.make_surface(lidar_image)
                
        except Exception as e:
            print(f"Error in Lidar callback: {str(e)}")
        
    def render_rgb_image(self, image: carla.Image, sensor_manager=None):
        try:
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]  # Convert from BGRA to RGB

            # Feed RGB image to data recorder if available
            if self.data_recorder is not None:
                self.data_recorder.update_rgb_image(array)

            if sensor_manager and sensor_manager.display_man.render_enable():
                sensor_manager.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                
        except Exception as e:
            print(f"Error in RGBcamera callback: {str(e)}")

    def render_depth_image(self, image: carla.Image, sensor_manager=None):
        try:
            image.convert(carla.ColorConverter.LogarithmicDepth)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            if sensor_manager and sensor_manager.display_man.render_enable():
                sensor_manager.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        except Exception as e:
            print(f"Error in Depthcamera callback: {str(e)}")

    def render_semantic_image(self, semantic_lidar_data, sensor_manager=None):
        try:
            display_size = sensor_manager.display_man.get_display_size()
            lidar_range = 2.0 * float(self.sensor_options['range'])

            points = np.frombuffer(semantic_lidar_data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 6), 6))
            # points = np.reshape(points, (-1, 6))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(display_size) / lidar_range
            lidar_data += (0.5 * display_size[0], 0.5 * display_size[1])
            lidar_data = np.fabs(lidar_data).astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_image = np.zeros((display_size[0], display_size[1], 3), dtype=np.uint8)

            lidar_image[tuple(lidar_data.T)] = (255, 255, 255)

            if sensor_manager and sensor_manager.display_man.render_enable():
                sensor_manager.surface = pygame.surfarray.make_surface(lidar_image)
                
        except Exception as e:
            print(f"Error in SemanticLidar callback: {str(e)}")

    def render_radar_image(self, radar_data, sensor_manager=None):
        try:
            points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (len(radar_data), 4))
            
        except Exception as e:
            print(f"Error in Radar callback: {str(e)}")

    def render_semantic_segmentation_image(self, image, sensor_manager=None):
        try:
            image.convert(carla.ColorConverter.CityScapesPalette)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            if sensor_manager and sensor_manager.display_man.render_enable():
                sensor_manager.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        except Exception as e:
            print(f"Error in SemanticSegmentation callback: {str(e)}")

    def render_instance_segmentation_image(self, image, sensor_manager=None):
        try:
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            if sensor_manager and sensor_manager.display_man.render_enable():
                sensor_manager.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        except Exception as e:
            print(f"Error in InstanceSegmentation callback: {str(e)}")