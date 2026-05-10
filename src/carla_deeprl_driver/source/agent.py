import carla
import numpy as np
import queue
from PIL import Image
from torchvision import transforms
import random
from typing import Tuple, Optional


class ActorCar:
    """ActorCar combines a vehicle with attached sensors for autonomous driving.
    
    Attributes:
        actor_car: The CARLA vehicle actor
        rgb_camera: RGB camera sensor
        col_sensor: Collision sensor
        front_camera: Processed camera image tensor
        collision_intensity: Collision impulse magnitude
    """

    def __init__(self, client: carla.Client, world: carla.World, bp: carla.BlueprintLibrary, 
                 spawn_points: list, config: dict):
        self.client = client
        self.actor_list = []
        
        car_bp = bp.filter('model3')[0]
        spawn_point = random.choice(spawn_points[config['car_num']:])
        self.actor_car = world.spawn_actor(car_bp, spawn_point)
        self.actor_list.append(self.actor_car)
        
        camera_bp = bp.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.2, z=1.7))
        self.rgb_camera = world.spawn_actor(camera_bp, camera_transform, attach_to=self.actor_car)
        self.actor_list.append(self.rgb_camera)
        
        collision_bp = bp.find('sensor.other.collision')
        self.col_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.actor_car)
        self.actor_list.append(self.col_sensor)

        self.front_camera: Optional[torch.Tensor] = None
        self.collision_intensity = 0.0
        
        self._camera_queue = queue.Queue()
        self._col_queue = queue.Queue()
        self.rgb_camera.listen(self._camera_queue.put)
        self.col_sensor.listen(self._col_queue.put)
        
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def retrieve_data(self, frame_index: int) -> Tuple[Optional[torch.Tensor], float]:
        """Retrieve sensor data for a specific frame.
        
        Args:
            frame_index: The expected frame number
            
        Returns:
            Tuple of (processed camera image, collision intensity)
        """
        self.process_img(frame_index)
        self.process_col_event(frame_index)
        return self.front_camera, self.collision_intensity

    def process_img(self, frame_index: int) -> bool:
        """Process camera image from queue.
        
        Args:
            frame_index: The expected frame number
            
        Returns:
            True if image was processed successfully, False otherwise
        """
        if not self._camera_queue.empty():
            img = self._camera_queue.get(timeout=2)
            if frame_index == img.frame:
                img_data = np.reshape(img.raw_data, (640, 480, 4))
                img_data = img_data[:, :, :3]
                img_data = img_data[:, :, ::-1]
                img_pil = Image.fromarray(np.uint8(img_data)).convert('RGB')
                self.front_camera = self.image_transform(img_pil)
                return True
        self.front_camera = None
        return False

    def process_col_event(self, frame_index: int):
        """Process collision event from queue.
        
        Args:
            frame_index: The expected frame number
        """
        if not self._col_queue.empty():
            event = self._col_queue.get(timeout=2)
            if frame_index == event.frame:
                impulse = event.normal_impulse
                self.collision_intensity = impulse.length()

    def cleanup(self):
        """Clean up all actors associated with this agent."""
        self.rgb_camera.stop()
        self.col_sensor.stop()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
