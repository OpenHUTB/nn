import carla
import numpy as np
import queue
from PIL import Image
from torchvision import transforms
import random


class ActorCar(object):
    """ActorCar is the combination of car and attached camera.
    important members:
        self.actor_car
        self.rgb_camera
        self.col_sensor
    """

    def __init__(self, client, world, bp, spawn_points, config):
        self.client = client
        self.actor_list = []
        car = bp.filter('model3')[0]
        spawn_point = random.choice(spawn_points[config['car_num']:])
        self.actor_car = world.spawn_actor(car, spawn_point)
        self.actor_list.append(self.actor_car)
        camera = bp.find('sensor.camera.rgb')
        camera.set_attribute('image_size_x', '640')
        camera.set_attribute('image_size_y', '480')
        camera.set_attribute('fov', '110')
        transform = carla.Transform(carla.Location(x=1.2, z=1.7))
        self.rgb_camera = world.spawn_actor(camera, transform,
                                            attach_to=self.actor_car)
        self.actor_list.append(self.rgb_camera)
        # tips: collision sensor only receive data when triggered
        collision_sensor = bp.find('sensor.other.collision')
        self.col_sensor = world.spawn_actor(collision_sensor,
                                            transform, attach_to=self.actor_car)
        self.actor_list.append(self.col_sensor)

        self.front_camera = None
        self.collision_intensity = 0
        self._camera_queue = queue.Queue()
        self._col_queue = queue.Queue()
        self.rgb_camera.listen(self._camera_queue.put)
        self.col_sensor.listen(self._col_queue.put)
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def retrieve_data(self, frame_index):
        while not self.process_img(frame_index):
            pass
        # self.process_img(frame_index)
        self.process_col_event(frame_index)
        return self.front_camera, self.collision_intensity

    def process_img(self, frame_index):
        if not self._camera_queue.empty():
            img = self._camera_queue.get(timeout=2)
            # print("current size of q images", self._col_queue.qsize())
            assert self._camera_queue.qsize() == 0, "Expected qsize of images 0"
            assert frame_index == img.frame, "not the corresponding frame image."
            print(f"current image frame is: {img.frame}")
            # BGRA
            img = np.reshape(img.raw_data, (640, 480, 4))
            # slice BGR
            img = img[:, :, :3]
            # reverse to RGB
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img)).convert('RGB')
            self.front_camera = self.image_transform(img)
            return True
        self.front_camera = None
        return False

    def process_col_event(self, frame_index):
        if not self._col_queue.empty():
            event = self._col_queue.get(timeout=2)
            assert frame_index == event.frame, "not the corresponding frame event."
            impulse = event.normal_impulse
            self.collision_intensity = impulse.length()
            print(f"collision length is: {self.collision_intensity}")
            print(
                f"current collision frame is: {event.frame}")

    def cleanup(self):
        """cleanup is to destroy all agent actors in the world."""
        self.rgb_camera.stop()
        self.col_sensor.stop()
        self.client.apply_batch([carla.command.DestroyActor(x)
                                 for x in self.actor_list])
        print("destroy all actors of agent")
