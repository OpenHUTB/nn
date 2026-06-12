
import time
import os

import numpy as np
from PIL import Image

from agents.agent import Agent
from airsim_utils import get_velocity, parse_state, is_new_collision


class VisionFlyer(Agent):
    def __init__(self, client, move_type):
        super(VisionFlyer, self).__init__(client, move_type)
        self.client.start()

    def get_state(self):
        return self.client.get_state()

    def get_image(self, camera_number):
        return self.client.get_images(camera_number=camera_number)

    def save_image(self, image, path):
        self.client.save_image(path, image)

    def act(self):
        # relative velocity
        state = self.client.get_state()
        _, kinematic, _ = parse_state(state)
        velocity = get_velocity(kinematic)
        # random move
        velocity = [
            velocity[0] + np.random.uniform(-0.5, 0.5),
            velocity[1] + np.random.uniform(-0.5, 0.5),
            velocity[2] + np.random.uniform(-0.5, 0.5)
        ]
        self.client.move(self.move_type, *velocity)

    def run(self, loop_cnt=100):
        for epoch in range(loop_cnt):
            self.act()
            for cam in ['0', '1']:
                images = self.get_image(cam)
                for idx, img in enumerate(images):
                    self._save_image(img, idx, self._get_repr(cam, epoch, idx))
            time.sleep(0.1)
            collision = self.client.get_collision_info()
            if is_new_collision(self.cur_collision, collision):
                print(collision)
                self.cur_collision = collision

    @staticmethod
    def _get_repr(camera, epoch, idx):
        return f'{camera}_{epoch}_{idx}'

    def _save_image(self, image, idx, name):
        height, width = image.height, image.width
        if idx == 0:
            image = np.fromstring(image.image_data_uint8, dtype=np.int8)
            image = image.reshape(height, width, 4)
            image = Image.fromarray(image, 'RGBA')
        else:
            image = np.array(image.image_data_float, dtype=np.float)
            image = 255 / np.maximum(np.ones(image.size), image)
            image = np.reshape(image, (height, width))
            image = Image.fromarray(image).convert('L')
        image.save(os.path.join(self.client.root_path, f'img_{name}.png'))
