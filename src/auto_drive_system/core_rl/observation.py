import gym
import numpy as np

class CarlaObservations():

    def __init__(self, img_height, img_width):

        self.img_height = img_height
        self.img_width = img_width

    def get_observation_space(self):
        return gym.spaces.Box(low=0.0, high=255.0, shape=(self.img_height, self.img_width, 3), dtype=np.uint8)
