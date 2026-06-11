
import numpy as np

from agents.agent import Agent
from airsim_utils import is_new_collision


class RandomWalker(Agent):
    def __init__(self, client, move_type, random_range):
        super(RandomWalker, self).__init__(client, move_type)
        self.client.start()
        self.random_range = random_range

    def get_state(self):
        return self.client.get_state()

    def act(self):
        target_vel = [
            np.random.uniform(*self.random_range,),
            np.random.uniform(*self.random_range,),
            np.random.uniform(*self.random_range,)
        ]
        self.client.move(self.move_type, *target_vel)

    def run(self, loop_cnt=100):
        for _ in range(loop_cnt):
            self.act()
            collision = self.client.get_collision_info()
            if is_new_collision(self.cur_collision, collision):
                print(collision)
                self.cur_collision = collision
