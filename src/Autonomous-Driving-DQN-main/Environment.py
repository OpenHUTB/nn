import numpy as np
import cv2
import time
import random
import math
from Hyperparameters import *

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    im_width = IM_WIDTH
    im_height = IM_HEIGHT

    def __init__(self):
        self.actor_list = []
        self.front_camera = None
        self.collision_history = []
        self.slow_counter = 0
        self.vehicle_x = -81.0  # 模拟车辆位置
        self.velocity_kmh = 0

    def reset(self):
        # 模拟重置
        self.collision_history = []
        self.slow_counter = 0
        self.vehicle_x = -81.0
        self.velocity_kmh = 0
        # 返回假摄像头图像
        self.front_camera = np.random.randint(0, 255, (self.im_height, self.im_width, 3), dtype=np.uint8)
        time.sleep(0.1)
        return self.front_camera

    def collision_data(self, event):
        self.collision_history.append(event)

    def process_img(self):
        # 模拟图像
        self.front_camera = np.random.randint(0, 255, (self.im_height, self.im_width, 3), dtype=np.uint8)
        if self.SHOW_CAM:
            cv2.imshow("", self.front_camera)
            cv2.waitKey(1)

    def reward(self):
        reward = 0
        done = False
        # 模拟速度
        velocity_kmh = self.velocity_kmh
        # 模拟最小距离
        min_dist = random.uniform(4, 20)

        if len(self.collision_history) > 0:
            reward = -5
            done = True
        elif min_dist < 4:
            reward = -2
        elif velocity_kmh == 0:
            reward += -1
        elif 15 < velocity_kmh < 25:
            reward += 1
        elif 35 < velocity_kmh < 45:
            reward += 2

        # 到达终点
        if self.vehicle_x > 155:
            done = True

        return reward, done

    def step(self, action):
        # 模拟动作：0刹车 1中速 2高速
        if action == 0:
            self.velocity_kmh = 0
        elif action == 1:
            self.velocity_kmh = 20
        elif action == 2:
            self.velocity_kmh = 40

        # 模拟车辆前进
        self.vehicle_x += random.uniform(0.5, 2.0)

        # 随机触发碰撞
        if random.random() < 0.01:
            self.collision_history.append(1)

        self.process_img()
        reward, done = self.reward()
        return self.front_camera, reward, done, None