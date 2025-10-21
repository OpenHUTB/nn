import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from Hyperparameters import *

avg_score = 0
average_reward = 0

# 如果已使用 pip 安装 carla，则无需加载 egg
# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

import carla
from carla import ColorConverter


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    # STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT

    # front_camera = None

    def __init__(self):
        self.actor_list = None
        self.sem_cam = None
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.front_camera = None

        # self.world = self.client.get_world()
        self.world = self.client.load_world('Town03')

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

        self.walker_list = []
        self.collision_history = []

        self.slow_counter = 0

    def spawn_pedestrians_general(self, number, isCross):
        for i in range(number):
            isLeft = random.choice([True, False])
            if isLeft:
                self.spawn_pedestrians_left(isCross)
            else:
                self.spawn_pedestrians_right(isCross)

    def spawn_pedestrians_right(self, isCross):

        blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        walker_bp = random.choice(blueprints_walkers)

        # global walker_list

        for i in range(1):
            walker_bp = random.choice(blueprints_walkers)

            min_x = -50
            max_x = 140
            min_y = -188
            max_y = -183

            if isCross:
                isFirstCross = random.choice([True, False])
                if isFirstCross:
                    min_x = -14
                    max_x = -10.5
                else:
                    min_x = 17
                    max_x = 20.5

            # Randomly select the position for the pedestrian
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)

            spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            while (-10 < spawn_point.location.x < 17) or (70 < spawn_point.location.x < 100):
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            if spawn_point:
                npc = self.world.try_spawn_actor(walker_bp, spawn_point)

            if npc is not None:
                ped_control = carla.WalkerControl()
                ped_control.speed = random.uniform(0.5, 1.0)
                ped_control.direction.y = -1
                ped_control.direction.x = 0.15
                npc.apply_control(ped_control)
                npc.set_simulate_physics(True)
                # self.walker_list.append(npc)

    def spawn_pedestrians_left(self, isCross):

        blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        walker_bp = random.choice(blueprints_walkers)

        # global walker_list

        for i in range(1):
            walker_bp = random.choice(blueprints_walkers)
            # spawn_points = self.world.get_map().get_spawn_points()  # Assuming spawn_points is defined elsewhere
            # npc = self.world.try_spawn_actor(walker_bp, random.choice(spawn_points))

            min_x = -50
            max_x = 140
            min_y = -216
            max_y = -210

            if (isCross):
                isFirstCross = random.choice([True, False])
                if isFirstCross:
                    min_x = -14
                    max_x = -10.5
                else:
                    min_x = 17
                    max_x = 20.5

            # Randomly select the position for the pedestrian
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)

            spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            while (-10 < spawn_point.location.x < 17) or (70 < spawn_point.location.x < 100):
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            if spawn_point:
                npc = self.world.try_spawn_actor(walker_bp, spawn_point)

            if npc is not None:
                ped_control = carla.WalkerControl()
                ped_control.speed = random.uniform(0.7, 1.3)
                ped_control.direction.y = 1
                ped_control.direction.x = -0.05
                npc.apply_control(ped_control)
                npc.set_simulate_physics(True)
                # self.walker_list.append(npc)

    def reset(self):

        walkers = self.world.get_actors().filter('walker.*')
        for walker in walkers:
            walker.destroy()

        vehicles = self.world.get_actors().filter('vehicle.*')
        for v in vehicles:
            v.destroy()

        self.spawn_pedestrians_general(30, True)
        self.spawn_pedestrians_general(10, False)

        self.collision_history = []
        self.actor_list = []

        # self.isHit = False
        self.slow_counter = 0

        # self.transform = random.choice(self.world.get_map().get_spawn_points())

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        spawn_point.location.x = -81.0
        spawn_point.location.y = -195.0
        spawn_point.location.z += 2.0
        spawn_point.rotation.roll = 0.0
        spawn_point.rotation.pitch = 0.0
        spawn_point.rotation.yaw = 0.0
        self.vehicle = self.world.spawn_actor(self.model_3, spawn_point)

        # self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        # self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        # self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        # self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        # self.rgb_cam.set_attribute("fov", f"110")
        self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.sem_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        # self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.sensor = self.world.spawn_actor(self.sem_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))

        return self.front_camera

    def collision_data(self, event):
        # self.isHit = True
        self.collision_history.append(event)

    def process_img(self, image):

        image.convert(carla.ColorConverter.CityScapesPalette)

        processed_image = np.array(image.raw_data)
        processed_image = processed_image.reshape((self.im_height, self.im_width, 4))
        processed_image = processed_image[:, :, :3]

        if self.SHOW_CAM:
            cv2.imshow("", processed_image)
            cv2.waitKey(1)

        self.front_camera = processed_image

    def reward(self):
        reward = 0
        done = False

        velocity = self.vehicle.get_velocity()
        velocity_kmh = int(3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))

        distances = []
        walkers = self.world.get_actors().filter('walker.*')
        for walker in walkers:
            player_transform = walker.get_transform()
            ped_location = player_transform.location
            player_direction = walker.get_control().direction
            if ped_location.y < -214 and player_direction.y == -1:
                walker.destroy()
                continue
            elif ped_location.y > -191 and player_direction.y == 1:
                walker.destroy()
                continue
            dx = self.vehicle.get_location().x - ped_location.x
            dy = self.vehicle.get_location().y - ped_location.y
            distance = math.sqrt((dx * dx) + (dy * dy))
            distances.append(distance)

        min_dist = min(distances)

        if len(self.collision_history) != 0:
            reward = -5
            done = True
        elif min_dist < 4:
            reward = -2
            done = False
        elif velocity_kmh == 0:
            reward += -1
            done = False
        elif 15 < velocity_kmh < 25:
            reward += 1
            done = False
        elif 35 < velocity_kmh < 45:
            reward += 2
            done = False

        if self.vehicle.get_location().x > 155:
            done = True

        return reward, done

    # def reward(self):
#
    #     reward = 0
    #     done = False
#
    #     velocity = self.vehicle.get_velocity()
    #     velocity_kmh = int(3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))
#
    #     if velocity_kmh == 0:
    #         self.slow_counter += 1
    #     else:
    #         self.slow_counter = 0
#
    #     # if len(self.collision_history) != 0:
    #     #     reward += -200 * len(self.collision_history)
    #     #     self.collision_history = []
    #     #     done = True
    #     # elif velocity_kmh < 30:
    #     #     reward += -1
    #     #     done = False
    #     # else:
    #     #     reward += 1
    #     #     done = False
#
    #     if len(self.collision_history) != 0:
    #         reward += -300 * len(self.collision_history)
    #         self.collision_history = []
    #         done = True
    #     elif velocity_kmh == 0:
    #         reward += -1
    #         done = False
    #     elif 15 < velocity_kmh < 25:
    #         reward += 1
    #     elif 35 < velocity_kmh < 45:
    #         reward += 2
    #         done = False
#
    #     walkers = self.world.get_actors().filter('walker.*')
    #     for walker in walkers:
#
    #         player_transform = walker.get_transform()
    #         ped_location = player_transform.location
    #         player_direction = walker.get_control().direction
#
    #         if ped_location.y < -214 and player_direction.y == -1:
    #             walker.destroy()
    #             continue
    #         elif ped_location.y > -191 and player_direction.y == 1:
    #             walker.destroy()
    #             continue
#
    #         dx = self.vehicle.get_location().x - ped_location.x
    #         dy = self.vehicle.get_location().y - ped_location.y
    #         distance = math.sqrt((dx * dx) + (dy * dy))
#
    #         # If the distance is lower than 8, give a negative reward of -20
    #         if distance < 5:
    #             reward -= 2
#
    #         # If there is a collision (vehicle position matches pedestrian position), give a negative reward of -200
    #         if distance < 2:
    #             reward -= 25
#
    #     reward += (80 / (166 - self.vehicle.get_location().x))**2
#
    #     if self.vehicle.get_location().x > 155 or self.slow_counter == SLOW_COUNTER:  # or self.episode_start + SECONDS_PER_EPISODE < time.time():
    #         done = True
#
    #     return reward, done

    def step(self, action):
        if action == 0:
            # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.6, steer=0.0))
            velocity = carla.Vector3D(x=0.00, y=0.0, z=0.0)
            self.vehicle.set_target_velocity(velocity)
        elif action == 1:
            # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
            velocity = carla.Vector3D(x=6.5, y=0.0, z=0.0)
            self.vehicle.set_target_velocity(velocity)
        elif action == 2:
            # self.vehicle.apply_control(carla.VehicleControl(throttle=0.6, brake=0.0, steer=0.0))
            velocity = carla.Vector3D(x=12, y=0.0, z=0.0)
            self.vehicle.set_target_velocity(velocity)

        reward, done = self.reward()

        return self.front_camera, reward, done, None