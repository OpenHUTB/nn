# Environment.py
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

import carla
from carla import ColorConverter


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    im_width = IM_WIDTH
    im_height = IM_HEIGHT

    def __init__(self):
        self.actor_list = None
        self.sem_cam = None
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.front_camera = None

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

    def spawn_pedestrians_left(self, isCross):
        blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        walker_bp = random.choice(blueprints_walkers)

        for i in range(1):
            walker_bp = random.choice(blueprints_walkers)

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

    def reset(self):
        walkers = self.world.get_actors().filter('walker.*')
        for walker in walkers:
            walker.destroy()

        vehicles = self.world.get_actors().filter('vehicle.*')
        for v in vehicles:
            v.destroy()

        # 课程学习 - 根据训练阶段调整难度
        self.spawn_pedestrians_general(30, True)
        self.spawn_pedestrians_general(10, False)

        self.collision_history = []
        self.actor_list = []
        self.slow_counter = 0

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        spawn_point.location.x = -81.0
        spawn_point.location.y = -195.0
        spawn_point.location.z = 2.0
        spawn_point.rotation.roll = 0.0
        spawn_point.rotation.pitch = 0.0
        spawn_point.rotation.yaw = 0.0
        
        self.vehicle = self.world.spawn_actor(self.model_3, spawn_point)
        self.actor_list.append(self.vehicle)

        self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.sem_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
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
        
        # 获取车辆位置和前进方向
        vehicle_location = self.vehicle.get_location()
        vehicle_rotation = self.vehicle.get_transform().rotation.yaw
        
        # 计算距离终点的进度奖励
        progress_reward = (vehicle_location.x + 81) / 236.0  # 从-81到155，总共236单位
        
        # 速度奖励 - 更加平滑
        if velocity_kmh == 0:
            reward -= 0.5  # 停车惩罚减少
        elif 20 <= velocity_kmh <= 40:  # 理想速度区间
            reward += 0.8
        elif 10 <= velocity_kmh < 20 or 40 < velocity_kmh <= 50:
            reward += 0.3  # 可接受速度区间
        else:
            reward -= 0.2  # 不理想速度
            
        # 方向奖励 - 确保车辆朝正确方向行驶
        if -45 <= vehicle_rotation <= 45:  # 大致朝东方向
            reward += 0.2
        else:
            reward -= 0.5
            
        # 行人距离检测
        min_dist = float('inf')
        walkers = self.world.get_actors().filter('walker.*')
        for walker in walkers:
            ped_location = walker.get_location()
            dx = vehicle_location.x - ped_location.x
            dy = vehicle_location.y - ped_location.y
            distance = math.sqrt(dx**2 + dy**2)
            min_dist = min(min_dist, distance)
            
            # 清理边界外的行人
            player_direction = walker.get_control().direction
            if (ped_location.y < -214 and player_direction.y == -1) or \
               (ped_location.y > -191 and player_direction.y == 1):
                walker.destroy()

        # 基于行人距离的奖励
        if min_dist < 3.0:  # 非常危险
            reward -= 3.0
            done = True
        elif min_dist < 5.0:  # 危险
            reward -= 1.0
        elif min_dist < 8.0:  # 警告
            reward -= 0.3
        elif min_dist > 15.0:  # 安全
            reward += 0.2
            
        # 碰撞检测
        if len(self.collision_history) != 0:
            reward = -10  # 增加碰撞惩罚
            done = True
            
        # 进度奖励
        reward += progress_reward * 0.5
        
        # 完成条件
        if vehicle_location.x > 155:
            reward += 10  # 成功到达奖励
            done = True
        elif vehicle_location.x < -90:  # 倒退太多
            reward -= 5
            done = True
            
        return reward, done

    def step(self, action):
        # 更平滑的控制
        if action == 0:  # 减速
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.3))
        elif action == 1:  # 保持/轻微加速
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, brake=0.0))
        elif action == 2:  # 加速
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, brake=0.0))

        # 等待物理更新
        time.sleep(0.05)
        
        reward, done = self.reward()
        
        # 限制极端奖励值
        reward = np.clip(reward, -10, 10)
        
        return self.front_camera, reward, done, None