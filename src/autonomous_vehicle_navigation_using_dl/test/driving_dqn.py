from __future__ import print_function

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录
parent_dir = os.path.dirname(current_dir)
# 将父目录添加到系统路径
sys.path.append(parent_dir)
from agents.navigation.global_route_planner import GlobalRoutePlanner

import carla
from carla import ColorConverter as cc
from carla import Transform 
from carla import Location
from carla import Rotation

from PIL import Image

import tensorflow as tf
from tensorflow import keras

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import time
import numpy as np
import cv2
from collections import deque
from tensorflow.keras.applications.xception import Xception 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Activation, Flatten, GlobalAveragePooling2D, Dense, Concatenate, Input

from threading import Thread
from tensorflow.keras import regularizers

from tqdm import tqdm

# 设置 TensorFlow 日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 20
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 2
MODEL_NAME = "Driving"

MEMORY_FRACTION = 0.8
MIN_REWARD = 0

EPISODES = 40
DISCOUNT = 0.99
epsilon = 0.5
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.01

AGGREGATE_STATS_EVERY = 1

# 路径配置 - 现在由外部传入
default_town2 = {1: [193.75,269.2610168457031, 5, 270], 2:[135.25,206]} #left turn
curves = [0, default_town2]

'''
Custom Tensorboard class. Updates logs after every episode
'''
class ModifiedTensorBoard(TensorBoard):
    def __init__(self, model, log_dir):
        super().__init__(log_dir)
        self.log_dir = log_dir
        self.model = model
        self.step = 1
        print("SELF: LOG DIR:      ", self.log_dir)
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
            self.writer.flush()

'''
Defining the Carla Environment Class. 
'''
class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self, start_point=None, end_point=None):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.find("vehicle.tesla.model3")
        self.via = 2
        self.crossing = 0
        self.curves = 1
        self.reached = 0
        
        # 允许从外部传入起点终点
        self.custom_start = start_point
        self.custom_end = end_point
        
        # 如果提供了自定义起点，使用它
        if start_point is not None:
            self.start = start_point
        else:
            self.start = default_town2[1]
            
        # 存储自定义终点
        self.end_point = end_point if end_point is not None else default_town2[2]
        
        self.phi = []
        self.dc = []
        self.vel = []
        self.time = []
        self.last_position = None
        self.current_waypoint_index = 0
        self.path_progress = 0  # 路径进度跟踪

    def reset(self):
        # store any collision detected
        self.collision_history = []
        # to store all the actors that are present in the environment
        self.actor_list = []
        # store the number of times the vehicles crosses the lane marking
        self.lanecrossing_history = []
        
        # 重置路径跟踪
        self.current_waypoint_index = 0
        self.path_progress = 0
        self.last_position = None
        
        '''
        To spawn the Vehicle (agent)
        '''
        initial_pos = self.start
        print(f"生成车辆在起点: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f}), 航向: {initial_pos[3]}°")
        
        self.transform = Transform(Location(x=initial_pos[0], y=initial_pos[1], z=initial_pos[2]), Rotation(yaw=initial_pos[3]))
        
        # 尝试生成车辆
        try:
            self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            self.actor_list.append(self.vehicle)
            print("✓ 车辆生成成功")
        except Exception as e:
            print(f"❌ 车辆生成失败: {e}")
            # 尝试使用默认位置
            default_transform = Transform(Location(x=0, y=0, z=5), Rotation(yaw=0))
            self.vehicle = self.world.spawn_actor(self.model_3, default_transform)
            self.actor_list.append(self.vehicle)

        # 相机传感器设置
        self.camera_spawn_point = carla.Transform(carla.Location(x=2, y=0, z=1.4))

        # to initialize the car quickly and get it going
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        time.sleep(1)  # 减少等待时间

        '''
        To spawn the collision sensor
        '''
        col_sensor = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_sensor, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        # to introduce the lanecrossing sensor
        lane_crossing_sensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lanecrossing_sensor = self.world.spawn_actor(lane_crossing_sensor, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.lanecrossing_sensor)
        self.lanecrossing_sensor.listen(lambda event: self.lanecrossing_data(event))
        
        # 生成轨迹
        traj = self.trajectory()
        self.path = []
        for el in traj:
            self.path.append(el[0])
        
        print(f"生成的路径点数量: {len(self.path)}")
        if len(self.path) > 0:
            print(f"第一个路径点方向: {self.path[0].transform.rotation.yaw:.1f}°")
            print(f"最后一个路径点方向: {self.path[-1].transform.rotation.yaw:.1f}°")
        
        # 获取车辆初始方向
        vehicle_transform = self.vehicle.get_transform()
        print(f"车辆初始方向: {vehicle_transform.rotation.yaw:.1f}°")
        
        # 检查车辆与第一个路径点的方向差
        if len(self.path) > 0:
            first_wp_direction = self.path[0].transform.rotation.yaw
            direction_diff = first_wp_direction - vehicle_transform.rotation.yaw
            phi_initial = direction_diff % 360 - 360 * (direction_diff % 360 > 180)
            print(f"初始方向差: {phi_initial:.1f}°")

        self.episode_start = time.time()
        
        # 设置初始控制，让车辆开始移动
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.5, brake = 0.0, steer = 0.0))
        time.sleep(0.5)  # 给车辆一点时间开始移动
        
        # 获取初始状态
        pos = self.vehicle.get_transform().location
        rot = self.vehicle.get_transform().rotation
        
        # 获取最近的路径点
        waypoint = self.client.get_world().get_map().get_waypoint(pos, project_to_road=True)
        closest_index = self.get_closest_waypoint(self.path, waypoint)
        self.current_waypoint_index = max(0, min(closest_index, len(self.path)-1))
        
        if len(self.path) > self.current_waypoint_index:
            wp = self.path[self.current_waypoint_index]
            wp_rot = wp.transform.rotation
            direction_diff = wp_rot.yaw - rot.yaw
            phi = direction_diff % 360 - 360 * (direction_diff % 360 > 180)
        else:
            phi = 0
        
        return [phi, 0]  # 初始状态

    def collision_data(self, event):
        self.collision_history.append(event)
    
    def lanecrossing_data(self, event):
        self.lanecrossing_history.append(event)

    def step(self, action, current_state):
        '''
        Take 5 actions; go straight, turn left, turn right, turn slightly left, turn slightly right
        '''
        # 增加油门值，让车辆更快移动
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=0*self.STEER_AMT))
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=-0.6*self.STEER_AMT))
        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.6*self.STEER_AMT))
        if action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=-0.1*self.STEER_AMT))
        if action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=0.1*self.STEER_AMT))

        # 检查移动距离
        current_pos = self.vehicle.get_transform().location
        if self.last_position:
            distance_moved = math.sqrt(
                (current_pos.x - self.last_position.x)**2 +
                (current_pos.y - self.last_position.y)**2
            )
            if distance_moved < 0.1 and hasattr(self, 'step_count') and self.step_count > 20:
                print(f"⚠️ 警告: 车辆移动距离太小! ({distance_moved:.2f}米)")
        self.last_position = current_pos

        # initialize a reward for a single action 
        reward = 0
        # to calculate the kmh of the vehicle
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        # to get the position and orientation of the car
        pos = self.vehicle.get_transform().location
        rot = self.vehicle.get_transform().rotation
        
        print(f"车辆位置: ({pos.x:.1f}, {pos.y:.1f}), 方向: {rot.yaw:.1f}°, 速度: {kmh} km/h")
        
        # to get the closest waypoint to the car
        waypoint = self.client.get_world().get_map().get_waypoint(pos, project_to_road=True)
        
        # 安全地获取路径点索引
        if not hasattr(self, 'path') or len(self.path) == 0:
            self.trajectory()
        
        # 使用改进的最近路径点查找
        closest_index = self.get_closest_waypoint(self.path, waypoint)
        
        # 确保索引在有效范围内
        closest_index = min(closest_index, len(self.path) - 1)
        closest_index = max(closest_index, 0)
        
        # 更新当前路径点索引（允许前进，限制回退）
        if closest_index > self.current_waypoint_index:
            self.current_waypoint_index = closest_index
        elif closest_index < self.current_waypoint_index - 2:  # 允许少量回退
            self.current_waypoint_index = max(0, closest_index)
        
        # 获取当前路径点
        if len(self.path) > self.current_waypoint_index:
            current_waypoint = self.path[self.current_waypoint_index]
        else:
            current_waypoint = self.path[-1] if len(self.path) > 0 else waypoint
        
        # 计算到当前路径点的距离
        wp_location = current_waypoint.transform.location
        distance_to_wp = math.sqrt((pos.x - wp_location.x)**2 + (pos.y - wp_location.y)**2)
        
        print(f"当前路径点: {self.current_waypoint_index}/{len(self.path)-1}")
        print(f"路径点位置: ({wp_location.x:.1f}, {wp_location.y:.1f})")
        print(f"到路径点距离: {distance_to_wp:.2f}")
        
        # 如果接近当前路径点且不是最后一个，前进到下一个
        if distance_to_wp < 5.0 and self.current_waypoint_index < len(self.path) - 1:
            next_index = self.current_waypoint_index + 1
            next_waypoint = self.path[next_index]
        else:
            next_index = self.current_waypoint_index
            next_waypoint = current_waypoint
        
        waypoint = current_waypoint
        next_waypoint_loc = next_waypoint.transform.location
        next_waypoint_rot = next_waypoint.transform.rotation
        
        waypoint_loc = waypoint.transform.location
        waypoint_rot = waypoint.transform.rotation
        
        # 计算到终点的距离
        final_destination = self.end_point
        dist_from_goal = np.sqrt((pos.x - final_destination[0])**2 + (pos.y - final_destination[1])**2)

        done = False

        '''
        TO DEFINE THE REWARDS
        '''
        # to get the orientation difference between the car and the road "phi"
        orientation_diff = waypoint_rot.yaw - rot.yaw
        phi = orientation_diff % 360 - 360 * (orientation_diff % 360 > 180)
        
        print(f"车辆方向: {rot.yaw:.1f}°, 路径点方向: {waypoint_rot.yaw:.1f}°")
        print(f"方向差: {orientation_diff:.1f}°, phi: {phi:.1f}°")
        
        current_state[1] = current_state[1] / 15
        
        # 计算横向偏移
        u = [waypoint_loc.x - next_waypoint_loc.x, waypoint_loc.y - next_waypoint_loc.y]
        v = [pos.x - next_waypoint_loc.x, pos.y - next_waypoint_loc.y]
        
        if np.linalg.norm(u) > 0.1 and np.linalg.norm(v) > 0.1:
            cross_product = u[0]*v[1] - u[1]*v[0]
            dot_product = u[0]*v[0] + u[1]*v[1]
            angle = np.arctan2(cross_product, dot_product)
            signed_dis = np.linalg.norm(v) * np.sin(angle)
        else:
            signed_dis = 0

        print(f"当前状态: phi={current_state[0]:.1f}, d={current_state[1]:.1f}")

        # Defining the Reward function by comparing the action taken to a suboptimal policy
        if abs(current_state[0]) < 5:
            if action == 0:
                reward += 2
            else:
                reward -= 1
        elif abs(current_state[0]) < 10:
            if current_state[0] < 0:
                if action == 3:
                    reward += 2
                elif action == 1:
                    reward += 1
                else:
                    reward -= 1
            else:
                if action == 4:
                    reward += 2
                elif action == 2:
                    reward += 1
                else:
                    reward -= 1
        else:
            if current_state[0] < 0:
                if action == 1:
                    reward += 2
                elif action == 3:
                    reward += 1
                else:
                    reward -= 1
            else:
                if action == 2:
                    reward += 2
                elif action == 4:
                    reward += 1
                else:
                    reward -= 1
                
        if abs(current_state[1]) < 0.1:
            if action == 0:
                reward += 4
            else:
                reward -= 2
        elif abs(current_state[1]) < 0.5:
            if current_state[1] < 0:
                if action == 3:
                    reward += 2
                elif action == 1:
                    reward += 1
                else:
                    reward -= 1
            else:
                if action == 4:
                    reward += 2
                elif action == 2:
                    reward += 1
                else:
                    reward -= 1
        else:
            if current_state[1] < 0:
                if action == 1:
                    reward += 2
                elif action == 3:
                    reward += 1
                else:
                    reward -= 1
            else:
                if action == 2:
                    reward += 2
                elif action == 4:
                    reward += 1
                else:
                    reward -= 1
                
        # 添加速度奖励
        if kmh > 20:
            reward += 1
        elif kmh < 5:
            reward -= 1
            
        if abs(signed_dis) > 2:
            reward -= 10
        
        # 检查距离路径点是否过远
        if distance_to_wp > 30:
            print(f"⚠️ 警告: 车辆距离路径点过远 ({distance_to_wp:.1f} > 30)")
            done = True
            reward = -100
            
        # to avoid collisions
        if len(self.collision_history) != 0:
            done = True
            reward = -200
            print("❌ 发生碰撞!")

        # to end the episode if phi value goes high
        if abs(phi) > 100:
            done = True
            reward = -200
            print("❌ 方向偏差过大!")
            
        # Ending the episode if the distance to the centerline of the road is greater than 3
        if abs(signed_dis) > 3:
            done = True
            reward = -200
            print("❌ 偏离道路中心线过远!")
            
        # to end the episode if the car reaches close to the final destination
        if dist_from_goal < 5:
            self.reached = 1
            done = True
            reward += 100
            print("✅ 成功到达目的地!")

        # to run each episode for just 30 seconds
        if self.episode_start + 200 < time.time():
            done = True
            print("⏰ 时间到!")

        print(f"奖励: {reward}")
        
        self.phi.append(phi)
        self.dc.append(signed_dis)
        self.vel.append(kmh)
        self.time.append(time.time())

        return [phi, signed_dis*15], reward, done, waypoint

    def trajectory(self, draw=False):
        amap = self.world.get_map()
        sampling_resolution = 0.5
        grp = GlobalRoutePlanner(amap, sampling_resolution)
        
        # 使用自定义终点或默认终点
        if hasattr(self, 'end_point') and self.end_point is not None:
            end_x, end_y = self.end_point
        else:
            end_x, end_y = default_town2[2][0], default_town2[2][1]
            
        start_location = carla.Location(x=self.start[0], y=self.start[1], z=0)
        end_location = carla.Location(x=end_x, y=end_y, z=0)
        
        print(f"生成轨迹: 从 ({self.start[0]:.1f}, {self.start[1]:.1f}) 到 ({end_x:.1f}, {end_y:.1f})")
        
        a = amap.get_waypoint(start_location, project_to_road=True)
        b = amap.get_waypoint(end_location, project_to_road=True)
        
        if a is None:
            print("❌ 无法找到起点对应的道路点!")
            # 尝试使用车辆当前位置
            if hasattr(self, 'vehicle'):
                vehicle_loc = self.vehicle.get_transform().location
                a = amap.get_waypoint(vehicle_loc, project_to_road=True)
        
        if b is None:
            print("❌ 无法找到终点对应的道路点!")
            return []
        
        a_loc = a.transform.location
        b_loc = b.transform.location
        
        w1 = grp.trace_route(a_loc, b_loc)
        
        print(f"生成的路径段数: {len(w1)}")
        
        i = 0
        if draw:
            for w in w1:
                if i % 10 == 0:
                    self.world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                    color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                    persistent_lines=True)
                else:
                    self.world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                    color = carla.Color(r=0, g=0, b=255), life_time=1000.0,
                    persistent_lines=True)
                i += 1
        return w1

    def get_closest_waypoint(self, waypoint_list, target_waypoint):
        if not waypoint_list:
            return 0
            
        closest_waypoint = self.current_waypoint_index
        closest_distance = float('inf')
        
        # 车辆位置
        vehicle_location = target_waypoint.transform.location
        
        # 从当前索引开始搜索，但允许查看前面的几个点
        start_index = max(0, self.current_waypoint_index - 3)
        
        for i in range(start_index, len(waypoint_list)):
            waypoint = waypoint_list[i]
            waypoint_location = waypoint.transform.location
            distance = math.sqrt(
                (waypoint_location.x - vehicle_location.x)**2 +
                (waypoint_location.y - vehicle_location.y)**2
            )
            
            if distance < closest_distance:
                closest_waypoint = i
                closest_distance = distance
        
        return closest_waypoint

'''
TO define the Deep Q Network agent class
'''
class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(self.model, log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        model3 = Sequential()
        model3.add(Dense(8, input_shape=(2,), activation='relu', name='dense1'))
        model3.add(Dense(5, activation='linear', name='output'))
        combined_model = Model(inputs=model3.input, outputs=model3.output)
        
        combined_model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        
        return combined_model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_data = np.array([[transition[0][i] for i in range(2)] for transition in minibatch])
        current_qs_list = self.model.predict(current_data, PREDICTION_BATCH_SIZE, verbose=0)

        new_current_data = np.array([[transition[3][i] for i in range(2)] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_data, PREDICTION_BATCH_SIZE, verbose=0)

        X_data = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X_data.append([current_state[i] for i in range(2)])
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        self.model.fit(np.array(X_data), np.array(y), batch_size=TRAINING_BATCH_SIZE, 
                      verbose=0, shuffle=False, 
                      callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *np.array(state).shape), verbose=0)[0]

    def train_in_loop(self):
        X2 = np.random.uniform(size=(1, 2)).astype(np.float32)
        y = np.random.uniform(size=(1, 5)).astype(np.float32)
        self.model.fit(X2, y, verbose=0, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)