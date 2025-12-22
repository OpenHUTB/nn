from __future__ import print_function

import glob
import os
import sys
import random

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# 获取当前文件的目录
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

import keras
import tensorflow as tf

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
from keras.applications.xception import Xception 
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard

from keras.models import Sequential, Model, load_model
from keras.layers import AveragePooling2D, Conv2D, Activation, Flatten, GlobalAveragePooling2D, Dense, Concatenate, Input

import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

from tqdm import tqdm

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
MODEL_NAME = "Braking"

MEMORY_FRACTION = 0.8
MIN_REWARD = 0

EPISODES = 20
DISCOUNT = 0.99
epsilon = 0.99
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.01

AGGREGATE_STATS_EVERY = 10

class CarEnv:

    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 0.88   # actions that the agent can take [-1, 0, 1] --> [turn left, go straight, turn right]
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self, start, end):
    # to initialize
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.find("vehicle.tesla.model3")
        self.front_model3 = self.blueprint_library.find("vehicle.tesla.model3")
        self.via = 2
        self.crossing = 0
        self.reached = 0
        self.final_destination  = end
        self.initial_pos = start
        self.distance = 0
        self.cam = None
        self.seg = None
        self.actor_list = []
    
    # 移除环境清理，保留周围环境
    # self.cleanup_environment()

    def cleanup_environment(self):
        """清理环境中的车辆和行人"""
        try:
            actors = self.world.get_actors()
            destroyed_count = 0
            
            for actor in actors:
                if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('walker.'):
                    try:
                        actor.destroy()
                        destroyed_count += 1
                    except:
                        pass
            
            if destroyed_count > 0:
                print(f"清理了 {destroyed_count} 个现有演员")
                
            time.sleep(2.0)
            
        except Exception as e:
            print(f"环境清理时出错: {e}")

    def cleanup(self):
        """清理所有演员"""
        print("清理环境中的演员...")
        
        try:
            # 首先销毁我们创建的演员
            for actor in self.actor_list:
                try:
                    if actor.is_alive:
                        actor.destroy()
                except Exception as e:
                    print(f"销毁演员失败: {e}")
            
            self.actor_list = []
            time.sleep(1.0)  # 等待销毁完成
            
        except Exception as e:
            print(f"清理时出错: {e}")

    def reset(self):
        # 只清理我们自己创建的演员，不清理环境
        self.cleanup_self_actors()
        
        # store any collision detected
        self.collision_history = []
        # to store all the actors that are present in the environment
        self.actor_list = []
        # store the number of times the vehicles crosses the lane marking
        self.lanecrossing_history = []
        
        # 先计算轨迹，以便获取正确的朝向
        traj = self.trajectory()
        self.path = []
        for el in traj:
            self.path.append(el[0])
        
        self.visualize_path(self.path)

        # 使用路径上的第一个点来确定正确的朝向
        if len(self.path) > 0:
            first_waypoint = self.path[0]
            correct_yaw = first_waypoint.transform.rotation.yaw
            # 确保朝向正确（与路径方向一致）
            print(f"路径朝向: {correct_yaw}°")
        else:
            correct_yaw = self.initial_pos[3]
        
        self.transform = Transform(
            Location(x=self.initial_pos[0], y=self.initial_pos[1], z=self.initial_pos[2]), 
            Rotation(yaw=correct_yaw)  # 使用路径的朝向
        )
        
        # 生成车辆
        self.vehicle = self.world.try_spawn_actor(self.model_3, self.transform)
        
        if self.vehicle is None:
            # 如果生成失败，尝试附近的位置
            for i in range(5):
                offset_x = random.uniform(-2.0, 2.0)
                offset_y = random.uniform(-2.0, 2.0)
                temp_transform = Transform(
                    Location(x=self.initial_pos[0] + offset_x, y=self.initial_pos[1] + offset_y, z=self.initial_pos[2]),
                    Rotation(yaw=correct_yaw)
                )
                self.vehicle = self.world.try_spawn_actor(self.model_3, temp_transform)
                if self.vehicle is not None:
                    break
        
        if self.vehicle is None:
            raise RuntimeError("无法生成车辆")
        
        self.actor_list.append(self.vehicle)
        
        print("Spawning my agent.....")
        print(f"车辆位置: ({self.initial_pos[0]:.2f}, {self.initial_pos[1]:.2f}, {self.initial_pos[2]:.2f})")
        print(f"车辆朝向: {-correct_yaw}°")

        # to use the RGB camera
        self.depth_camera = self.blueprint_library.find("sensor.camera.depth")
        self.depth_camera.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.depth_camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.depth_camera.set_attribute("fov", f"40")

        self.camera_spawn_point = carla.Transform(carla.Location(x=2, y = 0, z=1.4), Rotation(yaw=0))

        # to spawn the camera
        self.camera_sensor = self.world.spawn_actor(self.depth_camera, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.camera_sensor)

        # to record the data from the camera sensor
        self.camera_sensor.listen(lambda data: self.image_dep(data))

        '''
        To spawn the SEGMENTATION camera
        '''
        self.seg_camera = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        self.seg_camera.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.seg_camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.seg_camera.set_attribute("fov", f"40")

        # to spawn the segmentation camera exactly in between the 2 depth cameras
        self.seg_camera_spawn_point = carla.Transform(carla.Location(x=2, y = 0, z=1.4), Rotation(yaw=0))
        
        # to spawn the camera
        self.seg_camera_sensor = self.world.spawn_actor(self.seg_camera, self.seg_camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.seg_camera_sensor)

        self.seg_camera_sensor.listen(lambda data: self.image_seg(data))
        
        # 不要初始化车辆控制，保持静止
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)  # 等待传感器初始化

        # to introduce the collision sensor to detect what type of collision is happening
        col_sensor = self.blueprint_library.find("sensor.other.collision")
        
        # keeping the location of the sensor to be same as that of the RGB camera
        self.collision_sensor = self.world.spawn_actor(col_sensor, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.collision_sensor)

        # to record the data from the collision sensor
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        # to introduce the lanecrossing sensor to identify vehicles trajectory
        lane_crossing_sensor = self.blueprint_library.find("sensor.other.lane_invasion")

        # keeping the location of the sensor to be same as that of RGM Camera
        self.lanecrossing_sensor = self.world.spawn_actor(lane_crossing_sensor, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.lanecrossing_sensor)

        # to record the data from the lanecrossing_sensor
        self.lanecrossing_sensor.listen(lambda event: self.lanecrossing_data(event))

        # 等待传感器数据
        while self.cam is None or self.seg is None:
            time.sleep(0.01)
            
        self.process_images()

        # episode计时
        self.episode_start = time.time()

        # 返回初始状态，车辆保持静止
        return [(self.distance-300)/300, -1, 0, 0]

    def visualize_path(self, path_points, color=None):
        """可视化路径"""
        try:
            if color is None:
                color = carla.Color(255, 0, 0)  # 红色
            
            # 绘制路径点
            for i, waypoint in enumerate(path_points):
                location = waypoint.transform.location
                
                # 每10个点画一个标记
                if i % 10 == 0:
                    self.world.debug.draw_string(
                        carla.Location(location.x, location.y, location.z + 0.5),
                        '→',
                        draw_shadow=False,
                        color=carla.Color(0, 255, 0),
                        life_time=100.0,
                        persistent_lines=True
                    )
                
                # 画线连接路径点
                if i < len(path_points) - 1:
                    next_location = path_points[i + 1].transform.location
                    self.world.debug.draw_line(
                        carla.Location(location.x, location.y, location.z + 0.3),
                        carla.Location(next_location.x, next_location.y, next_location.z + 0.3),
                        thickness=0.1,
                        color=color,
                        life_time=100.0,
                        persistent_lines=True
                    )
            
            # 绘制起点和终点
            if path_points:
                start = path_points[0].transform.location
                end = path_points[-1].transform.location
                
                # 起点标记
                self.world.debug.draw_point(
                    carla.Location(start.x, start.y, start.z + 0.5),
                    size=0.2,
                    color=carla.Color(0, 255, 0),  # 绿色
                    life_time=100.0,
                    persistent_lines=True
                )
                
                # 终点标记
                self.world.debug.draw_point(
                    carla.Location(end.x, end.y, end.z + 0.5),
                    size=0.2,
                    color=carla.Color(255, 0, 0),  # 红色
                    life_time=100.0,
                    persistent_lines=True
                )
                
                # 添加文字标签
                self.world.debug.draw_string(
                    carla.Location(start.x, start.y, start.z + 2.0),
                    'START',
                    draw_shadow=True,
                    color=carla.Color(255, 255, 255),
                    life_time=100.0
                )
                
                self.world.debug.draw_string(
                    carla.Location(end.x, end.y, end.z + 2.0),
                    'GOAL',
                    draw_shadow=True,
                    color=carla.Color(255, 255, 255),
                    life_time=100.0
                )
                
            print(f"路径可视化完成，共绘制 {len(path_points)} 个点")
            
        except Exception as e:
            print(f"路径可视化失败: {e}")

    def collision_data(self, event):
        self.collision_history.append(event)

    
    # to record the lane crossing data
    def lanecrossing_data(self, event):
        self.lanecrossing_history.append(event)
        print("Lane crossing history: ", event)
        

    def image_dep(self, image):
        self.cam = image
        

    def image_seg(self, image):
        self.seg = image


    # to process the image
    def process_images(self):        
        # Convert depth image to array of depth values
        depth_array1 = np.frombuffer(self.cam.raw_data, dtype=np.dtype("uint8"))
        depth_array1 = np.reshape(depth_array1, (self.cam.height, self.cam.width, 4))
        depth_array1 = depth_array1.astype(np.int32)
        
        # Using this formula to get the distances
        depth_map = (depth_array1[:, :, 0]*255*255 + depth_array1[:, :, 1]*255 + depth_array1[:, :, 2])/1000
        
        # Making the sky at 0 distance
        x = np.where(depth_map >= 16646.655)
        depth_map[x] = 0

        # Showing the initial depth image
        #cv2.imshow("Initial: ", np.array(depth_array1, dtype = np.uint8))

        # Calculate distance from camera to each point in world coordinates
        distances = depth_map


        # uncomment the code below to get the distance map
        # # Plot the distance map
        #fig, ax = plt.subplots()
        #cmap = plt.cm.jet
        #cmap.set_bad(color='black')
        #im = ax.imshow(depth_array, cmap=cmap, vmin=0, vmax=50)#int(distances[int(cy),int(cx)]*2))
        #ax.set_title('Distance Map')
        #ax.set_xlabel('Pixel X')
        #ax.set_ylabel('Pixel Y')
        #cbar = ax.figure.colorbar(im, ax=ax)
        #cbar.ax.set_ylabel('Distance (m)', rotation=-90, va="bottom")
        #plt.savefig("pics/"+str(int(time.time()*100))+".jpg")
        
        image_array = np.frombuffer(self.seg.raw_data, dtype=np.dtype("uint8"))
        image_array = np.reshape(image_array, (self.seg.height, self.seg.width, 4))
        
        # removing the alpha channel
        image_array = image_array[:, :, :3]
        self.seg_array = image_array 
        
        colors = {
            0: [0, 0, 0],         # None
            1: [70, 70, 70],      # Buildings
            2: [190, 153, 153],   # Fences
            3: [72, 0, 90],       # Other
            4: [220, 20, 60],     # Pedestrians
            5: [153, 153, 153],   # Poles
            6: [157, 234, 50],    # RoadLines
            7: [128, 64, 128],    # Roads
            8: [244, 35, 232],    # Sidewalks
            9: [107, 142, 35],    # Vegetation
            10: [0, 0, 255],      # Vehicles
            11: [102, 102, 156],  # Walls
            12: [220, 220, 0],    # TrafficSigns
        }
        
        # to store the vehicle indices only
        lane = np.where((self.seg_array == [0, 0, 6]).all(axis = 2))
        
        copy_seg_img = np.copy(self.seg_array)
        

        if False:
            #sidewalk = np.where((copy_seg_img == [0, 0, 8]).all(axis = 2))
            #p2 = np.polyfit(sidewalk[0], sidewalk[1], 1)
            
            # Fit a polynomial of degree 2
            p = np.polyfit(lane[0], lane[1], 2)
            
            # Create a new set of x-values to plot the fitted curve
            x_fit = np.linspace(0, 479, 480)
            
            # Evaluate the fitted polynomial at the new x-values
            y_fit = np.polyval(p, x_fit)
            
            # Evaluate the fitted polynomial at the new x-values
            #y_fit2 = np.polyval(p2, x_fit)
            
            # Plot the data and the fitted curve
            #plt.plot(x, y, 'o', label='data')
            #plt.plot(x_fit, y_fit, '-', label='fit')
            #plt.legend()
            #plt.show()
            
            for i in range(480):
                for j in range(640):
                    if j < y_fit[i]:
                        copy_seg_img[i,j] = [0,0,0]
                        distances[i,j] = 0

        
        # to store the vehicle indices only
        self.vehicle_indices = np.where((copy_seg_img == [0, 0, 10]).all(axis = 2))
        
        # to store the pedestrian indices only
        self.pedestrian_indices = np.where((copy_seg_img == [0, 0, 4]).all(axis = 2))

        if len(self.vehicle_indices[0]) != 0:
            dis = np.sum(distances[self.vehicle_indices])/len(self.vehicle_indices[0])
        else:
            dis = 10000
            
        if len(self.pedestrian_indices[0]) != 0:
            dis_ped = np.sum(distances[self.pedestrian_indices])/len(self.pedestrian_indices[0])
        else:
            dis_ped = 10000
        
        copy_seg_img2 = np.copy(copy_seg_img)
        for key in colors:
            copy_seg_img2[np.where((copy_seg_img2 == [0, 0, key]).all(axis = 2))] = colors[key]

        # to save the image
        cv2.imwrite("pics/seg/seg_"+str(int(time.time()*100))+".jpg", copy_seg_img2)    

            
        self.distance = min(dis, dis_ped)

        return dis
        

    def step(self, action, current_state):
        '''
        To take 6 actions; brake, go straight, turn left, turn right, turn slightly left, turn slightly right
        '''
        # 根据当前速度和横向偏差动态调整转向灵敏度
        v = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        # 动态转向系数：速度越高，转向越小
        if speed > 30:  # 高速时
            steer_factor = 0.5
        elif speed > 15:  # 中速时
            steer_factor = 0.7
        else:  # 低速时
            steer_factor = 1.0
        
        # 应用动作控制 - 减小等待时间
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0))
            print("执行动作: 刹车")
        elif action == 1:
            # 直行时稍微增加油门，避免停车
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0))
            print("执行动作: 直行")
        elif action == 2:
            # 根据速度动态调整转向幅度
            steer_value = -0.25 * steer_factor
            throttle_value = 0.3 if speed < 10 else 0.1  # 低速时给更多油门
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_value, steer=steer_value))
            print("执行动作: 左转")
        elif action == 3:
            steer_value = 0.25 * steer_factor
            throttle_value = 0.3 if speed < 10 else 0.1
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_value, steer=steer_value))
            print("执行动作: 右转")
        elif action == 4:
            steer_value = -0.15 * steer_factor
            throttle_value = 0.4
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_value, steer=steer_value))
            print("执行动作: 微左")
        elif action == 5:
            steer_value = 0.15 * steer_factor
            throttle_value = 0.4
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_value, steer=steer_value))
            print("执行动作: 微右")
            
        time.sleep(0.01)

        # 处理图像
        self.process_images()
        
        # initialize a reward for a single action 
        reward = 0

        # to calculate the kmh of the vehicle
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        # to get the position and orientation of the car
        pos = self.vehicle.get_transform().location
        rot = self.vehicle.get_transform().rotation
        
        # 获取最近的路径点
        waypoint_ind = self.get_closest_waypoint(self.path, self.vehicle.get_transform())
        current_waypoint = self.path[waypoint_ind]
        
        # 计算到终点的距离
        dist_from_goal = np.sqrt((pos.x - self.final_destination[0])**2 + (pos.y-self.final_destination[1])**2)
        
        done = False

        # 计算角度差
        waypoint_rot = current_waypoint.transform.rotation
        orientation_diff = waypoint_rot.yaw - rot.yaw
        phi = ((orientation_diff + 180) % 360) - 180
        
        # 计算横向偏差
        if waypoint_ind < len(self.path) - 1:
            next_waypoint = self.path[waypoint_ind + 1]
            waypoint_loc = current_waypoint.transform.location
            next_waypoint_loc = next_waypoint.transform.location
            vehicle_loc = pos
            
            # 计算路径向量
            u = np.array([next_waypoint_loc.x - waypoint_loc.x, 
                         next_waypoint_loc.y - waypoint_loc.y])
            v = np.array([pos.x - waypoint_loc.x, 
                         pos.y - waypoint_loc.y])
                       
            if np.linalg.norm(u) > 0.1:
                # 计算投影长度
                t = np.dot(u, v) / np.dot(u, u)
                t = max(0, min(1, t))  # 限制在0-1之间
                
                # 计算最近点
                closest_point = np.array([waypoint_loc.x, waypoint_loc.y]) + t * u
                
                # 计算横向偏差
                lateral_vec = v - t * u
                signed_dis = np.linalg.norm(lateral_vec)
                
                # 判断偏差方向
                cross_product = np.cross(u, v)
                if cross_product > 0:
                    signed_dis = -signed_dis  # 车辆在路径左侧
            else:
                signed_dis = 0
        else:
            signed_dis = 0
        
        current_state[0] = (self.distance-300)/300
        current_state[1] = (kmh-30)/30 - (self.distance-300)/300
        current_state[2] = phi
        current_state[3] = signed_dis * 15
        
        print(f"角度差: {phi:.2f}°, 横向偏差: {signed_dis:.2f}m")
        print(f"距离障碍物: {self.distance:.2f}m")
        print(f"速度: {kmh:.2f}km/h")
        print(f"距离终点: {dist_from_goal:.2f}m")

        # 奖励计算 - 更精细的奖励函数
        base_reward = 0.1  # 基础生存奖励
        
        # 速度奖励（鼓励合理速度20-40 km/h）
        if 20 <= kmh <= 40:
            speed_reward = 0.2
        elif kmh < 5:
            speed_reward = -0.1  # 太慢惩罚
        else:
            speed_reward = 0.1 * (kmh / 40.0)
        
        # 路径跟随奖励
        heading_penalty = abs(phi) / 180.0  # 角度偏差惩罚
        lateral_penalty = min(abs(signed_dis) / 3.0, 1.0)  # 横向偏差惩罚
        
        path_follow_reward = 0.3 * (1.0 - heading_penalty) + 0.3 * (1.0 - lateral_penalty)
        
        # 进度奖励
        if not hasattr(self, 'initial_distance_to_goal'):
            self.initial_distance_to_goal = dist_from_goal
        
        if self.initial_distance_to_goal > 0:
            progress = 1.0 - (dist_from_goal / self.initial_distance_to_goal)
            progress_reward = 0.2 * max(0, progress)
        else:
            progress_reward = 0
        
        # 距离障碍物奖励
        obstacle_reward = 0.2 * max(0, 1.0 - self.distance / 50.0) if self.distance < 50 else 0
        
        reward = base_reward + speed_reward + path_follow_reward + progress_reward + obstacle_reward

        # 检查是否完成
        if len(self.collision_history) != 0:
            done = True
            reward = -200
            print("❌ 发生碰撞!")

        if abs(phi) > 100:
            done = True
            reward = -200
            print("❌ 方向偏差过大!")
            
        if abs(signed_dis) > 2:
            reward = -50
            print("⚠️ 横向偏差过大")
        
        # 到达终点
        if dist_from_goal < 10:
            self.reached = 1
            done = True
            reward = 1000
            print("✅ 成功到达终点!")


        print(f"奖励: {reward}")

        self.world.debug.draw_string(
            carla.Location(pos.x, pos.y, pos.z + 2.0),
            f'V: {kmh}km/h',
            draw_shadow=True,
            color=carla.Color(255, 255, 0),
            life_time=0.1
        )
        
        # 返回新的状态、奖励、完成标志和当前路径点
        return current_state, reward, done, current_waypoint
    def trajectory(self, draw = False):

        amap = self.world.get_map()
        sampling_resolution = 0.5
        # dao = GlobalRoutePlannerDAO(amap, sampling_resolution)
        grp = GlobalRoutePlanner(amap, sampling_resolution)
        # grp.setup()
        
        #start_location = self.vehicle.get_transform().location
        start_location = carla.Location(x=self.initial_pos[0], y=self.initial_pos[1], z=0)
        end_location = carla.Location(x=self.final_destination[0], y=self.final_destination[1], z=0)
        a = amap.get_waypoint(start_location, project_to_road=True)
        b = amap.get_waypoint(end_location, project_to_road=True)
        spawn_points = self.world.get_map().get_spawn_points()
        #print(spawn_points)
        a = a.transform.location
        b = b.transform.location
        w1 = grp.trace_route(a, b) # there are other funcations can be used to generate a route in GlobalRoutePlanner.
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
    def cleanup_self_actors(self):
        """只清理我们自己创建的演员，不清理环境中的其他车辆和行人"""
        print("清理自己创建的演员...")
        
        try:
            # 只清理我们自己的actor_list中的演员
            for actor in self.actor_list:
                try:
                    if actor.is_alive:
                        actor.destroy()
                except Exception as e:
                    print(f"销毁演员失败: {e}")
            
            self.actor_list = []
            time.sleep(1.0)  # 等待销毁完成
            
        except Exception as e:
            print(f"清理时出错: {e}")


    def get_closest_waypoint(self, waypoint_list, vehicle_transform):
        """获取最近的路径点，优化搜索算法"""
        if not waypoint_list:
            return 0
        
        # 使用更高效的距离计算方法
        vehicle_location = vehicle_transform.location
        closest_waypoint = 0
        closest_distance_sq = float('inf')
        
        # 如果已经保存了上一次的最近点，从它附近开始搜索
        if hasattr(self, 'last_waypoint_index') and self.last_waypoint_index < len(waypoint_list):
            start_index = max(0, self.last_waypoint_index - 10)
            end_index = min(len(waypoint_list), self.last_waypoint_index + 20)
        else:
            start_index = 0
            end_index = min(50, len(waypoint_list))  # 限制搜索范围，提高速度
        
        for i in range(start_index, end_index):
            waypoint_location = waypoint_list[i].transform.location
            dx = waypoint_location.x - vehicle_location.x
            dy = waypoint_location.y - vehicle_location.y
            distance_sq = dx*dx + dy*dy  # 使用平方距离，避免开方运算
            
            if distance_sq < closest_distance_sq:
                closest_waypoint = i
                closest_distance_sq = distance_sq
        
        # 保存最近的路径点索引，供下次使用
        self.last_waypoint_index = closest_waypoint
        
        # 向前看几个点，避免卡在同一个点
        look_ahead = 5
        if closest_waypoint < len(waypoint_list) - look_ahead:
            return closest_waypoint + look_ahead
        else:
            return len(waypoint_list) - 1