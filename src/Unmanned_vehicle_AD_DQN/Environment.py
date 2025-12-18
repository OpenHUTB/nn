# Environment.py
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque

import carla
from carla import ColorConverter

# 导入超参数
try:
    from Hyperparameters import *
except ImportError:
    SHOW_PREVIEW = False
    IM_WIDTH = 160
    IM_HEIGHT = 120


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    im_width = IM_WIDTH
    im_height = IM_HEIGHT

    def __init__(self):
        self.actor_list = []
        self.sem_cam = None
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.front_camera = None
        
        # 道路参数
        self.road_center_y = -195
        self.road_left = -216  # 道路左侧边界
        self.road_right = -183  # 道路右侧边界
        
        # 原有变量
        self.last_action = 1
        self.same_steer_counter = 0
        self.suggested_action = None
        self.episode_start_time = None
        self.last_ped_distance = float('inf')
        self.current_episode = 1
        self.obstacle_detected_time = None
        self.reaction_start_time = None
        self.proactive_action_count = 0
        self.frame_buffer = deque(maxlen=3)
        self.motion_detected = False

        # 加载世界
        try:
            self.world = self.client.load_world('Town03')
        except:
            self.world = self.client.get_world()
        
        # 设置观察者视角
        self.setup_observer_view()
        
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        
        # 行人列表和碰撞历史
        self.walker_list = []
        self.collision_history = []
        self.slow_counter = 0
        self.steer_counter = 0

    def setup_observer_view(self):
        """设置观察者视角"""
        try:
            spectator = self.world.get_spectator()
            transform = carla.Transform()
            transform.location.x = -81.0
            transform.location.y = -195.0
            transform.location.z = 15.0
            transform.rotation.pitch = -45.0
            transform.rotation.yaw = 0.0
            transform.rotation.roll = 0.0
            
            spectator.set_transform(transform)
            print("观察者视角已设置")
        except Exception as e:
            print(f"设置观察者视角时出错: {e}")

    def check_road_boundary(self, location):
        """检查是否在道路边界内 - 简化版"""
        # 检查是否在道路范围内
        if self.road_left <= location.y <= self.road_right:
            # 在道路内，计算到道路中心的距离
            distance_to_center = abs(location.y - self.road_center_y)
            return distance_to_center, False  # 在道路内
        else:
            # 超出道路边界
            return float('inf'), True

    def spawn_pedestrians_general(self, number, isCross):
        """生成指定数量的行人"""
        target_number = number
        success_count = 0
        attempts = 0
        max_attempts = number * 3
        
        while success_count < target_number and attempts < max_attempts:
            attempts += 1
            isLeft = random.choice([True, False])
            
            try:
                if isLeft:
                    if self.spawn_pedestrians_left(isCross):
                        success_count += 1
                else:
                    if self.spawn_pedestrians_right(isCross):
                        success_count += 1
            except Exception as e:
                continue
        
        print(f"成功生成 {success_count}/{target_number} 个行人 (isCross={isCross})")
        return success_count

    def spawn_pedestrians_right(self, isCross):
        """在右侧生成行人"""
        blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        
        # 设置生成区域 - 在道路右侧的人行道上
        min_x = -50
        max_x = 140
        min_y = self.road_right + 2  # 道路右侧外2米
        max_y = self.road_right + 5

        if isCross:
            isFirstCross = random.choice([True, False])
            if isFirstCross:
                min_x = -14
                max_x = -10.5
            else:
                min_x = 17
                max_x = 20.5

        for attempt in range(3):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)

            spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            try:
                walker_bp = random.choice(blueprints_walkers)
                npc = self.world.try_spawn_actor(walker_bp, spawn_point)

                if npc is not None:
                    ped_control = carla.WalkerControl()
                    ped_control.speed = random.uniform(0.5, 1.0)
                    ped_control.direction.y = -1  # 向道路方向移动
                    ped_control.direction.x = 0.15
                    npc.apply_control(ped_control)
                    npc.set_simulate_physics(True)
                    self.walker_list.append(npc)
                    return True
            except Exception as e:
                continue
        
        return False

    def spawn_pedestrians_left(self, isCross):
        """在左侧生成行人"""
        blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        
        min_x = -50
        max_x = 140
        min_y = self.road_left - 5  # 道路左侧外5米
        max_y = self.road_left - 2

        if isCross:
            isFirstCross = random.choice([True, False])
            if isFirstCross:
                min_x = -14
                max_x = -10.5
            else:
                min_x = 17
                max_x = 20.5

        for attempt in range(3):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)

            spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            try:
                walker_bp = random.choice(blueprints_walkers)
                npc = self.world.try_spawn_actor(walker_bp, spawn_point)

                if npc is not None:
                    ped_control = carla.WalkerControl()
                    ped_control.speed = random.uniform(0.7, 1.3)
                    ped_control.direction.y = 1  # 向道路方向移动
                    ped_control.direction.x = -0.05
                    npc.apply_control(ped_control)
                    npc.set_simulate_physics(True)
                    self.walker_list.append(npc)
                    return True
            except Exception as e:
                continue
        
        return False

    def reset(self, episode=1):
        """重置环境"""
        self.current_episode = episode
        
        # 清理现有的行人和车辆
        self.cleanup_actors()
        
        # 重置行人列表
        self.walker_list = []

        # 根据训练阶段生成行人
        if episode < 100:
            print(f"Episode {episode}: 生成少量行人 (4十字路口 + 2非十字路口)")
            self.spawn_pedestrians_general(4, True)
            self.spawn_pedestrians_general(2, False)
        elif episode < 400:
            print(f"Episode {episode}: 生成中等数量行人 (6十字路口 + 3非十字路口)")
            self.spawn_pedestrians_general(6, True)
            self.spawn_pedestrians_general(3, False)
        else:
            print(f"Episode {episode}: 生成正常数量行人 (8十字路口 + 4非十字路口)")
            self.spawn_pedestrians_general(8, True)
            self.spawn_pedestrians_general(4, False)

        # 重置状态变量
        self.collision_history = []
        self.actor_list = []
        self.slow_counter = 0
        self.steer_counter = 0
        self.same_steer_counter = 0
        self.suggested_action = None
        self.last_action = 1
        self.episode_start_time = time.time()
        self.last_ped_distance = float('inf')
        self.obstacle_detected_time = None
        self.reaction_start_time = None
        self.proactive_action_count = 0
        self.frame_buffer.clear()
        self.motion_detected = False

        # 设置车辆生成点 - 在道路中心
        spawn_point = carla.Transform()
        spawn_point.location.x = -81.0
        spawn_point.location.y = self.road_center_y  # 道路中心
        spawn_point.location.z = 2.0
        spawn_point.rotation.roll = 0.0
        spawn_point.rotation.pitch = 0.0
        spawn_point.rotation.yaw = 0.0
        
        # 生成主车辆
        self.vehicle = self.world.spawn_actor(self.model_3, spawn_point)
        self.actor_list.append(self.vehicle)

        # 设置语义分割摄像头
        self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.sem_cam.set_attribute("fov", f"110")

        # 安装摄像头传感器
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.sem_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        # 初始化车辆控制
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        time.sleep(2)

        # 设置碰撞传感器
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # 等待摄像头初始化完成
        start_time = time.time()
        while self.front_camera is None and time.time() - start_time < 5:
            time.sleep(0.01)

        # 设置跟随相机
        self.setup_follow_camera()

        # 记录episode开始时间并重置控制
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))

        # 返回增强的状态信息
        return self.get_enhanced_state()

    def get_enhanced_state(self):
        """获取增强的状态信息（图像+位置+速度+方向）"""
        # 基础图像状态
        base_state = self.front_camera
        
        # 获取车辆状态信息
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            vehicle_location = self.vehicle.get_location()
            vehicle_transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            
            # 计算速度
            speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # 计算方向
            heading = vehicle_transform.rotation.yaw
            
            # 计算到道路中心的距离
            distance_to_center = abs(vehicle_location.y - self.road_center_y)
            
            # 创建状态字典
            state_info = {
                'image': base_state,
                'location': np.array([vehicle_location.x, vehicle_location.y]),  # 2维
                'speed': np.array([speed_kmh]),  # 1维
                'heading': np.array([heading]),  # 1维
                'distance_to_center': np.array([distance_to_center]),  # 1维 - 新增
                'last_action': np.array([self.last_action]) if hasattr(self, 'last_action') else np.array([1])  # 1维
            }
            
            return state_info
        
        # 默认返回
        return {
            'image': base_state, 
            'location': np.zeros(2), 
            'speed': np.array([0]), 
            'heading': np.array([0]),
            'distance_to_center': np.array([0]),
            'last_action': np.array([1])
        }

    def cleanup_actors(self):
        """清理所有actors"""
        try:
            vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in vehicles:
                if vehicle.is_alive:
                    vehicle.destroy()
        except:
            pass
        
        try:
            walkers = self.world.get_actors().filter('walker.*')
            for walker in walkers:
                if walker.is_alive:
                    walker.destroy()
        except:
            pass
                
        for actor in self.actor_list:
            try:
                if actor.is_alive:
                    actor.destroy()
            except:
                pass
                
        self.actor_list = []
        self.walker_list = []

    def setup_follow_camera(self):
        """设置跟随车辆的相机"""
        try:
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '110')
            
            camera_transform = carla.Transform(carla.Location(x=-8, z=6), carla.Rotation(pitch=-20))
            follow_camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            self.actor_list.append(follow_camera)
            print("跟随相机已设置")
        except Exception as e:
            print(f"设置跟随相机时出错: {e}")

    def collision_data(self, event):
        """处理碰撞事件"""
        self.collision_history.append(event)

    def process_img(self, image):
        """处理摄像头图像"""
        try:
            image.convert(carla.ColorConverter.CityScapesPalette)

            processed_image = np.array(image.raw_data)
            processed_image = processed_image.reshape((self.im_height, self.im_width, 4))
            processed_image = processed_image[:, :, :3]
            
            # 图像增强
            processed_image = cv2.convertScaleAbs(processed_image, alpha=1.2, beta=10)
            
            # 运动检测
            if len(self.frame_buffer) == 0:
                self.frame_buffer.append(processed_image.copy())
            else:
                if len(self.frame_buffer) >= 2:
                    prev_frame = self.frame_buffer[-1]
                    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                    gray_current = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
                    
                    diff = cv2.absdiff(gray_prev, gray_current)
                    _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    
                    motion_pixels = np.sum(diff_thresh > 0)
                    motion_ratio = motion_pixels / (self.im_width * self.im_height)
                    self.motion_detected = motion_ratio > 0.01
                
                self.frame_buffer.append(processed_image.copy())

            if self.SHOW_CAM:
                cv2.imshow("", processed_image)
                cv2.waitKey(1)

            self.front_camera = processed_image
        except Exception as e:
            print(f"处理图像时出错: {e}")

    def reward(self, speed_kmh, current_steer):
        """增强的奖励函数 - 简化版"""
        reward = 0
        done = False
        
        # 获取车辆状态
        vehicle_location = self.vehicle.get_location()
        vehicle_rotation = self.vehicle.get_transform().rotation.yaw
        
        # 1. 道路保持奖励 - 基于到道路中心的距离
        distance_to_center = abs(vehicle_location.y - self.road_center_y)
        
        if distance_to_center < 5:  # 完美保持在中心
            reward += 1.0
        elif distance_to_center < 10:  # 良好保持
            reward += 0.5
        elif distance_to_center < 15:  # 可接受
            reward += 0.1
        else:  # 偏离较大
            reward -= 0.5
        
        # 2. 道路边界检查
        boundary_distance, out_of_boundary = self.check_road_boundary(vehicle_location)
        
        if out_of_boundary:
            reward -= 20.0  # 大幅惩罚
            done = True
            print(f"Episode {self.current_episode}: 驶出道路边界! y={vehicle_location.y:.1f}, 范围[{self.road_left}, {self.road_right}]")
        elif distance_to_center > 20:  # 严重偏离但还在道路内
            reward -= 10.0
            done = True
            print(f"Episode {self.current_episode}: 严重偏离道路中心! 距离: {distance_to_center:.1f}m")
        
        # 3. 方向保持奖励
        heading_error = abs(vehicle_rotation)
        
        if heading_error < 5:
            reward += 0.5
        elif heading_error < 15:
            reward += 0.2
        
        # 4. 行人避障
        min_ped_distance = float('inf')
        closest_pedestrian = None
        active_pedestrians = 0
        
        for walker in self.walker_list:
            if not walker.is_alive:
                continue
                
            active_pedestrians += 1
            ped_location = walker.get_location()
            dx = vehicle_location.x - ped_location.x
            dy = vehicle_location.y - ped_location.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < min_ped_distance:
                min_ped_distance = distance
                closest_pedestrian = walker
        
        current_time = time.time()
        
        if min_ped_distance < 100:
            if self.obstacle_detected_time is None:
                self.obstacle_detected_time = current_time
                self.reaction_start_time = current_time
            
            # 行人距离分级奖励/惩罚
            if min_ped_distance < 3.0:
                reward -= 25.0
                done = True
                print(f"Episode {self.current_episode}: 与行人碰撞! 距离: {min_ped_distance:.1f}m")
                
            elif min_ped_distance < 5.0:
                reward -= 8.0
                
                if closest_pedestrian:
                    ped_y = closest_pedestrian.get_location().y
                    veh_y = vehicle_location.y
                    if ped_y < veh_y:
                        self.suggested_action = 4  # 行人在左侧，右转
                    else:
                        self.suggested_action = 3  # 行人在右侧，左转
                        
            elif min_ped_distance < 8.0:
                reward -= 3.0
                
                if closest_pedestrian:
                    ped_y = closest_pedestrian.get_location().y
                    veh_y = vehicle_location.y
                    if ped_y < veh_y:
                        self.suggested_action = 4
                    else:
                        self.suggested_action = 3
                        
            elif min_ped_distance < 12.0:
                # 预警距离，轻微惩罚
                reward -= 0.5
                    
        else:
            self.obstacle_detected_time = None
            self.reaction_start_time = None
        
        self.last_ped_distance = min_ped_distance
        
        # 5. 速度奖励
        if 20 <= speed_kmh <= 35:
            reward += 0.5
        elif 10 <= speed_kmh < 20:
            reward += 0.3
        elif speed_kmh > 40:
            reward -= 1.0
        
        # 6. 转向平滑性奖励
        steer_penalty = abs(current_steer) * 0.1
        reward -= steer_penalty
        
        # 7. 碰撞检测
        if len(self.collision_history) != 0:
            reward = -30
            done = True
            print(f"Episode {self.current_episode}: 发生碰撞!")
        
        # 8. 进度奖励
        progress = (vehicle_location.x + 81) / 236.0
        reward += progress * 0.5
        
        # 9. 边界检查
        if vehicle_location.x > 155:
            reward += 30
            done = True
            print(f"Episode {self.current_episode}: 成功到达终点!")
        elif vehicle_location.x < -90:
            reward -= 10
            done = True
            print(f"Episode {self.current_episode}: 反向行驶过远!")
        
        # 10. 主动避障奖励
        if self.proactive_action_count > 0:
            reward += self.proactive_action_count * 0.5
        
        # 限制奖励范围
        reward = max(min(reward, 40), -35)
        
        return reward, done

    def step(self, action):
        """执行动作并返回新状态"""
        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # 获取当前位置和到道路中心的距离
        vehicle_location = self.vehicle.get_location()
        distance_to_center = abs(vehicle_location.y - self.road_center_y)
        
        # 道路保持因子 - 偏离越大，转向越谨慎
        road_keeping_factor = max(0.3, 1.0 - distance_to_center / 30.0)
        
        speed_factor = max(0.5, min(1.0, 30.0 / max(1.0, speed_kmh)))
        
        throttle = 0.0
        brake = 0.0
        steer = 0.0
        
        if action == 0:
            throttle = 0.0
            brake = 0.7
        elif action == 1:
            throttle = 0.4
            brake = 0.0
        elif action == 2:
            throttle = 0.8
            brake = 0.0
        elif action == 3:
            throttle = 0.5
            brake = 0.0
            steer = -0.25 * speed_factor * road_keeping_factor
        elif action == 4:
            throttle = 0.5
            brake = 0.0
            steer = 0.25 * speed_factor * road_keeping_factor
        
        # 自动道路保持：如果偏离较大，自动轻微回正
        if distance_to_center > 10:
            auto_correction = 0.05 * (distance_to_center - 10) / 10.0
            if vehicle_location.y < self.road_center_y:  # 在道路左侧
                steer += auto_correction  # 轻微右转
            else:  # 在道路右侧
                steer -= auto_correction  # 轻微左转
            throttle = min(throttle, 0.6)  # 减速
        
        # 如果有建议的避让动作，调整当前动作
        if self.suggested_action is not None:
            if self.suggested_action in [3, 4] and action in [3, 4]:
                if self.suggested_action != action:
                    if self.suggested_action == 3:
                        steer = -0.3 * speed_factor
                    else:
                        steer = 0.3 * speed_factor
            elif self.suggested_action == 0:
                brake = max(brake, 0.8)
                throttle = 0.0
        
        # 限制连续同向转向
        if (action == 3 and self.last_action == 3) or (action == 4 and self.last_action == 4):
            self.same_steer_counter += 1
            if self.same_steer_counter > 2:
                steer *= 0.3
                throttle *= 0.6
                brake = 0.2
        else:
            self.same_steer_counter = 0
        
        self.last_action = action
        
        # 应用控制
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, 
            brake=brake, 
            steer=steer
        ))
        
        time.sleep(0.03)
        
        reward, done = self.reward(speed_kmh, steer)
        
        reward = np.clip(reward, -35, 40)
        
        # 获取增强的状态
        new_enhanced_state = self.get_enhanced_state()
        
        return new_enhanced_state, reward, done, None