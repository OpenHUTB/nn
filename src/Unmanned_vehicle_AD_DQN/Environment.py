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

import carla
from carla import ColorConverter


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW  # 是否显示摄像头预览
    im_width = IM_WIDTH  # 图像宽度
    im_height = IM_HEIGHT  # 图像高度

    def __init__(self):
        self.actor_list = []  # 存储所有actor的列表
        self.sem_cam = None  # 语义分割摄像头
        self.client = carla.Client("localhost", 2000)  # CARLA客户端
        self.client.set_timeout(20.0)  # 连接超时设置
        self.front_camera = None  # 前置摄像头图像
        
        # 新增变量
        self.last_action = 1  # 上一个动作，默认保持
        self.same_steer_counter = 0  # 连续同向转向计数器
        self.suggested_action = None  # 建议的避让动作
        self.episode_start_time = None  # 每轮开始时间
        self.last_ped_distance = float('inf')  # 上次最近行人距离
        self.current_episode = 1  # 当前episode编号

        # 加载世界和蓝图
        self.world = self.client.load_world('Town03')
        
        # 设置观察者视角，让CARLA窗口显示
        self.setup_observer_view()
        
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]  # Tesla Model3车辆

        # 行人列表和碰撞历史
        self.walker_list = []
        self.collision_history = []
        self.slow_counter = 0  # 慢速计数器
        self.steer_counter = 0  # 转向计数器，用于限制过度转向

    def setup_observer_view(self):
        """设置观察者视角，让用户可以在CARLA窗口中看到场景"""
        try:
            # 获取当前地图的生成点
            spawn_points = self.world.get_map().get_spawn_points()
            if spawn_points:
                # 选择一个合适的观察者位置
                spectator = self.world.get_spectator()
                
                # 设置观察者位置在车辆起始位置附近
                transform = carla.Transform()
                transform.location.x = -81.0
                transform.location.y = -195.0
                transform.location.z = 15.0  # 提高视角高度
                transform.rotation.pitch = -45.0  # 向下倾斜视角
                transform.rotation.yaw = 0.0
                transform.rotation.roll = 0.0
                
                spectator.set_transform(transform)
                print("观察者视角已设置")
        except Exception as e:
            print(f"设置观察者视角时出错: {e}")

    def spawn_pedestrians_general(self, number, isCross):
        """生成指定数量的行人"""
        # 记录要生成的数量
        target_number = number
        
        # 记录成功生成的数量
        success_count = 0
        
        # 尝试生成指定数量的行人
        attempts = 0
        max_attempts = number * 3  # 最多尝试3倍次数
        
        while success_count < target_number and attempts < max_attempts:
            attempts += 1
            isLeft = random.choice([True, False])  # 随机选择左右侧
            
            try:
                if isLeft:
                    if self.spawn_pedestrians_left(isCross):
                        success_count += 1
                else:
                    if self.spawn_pedestrians_right(isCross):
                        success_count += 1
            except Exception as e:
                # 如果生成失败，继续尝试
                print(f"生成行人失败: {e}")
                continue
        
        print(f"成功生成 {success_count}/{target_number} 个行人 (isCross={isCross})")
        return success_count

    def spawn_pedestrians_right(self, isCross):
        """在右侧生成行人"""
        blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        
        # 设置生成区域
        min_x = -50
        max_x = 140
        min_y = -188
        max_y = -183

        # 如果是十字路口，调整生成位置
        if isCross:
            isFirstCross = random.choice([True, False])
            if isFirstCross:
                min_x = -14
                max_x = -10.5
            else:
                min_x = 17
                max_x = 20.5

        # 尝试多次生成直到成功
        for attempt in range(3):  # 尝试3次
            # 随机生成位置
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)

            spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            # 避免在特定区域生成
            while (-10 < spawn_point.location.x < 17) or (70 < spawn_point.location.x < 100):
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            # 尝试生成行人
            try:
                walker_bp = random.choice(blueprints_walkers)
                npc = self.world.try_spawn_actor(walker_bp, spawn_point)

                if npc is not None:
                    # 设置行人控制参数
                    ped_control = carla.WalkerControl()
                    ped_control.speed = random.uniform(0.5, 1.0)  # 随机速度
                    ped_control.direction.y = -1  # 主要移动方向
                    ped_control.direction.x = 0.15  # 轻微横向移动
                    npc.apply_control(ped_control)
                    npc.set_simulate_physics(True)  # 启用物理模拟
                    self.walker_list.append(npc)  # 添加到行人列表
                    return True  # 生成成功
            except Exception as e:
                print(f"生成右侧行人失败 (尝试 {attempt+1}): {e}")
                continue
        
        return False  # 生成失败

    def spawn_pedestrians_left(self, isCross):
        """在左侧生成行人"""
        blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        
        # 设置生成区域
        min_x = -50
        max_x = 140
        min_y = -216
        max_y = -210

        # 如果是十字路口，调整生成位置
        if isCross:
            isFirstCross = random.choice([True, False])
            if isFirstCross:
                min_x = -14
                max_x = -10.5
            else:
                min_x = 17
                max_x = 20.5

        # 尝试多次生成直到成功
        for attempt in range(3):  # 尝试3次
            # 随机生成位置
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)

            spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            # 避免在特定区域生成
            while (-10 < spawn_point.location.x < 17) or (70 < spawn_point.location.x < 100):
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            # 尝试生成行人
            try:
                walker_bp = random.choice(blueprints_walkers)
                npc = self.world.try_spawn_actor(walker_bp, spawn_point)

                if npc is not None:
                    # 设置行人控制参数
                    ped_control = carla.WalkerControl()
                    ped_control.speed = random.uniform(0.7, 1.3)  # 随机速度
                    ped_control.direction.y = 1  # 主要移动方向
                    ped_control.direction.x = -0.05  # 轻微横向移动
                    npc.apply_control(ped_control)
                    npc.set_simulate_physics(True)  # 启用物理模拟
                    self.walker_list.append(npc)  # 添加到行人列表
                    return True  # 生成成功
            except Exception as e:
                print(f"生成左侧行人失败 (尝试 {attempt+1}): {e}")
                continue
        
        return False  # 生成失败

    def reset(self, episode=1):
        """重置环境，根据episode参数生成不同数量的行人"""
        self.current_episode = episode
        
        # 清理现有的行人和车辆
        self.cleanup_actors()
        
        # 重置行人列表
        self.walker_list = []

        # 根据训练阶段生成不同数量的行人
        if episode < 100:  # 第一阶段：少量行人
            print(f"Episode {episode}: 生成少量行人 (4十字路口 + 2非十字路口)")
            self.spawn_pedestrians_general(4, True)   # 十字路口行人
            self.spawn_pedestrians_general(2, False)  # 非十字路口行人
        elif episode < 400:  # 第二阶段：中等数量行人
            print(f"Episode {episode}: 生成中等数量行人 (6十字路口 + 3非十字路口)")
            self.spawn_pedestrians_general(6, True)
            self.spawn_pedestrians_general(3, False)
        else:  # 第三阶段：正常难度
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

        # 设置车辆生成点
        spawn_point = carla.Transform()
        spawn_point.location.x = -81.0
        spawn_point.location.y = -195.0
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
        self.sem_cam.set_attribute("fov", f"110")  # 视野角度

        # 安装摄像头传感器
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.sem_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))  # 设置图像处理回调

        # 初始化车辆控制
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        time.sleep(2)  # 等待环境稳定

        # 设置碰撞传感器
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))  # 设置碰撞检测回调

        # 等待摄像头初始化完成
        while self.front_camera is None:
            time.sleep(0.01)

        # 设置跟随相机（用于观察）
        self.setup_follow_camera()

        # 记录episode开始时间并重置控制
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))

        return self.front_camera

    def cleanup_actors(self):
        """清理所有actors"""
        # 清理车辆
        vehicles = self.world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        
        # 清理行人
        walkers = self.world.get_actors().filter('walker.*')
        for walker in walkers:
            if walker.is_alive:
                walker.destroy()
                
        # 清理传感器
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
                
        self.actor_list = []
        self.walker_list = []  # 清空行人列表

    def setup_follow_camera(self):
        """设置跟随车辆的相机，用于在CARLA窗口中观察"""
        try:
            # 创建RGB相机
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '110')
            
            # 相机位置相对于车辆（后方上方）
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
        image.convert(carla.ColorConverter.CityScapesPalette)  # 转换为CityScapes调色板

        # 处理原始图像数据
        processed_image = np.array(image.raw_data)
        processed_image = processed_image.reshape((self.im_height, self.im_width, 4))
        processed_image = processed_image[:, :, :3]  # 移除alpha通道

        # 显示预览（如果启用）
        if self.SHOW_CAM:
            cv2.imshow("", processed_image)
            cv2.waitKey(1)

        self.front_camera = processed_image  # 更新前置摄像头图像

    def reward(self, speed_kmh, current_steer):
        """增强的奖励函数 - 特别强调行人避障"""
        reward = 0
        done = False
        
        # 获取车辆状态
        vehicle_location = self.vehicle.get_location()
        vehicle_rotation = self.vehicle.get_transform().rotation.yaw
        
        # 1. 道路保持奖励（重要的基础）
        heading_error = abs(vehicle_rotation)
        
        if heading_error < 5:  # 完美保持方向
            reward += 0.8  # 略微降低权重，给行人避障更多空间
        elif heading_error < 15:  # 良好保持
            reward += 0.4
        elif heading_error < 30:  # 可接受
            reward += 0.1
        else:  # 方向偏差过大
            reward -= 0.3 * (heading_error / 30.0)  # 降低惩罚强度
        
        # 2. 行人避障（最高优先级） - 增强奖励机制
        min_ped_distance = float('inf')
        closest_pedestrian = None
        
        # 统计有效行人数量
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
        
        # 如果当前没有有效行人，重置距离
        if active_pedestrians == 0:
            min_ped_distance = float('inf')
        
        # 行人距离分级奖励 - 增强避障激励
        if min_ped_distance < 100:  # 只考虑100米内的行人
            if min_ped_distance < 3.0:  # 紧急避让距离
                reward -= 8.0  # 增加惩罚
                done = True
                print(f"Episode {self.current_episode}: 与行人距离过近 ({min_ped_distance:.1f}m)!")
            elif min_ped_distance < 5.0:  # 危险距离
                reward -= 3.0  # 增加惩罚
                # 计算避让方向
                if closest_pedestrian:
                    ped_y = closest_pedestrian.get_location().y
                    veh_y = vehicle_location.y
                    # 如果行人在车辆左侧，鼓励右转；反之鼓励左转
                    if ped_y < veh_y:  # 行人在左侧
                        self.suggested_action = 4  # 右转
                    else:  # 行人在右侧
                        self.suggested_action = 3  # 左转
            elif min_ped_distance < 8.0:  # 预警距离
                reward -= 0.8
            elif min_ped_distance < 12.0:  # 安全距离
                reward += 0.5  # 增加安全距离奖励
            else:  # 非常安全
                reward += 0.2
        
        # 3. 成功避障奖励 - 当成功避开行人后给予额外奖励
        if self.last_ped_distance < 8.0 and min_ped_distance > self.last_ped_distance:
            # 如果上次距离危险，这次距离更远了，说明成功避让
            reward += 0.3
        
        self.last_ped_distance = min_ped_distance  # 保存当前距离
        
        # 4. 速度奖励（平衡避障和前进）
        if 15 <= speed_kmh <= 35:  # 理想速度区间
            reward += 0.4
        elif 5 <= speed_kmh < 15:  # 较慢但安全（避障时可能需要减速）
            reward += 0.2
        elif 35 < speed_kmh <= 45:  # 稍快
            reward += 0.1
        elif speed_kmh > 45:  # 过快，在行人环境中危险
            reward -= 0.5  # 增加惩罚
        else:  # 停车或极慢
            reward -= 0.05  # 轻微惩罚
        
        # 5. 转向平滑性奖励（避障时可能需要适当转向）
        steer_penalty = abs(current_steer) * 0.3  # 降低惩罚，给避障转向更多空间
        reward -= steer_penalty
        
        # 6. 碰撞检测
        if len(self.collision_history) != 0:
            reward = -15  # 增加碰撞惩罚
            done = True
            print(f"Episode {self.current_episode}: 发生碰撞!")
        
        # 7. 进度奖励（次要于避障）
        progress = (vehicle_location.x + 81) / 236.0  # 从-81到155
        reward += progress * 0.2  # 降低进度权重
        
        # 8. 边界检查
        if vehicle_location.x > 155:  # 成功到达终点
            reward += 20  # 增加到达终点的奖励
            done = True
            print(f"Episode {self.current_episode}: 成功到达终点!")
        elif vehicle_location.x < -90 or abs(vehicle_location.y + 195) > 30:  # 偏离道路
            reward -= 3  # 降低偏离惩罚
            done = True
            print(f"Episode {self.current_episode}: 偏离道路!")
        
        return reward, done

    def step(self, action):
        """执行动作并返回新状态 - 增强平滑性和安全性"""
        # 获取当前速度
        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # 5个动作: 0-减速, 1-保持, 2-加速, 3-左转, 4-右转
        
        # 根据速度调整转向幅度 - 速度越快，转向越平滑
        speed_factor = max(0.5, min(1.0, 30.0 / max(1.0, speed_kmh)))
        
        # 基础控制参数
        throttle = 0.0
        brake = 0.0
        steer = 0.0
        
        # 速度控制
        if action == 0:  # 减速
            throttle = 0.0
            brake = 0.5
        elif action == 1:  # 保持/轻微加速
            throttle = 0.3
            brake = 0.0
        elif action == 2:  # 加速
            throttle = 0.7
            brake = 0.0
        elif action == 3:  # 左转
            throttle = 0.4
            brake = 0.0
            steer = -0.2 * speed_factor  # 速度相关的转向幅度
        elif action == 4:  # 右转
            throttle = 0.4
            brake = 0.0
            steer = 0.2 * speed_factor   # 速度相关的转向幅度
        
        # 限制连续同向转向 - 防止过度转向
        if (action == 3 and self.last_action == 3) or (action == 4 and self.last_action == 4):
            self.same_steer_counter += 1
            if self.same_steer_counter > 3:  # 连续3次同向转向后强制回正
                steer *= 0.5  # 减小转向幅度
                throttle *= 0.8  # 减速
        else:
            self.same_steer_counter = 0
        
        # 记录上一个动作
        self.last_action = action
        
        # 应用控制
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, 
            brake=brake, 
            steer=steer
        ))
        
        # 等待物理更新
        time.sleep(0.05)
        
        # 计算奖励和完成状态
        reward, done = self.reward(speed_kmh, steer)
        
        # 限制极端奖励值
        reward = np.clip(reward, -15, 20)
        
        return self.front_camera, reward, done, None