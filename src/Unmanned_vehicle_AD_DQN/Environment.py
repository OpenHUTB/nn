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
        """生成指定数量的行人 - 大幅减少数量"""
        # 限制最大生成数量
        number = min(number, 8)  # 最多8个行人
        
        for i in range(number):
            isLeft = random.choice([True, False])  # 随机选择左右侧
            if isLeft:
                self.spawn_pedestrians_left(isCross)
            else:
                self.spawn_pedestrians_right(isCross)

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

    def reset(self):
        """重置环境"""
        # 清理现有的行人和车辆
        self.cleanup_actors()
        
        # 重置行人列表
        self.walker_list = []

        # 大幅减少行人数量 - 从30+10减少到8+4
        self.spawn_pedestrians_general(8, True)
        self.spawn_pedestrians_general(4, False)

        # 重置状态变量
        self.collision_history = []
        self.actor_list = []
        self.slow_counter = 0
        self.steer_counter = 0

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

    def reward(self):
        """计算奖励函数 - 增强方向控制奖励"""
        reward = 0
        done = False

        # 计算车辆速度
        velocity = self.vehicle.get_velocity()
        velocity_kmh = int(3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))
        
        # 获取车辆位置和方向
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
            
        # 增强方向奖励 - 确保车辆朝正确方向行驶
        if -20 <= vehicle_rotation <= 20:  # 严格限制在正东方向附近
            reward += 0.5  # 增加直行奖励
            self.steer_counter = max(0, self.steer_counter - 1)  # 减少转向计数
        elif -45 <= vehicle_rotation <= 45:
            reward += 0.1  # 较小奖励
        else:
            reward -= 1.0  # 严重偏离惩罚
            self.steer_counter += 2  # 增加转向计数
            
        # 行人距离检测
        min_dist = float('inf')  # 最小距离初始化为无穷大
        for walker in self.walker_list:
            if not walker.is_alive:
                continue
                
            ped_location = walker.get_location()
            dx = vehicle_location.x - ped_location.x
            dy = vehicle_location.y - ped_location.y
            distance = math.sqrt(dx**2 + dy**2)
            min_dist = min(min_dist, distance)  # 更新最小距离
            
            # 清理边界外的行人
            try:
                player_direction = walker.get_control().direction
                if (ped_location.y < -214 and player_direction.y == -1) or \
                   (ped_location.y > -191 and player_direction.y == 1):
                    if walker.is_alive:
                        walker.destroy()
            except:
                pass  # 如果无法获取控制信息，跳过

        # 基于行人距离的奖励
        if min_dist < 3.0:  # 非常危险距离
            reward -= 3.0
            done = True
        elif min_dist < 5.0:  # 危险距离
            reward -= 1.0
        elif min_dist < 8.0:  # 警告距离
            reward -= 0.3
        elif min_dist > 15.0:  # 安全距离
            reward += 0.2
            
        # 碰撞检测
        if len(self.collision_history) != 0:
            reward = -10  # 碰撞惩罚
            done = True
            
        # 进度奖励
        reward += progress_reward * 0.5
        
        # 完成条件判断
        if vehicle_location.x > 155:  # 成功到达终点
            reward += 10  # 成功到达奖励
            done = True
        elif vehicle_location.x < -90:  # 倒退太多
            reward -= 5
            done = True
            
        return reward, done

    def step(self, action):
        """执行动作并返回新状态 - 扩展为5个动作包含转向"""
        # 扩展的动作空间: 0-减速, 1-保持, 2-加速, 3-左转, 4-右转
        
        # 限制连续转向次数，避免过度转向
        max_continuous_steer = 3
        current_steer = 0.0
        
        if action == 3:  # 左转
            if self.steer_counter < max_continuous_steer:
                current_steer = -0.3  # 小角度左转
                self.steer_counter += 1
            else:
                # 强制直行一段时间
                current_steer = 0.0
                action = 1  # 改为保持动作
        elif action == 4:  # 右转
            if self.steer_counter < max_continuous_steer:
                current_steer = 0.3  # 小角度右转
                self.steer_counter += 1
            else:
                # 强制直行一段时间
                current_steer = 0.0
                action = 1  # 改为保持动作
        else:
            # 非转向动作时逐渐减少转向计数
            self.steer_counter = max(0, self.steer_counter - 0.5)
        
        # 速度控制
        throttle = 0.0
        brake = 0.0
        
        if action == 0:  # 减速
            throttle = 0.0
            brake = 0.3
        elif action == 1:  # 保持/轻微加速
            throttle = 0.3
            brake = 0.0
        elif action == 2:  # 加速
            throttle = 0.7
            brake = 0.0
        elif action == 3 or action == 4:  # 转向时保持适中速度
            throttle = 0.4
            brake = 0.0

        # 应用控制
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, 
            brake=brake, 
            steer=current_steer
        ))

        # 等待物理更新
        time.sleep(0.05)
        
        # 计算奖励和完成状态
        reward, done = self.reward()
        
        # 限制极端奖励值
        reward = np.clip(reward, -10, 10)
        
        return self.front_camera, reward, done, None