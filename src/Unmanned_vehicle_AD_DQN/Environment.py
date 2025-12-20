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
from collections import deque

import carla
from carla import ColorConverter


class EnhancedCarEnv:
    SHOW_CAM = SHOW_PREVIEW
    im_width = IM_WIDTH
    im_height = IM_HEIGHT

    def __init__(self, obstacle_detection_mode='advanced'):
        self.actor_list = []
        self.sem_cam = None
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('Town03')
        self.front_camera = None
        
        # 增强的状态跟踪
        self.last_action = 1
        self.same_steer_counter = 0
        self.suggested_action = None
        self.episode_start_time = None
        self.last_ped_distance = float('inf')
        self.current_episode = 1
        self.obstacle_detection_mode = obstacle_detection_mode
        
        # 障碍物检测增强
        self.obstacle_history = deque(maxlen=10)  # 障碍物历史
        self.obstacle_warning_level = 0  # 0-无警告, 1-轻微, 2-中等, 3-严重
        self.last_obstacle_type = None  # 'pedestrian', 'vehicle', 'building'
        
        # 碰撞和避障统计
        self.collision_history = []
        self.near_miss_history = []
        self.successful_avoidance = 0
        self.obstacle_encounters = 0
        
        # 初始化设置
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.walker_list = []
        self.slow_counter = 0
        self.steer_counter = 0
        
        # 设置观察者视角
        self.setup_observer_view()
        
        # 建筑物检测参数
        self.building_detection_enabled = True
        self.building_proximity_threshold = 15.0  # 建筑物接近阈值（米）
        
        # 语义分割颜色映射（用于障碍物检测）
        self.semantic_colors = {
            4: 'building',    # 建筑物 - 蓝色
            10: 'pedestrian', # 行人 - 红色
            12: 'vehicle',    # 车辆 - 绿色
        }

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

    def spawn_pedestrians_with_config(self, config):
        """根据配置生成行人"""
        cross_count = config.get('pedestrian_cross', 4)
        normal_count = config.get('pedestrian_normal', 2)
        
        print(f"生成行人: 十字路口={cross_count}, 普通路段={normal_count}")
        
        # 清理现有行人
        self.cleanup_pedestrians()
        
        # 生成十字路口行人
        for _ in range(cross_count):
            self.spawn_pedestrian_at_crossing()
        
        # 生成普通路段行人
        for _ in range(normal_count):
            self.spawn_pedestrian_normal()
    
    def spawn_pedestrian_at_crossing(self):
        """在十字路口生成行人"""
        blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        
        # 两个主要十字路口位置
        crossings = [
            {'x_range': (-14, -10.5), 'y_range': (-188, -183)},  # 第一个十字路口右侧
            {'x_range': (17, 20.5), 'y_range': (-188, -183)},    # 第二个十字路口右侧
            {'x_range': (-14, -10.5), 'y_range': (-216, -210)},  # 第一个十字路口左侧
            {'x_range': (17, 20.5), 'y_range': (-216, -210)}     # 第二个十字路口左侧
        ]
        
        crossing = random.choice(crossings)
        x = random.uniform(crossing['x_range'][0], crossing['x_range'][1])
        y = random.uniform(crossing['y_range'][0], crossing['y_range'][1])
        
        spawn_point = carla.Transform(carla.Location(x, y, 2.0))
        
        try:
            walker_bp = random.choice(blueprints_walkers)
            
            # 设置行人属性
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            
            npc = self.world.try_spawn_actor(walker_bp, spawn_point)
            
            if npc is not None:
                # 设置行人移动
                ped_control = carla.WalkerControl()
                
                # 十字路口行人更活跃
                ped_control.speed = random.uniform(0.8, 1.5)
                
                # 随机选择移动方向
                if random.random() > 0.5:
                    ped_control.direction.y = -1  # 南北向
                    ped_control.direction.x = random.uniform(-0.2, 0.2)
                else:
                    ped_control.direction.x = 1   # 东西向
                    ped_control.direction.y = random.uniform(-0.2, 0.2)
                
                npc.apply_control(ped_control)
                npc.set_simulate_physics(True)
                self.walker_list.append(npc)
                
                # 添加随机移动模式
                self.setup_pedestrian_ai(npc)
                
                return True
        except Exception as e:
            print(f"生成十字路口行人失败: {e}")
        
        return False

    def spawn_pedestrian_normal(self):
        """在普通路段生成行人"""
        blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        
        # 普通路段位置
        if random.random() > 0.5:  # 右侧
            x_range = (-50, 140)
            y_range = (-188, -183)
            direction_y = -1
        else:  # 左侧
            x_range = (-50, 140)
            y_range = (-216, -210)
            direction_y = 1
        
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        
        # 避开十字路口区域
        while (-14 < x < -10.5) or (17 < x < 20.5) or (70 < x < 100):
            x = random.uniform(x_range[0], x_range[1])
        
        spawn_point = carla.Transform(carla.Location(x, y, 2.0))
        
        try:
            walker_bp = random.choice(blueprints_walkers)
            npc = self.world.try_spawn_actor(walker_bp, spawn_point)
            
            if npc is not None:
                ped_control = carla.WalkerControl()
                ped_control.speed = random.uniform(0.5, 1.2)
                ped_control.direction.y = direction_y
                ped_control.direction.x = random.uniform(-0.1, 0.1)
                
                npc.apply_control(ped_control)
                npc.set_simulate_physics(True)
                self.walker_list.append(npc)
                
                return True
        except Exception as e:
            print(f"生成普通行人失败: {e}")
        
        return False

    def setup_pedestrian_ai(self, pedestrian):
        """设置行人AI行为"""
        # 这里可以添加更复杂的行人AI逻辑
        # 目前使用简单的随机移动
        pass

    def cleanup_pedestrians(self):
        """清理所有行人"""
        for walker in self.walker_list:
            if walker.is_alive:
                walker.destroy()
        self.walker_list = []

    def reset(self, episode=1, curriculum_config=None):
        """重置环境"""
        self.current_episode = episode
        
        # 清理现有的actors
        self.cleanup_actors()
        self.cleanup_pedestrians()
        
        # 重置状态变量
        self.collision_history = []
        self.near_miss_history = []
        self.actor_list = []
        self.slow_counter = 0
        self.steer_counter = 0
        self.same_steer_counter = 0
        self.suggested_action = None
        self.last_action = 1
        self.episode_start_time = time.time()
        self.last_ped_distance = float('inf')
        self.obstacle_warning_level = 0
        self.last_obstacle_type = None
        self.successful_avoidance = 0
        self.obstacle_encounters = 0
        
        # 根据课程学习配置或episode阶段生成行人
        if curriculum_config:
            self.spawn_pedestrians_with_config(curriculum_config)
        else:
            # 默认阶段划分
            if episode < 100:
                config = {'pedestrian_cross': 2, 'pedestrian_normal': 1}
            elif episode < 300:
                config = {'pedestrian_cross': 4, 'pedestrian_normal': 2}
            elif episode < 500:
                config = {'pedestrian_cross': 6, 'pedestrian_normal': 3}
            else:
                config = {'pedestrian_cross': 8, 'pedestrian_normal': 4}
            
            self.spawn_pedestrians_with_config(config)
        
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
        self.sem_cam.set_attribute("fov", f"110")
        
        # 安装摄像头传感器
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.sem_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img_enhanced(data))
        
        # 初始化车辆控制
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        time.sleep(1.5)  # 等待环境稳定
        
        # 设置碰撞传感器
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        
        # 设置车道入侵传感器（可选）
        try:
            lane_sensor = self.blueprint_library.find("sensor.other.lane_invasion")
            self.lane_sensor = self.world.spawn_actor(lane_sensor, transform, attach_to=self.vehicle)
            self.actor_list.append(self.lane_sensor)
            self.lane_sensor.listen(lambda event: self.lane_invasion_data(event))
        except:
            pass
        
        # 等待摄像头初始化完成
        while self.front_camera is None:
            time.sleep(0.01)
        
        # 设置跟随相机
        self.setup_follow_camera()
        
        # 记录episode开始时间并重置控制
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        
        print(f"Episode {episode} 环境重置完成，有 {len(self.walker_list)} 个行人")
        
        return self.front_camera

    def cleanup_actors(self):
        """清理所有actors"""
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        self.actor_list = []

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
        except Exception as e:
            print(f"设置跟随相机时出错: {e}")

    def collision_data(self, event):
        """处理碰撞事件"""
        self.collision_history.append({
            'timestamp': time.time(),
            'actor': event.other_actor,
            'impulse': event.normal_impulse,
            'type': event.other_actor.type_id if event.other_actor else 'unknown'
        })
        
        # 记录碰撞类型
        if event.other_actor:
            if 'walker' in event.other_actor.type_id:
                self.last_obstacle_type = 'pedestrian'
                print(f"⚠️ 与行人发生碰撞!")
            elif 'vehicle' in event.other_actor.type_id:
                self.last_obstacle_type = 'vehicle'
                print(f"⚠️ 与车辆发生碰撞!")
            else:
                self.last_obstacle_type = 'building'
                print(f"⚠️ 与建筑物发生碰撞!")

    def lane_invasion_data(self, event):
        """处理车道入侵事件"""
        # 可以用于检测偏离道路
        pass

    def process_img_enhanced(self, image):
        """增强的图像处理，包含障碍物检测"""
        image.convert(carla.ColorConverter.CityScapesPalette)
        
        # 处理原始图像数据
        processed_image = np.array(image.raw_data)
        processed_image = processed_image.reshape((self.im_height, self.im_width, 4))
        processed_image = processed_image[:, :, :3]
        
        # 缩小图像尺寸以减少内存使用
        target_height, target_width = 240, 320
        processed_image = cv2.resize(processed_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # 障碍物检测和分析
        if self.obstacle_detection_mode != 'none':
            obstacle_info = self.detect_obstacles_from_image(processed_image)
            
            # 更新障碍物历史
            self.obstacle_history.append(obstacle_info)
            
            # 计算警告级别
            self.update_obstacle_warning(obstacle_info)
        
        # 显示预览（如果启用）
        if self.SHOW_CAM:
            # 在图像上绘制障碍物信息
            if hasattr(self, 'obstacle_warning_level') and self.obstacle_warning_level > 0:
                warning_text = f"障碍物警告: 等级{self.obstacle_warning_level}"
                cv2.putText(processed_image, warning_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("语义分割视图", processed_image)
            cv2.waitKey(1)
        
        self.front_camera = processed_image

    def detect_obstacles_from_image(self, image):
        """从语义分割图像中检测障碍物"""
        obstacle_info = {
            'has_obstacle': False,
            'obstacle_type': None,
            'obstacle_distance': float('inf'),
            'obstacle_position': None,  # (x, y) 在图像中的位置
            'obstacle_size': 0,  # 障碍物像素数量
            'warning_level': 0
        }
        
        # 简化版的障碍物检测（基于颜色）
        # 在实际应用中，应该使用更复杂的语义分割结果
        
        # 检测红色区域（行人）
        red_mask = cv2.inRange(image, (200, 0, 0), (255, 50, 50))
        red_pixels = cv2.countNonZero(red_mask)
        
        # 检测蓝色区域（建筑物）
        blue_mask = cv2.inRange(image, (0, 0, 200), (50, 50, 255))
        blue_pixels = cv2.countNonZero(blue_mask)
        
        # 检测绿色区域（车辆）
        green_mask = cv2.inRange(image, (0, 200, 0), (50, 255, 50))
        green_pixels = cv2.countNonZero(green_mask)
        
        # 确定主要障碍物类型
        max_pixels = max(red_pixels, blue_pixels, green_pixels)
        
        if max_pixels > 100:  # 阈值
            obstacle_info['has_obstacle'] = True
            obstacle_info['obstacle_size'] = max_pixels
            
            if max_pixels == red_pixels:
                obstacle_info['obstacle_type'] = 'pedestrian'
                # 计算障碍物在图像中的位置（中心）
                moments = cv2.moments(red_mask)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    obstacle_info['obstacle_position'] = (cx, cy)
                    
                    # 根据位置估计距离（简化）
                    # 中心越靠近图像底部，距离越近
                    distance_estimate = (self.im_height - cy) / self.im_height * 50 + 5
                    obstacle_info['obstacle_distance'] = distance_estimate
                    
            elif max_pixels == blue_pixels:
                obstacle_info['obstacle_type'] = 'building'
            elif max_pixels == green_pixels:
                obstacle_info['obstacle_type'] = 'vehicle'
        
        return obstacle_info

    def update_obstacle_warning(self, obstacle_info):
        """更新障碍物警告级别"""
        if not obstacle_info['has_obstacle']:
            self.obstacle_warning_level = 0
            return
        
        distance = obstacle_info['obstacle_distance']
        size = obstacle_info['obstacle_size']
        
        # 根据距离和大小计算警告级别
        if distance < 5.0 or size > 1000:
            self.obstacle_warning_level = 3  # 严重
        elif distance < 10.0 or size > 500:
            self.obstacle_warning_level = 2  # 中等
        elif distance < 20.0 or size > 100:
            self.obstacle_warning_level = 1  # 轻微
        else:
            self.obstacle_warning_level = 0  # 无
        
        self.last_obstacle_type = obstacle_info['obstacle_type']

    def detect_obstacles_raycast(self):
        """使用射线投射检测障碍物（更精确）"""
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        
        obstacles = []
        
        # 在前方多个方向发射射线
        for angle in range(-30, 31, 10):  # -30度到30度，每10度一条射线
            # 计算射线方向
            ray_direction = vehicle_transform.rotation
            ray_direction.yaw += angle
            
            # 创建射线终点
            ray_end = vehicle_location + carla.Location(
                x=50 * math.cos(math.radians(ray_direction.yaw)),
                y=50 * math.sin(math.radians(ray_direction.yaw)),
                z=0
            )
            
            # 执行射线投射
            raycast_result = self.world.cast_ray(vehicle_location, ray_end)
            
            if raycast_result:
                hit_location, hit_actor = raycast_result
                distance = vehicle_location.distance(hit_location)
                
                if distance < 30:  # 只关心30米内的障碍物
                    obstacle_type = 'unknown'
                    if hit_actor:
                        if 'walker' in hit_actor.type_id:
                            obstacle_type = 'pedestrian'
                        elif 'vehicle' in hit_actor.type_id:
                            obstacle_type = 'vehicle'
                        elif 'building' in hit_actor.type_id or 'static' in hit_actor.type_id:
                            obstacle_type = 'building'
                    
                    obstacles.append({
                        'type': obstacle_type,
                        'distance': distance,
                        'direction': angle,
                        'location': hit_location
                    })
        
        return obstacles

    def calculate_avoidance_direction(self, obstacles):
        """计算最佳避让方向"""
        if not obstacles:
            return None
        
        # 分析障碍物分布
        left_obstacles = [o for o in obstacles if o['direction'] < 0]
        right_obstacles = [o for o in obstacles if o['direction'] > 0]
        center_obstacles = [o for o in obstacles if abs(o['direction']) <= 10]
        
        # 找到最近的障碍物
        min_distance = min(o['distance'] for o in obstacles)
        nearest_obstacles = [o for o in obstacles if o['distance'] == min_distance]
        
        if not nearest_obstacles:
            return None
        
        nearest = nearest_obstacles[0]
        
        # 根据最近的障碍物决定避让方向
        if nearest['distance'] < 5.0:  # 紧急情况
            if nearest['direction'] < 0:  # 障碍物在左侧，向右避让
                return 4  # 右转
            else:  # 障碍物在右侧，向左避让
                return 3  # 左转
        elif nearest['distance'] < 10.0:  # 警告情况
            # 选择障碍物较少的一侧
            if len(left_obstacles) < len(right_obstacles):
                return 4  # 向右避让
            else:
                return 3  # 向左避让
        
        return None

    def reward_enhanced(self, speed_kmh, current_steer):
        """增强的奖励函数，特别注重避障"""
        reward = 0
        done = False
        
        # 获取车辆状态
        vehicle_location = self.vehicle.get_location()
        vehicle_rotation = self.vehicle.get_transform().rotation.yaw
        
        # 1. 基础安全奖励
        reward += 0.1  # 每步存活奖励
        
        # 2. 障碍物检测与避让（最高优先级）
        obstacle_reward, obstacle_done, avoidance_success = self.calculate_obstacle_reward()
        reward += obstacle_reward
        if obstacle_done:
            done = True
            
        # 记录避障成功
        if avoidance_success:
            self.successful_avoidance += 1
        
        # 3. 道路保持奖励
        heading_error = abs(vehicle_rotation)
        if heading_error < 5:
            reward += 0.5
        elif heading_error < 15:
            reward += 0.2
        elif heading_error < 30:
            reward += 0.1
        else:
            reward -= 0.2
        
        # 4. 速度奖励（平衡避障和前进）
        if 15 <= speed_kmh <= 35:
            reward += 0.3
        elif 5 <= speed_kmh < 15:
            reward += 0.1
        elif 35 < speed_kmh <= 45:
            reward += 0.05
        elif speed_kmh > 45:
            reward -= 0.3
        else:
            reward -= 0.05
        
        # 5. 转向平滑性奖励
        steer_penalty = abs(current_steer) * 0.2
        reward -= steer_penalty
        
        # 6. 碰撞检测（已经在obstacle_reward中处理，但这里双重检查）
        if len(self.collision_history) > 0:
            # 检查最近2秒内是否有碰撞
            recent_collisions = [c for c in self.collision_history 
                                if time.time() - c['timestamp'] < 2.0]
            if recent_collisions:
                reward = -20
                done = True
                print(f"Episode {self.current_episode}: 发生碰撞!")
        
        # 7. 进度奖励
        progress = (vehicle_location.x + 81) / 236.0
        reward += progress * 0.3
        
        # 8. 边界检查
        if vehicle_location.x > 155:
            reward += 25  # 成功到达终点
            done = True
            print(f"Episode {self.current_episode}: 成功到达终点!")
            
            # 根据避障表现给予额外奖励
            if self.successful_avoidance > 0:
                reward += self.successful_avoidance * 2
                print(f"成功避让 {self.successful_avoidance} 次障碍物，获得额外奖励!")
                
        elif vehicle_location.x < -90 or abs(vehicle_location.y + 195) > 30:
            reward -= 5
            done = True
            print(f"Episode {self.current_episode}: 偏离道路!")
        
        # 9. 时间惩罚（防止停滞）
        elapsed_time = time.time() - self.episode_start_time
        if elapsed_time > 60 and progress < 0.3:  # 1分钟内进度不足30%
            reward -= 0.01 * elapsed_time
        
        # 限制极端奖励值
        reward = np.clip(reward, -25, 30)
        
        return reward, done, {
            'obstacle_avoidance_success': avoidance_success,
            'obstacle_encounter': self.obstacle_encounters > 0,
            'progress': progress
        }

    def calculate_obstacle_reward(self):
        """计算障碍物相关奖励"""
        reward = 0
        done = False
        avoidance_success = False
        
        vehicle_location = self.vehicle.get_location()
        
        # 检测最近的障碍物
        min_distance = float('inf')
        closest_obstacle = None
        
        # 检查行人
        for walker in self.walker_list:
            if not walker.is_alive:
                continue
                
            ped_location = walker.get_location()
            dx = vehicle_location.x - ped_location.x
            dy = vehicle_location.y - ped_location.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_obstacle = walker
        
        self.last_ped_distance = min_distance
        
        # 障碍物距离处理
        if min_distance < 100:  # 只考虑100米内的障碍物
            self.obstacle_encounters += 1
            
            if min_distance < 3.0:  # 碰撞距离
                reward -= 15
                done = True
                print(f"Episode {self.current_episode}: 与障碍物距离过近 ({min_distance:.1f}m)!")
                
            elif min_distance < 5.0:  # 危险距离
                reward -= 5
                # 计算避让方向
                if closest_obstacle:
                    ped_y = closest_obstacle.get_location().y
                    veh_y = vehicle_location.y
                    if ped_y < veh_y:  # 行人在左侧
                        self.suggested_action = 4  # 右转
                    else:  # 行人在右侧
                        self.suggested_action = 3  # 左转
                        
            elif min_distance < 8.0:  # 警告距离
                reward -= 1
                
            elif min_distance < 12.0:  # 安全距离，成功避让
                reward += 2
                avoidance_success = True
                
            else:  # 非常安全
                reward += 0.5
        
        # 检查历史避障成功（如果上次危险，这次安全了）
        if hasattr(self, 'last_min_distance') and self.last_min_distance < 8.0 and min_distance > self.last_min_distance:
            reward += 1.5  # 成功避让奖励
            avoidance_success = True
        
        # 更新历史距离
        if hasattr(self, 'last_min_distance'):
            self.last_min_distance = min_distance
        
        return reward, done, avoidance_success

    def step(self, action):
        """执行动作并返回新状态"""
        # 获取当前速度
        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # 根据障碍物警告级别调整动作
        adjusted_action = self.adjust_action_for_obstacles(action, speed_kmh)
        
        # 速度相关的转向幅度
        speed_factor = max(0.4, min(1.0, 25.0 / max(1.0, speed_kmh)))
        
        # 基础控制参数
        throttle = 0.0
        brake = 0.0
        steer = 0.0
        
        # 动作执行
        if adjusted_action == 0:  # 减速
            throttle = 0.0
            brake = 0.6
        elif adjusted_action == 1:  # 保持
            throttle = 0.3
            brake = 0.0
        elif adjusted_action == 2:  # 加速
            throttle = 0.7
            brake = 0.0
        elif adjusted_action == 3:  # 左转
            throttle = 0.4
            brake = 0.0
            steer = -0.25 * speed_factor
        elif adjusted_action == 4:  # 右转
            throttle = 0.4
            brake = 0.0
            steer = 0.25 * speed_factor
        
        # 防止过度转向
        if (adjusted_action == 3 and self.last_action == 3) or (adjusted_action == 4 and self.last_action == 4):
            self.same_steer_counter += 1
            if self.same_steer_counter > 3:
                steer *= 0.5
                throttle *= 0.7
        else:
            self.same_steer_counter = 0
        
        self.last_action = adjusted_action
        
        # 应用控制
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer,
            hand_brake=False,
            reverse=False
        ))
        
        # 等待物理更新
        time.sleep(0.05)
        
        # 计算奖励
        reward, done, extra_info = self.reward_enhanced(speed_kmh, steer)
        
        # 为多目标优化准备额外信息
        obstacle_info = {
            'min_distance': self.last_ped_distance,
            'warning_level': self.obstacle_warning_level,
            'obstacle_type': self.last_obstacle_type,
            'avoidance_success': extra_info['obstacle_avoidance_success']
        }
        
        return self.front_camera, reward, done, {
            'obstacle_info': obstacle_info,
            'speed_kmh': speed_kmh,
            'progress': extra_info['progress']
        }

    def adjust_action_for_obstacles(self, action, speed_kmh):
        """根据障碍物情况调整动作"""
        # 如果有建议的避让动作，优先执行
        if self.suggested_action is not None:
            adjusted = self.suggested_action
            self.suggested_action = None
            return adjusted
        
        # 根据障碍物警告级别调整
        if self.obstacle_warning_level >= 2:  # 中等或严重警告
            if action == 2:  # 如果原动作是加速，改为保持或减速
                return 1 if speed_kmh < 20 else 0
            elif action in [3, 4] and speed_kmh > 30:  # 高速时避免急转
                return 1  # 改为保持
        
        # 如果距离障碍物很近，强制减速
        if self.last_ped_distance < 6.0 and action == 2:
            return 0  # 强制减速
        
        return action

    def get_environment_info(self):
        """获取环境信息"""
        vehicle_location = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
        
        return {
            'vehicle_location': (vehicle_location.x, vehicle_location.y),
            'speed_kmh': speed_kmh,
            'pedestrian_count': len(self.walker_list),
            'collision_count': len(self.collision_history),
            'successful_avoidance': self.successful_avoidance,
            'obstacle_warning_level': self.obstacle_warning_level,
            'last_obstacle_distance': self.last_ped_distance
        }


# 为兼容性保留原类名
CarEnv = EnhancedCarEnv