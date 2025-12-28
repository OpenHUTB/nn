#!/usr/bin/env python3

import carla
import config as Config
import math
import numpy as np
from drawer import PyGameDrawer
from sync_pygame import SyncPyGame
from mpc import MPC


class Main():

    def __init__(self):
        # setup world
        self.client = carla.Client(Config.CARLA_SERVER, 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(Config.WORLD_NAME)
        self.map = self.world.get_map()

        # spawn ego
        ego_spawn_point = self.map.get_spawn_points()[100]
        bp = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        self.ego = self.world.spawn_actor(bp, ego_spawn_point)

        # init game and drawer
        self.game = SyncPyGame(self)
        self.drawer = PyGameDrawer(self)
        self.mpc = MPC(self.drawer, self.ego)

        # 刹车状态跟踪
        self.is_braking = False
        self.brake_history = []
        self.speed_history = []  # 速度历史记录
        self.target_speed_kmh = 40  # 目标速度40km/h
        self.brake_force = 0.0  # 当前刹车力度
        self.frame_count = 0  # 帧计数
        self.steer_angle = 0.0  # 转向角度
        self.throttle_value = 0.6  # 油门值
        self.control_mode = "AUTO"  # 控制模式
        self.collision_warning = False  # 碰撞警告
        self.collision_history = []  # 碰撞警告历史

        # 驾驶评分系统
        self.driving_score = 100.0  # 初始驾驶评分
        self.score_history = []  # 评分历史记录
        self.score_factors = {
            'speed_stability': 0.0,  # 速度稳定性
            'steering_smoothness': 0.0,  # 转向平滑度
            'brake_usage': 0.0,  # 刹车使用情况
            'path_following': 0.0,  # 路径跟踪
            'safety': 0.0  # 安全性
        }

        # 航点导航系统
        self.current_waypoint_index = 0  # 当前航点索引
        self.waypoint_positions = []  # 航点位置列表
        self.distance_to_waypoint = 0.0  # 到当前航点的距离
        self.waypoint_reached_count = 0  # 已到达航点计数
        self.waypoint_progress = 0.0  # 航点进度（0-1）

        # 障碍物检测系统
        self.obstacles = []  # 障碍物列表
        self.obstacle_detection_range = 50.0  # 障碍物检测范围（米）

        # 驾驶辅助系统状态
        self.lane_assist_active = True  # 车道保持辅助状态
        self.adaptive_cruise_active = True  # 自适应巡航状态
        self.collision_avoidance_active = True  # 碰撞避免状态

        # start game loop
        self.game.game_loop(self.world, self.on_tick)

    def on_tick(self):
        self.frame_count += 1

        # generate reference path (global frame)
        lookahead = 5
        wp = self.map.get_waypoint(self.ego.get_location())
        path = []

        for _ in range(lookahead):
            _wps = wp.next(1)
            if len(_wps) == 0:
                break
            wp = _wps[0]
            path.append(wp.transform.location)

        # 更新航点信息
        self.update_waypoint_navigation(path)

        # 检测障碍物
        self.detect_obstacles()

        # get forward speed
        velocity = self.ego.get_velocity()
        speed_m_s = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        # 计算当前速度（km/h）
        current_speed_kmh = speed_m_s * 3.6  # m/s to km/h

        # 记录速度历史（用于显示）
        self.speed_history.append(current_speed_kmh)
        if len(self.speed_history) > 100:  # 保留最近100帧的速度历史
            self.speed_history.pop(0)

        dt = 1 / Config.PYGAME_FPS

        # generate control signal
        control = carla.VehicleControl()

        # 智能速度控制逻辑
        if self.frame_count < 100:
            # 前100帧：全力加速
            control.throttle = 1.0  # 最大油门
            control.brake = 0.0
            self.brake_force = 0.0
            self.is_braking = False
            self.throttle_value = 1.0
        elif self.frame_count < 150:
            # 第100-150帧：维持油门，让速度继续上升
            control.throttle = 0.8
            control.brake = 0.0
            self.brake_force = 0.0
            self.is_braking = False
            self.throttle_value = 0.8
        else:
            # 150帧后：开始速度控制
            speed_error = current_speed_kmh - self.target_speed_kmh

            # 动态调整目标速度
            if current_speed_kmh > 45:  # 如果速度能到45以上，提高目标速度
                self.target_speed_kmh = 45

            # 根据转向角度调整目标速度
            if abs(self.steer_angle) > 0.1:  # 如果转向角度较大
                # 转弯时降低目标速度
                adjusted_target = self.target_speed_kmh * (1.0 - abs(self.steer_angle) * 2)
                speed_error = current_speed_kmh - adjusted_target
            else:
                speed_error = current_speed_kmh - self.target_speed_kmh

            # 根据前方障碍物调整速度
            if self.obstacles:
                # 如果有障碍物，进一步降低目标速度
                min_obstacle_distance = min(self.obstacles, key=lambda x: x['distance'])['distance']
                if min_obstacle_distance < 20.0:  # 20米内有障碍物
                    safety_factor = min_obstacle_distance / 20.0  # 距离越近，因子越小
                    self.target_speed_kmh = min(self.target_speed_kmh, 30.0 * safety_factor)
                    speed_error = current_speed_kmh - self.target_speed_kmh

            if speed_error > 5:  # 超过目标速度5km/h时强力刹车
                control.throttle = 0.0
                self.brake_force = min(self.brake_force + 0.1, 1.0)  # 快速增加刹车力度
                control.brake = self.brake_force
                self.is_braking = True
                self.throttle_value = 0.0
            elif speed_error > 2:  # 超过目标速度2km/h时轻微刹车
                control.throttle = 0.0
                self.brake_force = min(self.brake_force + 0.05, 0.5)  # 中等刹车
                control.brake = self.brake_force
                self.is_braking = True
                self.throttle_value = 0.0
            elif current_speed_kmh < self.target_speed_kmh - 5:  # 低于目标速度时全力加速
                control.throttle = 1.0
                self.brake_force = 0.0
                control.brake = 0.0
                self.is_braking = False
                self.throttle_value = 1.0
            elif current_speed_kmh < self.target_speed_kmh - 2:  # 接近目标速度但稍低
                control.throttle = 0.6
                self.brake_force = 0.0
                control.brake = 0.0
                self.is_braking = False
                self.throttle_value = 0.6
            else:  # 接近目标速度时维持
                control.throttle = 0.3
                self.brake_force = max(self.brake_force - 0.02, 0.0)  # 逐渐释放刹车
                control.brake = self.brake_force
                self.is_braking = self.brake_force > 0.05  # 只有刹车力度大于0.05时才显示刹车状态
                self.throttle_value = 0.3

        # 记录刹车状态历史（用于闪烁效果）
        self.brake_history.append(self.is_braking)
        if len(self.brake_history) > 20:  # 保持最近20帧的记录
            self.brake_history.pop(0)

        # MPC控制转向
        control.steer = self.mpc.run_step(path, speed_m_s, dt)
        self.steer_angle = control.steer  # 保存转向角度

        # 碰撞检测逻辑
        self.check_collision_warning(path, current_speed_kmh, self.steer_angle)

        # 计算驾驶评分
        self.calculate_driving_score(current_speed_kmh, self.steer_angle, self.brake_force, path)

        # 如果检测到碰撞风险，自动减速
        if self.collision_warning:
            # 紧急刹车
            control.throttle = 0.0
            control.brake = 0.7  # 中等刹车力度
            self.brake_force = 0.7
            self.is_braking = True
            self.throttle_value = 0.0
            print(f"碰撞警告！自动刹车，转向角度: {self.steer_angle:.3f}")

        # apply control signal
        self.ego.apply_control(control)

        # 在屏幕上显示所有信息
        self.drawer.display_speed(current_speed_kmh)
        self.drawer.display_brake_status(self.is_braking, self.brake_history, self.target_speed_kmh, self.frame_count)
        self.drawer.display_speed_history(self.speed_history, self.target_speed_kmh)
        self.drawer.display_steering(self.steer_angle)
        self.drawer.display_throttle_info(self.throttle_value, self.brake_force)
        self.drawer.display_control_mode(self.control_mode)
        self.drawer.display_frame_info(self.frame_count, dt)
        self.drawer.display_collision_warning(self.collision_warning, self.collision_history)
        self.drawer.display_driving_score(self.driving_score, self.score_factors, self.score_history)
        self.drawer.display_waypoint_navigation(
            self.current_waypoint_index,
            self.waypoint_positions,
            self.distance_to_waypoint,
            self.waypoint_reached_count,
            self.waypoint_progress
        )

        # 显示驾驶辅助线和雷达图
        self.drawer.display_driving_assist_lines(
            self.ego.get_location(),
            self.ego.get_transform(),
            self.steer_angle,
            path  # 传入路径用于绘制预期路径
        )

        # 显示雷达图（传入检测到的障碍物）
        self.drawer.display_simple_radar(self.ego.get_location(), self.obstacles)

    def check_collision_warning(self, path, speed_kmh, steer_angle):
        """检测可能的碰撞风险"""
        # 基于转向角度和速度的简单碰撞检测
        speed_factor = speed_kmh / 100.0  # 速度越快，风险越高
        steer_factor = abs(steer_angle)  # 转向角度越大，风险越高

        # 考虑障碍物距离
        obstacle_factor = 0.0
        if self.obstacles:
            min_obstacle_distance = min(self.obstacles, key=lambda x: x['distance'])['distance']
            if min_obstacle_distance < 10.0:
                obstacle_factor = 1.0 - (min_obstacle_distance / 10.0)

        # 计算碰撞风险
        collision_risk = speed_factor * (1.0 + steer_factor * 3) + obstacle_factor * 0.5

        # 检查是否超过阈值
        warning_threshold = 0.5
        was_warning = self.collision_warning
        self.collision_warning = collision_risk > warning_threshold

        # 记录警告历史
        self.collision_history.append(self.collision_warning)
        if len(self.collision_history) > 30:  # 保留最近30帧的记录
            self.collision_history.pop(0)

        # 如果状态改变，输出信息
        if self.collision_warning != was_warning:
            if self.collision_warning:
                print(f"碰撞警告激活！速度: {speed_kmh:.1f} km/h, 转向: {steer_angle:.3f}, 风险: {collision_risk:.2f}")
                if self.obstacles:
                    print(f"  最近障碍物距离: {min(self.obstacles, key=lambda x: x['distance'])['distance']:.1f}米")
            else:
                print("碰撞警告解除")

    def calculate_driving_score(self, current_speed, steer_angle, brake_force, path):
        """计算驾驶评分"""
        # 1. 速度稳定性评分 (权重30%)
        if len(self.speed_history) >= 10:
            recent_speeds = self.speed_history[-10:]
            speed_variance = np.var(recent_speeds) if len(recent_speeds) > 1 else 0
            # 速度变化越小，分数越高
            speed_stability = max(0, 100 - speed_variance * 5)
        else:
            speed_stability = 80  # 初始分数

        # 2. 转向平滑度评分 (权重25%)
        # 转向变化越小，分数越高
        if self.frame_count > 1:
            steer_variance = abs(steer_angle) * 50  # 转向角度越大，扣分越多
            steering_smoothness = max(0, 100 - steer_variance)
        else:
            steering_smoothness = 85

        # 3. 刹车使用评分 (权重20%)
        # 刹车使用越少，分数越高
        brake_usage = max(0, 100 - brake_force * 120)  # 刹车力度越大，扣分越多

        # 4. 路径跟踪评分 (权重15%)
        # 这里简化处理，使用转向角度作为路径跟踪的间接指标
        path_following = max(0, 100 - abs(steer_angle) * 40)

        # 5. 安全性评分 (权重10%)
        # 安全事件越少，分数越高
        safety_penalty = 0
        if self.collision_warning:
            safety_penalty += 30  # 碰撞警告扣分
        if brake_force > 0.5:
            safety_penalty += 20  # 紧急刹车扣分
        if self.obstacles and min(self.obstacles, key=lambda x: x['distance'])['distance'] < 5.0:
            safety_penalty += 30  # 距离障碍物太近扣分
        safety = max(0, 100 - safety_penalty)

        # 保存各项评分因子
        self.score_factors['speed_stability'] = speed_stability
        self.score_factors['steering_smoothness'] = steering_smoothness
        self.score_factors['brake_usage'] = brake_usage
        self.score_factors['path_following'] = path_following
        self.score_factors['safety'] = safety

        # 计算综合评分 (加权平均)
        weights = {
            'speed_stability': 0.30,
            'steering_smoothness': 0.25,
            'brake_usage': 0.20,
            'path_following': 0.15,
            'safety': 0.10
        }

        total_score = 0
        for factor, weight in weights.items():
            total_score += self.score_factors[factor] * weight

        # 应用平滑更新 (避免分数突变)
        self.driving_score = 0.7 * self.driving_score + 0.3 * total_score

        # 记录评分历史
        self.score_history.append(self.driving_score)
        if len(self.score_history) > 200:  # 保留最近200帧的评分历史
            self.score_history.pop(0)

        # 每100帧输出一次评分信息
        if self.frame_count % 100 == 0:
            print(f"\n=== 驾驶评分报告 (帧 {self.frame_count}) ===")
            print(f"综合评分: {self.driving_score:.1f}/100")
            print(f"速度稳定性: {speed_stability:.1f}")
            print(f"转向平滑度: {steering_smoothness:.1f}")
            print(f"刹车使用: {brake_usage:.1f}")
            print(f"路径跟踪: {path_following:.1f}")
            print(f"安全性: {safety:.1f}")
            print("=" * 40)

    def update_waypoint_navigation(self, path):
        """更新航点导航信息"""
        if len(path) == 0:
            return

        # 保存航点位置
        self.waypoint_positions = path

        # 计算车辆当前位置
        vehicle_location = self.ego.get_location()

        # 如果还没有设置当前航点，从第一个开始
        if self.current_waypoint_index >= len(self.waypoint_positions):
            self.current_waypoint_index = 0

        # 计算到当前航点的距离
        current_waypoint = self.waypoint_positions[self.current_waypoint_index]
        dx = current_waypoint.x - vehicle_location.x
        dy = current_waypoint.y - vehicle_location.y
        self.distance_to_waypoint = math.sqrt(dx * dx + dy * dy)

        # 检查是否到达当前航点（距离小于5米）
        waypoint_threshold = 5.0  # 到达阈值（米）
        if self.distance_to_waypoint < waypoint_threshold:
            # 到达当前航点，切换到下一个
            self.current_waypoint_index += 1
            self.waypoint_reached_count += 1

            # 如果到达所有航点，重新开始
            if self.current_waypoint_index >= len(self.waypoint_positions):
                self.current_waypoint_index = 0

            # 重新计算到新航点的距离
            if self.current_waypoint_index < len(self.waypoint_positions):
                new_waypoint = self.waypoint_positions[self.current_waypoint_index]
                dx = new_waypoint.x - vehicle_location.x
                dy = new_waypoint.y - vehicle_location.y
                self.distance_to_waypoint = math.sqrt(dx * dx + dy * dy)

            # 输出到达信息
            print(
                f"到达航点 #{self.waypoint_reached_count}，切换到航点 {self.current_waypoint_index + 1}/{len(self.waypoint_positions)}")

        # 计算航点进度（0-1）
        if len(self.waypoint_positions) > 0:
            self.waypoint_progress = self.current_waypoint_index / len(self.waypoint_positions)

    def detect_obstacles(self):
        """检测车辆周围的障碍物"""
        # 清空障碍物列表
        self.obstacles = []

        # 获取车辆位置和朝向
        vehicle_location = self.ego.get_location()
        vehicle_transform = self.ego.get_transform()
        vehicle_rotation = vehicle_transform.rotation

        # 获取车辆前方的航点作为参考方向
        wp = self.map.get_waypoint(vehicle_location)

        # 检测周围的车辆
        vehicle_list = self.world.get_actors().filter('vehicle.*')

        for vehicle in vehicle_list:
            # 排除自车
            if vehicle.id == self.ego.id:
                continue

            # 计算车辆距离
            other_location = vehicle.get_location()
            distance = vehicle_location.distance(other_location)

            # 只检测一定范围内的车辆
            if distance < self.obstacle_detection_range:
                # 计算相对方向
                dx = other_location.x - vehicle_location.x
                dy = other_location.y - vehicle_location.y

                # 计算相对于车辆前进方向的角度
                # 这里简化处理，使用航点方向作为参考
                forward_vector = vehicle_transform.get_forward_vector()
                relative_vector = carla.Vector3D(dx, dy, 0)

                # 计算点积和叉积
                dot_product = forward_vector.x * relative_vector.x + forward_vector.y * relative_vector.y
                cross_product = forward_vector.x * relative_vector.y - forward_vector.y * relative_vector.x

                # 计算角度（弧度）
                angle = math.atan2(cross_product, dot_product)

                # 转换为度
                angle_deg = math.degrees(angle)

                # 只考虑前方±60度范围内的障碍物
                if abs(angle_deg) < 60:
                    # 计算相对速度（简化）
                    other_velocity = vehicle.get_velocity()
                    relative_speed = math.sqrt(
                        (other_velocity.x - self.ego.get_velocity().x) ** 2 +
                        (other_velocity.y - self.ego.get_velocity().y) ** 2
                    )

                    # 添加到障碍物列表
                    self.obstacles.append({
                        'location': other_location,
                        'distance': distance,
                        'angle': angle_deg,
                        'relative_speed': relative_speed,
                        'type': 'vehicle'
                    })

        # 检测静态障碍物（简化：使用地图中的建筑）
        # 这里简化处理，实际上应该使用传感器数据
        if self.frame_count % 30 == 0:  # 每30帧检测一次静态障碍物
            # 随机添加一些模拟的静态障碍物用于演示
            import random
            for i in range(random.randint(0, 3)):
                # 在车辆前方随机位置添加模拟障碍物
                angle = random.uniform(-45, 45)
                distance = random.uniform(10, 40)

                # 计算障碍物位置
                angle_rad = math.radians(angle)
                obstacle_x = vehicle_location.x + distance * math.cos(angle_rad + math.radians(vehicle_rotation.yaw))
                obstacle_y = vehicle_location.y + distance * math.sin(angle_rad + math.radians(vehicle_rotation.yaw))
                obstacle_z = vehicle_location.z

                obstacle_location = carla.Location(x=obstacle_x, y=obstacle_y, z=obstacle_z)

                # 添加到障碍物列表
                self.obstacles.append({
                    'location': obstacle_location,
                    'distance': distance,
                    'angle': angle,
                    'relative_speed': 0.0,
                    'type': 'static'
                })

        # 按距离排序
        self.obstacles.sort(key=lambda x: x['distance'])

        # 保留最近的5个障碍物
        if len(self.obstacles) > 5:
            self.obstacles = self.obstacles[:5]

        # 每100帧输出一次障碍物信息
        if self.frame_count % 100 == 0 and self.obstacles:
            print(f"\n=== 障碍物检测报告 (帧 {self.frame_count}) ===")
            for i, obstacle in enumerate(self.obstacles):
                print(f"障碍物 {i + 1}: 距离={obstacle['distance']:.1f}米, 角度={obstacle['angle']:.1f}°, "
                      f"类型={obstacle['type']}, 相对速度={obstacle['relative_speed']:.1f} m/s")
            print("=" * 40)


if __name__ == '__main__':
    Main()