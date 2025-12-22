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

    def check_collision_warning(self, path, speed_kmh, steer_angle):
        """检测可能的碰撞风险"""
        # 基于转向角度和速度的简单碰撞检测
        speed_factor = speed_kmh / 100.0  # 速度越快，风险越高
        steer_factor = abs(steer_angle)  # 转向角度越大，风险越高

        # 计算碰撞风险
        collision_risk = speed_factor * (1.0 + steer_factor * 3)

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


if __name__ == '__main__':
    Main()