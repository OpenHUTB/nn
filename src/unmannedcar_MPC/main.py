#!/usr/bin/env python3

import carla
import config as Config
import math
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


if __name__ == '__main__':
    Main()