#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA 多交通灯版：Town04密集交通灯+状态循环+车辆响应
"""

import sys
import os
import carla
import numpy as np
import math
import pygame
import traceback
import time

# ===================== 全局配置 ======================
# CARLA连接
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

# 车辆配置
VEHICLE_MODEL = "vehicle.tesla.model3"
VEHICLE_WHEELBASE = 2.9
VEHICLE_REAR_AXLE_OFFSET = 1.45

# 转向控制
LOOKAHEAD_DIST_STRAIGHT = 7.0
LOOKAHEAD_DIST_CURVE = 4.0
STEER_GAIN_STRAIGHT = 0.7
STEER_GAIN_CURVE = 1.0
STEER_DEADZONE = 0.05
STEER_LOWPASS_ALPHA = 0.6
MAX_STEER = 1.0

# 弯道等级
DIR_CHANGE_GENTLE = 0.03
DIR_CHANGE_SHARP = 0.08

# 速度控制
BASE_SPEED = 25.0
PID_KP = 0.2
PID_KI = 0.01
PID_KD = 0.02

# 相机
CAMERA_POS = carla.Transform(carla.Location(x=-5.0, z=2.0))
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
CAMERA_FOV = 90

# 交通规则
TRAFFIC_LIGHT_STOP_DISTANCE = 4.0
TRAFFIC_LIGHT_DETECTION_RANGE = 50.0  # 扩大检测范围，能检测更多交通灯
TRAFFIC_LIGHT_ANGLE_THRESHOLD = 60.0  # 扩大检测角度
STOP_SPEED_THRESHOLD = 0.2
GREEN_LIGHT_ACCEL_FACTOR = 0.25
STOP_LINE_SIM_DISTANCE = 5.0

# 交通灯状态循环配置（单位：秒）
RED_LIGHT_DURATION = 3.0
GREEN_LIGHT_DURATION = 5.0
YELLOW_LIGHT_DURATION = 2.0

# 车道行驶配置
ROAD_DIRECTION_DOT_THRESHOLD = 0.0
LANE_KEEP_STRICTNESS = 1.2

# ===================== 核心兼容工具函数 ======================
def is_actor_alive(actor):
    """兼容不同版本的Actor存活状态判断"""
    try:
        return actor.is_alive()
    except TypeError:
        return actor.is_alive

def get_traffic_light_stop_line(traffic_light):
    """兼容不同版本的交通灯停止线获取"""
    try:
        return traffic_light.get_stop_line_location()
    except AttributeError:
        tl_transform = traffic_light.get_transform()
        forward_vec = tl_transform.get_forward_vector()
        stop_line_loc = tl_transform.location - forward_vec * STOP_LINE_SIM_DISTANCE
        stop_line_loc.z = tl_transform.location.z
        return stop_line_loc

def get_spawn_point_near_traffic_light(world, map):
    """自动查找距离交通灯最近的出生点"""
    traffic_lights = world.get_actors().filter("traffic.traffic_light")
    if not traffic_lights:
        print("警告：当前地图中未找到交通灯，使用默认出生点")
        spawn_points = map.get_spawn_points()
        return spawn_points[0] if spawn_points else carla.Transform()

    spawn_points = map.get_spawn_points()
    if not spawn_points:
        print("警告：未找到默认出生点，使用交通灯旁位置")
        tl_transform = traffic_lights[0].get_transform()
        return carla.Transform(tl_transform.location + carla.Location(x=-5.0, z=0.5), tl_transform.rotation)

    min_distance = float('inf')
    best_spawn_point = spawn_points[0]

    for spawn_point in spawn_points:
        for tl in traffic_lights:
            if not is_actor_alive(tl):
                continue
            tl_loc = tl.get_transform().location
            spawn_loc = spawn_point.location
            distance = math.sqrt((tl_loc.x - spawn_loc.x)**2 + (tl_loc.y - spawn_loc.y)**2)
            if distance < min_distance:
                min_distance = distance
                best_spawn_point = spawn_point

    print(f"找到距离交通灯最近的出生点，距离：{min_distance:.2f}米")
    return best_spawn_point

def cycle_traffic_light_states(world):
    """
    循环控制所有交通灯的状态：红→绿→黄→红
    作为后台线程运行，确保交通灯持续变化
    """
    while True:
        traffic_lights = world.get_actors().filter("traffic.traffic_light")
        # 设置所有交通灯为红灯
        for tl in traffic_lights:
            if is_actor_alive(tl):
                try:
                    tl.set_state(carla.TrafficLightState.Red)
                except:
                    pass
        time.sleep(RED_LIGHT_DURATION)

        # 设置所有交通灯为绿灯
        for tl in traffic_lights:
            if is_actor_alive(tl):
                try:
                    tl.set_state(carla.TrafficLightState.Green)
                except:
                    pass
        time.sleep(GREEN_LIGHT_DURATION)

        # 设置所有交通灯为黄灯
        for tl in traffic_lights:
            if is_actor_alive(tl):
                try:
                    tl.set_state(carla.TrafficLightState.Yellow)
                except:
                    pass
        time.sleep(YELLOW_LIGHT_DURATION)

# ===================== 纯追踪控制器 =====================
class AdaptivePurePursuit:
    def __init__(self, wheelbase):
        self.wheelbase = wheelbase
        self.last_steer = 0.0
        self.last_lookahead = LOOKAHEAD_DIST_STRAIGHT

    def calculate_steer(self, vehicle_transform, target_point, dir_change):
        # 1. 后轴位置
        forward_vec = vehicle_transform.get_forward_vector()
        rear_axle_loc = carla.Location(
            x=vehicle_transform.location.x - forward_vec.x * VEHICLE_REAR_AXLE_OFFSET,
            y=vehicle_transform.location.y - forward_vec.y * VEHICLE_REAR_AXLE_OFFSET,
            z=vehicle_transform.location.z
        )

        # 2. 车辆坐标系转换
        dx = target_point.x - rear_axle_loc.x
        dy = target_point.y - rear_axle_loc.y
        yaw = math.radians(vehicle_transform.rotation.yaw)

        dx_vehicle = dx * math.cos(yaw) + dy * math.sin(yaw)
        dy_vehicle = -dx * math.sin(yaw) + dy * math.cos(yaw)

        # 3. 转向增益
        steer_gain = np.interp(
            dir_change,
            [0, DIR_CHANGE_SHARP],
            [STEER_GAIN_STRAIGHT, STEER_GAIN_CURVE]
        )
        steer_gain = np.clip(steer_gain, STEER_GAIN_STRAIGHT, STEER_GAIN_CURVE)

        # 4. 纯追踪计算
        if dx_vehicle < 0.1:
            steer = self.last_steer
        else:
            steer_rad = math.atan2(2 * self.wheelbase * dy_vehicle * LANE_KEEP_STRICTNESS, dx_vehicle ** 2 + dy_vehicle ** 2)
            steer = steer_rad / math.pi
            steer *= steer_gain

        # 5. 死区+滤波
        if abs(steer) < STEER_DEADZONE:
            steer = 0.0
        steer = STEER_LOWPASS_ALPHA * steer + (1 - STEER_LOWPASS_ALPHA) * self.last_steer
        steer = np.clip(steer, -MAX_STEER, MAX_STEER)

        self.last_steer = steer
        return steer

    def get_adaptive_lookahead(self, dir_change):
        lookahead_dist = np.interp(
            dir_change,
            [0, DIR_CHANGE_SHARP],
            [LOOKAHEAD_DIST_STRAIGHT, LOOKAHEAD_DIST_CURVE]
        )
        lookahead_dist = np.clip(lookahead_dist, LOOKAHEAD_DIST_CURVE, LOOKAHEAD_DIST_STRAIGHT)
        self.last_lookahead = lookahead_dist
        return lookahead_dist

# ===================== 速度控制器 =====================
class SpeedController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = 0.0
        self.integral = 0.0

    def calculate(self, target_speed, current_speed):
        error = target_speed - current_speed
        p = self.kp * error
        self.integral += self.ki * error
        self.integral = np.clip(self.integral, -1.0, 1.0)
        i = self.integral
        d = self.kd * (error - self.last_error)
        self.last_error = error
        return np.clip(p + i + d, 0.0, 1.0)

# ===================== 交通灯管理类 ======================
class TrafficLightManager:
    def __init__(self):
        self.tracked_light = None
        self.is_stopped_at_red = False
        self.red_light_stop_time = 0

    def _calculate_angle_between_vehicle_and_light(self, vehicle_transform, light_transform):
        """计算车辆前进方向与交通灯的夹角"""
        vehicle_forward = vehicle_transform.get_forward_vector()
        vehicle_forward = np.array([vehicle_forward.x, vehicle_forward.y])
        vehicle_forward = vehicle_forward / np.linalg.norm(vehicle_forward)

        light_dir = light_transform.location - vehicle_transform.location
        light_dir = np.array([light_dir.x, light_dir.y])
        if np.linalg.norm(light_dir) < 0.1:
            return 0.0
        light_dir = light_dir / np.linalg.norm(light_dir)

        angle = math.acos(np.clip(np.dot(vehicle_forward, light_dir), -1.0, 1.0))
        angle = math.degrees(angle)
        return angle

    def get_lane_traffic_light(self, vehicle, world):
        """扩大检测范围，检测更多交通灯"""
        vehicle_transform = vehicle.get_transform()
        vehicle_loc = vehicle_transform.location

        # 检查跟踪的交通灯是否存活
        if self.tracked_light and is_actor_alive(self.tracked_light):
            dist = self.tracked_light.get_transform().location.distance(vehicle_loc)
            angle = self._calculate_angle_between_vehicle_and_light(vehicle_transform, self.tracked_light.get_transform())
            if dist < TRAFFIC_LIGHT_DETECTION_RANGE and angle < TRAFFIC_LIGHT_ANGLE_THRESHOLD:
                return self.tracked_light

        # 获取所有交通灯并筛选有效灯
        traffic_lights = world.get_actors().filter("traffic.traffic_light")
        valid_lights = []

        for light in traffic_lights:
            if not is_actor_alive(light):
                continue
            dist = light.get_transform().location.distance(vehicle_loc)
            if dist > TRAFFIC_LIGHT_DETECTION_RANGE:
                continue
            angle = self._calculate_angle_between_vehicle_and_light(vehicle_transform, light.get_transform())
            if angle < TRAFFIC_LIGHT_ANGLE_THRESHOLD:
                valid_lights.append((dist, light))

        if valid_lights:
            valid_lights.sort(key=lambda x: x[0])
            self.tracked_light = valid_lights[0][1]
            return self.tracked_light

        self.tracked_light = None
        return None

    def handle_traffic_light_logic(self, vehicle, current_speed, base_target_speed):
        """红灯强制停车，绿灯恢复行驶"""
        world = vehicle.get_world()
        traffic_light = self.get_lane_traffic_light(vehicle, world)

        if not traffic_light:
            self.is_stopped_at_red = False
            self.red_light_stop_time = 0
            return base_target_speed, "No Light (Lane)"

        # 核心：使用兼容函数获取停止线位置
        stop_line_loc = get_traffic_light_stop_line(traffic_light)
        dist_to_stop_line = vehicle.get_transform().location.distance(stop_line_loc)

        if traffic_light.get_state() == carla.TrafficLightState.Green:
            if self.is_stopped_at_red:
                recovery_speed = current_speed + (base_target_speed - current_speed) * GREEN_LIGHT_ACCEL_FACTOR
                target_speed = max(STOP_SPEED_THRESHOLD, recovery_speed)
                if abs(target_speed - base_target_speed) < 0.5:
                    self.is_stopped_at_red = False
                return target_speed, "Green (Recovering)"
            return base_target_speed, "Green (Lane)"

        elif traffic_light.get_state() == carla.TrafficLightState.Yellow:
            self.is_stopped_at_red = False
            yellow_speed = max(5.0, base_target_speed * 0.3)
            return yellow_speed, "Yellow (Stop Soon)"

        elif traffic_light.get_state() == carla.TrafficLightState.Red:
            if dist_to_stop_line > TRAFFIC_LIGHT_STOP_DISTANCE:
                self.is_stopped_at_red = False
                red_speed = max(2.0, current_speed * 0.1)
                return red_speed, f"Red (Decelerating: {dist_to_stop_line:.1f}m)"
            else:
                if current_speed <= STOP_SPEED_THRESHOLD:
                    self.is_stopped_at_red = True
                    self.red_light_stop_time += 1
                    wait_seconds = self.red_light_stop_time // 30
                    return 0.0, f"Red (Stopped: {wait_seconds}s)"
                else:
                    return 0.0, "Red (Emergency Stop)"

        return base_target_speed, "Unknown Light"

# ===================== 车道行驶辅助函数 ======================
def calculate_dir_change(current_wp):
    """计算方向变化量，判断弯道等级"""
    waypoints = [current_wp]
    for i in range(5):
        next_wps = waypoints[-1].next(1.0)
        if next_wps:
            waypoints.append(next_wps[0])
        else:
            break

    if len(waypoints) < 4:
        return 0.0, 0

    dirs = []
    for i in range(1, len(waypoints)):
        wp_prev = waypoints[i-1]
        wp_curr = waypoints[i]
        dir_rad = math.atan2(
            wp_curr.transform.location.y - wp_prev.transform.location.y,
            wp_curr.transform.location.x - wp_prev.transform.location.x
        )
        dirs.append(dir_rad)

    dir_change = 0.0
    for i in range(1, len(dirs)):
        dir_change += abs(dirs[i] - dirs[i-1]) * 2

    if dir_change < DIR_CHANGE_GENTLE:
        curve_level = 0
    elif dir_change < DIR_CHANGE_SHARP:
        curve_level = 1
    else:
        curve_level = 2

    return dir_change, curve_level

def get_forward_waypoint(vehicle, map):
    """获取车辆当前车道的前进方向路点（极低版本兼容）"""
    vehicle_transform = vehicle.get_transform()
    # 1. 投影到道路
    current_wp = map.get_waypoint(
        vehicle_transform.location,
        project_to_road=True
    )

    # 2. 纯数学判断方向是否相反
    road_direction = current_wp.transform.get_forward_vector()
    vehicle_direction = vehicle_transform.get_forward_vector()
    dot_product = road_direction.x * vehicle_direction.x + road_direction.y * vehicle_direction.y

    # 3. 方向相反则取前方点
    if dot_product < ROAD_DIRECTION_DOT_THRESHOLD:
        forward_wps = current_wp.next(10.0)
        if forward_wps:
            current_wp = forward_wps[0]
        else:
            current_wp = map.get_waypoint(
                vehicle_transform.location + vehicle_direction * 5.0,
                project_to_road=True
            )

    return current_wp

# ===================== 相机管理器 =====================
class CameraManager:
    def __init__(self, world, vehicle, display):
        self.world = world
        self.vehicle = vehicle
        self.display = display
        self.camera = None
        self.traffic_light_status = "No Light"
        self._create_camera()

    def _create_camera(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        bp.set_attribute("fov", str(CAMERA_FOV))
        self.camera = self.world.spawn_actor(bp, CAMERA_POS, attach_to=self.vehicle)
        self.camera.listen(self._on_image)

    def _on_image(self, image):
        """修正数组处理逻辑"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1].swapaxes(0, 1)
        self.display.blit(pygame.surfarray.make_surface(array), (0, 0))
        self._draw_info()
        pygame.display.flip()

    def _draw_info(self):
        font = pygame.font.SysFont("Arial", 24, bold=True)
        if "Red" in self.traffic_light_status:
            color = (255, 0, 0)
        elif "Green" in self.traffic_light_status:
            color = (0, 255, 0)
        elif "Yellow" in self.traffic_light_status:
            color = (255, 255, 0)
        else:
            color = (255, 255, 255)

        text = font.render(f"Traffic Light: {self.traffic_light_status}", True, color)
        bg = pygame.Surface((text.get_width() + 10, text.get_height() + 5))
        bg.fill((0, 0, 0))
        self.display.blit(bg, (5, 5))
        self.display.blit(text, (10, 7))

    def update_traffic_light_status(self, status):
        self.traffic_light_status = status

    def destroy(self):
        if self.camera:
            if is_actor_alive(self.camera):
                self.camera.stop()
            self.camera.destroy()

# ===================== 主函数（核心：多交通灯配置）=====================
def main():
    pygame.init()
    display = pygame.display.set_mode((CAMERA_WIDTH, CAMERA_HEIGHT))
    pygame.display.set_caption("CARLA 多交通灯版（Town04密集交通灯+状态循环）")

    client = None
    world = None
    vehicle = None
    camera_manager = None
    pp_controller = None
    speed_controller = None
    traffic_light_manager = None

    try:
        # 1. 连接CARLA，加载**Town04**（交通灯最密集的地图）
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(CARLA_TIMEOUT)
        try:
            client.load_world("Town04")  # 替换为Town04，交通灯数量远多于Town03
        except:
            world = client.get_world()
            print("警告：Town04地图不存在，使用当前地图")
        else:
            world = client.get_world()
            print("成功加载Town04地图（交通灯密集）")
        map = world.get_map()

        # 2. 清理残留演员
        for actor in world.get_actors():
            if actor.type_id.startswith(("vehicle.", "walker.", "sensor.", "controller.")):
                if is_actor_alive(actor):
                    actor.destroy()
        print("残留演员清理完成")

        # 3. 启动交通灯状态循环线程（后台持续切换交通灯状态）
        import threading
        tl_cycle_thread = threading.Thread(target=cycle_traffic_light_states, args=(world,), daemon=True)
        tl_cycle_thread.start()
        print("交通灯状态循环线程已启动（红3秒→绿5秒→黄2秒）")

        # 4. 设置车辆起始位置（Town04交通灯密集区）
        vehicle_bp = world.get_blueprint_library().find(VEHICLE_MODEL)
        
        # 手动指定Town04的核心交通灯路口坐标（经测试：多个交通灯环绕）
        spawn_point = carla.Transform(
            carla.Location(x=220.0, y=150.0, z=0.5),  # Town04核心交通灯密集区
            carla.Rotation(yaw=90.0)
        )
        spawn_point.location.z += 0.2
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        # 备用方案：自动查找交通灯附近的出生点
        if not vehicle:
            print("手动坐标生成失败，自动查找交通灯附近的出生点...")
            spawn_point = get_spawn_point_near_traffic_light(world, map)
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        # 最终备用方案
        if not vehicle:
            spawn_points = map.get_spawn_points()
            spawn_point = spawn_points[0] if spawn_points else carla.Transform()
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        print(f"车辆生成成功：{vehicle.type_id}（起始于交通灯密集区）")

        # 5. 初始化组件
        pp_controller = AdaptivePurePursuit(VEHICLE_WHEELBASE)
        speed_controller = SpeedController(PID_KP, PID_KI, PID_KD)
        traffic_light_manager = TrafficLightManager()
        camera_manager = CameraManager(world, vehicle, display)

        # 6. 主循环
        print("仿真启动，按ESC退出...（车辆将经过多个交通灯）")
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            # 获取车辆状态
            vehicle_transform = vehicle.get_transform()
            vehicle_vel = vehicle.get_velocity()
            current_speed = math.hypot(vehicle_vel.x, vehicle_vel.y) * 3.6

            # 核心：获取前进方向路点
            current_wp = get_forward_waypoint(vehicle, map)

            # 计算方向变化
            dir_change, curve_level = calculate_dir_change(current_wp)

            # 预瞄距离
            lookahead_dist = pp_controller.get_adaptive_lookahead(dir_change)

            # 目标路点
            target_wps = current_wp.next(lookahead_dist)
            target_point = target_wps[0].transform.location if target_wps else vehicle_transform.location

            # 基础速度
            curve_speed_factors = [1.0, 0.7, 0.4]
            speed_factor = curve_speed_factors[min(curve_level, 2)]
            base_target_speed = BASE_SPEED * speed_factor
            base_target_speed = max(8.0, base_target_speed)

            # 交通灯处理（检测多个交通灯）
            target_speed, traffic_light_status = traffic_light_manager.handle_traffic_light_logic(
                vehicle, current_speed, base_target_speed
            )
            camera_manager.update_traffic_light_status(traffic_light_status)

            # 控制计算
            steer = pp_controller.calculate_steer(vehicle_transform, target_point, dir_change)
            throttle = speed_controller.calculate(target_speed, current_speed)
            brake = 1.0 - throttle if current_speed > target_speed + 1 else 0.0

            # 红灯刹车
            if "Red (Stopped)" in traffic_light_status or target_speed == 0.0:
                throttle = 0.0
                brake = 1.0

            # 应用控制
            control = carla.VehicleControl()
            control.steer = steer
            control.throttle = throttle
            control.brake = brake
            vehicle.apply_control(control)

            # 打印状态（显示当前交通灯状态）
            curve_names = ["直道", "缓弯", "急弯"]
            lane_id = current_wp.lane_id
            print(f"速度：{current_speed:5.1f}km/h | 目标：{target_speed:5.1f} | 弯道：{curve_names[curve_level]:<3} | 车道ID：{lane_id} | 灯状态：{traffic_light_status}")

            clock.tick(30)

    except Exception as e:
        print(f"错误：{e}")
        traceback.print_exc()

    finally:
        print("清理资源...")
        if camera_manager:
            camera_manager.destroy()
        if vehicle and is_actor_alive(vehicle):
            vehicle.destroy()
        pygame.quit()
        print("仿真结束")

if __name__ == "__main__":
    main()