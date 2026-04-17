#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function
import time
import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref


# ==============================================================================
# -- 导入RL模块 ----------------------------------------------------------------
# ==============================================================================

try:
    from rl_agent import (
        CARLARLEnvironment, DQNAgent, PPOAgent,
        RLTrainer
    )
    RL_AVAILABLE = True
    print("✓ RL模块导入成功")
except ImportError as e:
    print(f"⚠ RL模块导入失败: {e}")
    RL_AVAILABLE = False
    CARLARLEnvironment = None
    DQNAgent = None
    PPOAgent = None
    RLTrainer = None



try:
    import pygame
    from pygame.locals import *
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================

# 添加 CARLA Python API 路径（使用你的本地路径）
carla_base = 'H:/carla0.9.15/WindowsNoEditor/PythonAPI/carla'
egg_dir = os.path.join(carla_base, 'dist')

# 添加 egg 文件
try:
    egg_files = glob.glob(os.path.join(egg_dir, 'carla-*.egg'))
    if egg_files:
        sys.path.append(egg_files[0])
        print(f"✓ 已添加 CARLA egg: {os.path.basename(egg_files[0])}")
    else:
        print(f"⚠ 警告: 未找到 CARLA egg 文件: {egg_dir}")
except Exception as e:
    print(f"⚠ 添加 egg 文件时出错: {e}")

# 添加 agents 模块路径
if carla_base not in sys.path:
    sys.path.append(carla_base)
    print(f"✓ 已添加 agents 路径: {carla_base}")

# ==============================================================================
# -- 修复 CARLA 0.9.12 agents 模块缺失的函数 -------------------------------------
# ==============================================================================

try:
    import agents.tools.misc as misc_module

    if not hasattr(misc_module, 'is_within_distance_ahead'):
        def is_within_distance_ahead(transform, location, distance):
            forward_vector = transform.get_forward_vector()
            relative_vector = location - transform.location
            dot_product = forward_vector.x * relative_vector.x + forward_vector.y * relative_vector.y
            return dot_product > 0 and relative_vector.length() < distance
        misc_module.is_within_distance_ahead = is_within_distance_ahead
        print("✓ 已添加缺失函数: is_within_distance_ahead")

    if not hasattr(misc_module, 'is_within_distance'):
        def is_within_distance(target_transform, reference_transform, max_distance):
            return target_transform.location.distance(reference_transform.location) < max_distance
        misc_module.is_within_distance = is_within_distance
        print("✓ 已添加缺失函数: is_within_distance")

    if not hasattr(misc_module, 'compute_distance'):
        def compute_distance(location1, location2):
            return location1.distance(location2)
        misc_module.compute_distance = compute_distance
        print("✓ 已添加缺失函数: compute_distance")

except ImportError as e:
    print(f"⚠ 无法导入 agents.tools.misc: {e}")

print("=" * 60)

# ==============================================================================
# -- Import CARLA and Agents modules -------------------------------------------
# ==============================================================================

try:
    import carla
    from carla import ColorConverter as cc
    print("✓ CARLA 模块导入成功")
except ImportError as e:
    print(f"✗ CARLA 模块导入失败: {e}")
    sys.exit(1)

try:
    from agents.navigation.behavior_agent import BehaviorAgent
    from agents.navigation.roaming_agent import RoamingAgent
    from agents.navigation.basic_agent import BasicAgent
    print("✓ Agents 模块导入成功")
    print("  - BehaviorAgent")
    print("  - RoamingAgent")
    print("  - BasicAgent")
except ImportError as e:
    print(f"✗ Agents 模块导入失败: {e}")
    print("请检查 agents 文件夹是否存在于: " + carla_base)
    sys.exit(1)

print("=" * 60)
print("所有模块导入完成，开始启动程序...")
print("=" * 60)
print()


# ==============================================================================
# -- 辅助驾驶系统 (修复版) -------------------------------------------------------
# ==============================================================================

class AssistedDrivingSystem:
    """鲁棒版自动避障系统 - 修复ActorList合并错误"""

    def __init__(self, vehicle, world, hud):
        self.vehicle = vehicle
        self.world = world
        self.hud = hud
        self.enabled = True
        self.obstacle_avoidance_enabled = True

        # 避障参数
        self.emergency_brake_distance = 10.0
        self.takeover_distance = 30.0
        self.safe_distance = 40.0

        # 状态机
        self.is_taking_over = False
        self.takeover_start_time = 0
        self.takeover_duration = 3.0
        self.avoidance_direction = 0
        self.avoidance_phase = "none"

        # 控制平滑
        self.current_steer = 0.0
        self.steer_smooth = 0.3

        # 调试
        self._debug_print = True

    def toggle(self):
        self.enabled = not self.enabled
        status = "ON" if self.enabled else "OFF"
        self.hud.notification(f"Assist {status}", seconds=2.0,
                              color=(0, 255, 0) if self.enabled else (255, 0, 0))
        return self.enabled

    def toggle_obstacle_avoidance(self):
        self.obstacle_avoidance_enabled = not self.obstacle_avoidance_enabled
        status = "ON" if self.obstacle_avoidance_enabled else "OFF"
        self.hud.notification(f"Avoidance {status}", seconds=2.0,
                              color=(0, 255, 0) if self.obstacle_avoidance_enabled else (255, 0, 0))
        return self.obstacle_avoidance_enabled

    def get_dynamic_distance(self):
        v = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        if speed > 70:
            return 50.0
        elif speed > 40:
            return 40.0
        else:
            return 30.0

    def get_front_obstacle(self):
        """检测前方障碍物，返回 (距离, 角度, actor)"""
        vehicle_transform = self.vehicle.get_transform()
        veh_loc = vehicle_transform.location
        forward = vehicle_transform.get_forward_vector()
        right = vehicle_transform.get_right_vector()

        check_dist = self.get_dynamic_distance()
        min_dist = check_dist
        min_angle = 0.0
        hit_actor = None

        # 获取所有 actors 并过滤（修复 ActorList 相加错误）
        all_actors = self.world.get_actors()
        for actor in all_actors:
            if actor.id == self.vehicle.id:
                continue
            # 过滤类型
            actor_type = actor.type_id.lower()
            if not any(x in actor_type for x in
                       ['vehicle', 'walker', 'static', 'building', 'pole', 'tree', 'wall', 'barrier']):
                continue

            loc = actor.get_location()
            dx = loc.x - veh_loc.x
            dy = loc.y - veh_loc.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > check_dist or dist < 1.0:
                continue

            forward_dot = (dx*forward.x + dy*forward.y) / dist
            if forward_dot < 0.707:  # 前方140度视野
                continue

            right_dot = (dx*right.x + dy*right.y) / dist
            angle = math.degrees(math.asin(max(-1.0, min(1.0, right_dot))))

            # 考虑障碍物宽度
            bbox_extent = 1.0
            if hasattr(actor, 'bounding_box'):
                bbox_extent = actor.bounding_box.extent.y

            lateral_offset = abs(dist * math.sin(math.radians(angle)))
            safe_margin = self.vehicle.bounding_box.extent.y + bbox_extent + 0.8

            if lateral_offset < safe_margin and dist < min_dist:
                min_dist = dist
                min_angle = angle
                hit_actor = actor

        return min_dist, min_angle, hit_actor

    def check_side_space(self, direction):
        """检查侧向是否有车道空间（基于waypoint）"""
        try:
            waypoint = self.world.get_map().get_waypoint(
                self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            if direction == -1:
                left = waypoint.get_left_lane()
                return left is not None and left.lane_type == carla.LaneType.Driving
            else:
                right = waypoint.get_right_lane()
                return right is not None and right.lane_type == carla.LaneType.Driving
        except:
            return True  # 默认安全

    def decide_direction(self, obstacle_angle):
        left_ok = self.check_side_space(-1)
        right_ok = self.check_side_space(1)
        if left_ok and right_ok:
            return -1 if obstacle_angle > 0 else 1
        elif left_ok:
            return -1
        elif right_ok:
            return 1
        else:
            return -1 if random.random() > 0.5 else 1

    def apply_assistance(self, control):
        if not self.enabled or not self.obstacle_avoidance_enabled:
            self.is_taking_over = False
            return control

        current_time = time.time()
        distance, angle, actor = self.get_front_obstacle()
        emergency = distance < self.emergency_brake_distance

        if actor and distance < self.safe_distance and self._debug_print:
            print(f"[DETECT] {actor.type_id} at {distance:.1f}m, angle={angle:.1f}°")

        need_takeover = (distance < self.takeover_distance) and actor is not None

        # 开始接管
        if not self.is_taking_over and (need_takeover or emergency):
            self.is_taking_over = True
            self.takeover_start_time = current_time
            self.avoidance_phase = "avoiding"
            self.avoidance_direction = self.decide_direction(angle)
            dir_str = "LEFT" if self.avoidance_direction == -1 else "RIGHT"
            self.hud.notification(f"⚠ AVOID {dir_str}!", seconds=1.5, color=(255, 50, 50))
            print(f"[TAKEOVER] Start avoiding {dir_str}, distance={distance:.1f}m")

        if self.is_taking_over:
            takeover_time = current_time - self.takeover_start_time

            should_end = False
            if distance > self.takeover_distance + 5.0:
                should_end = True
            elif takeover_time > self.takeover_duration:
                should_end = True
            elif emergency and distance > self.emergency_brake_distance + 2.0:
                should_end = True

            if should_end:
                self.is_taking_over = False
                self.avoidance_phase = "none"
                self.current_steer = 0.0
                self.hud.notification("✓ Control returned", seconds=1.0, color=(0, 255, 0))
                print("[TAKEOVER] Control returned")
                return control

            # 执行控制
            if emergency:
                control.brake = 1.0
                control.throttle = 0.0
                control.steer = self.avoidance_direction * 0.6
                self.hud.notification("EMERGENCY BRAKE!", seconds=0.3, color=(255, 0, 0))
            else:
                brake_ratio = 1.0 - (distance - self.emergency_brake_distance) / (self.takeover_distance - self.emergency_brake_distance)
                brake_ratio = max(0.2, min(0.8, brake_ratio))
                control.brake = brake_ratio
                control.throttle = 0.1

                base_steer = 0.8
                if distance < 15:
                    base_steer = 1.0
                target_steer = self.avoidance_direction * base_steer

                self.current_steer += (target_steer - self.current_steer) * self.steer_smooth
                control.steer = self.current_steer

                self.hud.notification(f"AVOID {distance:.1f}m", seconds=0.2, color=(255, 150, 0))

        return control

    def detect_obstacles(self):
        """保持向后兼容"""
        dist, angle, actor = self.get_front_obstacle()
        emergency = dist < self.emergency_brake_distance
        obs_type = "none"
        if actor:
            t = actor.type_id.lower()
            if 'vehicle' in t:
                obs_type = "vehicle"
            elif 'walker' in t:
                obs_type = "pedestrian"
            else:
                obs_type = "static"
        return dist, angle, emergency, obs_type


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self.assisted_driving = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.rl_env = None
        self.rl_agent = None
        self.rl_trainer = None
        self.rl_enabled = getattr(args, 'rl_mode', False)
        self.rl_training = getattr(args, 'rl_train', False)

    def restart(self, args):
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0
        if args.seed is not None:
            random.seed(args.seed)

        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        print("Spawning the player")
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        self.assisted_driving = AssistedDrivingSystem(self.player, self.world, self.hud)

        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("按 H 键查看帮助", seconds=4.0)
        self.world = world

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_r:
                    if self.world.assisted_driving:
                        self.world.assisted_driving.toggle()
                elif event.key == K_t:
                    if self.world.assisted_driving:
                        self.world.assisted_driving.toggle_obstacle_avoidance()
                elif event.key == K_h:
                    self.world.hud.help.toggle()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._font_large = pygame.font.Font(mono, 18)

        try:
            chinese_fonts = ['simsun', 'simhei', 'microsoft yahei', 'fang song', 'kaiti']
            for cf in chinese_fonts:
                cf_path = pygame.font.match_font(cf)
                if cf_path:
                    self._font_chinese = pygame.font.Font(cf_path, 14)
                    break
            else:
                self._font_chinese = self._font_mono
        except:
            self._font_chinese = self._font_mono

        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 20), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()

        speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        heading = ''
        if abs(transform.rotation.yaw) < 89.5:
            heading += '北'
        if abs(transform.rotation.yaw) > 90.5:
            heading += '南'
        if 179.5 > transform.rotation.yaw > 0.5:
            heading += '东'
        if -0.5 > transform.rotation.yaw > -179.5:
            heading += '西'

        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        assist_status = "✓开启" if world.assisted_driving and world.assisted_driving.enabled else "✗关闭"
        obstacle_status = "✓开启" if world.assisted_driving and world.assisted_driving.obstacle_avoidance_enabled else "✗关闭"

        gear_display = ""
        if control.reverse:
            gear_display = "倒车档 R"
        elif control.gear == 0:
            gear_display = "空档 N"
        elif control.gear > 0:
            gear_display = f"{control.gear} 档"
        else:
            gear_display = "前进档 D"

        obstacle_info = ""
        if world.assisted_driving and world.assisted_driving.enabled:
            result = world.assisted_driving.detect_obstacles()
            if len(result) == 4:
                distance, angle, emergency, obs_type = result
            elif len(result) == 3:
                distance, angle, emergency = result
            else:
                distance, angle, emergency = 20.0, 0, False
            if distance < 15:
                obstacle_info = f" | 障碍物: {distance:.1f}m"

        self._info_text = [
            f'服务器 FPS: {self.server_fps:16.0f}',
            f'客户端 FPS: {clock.get_fps():16.0f}',
            '',
            f'车辆: {get_actor_display_name(world.player, truncate=20):20s}',
            f'地图: {world.map.name:20s}',
            f'模拟时间: {str(datetime.timedelta(seconds=int(self.simulation_time))):12s}',
            '',
            f'速度: {speed:15.0f} km/h',
            f'方向: {transform.rotation.yaw:16.0f}° {heading:2s}',
            f'位置: ({transform.location.x:5.1f}, {transform.location.y:5.1f})',
            f'档位: {gear_display:20s}',
            f'高度: {transform.location.z:18.0f} m',
            '',
            f'[R]辅助驾驶: {assist_status}  [T]自动避障: {obstacle_status}{obstacle_info}',
            '']

        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('油门:', control.throttle, 0.0, 1.0),
                ('转向:', control.steer, -1.0, 1.0),
                ('刹车:', control.brake, 0.0, 1.0),
                ('倒车:', control.reverse),
                ('手刹:', control.hand_brake),
                ('手动档:', control.manual_gear_shift),
                f'档位: {gear_display}']
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('速度:', control.speed, 0.0, 5.556),
                ('跳跃:', control.jump)]
        self._info_text += [
            '',
            '碰撞历史:',
            collision,
            '',
            f'附近车辆数: {len(vehicles):8d}']

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x) ** 2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z) ** 2)

        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist_val, vehicle in sorted(vehicles):
            if dist_val > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append(f'{int(dist_val):4d}m {vehicle_type}')

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0, color=(255, 255, 255)):
        self._notifications.set_text(text, seconds=seconds, color=color)

    def error(self, text):
        self._notifications.set_text(f'错误: {text}', (255, 0, 0))

    def render(self, display):
        self.help.render(display)

        if self._show_info:
            info_surface = pygame.Surface((300, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18

        self.render_keyboard_help(display)
        self._notifications.render(display)

    def render_keyboard_help(self, display):
        help_y = self.dim[1] - 160
        line_height = 18

        title = self._font_large.render("=== 按键说明 ===", True, (255, 255, 0))
        display.blit(title, (10, help_y))

        tips = [
            ("F", "切换自动驾驶/手动", (0, 255, 0)),
            ("R", "开关辅助驾驶", (0, 255, 0)),
            ("T", "开关自动避障", (0, 255, 0)),
            ("H", "显示/隐藏帮助", (0, 255, 0)),
            ("", "", None),
            ("W / ↑", "前进 / 加速", (255, 255, 255)),
            ("S / ↓", "刹车 / 减速", (255, 255, 255)),
            ("A / ←", "左转", (255, 255, 255)),
            ("D / →", "右转", (255, 255, 255)),
            ("", "", None),
            ("空格", "倒车（按住）", (255, 200, 100)),
            ("", "", None),
            ("ESC / Ctrl+Q", "退出程序", (255, 100, 100)),
        ]

        for i, (key, desc, color) in enumerate(tips):
            if not key:
                continue
            y_pos = help_y + 22 + i * line_height
            if y_pos > self.dim[1] - 10:
                break
            key_surface = self._font_mono.render(f"[{key}]", True, color if color else (255, 255, 0))
            desc_surface = self._font_mono.render(desc, True, (200, 200, 200))
            display.blit(key_surface, (15, y_pos))
            display.blit(desc_surface, (85, y_pos))


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================

class HelpText(object):
    def __init__(self, font, width, height):
        help_lines = [
            "==================== 帮助菜单 ====================",
            "",
            "驾驶模式:",
            "  F - 切换全自动驾驶 / 手动驾驶",
            "",
            "辅助功能 (手动模式下有效):",
            "  R - 开启/关闭辅助驾驶系统",
            "  T - 开启/关闭自动避障",
            "",
            "手动驾驶控制:",
            "  W / ↑ - 前进 / 加速",
            "  S / ↓ - 刹车 / 减速",
            "  A / ← - 左转",
            "  D / → - 右转",
            "  空格键 - 倒车（按住）",
            "",
            "其他功能:",
            "  H - 显示/隐藏本菜单",
            "  ESC / Ctrl+Q - 退出程序",
            "",
            "================================================="
        ]

        self.font = font
        self.dim = (500, len(help_lines) * 22 + 20)
        self.pos = (width - self.dim[0] - 20, 20)
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0))
        self.surface.set_alpha(200)

        for i, line in enumerate(help_lines):
            text_color = (255, 255, 0) if "====" in line else (255, 255, 255)
            text_texture = self.font.render(line, True, text_color)
            self.surface.blit(text_texture, (10, i * 22 + 5))

        self._render = False

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification(f'碰撞: {actor_type}', seconds=1.0, color=(255, 0, 0))
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================

class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification(f'压线: {" ".join(text)}', seconds=0.5, color=(255, 255, 0))


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================

class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
                if blp.has_attribute('gamma'):
                    blp.set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
                force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    tot_target_reached = 0
    num_min_waypoints = 21
    autopilot_enabled = True
    last_f_key_time = 0

    rl_enabled = getattr(args, 'rl_mode', False) and RL_AVAILABLE
    rl_training = getattr(args, 'rl_train', False) and rl_enabled
    rl_episode_count = 0
    rl_episode_reward = 0
    rl_step_count = 0

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        if rl_enabled:
            world.rl_env = CARLARLEnvironment(world.player, world.world, hud, world.assisted_driving)
            state_dim = world.rl_env.state_dim
            action_dim = world.rl_env.action_dim
            algorithm = getattr(args, 'rl_algorithm', 'ppo')

            if algorithm == 'dqn':
                world.rl_agent = DQNAgent(state_dim, action_dim)
                world.rl_env.continuous_actions = False
                hud.notification("RL Agent: DQN", seconds=2.0, color=(0, 255, 255))
            else:
                world.rl_agent = PPOAgent(state_dim, action_dim, continuous=True)
                world.rl_env.continuous_actions = True
                hud.notification("RL Agent: PPO", seconds=2.0, color=(0, 255, 255))

            model_path = getattr(args, 'rl_model', None)
            if model_path and os.path.exists(model_path):
                world.rl_agent.load(model_path)
                hud.notification(f"RL Model Loaded", seconds=2.0, color=(0, 255, 0))

            if rl_training:
                # 每帧强制禁用辅助驾驶
                if world.assisted_driving:
                    world.assisted_driving.enabled = False
                    world.assisted_driving.is_taking_over = False
                    hud.notification("Assist DISABLED for training", seconds=2.0, color=(255, 150, 0))

                save_dir = getattr(args, 'rl_save_dir', './rl_checkpoints')
                world.rl_trainer = RLTrainer(world.rl_env, world.rl_agent, save_dir)
                hud.notification("RL Training Mode", seconds=3.0, color=(255, 255, 0))

                spawn_points = world.map.get_spawn_points()
                if spawn_points:
                    # --- 物理卡死修复（与测试阶段保持一致）---
                    new_transform = random.choice(spawn_points)
                    new_transform.location.z += 1.5
                    new_transform.rotation.yaw = random.uniform(0, 360)
                    world.player.set_transform(new_transform)

                    # 强制松开所有控制
                    world.player.apply_control(carla.VehicleControl())

                    # 选择远距离目标
                    vehicle_loc = new_transform.location
                    far_spawns = [sp for sp in spawn_points
                                  if sp.location.distance(vehicle_loc) > 30.0]
                    if far_spawns:
                        target_point = random.choice(far_spawns)
                    else:
                        target_point = random.choice(spawn_points)
                    world.rl_env.set_target(target_point.location)
                    print(f"New episode target distance: {vehicle_loc.distance(target_point.location):.1f}m")

            world.rl_enabled = True

        if args.agent == "Roaming":
            agent = RoamingAgent(world.player)
        elif args.agent == "Basic":
            agent = BasicAgent(world.player)
            spawn_point = world.map.get_spawn_points()[0]
            agent.set_destination((spawn_point.location.x, spawn_point.location.y, spawn_point.location.z))
        else:
            agent = BehaviorAgent(world.player, behavior=args.behavior)
            spawn_points = world.map.get_spawn_points()
            random.shuffle(spawn_points)
            if spawn_points[0].location != world.player.get_location():
                destination = spawn_points[0].location
            else:
                destination = spawn_points[1].location
            agent.set_destination(destination)

        clock = pygame.time.Clock()

        hud.notification("=== DRIVING MODES ===", seconds=5.0)
        hud.notification("F: Toggle Auto/Manual", seconds=5.0)
        hud.notification("R: Assist ON/OFF", seconds=5.0)
        hud.notification("T: Avoidance ON/OFF", seconds=5.0)
        hud.notification("SPACE: Reverse", seconds=5.0)

        if rl_enabled:
            mode_str = "RL TRAIN" if rl_training else "RL INFERENCE"
            hud.notification(f"=== {mode_str} ===", seconds=3.0, color=(255, 0, 255))
            if rl_training:
                hud.notification("L: Save checkpoint", seconds=3.0)

        # 可选：训练前先测试一下车辆是否能动
        if rl_training:
            print("Testing vehicle control for 2 seconds...")
            # 先确保车辆处于安全状态
            spawn_points = world.map.get_spawn_points()
            if spawn_points:
                safe_transform = random.choice(spawn_points)
                safe_transform.location.z += 1.5
                world.player.set_transform(safe_transform)
            test_control = carla.VehicleControl()
            test_control.throttle = 0.8
            test_control.steer = 0.0
            test_control.hand_brake = False
            test_control.reverse = False
            world.player.apply_control(test_control)
            for i in range(40):
                world.world.wait_for_tick(10.0)
                world.tick(clock)
                world.render(display)
                pygame.display.flip()
                if i % 10 == 0:
                    vel = world.player.get_velocity()
                    speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
                    print(f"Test step {i}: speed={speed:.2f} km/h")
            print("Test complete.")

        while True:
            clock.tick_busy_loop(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or (event.key == K_q and pygame.key.get_mods() & KMOD_CTRL):
                        return
                    elif event.key == K_f:
                        current_time = time.time()
                        if current_time - last_f_key_time > 0.3:
                            last_f_key_time = current_time
                            autopilot_enabled = not autopilot_enabled
                            mode = "AUTO MODE" if autopilot_enabled else "MANUAL MODE"
                            hud.notification(mode, seconds=2.0,
                                             color=(0, 255, 0) if autopilot_enabled else (255, 255, 0))
                            if world.assisted_driving:
                                world.assisted_driving.is_taking_over = False
                    elif event.key == K_r:
                        if world.assisted_driving and not autopilot_enabled:
                            world.assisted_driving.toggle()
                    elif event.key == K_t:
                        if world.assisted_driving:
                            world.assisted_driving.toggle_obstacle_avoidance()
                    elif event.key == K_h:
                        hud.help.toggle()
                    elif event.key == K_l and rl_enabled:
                        if world.rl_agent:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_path = f"./rl_checkpoints/model_{timestamp}.pth"
                            os.makedirs("./rl_checkpoints", exist_ok=True)
                            world.rl_agent.save(save_path)
                            hud.notification(f"Model saved: {timestamp}", seconds=2.0, color=(0, 255, 0))

            if not world.world.wait_for_tick(10.0):
                continue

            keys = pygame.key.get_pressed()
            reverse_pressed = keys[K_SPACE]

            if rl_enabled:
                world.tick(clock)
                world.render(display)
                pygame.display.flip()

                if rl_training:
                    state = world.rl_env.get_state()

                    if isinstance(world.rl_agent, DQNAgent):
                        action = world.rl_agent.select_action(state)
                    else:
                        action, log_prob, value = world.rl_agent.select_action(state)

                    # 调试输出（每100步打印一次）
                    if rl_step_count % 100 == 0:
                        print(f"[RL Step {rl_step_count}] Action: {action}")

                    next_state, reward, done, _ = world.rl_env.step(action)
                    rl_episode_reward += reward
                    rl_step_count += 1

                    if isinstance(world.rl_agent, DQNAgent):
                        world.rl_agent.memory.push(state, action, reward, next_state, done)
                        loss = world.rl_agent.update()
                    else:
                        world.rl_agent.store_transition(state, action, log_prob, reward, done, value)

                    if done or rl_step_count >= getattr(args, 'rl_max_steps', 2000):
                        if isinstance(world.rl_agent, PPOAgent):
                            loss = world.rl_agent.update()

                        rl_episode_count += 1
                        print(f"Episode {rl_episode_count} | Reward: {rl_episode_reward:.2f} | Steps: {rl_step_count}")

                        if rl_episode_count % 100 == 0:
                            save_path = f"./rl_checkpoints/episode_{rl_episode_count}.pth"
                            world.rl_agent.save(save_path)
                            hud.notification(f"Checkpoint saved: Ep {rl_episode_count}", seconds=2.0)

                        total_episodes = getattr(args, 'rl_episodes', 1000)
                        if rl_episode_count >= total_episodes:
                            hud.notification("Training Complete!", seconds=5.0, color=(0, 255, 0))
                            final_path = "./rl_checkpoints/final_model.pth"
                            world.rl_agent.save(final_path)
                            break

                        rl_episode_reward = 0
                        rl_step_count = 0
                        world.rl_env.reset()

                        spawn_points = world.map.get_spawn_points()
                        if spawn_points:
                            world.player.set_transform(random.choice(spawn_points))
                            target_point = random.choice(spawn_points)
                            world.rl_env.set_target(target_point.location)

                else:
                    state = world.rl_env.get_state()

                    if isinstance(world.rl_agent, DQNAgent):
                        action = world.rl_agent.select_action(state, evaluate=True)
                    else:
                        action, _, _ = world.rl_agent.select_action(state, deterministic=True)

                    control = carla.VehicleControl()
                    if world.rl_env.continuous_actions:
                        control.steer = float(np.clip(action[0], -1, 1))
                        throttle_brake = float(action[1])
                        if throttle_brake >= 0:
                            control.throttle = np.clip(throttle_brake, 0, 1)
                            control.brake = 0.0
                        else:
                            control.throttle = 0.0
                            control.brake = np.clip(-throttle_brake, 0, 1)
                    else:
                        if action == 0:
                            control.steer = -0.5; control.throttle = 0.5
                        elif action == 1:
                            control.steer = 0.0; control.throttle = 0.5
                        elif action == 2:
                            control.steer = 0.5; control.throttle = 0.5
                        elif action == 3:
                            control.steer = 0.0; control.throttle = 0.8
                        elif action == 4:
                            control.steer = 0.0; control.brake = 0.8

                    if world.assisted_driving and world.assisted_driving.enabled:
                        control = world.assisted_driving.apply_assistance(control)

                    world.player.apply_control(control)

                    speed = 3.6 * math.sqrt(world.player.get_velocity().x**2 + world.player.get_velocity().y**2)
                    if hasattr(world.rl_env, 'target_location') and world.rl_env.target_location:
                        dist = world.player.get_location().distance(world.rl_env.target_location)
                        hud.notification(f"RL | Speed: {speed:.0f} | Target: {dist:.1f}m",
                                         seconds=0.1, color=(200, 200, 255))

            else:
                # 原有的手动/自动模式代码保持不变
                if args.agent == "Roaming" or args.agent == "Basic":
                    world.world.wait_for_tick(10.0)
                    world.tick(clock)
                    world.render(display)
                    pygame.display.flip()
                    control = agent.run_step()
                    control.manual_gear_shift = False
                    if world.assisted_driving and not autopilot_enabled:
                        control = world.assisted_driving.apply_assistance(control)
                    world.player.apply_control(control)
                else:
                    world.tick(clock)
                    world.render(display)
                    pygame.display.flip()

                    if autopilot_enabled:
                        local_planner = agent.get_local_planner()
                        if hasattr(local_planner, '_waypoints_queue'):
                            waypoints_len = len(local_planner._waypoints_queue)
                        elif hasattr(local_planner, 'waypoints_queue'):
                            waypoints_len = len(local_planner.waypoints_queue)
                        else:
                            waypoints_len = 0

                        if waypoints_len < num_min_waypoints and args.loop and spawn_points:
                            agent.reroute(spawn_points)
                            tot_target_reached += 1
                            world.hud.notification(f"Target reached x{tot_target_reached}", seconds=4.0)
                        elif waypoints_len == 0 and not args.loop:
                            print("Target reached, mission accomplished...")
                            break

                        speed_limit = world.player.get_speed_limit()
                        agent.get_local_planner().set_speed(speed_limit)
                        control = agent.run_step()

                        if world.assisted_driving and world.assisted_driving.obstacle_avoidance_enabled:
                            result = world.assisted_driving.detect_obstacles()
                            if len(result) >= 3:
                                distance, angle, emergency = result[0], result[1], result[2]
                                if distance < 8.0:
                                    if distance < 5.0 or emergency:
                                        control.brake = 1.0
                                        control.throttle = 0.0
                    else:
                        control = carla.VehicleControl()
                        control.throttle = 0.0
                        control.steer = 0.0
                        control.brake = 0.0
                        control.reverse = False

                        if reverse_pressed:
                            control.reverse = True
                            if keys[K_UP] or keys[K_w]:
                                control.throttle = 0.5
                            elif keys[K_DOWN] or keys[K_s]:
                                control.brake = 0.5
                            else:
                                control.throttle = 0.3
                            if keys[K_LEFT] or keys[K_a]:
                                control.steer = -0.5
                            if keys[K_RIGHT] or keys[K_d]:
                                control.steer = 0.5
                        else:
                            if keys[K_UP] or keys[K_w]:
                                control.throttle = 0.7
                            if keys[K_DOWN] or keys[K_s]:
                                control.brake = 0.7
                            if keys[K_LEFT] or keys[K_a]:
                                control.steer = -0.5
                            if keys[K_RIGHT] or keys[K_d]:
                                control.steer = 0.5

                        if world.assisted_driving:
                            control = world.assisted_driving.apply_assistance(control)

                    world.player.apply_control(control)

    finally:
        if rl_enabled and rl_training and world.rl_agent:
            final_path = "./rl_checkpoints/final_model.pth"
            os.makedirs("./rl_checkpoints", exist_ok=True)
            world.rl_agent.save(final_path)
            print(f"Final model saved to {final_path}")

        if world is not None:
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(description='CARLA 自动驾驶控制客户端')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='打印调试信息')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='服务器IP地址 (默认: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP端口 (默认: 2000)')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='窗口分辨率 (默认: 1280x720)')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='角色过滤器 (默认: "vehicle.*")')
    argparser.add_argument('--gamma', default=2.2, type=float, help='相机伽马校正 (默认: 2.2)')
    argparser.add_argument('-l', '--loop', action='store_true', dest='loop', help='到达目的地后自动规划新路线 (默认: False)')
    argparser.add_argument('-b', '--behavior', type=str, choices=["cautious", "normal", "aggressive"],
                           help='选择自动驾驶行为模式 (默认: normal) ', default='normal')
    argparser.add_argument("-a", "--agent", type=str, choices=["Behavior", "Roaming", "Basic"],
                           help="选择使用的AI代理", default="Behavior")
    argparser.add_argument('-s', '--seed', help='设置随机种子 (默认: None)', default=None, type=int)
    # 添加RL相关参数
    argparser.add_argument('--rl-mode', action='store_true',
                           help='启用RL控制模式')
    argparser.add_argument('--rl-train', action='store_true',
                           help='启用RL训练模式')
    argparser.add_argument('--rl-algorithm', type=str, choices=['dqn', 'ppo'],
                           default='ppo', help='RL算法类型 (默认: ppo)')
    argparser.add_argument('--rl-model', type=str, default='./rl_checkpoints/best_model.pth',
                           help='加载的RL模型路径')
    argparser.add_argument('--rl-save-dir', type=str, default='./rl_checkpoints',
                           help='RL模型保存目录')
    argparser.add_argument('--rl-episodes', type=int, default=1000,
                           help='训练episode数量')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('连接到服务器 %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\n用户取消。再见！')


if __name__ == '__main__':
    main()