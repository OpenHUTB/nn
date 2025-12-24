"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import sys
import os

# 获取当前脚本所在的目录，并找到agents文件夹的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_path = os.path.join(current_dir, 'agents')
# 将该路径添加到模块搜索路径中
if os.path.exists(agents_path):
    sys.path.insert(0, agents_path)
import argparse
import collections
import datetime
import glob
import logging
import math
import os
import cv2
import random
import re
import sys
import weakref
from enum import Enum
from collections import deque

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
    from pygame.locals import K_l
    from pygame.locals import K_v
    from pygame.locals import K_c
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
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
from torch.utils.tensorboard import SummaryWriter
import torch

# import tensorflow as tf

# Create a SummaryWriter object to write the metrics to Tensorboard
writer = SummaryWriter()

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.types_behavior import Cautious, Aggressive, Normal

from agents.tools.misc import get_speed, positive, draw_waypoints, distance_vehicle


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))

    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """Class representing the surrounding environment"""

    # 常量提取
    DEFAULT_VEHICLE_MODEL = 'model3'
    DEFAULT_VEHICLE_COLOR = '236,102,17'

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self.world = carla_world
        self.hud = hud
        self.args = args

        self._initialize_map()
        self._weather_presets = find_weather_presets()
        self._weather_index = 0

        # 初始化所有传感器和玩家为None
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None

        self.recording_enabled = False
        self.recording_start = 0

        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def _initialize_map(self):
        """初始化地图，处理可能的错误"""
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print(f'RuntimeError: {error}')
            print('The server could not send the OpenDRIVE (.xodr) file:')
            print('Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

    def restart(self):
        """Restart the world"""
        # 保存相机状态
        cam_index = self.camera_manager.index if self.camera_manager else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager else 0

        # 设置随机种子
        if self.args.seed is not None:
            random.seed(self.args.seed)

        # 创建并配置车辆
        blueprint = self._create_vehicle_blueprint()

        # 生成玩家车辆
        self._spawn_player(blueprint)

        # 设置传感器
        self._setup_sensors(cam_index, cam_pos_id)

    def _create_vehicle_blueprint(self):
        """创建并配置车辆蓝图"""
        blueprint_lib = self.world.get_blueprint_library()
        blueprint = blueprint_lib.filter(self.DEFAULT_VEHICLE_MODEL)[0]
        blueprint.set_attribute('role_name', 'hero')

        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', self.DEFAULT_VEHICLE_COLOR)

        return blueprint

    def _spawn_player(self, blueprint):
        """生成玩家车辆"""
        # 如果玩家已存在，在当前位置重新生成
        if self.player is not None:
            spawn_point = self._get_respawn_point()
            self.destroy()  # 销毁现有actor
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        # 首次生成或重新生成失败时
        if self.player is None:
            spawn_point = self._get_random_spawn_point()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

    def _get_respawn_point(self):
        """获取重新生成点（当前位置上方）"""
        spawn_point = self.player.get_transform()
        spawn_point.location.z += 2.0
        spawn_point.rotation.roll = 0.0
        spawn_point.rotation.pitch = 0.0
        return spawn_point

    def _get_random_spawn_point(self):
        """获取随机生成点"""
        spawn_points = self.map.get_spawn_points()

        if not spawn_points:
            print('There are no spawn points available in your map/town.')
            print('Please add some Vehicle Spawn Point to your UE4 scene.')
            sys.exit(1)

        return random.choice(spawn_points)

    def _setup_sensors(self, cam_index, cam_pos_id):
        """设置所有传感器"""
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)

        self.camera_manager = CameraManager(self.player, self.hud, self.args.gamma)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)

        # 通知玩家类型
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """切换到下一个天气预设"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)

        preset = self._weather_presets[self._weather_index]
        self.hud.notification(f'Weather: {preset[1]}')
        self.world.set_weather(preset[0])

    def tick(self, clock):
        """每一帧的更新方法"""
        self.hud.tick(self, clock)

    def render(self, display):
        """渲染世界"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """销毁所有传感器"""
        if self.camera_manager and self.camera_manager.sensor:
            self.camera_manager.sensor.destroy()
            self.camera_manager.sensor = None
            self.camera_manager.index = None

    def destroy(self):
        """销毁所有actor"""
        actors = [
            self.camera_manager.sensor if self.camera_manager else None,
            self.collision_sensor.sensor if self.collision_sensor else None,
            self.lane_invasion_sensor.sensor if self.lane_invasion_sensor else None,
            self.gnss_sensor.sensor if self.gnss_sensor else None,
            self.player
        ]

        for actor in actors:
            if actor is not None:
                actor.destroy()

        # 重置引用
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, start_locked=True):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        # 可选：启动时自动锁定
        if start_locked and world.camera_manager is not None:
            world.camera_manager.set_lock_mode(True)
            world.camera_manager.hud.notification("Camera Lock: ON", seconds=2.0)

    def parse_events(self, world):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_l:  # 按L键切换锁定
                    if world.camera_manager is not None:
                        world.camera_manager.toggle_lock()
                elif event.key == K_v:  # 按V键切换锁定视角
                    if world.camera_manager is not None and world.camera_manager.lock_mode:
                        world.camera_manager.next_lock_view()
                elif event.key == K_c:  # 按C键切换相机（原有功能）
                    if world.camera_manager is not None:
                        world.camera_manager.toggle_camera()

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x) ** 2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z) ** 2)

        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
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
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """Class for fading text"""

    # 常量定义
    ALPHA_MULTIPLIER = 500.0  # 透明度计算系数
    TEXT_OFFSET = (10, 11)  # 文本偏移量
    DEFAULT_COLOR = (255, 255, 255)  # 默认文本颜色
    DEFAULT_SECONDS = 2.0  # 默认显示时间

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self._init_surface()

    def _init_surface(self):
        """初始化Surface"""
        self.surface = pygame.Surface(self.dim)
        self._clear_surface()

    def _clear_surface(self):
        """清空Surface"""
        self.surface.fill((0, 0, 0, 0))

    def set_text(self, text, color=None, seconds=None):
        """Set fading text"""
        # 使用默认值
        if color is None:
            color = self.DEFAULT_COLOR
        if seconds is None:
            seconds = self.DEFAULT_SECONDS

        # 渲染文本
        text_texture = self.font.render(text, True, color)

        # 准备Surface
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self._clear_surface()

        # 绘制文本
        self.surface.blit(text_texture, self.TEXT_OFFSET)

    def tick(self, _, clock):
        """Fading text method for every tick"""
        # 计算经过的时间
        delta_seconds = clock.get_time() / 1000.0  # 转换为秒

        # 更新时间
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)

        # 更新透明度
        self._update_alpha()

    def _update_alpha(self):
        """更新Surface的透明度"""
        alpha_value = self.seconds_left * self.ALPHA_MULTIPLIER
        self.surface.set_alpha(alpha_value)

    def render(self, display):
        """Render fading text method"""
        if self.seconds_left > 0:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = [
            'Welcome to CARLA automatic control client',
            '',
            'Camera Control:',
            '    C          : Change camera position',
            '    L          : Toggle camera lock on vehicle',
            '    V          : Change lock view (when locked)',
            '    PageUp     : Toggle recording',
            '',
            'Vehicle Control:',
            '    W          : Throttle',
            '    S          : Brake',
            '    A/D        : Steer left/right',
            '    Q          : Toggle reverse',
            '    P          : Toggle autopilot',
            '',
            'Miscellaneous:',
            '    H          : Toggle help',
            '    ESC        : Quit',
            '    CTRL + Q   : Quit',
        ]

        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud, gamma_correction):
        """Constructor method"""
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

        # 添加锁定模式相关属性
        self.lock_mode = True  # 默认启用锁定模式
        self.lock_target = parent_actor  # 锁定目标
        self.lock_transform_index = 0  # 锁定模式的相机位置索引

        # 定义锁定模式的相机变换（可以自定义不同角度）
        self._lock_transforms = [
            (carla.Transform(
                carla.Location(x=-6.0, z=3.5), carla.Rotation(pitch=-10.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=0.0)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.0, y=2.0, z=2.0), carla.Rotation(pitch=-5.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-10.0, z=5.0), carla.Rotation(pitch=-15.0)), attachment.SpringArm),
        ]

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
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
                force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None

            # 选择变换和附着类型
            if self.lock_mode:
                # 锁定模式使用锁定变换
                transform, attachment_type = self._lock_transforms[self.lock_transform_index]
            else:
                # 非锁定模式使用原有变换
                transform, attachment_type = self._camera_transforms[self.transform_index]

            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                transform,
                attach_to=self._parent,
                attachment_type=attachment_type)

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            sensor_name = self.sensors[index][2]
            if self.lock_mode:
                sensor_name += " (Locked)"
            self.hud.notification(sensor_name)
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def toggle_lock(self):
        """切换锁定模式"""
        self.lock_mode = not self.lock_mode
        mode_text = "Camera Lock: ON" if self.lock_mode else "Camera Lock: OFF"
        self.hud.notification(mode_text, seconds=2.0)
        # 重新设置传感器以应用锁定模式
        if self.sensor is not None:
            self.set_sensor(self.index, notify=False, force_respawn=True)
        return self.lock_mode

    def set_lock_mode(self, enable=True):
        """设置锁定模式"""
        self.lock_mode = enable
        return self.lock_mode

    def next_lock_view(self):
        """切换到下一个锁定视角"""
        if self.lock_mode:
            self.lock_transform_index = (self.lock_transform_index + 1) % len(self._lock_transforms)
            self.hud.notification(f'Lock View {self.lock_transform_index + 1}', seconds=1.0)
            # 重新设置传感器以应用新的锁定视角
            if self.sensor is not None:
                self.set_sensor(self.index, notify=False, force_respawn=True)

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
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

            #########################################################################################################################################

        ############################################################  yolov3 ####################################################################
        def calculate_iou(box1, box2):
            """
            Calculate the Intersection over Union (IoU) between two bounding boxes.

            Args:
                box1 (list): List containing the coordinates of the first bounding box [x, y, w, h].
                box2 (list): List containing the coordinates of the second bounding box [x, y, w, h].

            Returns:
                float: The IoU value.

            """
            # Extract box coordinates
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2

            # Calculate box coordinates
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)

            # Calculate intersection area
            intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

            # Calculate union area
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - intersection_area

            # Calculate IoU
            iou = intersection_area / union_area

            return iou

        test_im = np.array(image.raw_data)
        test_im = test_im.copy()
        test_im = test_im.reshape((image.height, image.width, 4))
        test_im = test_im[:, :, :3]

        # Load YOLO
        model_weights = 'yolov3.cfg'
        model_cfg = 'yolov3.weights'
        # frameWidth = 1200
        # frameHeight = 720
        frameWidth = 640
        frameHeight = 480
        net = cv2.dnn.readNet(model_weights, model_cfg)

        # Define your loss function
        # loss_fn = tf.keras.losses.BinaryCrossentropy()

        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layers_names = net.getLayerNames()
        # output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # Loading image
        img = cv2.resize(test_im, (frameWidth, frameHeight), None)
        height, width, channels = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect objects
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Perform object detection
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        detected_labels = []
        detected_boxes = []
        predicted_confidences = []
        predicted_class_probs = []

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence, (x + 60, y + 60), font, 0.8, (255, 255, 200), 1)
                detected_labels.append(label)
                detected_boxes.append([x, y, w, h])
                predicted_confidences.append(float(confidence))
                predicted_class_probs.append(label)

        # Ground truth annotations
        ground_truth_labels = ['person', 'bicycle', 'car', 'traffic light', 'fire hydrant', 'pottedplant', 'stop sign',
                               'bench']
        ground_truth_boxes = [[250, 220, 160, 190], [200, 200, 80, 80], [200, 200, 80, 80], [450, 90, 20, 50],
                              [200, 200, 80, 80], [200, 200, 80, 80], [200, 200, 80, 80],
                              [200, 200, 80, 80]]  # [x, y, w, h] format

        # def normalize_boxes(boxes, image_width, image_height):
        # Convert box coordinates to numpy array
        # boxes = np.array(boxes)

        # Extract box coordinates
        # x, y, w, h = np.split(boxes, 4, axis=1)

        # Normalize box coordinates
        # normalized_x = x / image_width
        # normalized_y = y / image_height
        # normalized_w = w / image_width
        # normalized_h = h / image_height

        # Concatenate normalized coordinates
        # normalized_boxes = np.concatenate((normalized_x, normalized_y, normalized_w, normalized_h), axis=1)

        # return normalized_boxes
        '''
        def preprocess_ground_truth(ground_truth_labels, ground_truth_boxes):
            # Define a dictionary to map class labels to class IDs
            label_to_id = {
                'person': 0,
                'bicycle': 1,
                'car': 2,
                'traffic light': 3,
                'fire hydrant': 4,
                'pottedplant': 5,
                'stop sign': 6,
                'bench': 7
            }
        '''

        # Convert class labels to class IDs#
        '''
            class_ids = [label_to_id[label] for label in ground_truth_labels]

            # Convert class IDs to one-hot encoding
            num_classes = len(label_to_id)
            one_hot_labels = tf.one_hot(class_ids, num_classes)

            # Normalize bounding box coordinates
            normalized_boxes = normalize_boxes(ground_truth_boxes, width, height)  # Implement 'normalize_boxes' function to scale box coordinates between 0 and 1

            print('Normalized Boxes Shape:', normalized_boxes.shape)
            print('One-Hot Labels Shape:', one_hot_labels.shape)
            print('Normalized Boxes Data Type:', normalized_boxes.dtype)
            print('One-Hot Labels Data Type:', one_hot_labels.dtype)

            if normalized_boxes.shape != one_hot_labels.shape:
                print('Error: Shapes of normalized_boxes and one_hot_labels do not match.')
                return None

            if normalized_boxes.dtype != one_hot_labels.dtype:
                print('Error: Data types of normalized_boxes and one_hot_labels do not match.')
                return None

            # Combine labels and boxes into a single tensor
            ground_truth_targets = tf.concat([normalized_boxes, one_hot_labels], axis=-1)

            return ground_truth_targets, normalized_boxes, one_hot_labels, label_to_id
        # Call the function and store the returned values
        #ground_truth_targets, normalized_boxes, one_hot_labels, label_to_id = preprocess_ground_truth(ground_truth_labels, ground_truth_boxes)

        #num_classes = len(label_to_id)
    '''

        # Perform object matching for ground truth and detected objects
        TP = 0  # True positives
        FP = 0  # False positives
        FN = 0  # False negatives
        threshold = 0.5
        print("Detected Labels: ", detected_labels)
        print("Detected boxes: ", detected_boxes)

        for i, (gt_label, gt_box) in enumerate(zip(ground_truth_labels, ground_truth_boxes)):
            max_iou = 0
            max_iou_index = -1

            for j, (det_label, det_box) in enumerate(zip(detected_labels, detected_boxes)):
                iou = calculate_iou(gt_box, det_box)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_index = j

            if max_iou >= threshold:
                # Matched object found (True positive)
                TP += 1
                # Remove the matched object from detected objects
                if max_iou_index != -1:
                    detected_labels.pop(max_iou_index)
                    detected_boxes.pop(max_iou_index)
            else:
                # No match found (False negative)
                FN += 1

        # Any remaining unmatched detected objects are false positives
        FP = len(detected_labels)

        eps = 1e-7  # Small value to avoid division by zero

        # Calculate performance metrics
        accuracy = TP / (TP + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        matt_corr_coeff = (TP - (FP * FN)) / math.sqrt((TP + FP + eps) * (TP + FN + eps) * (FP + eps) * (FN + eps))

        # Calculate the loss
        # ground_truth_targets = preprocess_ground_truth(ground_truth_labels, ground_truth_boxes)  # Preprocess the ground truth targets

        # localization_loss = loss_fn(detected_boxes, ground_truth_targets)  # Calculate the localization loss
        # confidence_loss = loss_fn(predicted_confidences, ground_truth_targets)  # Calculate the confidence loss
        # class_prob_loss = loss_fn(predicted_class_probs, ground_truth_targets)  # Calculate the class probability loss

        # Log the metrics to Tensorboard
        log_dir = "./logs3"  # Specify the log directory her
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_scalar('Accuracy', accuracy, image.frame)
        writer.add_scalar('Precision', precision, image.frame)
        writer.add_scalar('Recall', recall, image.frame)
        writer.add_scalar('IOU', max_iou, image.frame)
        # writer.add_scalar('Matthews Correlation Coefficient', matt_corr_coeff)
        # writer.add_scalar('Localization Loss', localization_loss, image.frame)
        # writer.add_scalar('Confidence Loss', confidence_loss, image.frame)
        # writer.add_scalar('Class Probability Loss', class_prob_loss, image.frame)

        # Show the annotated image
        # cv2.imshow("Annotated Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        self.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('output/%08d' % image.frame)


# ==============================================================================
# -- KK PID Controllers  ---------------------------------------------------------
# ==============================================================================

class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(self, vehicle, args_lateral, args_longitudinal, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal
        PID controller using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.past_steering = self._vehicle.get_control().steer
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, **args_lateral)

    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint
        at a given target_speed.

            :param target_speed: desired vehicle speed
            :param waypoint: target location encoded as a waypoint
            :return: distance (in meters) to the waypoint
        """

        acceleration = self._lon_controller.run_step(target_speed)
        current_steering = self._lat_controller.run_step(waypoint)
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_speed = get_speed(self._vehicle) / 2
        target_speed = target_speed / 2
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)


class PIDLateralController:
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.

        Args:
            vehicle: Actor to apply local planner logic onto
            K_P: Proportional term coefficient (default: 1.0)
            K_D: Differential term coefficient (default: 0.0)
            K_I: Integral term coefficient (default: 0.0)
            dt: Time differential in seconds (default: 0.03)
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer the vehicle towards a waypoint.

        Args:
            waypoint: Target waypoint

        Returns:
            float: Steering control in range [-1, 1] where:
                   -1 = maximum steering to left
                   +1 = maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate steering angle based on PID equations.

        Args:
            waypoint: Target waypoint
            vehicle_transform: Current transform of the vehicle

        Returns:
            float: Steering control in range [-1, 1]
        """
        # Calculate vehicle direction vector
        v_begin = vehicle_transform.location
        yaw_rad = math.radians(vehicle_transform.rotation.yaw)
        v_end = v_begin + carla.Location(
            x=math.cos(yaw_rad),
            y=math.sin(yaw_rad)
        )

        # Calculate vectors
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([
            waypoint.transform.location.x - v_begin.x,
            waypoint.transform.location.y - v_begin.y,
            0.0
        ])

        # Calculate angle difference
        v_norm = np.linalg.norm(v_vec)
        w_norm = np.linalg.norm(w_vec)
        dot_product = np.dot(w_vec, v_vec) / (v_norm * w_norm)
        dot_product_clipped = np.clip(dot_product, -1.0, 1.0)
        angle_diff = math.acos(dot_product_clipped)

        # Determine sign based on cross product
        cross_product = np.cross(v_vec, w_vec)
        if cross_product[2] < 0:
            angle_diff *= -1.0

        # Update error buffer and calculate PID terms
        self._error_buffer.append(angle_diff)

        # Calculate derivative and integral terms
        derivative_error = self._calculate_derivative_error()
        integral_error = self._calculate_integral_error()

        # PID calculation with clamping
        pid_output = (
                self._k_p * angle_diff +
                self._k_d * derivative_error +
                self._k_i * integral_error
        )

        return np.clip(pid_output, -1.0, 1.0)

    def _calculate_derivative_error(self):
        """Calculate derivative error term."""
        if len(self._error_buffer) >= 2:
            return (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
        return 0.0

    def _calculate_integral_error(self):
        """Calculate integral error term."""
        return sum(self._error_buffer) * self._dt


# ==============================================================================
# -- KK Local Planner  ---------------------------------------------------------
# ==============================================================================

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations
    when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory
    of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers,
    one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections)
    this local planner makes a random choice.
    """

    # Minimum distance to target waypoint as a percentage
    # (e.g. within 80% of total distance)

    # FPS used for dt
    FPS = 20

    def __init__(self, agent):
        """
        :param agent: agent that regulates the vehicle
        :param vehicle: actor to apply to local planner logic onto
        """
        self._vehicle = agent.vehicle
        self._map = agent.vehicle.get_world().get_map()

        self._target_speed = None
        self.sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self.target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        self._pid_controller = None
        self.waypoints_queue = deque(maxlen=20000)  # queue with tuples of (waypoint, RoadOption)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        self._init_controller()  # initializing controller

    def get_distance_to_nearest_obstacle(self):
        ego_vehicle_transform = self._vehicle.get_transform()
        ego_vehicle_location = ego_vehicle_transform.location

        # Get the bounding box of the ego vehicle
        ego_vehicle_bbox = self._vehicle.bounding_box

        # Calculate the minimum distance to other vehicles
        distance_to_nearest_obstacle = None

        for actor in self._vehicle.get_world().get_actors().filter('vehicle.*'):
            if actor.id != self._vehicle.id:
                actor_transform = actor.get_transform()
                actor_location = actor_transform.location

                # Calculate the distance between the ego vehicle and the other vehicle
                distance = ego_vehicle_location.distance(actor_location)

                # Calculate the minimum distance to the other vehicle's bounding box
                distance_to_obstacle = distance - ego_vehicle_bbox.extent.x - actor.bounding_box.extent.x

                if distance_to_nearest_obstacle is None or distance_to_obstacle < distance_to_nearest_obstacle:
                    distance_to_nearest_obstacle = distance_to_obstacle

        return distance_to_nearest_obstacle

    def reset_vehicle(self):
        """Reset the ego-vehicle"""
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def _init_controller(self):
        """
        Controller initialization.

        dt -- time difference between physics control in seconds.
        This is can be fixed from server side
        using the arguments -benchmark -fps=F, since dt = 1/F

        target_speed -- desired cruise speed in km/h

        min_distance -- minimum distance to remove waypoint from queue

        lateral_dict -- dictionary of arguments to setup the lateral PID controller
                            {'K_P':, 'K_D':, 'K_I':, 'dt'}

        longitudinal_dict -- dictionary of arguments to setup the longitudinal PID controller
                            {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        # Default parameters
        self.args_lat_hw_dict = {
            'K_P': 0.75,
            'K_D': 0.02,
            'K_I': 0.4,
            'dt': 1.0 / self.FPS}
        self.args_lat_city_dict = {
            'K_P': 0.58,
            'K_D': 0.02,
            'K_I': 0.5,
            'dt': 1.0 / self.FPS}
        self.args_long_hw_dict = {
            'K_P': 0.37,
            'K_D': 0.024,
            'K_I': 0.032,
            'dt': 1.0 / self.FPS}
        self.args_long_city_dict = {
            'K_P': 0.15,
            'K_D': 0.05,
            'K_I': 0.07,
            'dt': 1.0 / self.FPS}

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        self._global_plan = False

        self._target_speed = self._vehicle.get_speed_limit()

        self._min_distance = 3

    def set_speed(self, speed):
        """
        Request new target speed.

            :param speed: new target speed in km/h
        """

        self._target_speed = speed

    def set_global_plan(self, current_plan):
        """
        Sets new global plan.

            :param current_plan: list of waypoints in the actual plan
        """
        for elem in current_plan:
            self.waypoints_queue.append(elem)
        self._global_plan = True

    def get_incoming_waypoint_and_direction(self, steps=3):
        """
        Returns direction and waypoint at a distance ahead defined by the user.

            :param steps: number of steps to get the incoming waypoint.
        """
        if len(self.waypoints_queue) > steps:
            return self.waypoints_queue[steps]

        else:
            try:
                wpt, direction = self.waypoints_queue[-1]
                return wpt, direction
            except IndexError as i:
                print(i)
                return None, RoadOption.VOID
        return None, RoadOption.VOID

    def run_step(self, target_speed=None, debug=False):
        """
        Execute one step of local planning which involves
        running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

            :param target_speed: desired speed
            :param debug: boolean flag to activate waypoints debugging
            :return: control
        """

        if target_speed is not None:
            self._target_speed = target_speed
        else:
            self._target_speed = self._vehicle.get_speed_limit()

        if len(self.waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
            return control

        # Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self.waypoints_queue:
                    self._waypoint_buffer.append(
                        self.waypoints_queue.popleft())
                else:
                    break

        # Current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        # Target waypoint
        self.target_waypoint, self.target_road_option = self._waypoint_buffer[0]

        if target_speed > 50:
            args_lat = self.args_lat_hw_dict
            args_long = self.args_long_hw_dict
        else:
            args_lat = self.args_lat_city_dict
            args_long = self.args_long_city_dict

        self._pid_controller = VehiclePIDController(self._vehicle,
                                                    args_lateral=args_lat,
                                                    args_longitudinal=args_long)

        control = self._pid_controller.run_step(self._target_speed, self.target_waypoint)

        # Purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        if debug:
            draw_waypoints(self._vehicle.get_world(),
                           [self.target_waypoint], 1.0)
        return control


# ==============================================================================
# -- KK Agent Class ---------------------------------------------------------
# ==============================================================================

class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_RED_LIGHT = 3


class Agent:
    """Base class to define agents in CARLA"""

    def __init__(self, vehicle):
        """
        Constructor method.

        Args:
            vehicle: Actor to apply local planner logic onto
        """
        self._vehicle = vehicle
        self._proximity_tlight_threshold = 5.0
        self._proximity_vehicle_threshold = 10.0
        self._local_planner = None
        self._world = self._vehicle.get_world()
        self._last_traffic_light = None
        self._map = None

        self._init_map()

    def _init_map(self):
        """Initialize the map with error handling."""
        try:
            self._map = self._world.get_map()
        except RuntimeError as error:
            self._handle_map_error(error)

    def _handle_map_error(self, error):
        """Handle map initialization error."""
        error_msg = (
            f'RuntimeError: {error}\n'
            '  The server could not send the OpenDRIVE (.xodr) file:\n'
            '  Make sure it exists, has the same name of your town, and is correct.'
        )
        print(error_msg)
        sys.exit(1)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    @property
    def local_planner(self):
        """Get the local planner instance (property version)."""
        return self._local_planner

    @staticmethod
    def run_step(debug=False):
        """
        Execute one step of navigation.

        Args:
            debug: Boolean flag for debugging

        Returns:
            control: Vehicle control command
        """
        control = carla.VehicleControl()

        if debug:
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False

        return control

    @staticmethod
    def emergency_stop():
        """
        Send an emergency stop command to the vehicle.

        Returns:
            control: Emergency stop control command
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control


# ==============================================================================
# -- KK Behavior Agent Class ---------------------------------------------------------
# ==============================================================================

class BehaviorAgent(Agent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment,
    such as overtaking or tailgating avoidance. Adding to these are possible
    behaviors, the agent can also keep safety distance from a car in front of it
    by tracking the instantaneous time to collision and keeping it in a certain range.
    Finally, different sets of behaviors are encoded in the agent, from cautious
    to a more aggressive ones.
    """

    def __init__(self, vehicle, ignore_traffic_light=False, behavior='normal'):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param ignore_traffic_light: boolean to ignore any traffic light
            :param behavior: type of agent to apply
        """

        super(BehaviorAgent, self).__init__(vehicle)
        self.vehicle = vehicle
        self.ignore_traffic_light = ignore_traffic_light
        self._local_planner = LocalPlanner(self)
        self._grp = None
        self.look_ahead_steps = 0

        # Vehicle information
        self.speed = 0
        self.speed_limit = 0
        self.direction = None
        self.incoming_direction = None
        self.incoming_waypoint = None
        self.start_waypoint = None
        self.end_waypoint = None
        self.is_at_traffic_light = 0
        self.light_state = "Green"
        self.light_id_to_ignore = -1
        self.min_speed = 5
        self.behavior = None
        self._sampling_resolution = 4.5

        # Parameters for agent behavior
        if behavior == 'cautious':
            self.behavior = Cautious()

        elif behavior == 'normal':
            self.behavior = Normal()

        elif behavior == 'aggressive':
            self.behavior = Aggressive()

    def update_information(self, world):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.

            :param world: carla.world object
        """
        self.speed = get_speed(self.vehicle)
        self.speed_limit = world.player.get_speed_limit()
        # self.speed_limit = 100  # KK set speed to 100 km/h
        self._local_planner.set_speed(self.speed_limit)
        self.direction = self._local_planner.target_road_option
        if self.direction is None:
            self.direction = RoadOption.LANEFOLLOW

        self.look_ahead_steps = int(self.speed_limit / 10)

        self.incoming_waypoint, self.incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self.look_ahead_steps)
        if self.incoming_direction is None:
            self.incoming_direction = RoadOption.LANEFOLLOW

        self.is_at_traffic_light = world.player.is_at_traffic_light()
        if self.ignore_traffic_light:
            self.light_state = "Green"
        else:
            if self.is_at_traffic_light:
                traffic_light = world.player.get_traffic_light()
                self.light_state = str(traffic_light.state)
                if self.light_id_to_ignore != -1 and traffic_light.id == self.light_id_to_ignore:
                    # Ignore this traffic light for this iteration
                    self.light_state = "Green"
        # Get distance to the nearest obstacle
        distance_to_nearest_obstacle = self._local_planner.get_distance_to_nearest_obstacle()

        # Check if the obstacle is closer than the minimum distance
        if distance_to_nearest_obstacle is not None and distance_to_nearest_obstacle < 3.0:
            # Apply braking if the distance to the nearest obstacle is below the threshold
            self._local_planner.brake = 1.0
        else:
            self._local_planner.brake = 0.0
        # else:
        # This method also includes stop signs and intersections.
        # self.light_state = str(self.vehicle.get_traffic_light_state())

    def set_destination(self, start_location, end_location, clean=False):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router.

            :param start_location: initial position
            :param end_location: final position
            :param clean: boolean to clean the waypoint queue
        """
        if clean:
            self._local_planner.waypoints_queue.clear()
        self.start_waypoint = self._map.get_waypoint(start_location)
        self.end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self._trace_route(self.start_waypoint, self.end_waypoint)

        self._local_planner.set_global_plan(route_trace)

    def reroute(self, spawn_points):
        """
        This method implements re-routing for vehicles approaching its destination.
        It finds a new target and computes another path to reach it.

            :param spawn_points: list of possible destinations for the agent
        """

        print("Target almost reached, setting new destination...")
        random.shuffle(spawn_points)
        new_start = self._local_planner.waypoints_queue[-1][0].transform.location
        destination = spawn_points[0].location if spawn_points[0].location != new_start else spawn_points[1].location
        print("New destination: " + str(destination))

        self.set_destination(new_start, destination)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the
        optimal route from start_waypoint to end_waypoint.

            :param start_waypoint: initial position
            :param end_waypoint: final position
        """
        # Setting up global router
        if self._grp is None:
            wld = self.vehicle.get_world()
            dao = GlobalRoutePlannerDAO(
                wld.get_map(), sampling_resolution=self._sampling_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        control = carla.VehicleControl()

        ego_vehicle_loc = self.vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        if self.is_at_traffic_light and self.light_state != "Green":
            # Stop at red traffic light
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 1.0
            control.steer = 0.0
        else:
            # Get distance to the nearest obstacle
            distance_to_nearest_obstacle = self._local_planner.get_distance_to_nearest_obstacle()

            # Check if the obstacle is closer than the minimum distance
            if distance_to_nearest_obstacle is not None and distance_to_nearest_obstacle < 3.0:
                # Apply braking if the distance to the nearest obstacle is below the threshold
                control.throttle = 0.0
                control.brake = 1.0
                control.steer = 0.0
            else:
                # Continue following the planned route
                control = self._local_planner.run_step(
                    target_speed=min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist),
                    debug=debug)
        self.vehicle.apply_control(control)

        return control


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """ Main loop for agent"""

    # 初始化部分保持不变
    pygame.init()
    pygame.font.init()
    world = None
    tot_target_reached = 0
    num_min_waypoints = 21

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(70.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        selected_world = client.get_world()
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)
        agent = BehaviorAgent(world.player, behavior=args.behavior)

        spawn_points = world.map.get_spawn_points()
        random.shuffle(spawn_points)

        if spawn_points[0].location != agent.vehicle.get_location():
            destination = spawn_points[0].location
        else:
            destination = spawn_points[1].location

        agent.set_destination(agent.vehicle.get_location(), destination, clean=True)

        clock = pygame.time.Clock()

        # 主循环开始
        while True:
            clock.tick_busy_loop(60)

            # 统一处理事件
            if controller.parse_events(world):
                return

            # 等待世界更新
            if not world.world.wait_for_tick(10.0):
                continue

            # 根据代理类型执行不同的逻辑
            if args.agent in ["Roaming", "Basic"]:
                world.world.wait_for_tick(10.0)  # 等待服务器
                world.tick(clock)
                world.render(display)
                pygame.display.flip()

                # 为Roaming/Basic代理生成控制
                control = agent.run_step()
                control.manual_gear_shift = False
                world.player.apply_control(control)
                continue  # 跳过后续的控制应用

            # 其他代理类型的逻辑
            agent.update_information(world)
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            # 检查是否到达目的地
            if len(agent.get_local_planner().waypoints_queue) < num_min_waypoints and args.loop:
                agent.reroute(spawn_points)
                tot_target_reached += 1
                world.hud.notification("The target has been reached " +
                                       str(tot_target_reached) + " times.", seconds=4.0)
            elif len(agent.get_local_planner().waypoints_queue) == 0 and not args.loop:
                print("Target reached, mission accomplished...")
                break

            # 设置速度限制并应用控制
            speed_limit = world.player.get_speed_limit()
            agent.get_local_planner().set_speed(speed_limit)
            control = agent.run_step()
            world.player.apply_control(control)

    finally:
        if world is not None:
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    # 参数解析器配置
    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client'
    )

    # 参数定义
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information'
    )
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)'
    )
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)'
    )
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)'
    )
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")'
    )
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)'
    )
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)'
    )
    argparser.add_argument(
        '-b', '--behavior',
        type=str,
        choices=['cautious', 'normal', 'aggressive'],
        help='Choose one of the possible agent behaviors (default: normal)',
        default='normal'
    )
    argparser.add_argument(
        '-a', '--agent',
        type=str,
        choices=['Behavior', 'Roaming', 'Basic'],
        help='Select which agent to run',
        default='Behavior'
    )
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int
    )

    # 解析参数
    args = argparser.parse_args()

    # 解析分辨率
    args.width, args.height = map(int, args.res.split('x'))

    # 配置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=log_level
    )

    logging.info('Listening to server %s:%s', args.host, args.port)

    # 打印文档字符串（假设有外部文档）
    if __doc__:
        print(__doc__)

    # 运行主循环
    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()