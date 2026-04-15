#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""客户端侧自动车辆控制示例"""

from __future__ import print_function

import argparse  # 解析命令行参数
import collections  # 提供额外的数据结构（如默认字典）
import datetime  # 处理日期和时间
import glob  # 查找匹配特定模式的文件路径
import logging  # 日志记录
import math  # 数学函数
import os  # 操作系统接口
import random  # 随机数生成
import re  # 正则表达式
import sys  # 系统相关参数和函数
import weakref  # 弱引用（避免循环引用）

# 尝试导入pygame（用于图形界面和键盘输入）
try:
    import pygame
    from pygame.locals import KMOD_CTRL  # Ctrl键修饰符
    from pygame.locals import K_ESCAPE   # ESC键
    from pygame.locals import K_q        # Q键
except ImportError:
    raise RuntimeError('无法导入pygame，请确保已安装pygame包')

# 尝试导入numpy（用于数值计算）
try:
    import numpy as np
except ImportError:
    raise RuntimeError('无法导入numpy，请确保已安装numpy包')

# ==============================================================================
# -- 查找CARLA模块 ---------------------------------------------------------
# ==============================================================================
try:
    # 尝试添加CARLA的egg文件到系统路径
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,      # Python主版本号
        sys.version_info.minor,      # Python次版本号
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])  # 根据操作系统选择
except IndexError:
    pass  # 如果找不到egg文件，继续尝试其他方法

# ==============================================================================
# -- 为发布模式添加PythonAPI --------------------------------------------
# ==============================================================================
try:
    # 添加carla模块路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla  # 导入CARLA核心模块
from carla import ColorConverter as cc  # 颜色转换器

# 导入导航代理模块
from agents.navigation.behavior_agent import BehaviorAgent  # 行为代理（谨慎/正常/激进）
from agents.navigation.roaming_agent import RoamingAgent   # 漫游代理（随机移动）
from agents.navigation.basic_agent import BasicAgent       # 基础代理（点到点导航）


# ==============================================================================
# -- 全局函数 ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """查找天气预设的方法"""
    # 正则表达式：用于分割驼峰命名字符串
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    # 获取carla.WeatherParameters中所有以大写字母开头的属性（天气预设）
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    # 返回(天气对象, 显示名称)的列表
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """获取角色的显示名称"""
    # 将类型ID中的下划线替换为点，然后按点分割，取第2部分之后的内容
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    # 如果名称过长则截断并添加省略号
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- 世界类 ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """代表周围环境的世界类"""

    def __init__(self, carla_world, hud, args):
        """构造函数"""
        self.world = carla_world  # CARLA世界对象
        try:
            self.map = self.world.get_map()  # 获取地图
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('服务器无法发送OpenDRIVE (.xodr)文件：')
            print('请确保文件存在，名称与城镇相同，并且格式正确。')
            sys.exit(1)
        self.hud = hud  # 抬头显示（HUD）
        self.player = None  # 玩家角色（车辆）
        self.collision_sensor = None  # 碰撞传感器
        self.lane_invasion_sensor = None  # 车道入侵传感器
        self.gnss_sensor = None  # GNSS/GPS传感器
        self.camera_manager = None  # 相机管理器
        self._weather_presets = find_weather_presets()  # 天气预设列表
        self._weather_index = 0  # 当前天气索引
        self._actor_filter = args.filter  # 角色过滤器
        self._gamma = args.gamma  # 伽马校正值
        self.restart(args)  # 启动世界
        self.world.on_tick(hud.on_world_tick)  # 注册世界时钟回调
        self.recording_enabled = False  # 录制状态
        self.recording_start = 0  # 录制开始时间

    def restart(self, args):
        """重启世界"""
        # 保留相机配置（如果相机管理器存在）
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # 如果用户请求了随机种子
        if args.seed is not None:
            random.seed(args.seed)

        # 随机选择一个符合条件的蓝图
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')  # 设置角色为"英雄"
        # 如果有颜色属性，随机选择颜色
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # 生成玩家角色
        print("正在生成玩家角色")
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0  # 提高位置
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        # 确保玩家生成成功
        while self.player is None:
            if not self.map.get_spawn_points():
                print('您的地图/城镇中没有生成点。')
                print('请在UE4场景中添加车辆生成点。')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # 设置传感器
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """切换到下一个天气设置"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('天气: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        """每帧更新的方法"""
        self.hud.tick(self, clock)

    def render(self, display):
        """渲染世界"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """销毁所有传感器"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """销毁所有角色"""
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
# -- 键盘控制 -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    """键盘控制类"""
    
    def __init__(self, world):
        world.hud.notification("按 'H' 或 '?' 查看帮助", seconds=4.0)

    def parse_events(self):
        """解析键盘事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # 关闭窗口
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):  # 按退出快捷键
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """判断是否为退出快捷键"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD 抬头显示 -------------------------------------------------------
# ==============================================================================

class HUD(object):
    """抬头显示（HUD）文本类"""

    def __init__(self, width, height):
        """构造函数"""
        self.dim = (width, height)  # 显示尺寸
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0  # 服务器帧率
        self.frame = 0       # 当前帧数
        self.simulation_time = 0  # 模拟时间
        self._show_info = True    # 是否显示信息
        self._info_text = []      # 信息文本列表
        self._server_clock = pygame.time.Clock()  # 服务器时钟

    def on_world_tick(self, timestamp):
        """每帧从世界获取信息"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()  # 获取FPS
        self.frame = timestamp.frame_count  # 帧计数
        self.simulation_time = timestamp.elapsed_seconds  # 已过时间

    def tick(self, world, clock):
        """每帧更新HUD"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        # 获取车辆状态信息
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        # 计算朝向（北/南/东/西）
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        # 碰撞历史
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')  # 所有车辆

        # 构建显示信息
        self._info_text = [
            '服务器:  % 16.0f FPS' % self.server_fps,
            '客户端:  % 16.0f FPS' % clock.get_fps(),
            '',
            '车辆: % 20s' % get_actor_display_name(world.player, truncate=20),
            '地图:     % 20s' % world.map.name,
            '模拟时间: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            '速度:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'朝向:% 16.0f° % 2s' % (transform.rotation.yaw, heading),
            '位置:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            '高度:  % 18.0f m' % transform.location.z,
            '']
        
        # 添加控制信息
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('油门:', control.throttle, 0.0, 1.0),
                ('转向:', control.steer, -1.0, 1.0),
                ('刹车:', control.brake, 0.0, 1.0),
                ('倒车:', control.reverse),
                ('手刹:', control.hand_brake),
                ('手动档:', control.manual_gear_shift),
                '档位:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        
        self._info_text += [
            '',
            '碰撞:',
            collision,
            '',
            '车辆数量: % 8d' % len(vehicles)]

        # 添加附近车辆信息
        if len(vehicles) > 1:
            self._info_text += ['附近车辆:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """切换信息显示开关"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """显示通知文本"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """显示错误文本"""
        self._notifications.set_text('错误: %s' % text, (255, 0, 0))

    def render(self, display):
        """渲染HUD"""
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
                    # 渲染碰撞历史图表
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    # 渲染控制栏
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
                if item:  # 渲染文本
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- 淡入淡出文本 ----------------------------------------------------------------
# ==============================================================================

class FadingText(object):
    """淡入淡出文本类"""

    def __init__(self, font, dim, pos):
        """构造函数"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0  # 剩余显示秒数
        self.surface = pygame.Surface(self.dim)  # 文本表面

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """设置淡入淡出文本"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """每帧更新（处理淡出效果）"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)  # 根据剩余时间调整透明度

    def render(self, display):
        """渲染淡入淡出文本"""
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- 帮助文本 ------------------------------------------------------------------
# ==============================================================================

class HelpText(object):
    """帮助文本辅助类"""

    def __init__(self, font, width, height):
        """构造函数"""
        lines = __doc__.split('\n')  # 获取文档字符串
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
        """切换帮助文本的显示状态"""
        self._render = not self._render

    def render(self, display):
        """渲染帮助文本"""
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- 碰撞传感器 -----------------------------------------------------------
# ==============================================================================

class CollisionSensor(object):
    """碰撞传感器类"""

    def __init__(self, parent_actor, hud):
        """构造函数"""
        self.sensor = None
        self.history = []  # 碰撞历史记录
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # 使用弱引用来避免循环引用
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """获取碰撞历史"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """碰撞事件回调"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('与 %r 发生碰撞' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)  # 碰撞强度
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)  # 保持历史记录长度


# ==============================================================================
# -- 车道入侵传感器 --------------------------------------------------------
# ==============================================================================

class LaneInvasionSensor(object):
    """车道入侵传感器类"""

    def __init__(self, parent_actor, hud):
        """构造函数"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # 使用弱引用来避免循环引用
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """车道入侵事件回调"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)  # 获取车道线类型
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('越过车道线 %s' % ' 和 '.join(text))


# ==============================================================================
# -- GNSS/GPS传感器 --------------------------------------------------------
# ==============================================================================

class GnssSensor(object):
    """GNSS/GPS传感器类"""

    def __init__(self, parent_actor):
        """构造函数"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0  # 纬度
        self.lon = 0.0  # 经度
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # 使用弱引用来避免循环引用
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS事件回调"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- 相机管理器 -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    """相机管理类"""

    def __init__(self, parent_actor, hud, gamma_correction):
        """构造函数"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False  # 是否录制
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        # 相机位置预设
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
        self.transform_index = 1  # 当前相机位置索引
        # 可用的传感器类型
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, '相机 RGB'],
            ['sensor.camera.depth', cc.Raw, '相机深度 (原始)'],
            ['sensor.camera.depth', cc.Depth, '相机深度 (灰度)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, '相机深度 (对数灰度)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, '相机语义分割 (原始)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             '相机语义分割 (CityScapes调色板)'],
            ['sensor.lidar.ray_cast', None, '激光雷达 (射线投射)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        # 配置每个传感器的属性
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
        """切换相机位置"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """设置传感器"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            # 生成新传感器
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # 使用弱引用监听传感器数据
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """切换到下一个传感器"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """切换录制状态"""
        self.recording = not self.recording
        self.hud.notification('录制 %s' % ('开启' if self.recording else '关闭'))

    def render(self, display):
        """渲染相机画面"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        """解析传感器图像数据"""
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            # 处理激光雷达数据
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
            # 处理相机图像
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]  # RGB转BGR（pygame格式）
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)  # 保存图像


# ==============================================================================
# -- 游戏主循环 ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    """智能体主循环"""
    
    pygame.init()
    pygame.font.init()
    world = None
    tot_target_reached = 0  # 总到达目标次数
    num_min_waypoints = 21  # 最小路径点数量

    try:
        # 连接CARLA服务器
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        # 创建显示窗口
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        # 根据参数选择不同的导航代理
        if args.agent == "Roaming":
            agent = RoamingAgent(world.player)  # 漫游模式
        elif args.agent == "Basic":
            agent = BasicAgent(world.player)  # 基础模式
            spawn_point = world.map.get_spawn_points()[0]
            agent.set_destination((spawn_point.location.x,
                                   spawn_point.location.y,
                                   spawn_point.location.z))
        else:
            agent = BehaviorAgent(world.player, behavior=args.behavior)  # 行为模式
            
            spawn_points = world.map.get_spawn_points()
            random.shuffle(spawn_points)
            
            # 设置目的地
            if spawn_points[0].location != agent.vehicle.get_location():
                destination = spawn_points[0].location
            else:
                destination = spawn_points[1].location
            
            agent.set_destination(agent.vehicle.get_location(), destination, clean=True)

        clock = pygame.time.Clock()

        # 主循环
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events():
                return

            # 等待服务器就绪
            if not world.world.wait_for_tick(10.0):
                continue

            # 根据不同的代理模式执行控制逻辑
            if args.agent == "Roaming" or args.agent == "Basic":
                # 漫游或基础模式
                if controller.parse_events():
                    return
                
                world.world.wait_for_tick(10.0)
                world.tick(clock)
                world.render(display)
                pygame.display.flip()
                control = agent.run_step()
                control.manual_gear_shift = False
                world.player.apply_control(control)
            else:
                # 行为模式（更复杂的导航逻辑）
                agent.update_information()
                
                world.tick(clock)
                world.render(display)
                pygame.display.flip()
                
                # 到达目标后重新规划路线（循环模式）
                if len(agent.get_local_planner().waypoints_queue) < num_min_waypoints and args.loop:
                    agent.reroute(spawn_points)
                    tot_target_reached += 1
                    world.hud.notification("已到达目标 " +
                                           str(tot_target_reached) + " 次。", seconds=4.0)
                # 没有路径点且不循环，则退出
                elif len(agent.get_local_planner().waypoints_queue) == 0 and not args.loop:
                    print("已到达目标，任务完成...")
                    break
                
                # 应用速度限制
                speed_limit = world.player.get_speed_limit()
                agent.get_local_planner().set_speed(speed_limit)
                
                control = agent.run_step()
                world.player.apply_control(control)

    finally:
        if world is not None:
            world.destroy()
        
        pygame.quit()


# ==============================================================================
# -- 主函数 --------------------------------------------------------------
# ==============================================================================

def main():
    """主函数"""
    
    # 命令行参数解析器
    argparser = argparse.ArgumentParser(
        description='CARLA 自动控制客户端')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='打印调试信息')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='服务器主机IP（默认：127.0.0.1）')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP监听端口（默认：2000）')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='窗口分辨率（默认：1280x720）')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='角色过滤器（默认："vehicle.*"）')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='相机的伽马校正值（默认：2.2）')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='到达目标后设置新的随机目的地（默认：False）')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='选择智能体行为模式（默认：normal）',
        default='normal')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Behavior", "Roaming", "Basic"],
                           help="选择要运行的智能体",
                           default="Behavior")
    argparser.add_argument(
        '-s', '--seed',
        help='设置随机种子（默认：None）',
        default=None,
        type=int)

    args = argparser.parse_args()

    # 解析分辨率参数
    args.width, args.height = [int(x) for x in args.res.split('x')]

    # 设置日志级别
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('正在监听服务器 %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\n用户取消。再见！')


if __name__ == '__main__':
    main()
