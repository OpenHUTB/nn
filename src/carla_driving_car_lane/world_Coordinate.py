#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@gmail.com)
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>



from __future__ import print_function

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
import time
import weakref

# ==========================================
#  你的 CARLA 路径（保持不变）
# ==========================================
CARLA_ROOT = r"D:\carla0.9.15"
PYTHON_API = os.path.join(CARLA_ROOT, "PythonAPI")
sys.path.append(PYTHON_API)
sys.path.append(os.path.join(PYTHON_API, "carla"))

try:
    eggs = glob.glob(os.path.join(PYTHON_API, "carla", "dist", "*.egg"))
    for e in eggs:
        sys.path.append(e)
except:
    pass

# ================= 依赖 =================
try:
    import pygame
    from pygame.locals import K_ESCAPE, K_q, KMOD_CTRL, K_PAGEUP, K_PAGEDOWN
except ImportError:
    raise RuntimeError('请安装：pip install pygame')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('请安装：pip install numpy')

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent


# ==============================================================================
# -- Global functions
# ==============================================================================
def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?=[A-Z])(?=[A-Z][a-z])|$)')

    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))

    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World
# ==============================================================================
class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = 2.2
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)

    def restart(self, args):
        cam_index = self.camera_manager.index if self.camera_manager else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager else 0

        blueprint_library = self.world.get_blueprint_library()
        blueprint = random.choice(blueprint_library.filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')

        # 无绝对坐标：使用当前车辆位置重生（相对偏移）
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0  # 仅相对抬高，无绝对坐标
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        # 无绝对坐标：使用地图原生生成点
        while self.player is None:
            spawn_point = random.choice(self.map.get_spawn_points())
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('天气: %s' % preset[1])
        self.world.set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(clock, self)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None and actor.is_alive:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl
# ==============================================================================
class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("按 ESC 或 Ctrl+Q 退出", 2)
        world.hud.notification("PageUp/PageDown: 切换天气", 2)

    def parse_events(self, world):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                if event.key == K_PAGEUP:
                    world.next_weather(reverse=True)
                if event.key == K_PAGEDOWN:
                    world.next_weather()
        return False

    @staticmethod
    def _is_quit_shortcut(key):
        return key == K_ESCAPE or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD (显示车速、GPS、信息)
# ==============================================================================
class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.font = font
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True

    def on_world_tick(self, timestamp):
        self.server_fps = 1.0 / max(timestamp.delta_seconds, 0.01)
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, clock, world):
        self._notifications.tick(clock)
        if not world.player:
            return
        v = world.player.get_velocity()
        speed = round(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2), 1)
        lat = round(world.gnss_sensor.lat, 6)
        lon = round(world.gnss_sensor.lon, 6)
        self.speed = speed
        self.lat = lat
        self.lon = lon

    def notification(self, text, seconds=2):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('错误: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_text = [
                "FPS: %.1f" % self.server_fps,
                "车速: %.1f km/h" % getattr(self, 'speed', 0),
                "纬度: %.6f" % getattr(self, 'lat', 0),
                "经度: %.6f" % getattr(self, 'lon', 0)
            ]
            y = 10
            for s in info_text:
                surf = self.font.render(s, True, (255, 255, 255))
                display.blit(surf, (10, y))
                y += 20
        self._notifications.render(display)


# ==============================================================================
# -- FadingText
# ==============================================================================
class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        self.seconds_left = seconds
        self.surface = self.font.render(text, True, color)

    def tick(self, clock):
        self.seconds_left = max(0.0, self.seconds_left - clock.get_time() / 1000.0)

    def render(self, display):
        alpha = int(255 * (self.seconds_left / 2.0)) if self.seconds_left > 0 else 0
        self.surface.set_alpha(alpha)
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor
# ==============================================================================
class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        # 无绝对坐标：默认挂载点
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=parent_actor)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification(f"碰撞 → {actor_type}")


# ==============================================================================
# -- LaneInvasionSensor
# ==============================================================================
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        # 无绝对坐标：默认挂载点
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=parent_actor)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_lane(weak_self, event))

    @staticmethod
    def _on_lane(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.hud.notification("车道偏离！")


# ==============================================================================
# -- GnssSensor
# ==============================================================================
class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        # 无绝对坐标：仅相对高度，无 x/y
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(z=2.0)), attach_to=parent_actor)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss(weak_self, event))

    @staticmethod
    def _on_gnss(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager (✅ 已移除所有硬编码相机坐标)
# ==============================================================================
class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.transform_index = 0
        self.sensors = [['sensor.camera.rgb', cc.Raw, 'RGB Camera']]

        # ✅ 动态计算相机位置：基于车辆包围盒，无任何绝对坐标
        self._auto_camera_transform()

        self.index = None
        world = parent_actor.get_world()
        for i, it in enumerate(self.sensors):
            bp = world.get_blueprint_library().find(it[0])
            bp.set_attribute('image_size_x', str(hud.dim[0]))
            bp.set_attribute('image_size_y', str(hud.dim[1]))
            self.sensors[i].append(bp)

    def _auto_camera_transform(self):
        """ 自动计算相机位置：基于车辆尺寸，自适应所有车型，无硬编码坐标 """
        vehicle = self._parent
        bounds = vehicle.bounding_box
        ext = bounds.extent
        # 相机放在车辆后上方，自适应尺寸
        x = -ext.x * 2.5
        z = ext.z * 2.0
        self._camera_transforms = [
            carla.Transform(carla.Location(x=x, z=z), carla.Rotation(pitch=8.0))
        ]

    def set_sensor(self, index, notify=True):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()
        self.index = index
        self.sensor = self._parent.get_world().spawn_actor(
            self.sensors[index][-1],
            self._camera_transforms[0],
            attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))


# ==============================================================================
# -- Game Loop
# ==============================================================================
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        agent = BehaviorAgent(world.player, behavior=args.behavior)
        # ✅ 无绝对坐标：随机目标点来自地图生成点
        spawn_points = world.map.get_spawn_points()
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)
        world.hud.notification("已设置自动导航目标点")

        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(world):
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)

    finally:
        if world is not None:
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- main()
# ==============================================================================
def main():
    argparser = argparse.ArgumentParser(description='CARLA 自动行驶客户端')
    argparser.add_argument('--host', default='127.0.0.1')
    argparser.add_argument('--port', default=2000, type=int)
    argparser.add_argument('--res', default='1280x720')
    argparser.add_argument('--filter', default='vehicle.*')
    argparser.add_argument('-b', '--behavior', default='normal', choices=['cautious', 'normal', 'aggressive'])
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    game_loop(args)


if __name__ == '__main__':
    main()
