#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>

"""Example of automatic vehicle control from client side."""

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
import weakref

# ==========================================
#  适配你的路径 D:\carla0.9.15
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

# ================= 依赖导入 =================
try:
    import pygame
    from pygame.locals import *
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

        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')

        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

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
        self.hud.notification('Weather: %s' % preset[1])
        self.world.set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(clock)

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
            if actor is not None:
                actor.destroy()

# ==============================================================================
# -- KeyboardControl
# ==============================================================================
class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press ESC or Ctrl+Q to quit", seconds=2)
        world.hud.notification("PageUp / PageDown: Change Weather", seconds=2)

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
        return key == K_ESCAPE or (key == K_q and pygame.key.get_mod() & KMOD_CTRL)

# ==============================================================================
# -- HUD
# ==============================================================================
class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True

    def on_world_tick(self, timestamp):
        self.server_fps = 1.0 / timestamp.delta_seconds
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, clock):
        self._notifications.tick(clock)

    def notification(self, text, seconds=2):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
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
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor
# ==============================================================================
class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        bp = parent_actor.get_world().get_blueprint_library().find('sensor.other.collision')
        self.sensor = parent_actor.get_world().spawn_actor(bp, carla.Transform(), attach_to=parent_actor)
        self.sensor.listen(lambda e: self.hud.notification(f"碰撞: {get_actor_display_name(e.other_actor)}"))

# ==============================================================================
# -- LaneInvasionSensor
# ==============================================================================
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        bp = parent_actor.get_world().get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = parent_actor.get_world().spawn_actor(bp, carla.Transform(), attach_to=parent_actor)

# ==============================================================================
# -- GnssSensor
# ==============================================================================
class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.lat = self.lon = 0.0
        bp = parent_actor.get_world().get_blueprint_library().find('sensor.other.gnss')
        self.sensor = parent_actor.get_world().spawn_actor(bp, carla.Transform(carla.Location(z=2.0)), attach_to=parent_actor)

# ==============================================================================
# -- CameraManager
# ==============================================================================
class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.transform_index = 0
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8)), carla.AttachmentType.SpringArm)
        ]
        self.sensors = [['sensor.camera.rgb', cc.Raw, 'RGB']]
        world = parent_actor.get_world()
        for item in self.sensors:
            b = world.get_blueprint_library().find(item[0])
            b.set_attribute('image_size_x', str(hud.dim[0]))
            b.set_attribute('image_size_y', str(hud.dim[1]))
            item.append(b)
        self.index = None

    def set_sensor(self, index, notify=True):
        if self.sensor:
            self.sensor.destroy()
        self.sensor = self._parent.get_world().spawn_actor(
            self.sensors[0][-1],
            self._camera_transforms[0][0],
            attach_to=self._parent
        )
        self.sensor.listen(lambda img: self._parse_image(img))
        self.index = 0

    def _parse_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape(image.height, image.width, 4)[:, :, :3]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1)[:, :, ::-1])

    def render(self, display):
        if self.surface:
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
        spawn_points = world.map.get_spawn_points()
        agent.set_destination(random.choice(spawn_points).location)

        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(world):
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            control = agent.run_step()
            world.player.apply_control(control)

    finally:
        if world:
            world.destroy()
        pygame.quit()

# ==============================================================================
# -- main()
# ==============================================================================
def main():
    argparser = argparse.ArgumentParser(description='CARLA Automatic Control Client')
    argparser.add_argument('--host', default='127.0.0.1')
    argparser.add_argument('--port', default=2000, type=int)
    argparser.add_argument('--res', default='1280x720')
    argparser.add_argument('--filter', default='vehicle.*')
    argparser.add_argument('-b', '--behavior', default='normal')
    args = argparser.parse_args()
    args.width, args.height = map(int, args.res.split('x'))
    game_loop(args)

if __name__ == '__main__':
    main()
