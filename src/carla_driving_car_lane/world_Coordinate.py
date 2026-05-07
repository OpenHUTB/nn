#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import math
import random
import weakref
import glob

# ==============================================
# 强制阻断所有冲突库加载（终极方案）
# ==============================================
for lib in ['matplotlib', 'scipy', 'tensorflow', 'torch', 'keras']:
    sys.modules[lib] = None
    sys.modules[lib + '.core'] = None
    sys.modules[lib + '.api'] = None

# 加载 CARLA Python API
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    eggs = glob.glob(os.path.join(current_dir, "carla", "dist", "*.egg"))
    for e in eggs:
        sys.path.append(e)
except:
    pass

# 仅加载必要库
import pygame
from pygame.locals import *
import numpy as np
import carla
from carla import ColorConverter as cc


# ==============================================
# 自己实现极简自动导航（不依赖任何第三方库）
# ==============================================
class SimpleAutoAgent:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()
        self.target = None

    def set_destination(self, location):
        self.target = location

    def run_step(self):
        control = carla.VehicleControl()
        if not self.target:
            return control

        # 基础自动行驶逻辑
        transform = self.vehicle.get_transform()
        forward = transform.get_forward_vector()
        target_dir = self.target - transform.location
        target_dir.z = 0

        # 计算方向
        dot = forward.x * target_dir.x + forward.y * target_dir.y
        cross = forward.x * target_dir.y - forward.y * target_dir.x
        distance = math.sqrt(target_dir.x ** 2 + target_dir.y ** 2)

        # 控制
        if distance > 5.0:
            control.throttle = 0.5
            control.steer = max(-0.3, min(0.3, cross * 0.15))
        else:
            control.brake = 1.0

        return control


# ===================== 以下是完整 CARLA 运行代码 =====================
def find_weather_presets():
    presets = []
    for name in dir(carla.WeatherParameters):
        if name[0].isupper():
            presets.append((getattr(carla.WeatherParameters, name), name))
    return presets


def get_actor_display_name(actor):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return name


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)

    def restart(self, args):
        blueprint_library = self.world.get_blueprint_library()
        blueprint = random.choice(blueprint_library.filter(self._actor_filter))
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
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.set_sensor(0)

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
        actors = [self.camera_manager.sensor, self.collision_sensor.sensor,
                  self.lane_invasion_sensor.sensor, self.player]
        for actor in actors:
            if actor and actor.is_alive:
                actor.destroy()


class CollisionSensor(object):
    def __init__(self, parent, hud):
        self.sensor = parent.get_world().spawn_actor(
            parent.get_world().get_blueprint_library().find('sensor.other.collision'),
            carla.Transform(), attach_to=parent)
        self.hud = hud
        self.sensor.listen(lambda e: self.hud.notification(f"碰撞: {get_actor_display_name(e.other_actor)}"))


class LaneInvasionSensor(object):
    def __init__(self, parent, hud):
        self.sensor = parent.get_world().spawn_actor(
            parent.get_world().get_blueprint_library().find('sensor.other.lane_invasion'),
            carla.Transform(), attach_to=parent)
        self.hud = hud
        self.sensor.listen(lambda e: self.hud.notification("车道偏离"))


class CameraManager(object):
    def __init__(self, parent, hud):
        self.sensor = None
        self.surface = None
        self.parent = parent
        self.hud = hud
        bp = parent.get_world().get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(hud.dim[0]))
        bp.set_attribute('image_size_y', str(hud.dim[1]))
        self.bp = bp

    def set_sensor(self, index):
        if self.sensor: self.sensor.destroy()
        v = self.parent
        ext = v.bounding_box.extent
        trans = carla.Transform(carla.Location(x=-ext.x * 2.5, z=ext.z * 2), carla.Rotation(pitch=8))
        self.sensor = v.get_world().spawn_actor(self.bp, trans, attach_to=v)
        self.sensor.listen(lambda img: self._parse(img))

    def _parse(self, image):
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape(image.height, image.width, 4)
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def render(self, display):
        if self.surface: display.blit(self.surface, (0, 0))


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self._notifications = FadingText(self.font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.speed = 0

    def on_world_tick(self, timestamp):
        self.server_fps = 1.0 / max(timestamp.delta_seconds, 0.01)

    def tick(self, clock, world):
        self._notifications.tick(clock)
        if world.player:
            v = world.player.get_velocity()
            self.speed = round(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2), 1)

    def notification(self, text, seconds=2):
        self._notifications.set_text(text, seconds=seconds)

    def render(self, display):
        info = [f"FPS: {self.server_fps:.1f}", f"车速: {self.speed:.1f} km/h", "状态: 自动行驶中"]
        for i, line in enumerate(info):
            surf = self.font.render(line, True, (255, 255, 255))
            display.blit(surf, (10, 10 + i * 20))
        self._notifications.render(display)


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = font.render("", True, (0, 0, 0))

    def set_text(self, text, color=(255, 255, 255), seconds=2):
        self.seconds_left = seconds
        self.surface = self.font.render(text, True, color)

    def tick(self, clock):
        self.seconds_left = max(0.0, self.seconds_left - clock.get_time() / 1000)

    def render(self, display):
        alpha = int(255 * self.seconds_left / 2) if self.seconds_left else 0
        self.surface.set_alpha(alpha)
        display.blit(self.surface, self.pos)


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("ESC 退出", 2)
        world.hud.notification("PageUp/Down 切换天气", 2)

    def parse_events(self, world):
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
                return True
            if event.type == KEYUP:
                if event.key == K_PAGEUP:
                    world.next_weather(True)
                elif event.key == K_PAGEDOWN:
                    world.next_weather()
        return False


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        display = pygame.display.set_mode((args.width, args.height))
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        # 使用我们自己的极简自动代理（无任何依赖）
        agent = SimpleAutoAgent(world.player)
        dest = random.choice(world.map.get_spawn_points()).location
        agent.set_destination(dest)
        hud.notification("自动导航已启动")

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(world): return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            control = agent.run_step()
            world.player.apply_control(control)

    finally:
        if world: world.destroy()
        pygame.quit()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--host', default='127.0.0.1')
    argparser.add_argument('--port', default=2000, type=int)
    argparser.add_argument('--res', default='1280x720')
    argparser.add_argument('--filter', default='vehicle.*')
    args = argparser.parse_args()
    args.width, args.height = map(int, args.res.split('x'))
    game_loop(args)


if __name__ == '__main__':
    main()
