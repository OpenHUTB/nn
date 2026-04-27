#!/usr/bin/env python

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
import threading

# --------------- Python 3.9 专用修复 ---------------
# 清理冲突路径
for path in list(sys.path):
    if "carla/dist" in path and path.endswith(".egg"):
        sys.path.remove(path)

# 正确添加 CARLA PythonAPI 路径
CARLA_ROOT = r"D:\carla0.9.15"
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI"))
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))

# ---------------- 依赖 ----------------
try:
    import pygame
    from pygame.locals import *
except ImportError:
    raise RuntimeError('请安装: pip install pygame')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('请安装: pip install numpy')

import carla
from carla import ColorConverter as cc

# 导入自动驾驶代理
from agents.navigation.behavior_agent import BehaviorAgent


# ==============================================================================
# -- 工具函数
# ==============================================================================
def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def play_warning_sound():
    """车道偏离语音提醒（异步播放不卡顿）"""
    def play():
        try:
            pygame.mixer.init()
            warning = pygame.mixer.Sound("warning.wav")
            warning.play()
        except:
            pass
    threading.Thread(target=play, daemon=True).start()


# ==============================================================================
# -- World 场景管理
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
        self._actor_filter = args.filter
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)

    def restart(self, args):
        # 清理旧车辆
        if self.player is not None:
            self.destroy()

        # 生成车辆
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')

        while self.player is None:
            spawn_point = random.choice(self.map.get_spawn_points())
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        # 传感器
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.set_sensor(0)

    def tick(self, clock):
        self.hud.tick(clock)
        # 更新车速
        v = self.player.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        self.hud.speed = int(speed)

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
# -- 键盘控制（新增 P 键切换自动驾驶）
# ==============================================================================
class KeyboardControl(object):
    def __init__(self):
        self.auto_pilot = True

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if event.key == K_ESCAPE or (event.key == K_q and pygame.key.get_mod() & KMOD_CTRL):
                    return True
                if event.key == K_p:
                    self.auto_pilot = not self.auto_pilot
        return False


# ==============================================================================
# -- HUD 显示（新增车速 + 模式显示）
# ==============================================================================
class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.font_big = pygame.font.Font(pygame.font.get_default_font(), 32)
        self._notifications = FadingText(self.font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.speed = 0

    def on_world_tick(self, timestamp):
        self.server_fps = 1.0 / max(timestamp.delta_seconds, 0.01)

    def tick(self, clock):
        self._notifications.tick(clock)

    def notification(self, text, seconds=2):
        self._notifications.set_text(text, seconds=seconds)

    def render(self, display):
        # 车速
        speed_text = self.font_big.render(f"{self.speed} km/h", True, (0,255,255))
        display.blit(speed_text, (20,20))
        self._notifications.render(display)


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2):
        self.seconds_left = seconds
        self.surface = self.font.render(text, True, color)

    def tick(self, clock):
        self.seconds_left = max(0, self.seconds_left - clock.get_time() / 1000)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- 传感器
# ==============================================================================
class CollisionSensor(object):
    def __init__(self, parent, hud):
        self.sensor = parent.get_world().spawn_actor(
            parent.get_world().get_blueprint_library().find('sensor.other.collision'),
            carla.Transform(), attach_to=parent)
        self.sensor.listen(lambda e: hud.notification(f"碰撞: {get_actor_display_name(e.other_actor)}"))


class LaneInvasionSensor(object):
    def __init__(self, parent, hud):
        self.sensor = parent.get_world().spawn_actor(
            parent.get_world().get_blueprint_library().find('sensor.other.lane_invasion'),
            carla.Transform(), attach_to=parent)
        self.sensor.listen(lambda e: self._on_lane(e, hud))

    def _on_lane(self, event, hud):
        hud.notification("车道偏离！")
        play_warning_sound()


class GnssSensor(object):
    def __init__(self, parent):
        self.sensor = parent.get_world().spawn_actor(
            parent.get_world().get_blueprint_library().find('sensor.other.gnss'),
            carla.Transform(carla.Location(z=2.0)), attach_to=parent)


# ==============================================================================
# -- 相机
# ==============================================================================
class CameraManager(object):
    def __init__(self, parent, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent
        self.hud = hud
        self.index = 0

    def set_sensor(self, index):
        if self.sensor: self.sensor.destroy()
        bp = self._parent.get_world().get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.hud.dim[0]))
        bp.set_attribute('image_size_y', str(self.hud.dim[1]))

        transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8))
        self.sensor = self._parent.get_world().spawn_actor(
            bp, transform, attach_to=self._parent, attachment_type=carla.AttachmentType.SpringArm)
        self.sensor.listen(lambda img: self._parse(img))

    def _parse(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape(image.height, image.width, 4)[:, :, :3]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1)[:, :, ::-1])

    def render(self, display):
        if self.surface: display.blit(self.surface, (0, 0))


# ==============================================================================
# -- 主循环
# ==============================================================================
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    controller = KeyboardControl()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        display = pygame.display.set_mode((args.width, args.height))
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)

        # 自动导航
        agent = BehaviorAgent(world.player, behavior=args.behavior)
        spawn_points = world.map.get_spawn_points()
        target = random.choice(spawn_points).location
        agent.set_destination(target)
        hud.notification("自动导航已启动", 3)

        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events():
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            # 自动驾驶开关
            if controller.auto_pilot:
                control = agent.run_step()
                world.player.apply_control(control)
                hud.notification("模式：自动驾驶", 1)
                # 到达目标换新
                if agent.done():
                    new_target = random.choice(spawn_points).location
                    agent.set_destination(new_target)
            else:
                hud.notification("模式：手动驾驶", 1)

    finally:
        if world:
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- 主函数
# ==============================================================================
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--host', default='127.0.0.1')
    argparser.add_argument('--port', default=2000, type=int)
    argparser.add_argument('--res', default='1280x720')
    argparser.add_argument('--filter', default='vehicle.*')
    argparser.add_argument('--behavior', default='normal')
    args = argparser.parse_args()
    args.width, args.height = map(int, args.res.split('x'))
    game_loop(args)


if __name__ == '__main__':
    main()
