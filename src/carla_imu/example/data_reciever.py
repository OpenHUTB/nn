"""
CARLA IMU 采集 + 中英文界面 + 车辆状态实时显示
空格键：切换中英文
ESC：退出
鼠标拖拽：旋转视角
"""

from __future__ import print_function

import pandas as pd
import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import math
import random

try:
    import pygame
    from pygame.locals import *
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# ====================== 多语言配置 ======================
LANG = "en"
TEXTS = {
    "en": {
        "server": "Server",
        "client": "Client",
        "vehicle": "Vehicle",
        "map": "Map",
        "speed": "Speed",
        "autopilot": "AUTOPILOT ENABLED",
        "fps": "FPS",
        "kmh": "km/h",
        "status": "Status",
        "straight": "Straight",
        "accelerate": "Accelerate",
        "brake": "Brake",
        "left": "Turn Left",
        "right": "Turn Right",
        "reverse": "Reverse",
        "handbrake": "Handbrake"
    },
    "cn": {
        "server": "服务器",
        "client": "客户端",
        "vehicle": "车辆",
        "map": "地图",
        "speed": "速度",
        "autopilot": "自动驾驶已启用",
        "fps": "帧率",
        "kmh": "千米/小时",
        "status": "车辆状态",
        "straight": "直行",
        "accelerate": "加速",
        "brake": "刹车",
        "left": "左转",
        "right": "右转",
        "reverse": "倒车",
        "handbrake": "手刹"
    }
}

def t(key):
    return TEXTS[LANG].get(key, key)

# ====================== 工具函数 ======================
def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate-1] + "…") if len(name) > truncate else name

# ====================== 世界管理 ======================
class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.camera_manager = None
        self._actor_filter = args.filter
        self.restart()

    def restart(self):
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', "hero")
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        if self.player:
            self.player.destroy()

        spawn_points = self.map.get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()
        self.player = self.world.spawn_actor(blueprint, spawn_point)
        self.player.set_autopilot(True)

        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player)
        self.hud.notification(t("autopilot"))

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        actors = [self.player, self.imu_sensor.sensor, self.camera_manager.sensor]
        for a in actors:
            if a:
                a.destroy()

# ====================== 按键控制 ======================
class KeyboardControl(object):
    def parse_events(self, world):
        global LANG
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if event.key == K_ESCAPE:
                    return True
                if event.key == K_SPACE:
                    LANG = "cn" if LANG == "en" else "en"
                    tip = "已切换为中文" if LANG=="cn" else "Switched to English"
                    world.hud.notification(tip)
            if event.type == MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                dx, dy = event.rel
                world.camera_manager.yaw += dx * 0.2
                world.camera_manager.pitch = max(-60, min(60, world.camera_manager.pitch - dy * 0.2))
        return False

# ====================== HUD界面（带车辆状态） ======================
class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font_path = r"C:\Windows\Fonts\msyh.ttc"
        self._font = pygame.font.Font(font_path, 14)
        self.server_fps = 0
        self._clock = pygame.time.Clock()
        self.notice_text = ""
        self.notice_time = 0
        self.vehicle_status = t("straight")

    def notification(self, text, dur=2.0):
        self.notice_text = text
        self.notice_time = dur

    def tick(self, world, clock):
        self._clock.tick()
        self.server_fps = self._clock.get_fps()
        if self.notice_time > 0:
            self.notice_time -= 0.016

        # --- 车辆状态识别 ---
        v = world.player.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        control = world.player.get_control()
        handbrake = control.hand_brake
        reverse = control.reverse
        throttle = control.throttle
        brake = control.brake
        steer = control.steer

        status = t("straight")

        if handbrake:
            status = t("handbrake")
        elif reverse:
            status = t("reverse")
        elif brake > 0.1:
            status = t("brake")
        elif throttle > 0.1:
            status = t("accelerate")
        elif steer < -0.1:
            status = t("right")
        elif steer > 0.1:
            status = t("left")

        self.vehicle_status = status

        # --- 界面信息 ---
        self._info_text = [
            f"{t('server')}: {self.server_fps:.1f} {t('fps')}",
            f"{t('client')}: {clock.get_fps():.1f} {t('fps')}",
            "",
            f"{t('vehicle')}: {get_actor_display_name(world.player,20)}",
            f"{t('map')}: {world.map.name}",
            f"{t('speed')}: {speed:.1f} {t('kmh')}",
            "",
            f"{t('status')}: {self.vehicle_status}",  # 这里显示车辆状态
        ]

    def render(self, display):
        bg = pygame.Surface((280, self.dim[1]))
        bg.set_alpha(90)
        display.blit(bg, (0,0))

        y = 8
        for line in self._info_text:
            surf = self._font.render(line, True, (255,255,255))
            display.blit(surf, (12, y))
            y += 22

        if self.notice_time > 0:
            tip = self._font.render(self.notice_text, True, (255,255,0))
            display.blit(tip, (self.dim[0]//2 - 110, 80))

# ====================== IMU传感器 ======================
class IMUSensor:
    def __init__(self, parent):
        self.sensor = parent.get_world().spawn_actor(
            parent.get_world().get_blueprint_library().find("sensor.other.imu"),
            carla.Transform(), attach_to=parent)
        self.accelerometer = (0.0,0.0,0.0)
        self.gyroscope = (0.0,0.0,0.0)
        self.sensor.listen(self.callback)
    def callback(self, data):
        self.accelerometer = (data.accelerometer.x, data.accelerometer.y, data.accelerometer.z)
        self.gyroscope = (data.gyroscope.x, data.gyroscope.y, data.gyroscope.z)

# ====================== 摄像头 ======================
class CameraManager:
    def __init__(self, parent):
        self._parent = parent
        self.sensor = None
        self.surface = None
        self.yaw = 0.0
        self.pitch = 10.0
        self.dist = 5.5
        self.reset_cam()

    def reset_cam(self):
        if self.sensor:
            self.sensor.destroy()
        bp = self._parent.get_world().get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x","1280")
        bp.set_attribute("image_size_y","720")
        trans = carla.Transform(carla.Location(x=-self.dist,z=2.3), carla.Rotation(pitch=self.pitch,yaw=self.yaw))
        self.sensor = self._parent.get_world().spawn_actor(bp, trans, attach_to=self._parent, attachment_type=carla.AttachmentType.SpringArm)
        self.sensor.listen(self.on_img)

    def on_img(self, img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height,img.width,4)[:,:,:3]
        self.surface = pygame.surfarray.make_surface(arr.swapaxes(0,1))

    def render(self, display):
        if self.surface:
            display.blit(self.surface, (0,0))
        new_trans = carla.Transform(carla.Location(x=-self.dist,z=2.3), carla.Rotation(pitch=self.pitch,yaw=self.yaw))
        self.sensor.set_transform(new_trans)

# ====================== 主循环 ======================
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    data = pd.DataFrame()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(3.0)
        display = pygame.display.set_mode((1280,720), pygame.HWSURFACE|pygame.DOUBLEBUF)
        hud = HUD(1280,720)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl()
        clock = pygame.time.Clock()

        while True:
            clock.tick(60)
            if controller.parse_events(world):
                break
            world.tick(clock)
            world.render(display)

            data = pd.concat([data, pd.DataFrame([{
                "accelX":world.imu_sensor.accelerometer[0],
                "accelY":world.imu_sensor.accelerometer[1],
                "accelZ":world.imu_sensor.accelerometer[2],
                "gyroX":world.imu_sensor.gyroscope[0],
                "gyroY":world.imu_sensor.gyroscope[1],
                "gyroZ":world.imu_sensor.gyroscope[2],
                "class":args.name
            }])], ignore_index=True)
            pygame.display.flip()

    finally:
        if world:
            world.destroy()
        data.to_csv(f"out_{args.name}.csv", index=False)
        print(f"✅ CSV 已保存: out_{args.name}.csv")
        pygame.quit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",default=2000,type=int)
    parser.add_argument("--filter",default="vehicle.*")
    parser.add_argument("--name",default="mehdi")
    args = parser.parse_args()
    game_loop(args)

if __name__ == "__main__":
    main()
