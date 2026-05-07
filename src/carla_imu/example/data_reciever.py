"""
CARLA IMU 采集 + 中英文界面 + 车辆状态实时显示 + 打开输出文件夹按钮
空格键：切换中英文
ESC：退出
鼠标拖拽：旋转视角
点击左侧【打开输出文件】按钮：直接打开CSV保存目录
点击【手动/自动驾驶】按钮：切换驾驶模式
手动模式：↑加速 ↓刹车 ←左转 →右转 底部按钮倒车
"""

from __future__ import print_function

import pandas as pd
import glob
import os
import sys
import subprocess
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
        "autopilot": "AUTOPILOT",
        "manual": "MANUAL",
        "fps": "FPS",
        "kmh": "km/h",
        "status": "Status",
        "straight": "Straight",
        "accelerate": "Accelerate",
        "brake": "Brake",
        "left": "Turn Left",
        "right": "Turn Right",
        "reverse": "Reverse",
        "handbrake": "Handbrake",
        "open_file": "Open Output Folder",
        "switch": "Switch Mode",
        "rev_btn": "REV"
    },
    "cn": {
        "server": "服务器",
        "client": "客户端",
        "vehicle": "车辆",
        "map": "地图",
        "speed": "速度",
        "autopilot": "自动驾驶",
        "manual": "手动驾驶",
        "fps": "帧率",
        "kmh": "千米/小时",
        "status": "车辆状态",
        "straight": "直行",
        "accelerate": "加速",
        "brake": "刹车",
        "left": "左转",
        "right": "右转",
        "reverse": "倒车",
        "handbrake": "手刹",
        "open_file": "打开输出文件",
        "switch": "切换模式",
        "rev_btn": "倒车"
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
        self.imu_sensor = None
        self.camera_manager = None
        self._actor_filter = args.filter
        self.autopilot = True
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
        self.player.set_autopilot(self.autopilot)

        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player)
        self.hud.notification(t("autopilot") if self.autopilot else t("manual"))

    def toggle_mode(self):
        self.autopilot = not self.autopilot
        self.player.set_autopilot(self.autopilot)
        self.hud.notification(t("autopilot") if self.autopilot else t("manual"))

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display, self)

    def destroy(self):
        actors = [self.player, self.imu_sensor.sensor, self.camera_manager.sensor]
        for a in actors:
            if a:
                a.destroy()

# ====================== 按键控制 ======================
class KeyboardControl(object):
    def __init__(self):
        self.control = carla.VehicleControl()

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

            if event.type == pygame.MOUSEBUTTONDOWN:
                if world.hud.mode_btn.collidepoint(event.pos):
                    world.toggle_mode()
                if world.hud.open_btn.collidepoint(event.pos):
                    path = os.path.abspath(".")
                    try:
                        if os.name == "nt": os.startfile(path)
                        else: subprocess.Popen(["open", path])
                        world.hud.notification("文件夹已打开")
                    except:
                        world.hud.notification("打开失败")

                if not world.autopilot:
                    self.control.hand_brake = False
                    self.control.throttle = 0.0
                    self.control.brake = 0.0
                    self.control.steer = 0.0
                    self.control.reverse = False

                    if world.hud.btn_up.collidepoint(event.pos):
                        self.control.throttle = 0.8
                    elif world.hud.btn_down.collidepoint(event.pos):
                        self.control.brake = 1.0
                    elif world.hud.btn_left.collidepoint(event.pos):
                        self.control.steer = 0.4
                    elif world.hud.btn_right.collidepoint(event.pos):
                        self.control.steer = -0.4
                    # 倒车按钮
                    elif world.hud.btn_rev.collidepoint(event.pos):
                        self.control.reverse = True
                        self.control.throttle = 0.6

            if event.type == pygame.MOUSEBUTTONUP:
                self.control.throttle = 0.0
                self.control.brake = 0.0
                self.control.steer = 0.0
                self.control.reverse = False

            if event.type == MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                dx, dy = event.rel
                world.camera_manager.yaw += dx * 0.2
                world.camera_manager.pitch = max(-60, min(60, world.camera_manager.pitch - dy * 0.2))

        if not world.autopilot:
            world.player.apply_control(self.control)
        return False

# ====================== HUD界面 ======================
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

        self.open_btn = pygame.Rect(10, height - 50, 200, 35)
        self.mode_btn = pygame.Rect(10, height - 100, 200, 35)

        btn_size = 70
        cx = width // 2
        by = height - 130

        self.btn_up    = pygame.Rect(cx-btn_size//2, by-90, btn_size, btn_size)
        self.btn_left  = pygame.Rect(cx-btn_size-20, by, btn_size, btn_size)
        self.btn_right = pygame.Rect(cx+20, by, btn_size, btn_size)
        self.btn_down  = pygame.Rect(cx-btn_size//2, by, btn_size, btn_size)
        # 倒车按钮
        self.btn_rev   = pygame.Rect(cx-btn_size//2, by+90, btn_size+20, btn_size)

    def notification(self, text, dur=2.0):
        self.notice_text = text
        self.notice_time = dur

    def tick(self, world, clock):
        self._clock.tick()
        self.server_fps = self._clock.get_fps()
        if self.notice_time > 0:
            self.notice_time -= 0.016

        v = world.player.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        control = world.player.get_control()

        status = t("straight")
        if control.hand_brake:
            status = t("handbrake")
        elif control.reverse:
            status = t("reverse")
        elif control.brake > 0.1:
            status = t("brake")
        elif control.throttle > 0.1:
            status = t("accelerate")
        elif control.steer < -0.1:
            status = t("right")
        elif control.steer > 0.1:
            status = t("left")

        self.vehicle_status = status

        mode_text = t("autopilot") if world.autopilot else t("manual")
        self._info_text = [
            f"{t('server')}: {self.server_fps:.1f} {t('fps')}",
            f"{t('client')}: {clock.get_fps():.1f} {t('fps')}",
            "",
            f"{t('vehicle')}: {get_actor_display_name(world.player,20)}",
            f"{t('map')}: {world.map.name}",
            f"{t('speed')}: {speed:.1f} {t('kmh')}",
            f"Mode: {mode_text}",
            "",
            f"{t('status')}: {self.vehicle_status}",
        ]

    def render(self, display, world):
        bg = pygame.Surface((280, self.dim[1]))
        bg.set_alpha(90)
        display.blit(bg, (0,0))

        y = 8
        for line in self._info_text:
            surf = self._font.render(line, True, (255,255,255))
            display.blit(surf, (12, y))
            y += 22

        pygame.draw.rect(display, (0,160,90), self.mode_btn)
        pygame.draw.rect(display, (255,255,255), self.mode_btn, 2)
        txt = self._font.render(t("switch"), True, (255,255,255))
        display.blit(txt, txt.get_rect(center=self.mode_btn.center))

        pygame.draw.rect(display, (0,120,210), self.open_btn)
        pygame.draw.rect(display, (255,255,255), self.open_btn, 2)
        btn_text = self._font.render(t("open_file"), True, (255,255,255))
        display.blit(btn_text, btn_text.get_rect(center=self.open_btn.center))

        if not world.autopilot:
            c = (200,200,200)
            bw = 2
            pygame.draw.rect(display, c, self.btn_up, bw)
            pygame.draw.rect(display, c, self.btn_left, bw)
            pygame.draw.rect(display, c, self.btn_right, bw)
            pygame.draw.rect(display, c, self.btn_down, bw)
            pygame.draw.rect(display, (255,80,80), self.btn_rev, bw)

            up = self._font.render("↑", True, (255,255,255))
            left = self._font.render("←", True, (255,255,255))
            right = self._font.render("→", True, (255,255,255))
            down = self._font.render("↓", True, (255,255,255))
            rev = self._font.render(t("rev_btn"), True, (255,255,255))

            display.blit(up, up.get_rect(center=self.btn_up.center))
            display.blit(left, left.get_rect(center=self.btn_left.center))
            display.blit(right, right.get_rect(center=self.btn_right.center))
            display.blit(down, down.get_rect(center=self.btn_down.center))
            display.blit(rev, rev.get_rect(center=self.btn_rev.center))

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
        self.gyroscope = (math.degrees(data.gyroscope.x), math.degrees(data.gyroscope.y), math.degrees(data.gyroscope.z))

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
