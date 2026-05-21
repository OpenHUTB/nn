from __future__ import print_function

"""Example of automatic vehicle control from client side."""
"CARLA-Native-Assisted-Driving-System "
import argparse
import collections
import datetime
import glob
import math
import os
import random
import re
import sys
import weakref
import cv2
import pygame
from pygame.locals import *
import numpy as np

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


# ==============================================================================
# -- PID Controller ------------------------------------------------------------
# ==============================================================================
class PIDController:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error = 0.0
        self.last_error = 0.0
        self.integral = 0.0

    def step(self, target, current, dt=0.05):
        self.error = target - current
        self.integral += self.error * dt
        self.integral = np.clip(self.integral, -10, 10)
        derivative = (self.error - self.last_error) / dt
        output = self.Kp * self.error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = self.error
        return max(0.0, min(1.0, output))


def find_weather_presets():
    # 驼峰命名拆分正则（不变，用来美化名字）
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?=[A-Z])(?=[A-Z][a-z])|$)')

    def name(x):
        return ' '.join(m.group(0) for m in rgx.finditer(x))

    weather_class = carla.WeatherParameters
    presets = []
    for attr_name in dir(weather_class):
        # 筛选规则：大写开头 + 不是私有属性(__开头) + 不是方法/函数
        if attr_name[0].isupper() and not attr_name.startswith('__'):
            attr_value = getattr(weather_class, attr_name)
            # 额外校验：确保是天气参数对象（排除无效属性）
            if isinstance(attr_value, carla.WeatherParameters):
                presets.append(attr_name)

    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
def traffic_light_detect(image):
    # 空图像保护
    if image is None:
        return 'none'

    h, w = image.shape[:2]
    roi = image[int(h * 0.1):int(h * 0.25), int(w * 0.42):int(w * 0.58)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 150])
    upper_red2 = np.array([180, 255, 255])

    # 黄灯阈值完全保留（之前标定完美，不动）
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([27, 255, 255])

    # 绿灯阈值完全保留（之前黄绿分割完美，不动）
    lower_green = np.array([28, 100, 100])
    upper_green = np.array([55, 255, 255])

    # 掩码生成
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 形态学开运算去噪点
    kernel = np.ones((2, 2), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    # 有效发光像素统计
    red_cnt = cv2.countNonZero(mask_red)
    yellow_cnt = cv2.countNonZero(mask_yellow)
    green_cnt = cv2.countNonZero(mask_green)

    detect_threshold = 60

    if red_cnt > detect_threshold:
        return 'red'
    elif yellow_cnt > detect_threshold:
        return 'yellow'
    elif green_cnt > detect_threshold:
        return 'green'
    else:
        return 'none'


# ==============================================================================
# 【新增】交通标志识别（停车牌 / 限速牌）- 复用红绿灯视觉逻辑
# ==============================================================================
def traffic_sign_detect(image):
    if image is None:
        return 'none'

    h, w = image.shape[:2]
    # 扩大ROI：覆盖道路上方的交通标志区域（比红绿灯ROI更大）
    roi = image[int(h * 0.05):int(h * 0.35), int(w * 0.35):int(w * 0.65)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # 1. 停车标志：红色（和红绿灯红色阈值一致）
    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 150])
    upper_red2 = np.array([180, 255, 255])
    mask_stop = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # 2. 限速标志：CARLA标准蓝色（HSV蓝色区间）
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask_speed = cv2.inRange(hsv, lower_blue, upper_blue)

    # 形态学去噪
    kernel = np.ones((3, 3), np.uint8)
    mask_stop = cv2.morphologyEx(mask_stop, cv2.MORPH_OPEN, kernel)
    mask_speed = cv2.morphologyEx(mask_speed, cv2.MORPH_OPEN, kernel)

    # 像素计数阈值
    stop_cnt = cv2.countNonZero(mask_stop)
    speed_cnt = cv2.countNonZero(mask_speed)
    sign_threshold = 80

    # 识别逻辑：停车牌优先级 > 限速牌
    if stop_cnt > sign_threshold:
        return 'stop'
    elif speed_cnt > sign_threshold:
        # 简化版：CARLA常见限速 30/50/60，可根据地图调整
        return 'speed_50'
    else:
        return 'none'


# ==============================================================================
#  行人检测 AEB
# ==============================================================================
def pedestrian_detect(image):
    if image is None:
        return False

    h, w = image.shape[:2]
    # 严格缩小ROI：只看车辆正前方路面，砍掉天空、路边建筑、围墙干扰
    roi = image[int(h * 0.6):int(h * 0.9), int(w * 0.4):int(w * 0.6)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # 极度收紧行人肤色阈值，过滤路面/黄土/墙壁
    lower_ped = np.array([0, 80, 120])
    upper_ped = np.array([12, 130, 200])

    mask = cv2.inRange(hsv, lower_ped, upper_ped)

    # 强力降噪
    kernel = np.ones((6, 6), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 只有足够大的物体才判定为行人
        if area > 600:
            return True
    return False
# ==============================================================================
# 【新增】天气自适应控制系统 - 自动车灯 + 自动相机参数
# ==============================================================================
# 1. 判断当前天气类型（黑夜/雨天/雾天/晴天）
def get_weather_type(world):
    weather = world.get_weather()
    # 光照强度 (0=黑夜, 100=大晴天)
    illumination = weather.sun_altitude_angle
    # 雨/雾强度
    rain = weather.precipitation
    fog = weather.fog_density

    if illumination < 5:
        return "night"  # 黑夜
    elif rain > 30:
        return "rain"  # 雨天
    elif fog > 40:
        return "fog"  # 雾天
    else:
        return "sunny"  # 晴天


# 2. 自动控制车辆车灯（近光灯 + 雾灯）
# 2. 自动控制车辆车灯（CARLA官方正确写法，无报错版）
def auto_vehicle_lights(vehicle, weather_type):
    light_state = carla.VehicleLightState()

    if weather_type == "night":
        # 黑夜：开启近光灯
        light_state = carla.VehicleLightState.LowBeam
    elif weather_type == "rain":
        # 雨天：近光灯 + 雾灯
        light_state = carla.VehicleLightState.LowBeam | carla.VehicleLightState.Fog
    elif weather_type == "fog":
        # 雾天：雾灯
        light_state = carla.VehicleLightState.Fog
    else:
        # 晴天：关闭所有灯光
        light_state = carla.VehicleLightState.NONE
    vehicle.set_light_state(light_state)


# 3. 自动调整相机参数（曝光/对比度）
def auto_camera_settings(camera_manager, weather_type):
    if camera_manager.sensor is None:
        return
    sensor = camera_manager.sensor
    # 只对RGB相机调整参数
    if 'sensor.camera.rgb' in sensor.type_id:
        if weather_type == "night":
            # 黑夜：提高曝光，提亮画面
            sensor.set_attribute('exposure_mode', 'manual')
            sensor.set_attribute('exposure_compensation', '3.0')
            sensor.set_attribute('contrast', '1.4')
        elif weather_type == "fog":
            # 雾天：提高对比度，看清道路
            sensor.set_attribute('exposure_compensation', '1.5')
            sensor.set_attribute('contrast', '1.6')
            sensor.set_attribute('saturation', '1.2')
        elif weather_type == "rain":
            # 雨天：柔和参数
            sensor.set_attribute('exposure_compensation', '1.0')
            sensor.set_attribute('contrast', '1.2')
        else:
            # 晴天：默认参数
            sensor.set_attribute('exposure_mode', 'auto')
            sensor.set_attribute('exposure_compensation', '0.0')
            sensor.set_attribute('contrast', '1.0')
            sensor.set_attribute('saturation', '1.0')


# -- World --------------------------------------------------------------------
# ==============================================================================
class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = 1
        if args.seed is not None:
            random.seed(args.seed)

        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

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
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

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
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

        # 【新增】LDW车道偏离预警状态
        self.ldw_warning = False
        self.ldw_warning_duration = 1.0  # 警告持续1秒
        self.ldw_timer = 0.0
        # 预警大字体
        self.ldw_font = pygame.font.Font(mono, 48)

    # 【新增FCW】前向碰撞预警状态
        self.fcw_warning = False
        self.fcw_warning_duration = 0.8
        self.fcw_timer = 0.0
        self.fcw_font = pygame.font.Font(mono, 52)
        # ===================== 【新增：BSD盲区监测】 =====================
        self.bsd_warning = False
        self.bsd_side = ""  # 记录左侧/右侧
        self.bsd_warning_duration = 0.8
        self.bsd_timer = 0.0
        self.bsd_font = pygame.font.Font(mono, 46)
    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return

        # 【新增】更新LDW警告计时器
        delta_seconds = 1e-3 * clock.get_time()
        if self.ldw_warning:
            self.ldw_timer -= delta_seconds
            if self.ldw_timer <= 0:
                self.ldw_warning = False
        # 【新增FCW】更新碰撞预警计时器
        if self.fcw_warning:
            self.fcw_timer -= delta_seconds
            if self.fcw_timer <= 0:
                self.fcw_warning = False
            # ===================== 【新增：BSD计时器】 =====================
        if self.bsd_warning:
            self.bsd_timer -= delta_seconds
            if self.bsd_timer <= 0:
                self.bsd_warning = False

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

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    # 【新增】触发LDW车道偏离警告
    def trigger_ldw_warning(self):
        self.ldw_warning = True
        self.ldw_timer = self.ldw_warning_duration

    # 【新增FCW】触发前向碰撞警告
    def trigger_fcw_warning(self):
        self.fcw_warning = True
        self.fcw_timer = self.fcw_warning_duration

    def render(self, display):
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
                            rect = pygame.Rect((bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

        # 【新增】绘制LDW车道偏离警告（屏幕中央红色大字）
        if self.ldw_warning:
            warning_text = self.ldw_font.render("车道偏离警告！", True, (255, 0, 0))
            text_rect = warning_text.get_rect(center=(self.dim[0] // 2, self.dim[1] // 2))
            display.blit(warning_text, text_rect)
        # 【新增FCW】绘制前向碰撞警告
        if self.fcw_warning:
            warning_text = self.fcw_font.render("碰撞危险！", True, (255, 0, 0))
            text_rect = warning_text.get_rect(center=(self.dim[0] // 2, self.dim[1] // 2 + 80))
            display.blit(warning_text, text_rect)

            # ===================== 【新增：绘制BSD盲区警告】 =====================
        if self.bsd_warning:
            warning_text = self.bsd_font.render(f"{self.bsd_side}盲区有车！", True, (0, 255, 255))
            text_rect = warning_text.get_rect(center=(self.dim[0] // 2, self.dim[1] // 2 - 80))
            display.blit(warning_text, text_rect)
# ==============================================================================
# -- 传感器基础类 --------------------------------------------------------------
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


class HelpText(object):
    def __init__(self, font, width, height):
        lines = [
            "CARLA-Native-Assisted-Driving-System",
            "PID定速巡航 | 精准红绿灯识别 | 交通标志识别",
            "按键: 1/2/3/4/5 切换视角 | C/V切换天气 | ESC 退出"
        ]
        self.font = font
        self.dim = (720, len(lines) * 22 + 12)
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
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


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
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


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
        # 【新增】触发LDW预警（屏幕警告+提示音）
        self.hud.trigger_ldw_warning()
        try:
            self.hud.ldw_sound.play()  # 播放提示音
        except:
            pass

        # 原有提示保留
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


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


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.rgb_image = None
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)
        ]
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
                    # 新增：默认曝光模式
                if blp.has_attribute('exposure_mode'):
                    blp.set_attribute('exposure_mode', 'auto')
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def set_camera_view(self, index):
        self.transform_index = index % len(self._camera_transforms)
        self.set_sensor(self.index, notify=True, force_respawn=True)

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
        self.rgb_image = array.copy()
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- 主循环 --------------------------------------------------------------------
# ==============================================================================
def game_loop(args):
    pygame.init()
    pygame.font.init()
    # 【新增】初始化Pygame音频模块
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
    world = None

    pid = PIDController(Kp=1.0, Ki=0.2, Kd=0.1)
    steer_pid = PIDController(Kp=0.3, Ki=0.0, Kd=0.05)
    target_speed = 30.0
    # 【新增】默认巡航速度（未识别到限速标志时使用）
    DEFAULT_CRUISE_SPEED = 30.0

    # 【新增】生成LDW预警提示音（无外部文件依赖）
    def generate_ldw_beep():
        sample_rate = 22050
        duration = 0.15
        freq = 900
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * freq * t)
        wave = (wave * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        return pygame.sndarray.make_sound(stereo_wave)

    ldw_warning_sound = generate_ldw_beep()

    # 【新增FCW】生成碰撞预警提示音
    def generate_fcw_beep():
        sample_rate = 22050
        duration = 0.2
        freq = 1200
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * freq * t)
        wave = (wave * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        return pygame.sndarray.make_sound(stereo_wave)

    fcw_warning_sound = generate_fcw_beep()

    # ===================== 【新增：BSD盲区提示音】 =====================
    def generate_bsd_beep():
        sample_rate = 22050
        duration = 0.18
        freq = 800
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * freq * t)
        wave = (wave * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        return pygame.sndarray.make_sound(stereo_wave)

    bsd_warning_sound = generate_bsd_beep()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        # 【新增】将提示音绑定到HUD
        hud.ldw_sound = ldw_warning_sound
        hud.fcw_sound = fcw_warning_sound
        # ===================== 【新增】 =====================
        hud.bsd_sound = bsd_warning_sound
        world = World(client.get_world(), hud, args)
        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)

            # 视角切换按键
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
                    return
                if event.type == KEYDOWN:
                    if event.key == K_1:
                        world.camera_manager.set_camera_view(0)
                    elif event.key == K_2:
                        world.camera_manager.set_camera_view(1)
                    elif event.key == K_3:
                        world.camera_manager.set_camera_view(2)
                    elif event.key == K_4:
                        world.camera_manager.set_camera_view(3)
                    elif event.key == K_5:
                        world.camera_manager.set_camera_view(4)
                    elif event.key == K_c:  # 按 C 键 → 切换下一个天气（白天→黑夜→雨天→雾天）
                        world.next_weather()
                    elif event.key == K_v:  # 按 V 键 → 切换上一个天气
                        world.next_weather(reverse=True)
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            ego = world.player
            transform = ego.get_transform()
            vehicle_loc = transform.location
            vehicle_forward = transform.get_forward_vector()

            # PID定速巡航
            vel = ego.get_velocity()
            current_speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            dt = clock.get_time() / 1000.0
            throttle = pid.step(target_speed, current_speed, dt)
            throttle = max(throttle, 0.25)
            brake = 0.0

            # 车道居中
            waypoint = world.world.get_map().get_waypoint(vehicle_loc)
            dx = waypoint.transform.location.x - vehicle_loc.x
            dy = waypoint.transform.location.y - vehicle_loc.y
            cross = dx * vehicle_forward.y - dy * vehicle_forward.x
            steer = steer_pid.step(0, cross, dt)
            steer = max(-0.25, min(0.25, steer))

            # 前车障碍物避障
            safe_dist = 30.0
            danger_dist = 10.0
            min_dist = 9999
            for vehicle in world.world.get_actors().filter('vehicle.*'):
                if vehicle.id == ego.id:
                    continue
                diff = vehicle.get_transform().location - vehicle_loc
                dot = diff.x * vehicle_forward.x + diff.y * vehicle_forward.y
                if dot > 0:
                    dist = math.hypot(diff.x, diff.y)
                    if dist < min_dist:
                        min_dist = dist

            if min_dist < danger_dist:
                brake = 0.6
                throttle *= 0.2
            elif min_dist < safe_dist:
                brake = 0.2
                throttle *= 0.5
                # 【新增FCW】前向碰撞预警逻辑
                if min_dist < 12:  # 危险距离触发警告
                    hud.trigger_fcw_warning()
                    try:
                        hud.fcw_sound.play()
                    except:
                        pass
                # ===================== 【新增：BSD盲区监测核心代码】 =====================
                bsd_left = False
                bsd_right = False
                blind_radius = 8.0  # 盲区范围
                for vehicle in world.world.get_actors().filter('vehicle.*'):
                    if vehicle.id == ego.id:
                        continue
                    diff = vehicle.get_transform().location - vehicle_loc
                    dist = math.hypot(diff.x, diff.y)
                    if 2 < dist < blind_radius:
                        cross = diff.x * vehicle_forward.y - diff.y * vehicle_forward.x
                        if cross < -1.5:
                            bsd_right = True
                        elif cross > 1.5:
                            bsd_left = True

                if bsd_left or bsd_right:
                    hud.bsd_warning = True
                    hud.bsd_timer = hud.bsd_warning_duration
                    hud.bsd_side = "左侧" if bsd_left else "右侧"
                    try:
                        hud.bsd_sound.play()
                    except:
                        pass
            # ===================== 天气自适应控制（全自动） =====================
            weather_type = get_weather_type(world.world)
            auto_vehicle_lights(ego, weather_type)

            print("当前天气：", weather_type)

            # ===================== 交通标志识别控制（新增） =====================
            sign_state = traffic_sign_detect(world.camera_manager.rgb_image)
            print("交通标志识别：", sign_state)
            # ===================== 行人检测 + AEB自动刹车（新增） =====================
            has_pedestrian = pedestrian_detect(world.camera_manager.rgb_image)
            print("前方行人：", "检测到" if has_pedestrian else "无")

            # ===================== 控制优先级：行人AEB > 停车牌 > 红绿灯 > 限速 =====================
            # 1. 行人检测（最高优先级，必须第一）
            if has_pedestrian:
                brake = 1.0
                throttle = 0.0

            # 2. 停车标志
            if sign_state == 'stop':
                brake = 1.0
                throttle = 0.0
            elif sign_state == 'speed_30':
                target_speed = 30.0
            elif sign_state == 'speed_50':
                target_speed = 50.0
            elif sign_state == 'speed_60':
                target_speed = 60.0
            else:
                target_speed = DEFAULT_CRUISE_SPEED


            # ===================== 红绿灯控制逻辑 =====================
            light_state = traffic_light_detect(world.camera_manager.rgb_image)
            print("红绿灯识别：", light_state)

            # 交规逻辑：红灯、黄灯 强制停车；绿灯正常行驶
            if light_state == 'red' or light_state == 'yellow' and sign_state != 'stop':
                brake = 1.0
                throttle = 0.0

            # 下发车辆控制
            control = carla.VehicleControl()
            control.throttle = throttle
            control.brake = brake
            control.steer = steer
            control.hand_brake = False
            ego.apply_control(control)

    finally:
        if world is not None:
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- main ---------------------------------------------------------------------
# ==============================================================================
def main():
    argparser = argparse.ArgumentParser(description='CARLA Client')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug')
    argparser.add_argument('--host', default='127.0.0.1')
    argparser.add_argument('-p', '--port', default=2000, type=int)
    argparser.add_argument('--res', default='1280x720')
    argparser.add_argument('--filter', default='vehicle.*')
    argparser.add_argument('--gamma', default=2.2, type=int)
    argparser.add_argument('-s', '--seed', default=None, type=int)

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()