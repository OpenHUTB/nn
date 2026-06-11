import sys
import os
import logging
import argparse
import collections
import datetime
import math
import random
import re
import weakref
import json  # 新增：JSON处理
from pathlib import Path

# ========== 全局常量 ==========
MIN_WAYPOINTS_QUEUE = 21
CARLA_API_PATH = "WindowsNoEditor/PythonAPI/carla"
LOG_SAVE_DIR = "logs"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2000

# ========== 路径初始化 ==========
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
carla_api_full_path = script_dir / CARLA_API_PATH

if not carla_api_full_path.exists():
    raise FileNotFoundError(
        f"未找到CARLA API: {carla_api_full_path}\n"
        "请确认WindowsNoEditor文件夹与脚本在同一目录"
    )
sys.path.append(str(carla_api_full_path))

# ========== 第三方库导入 ==========
try:
    import pygame
    from pygame.locals import KMOD_CTRL, K_ESCAPE, K_q, K_r, K_h
except ImportError:
    raise RuntimeError("请安装pygame: pip install pygame")

try:
    import numpy as np
except ImportError:
    raise RuntimeError("请安装numpy: pip install numpy")

try:
    import carla
    from carla import ColorConverter as cc
    from agents.navigation.behavior_agent import BehaviorAgent
    from agents.navigation.basic_agent import BasicAgent
except ImportError as e:
    raise RuntimeError(f"CARLA API导入失败: {e}")

# ========== 日志配置 ==========
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ========== 新增：行驶日志记录器 ==========
class DrivingLogger:
    """自动记录行驶数据到CSV文件"""
    def __init__(self):
        self.log_dir = Path(LOG_SAVE_DIR)
        self.log_dir.mkdir(exist_ok=True)
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"driving_log_{time_str}.csv"
        self._init_header()
        self.last_collision_frame = -1

    def _init_header(self):
        headers = ["timestamp", "x", "y", "z", "speed_kmh", "is_collision", "target_reached"]
        with open(self.log_file, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(headers) + "\n")

    def record_frame(self, world, target_count):
        """记录单帧数据"""
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # 碰撞判断
        is_collision = 0
        collision_hist = world.collision_sensor.get_collision_history()
        current_frame = world.hud.frame
        if current_frame in collision_hist and current_frame != self.last_collision_frame:
            is_collision = 1
            self.last_collision_frame = current_frame

        # 写入CSV
        row = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            f"{transform.location.x:.2f}",
            f"{transform.location.y:.2f}",
            f"{transform.location.z:.2f}",
            f"{speed:.1f}",
            str(is_collision),
            str(target_count)
        ]
        with open(self.log_file, "a", encoding="utf-8", newline="") as f:
            f.write(",".join(row) + "\n")

    def get_file_path(self):
        return str(self.log_file)

# ========== 新增：轨迹记录器（JSON格式） ==========
class TrajectoryLogger:
    """记录车辆完整行驶轨迹为JSON文件，支持实时追加和最终格式化"""
    def __init__(self):
        self.log_dir = Path(LOG_SAVE_DIR)
        self.log_dir.mkdir(exist_ok=True)
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.traj_file = self.log_dir / f"trajectory_log_{time_str}.json"
        self.trajectory_data = {
            "metadata": {
                "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "total_frames": 0,
                "collision_frames": [],
                "target_reached_count": 0
            },
            "frames": []
        }
        # 初始化空JSON文件
        self._save_to_file()

    def record_frame(self, world, target_count):
        """记录单帧轨迹数据"""
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        current_frame = world.hud.frame

        # 碰撞判断
        is_collision = 0
        collision_hist = world.collision_sensor.get_collision_history()
        if current_frame in collision_hist:
            is_collision = 1
            if current_frame not in self.trajectory_data["metadata"]["collision_frames"]:
                self.trajectory_data["metadata"]["collision_frames"].append(current_frame)

        # 构造单帧数据
        frame_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "frame_id": current_frame,
            "location": {
                "x": round(transform.location.x, 2),
                "y": round(transform.location.y, 2),
                "z": round(transform.location.z, 2)
            },
            "rotation": {
                "pitch": round(transform.rotation.pitch, 2),
                "yaw": round(transform.rotation.yaw, 2),
                "roll": round(transform.rotation.roll, 2)
            },
            "speed_kmh": round(speed, 1),
            "is_collision": is_collision,
            "target_reached_count": target_count
        }

        # 追加数据并更新元信息
        self.trajectory_data["frames"].append(frame_data)
        self.trajectory_data["metadata"]["total_frames"] = len(self.trajectory_data["frames"])
        self.trajectory_data["metadata"]["target_reached_count"] = target_count

        # 实时保存（轻量化写入）
        self._save_to_file()

    def _save_to_file(self):
        """将轨迹数据写入JSON文件"""
        with open(self.traj_file, "w", encoding="utf-8") as f:
            json.dump(self.trajectory_data, f, ensure_ascii=False, indent=2)

    def get_file_path(self):
        """获取轨迹文件路径"""
        return str(self.traj_file)

    def finalize(self):
        """程序结束时最终化数据（补充结束时间）"""
        self.trajectory_data["metadata"]["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._save_to_file()
        logger.info(f"轨迹日志已最终化，共记录 {self.trajectory_data['metadata']['total_frames']} 帧")

# ========== 工具函数 ==========
def get_random_destination(current_loc, spawn_points):
    """获取非当前位置的随机目的地，兼容浮点误差"""
    if not spawn_points:
        raise ValueError("无可用生成点")
    valid_points = [
        p for p in spawn_points
        if math.hypot(p.location.x - current_loc.x, p.location.y - current_loc.y) > 2.0
    ]
    return random.choice(valid_points).location if valid_points else spawn_points[0].location

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

# ========== World类 ==========
class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
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

        try:
            self.map = self.world.get_map()
        except RuntimeError as e:
            logger.error(f"加载地图失败: {e}")
            sys.exit(1)

        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        cam_index = self.camera_manager.index if self.camera_manager else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager else 0

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

        # 重试生成车辆
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            logger.error("地图无可用生成点")
            sys.exit(1)

        for _ in range(5):
            spawn_point = random.choice(spawn_points)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            if self.player:
                break

        if not self.player:
            logger.error("车辆生成失败")
            sys.exit(1)

        # 初始化传感器
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

    # 重置车辆到最近的生成点
    def reset_vehicle(self):
        """重置车辆到最近的生成点"""
        if not self.player:
            return

        # 找最近的生成点
        spawn_points = self.map.get_spawn_points()
        current_loc = self.player.get_location()
        nearest = min(
            spawn_points,
            key=lambda p: math.hypot(p.location.x - current_loc.x, p.location.y - current_loc.y)
        )

        # 销毁旧车辆和传感器
        self.destroy()

        # 重生车辆
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        self.player = self.world.try_spawn_actor(blueprint, nearest)
        if not self.player:
            self.hud.error("车辆重置失败")
            return

        # 重新初始化所有传感器
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.set_sensor(0, notify=False)

        self.hud.notification("车辆已重置到最近生成点", seconds=3.0)
        logger.info(f"车辆重置到生成点 ({nearest.location.x:.1f}, {nearest.location.y:.1f})")

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        sensors = [
            self.camera_manager.sensor if self.camera_manager else None,
            self.collision_sensor.sensor if self.collision_sensor else None,
            self.lane_invasion_sensor.sensor if self.lane_invasion_sensor else None,
            self.gnss_sensor.sensor if self.gnss_sensor else None,
            self.player
        ]
        for sensor in sensors:
            if sensor is not None and sensor.is_alive:
                sensor.destroy()

# ========== 键盘控制 ==========
class KeyboardControl(object):
    def __init__(self, world):
        self.world = world
        self.hud = world.hud
        self.hud.notification("按R重置车辆 | 按H查看帮助 | 按ESC退出", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                # R键重置车辆
                if event.key == K_r:
                    self.world.reset_vehicle()
                # H键切换帮助
                if event.key == K_h:
                    self.hud.help.toggle()
        return False

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ========== HUD & 传感器类 ==========
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

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
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
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']

        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]

        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

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
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

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
            "CARLA 自动控制客户端",
            "",
            "快捷键：",
            "ESC / Ctrl+Q - 退出程序",
            "R - 重置车辆到最近生成点",
            "H - 显示/隐藏本帮助",
            "",
            "启动参数：",
            "-l / --loop - 到达目标后自动设置新目的地",
            "-b / --behavior - 智能体行为: cautious/normal/aggressive",
            "-a / --agent - 智能体类型: Behavior/Basic"
        ]
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 180))
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
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
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
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
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
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
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
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
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

# ========== 主游戏循环 ==========
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    driving_logger = None
    trajectory_logger = None  # 新增：轨迹日志器
    tot_target_reached = 0

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        # 初始化行驶日志和轨迹日志
        driving_logger = DrivingLogger()
        trajectory_logger = TrajectoryLogger()  # 新增：初始化轨迹日志
        logger.info(f"行驶日志已创建: {driving_logger.get_file_path()}")
        logger.info(f"轨迹日志已创建: {trajectory_logger.get_file_path()}")

        # 初始化智能体（只保留你用的BehaviorAgent和BasicAgent）
        if args.agent == "Basic":
            agent = BasicAgent(world.player)
            spawn_point = world.map.get_spawn_points()[0]
            agent.set_destination((spawn_point.location.x,
                                   spawn_point.location.y,
                                   spawn_point.location.z))
        else:
            agent = BehaviorAgent(world.player, behavior=args.behavior)
            spawn_points = world.map.get_spawn_points()
            random.shuffle(spawn_points)
            current_location = world.player.get_location()
            destination = get_random_destination(current_location, spawn_points)
            agent.set_destination(destination, start_location=current_location)

        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events():
                return

            if not world.world.wait_for_tick(10.0):
                continue

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            # 记录行驶日志和轨迹日志
            driving_logger.record_frame(world, tot_target_reached)
            trajectory_logger.record_frame(world, tot_target_reached)  # 新增：记录轨迹

            if args.agent == "Basic":
                control = agent.run_step()
                control.manual_gear_shift = False
                world.player.apply_control(control)
            else:
                # 判断是否需要设置新目的地
                if len(agent.get_local_planner()._waypoints_queue) < MIN_WAYPOINTS_QUEUE:
                    if args.loop:
                        spawn_points = world.map.get_spawn_points()
                        random.shuffle(spawn_points)
                        current_loc = world.player.get_location()
                        new_dest = get_random_destination(current_loc, spawn_points)
                        agent.set_destination(new_dest, start_location=current_loc)
                        agent.run_step()

                        tot_target_reached += 1
                        world.hud.notification(f"目标已到达 {tot_target_reached} 次", seconds=4.0)
                        logger.info(f"到达第 {tot_target_reached} 个目标")
                    else:
                        print("Target reached, mission accomplished...")
                        break

                speed_limit = world.player.get_speed_limit()
                agent.get_local_planner().set_speed(speed_limit)
                control = agent.run_step()
                world.player.apply_control(control)

    finally:
        if world is not None:
            world.destroy()
        pygame.quit()
        # 保存日志
        if driving_logger:
            logger.info(f"行驶日志已保存至: {driving_logger.get_file_path()}")
        if trajectory_logger:  # 新增：最终化并保存轨迹日志
            trajectory_logger.finalize()
            logger.info(f"轨迹日志已保存至: {trajectory_logger.get_file_path()}")

# ========== 主函数 ==========
def main():
    argparser = argparse.ArgumentParser(description='CARLA Automatic Control Client')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='打印调试信息')
    argparser.add_argument('--host', metavar='H', default=DEFAULT_HOST, help='服务器IP')
    argparser.add_argument('-p', '--port', metavar='P', default=DEFAULT_PORT, type=int, help='端口')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default="1280x720", help='窗口分辨率')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='车辆过滤器')
    argparser.add_argument('--gamma', default=2.2, type=float, help='相机伽马值')
    argparser.add_argument('-l', '--loop', action='store_true', dest='loop', help='循环生成新目的地')
    argparser.add_argument('-b', '--behavior', type=str, choices=["cautious", "normal", "aggressive"],
                           default='normal', help='智能体行为模式')
    argparser.add_argument("-a", "--agent", type=str, choices=["Behavior", "Basic"],
                           default="Behavior", help="智能体类型")
    argparser.add_argument('-s', '--seed', default=None, type=int, help='随机种子')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger().setLevel(log_level)

    logging.info('listening to server %s:%s', args.host, args.port)
    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()


    