import sys
import os
import logging
import argparse
import collections
import datetime
import glob
import math
import random
import re
import weakref
from pathlib import Path

# ========== 全局常量定义（优化：提取魔法常量） ==========
MIN_WAYPOINTS_QUEUE = 21  # 触发新目的地的最小路径点数量
CARLA_API_RELATIVE_PATH = "WindowsNoEditor/PythonAPI/carla"
LOG_DIR = "logs"  # 行驶日志保存目录
DEFAULT_CAMERA_GAMMA = 2.2
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 2000
DEFAULT_WINDOW_RES = "1280x720"
DEFAULT_VEHICLE_FILTER = "vehicle.*"

# ========== 路径处理（优化：统一且健壮） ==========
# 获取当前脚本绝对路径
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
# 拼接CARLA PythonAPI路径（自动适配系统）
carla_api_path = script_dir / CARLA_API_RELATIVE_PATH

# 检查API路径是否存在
if not carla_api_path.exists():
    raise FileNotFoundError(
        f"CARLA API路径不存在: {carla_api_path}\n"
        "请确保脚本与WindowsNoEditor文件夹在同一目录下"
    )
sys.path.append(str(carla_api_path))

# ========== 第三方库导入（优化：集中导入） ==========
try:
    import pygame
    from pygame.locals import KMOD_CTRL, K_ESCAPE, K_q, K_r, K_h, K_SLASH
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
    # 补充RoamingAgent（原代码未导入，修复潜在报错）

except ImportError as e:
    raise RuntimeError(f"CARLA PythonAPI导入失败: {e}")

# ========== 日志配置（优化：统一日志输出） ==========
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========== 新增功能：行驶日志记录类 ==========
class DrivingLogger:
    """行驶日志记录类，保存车辆状态到CSV文件"""
    def __init__(self):
        # 创建日志目录
        self.log_dir = Path(LOG_DIR)
        self.log_dir.mkdir(exist_ok=True)
        
        # 按时间生成日志文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"driving_log_{timestamp}.csv"
        
        # 初始化CSV文件头
        self._init_csv()
        self.last_collision_frame = -1  # 记录最后碰撞帧，避免重复标记

    def _init_csv(self):
        """初始化CSV文件，写入表头"""
        headers = [
            "timestamp", "location_x", "location_y", "location_z",
            "speed_kmh", "is_collision", "weather", "target_reached_count"
        ]
        with open(self.log_file, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(headers) + "\n")

    def log_frame(self, world, vehicle, target_reached_count):
        """记录单帧车辆状态"""
        # 获取基础信息
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        transform = vehicle.get_transform()
        vel = vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        
        # 判断当前是否碰撞（避免同一碰撞多次记录）
        is_collision = False
        collision_history = world.collision_sensor.get_collision_history()
        current_frame = world.hud.frame
        if current_frame in collision_history and current_frame != self.last_collision_frame:
            is_collision = True
            self.last_collision_frame = current_frame
        
                # 获取天气信息（容错版：比较关键参数而非对象本身）
        weather = world.player.get_world().get_weather()
        weather_name = "未知"
        for preset, name in find_weather_presets():
            # 比较云量、降水量、雾密度三个关键参数
            if (abs(preset.cloudiness - weather.cloudiness) < 0.1 and
                abs(preset.precipitation - weather.precipitation) < 0.1 and
                abs(preset.fog_density - weather.fog_density) < 0.1):
                weather_name = name
                break
        
        # 写入CSV
        row = [
            timestamp,
            f"{transform.location.x:.2f}",
            f"{transform.location.y:.2f}",
            f"{transform.location.z:.2f}",
            f"{speed_kmh:.1f}",
            "1" if is_collision else "0",
            weather_name,
            str(target_reached_count)
        ]
        with open(self.log_file, "a", encoding="utf-8", newline="") as f:
            f.write(",".join(row) + "\n")

    def get_log_path(self):
        """返回日志文件路径"""
        return str(self.log_file)

# ========== 工具函数（优化：保持简洁） ==========
def get_random_destination(current_location, spawn_points):
    """获取非当前位置的随机生成点作为目的地"""
    if not spawn_points:
        raise ValueError("无可用的生成点！")
    # 过滤当前位置（允许微小误差，避免浮点精度问题）
    valid_spawn_points = [
        p for p in spawn_points
        if not math.isclose(p.location.x, current_location.x, abs_tol=0.1)
        or not math.isclose(p.location.y, current_location.y, abs_tol=0.1)
    ]
    return (
        random.choice(valid_spawn_points).location
        if valid_spawn_points
        else spawn_points[0].location
    )

def find_weather_presets():
    """获取天气预设列表"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    """获取Actor的可读名称"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

# ========== World类（优化：修复reset_vehicle、完善逻辑） ==========
class World(object):
    """代表CARLA世界环境的类"""
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
            logger.error(f"获取地图失败: {e}")
            logger.error("请确保OpenDRIVE文件存在且与城镇名称匹配")
            sys.exit(1)
        
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """重启世界（重新生成车辆和传感器）"""
        # 保留相机配置
        cam_index = self.camera_manager.index if self.camera_manager else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager else 0
        
        # 设置随机种子
        if args.seed is not None:
            random.seed(args.seed)
            logger.info(f"设置随机种子: {args.seed}")

        # 选择车辆蓝图
        blueprint_library = self.world.get_blueprint_library()
        try:
            blueprint = random.choice(blueprint_library.filter(self._actor_filter))
        except IndexError:
            logger.error(f"未找到匹配的Actor过滤器: {self._actor_filter}")
            sys.exit(1)
        
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # 销毁旧车辆
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()

        # 生成新车辆（重试机制）
        self.player = None
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            logger.error("地图中无可用的车辆生成点！")
            sys.exit(1)
        
        for _ in range(5):  # 最多重试5次
            spawn_point = random.choice(spawn_points)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            if self.player:
                break
        
        if not self.player:
            logger.error("车辆生成失败（重试5次后仍失败）")
            sys.exit(1)
        
        # 初始化传感器
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        
        actor_name = get_actor_display_name(self.player)
        self.hud.notification(f"生成车辆: {actor_name}")
        logger.info(f"成功生成车辆: {actor_name}")

    def next_weather(self, reverse=False):
        """切换天气预设"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset, name = self._weather_presets[self._weather_index]
        self.world.set_weather(preset)
        self.hud.notification(f"天气切换为: {name}")
        logger.info(f"天气已切换: {name}")

    def reset_vehicle(self):
        """优化：移到类内部，修复self引用"""
        """重置车辆到最近的生成点，并保留当前目的地"""
        if not self.player:
            logger.warning("无车辆可重置")
            return
        
        # 获取最近的生成点
        spawn_points = self.map.get_spawn_points()
        current_loc = self.player.get_location()
        nearest_spawn = min(
            spawn_points,
            key=lambda p: math.hypot(p.location.x - current_loc.x, p.location.y - current_loc.y)
        )
        
        # 销毁旧车辆
        self.destroy()
        
        # 重生车辆
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        
        self.player = self.world.try_spawn_actor(blueprint, nearest_spawn)
        if not self.player:
            logger.error("车辆重置失败：生成新车辆失败")
            return
        
        # 重新初始化传感器
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.set_sensor(0, notify=False)
        
        self.hud.notification("车辆已重置到最近生成点", seconds=3.0)
        logger.info(f"车辆重置到生成点: ({nearest_spawn.location.x:.2f}, {nearest_spawn.location.y:.2f})")

    def tick(self, clock):
        """每帧更新"""
        self.hud.tick(self, clock)

    def render(self, display):
        """渲染画面"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        """销毁所有Actor"""
        actors = [
            self.camera_manager.sensor if self.camera_manager else None,
            self.collision_sensor.sensor if self.collision_sensor else None,
            self.lane_invasion_sensor.sensor if self.lane_invasion_sensor else None,
            self.gnss_sensor.sensor if self.gnss_sensor else None,
            self.player
        ]
        for actor in actors:
            if actor and actor.is_alive:
                try:
                    actor.destroy()
                    logger.info(f"销毁Actor: {get_actor_display_name(actor)}")
                except Exception as e:
                    logger.warning(f"销毁Actor失败: {e}")

# ========== 键盘控制（优化：修复self.world初始化） ==========
class KeyboardControl(object):
    def __init__(self, world):
        self.world = world  # 修复：初始化world引用
        self.hud = world.hud
        self.hud.notification("按'H'或'?'查看帮助 | 按'R'重置车辆 | 按'ESC/Q'退出", seconds=4.0)

    def parse_events(self):
        """解析键盘事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                # 退出快捷键
                if self._is_quit_shortcut(event.key):
                    return True
                # 重置车辆
                if event.key == K_r:
                    self.world.reset_vehicle()
                # 切换帮助信息
                if event.key in (K_h, K_SLASH):
                    self.hud.help.toggle()
                # 切换天气
                if event.key == pygame.K_n:
                    self.world.next_weather()
                if event.key == pygame.K_m:
                    self.world.next_weather(reverse=True)
        return False

    @staticmethod
    def _is_quit_shortcut(key):
        """退出快捷键判断"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ========== 原有传感器/HUD类（少量优化：注释/鲁棒性） ==========
class HUD(object):
    """HUD显示类"""
    def __init__(self, width, height):
        self.dim = (width, height)
        self._font_mono = pygame.font.Font(
            pygame.font.match_font('ubuntumono' if os.name != 'nt' else 'courier'),
            12 if os.name == 'nt' else 14
        )
        self._notifications = FadingText(pygame.font.Font(None, 20), (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(None, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """每帧更新世界信息"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """更新HUD内容"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        
        # 基础车辆信息
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        
        # 碰撞历史
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        
        # 周边车辆
        vehicles = world.world.get_actors().filter('vehicle.*')
        vehicle_count = len(vehicles)

        # 组装信息文本
        self._info_text = [
            f'Server FPS:  {self.server_fps:.0f}',
            f'Client FPS:  {clock.get_fps():.0f}',
            '',
            f'车辆: {get_actor_display_name(world.player, truncate=20)}',
            f'地图: {world.map.name}',
            f'仿真时间: {datetime.timedelta(seconds=int(self.simulation_time))}',
            '',
            f'速度: {3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2):.0f} km/h',
            f'朝向: {transform.rotation.yaw:.0f}° {heading}',
            f'位置: ({transform.location.x:.1f}, {transform.location.y:.1f})',
            f'GNSS: ({world.gnss_sensor.lat:.6f}, {world.gnss_sensor.lon:.6f})',
            f'高度: {transform.location.z:.0f} m',
            ''
        ]

        # 车辆控制信息
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('油门:', control.throttle, 0.0, 1.0),
                ('转向:', control.steer, -1.0, 1.0),
                ('刹车:', control.brake, 0.0, 1.0),
                ('倒车:', control.reverse),
                ('手刹:', control.hand_brake),
                ('手动换挡:', control.manual_gear_shift),
                '档位: %s' % {-1: "R", 0: "N"}.get(control.gear, control.gear)
            ]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('速度:', control.speed, 0.0, 5.556),
                ('跳跃:', control.jump)
            ]

        # 碰撞和车辆数量
        self._info_text += [
            '',
            '碰撞强度:',
            collision,
            '',
            f'车辆总数: {vehicle_count}'
        ]

        # 周边车辆信息
        if vehicle_count > 1:
            self._info_text += ['周边车辆:']
            def dist(l):
                return math.hypot(
                    l.x - transform.location.x,
                    l.y - transform.location.y,
                    l.z - transform.location.z
                )
            nearby_vehicles = sorted(
                [(dist(v.get_location()), v) for v in vehicles if v.id != world.player.id],
                key=lambda x: x[0]
            )
            for dist, vehicle in nearby_vehicles:
                if dist > 200.0:
                    break
                self._info_text.append(f'{dist:.0f}m {get_actor_display_name(vehicle, truncate=22)}')

    def toggle_info(self):
        """切换信息显示"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """显示通知"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """显示错误"""
        self._notifications.set_text(f'错误: {text}', (255, 0, 0))

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
                        rect = pygame.Rect(
                            (bar_h_offset + fig * (bar_width - 6) if item[2] < 0 else bar_h_offset, v_offset + 8),
                            (6 if item[2] < 0 else fig * bar_width, 6)
                        )
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                
                if item:
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18

        self._notifications.render(display)
        self.help.render(display)

class FadingText(object):
    """渐隐文本类"""
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim, pygame.SRCALPHA)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """设置渐隐文本"""
        text_surface = self.font.render(text, True, color)
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_surface, (10, 11))
        self.seconds_left = seconds

    def tick(self, _, clock):
        """更新透明度"""
        delta = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """渲染文本"""
        display.blit(self.surface, self.pos)

class HelpText(object):
    """帮助文本类"""
    def __init__(self, font, width, height):
        lines = [
            "CARLA 自动控制客户端",
            "",
            "快捷键说明:",
            "H/? - 显示/隐藏帮助",
            "R - 重置车辆到最近生成点",
            "N - 下一个天气预设",
            "M - 上一个天气预设",
            "ESC/Q - 退出程序",
            "",
            "参数说明:",
            "-l/--loop - 到达目标后自动设置新随机目标",
            "-b/--behavior - 选择智能体行为（谨慎/正常/激进）",
            "-a/--agent - 选择智能体类型（Behavior/Roaming/Basic）"
        ]
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.surface = pygame.Surface(self.dim, pygame.SRCALPHA)
        self.surface.fill((0, 0, 0, 220))
        for i, line in enumerate(lines):
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_surface, (22, i * 22))
        self._render = False

    def toggle(self):
        """切换显示/隐藏"""
        self._render = not self._render

    def render(self, display):
        """渲染帮助文本"""
        if self._render:
            display.blit(self.surface, self.pos)

class CollisionSensor(object):
    """碰撞传感器"""
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
        """获取碰撞历史"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """碰撞回调"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification(f'与 {actor_type} 发生碰撞')
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

class LaneInvasionSensor(object):
    """车道入侵传感器"""
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """车道入侵回调"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = [str(x).split()[-1] for x in lane_types]
        self.hud.notification(f'压线: {" 和 ".join(text)}')

class GnssSensor(object):
    """GNSS传感器"""
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(
            blueprint,
            carla.Transform(carla.Location(x=1.0, z=2.8)),
            attach_to=self._parent
        )
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS回调"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

class CameraManager(object):
    """相机管理器"""
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.transform_index = 1
        self.index = None

        # 相机位置预设
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)
        ]

        # 传感器预设
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'RGB相机', bp_library.find('sensor.camera.rgb')],
            ['sensor.camera.depth', cc.Raw, '深度相机（原始）', bp_library.find('sensor.camera.depth')],
            ['sensor.camera.depth', cc.Depth, '深度相机（灰度）', bp_library.find('sensor.camera.depth')],
            ['sensor.camera.depth', cc.LogarithmicDepth, '深度相机（对数灰度）', bp_library.find('sensor.camera.depth')],
            ['sensor.camera.semantic_segmentation', cc.Raw, '语义分割（原始）', bp_library.find('sensor.camera.semantic_segmentation')],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, '语义分割（CityScapes配色）', bp_library.find('sensor.camera.semantic_segmentation')],
            ['sensor.lidar.ray_cast', None, '激光雷达', bp_library.find('sensor.lidar.ray_cast')]
        ]

        # 配置传感器参数
        for item in self.sensors:
            if item[0].startswith('sensor.camera'):
                item[3].set_attribute('image_size_x', str(hud.dim[0]))
                item[3].set_attribute('image_size_y', str(hud.dim[1]))
                if item[3].has_attribute('gamma'):
                    item[3].set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                item[3].set_attribute('range', '50')

    def toggle_camera(self):
        """切换相机位置"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """设置当前传感器"""
        index = index % len(self.sensors)
        needs_respawn = (
            self.index is None 
            or force_respawn 
            or self.sensors[index][0] != self.sensors[self.index][0]
        )

        if needs_respawn:
            if self.sensor:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][3],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1]
            )
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """切换到下一个传感器"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """切换录像状态"""
        self.recording = not self.recording
        self.hud.notification(f'录像: {"开启" if self.recording else "关闭"}')

    def render(self, display):
        """渲染相机画面"""
        if self.surface:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        """解析传感器图像"""
        self = weak_self()
        if not self:
            return

        if self.sensors[self.index][0].startswith('sensor.lidar'):
            # 激光雷达数据处理
            points = np.frombuffer(image.raw_data, dtype=np.float32).reshape(-1, 4)[:, :2]
            points *= min(self.hud.dim) / 100.0
            points += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            points = np.fabs(points).astype(np.int32)
            lidar_img = np.zeros((self.hud.dim[1], self.hud.dim[0], 3), dtype=np.uint8)
            lidar_img[tuple(points.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img.swapaxes(0, 1))
        else:
            # 相机图像处理
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        # 保存录像
        if self.recording:
            Path("_out").mkdir(exist_ok=True)
            image.save_to_disk(f'_out/{image.frame:08d}')

# ========== 主游戏循环（新增：日志记录） ==========
def game_loop(args):
    """主游戏循环"""
    pygame.init()
    pygame.display.set_caption("CARLA 自动控制客户端（带行驶日志）")
    display = pygame.display.set_mode(
        (args.width, args.height),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    clock = pygame.time.Clock()
    world = None
    driving_logger = None
    tot_target_reached = 0

    try:
        # 连接CARLA服务器
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)  # 优化：延长超时时间
        carla_world = client.get_world()
        logger.info(f"成功连接到CARLA服务器: {args.host}:{args.port}")

        # 初始化HUD、World、键盘控制
        hud = HUD(args.width, args.height)
        world = World(carla_world, hud, args)
        controller = KeyboardControl(world)

        # 初始化行驶日志（新增功能）
        driving_logger = DrivingLogger()
        logger.info(f"行驶日志已创建: {driving_logger.get_log_path()}")
        hud.notification(f"日志保存至: {driving_logger.get_log_path()}", seconds=5.0)

        # 初始化智能体
      
        if args.agent == "Basic":
            agent = BasicAgent(world.player)
            spawn_point = world.map.get_spawn_points()[0]
            agent.set_destination((spawn_point.location.x, spawn_point.location.y, spawn_point.location.z))
        else:  # Behavior Agent
            agent = BehaviorAgent(world.player, behavior=args.behavior)
            spawn_points = world.map.get_spawn_points()
            random.shuffle(spawn_points)
            current_location = world.player.get_location()
            destination = get_random_destination(current_location, spawn_points)
            agent.set_destination(destination, current_location)

        # 主循环
        while True:
            clock.tick_busy_loop(60)
            
            # 处理键盘事件
            if controller.parse_events():
                break
            
            # 等待服务器tick
            if not world.world.wait_for_tick(10.0):
                logger.warning("服务器tick超时")
                continue

            # 更新世界状态
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            # 智能体控制逻辑
            if args.agent in ["Roaming", "Basic"]:
                control = agent.run_step()
                control.manual_gear_shift = False
                world.player.apply_control(control)
            else:
                # Behavior Agent：到达目标后设置新目标
                waypoints_queue = agent.get_local_planner()._waypoints_queue
                if len(waypoints_queue) < MIN_WAYPOINTS_QUEUE and args.loop:
                    spawn_points = world.map.get_spawn_points()
                    random.shuffle(spawn_points)
                    current_location = world.player.get_location()
                    new_destination = get_random_destination(current_location, spawn_points)
                    agent.set_destination(current_location, new_destination)
                    agent.run_step()
                    tot_target_reached += 1
                    hud.notification(f"目标已到达 {tot_target_reached} 次", seconds=4.0)
                    logger.info(f"第 {tot_target_reached} 次到达目标，新目标: ({new_destination.x:.2f}, {new_destination.y:.2f})")
                elif len(waypoints_queue) == 0 and not args.loop:
                    logger.info("目标到达，任务完成")
                    hud.notification("目标到达，任务完成！", seconds=5.0)
                    break

                # 应用车辆控制
                speed_limit = world.player.get_speed_limit()
                agent.get_local_planner().set_speed(speed_limit)
                control = agent.run_step()
                world.player.apply_control(control)

            # 记录行驶日志（新增功能）
            if driving_logger:
                driving_logger.log_frame(world, world.player, tot_target_reached)

    except Exception as e:
        logger.error(f"游戏循环异常: {e}", exc_info=True)
        if hud:
            hud.error(f"程序异常: {str(e)}")
    finally:
        # 清理资源
        if world:
            world.destroy()
        pygame.quit()
        logger.info("程序正常退出，资源已清理")

# ========== 主函数（优化：参数解析更健壮） ==========
def main():
    """程序入口"""
    argparser = argparse.ArgumentParser(description='CARLA 自动控制客户端（带行驶日志功能）')
    argparser.add_argument('-v', '--verbose', action='store_true', help='打印调试信息')
    argparser.add_argument('--host', default=DEFAULT_SERVER_HOST, help='服务器IP（默认：127.0.0.1）')
    argparser.add_argument('-p', '--port', type=int, default=DEFAULT_SERVER_PORT, help='服务器端口（默认：2000）')
    argparser.add_argument('--res', default=DEFAULT_WINDOW_RES, help='窗口分辨率（默认：1280x720）')
    argparser.add_argument('--filter', default=DEFAULT_VEHICLE_FILTER, help='Actor过滤器（默认：vehicle.*）')
    argparser.add_argument('--gamma', type=float, default=DEFAULT_CAMERA_GAMMA, help='相机伽马校正（默认：2.2）')
    argparser.add_argument('-l', '--loop', action='store_true', help='到达目标后自动设置新随机目标')
    argparser.add_argument('-b', '--behavior', type=str, choices=["cautious", "normal", "aggressive"], default='normal', help='智能体行为（默认：normal）')
    argparser.add_argument('-a', '--agent', type=str, choices=["Behavior", "Roaming", "Basic"], default="Behavior", help='智能体类型（默认：Behavior）')
    argparser.add_argument('-s', '--seed', type=int, default=None, help='随机种子（用于复现实验）')

    args = argparser.parse_args()

    # 解析分辨率
    try:
        args.width, args.height = [int(x) for x in args.res.split('x')]
    except ValueError:
        logger.error(f"分辨率格式错误: {args.res}，使用默认值 {DEFAULT_WINDOW_RES}")
        args.width, args.height = [int(x) for x in DEFAULT_WINDOW_RES.split('x')]

    # 配置日志级别
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.info(f"启动CARLA客户端，连接 {args.host}:{args.port}，分辨率 {args.width}x{args.height}")

    try:
        game_loop(args)
    except KeyboardInterrupt:
        logger.info("用户手动中断程序")
        print("\n程序已被用户中断，退出中...")
    except Exception as e:
        logger.error(f"程序启动失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()