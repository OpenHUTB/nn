"""CARLA 碰撞检测与巡航测试系统

支持键盘控制车辆巡航，视觉感知模块检测前方障碍物，自动触发
变道绕行（AES）或紧急制动（AEB）。
"""

import carla
import time
import pygame
import math
import keyboard
import numpy as np
from vision_module import VisionSystem
from planner import LanePlanner

# ── 常量配置 ──────────────────────────────────────────────
CARLA_HOST = "localhost"
CARLA_PORT = 2000
TIMEOUT = 10.0
FIXED_DT = 0.05

# 速度控制
SPEED_STEP = 5.0            # 每次按键调整速度的步长 (km/h)
PID_KP = 0.15               # 比例系数
PID_KI = 0.02               # 积分系数
PID_MAX_INTEGRAL = 40.0     # 积分饱和上限
MAX_THROTTLE = 0.75
MAX_BRAKE_COAST = 0.2       # 0 目标速度时轻刹
MAX_BRAKE_STOP = 1.0        # 完全停车时重刹

# AEB 参数
DECELERATION = 6.0          # 制动减速度 (m/s²)
REACTION_DIST = 3.0         # 反应距离 (m)
LANE_CHANGE_SPEED = 15.0    # 变道时的临时巡航速度 (km/h)
AEB_MIN_SPEED = 2.0         # AEB触发的最低速度 (km/h)

# 显示
SCREEN_SIZE = (400, 240)
FONT_NAME = "simhei"
FONT_SIZE = 24

# 靶标前方距离
TARGET_DISTANCE = 60.0
FALLBACK_DISTANCE = 40.0


# ── 辅助类 ────────────────────────────────────────────────

class SpeedController:
    """PI速度控制器，支持前进/后退"""

    def __init__(self):
        self.target_kmh = 0.0
        self.error_sum = 0.0
        self.is_reverse = False

    def set_target(self, kmh: float):
        self.target_kmh = max(0.0, kmh)

    def adjust_target(self, delta: float):
        self.target_kmh = max(0.0, self.target_kmh + delta)

    def toggle_reverse(self):
        self.is_reverse = not self.is_reverse

    def reset(self):
        self.target_kmh = 0.0
        self.error_sum = 0.0

    def compute(self, current_speed_kmh: float) -> tuple:
        """返回 (throttle, brake)"""
        error = self.target_kmh - current_speed_kmh

        if self.target_kmh > 0:
            self.error_sum = max(min(self.error_sum + error, PID_MAX_INTEGRAL), -PID_MAX_INTEGRAL)
        else:
            self.error_sum = 0.0

        if self.target_kmh == 0.0:
            return 0.0, MAX_BRAKE_COAST if current_speed_kmh > 0.5 else MAX_BRAKE_STOP
        elif error > 0:
            throttle = min(max((error * PID_KP) + (self.error_sum * PID_KI), 0.0), MAX_THROTTLE)
            return throttle, 0.0
        else:
            brake = min(max((-error * PID_KP) - (self.error_sum * PID_KI), 0.0), 0.5)
            return 0.0, brake


class DisplayPanel:
    """Pygame 控制面板"""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("CARLA 控制面板")
        pygame.font.init()
        self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

    def render(self, target_kmh: float, current_kmh: float,
               throttle: float, brake: float, reverse: bool, aeb_active: bool):
        self.screen.fill((30, 30, 30))

        aeb_text = "AEB 制动中!" if aeb_active else "巡航系统已启动 (W/S 调速)"
        aeb_color = (255, 50, 50) if aeb_active else (255, 200, 0)

        lines = [
            (aeb_text, aeb_color),
            (f"设定巡航: {target_kmh:.1f} km/h", (255, 150, 200)),
            (f"当前车速: {current_kmh:.1f} km/h", (0, 255, 255)),
            (f"油门:[{'开' if throttle > 0.01 else '关'}]  刹车:[{'开' if brake > 0.01 else '关'}]", (150, 150, 150)),
            (f"档位: {'[R] 倒车' if reverse else '[D] 前进'}", (255, 255, 255)),
        ]

        for i, (text, color) in enumerate(lines):
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, (20, 20 + i * 40))

        pygame.display.flip()

    def quit(self):
        pygame.quit()


# ── 场景初始化 ────────────────────────────────────────────

def get_user_choice():
    """获取用户选择的靶标类型"""
    print("=" * 40)
    print("CARLA 碰撞与巡航测试系统")
    print("   [1] 测试车辆")
    print("   [2] 测试行人")
    print("=" * 40)
    choice = input("请输入选项 (默认 1): ").strip()
    return choice if choice in ('1', '2') else '1'


def spawn_ego_vehicle(world, bp_lib):
    """在主车生成点生成车辆，优先选择前方非路口的直道"""
    spawn_points = world.get_map().get_spawn_points()
    target_wp = None

    for sp in spawn_points:
        wp = world.get_map().get_waypoint(sp.location)
        if not wp.is_junction:
            fwd = wp.next(TARGET_DISTANCE)
            if fwd and not fwd[0].is_junction:
                ego_vehicle = world.try_spawn_actor(bp_lib.find('vehicle.lincoln.mkz_2017'), sp)
                return ego_vehicle, fwd[0]

    # 回退：使用第一个生成点
    sp = spawn_points[0]
    wp = world.get_map().get_waypoint(sp.location)
    ego_vehicle = world.try_spawn_actor(bp_lib.find('vehicle.lincoln.mkz_2017'), sp)
    return ego_vehicle, wp.next(FALLBACK_DISTANCE)[0]


def spawn_target(world, bp_lib, waypoint, target_type: str):
    """在指定航点生成靶标"""
    transform = waypoint.transform
    transform.location.z += 0.5

    if target_type == '2':
        target_bp = bp_lib.filter('walker.pedestrian.*')[0]
    else:
        target_bp = bp_lib.find('vehicle.tesla.model3')

    actor = world.try_spawn_actor(target_bp, transform)
    if actor:
        type_name = "行人" if target_type == '2' else "车辆"
        print(f"靶标 [{type_name}] 已生成在 ({transform.location.x:.1f}, {transform.location.y:.1f})")
    else:
        print("靶标生成失败，空间可能受限")
    return actor


def setup_collision_sensor(world, bp_lib, vehicle):
    """挂载碰撞传感器，返回 (sensor, flag_list)"""
    bp = bp_lib.find('sensor.other.collision')
    sensor = world.try_spawn_actor(bp, carla.Transform(), attach_to=vehicle)
    flag = [False]

    def on_collision(event):
        if flag[0]:
            return
        flag[0] = True
        loc = event.actor.get_transform().location
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        hit_type = "行人" if "walker" in event.other_actor.type_id else "车辆"
        print(f"\033[91m\n[COLLISION] {hit_type} ({event.other_actor.type_id}), "
              f"冲量={intensity:.0f}, 坐标=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})\033[0m")

    if sensor:
        sensor.listen(on_collision)
        print("碰撞传感器已挂载")
    return sensor, flag


# ── 主循环逻辑 ────────────────────────────────────────────

def handle_keyboard(speed_ctrl: SpeedController):
    """处理键盘输入，返回是否退出"""
    prev_w = getattr(handle_keyboard, 'prev_w', False)
    prev_s = getattr(handle_keyboard, 'prev_s', False)
    prev_q = getattr(handle_keyboard, 'prev_q', False)

    curr_q = keyboard.is_pressed('q')
    if curr_q and not prev_q:
        speed_ctrl.toggle_reverse()

    curr_w = keyboard.is_pressed('w')
    if curr_w and not prev_w:
        speed_ctrl.adjust_target(SPEED_STEP)

    curr_s = keyboard.is_pressed('s')
    if curr_s and not prev_s:
        speed_ctrl.adjust_target(-SPEED_STEP)

    handle_keyboard.prev_w = curr_w
    handle_keyboard.prev_s = curr_s
    handle_keyboard.prev_q = curr_q

    return keyboard.is_pressed('esc') or pygame.event.peek(pygame.QUIT)


def handle_vision_aeb(ego_vehicle, vision_system, lane_planner, speed_ctrl: SpeedController,
                      control: carla.VehicleControl):
    """视觉感知 + 变道/AEB 决策"""
    if not vision_system:
        return False

    _, min_dist = vision_system.process_and_render()
    safe_dist = min_dist if min_dist != float('inf') else 100.0

    v = ego_vehicle.get_velocity()
    speed_m_s = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
    speed_kmh = speed_m_s * 3.6

    # 横向控制
    planner_steer = lane_planner.get_lateral_control(safe_dist, speed_kmh)
    if not speed_ctrl.is_reverse and planner_steer is not None:
        control.steer = planner_steer

    # 纵向 AEB 决策
    if min_dist == float('inf'):
        return False

    braking_dist = (speed_m_s ** 2) / (2 * DECELERATION) + REACTION_DIST

    if lane_planner.is_changing_lane and not speed_ctrl.is_reverse:
        # 变道时降低速度而非急刹
        speed_ctrl.target_kmh = LANE_CHANGE_SPEED
        control.brake = 0.0
        return False

    if min_dist <= braking_dist and speed_kmh > AEB_MIN_SPEED and not speed_ctrl.is_reverse:
        print("\033[91m路径受阻，触发 AEB 紧急刹车!\033[0m")
        speed_ctrl.reset()
        control.throttle = 0.0
        control.brake = MAX_BRAKE_STOP
        control.hand_brake = True
        return True

    return False


def update_spectator(world, ego_vehicle):
    """更新观察者视角到车辆后方"""
    transform = ego_vehicle.get_transform()
    loc = transform.location + carla.Location(z=5) - transform.get_forward_vector() * 10
    rot = carla.Rotation(pitch=-20, yaw=transform.rotation.yaw)
    world.get_spectator().set_transform(carla.Transform(loc, rot))


def cleanup(world, actor_list: list, display: DisplayPanel):
    """清理所有 Actor 和设置"""
    keyboard.unhook_all()
    for actor in actor_list:
        if actor is not None:
            actor.destroy()
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    display.quit()
    print("环境已清理")


# ── 主函数 ────────────────────────────────────────────────

def main():
    choice = get_user_choice()
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(TIMEOUT)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    display = DisplayPanel()
    bp_lib = world.get_blueprint_library()

    ego_vehicle, target_wp = spawn_ego_vehicle(world, bp_lib)
    if not ego_vehicle:
        print("主车生成失败")
        cleanup(world, [], display)
        return

    print("主车已生成，定速巡航就绪")
    vision_system = VisionSystem(ego_vehicle, world)
    lane_planner = LanePlanner(ego_vehicle, world)

    dummy_target = spawn_target(world, bp_lib, target_wp, choice)
    collision_sensor, collision_flag = setup_collision_sensor(world, bp_lib, ego_vehicle)

    speed_ctrl = SpeedController()
    control = carla.VehicleControl()
    aeb_active = False
    was_aeb = False

    actors = [ego_vehicle, dummy_target, collision_sensor,
              vision_system if hasattr(vision_system, 'destroy') else None]

    try:
        running = True
        while running:
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if keyboard.is_pressed('esc'):
                running = False

            # 键盘控制
            handle_keyboard(speed_ctrl)
            if keyboard.is_pressed('space') or collision_flag[0]:
                speed_ctrl.reset()
                control.hand_brake, control.throttle, control.brake = True, 0.0, MAX_BRAKE_STOP
            else:
                control.hand_brake = False

            # 速度控制
            v = ego_vehicle.get_velocity()
            speed_kmh = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) * 3.6
            control.throttle, control.brake = speed_ctrl.compute(speed_kmh)

            # 视觉 AEB
            was_aeb = aeb_active
            aeb_active = handle_vision_aeb(ego_vehicle, vision_system, lane_planner,
                                           speed_ctrl, control)
            if aeb_active and not was_aeb:
                pass  # 日志已在 handle_vision_aeb 中打印

            # 执行控制
            control.reverse = speed_ctrl.is_reverse
            ego_vehicle.apply_control(control)
            world.tick()

            # 渲染
            update_spectator(world, ego_vehicle)
            display.render(speed_ctrl.target_kmh, speed_kmh,
                           control.throttle, control.brake,
                           speed_ctrl.is_reverse, aeb_active)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        if hasattr(vision_system, 'destroy'):
            vision_system.destroy()
        cleanup(world, [ego_vehicle, dummy_target, collision_sensor], display)


if __name__ == '__main__':
    main()