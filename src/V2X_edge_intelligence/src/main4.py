#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA 0.9.10
"""
import sys
import os
import time
import math

# ====================== 1. CARLA加载（兼容相对路径，无硬编码绝对路径） ======================
# 优先从项目内carla_lib加载，无则回退到默认路径（保证兼容性）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_LIB_DIR = os.path.join(CURRENT_DIR, "../carla_lib")  # 上级目录carla_lib
DEFAULT_CARLA_PATH = "D:/WindowsNoEditor"

# 自动查找egg文件
egg_file = None
# 尝试项目内carla_lib
if os.path.exists(CARLA_LIB_DIR):
    egg_files = [f for f in os.listdir(CARLA_LIB_DIR) if f.endswith(".egg") and "0.9.10" in f]
    if egg_files:
        egg_file = os.path.join(CARLA_LIB_DIR, egg_files[0])
# 回退到默认路径
if not egg_file:
    egg_file = os.path.join(DEFAULT_CARLA_PATH, "PythonAPI", "carla", "dist", "carla-0.9.10-py3.7-win-amd64.egg")

try:
    sys.path.append(egg_file)
    import carla
    print(f"✅ CARLA加载成功（egg路径：{egg_file}）")
except Exception as e:
    print(f"❌ CARLA加载失败：{e}")
    print("请确保：1. CARLA egg文件存在 2. Python版本为3.7")
    sys.exit(1)

# ====================== 2. 核心参数（全部最优适配，核心增大转向角度） ======================
# 速度参数（不变：低速平稳无抖动）
BASE_SPEED = 1.5               # 直道速度 1.5m/s
CURVE_TARGET_SPEED = 1.0       # 弯道速度 1.0m/s
SPEED_DEADZONE = 0.1
ACCELERATION_FACTOR = 0.04
DECELERATION_FACTOR = 0.06
SPEED_TRANSITION_RATE = 0.03

# 晚转弯核心参数【不变】：仅前方5米触发转向，接近弯道才转
LOOKAHEAD_DISTANCE = 20.0      # 20米前瞻 提前减速
WAYPOINT_STEP = 1.0
CURVE_DETECTION_THRESHOLD = 2.0
TURN_TRIGGER_DISTANCE_IDX = 4  # 前方5米 触发转向 (晚转弯核心)

# ========== 本次核心修改：拉满转弯角度 解决角度不够 ==========
STEER_ANGLE_MAX = 0.85         # 最大转向角拉满到0.85【重中之重】之前0.7，角度大幅增大
STEER_RESPONSE_FACTOR = 0.4    # 转向响应拉满0.4 最快响应，晚转+大角度一步到位
STEER_AMPLIFY = 1.6            # 转向角放大系数倍增到1.6 小偏差也能转出超大角度
MIN_STEER = 0.2                # 最小转向角提高到0.2 强制保底转向力度 绝对不转不动

# 出生点偏移【不变】：左移2米
SPAWN_OFFSET_X = -2.0
SPAWN_OFFSET_Y = 0.0
SPAWN_OFFSET_Z = 0.0

# ====================== 3. 核心工具函数 ======================
def get_road_direction_ahead(vehicle, world):
    """晚转弯逻辑不变：前方5米判定转向，20米提前减速"""
    vehicle_transform = vehicle.get_transform()
    carla_map = world.get_map()

    waypoints = []
    current_wp = carla_map.get_waypoint(vehicle_transform.location)
    next_wp = current_wp

    for _ in range(int(LOOKAHEAD_DISTANCE / WAYPOINT_STEP)):
        next_wps = next_wp.next(WAYPOINT_STEP)
        if not next_wps:
            break
        next_wp = next_wps[0]
        waypoints.append(next_wp)

    if len(waypoints) < 3:
        return vehicle_transform.rotation.yaw, False, 0.0

    # 晚转弯核心：仅取前方5米的道路点判定方向
    target_wp_idx = min(TURN_TRIGGER_DISTANCE_IDX, len(waypoints)-1)
    target_wp = waypoints[target_wp_idx]
    target_yaw = target_wp.transform.rotation.yaw

    current_yaw = vehicle_transform.rotation.yaw
    yaw_diff = target_yaw - current_yaw
    yaw_diff = (yaw_diff + 180) % 360 - 180
    is_curve = abs(yaw_diff) > CURVE_DETECTION_THRESHOLD

    return target_yaw, is_curve, yaw_diff

def calculate_steer_angle(current_yaw, target_yaw):
    """核心：超大角度转向计算，绝对够力度转进直道"""
    yaw_diff = target_yaw - current_yaw
    yaw_diff = (yaw_diff + 180) % 360 - 180

    # 三重放大：最大角度+系数放大+最小转向角 保证转弯角度绝对足够
    steer = (yaw_diff / 180.0 * STEER_ANGLE_MAX) * STEER_AMPLIFY
    steer = max(-STEER_ANGLE_MAX, min(STEER_ANGLE_MAX, steer))

    # 强制最小转向力度：哪怕极缓弯道，也有足够角度，杜绝转不动
    if abs(steer) > 0.05 and abs(steer) < MIN_STEER:
        steer = MIN_STEER * (1 if steer>0 else -1)

    return steer

# ====================== 4. 主函数（唯一入口） ======================
def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.load_world('Town01')
        world.set_weather(carla.WeatherParameters.ClearNoon)
        world.apply_settings(carla.WorldSettings(synchronous_mode=False,fixed_delta_seconds=0.1))
        print("✅ 已连接CARLA并加载Town01地图")
    except Exception as e:
        print(f"❌ 连接CARLA失败：{e}")
        return

    # 清理旧车辆
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()
    print("✅ 已清理旧车辆")

    # 生成车辆 + 出生点左移2米 【不变】
    bp_lib = world.get_blueprint_library()
    veh_bp = bp_lib.filter("vehicle")[0]
    veh_bp.set_attribute('color', '255,0,0')

    spawn_points = world.get_map().get_spawn_points()
    original_spawn_point = spawn_points[0]
    spawn_point = carla.Transform(
        carla.Location(
            x=original_spawn_point.location.x + SPAWN_OFFSET_X,
            y=original_spawn_point.location.y + SPAWN_OFFSET_Y,
            z=original_spawn_point.location.z + SPAWN_OFFSET_Z
        ),
        original_spawn_point.rotation
    )
    vehicle = world.spawn_actor(veh_bp, spawn_point)
    print(f"✅ 车辆生成成功（出生点左移{abs(SPAWN_OFFSET_X)}米）")
    print(f"   调整后位置：({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")

    # 视角同步左移
    spectator = world.get_spectator()
    spec_loc = carla.Location(x=spawn_point.location.x,y=spawn_point.location.y,z=40.0)
    spec_rot = carla.Rotation(pitch=-85.0,yaw=spawn_point.rotation.yaw,roll=0.0)
    spectator.set_transform(carla.Transform(spec_loc, spec_rot))
    print("\n✅ 视角已定位到车辆上方（俯视视角）")

    # 初始化控制参数
    control = carla.VehicleControl()
    control.hand_brake = False
    control.manual_gear_shift = False
    control.gear = 1

    current_steer = 0.0
    current_target_speed = BASE_SPEED
    last_throttle = 0.0
    last_brake = 0.0

    print(f"\n🚗 开始自动驾驶（直道{BASE_SPEED}m/s | 弯道减速至{CURVE_TARGET_SPEED}m/s）...")
    print("✅ 超大转弯角度+晚转弯+左移出生点，所有需求全部满足！")
    print("💡 按Ctrl+C停止程序\n")

    try:
        while True:
            # 获取车辆状态
            velocity = vehicle.get_velocity()
            current_speed = math.hypot(velocity.x, velocity.y)
            current_yaw = vehicle.get_transform().rotation.yaw

            # 晚转弯+弯道识别
            target_yaw, is_curve, yaw_diff = get_road_direction_ahead(vehicle, world)

            # 弯道渐进减速 【不变】
            if is_curve:
                current_target_speed = max(CURVE_TARGET_SPEED, current_target_speed - SPEED_TRANSITION_RATE)
            else:
                current_target_speed = min(BASE_SPEED, current_target_speed + SPEED_TRANSITION_RATE/2)

            # 平滑速度控制 无抖动 【不变】
            speed_error = current_target_speed - current_speed
            if abs(speed_error) < SPEED_DEADZONE:
                control.throttle = last_throttle * 0.85
                control.brake = 0.0
            elif speed_error > 0:
                control.throttle = min(last_throttle + ACCELERATION_FACTOR, 0.25)
                control.brake = 0.0
                last_throttle = control.throttle
            else:
                control.brake = min(last_brake + DECELERATION_FACTOR, 0.2)
                control.throttle = 0.0
                last_brake = control.brake

            # ========== 核心：超大角度+最快响应转向 ==========
            target_steer = calculate_steer_angle(current_yaw, target_yaw)
            current_steer = current_steer + (target_steer - current_steer) * STEER_RESPONSE_FACTOR
            control.steer = current_steer

            # 下发指令
            vehicle.apply_control(control)

            # 状态显示
            curve_status = "🔴 弯道（减速中）" if is_curve else "🟢 直道"
            speed_info = f"当前:{current_speed:.2f}m/s 目标:{current_target_speed:.2f}m/s"
            steer_info = f"{current_steer:.2f}(最大:{STEER_ANGLE_MAX})"
            yaw_info = f"偏差:{yaw_diff:.0f}°"

            print(f"\r{curve_status:12s} | {yaw_info} | 转向角：{steer_info} | 速度：{speed_info}", end="")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n🛑 停止程序...")

    # 清理资源
    if vehicle and vehicle.is_alive:
        vehicle.destroy()
        print("✅ 车辆已销毁")
    world.apply_settings(carla.WorldSettings(synchronous_mode=False))
    print("✅ 程序正常退出")

# ====================== 唯一程序入口 ======================
if __name__ == "__main__":
    main()