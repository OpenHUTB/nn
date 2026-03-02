#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA 0.9.10 - 无绝对路径优化版
保持原有自动驾驶功能，仅优化CARLA加载逻辑
"""
import sys
import os
import time
import math
from typing import Optional


# ====================== 1. 智能加载CARLA（无硬编码绝对路径） ======================
def load_carla() -> Optional[object]:
    """
    智能加载CARLA Python API，优先级：
    1. 系统环境变量 CARLA_ROOT（推荐）
    2. 自动搜索常见目录（当前目录、用户目录、上级目录）
    3. 引导用户手动输入路径
    """
    python_version = f"py{sys.version_info.major}.{sys.version_info.minor}"
    egg_file_patterns = [
        f"carla-0.9.10-{python_version}-win-amd64.egg",
        "carla-0.9.10-py3.7-win-amd64.egg",  # 兼容Python3.7（CARLA 0.9.10主流版本）
        "carla-0.9.10-*.egg"  # 兜底匹配所有0.9.10版本的egg文件
    ]

    # 候选路径列表（无任何硬编码绝对路径）
    candidate_paths = []

    # 优先级1：从环境变量CARLA_ROOT读取
    carla_root = os.getenv("CARLA_ROOT")
    if carla_root and os.path.isdir(carla_root):
        candidate_paths.append(os.path.join(carla_root, "PythonAPI", "carla", "dist"))

    # 优先级2：自动搜索常见目录
    search_bases = [
        os.getcwd(),  # 当前工作目录
        os.path.dirname(os.getcwd()),  # 上级目录
        os.path.expanduser("~"),  # 用户主目录
        os.path.expanduser("~/Documents"),  # 文档目录
    ]
    for base in search_bases:
        candidate_paths.append(os.path.join(base, "PythonAPI", "carla", "dist"))
        candidate_paths.append(os.path.join(base, "WindowsNoEditor", "PythonAPI", "carla", "dist"))

    # 遍历候选路径，查找有效的egg文件
    for dist_path in candidate_paths:
        if not os.path.isdir(dist_path):
            continue

        # 匹配egg文件
        for file in os.listdir(dist_path):
            if any(file.startswith(pattern.replace("*", "")) or file == pattern for pattern in egg_file_patterns):
                egg_path = os.path.join(dist_path, file)
                sys.path.append(egg_path)
                try:
                    import carla
                    print(f"✅ CARLA Python API 加载成功：{egg_path}")
                    return carla
                except ImportError as e:
                    print(f"⚠️  加载{egg_path}失败：{str(e)[:50]}")
                    continue

    # 优先级3：引导用户手动输入路径
    print("\n❌ 未自动找到CARLA egg文件！")
    print("📌 推荐配置环境变量（一劳永逸）：")
    print("   Windows: set CARLA_ROOT=你的CARLA安装目录（如D:\WindowsNoEditor）")
    print("   Linux/Mac: export CARLA_ROOT=你的CARLA安装目录")

    while True:
        manual_egg_path = input("\n请输入CARLA egg文件的完整路径：").strip()
        if not manual_egg_path:
            continue
        if os.path.isfile(manual_egg_path) and manual_egg_path.endswith(".egg"):
            sys.path.append(manual_egg_path)
            try:
                import carla
                print(f"✅ 手动加载CARLA成功：{manual_egg_path}")
                return carla
            except ImportError:
                print("❌ 该egg文件与当前Python版本不兼容，请重新输入！")
        else:
            print("❌ 路径无效或不是egg文件，请重新输入！")

    return None


# 加载CARLA核心模块
carla = load_carla()
if not carla:
    print("❌ CARLA加载失败，程序退出")
    sys.exit(1)

# ====================== 2. 核心配置参数（完全保留原有设置） ======================
# 速度控制（低速平稳）
BASE_SPEED = 1.5  # 直道基础速度 (m/s)
CURVE_TARGET_SPEED = 1.0  # 弯道目标速度 (m/s)
SPEED_DEADZONE = 0.1  # 速度死区（避免微小波动）
ACCELERATION_FACTOR = 0.04  # 油门调整幅度
DECELERATION_FACTOR = 0.06  # 刹车调整幅度
SPEED_TRANSITION_RATE = 0.03  # 速度过渡率（渐进减速/加速）

# 弯道识别与晚转弯控制
LOOKAHEAD_DISTANCE = 20.0  # 前瞻距离（提前减速）
WAYPOINT_STEP = 1.0  # 道路点步长
CURVE_DETECTION_THRESHOLD = 2.0  # 弯道判定阈值（角度偏差>2度）
TURN_TRIGGER_DISTANCE_IDX = 4  # 晚转弯触发点（前方5米）

# 转向控制（超大角度+快速响应）
STEER_ANGLE_MAX = 0.85  # 最大转向角（拉满）
STEER_RESPONSE_FACTOR = 0.4  # 转向响应速度
STEER_AMPLIFY = 1.6  # 转向角放大系数
MIN_STEER = 0.2  # 最小转向力度

# 出生点偏移
SPAWN_OFFSET_X = -2.0  # X轴左移2米
SPAWN_OFFSET_Y = 0.0  # Y轴不偏移
SPAWN_OFFSET_Z = 0.0  # Z轴不偏移


# ====================== 3. 核心工具函数（功能完全不变） ======================
def get_road_direction_ahead(vehicle, world):
    """
    获取前方道路方向，判定是否为弯道
    返回：目标航向角、是否为弯道、航向偏差
    """
    vehicle_transform = vehicle.get_transform()
    carla_map = world.get_map()

    # 收集前方道路点
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

    # 取前方5米处的道路点（晚转弯核心）
    target_wp_idx = min(TURN_TRIGGER_DISTANCE_IDX, len(waypoints) - 1)
    target_wp = waypoints[target_wp_idx]
    target_yaw = target_wp.transform.rotation.yaw

    # 计算航向偏差
    current_yaw = vehicle_transform.rotation.yaw
    yaw_diff = target_yaw - current_yaw
    yaw_diff = (yaw_diff + 180) % 360 - 180  # 标准化到-180~180°
    is_curve = abs(yaw_diff) > CURVE_DETECTION_THRESHOLD

    return target_yaw, is_curve, yaw_diff


def calculate_steer_angle(current_yaw, target_yaw):
    """计算超大角度转向角，保证足够转向力度"""
    yaw_diff = target_yaw - current_yaw
    yaw_diff = (yaw_diff + 180) % 360 - 180

    # 计算并放大转向角
    steer = (yaw_diff / 180.0 * STEER_ANGLE_MAX) * STEER_AMPLIFY
    steer = max(-STEER_ANGLE_MAX, min(STEER_ANGLE_MAX, steer))

    # 强制最小转向力度
    if abs(steer) > 0.05 and abs(steer) < MIN_STEER:
        steer = MIN_STEER * (1 if steer > 0 else -1)

    return steer


# ====================== 4. 主驾驶逻辑（功能完全不变） ======================
def main():
    # 1. 连接CARLA服务器
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.load_world('Town01')
        world.set_weather(carla.WeatherParameters.ClearNoon)
        # 设置世界参数（非同步模式，降低复杂度）
        world.apply_settings(carla.WorldSettings(
            synchronous_mode=False,
            fixed_delta_seconds=0.1
        ))
        print("✅ 已连接CARLA并加载Town01地图")
    except Exception as e:
        print(f"❌ 连接CARLA失败：{e}")
        return

    # 2. 清理场景中旧车辆
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()
    print("✅ 已清理场景中旧车辆")

    # 3. 生成车辆（出生点左移2米）
    bp_lib = world.get_blueprint_library()
    veh_bp = bp_lib.filter("vehicle")[0]
    veh_bp.set_attribute('color', '255,0,0')  # 红色车辆

    # 获取原始生成点并调整偏移
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

    # 生成车辆
    vehicle = world.spawn_actor(veh_bp, spawn_point)
    print(f"✅ 车辆生成成功（出生点左移{abs(SPAWN_OFFSET_X)}米）")
    print(f"   生成位置：X={spawn_point.location.x:.1f}, Y={spawn_point.location.y:.1f}")

    # 4. 设置俯视视角（同步车辆位置）
    spectator = world.get_spectator()
    spec_transform = carla.Transform(
        carla.Location(spawn_point.location.x, spawn_point.location.y, 40.0),
        carla.Rotation(pitch=-85.0, yaw=spawn_point.rotation.yaw, roll=0.0)
    )
    spectator.set_transform(spec_transform)
    print("✅ 已设置俯视视角，对准车辆")

    # 5. 初始化控制参数
    control = carla.VehicleControl()
    control.hand_brake = False
    control.manual_gear_shift = False
    control.gear = 1

    current_steer = 0.0
    current_target_speed = BASE_SPEED
    last_throttle = 0.0
    last_brake = 0.0

    # 6. 核心驾驶循环
    print(f"\n🚗 开始自动驾驶 | 直道{BASE_SPEED}m/s | 弯道{CURVE_TARGET_SPEED}m/s")
    print("💡 按 Ctrl+C 停止程序\n")

    try:
        while True:
            # 获取车辆当前状态
            velocity = vehicle.get_velocity()
            current_speed = math.hypot(velocity.x, velocity.y)
            current_yaw = vehicle.get_transform().rotation.yaw

            # 识别弯道与目标航向
            target_yaw, is_curve, yaw_diff = get_road_direction_ahead(vehicle, world)

            # 弯道渐进减速/直道恢复速度
            if is_curve:
                current_target_speed = max(CURVE_TARGET_SPEED, current_target_speed - SPEED_TRANSITION_RATE)
            else:
                current_target_speed = min(BASE_SPEED, current_target_speed + SPEED_TRANSITION_RATE / 2)

            # 平滑速度控制（无抖动）
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

            # 超大角度转向控制
            target_steer = calculate_steer_angle(current_yaw, target_yaw)
            current_steer = current_steer + (target_steer - current_steer) * STEER_RESPONSE_FACTOR
            control.steer = current_steer

            # 下发控制指令
            vehicle.apply_control(control)

            # 实时状态显示
            curve_status = "🔴 弯道（减速中）" if is_curve else "🟢 直道"
            status_info = (
                f"{curve_status:12s} | 航向偏差:{yaw_diff:.0f}° "
                f"| 转向角:{current_steer:.2f}(最大:{STEER_ANGLE_MAX}) "
                f"| 速度:{current_speed:.2f}m/s(目标:{current_target_speed:.2f})"
            )
            print(f"\r{status_info}", end="")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n🛑 接收到停止指令，正在清理资源...")
    finally:
        # 销毁车辆，恢复世界设置
        if vehicle and vehicle.is_alive:
            vehicle.destroy()
            print("✅ 车辆已销毁")
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        print("✅ 程序正常退出")


# ====================== 程序入口 ======================
if __name__ == "__main__":
    main()