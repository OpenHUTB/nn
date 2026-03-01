#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carla 0.9.10 路侧感知采集（可视化版）
适配0.9.10：移除draw_circle，用draw_line模拟激光雷达范围
运行前：启动CarlaUE4.exe，等待1分钟初始化.
"""
import sys
import os
import time
import json
import math
from typing import Dict, Any, List, Optional
import carla

# ===================== 全局配置常量（仅使用相对路径） =====================
# Carla连接配置
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 20.0
CARLA_MAP = "Town01"

# 数据采集配置（全部使用相对路径）
SAVE_DIR = "carla_sensor_data"  # 相对当前运行目录
VEHICLE_NUM = 3
TARGET_VEHICLE_MODEL = "vehicle.tesla.model3"
VEHICLE_COLOR = "0,0,0"  # 黑色

# 可视化配置
VISUALIZATION_DURATION = 30.0  # 可视化效果持续30秒
LIDAR_RANGE = 100.0  # 激光雷达范围
LIDAR_SEGMENTS = 36  # 激光雷达模拟圆的线段数
RSU_BOX_SIZE = carla.Vector3D(1, 1, 2)  # 路侧单元可视化尺寸
VEHICLE_BOX_SIZE = carla.Vector3D(2, 1, 1)  # 车辆可视化尺寸

# 颜色配置（RGB）
COLOR_RED = carla.Color(255, 0, 0)  # 路侧单元
COLOR_BLUE = carla.Color(0, 0, 255)  # 激光雷达范围
COLOR_GREEN = carla.Color(0, 255, 0)  # 车辆边框
COLOR_YELLOW = carla.Color(255, 255, 0)  # 车辆文字


# ===================== 路径工具函数（纯相对路径） =====================
def get_relative_path(*path_parts: str) -> str:
    """
    构建并返回相对路径（基于当前工作目录）
    Args:
        path_parts: 路径片段，如("data", "sensor", "file.json")
    Returns:
        规范化的相对路径字符串
    """
    # 拼接路径片段并规范化
    relative_path = os.path.join(*path_parts)
    # 返回相对路径（确保不以/开头）
    return os.path.normpath(relative_path)


def setup_carla_environment() -> None:
    """
    配置Carla Python环境（仅使用环境变量或相对路径，无绝对路径硬编码）
    优先从环境变量CARLA_EGG_PATH读取，不依赖任何固定路径
    """
    # 1. 仅从环境变量获取egg路径（不再尝试任何绝对默认路径）
    carla_egg_path = os.getenv("CARLA_EGG_PATH")

    if carla_egg_path:
        # 处理环境变量中的相对路径
        egg_path = get_relative_path(carla_egg_path) if not os.path.isabs(carla_egg_path) else carla_egg_path

        if os.path.exists(egg_path):
            sys.path.append(egg_path)
            print(f"✅ 从环境变量加载Carla egg文件: {egg_path}")
            return
        else:
            print(f"\n⚠️  环境变量CARLA_EGG_PATH指定的路径不存在: {egg_path}")

    # 2. 路径配置失败（仅提示，不提供绝对路径示例）
    print("\n❌ 未找到Carla egg文件！请按以下方式配置：")
    print("   1. 设置环境变量（支持相对路径）：")
    print("      Windows: set CARLA_EGG_PATH=相对路径/到/carla.egg")
    print("      Linux/Mac: export CARLA_EGG_PATH=相对路径/到/carla.egg")
    print("   2. 确保路径相对于当前运行目录，或使用完整绝对路径")
    sys.exit(1)


# ===================== Carla连接与初始化 =====================
def connect_to_carla() -> tuple[carla.Client, carla.World, carla.Transform]:
    """
    连接Carla模拟器并初始化世界（无路径依赖）

    Returns:
        tuple: (client实例, world实例, 观察者初始变换)
    """
    try:
        # 建立连接
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(CARLA_TIMEOUT)

        # 加载地图并等待初始化
        world = client.load_world(CARLA_MAP)
        time.sleep(3)

        # 获取观察者（视角）初始位置
        spectator = world.get_spectator()
        spectator_transform = spectator.get_transform()

        # 打印连接信息
        loc = spectator_transform.location
        print(f"\n✅ 成功连接Carla模拟器: {CARLA_HOST}:{CARLA_PORT}")
        print(f"✅ 加载地图: {CARLA_MAP}")
        print(f"✅ 初始视角位置: x={loc.x:.1f}, y={loc.y:.1f}, z={loc.z:.1f}")

        return client, world, spectator_transform

    except Exception as e:
        print(f"\n❌ 连接Carla失败: {str(e)}")
        sys.exit(1)


# ===================== 车辆管理 =====================
def clear_existing_vehicles(world: carla.World) -> None:
    """清除世界中所有现有车辆"""
    vehicles = world.get_actors().filter("vehicle.*")
    for vehicle in vehicles:
        vehicle.destroy()
    print(f"🗑️  已清除 {len(vehicles)} 辆现有车辆")


def get_vehicle_blueprint(world: carla.World) -> carla.ActorBlueprint:
    """
    获取车辆蓝图（优先特斯拉Model3，黑色）

    Args:
        world: Carla世界实例

    Returns:
        配置好的车辆蓝图
    """
    blueprint_lib = world.get_blueprint_library()

    # 优先获取指定车型
    try:
        vehicle_bp = blueprint_lib.find(TARGET_VEHICLE_MODEL)
        vehicle_bp.set_attribute("color", VEHICLE_COLOR)
        print(f"✅ 使用车辆蓝图: {TARGET_VEHICLE_MODEL} (黑色)")
        return vehicle_bp
    except IndexError:
        # 备选方案：使用第一个可用车辆蓝图
        vehicle_bp = blueprint_lib.filter("vehicle.*")[0]
        print(f"⚠️  指定车型不可用，使用默认车型: {vehicle_bp.id}")
        return vehicle_bp


def calculate_vehicle_spawn_positions(
        spectator_transform: carla.Transform
) -> List[carla.Location]:
    """
    计算视角前方的车辆生成位置

    Args:
        spectator_transform: 观察者变换信息

    Returns:
        车辆生成位置列表
    """
    yaw_rad = math.radians(spectator_transform.rotation.yaw)
    base_distance = 5  # 第一辆车距离视角的基础距离
    position_offset = 3  # 车辆之间的间距

    spawn_positions = []
    for i in range(VEHICLE_NUM):
        # 计算当前车辆的生成位置（视角前方，略有偏移）
        distance = base_distance + i * position_offset
        x = spectator_transform.location.x + distance * math.cos(yaw_rad)
        y = spectator_transform.location.y + distance * math.sin(yaw_rad) + (i % 2 - 0.5) * 2

        spawn_positions.append(carla.Location(x=x, y=y, z=0.5))

    return spawn_positions


def spawn_vehicles(
        world: carla.World,
        spectator_transform: carla.Transform
) -> List[carla.Actor]:
    """
    在视角前方生成指定数量的车辆

    Args:
        world: Carla世界实例
        spectator_transform: 观察者变换信息

    Returns:
        成功生成的车辆列表
    """
    # 清除旧车辆
    clear_existing_vehicles(world)

    # 获取车辆蓝图
    vehicle_bp = get_vehicle_blueprint(world)

    # 计算生成位置
    spawn_positions = calculate_vehicle_spawn_positions(spectator_transform)

    # 生成车辆
    spawned_vehicles = []
    for i, position in enumerate(spawn_positions):
        try:
            # 车辆朝向与视角相反
            vehicle_transform = carla.Transform(
                position,
                carla.Rotation(yaw=spectator_transform.rotation.yaw + 180)
            )

            # 生成车辆
            vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
            if vehicle:
                spawned_vehicles.append(vehicle)
                print(f"🚗 成功生成第{i + 1}辆车 (位置: x={position.x:.1f}, y={position.y:.1f})")
                time.sleep(1)

        except Exception as e:
            print(f"⚠️  第{i + 1}辆车生成失败: {str(e)}")
            continue

    print(f"\n✅ 车辆生成完成: 成功 {len(spawned_vehicles)}/{VEHICLE_NUM} 辆")
    return spawned_vehicles


# ===================== 可视化 =====================
def draw_rsu_marker(
        debug: carla.DebugHelper,
        rsu_transform: carla.Transform
) -> None:
    """绘制路侧单元（RSU）标记"""
    # 绘制红色立方体
    debug.draw_box(
        box=carla.BoundingBox(rsu_transform.location, RSU_BOX_SIZE),
        rotation=rsu_transform.rotation,
        thickness=0.1,
        color=COLOR_RED,
        life_time=VISUALIZATION_DURATION
    )

    # 绘制文字标注
    debug.draw_string(
        rsu_transform.location + carla.Location(z=2),
        "RSU_001（路侧单元）",
        color=COLOR_RED,
        life_time=VISUALIZATION_DURATION
    )


def draw_lidar_range(
        debug: carla.DebugHelper,
        center: carla.Location
) -> None:
    """用线段模拟绘制激光雷达范围（圆形）"""
    # 绘制圆形边框
    for i in range(LIDAR_SEGMENTS):
        angle1 = math.radians(i * (360 / LIDAR_SEGMENTS))
        angle2 = math.radians((i + 1) * (360 / LIDAR_SEGMENTS))

        start = carla.Location(
            x=center.x + LIDAR_RANGE * math.cos(angle1),
            y=center.y + LIDAR_RANGE * math.sin(angle1),
            z=center.z + 0.1
        )

        end = carla.Location(
            x=center.x + LIDAR_RANGE * math.cos(angle2),
            y=center.y + LIDAR_RANGE * math.sin(angle2),
            z=center.z + 0.1
        )

        debug.draw_line(
            start, end,
            thickness=0.5,
            color=COLOR_BLUE,
            life_time=VISUALIZATION_DURATION
        )

    # 绘制激光雷达范围文字
    debug.draw_string(
        center + carla.Location(z=3),
        f"激光雷达范围：{LIDAR_RANGE}m",
        color=COLOR_BLUE,
        life_time=VISUALIZATION_DURATION
    )


def draw_vehicle_markers(
        debug: carla.DebugHelper,
        vehicles: List[carla.Actor]
) -> None:
    """为每辆车绘制可视化标记"""
    for idx, vehicle in enumerate(vehicles):
        v_transform = vehicle.get_transform()
        v_loc = v_transform.location

        # 绘制绿色立方体边框
        debug.draw_box(
            box=carla.BoundingBox(v_loc, VEHICLE_BOX_SIZE),
            rotation=v_transform.rotation,
            thickness=0.1,
            color=COLOR_GREEN,
            life_time=VISUALIZATION_DURATION
        )

        # 绘制黄色文字标注
        debug.draw_string(
            v_loc + carla.Location(z=1.5),
            f"车辆{idx + 1}\nID:{vehicle.id}\nx:{v_loc.x:.1f}, y:{v_loc.y:.1f}",
            color=COLOR_YELLOW,
            life_time=VISUALIZATION_DURATION
        )


def visualize_scene(
        world: carla.World,
        spectator_transform: carla.Transform,
        vehicles: List[carla.Actor]
) -> None:
    """
    可视化整个场景：路侧单元、激光雷达范围、车辆标记

    Args:
        world: Carla世界实例
        spectator_transform: 路侧单元位置
        vehicles: 已生成的车辆列表
    """
    debug = world.debug

    # 绘制路侧单元
    draw_rsu_marker(debug, spectator_transform)

    # 绘制激光雷达范围
    draw_lidar_range(debug, spectator_transform.location)

    # 绘制车辆标记
    draw_vehicle_markers(debug, vehicles)

    print(f"✅ 场景可视化完成（效果持续{VISUALIZATION_DURATION}秒）")


# ===================== 数据采集与保存（纯相对路径） =====================
def collect_roadside_data(
        world: carla.World,
        vehicles: List[carla.Actor],
        rsu_transform: carla.Transform
) -> Dict[str, Any]:
    """
    采集路侧感知数据

    Args:
        world: Carla世界实例
        vehicles: 检测到的车辆列表
        rsu_transform: 路侧单元变换信息

    Returns:
        结构化的感知数据字典
    """
    try:
        # 传感器配置信息
        sensor_config = {
            "lidar": {"range": f"{LIDAR_RANGE}m", "frequency": "10Hz"},
            "camera": {"resolution": "1920x1080"}
        }

        # 车辆数据采集
        vehicle_data = []
        for vehicle in vehicles:
            trans = vehicle.get_transform()
            vehicle_data.append({
                "id": vehicle.id,
                "model": vehicle.type_id,
                "location": {
                    "x": float(trans.location.x),
                    "y": float(trans.location.y),
                    "z": float(trans.location.z)
                },
                "rotation": {
                    "yaw": float(trans.rotation.yaw),
                    "pitch": float(trans.rotation.pitch),
                    "roll": float(trans.rotation.roll)
                }
            })

        # 组装完整数据
        collected_data = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "roadside_unit": {
                "id": "RSU_001",
                "location": {
                    "x": float(rsu_transform.location.x),
                    "y": float(rsu_transform.location.y),
                    "z": float(rsu_transform.location.z)
                }
            },
            "sensor_config": sensor_config,
            "detected_vehicles": vehicle_data,
            "vehicle_count": len(vehicle_data),
            "working_directory": os.getcwd()  # 记录当前工作目录（便于调试）
        }

        return collected_data

    except Exception as e:
        print(f"⚠️  数据采集失败: {str(e)}")
        return {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "error": str(e),
            "vehicle_count": 0,
            "working_directory": os.getcwd()
        }


def save_roadside_data(data: Dict[str, Any]) -> None:
    """
    保存采集的数据到JSON文件（仅使用相对路径）

    Args:
        data: 要保存的感知数据字典
    """
    # 创建保存目录（纯相对路径）
    save_dir = get_relative_path(SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)

    # 生成文件名并保存（纯相对路径）
    filename = f"roadside_data_{data['timestamp']}.json"
    filepath = get_relative_path(save_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # 打印相对路径和绝对路径（便于用户查看）
    abs_filepath = os.path.abspath(filepath)
    print(f"✅ 数据已保存至:")
    print(f"   相对路径: {filepath}")
    print(f"   绝对路径: {abs_filepath}")


# ===================== 主函数 =====================
def main() -> None:
    """程序主入口（无任何绝对路径依赖）"""
    print("=" * 70)
    print("Carla 0.9.10 路侧感知采集系统（纯相对路径版）")
    print(f"当前工作目录: {os.getcwd()}")
    print("=" * 70)

    try:
        # 1. 配置Carla环境（仅通过环境变量）
        setup_carla_environment()

        # 2. 连接Carla模拟器
        client, world, spectator_transform = connect_to_carla()

        # 3. 生成车辆
        spawned_vehicles = spawn_vehicles(world, spectator_transform)

        # 4. 场景可视化
        visualize_scene(world, spectator_transform, spawned_vehicles)

        # 5. 调整视角（略微向下）
        spectator = world.get_spectator()
        new_rotation = carla.Rotation(
            pitch=spectator_transform.rotation.pitch - 5,
            yaw=spectator_transform.rotation.yaw,
            roll=spectator_transform.rotation.roll
        )
        spectator.set_transform(carla.Transform(spectator_transform.location, new_rotation))
        print("✅ 视角已调整（略微向下）")

        # 6. 采集并保存数据（纯相对路径）
        print("\n🔍 开始采集路侧感知数据...")
        sensor_data = collect_roadside_data(world, spawned_vehicles, spectator_transform)
        save_roadside_data(sensor_data)

        # 7. 输出采集结果
        print("\n" + "=" * 70)
        print(f"📊 采集完成！共检测到 {sensor_data['vehicle_count']} 辆车辆")
        print(f"💡 可视化效果将持续{VISUALIZATION_DURATION}秒，可开始录制视频")
        print(f"💾 数据保存目录（相对路径）: {SAVE_DIR}")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 用户中断程序执行")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 致命错误: {str(e)}")
        sys.exit(1)