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
from typing import Dict, Any


# ========== 加载Carla egg文件（移除绝对路径，适配多环境） ==========
def load_carla_egg():
    """
    加载Carla egg文件的容错逻辑：
    1. 优先从CARLA_EGG_PATH环境变量读取
    2. 其次从Carla默认安装路径查找
    3. 最后提示用户手动指定
    """
    # 1. 从环境变量获取（推荐，用户可灵活配置）
    carla_egg_path = os.getenv("CARLA_EGG_PATH")
    if carla_egg_path and os.path.exists(carla_egg_path):
        sys.path.append(carla_egg_path)
        return True

    # 2. 尝试Carla默认安装路径（Windows）
    default_paths = [
        # 默认安装路径
        r"CarlaUE4\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg",
        # 用户原路径（作为备选，兼容本地运行）
        r"D:\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg"
    ]
    for path in default_paths:
        if os.path.exists(path):
            sys.path.append(path)
            return True

    # 3. 未找到egg文件，提示用户配置
    print("❌ 未找到Carla egg文件！请按以下方式配置：")
    print("   1. 设置环境变量：set CARLA_EGG_PATH=你的Carla egg文件路径")
    print("   2. 或手动修改代码中的default_paths为你的Carla安装路径")
    return False


# 加载Carla并容错
if load_carla_egg():
    try:
        import carla

        print(f"✅ 成功加载Carla API（0.9.10适配版）")
    except Exception as e:
        print(f"❌ 加载Carla API失败：{str(e)}")
        sys.exit(1)
else:
    sys.exit(1)

# ========== 配置项（移除硬编码绝对路径） ==========
CARLA_HOST = "localhost"
CARLA_PORT = 2000
TIMEOUT = 20.0
SAVE_DIR = "carla_sensor_data"
VEHICLE_NUM = 3
# 可视化配置
VISUALIZATION_DURATION = 30.0  # 可视化效果持续30秒
LIDAR_RANGE = 100.0  # 激光雷达范围


# ========== 连接模拟器 ==========
def connect_carla():
    """连接Carla，获取client、world、视角原点"""
    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(TIMEOUT)
        world = client.load_world("Town01")
        time.sleep(3)

        # 获取视角当前的位置
        spectator = world.get_spectator()
        spectator_transform = spectator.get_transform()
        print(f"✅ 视角当前位置：x={spectator_transform.location.x:.1f}, y={spectator_transform.location.y:.1f}")
        print(f"✅ 成功连接Carla（Town01地图）：{CARLA_HOST}:{CARLA_PORT}")
        return client, world, spectator_transform
    except Exception as e:
        print(f"❌ 连接失败：{str(e)}")
        sys.exit(1)


# ========== 在视角前生成车辆 ==========
def spawn_vehicles_in_view(world, spectator_transform):
    """在视角正前方生成车辆，返回生成的车辆列表"""
    # 1. 清除现有车辆
    vehicles = world.get_actors().filter("vehicle.*")
    for v in vehicles:
        v.destroy()
    print(f"🗑️  已清除 {len(vehicles)} 辆旧车辆")

    # 2. 选择黑色特斯拉
    blueprint_lib = world.get_blueprint_library()
    vehicle_bp = blueprint_lib.find("vehicle.tesla.model3")
    vehicle_bp.set_attribute("color", "0,0,0")
    if not vehicle_bp:
        vehicle_bp = blueprint_lib.filter("vehicle.*")[0]

    # 3. 计算视角正前方生成位置
    spawn_positions = [
        carla.Location(
            x=spectator_transform.location.x + 5 * math.cos(math.radians(spectator_transform.rotation.yaw)),
            y=spectator_transform.location.y + 5 * math.sin(math.radians(spectator_transform.rotation.yaw)) + 1,
            z=0.5
        ),
        carla.Location(
            x=spectator_transform.location.x + 8 * math.cos(math.radians(spectator_transform.rotation.yaw)),
            y=spectator_transform.location.y + 8 * math.sin(math.radians(spectator_transform.rotation.yaw)) - 1,
            z=0.5
        ),
        carla.Location(
            x=spectator_transform.location.x + 11 * math.cos(math.radians(spectator_transform.rotation.yaw)),
            y=spectator_transform.location.y + 11 * math.sin(math.radians(spectator_transform.rotation.yaw)),
            z=0.5
        )
    ]

    # 4. 逐个生成车辆并记录
    spawned_vehicles = []
    for i in range(VEHICLE_NUM):
        try:
            vehicle_yaw = spectator_transform.rotation.yaw + 180
            transform = carla.Transform(spawn_positions[i], carla.Rotation(yaw=vehicle_yaw))
            vehicle = world.spawn_actor(vehicle_bp, transform)
            if vehicle:
                spawned_vehicles.append(vehicle)
                print(f"🚗 成功生成第{i + 1}辆车（在视角前{5 + i * 3}米处）")
                time.sleep(1)
        except Exception as e:
            print(f"⚠️  第{i + 1}辆车生成失败：{str(e)}")
            continue

    print(f"✅ 车辆生成完成：成功 {len(spawned_vehicles)}/{VEHICLE_NUM} 辆")
    return spawned_vehicles


# ========== 在CarlaUE4中可视化运行效果（适配0.9.10） ==========
def visualize_in_carla(world, spectator_transform, spawned_vehicles):
    """在CarlaUE4窗口中绘制：车辆ID标注、激光雷达范围（线模拟）、路侧单元位置"""
    debug = world.debug  # Carla 0.9.10调试工具

    # 1. 绘制路侧单元（RSU）位置（红色立方体+文字）
    rsu_location = spectator_transform.location
    debug.draw_box(
        box=carla.BoundingBox(rsu_location, carla.Vector3D(1, 1, 2)),
        rotation=spectator_transform.rotation,
        thickness=0.1,
        color=carla.Color(255, 0, 0),  # 红色
        life_time=VISUALIZATION_DURATION
    )
    debug.draw_string(
        rsu_location + carla.Location(z=2),
        "RSU_001（路侧单元）",
        color=carla.Color(255, 0, 0),
        life_time=VISUALIZATION_DURATION
    )

    # 2. 模拟绘制激光雷达范围（0.9.10支持，线组成圆形）
    center = rsu_location
    num_segments = 36  # 36条线组成圆形，足够平滑
    for i in range(num_segments):
        angle1 = math.radians(i * 10)
        angle2 = math.radians((i + 1) * 10)
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
            color=carla.Color(0, 0, 255),  # 蓝色
            life_time=VISUALIZATION_DURATION
        )
    # 标注激光雷达范围文字
    debug.draw_string(
        center + carla.Location(z=3),
        f"激光雷达范围：{LIDAR_RANGE}m",
        color=carla.Color(0, 0, 255),
        life_time=VISUALIZATION_DURATION
    )

    # 3. 为每辆车添加3D标注（绿色立方体+黄色文字）
    for idx, vehicle in enumerate(spawned_vehicles):
        v_loc = vehicle.get_transform().location
        debug.draw_box(
            box=carla.BoundingBox(v_loc, carla.Vector3D(2, 1, 1)),
            rotation=vehicle.get_transform().rotation,
            thickness=0.1,
            color=carla.Color(0, 255, 0),  # 绿色
            life_time=VISUALIZATION_DURATION
        )
        debug.draw_string(
            v_loc + carla.Location(z=1.5),
            f"车辆{idx + 1}\nID:{vehicle.id}\nx:{v_loc.x:.1f}, y:{v_loc.y:.1f}",
            color=carla.Color(255, 255, 0),  # 黄色
            life_time=VISUALIZATION_DURATION
        )

    print(f"✅ 可视化效果已绘制在CarlaUE4窗口（持续{VISUALIZATION_DURATION}秒）")


# ========== 采集路侧数据 ==========
def get_roadside_data(world, spawned_vehicles, spectator_transform):
    """采集数据，兼容可视化场景"""
    try:
        lidar_cfg = {"range": f"{LIDAR_RANGE}m", "freq": "10Hz"}
        camera_cfg = {"resolution": "1920x1080"}

        vehicle_data = []
        for v in spawned_vehicles:
            trans = v.get_transform()
            vehicle_data.append({
                "id": v.id,
                "model": v.type_id,
                "x": float(trans.location.x),
                "y": float(trans.location.y),
                "z": float(trans.location.z),
                "yaw": float(trans.rotation.yaw)
            })

        return {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "roadside_id": "RSU_001",
            "rsu_location": {
                "x": float(spectator_transform.location.x),
                "y": float(spectator_transform.location.y),
                "z": float(spectator_transform.location.z)
            },
            "lidar_config": lidar_cfg,
            "camera_config": camera_cfg,
            "detected_vehicles": vehicle_data,
            "vehicle_count": len(vehicle_data)
        }
    except Exception as e:
        print(f"⚠️  采集数据异常：{str(e)}")
        return {"timestamp": time.strftime("%Y%m%d_%H%M%S"), "vehicle_count": 0}


# ========== 保存数据 ==========
def save_data(data):
    """保存数据到相对路径（避免绝对路径）"""
    # 使用相对路径+绝对化，兼容不同运行目录
    save_path = os.path.abspath(SAVE_DIR)
    os.makedirs(save_path, exist_ok=True)
    file_name = f"roadside_data_{data['timestamp']}.json"
    file_path = os.path.join(save_path, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ 数据已保存：{file_path}")


# ========== 主函数 ==========
def main():
    print("===== Carla 0.9.10 路侧感知采集（可视化版） =====\n")
    # 1. 连接模拟器
    client, world, spectator_transform = connect_carla()

    # 2. 生成车辆
    spawned_vehicles = spawn_vehicles_in_view(world, spectator_transform)

    # 3. 可视化运行效果
    visualize_in_carla(world, spectator_transform, spawned_vehicles)

    # 4. 调整视角
    spectator = world.get_spectator()
    new_rotation = carla.Rotation(
        pitch=spectator_transform.rotation.pitch - 5,
        yaw=spectator_transform.rotation.yaw,
        roll=spectator_transform.rotation.roll
    )
    spectator.set_transform(carla.Transform(spectator_transform.location, new_rotation))

    # 5. 采集数据
    print("🔍 正在采集路侧感知数据...")
    sensor_data = get_roadside_data(world, spawned_vehicles, spectator_transform)

    # 6. 保存数据
    save_data(sensor_data)

    # 7. 输出结果
    print(f"\n📊 采集完成！共检测到 {sensor_data['vehicle_count']} 辆车辆")
    print(f"\n💡 可视化效果在CarlaUE4窗口持续{VISUALIZATION_DURATION}秒，可开始录视频！")
    print("===== 操作结束 =====\n")


if __name__ == "__main__":
    main()