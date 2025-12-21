#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carla 0.9.10 路侧感知采集（车辆生成在视角前，方便录视频）
运行前：启动D:\WindowsNoEditor\CarlaUE4.exe，等待1分钟初始化
"""
import sys
import os
import time
import json
import math  # 核心修正：添加math库导入
from typing import Dict, Any

# ========== 加载Carla egg文件 ==========
CARLA_EGG_PATH = r"D:\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg"
sys.path.append(CARLA_EGG_PATH)

# 导入Carla并容错
try:
    import carla

    print(f"✅ 成功加载Carla API（0.9.10适配版）")
except Exception as e:
    print(f"❌ 加载Carla API失败：{str(e)}")
    sys.exit(1)

# ========== 配置项 ==========
CARLA_HOST = "localhost"
CARLA_PORT = 2000
TIMEOUT = 20.0
SAVE_DIR = "carla_sensor_data"
VEHICLE_NUM = 3  # 生成3辆（避免画面拥挤，适合录视频）


# ========== 连接模拟器 ==========
def connect_carla():
    """连接Carla，获取client、world、视角原点"""
    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(TIMEOUT)
        world = client.load_world("Town01")
        time.sleep(3)

        # 获取视角当前的位置（第一人称视角原点）
        spectator = world.get_spectator()  # 视角对象
        spectator_transform = spectator.get_transform()
        print(f"✅ 视角当前位置：x={spectator_transform.location.x:.1f}, y={spectator_transform.location.y:.1f}")
        print(f"✅ 成功连接Carla（Town01地图）：{CARLA_HOST}:{CARLA_PORT}")
        return client, world, spectator_transform
    except Exception as e:
        print(f"❌ 连接失败：{str(e)}")
        sys.exit(1)


# ========== 在视角前生成车辆（录视频专用） ==========
def spawn_vehicles_in_view(world, spectator_transform):
    """在视角正前方5-15米处生成车辆，录视频时直接可见"""
    # 1. 清除现有车辆
    vehicles = world.get_actors().filter("vehicle.*")
    for v in vehicles:
        v.destroy()
    print(f"🗑️  已清除 {len(vehicles)} 辆旧车辆")

    # 2. 选择显眼的车型（黑色特斯拉，录视频更清晰）
    blueprint_lib = world.get_blueprint_library()
    vehicle_bp = blueprint_lib.find("vehicle.tesla.model3")
    vehicle_bp.set_attribute("color", "0,0,0")  # 设置黑色（RGB）
    if not vehicle_bp:
        vehicle_bp = blueprint_lib.filter("vehicle.*")[0]

    # 3. 计算视角正前方的生成位置（核心！）
    # 视角正前方5米、8米、11米处，左右偏移1-2米（避免重叠）
    spawn_positions = [
        # 正前方5米，偏右1米
        carla.Location(
            x=spectator_transform.location.x + 5 * math.cos(math.radians(spectator_transform.rotation.yaw)),
            y=spectator_transform.location.y + 5 * math.sin(math.radians(spectator_transform.rotation.yaw)) + 1,
            z=0.5
        ),
        # 正前方8米，偏左1米
        carla.Location(
            x=spectator_transform.location.x + 8 * math.cos(math.radians(spectator_transform.rotation.yaw)),
            y=spectator_transform.location.y + 8 * math.sin(math.radians(spectator_transform.rotation.yaw)) - 1,
            z=0.5
        ),
        # 正前方11米，正中间
        carla.Location(
            x=spectator_transform.location.x + 11 * math.cos(math.radians(spectator_transform.rotation.yaw)),
            y=spectator_transform.location.y + 11 * math.sin(math.radians(spectator_transform.rotation.yaw)),
            z=0.5
        )
    ]

    # 4. 逐个生成车辆（面向视角，录视频更美观）
    spawned_num = 0
    for i in range(VEHICLE_NUM):
        try:
            # 车辆朝向视角（yaw和视角一致+180度）
            vehicle_yaw = spectator_transform.rotation.yaw + 180
            transform = carla.Transform(spawn_positions[i], carla.Rotation(yaw=vehicle_yaw))

            vehicle = world.spawn_actor(vehicle_bp, transform)
            if vehicle:
                spawned_num += 1
                print(f"🚗 成功生成第{i + 1}辆车（在视角前{5 + i * 3}米处）")
                time.sleep(1)
        except Exception as e:
            print(f"⚠️  第{i + 1}辆车生成失败：{str(e)}")
            continue

    print(f"✅ 车辆生成完成：成功 {spawned_num}/{VEHICLE_NUM} 辆")
    return spawned_num


# ========== 采集路侧数据 ==========
def get_roadside_data(world):
    """采集数据，兼容录视频场景"""
    try:
        lidar_cfg = {"range": "100m", "freq": "10Hz"}
        camera_cfg = {"resolution": "1920x1080"}

        vehicles = world.get_actors().filter("vehicle.*")
        vehicle_data = []
        for v in vehicles:
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
    """保存数据到绝对路径"""
    save_path = os.path.abspath(SAVE_DIR)
    os.makedirs(save_path, exist_ok=True)
    file_name = f"roadside_data_{data['timestamp']}.json"
    file_path = os.path.join(save_path, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ 数据已保存：{file_path}")


# ========== 主函数 ==========
def main():
    print("===== Carla 0.9.10 路侧数据采集（录视频专用） =====\n")
    # 1. 连接模拟器，获取视角位置
    client, world, spectator_transform = connect_carla()

    # 2. 在视角前生成车辆
    spawn_vehicles_in_view(world, spectator_transform)

    # 3. 调整视角稍微向下（录视频时车辆更完整）
    spectator = world.get_spectator()
    new_rotation = carla.Rotation(
        pitch=spectator_transform.rotation.pitch - 5,  # 向下5度
        yaw=spectator_transform.rotation.yaw,
        roll=spectator_transform.rotation.roll
    )
    spectator.set_transform(carla.Transform(spectator_transform.location, new_rotation))

    # 4. 等待车辆加载
    time.sleep(2)

    # 5. 采集数据
    print("🔍 正在采集路侧感知数据...")
    sensor_data = get_roadside_data(world)

    # 6. 保存数据
    save_data(sensor_data)

    # 7. 输出结果
    print(f"\n📊 采集完成！共检测到 {sensor_data['vehicle_count']} 辆车辆")
    print("\n💡 提示：现在可以开始录制Carla窗口视频，车辆就在视角前！")
    print("===== 操作结束 =====\n")


if __name__ == "__main__":
    main()