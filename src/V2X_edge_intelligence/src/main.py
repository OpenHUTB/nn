#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路侧感知数据集预处理（Carla 0.9.10终极适配版）
运行前：先启动D:\WindowsNoEditor\CarlaUE4.exe
"""
import sys
import os
import time
import json
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
TIMEOUT = 10.0
SAVE_DIR = "carla_sensor_data"

# ========== 连接模拟器 ==========
def connect_carla() -> carla.World:
    """连接Carla 0.9.10模拟器"""
    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(TIMEOUT)
        world = client.get_world()
        print(f"✅ 成功连接Carla模拟器：{CARLA_HOST}:{CARLA_PORT}")
        return world
    except Exception as e:
        print(f"❌ 连接失败：{str(e)}")
        sys.exit(1)

# ========== 获取路侧数据（完全适配0.9.10） ==========
def get_roadside_data(world: carla.World) -> Dict[str, Any]:
    """获取路侧感知数据（避开所有新版API）"""
    blueprint_lib = world.get_blueprint_library()

    # 1. 激光雷达配置（仅设置参数，不获取返回值，避免API冲突）
    lidar_bp = blueprint_lib.find("sensor.lidar.ray_cast")
    # 0.9.10仅支持基础参数，且无需获取返回值
    lidar_bp.set_attribute("range", "100")
    lidar_bp.set_attribute("rotation_frequency", "10")

    # 2. 摄像头配置（同样仅设置，不获取）
    camera_bp = blueprint_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "1920")
    camera_bp.set_attribute("image_size_y", "1080")

    # 3. 车辆检测（0.9.10核心API兼容）
    vehicles = world.get_actors().filter("vehicle.*")
    vehicle_list = []
    for v in vehicles:
        trans = v.get_transform()
        vehicle_list.append({
            "id": v.id,
            "model": v.type_id,
            "x": float(trans.location.x),
            "y": float(trans.location.y),
            "z": float(trans.location.z),
            "yaw": float(trans.rotation.yaw)
        })

    # 4. 整合数据（不依赖传感器属性获取，避免API错误）
    return {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "roadside_id": "RSU_001",
        "lidar_config": {
            "range": "100m",
            "rotation_frequency": "10Hz"
        },
        "camera_config": {
            "resolution": "1920x1080"
        },
        "detected_vehicles": vehicle_list,
        "vehicle_count": len(vehicle_list)
    }

# ========== 保存数据 ==========
def save_data(data: Dict[str, Any]) -> None:
    """保存数据到JSON文件"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    file_name = f"roadside_data_{data['timestamp']}.json"
    file_path = os.path.join(SAVE_DIR, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ 数据已保存：{file_path}")

# ========== 主函数 ==========
def main():
    print("===== Carla 0.9.10 路侧数据采集 =====\n")
    world = connect_carla()
    print("🔍 正在采集路侧感知数据...")
    sensor_data = get_roadside_data(world)
    save_data(sensor_data)
    print(f"\n📊 采集完成！共检测到 {sensor_data['vehicle_count']} 辆车辆")
    print("\n===== 操作结束 =====\n")

if __name__ == "__main__":
    main()