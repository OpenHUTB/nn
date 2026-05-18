#!/usr/bin/env python
# 生成会卡住不动的车辆（专门用于测试阻塞检测）
import carla
import time
import random

def main():
    try:
        # 连接 CARLA
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        print("✅ 连接成功")
        print("🚗 正在生成会卡住的车辆...")

        # 固定在同一个点生成多辆车 → 直接堵车卡死
        spawn_point = carla.Transform(
            carla.Location(x=-33.4, y=134.5, z=0.3),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )

        # 一次性在同一个位置生成 10 辆车 → 必堵死
        vehicles = []
        for i in range(10):
            bp = random.choice(blueprint_library.filter('vehicle.*'))
            vehicle = world.try_spawn_actor(bp, spawn_point)
            if vehicle:
                vehicles.append(vehicle)
                print(f"生成车辆 {vehicle.id}")

        print("\n✅ 车辆已全部卡死！现在可以录制 .rec 文件了！")
        print("▶ 运行你的录制脚本，再运行检测脚本，一定能看到结果！")

        # 保持 20 秒不动，方便录制
        time.sleep(20)

    except Exception as e:
        print(f"❌ 错误：{e}")

if __name__ == '__main__':
    main()