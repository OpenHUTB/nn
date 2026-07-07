"""
简单的CARLA车辆演示
生成一辆车并显示基本信息
"""
import carla
import sys
import time


def main():
    print("=" * 50)
    print("CARLA 简单车辆演示")
    print("=" * 50)

    try:
        # 连接到CARLA服务器
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        print("[OK] 已连接到CARLA服务器")

        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # 选择车辆蓝图
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("color", "255, 0, 0")  # 红色
        print("[OK] 已选择红色特斯拉")

        # 获取生成点
        spawn_points = world.get_map().get_spawn_points()

        # 尝试生成车辆
        vehicle = None
        for i, spawn_point in enumerate(spawn_points[:10]):
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                print(f"[OK] 车辆已生成！位置: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
                break
            except RuntimeError as e:
                if "collision" in str(e).lower():
                    continue
                raise

        if vehicle is None:
            print("[ERROR] 无法生成车辆")
            return

        # 启动自动驾驶
        vehicle.set_autopilot(True)
        print("[OK] 自动驾驶已开启\n")

        print("按 Ctrl+C 停止程序\n")

        try:
            # 循环显示车辆信息
            while True:
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                speed = ((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5) * 3.6

                print(f"\r位置: ({location.x:6.1f}, {location.y:6.1f}) | 速度: {speed:5.1f} km/h", end="")
                time.sleep(0.2)

        except KeyboardInterrupt:
            print("\n\n[INFO] 程序已停止")

        finally:
            print("[INFO] 清理中...")
            vehicle.destroy()
            print("[OK] 车辆已销毁")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("请确保CARLA服务器正在运行！")
        sys.exit(1)


if __name__ == "__main__":
    main()
