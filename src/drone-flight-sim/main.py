<<<<<<< HEAD
# main.py
"""无人机飞行主程序"""

import airsim
import time
from drone_controller import DroneController
from flight_path import FlightPath
from utils import print_separator


def main():
    print_separator()
    print("🚁 AirSim 无人机控制程序启动")
    print_separator()

    drone = None
    try:
        # 初始化无人机
        drone = DroneController()

        # 起飞
        if not drone.takeoff():
            print("❌ 起飞失败")
            return

        time.sleep(1)

        # 定义飞行路径（可根据需要修改）
        waypoints = [
            (5, 0, -3),  # 点1
            (5, -5, -3),  # 点2
            (0, -5, -3),  # 点3
            (0, 0, -3),  # 点4
        ]

        # 或者使用预设路径：
        # waypoints = FlightPath.square_path(size=15, height=-3)
        # waypoints = FlightPath.rectangle_path(width=20, height=10, altitude=-3)

        FlightPath.print_path(waypoints)

        # 执行飞行任务
        collision_occurred = False
        for i, (x, y, z) in enumerate(waypoints, 1):
            print(f"\n{'=' * 40}")
            print(f"第 {i} 段飞行")
            print(f"{'=' * 40}")

            if not drone.fly_to_position(x, y, z, velocity=3):
                print("⚠️  任务因碰撞而中断")
                collision_occurred = True
                break

            time.sleep(1)

        # 降落
        print_separator()
        if collision_occurred:
            print("⚠️  碰撞后执行紧急降落程序")
        else:
            print("✅ 任务完成，执行正常降落")
        print_separator()

        if not drone.safe_land():
            drone.emergency_stop()

    except KeyboardInterrupt:
        print("\n⚠️  用户中断程序")
        if drone:
            drone.emergency_stop()
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        import traceback

        traceback.print_exc()
        if drone:
            drone.emergency_stop()
    finally:
        if drone:
            drone.cleanup()

    print("\n🏁 程序结束")


if __name__ == "__main__":
    main()
