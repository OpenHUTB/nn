"""无人机飞行控制主程序

这是无人机飞行控制程序的入口文件。
1. 自动航点飞行模式 - 无人机按预设航点自动飞行
2. 键盘手动控制模式 - 使用键盘手动控制无人机
"""

# 导入 time 模块，用于延时操作
import time

# 从 drone_controller 模块导入无人机控制器
from drone_controller import DroneController

# 从 flight_path 模块导入航点规划类
from flight_path import FlightPath

# 从 utils 模块导入分隔线打印函数
from utils import print_separator


def auto_flight_mode(drone):
    """自动航点飞行模式

    无人机按照预设的航点列表自动飞行，并在每个航点拍照。

    参数:
        drone: DroneController 实例
    """
    print("\n🚀 进入自动航点飞行模式")
    print_separator()

    # 起飞
    if not drone.takeoff():
        print("❌ 起飞失败")
        return False

    time.sleep(1)

    # 定义飞行航点列表，每个航点是 (x, y, z) 坐标元组
    # 注意：AirSim 中 Z 轴向下为正，所以负值表示向上飞行
    waypoints = [
        (5, 0, -3),  # 航点1：向右飞行 5 米
        (5, -5, -3),  # 航点2：向前飞行 5 米
        (0, -5, -3),  # 航点3：向左飞行 5 米
        (0, 0, -3),  # 航点4：向后飞行 5 米，回到原点
    ]

    # ===== 使用预设路径的示例代码（可替换上方 waypoints）=====
    # 生成正方形路径：边长 15 米，高度 -3 米
    # waypoints = FlightPath.square_path(size=15, height=-3)
    # 生成矩形路径：宽 20 米，长 10 米，高度 -3 米
    # waypoints = FlightPath.rectangle_path(width=20, length=10, altitude=-3)

    # 打印飞行路径信息
    FlightPath.print_path(waypoints)

    # ===== 执行飞行任务阶段 =====
    collision_occurred = False

    for i, (x, y, z) in enumerate(waypoints, 1):
        print(f"\n{'=' * 40}")
        print(f"第 {i} 段飞行")
        print(f"{'=' * 40}")

        # 飞向当前航点，速度 3 m/s
        if not drone.fly_to_position(x, y, z, velocity=3):
            print("⚠️  任务因碰撞而中断")
            collision_occurred = True
            break

        # 到达航点后拍照
        print(f"\n📷 航点 {i} 拍照...")
        drone.capture_image()

        time.sleep(1)

    # 降落阶段
    print_separator()
    if collision_occurred:
        print("⚠️  碰撞后执行紧急降落程序")
    else:
        print("✅ 任务完成，执行正常降落")
    print_separator()

    if not drone.safe_land():
        drone.emergency_stop()

    return True


def keyboard_control_mode(drone):
    """键盘手动控制模式

    启动键盘监听，允许用户手动控制无人机飞行。

    参数:
        drone: DroneController 实例
    """
    print("\n🎮 进入键盘手动控制模式")
    print_separator()

    # 起飞
    if not drone.takeoff():
        print("❌ 起飞失败")
        return False

    time.sleep(1)

    # 导入键盘控制模块
    from keyboard_control import KeyboardController, print_control_help

    # 打印控制说明
    print_control_help()

    # 创建键盘控制器
    keyboard_controller = KeyboardController(drone)

    # 启动键盘监听
    print("🕹️ 键盘控制已启动，开始控制无人机吧！")
    print("📌 按 ESC 或 L 键退出键盘控制模式\n")

    keyboard_controller.start()

    # 退出后执行降落
    print("\n🛬 键盘控制结束，开始降落...")
    drone.safe_land()

    return True


def main():
    """主函数，程序入口"""
    # 创建无人机控制器实例
    drone = DroneController()

    try:
        # 选择飞行模式
        print("\n请选择飞行模式：")
        print("1 - 自动航点飞行模式")
        print("2 - 键盘手动控制模式")
        choice = input("请输入模式编号：")

        if choice == "1":
            auto_flight_mode(drone)
        elif choice == "2":
            keyboard_control_mode(drone)
        else:
            print("❌ 无效输入，请输入 1 或 2！")

    except KeyboardInterrupt:
        print("\n\n⚠️  检测到中断信号，正在安全降落...")
        drone.safe_land()
    except Exception as e:
        print(f"\n❌ 程序异常：{e}")
        drone.emergency_stop()
    finally:
        # 无论如何都执行资源清理
        drone.cleanup()
        print("\n👋 程序已退出")


if __name__ == "__main__":
    main()
