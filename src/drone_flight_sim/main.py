"""无人机飞行控制主程序

这是无人机飞行控制程序的入口文件。
1. 自动航点飞行模式 - 无人机按预设航点自动飞行，支持碰撞自动恢复和手动接管
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
    发生碰撞时自动尝试恢复，失败后请求手动接管。

    参数:
        drone: DroneController 实例

    返回:
        bool: 任务正常完成返回 True，需要手动接管返回 False
    """
    print("\n🚀 进入自动航点飞行模式")
    print_separator()

    # 起飞
    if not drone.takeoff():
        print("❌ 起飞失败")
        return False

    time.sleep(1)

    # ===== 定义大范围飞行航点列表 =====
    # 注意：AirSim 中 Z 轴向下为正，所以负值表示向上飞行
    # 飞行高度设为 -5 米，起飞高度也设为 -5 米
    waypoints = [
        (20, 0, -5),  # 航点1：向右飞行 10 米
        (10, -20, -5),  # 航点2：向前飞行 10 米
        (0, -10, -5),  # 航点3：向左飞行 10 米
        (0, 0, -5),  # 航点4：向后飞行 10 米，回到原点
        (-20, 0, -5),  # 航点5：继续向左飞行
        (-10, -10, -5),  # 航点6：继续向前飞行
        (10, -10, -5),  # 航点7：回到右侧
        (10, 10, -5),  # 航点8：向上飞行
        (-10, 10, -5),  # 航点9：向左飞行
        (-10, 0, -5),  # 航点10：向后飞行
        (0, 0, -5),  # 航点11：回到原点
    ]

    # 打印飞行路径信息
    FlightPath.print_path(waypoints)

    # ===== 执行飞行任务阶段 =====
    manual_takeover = False

    for i, (x, y, z) in enumerate(waypoints, 1):
        print(f"\n{'=' * 40}")
        print(f"第 {i} 段飞行 -> 目标: ({x}, {y}, {z})")
        print(f"{'=' * 40}")

        # 飞向当前航点，速度 3 m/s
        success = drone.fly_to_position(x, y, z, velocity=3)

        if not success:
            # 发生碰撞，尝试自动恢复
            print("\n⚠️  检测到碰撞，开始自动恢复...")

            # 最多尝试3次自动恢复
            recovery_success = False
            for attempt in range(3):
                if drone.collision_handler.auto_recover():
                    recovery_success = True
                    print("✅ 自动恢复成功，继续任务")
                    break
                else:
                    if attempt < 2:
                        print(f"⚠️  第 {attempt + 1} 次恢复失败，重试...")

            if not recovery_success:
                # 自动恢复全部失败，请求手动接管
                drone.collision_handler.request_manual_control()
                manual_takeover = True
                break

            # 重新尝试飞向当前航点
            print(f"\n重新飞向航点 {i}...")
            if not drone.fly_to_position(x, y, z, velocity=3):
                # 再次失败，再次尝试自动恢复
                if not drone.collision_handler.auto_recover():
                    drone.collision_handler.request_manual_control()
                    manual_takeover = True
                    break

        # 到达航点后拍照
        print(f"\n📷 航点 {i} 拍照...")
        drone.capture_image()

        time.sleep(1)

    # 降落阶段
    print_separator()
    if manual_takeover:
        print("⚠️  进入手动接管模式")
        print_separator()
        # 进入键盘控制模式，让用户手动解决
        from keyboard_control import KeyboardController, print_control_help

        print_control_help()
        keyboard_controller = KeyboardController(drone)
        print("🕹️ 请手动控制无人机脱离困境后按 L 降落")
        keyboard_controller.start()
    elif drone.collision_handler.collision_count > 0:
        print(
            f"⚠️  任务完成（共发生 {drone.collision_handler.collision_count} 次碰撞），执行降落"
        )
        drone.safe_land()
    else:
        print("✅ 任务完成，执行正常降落")
        drone.safe_land()
    print_separator()

    return not manual_takeover


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
