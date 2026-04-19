# main.py
"""无人机飞行控制主程序

这是无人机飞行控制程序的入口文件。
程序实现以下功能：
1. 连接 AirSim 仿真器
2. 控制无人机起飞
3. 按照预设航点飞行
4. 安全降落或紧急降落
"""

# 导入 time 模块，用于延时操作
import time

# 从 drone_controller 模块导入无人机控制器
from drone_controller import DroneController

# 从 flight_path 模块导入航点规划类
from flight_path import FlightPath

# 从 utils 模块导入分隔线打印函数
from utils import print_separator


def main():
    """主函数：控制无人机执行完整飞行任务

    流程包括：
    1. 初始化无人机控制器
    2. 执行起飞
    3. 按航点依次飞行
    4. 执行降落
    5. 清理资源

    包含异常处理和键盘中断处理。
    """
    # 打印顶部分隔线
    print_separator()
    # 打印程序启动信息
    print("🚁 AirSim 无人机控制程序启动")
    # 打印分隔线
    print_separator()

    # 初始化无人机控制器对象
    drone = None
    try:
        # 创建无人机控制器实例
        drone = DroneController()

        # ===== 起飞阶段 =====
        # 执行起飞操作
        if not drone.takeoff():
            # 起飞失败，打印错误信息并退出
            print("❌ 起飞失败")
            return

        # 起飞后等待 1 秒稳定
        time.sleep(1)

        # ===== 路径规划阶段 =====
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

        # ===== 相机设置阶段 =====
        # 设置图片保存目录（可选，默认保存在 drone_images 文件夹）
        # drone.set_output_dir("my_drone_photos")

        # 打印飞行路径信息
        FlightPath.print_path(waypoints)

        # ===== 执行飞行任务阶段 =====
        # 标记是否发生碰撞
        collision_occurred = False
        # 遍历所有航点，依次飞向每个点
        for i, (x, y, z) in enumerate(waypoints, 1):
            # 打印当前飞行段的分隔线和编号
            print(f"\n{'=' * 40}")
            print(f"第 {i} 段飞行")
            print(f"{'=' * 40}")

            # 飞向当前航点，速度 3 m/s
            if not drone.fly_to_position(x, y, z, velocity=3):
                # 飞行失败（可能因碰撞中断）
                print("⚠️  任务因碰撞而中断")
                collision_occurred = True
                # 跳出飞行循环
                break

            # ===== 到达航点后拍照 =====
            print(f"\n📷 航点 {i} 拍照...")
            # 拍照并保存 RGB 图像
            drone.capture_image()
            # 也可以保存其他类型图像（取消注释使用）：
            # drone.capture_depth_image()
            # drone.capture_segmentation_image()
            # 同时保存所有类型图像：
            # drone.capture_all_cameras()

            # 到达当前航点后等待 1 秒
            time.sleep(1)

        # ===== 降落阶段 =====
        # 打印分隔线
        print_separator()
        # 根据是否发生碰撞选择降落方式
        if collision_occurred:
            # 碰撞后执行紧急降落程序
            print("⚠️  碰撞后执行紧急降落程序")
        else:
            # 正常完成任务，执行正常降落
            print("✅ 任务完成，执行正常降落")
        print_separator()

        # 执行安全降落
        if not drone.safe_land():
            # 安全降落失败，执行紧急停止
            drone.emergency_stop()

    except KeyboardInterrupt:
        """捕获键盘中断（Ctrl+C）

        当用户按下 Ctrl+C 时执行此代码块，
        确保无人机安全停止。
        """
        # 打印用户中断提示
        print("\n⚠️  用户中断程序")
        # 如果无人机对象已创建，执行紧急停止
        if drone:
            drone.emergency_stop()

    except Exception as e:
        """捕获所有其他异常

        当程序发生错误时执行此代码块，
        打印错误信息并尝试安全停止无人机。
        """
        # 打印异常信息
        print(f"❌ 发生异常: {e}")
        # 导入 traceback 模块用于打印详细错误信息
        import traceback

        # 打印异常的完整调用栈信息
        traceback.print_exc()
        # 如果无人机对象已创建，执行紧急停止
        if drone:
            drone.emergency_stop()

    finally:
        """无论是否发生异常都会执行的代码块

        用于确保资源被正确清理。
        """
        # 如果无人机对象已创建，执行清理操作
        if drone:
            drone.cleanup()

    # 打印程序结束信息
    print("\n🏁 程序结束")


# 程序入口点
# 只有直接运行此脚本时才会执行 main() 函数
# 被其他模块导入时不会自动执行
if __name__ == "__main__":
    main()
