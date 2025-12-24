import sys
import os
import time

# ====================== 1. 相对路径配置（核心：移除绝对路径） ======================
# 方法：将CARLA的egg文件放到项目根目录的「carla_lib」文件夹下
# 你需要手动执行：把 D:\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg
# 复制到 当前项目根目录/carla_lib/ 文件夹中
CARLA_LIB_DIR = os.path.join(os.path.dirname(__file__), "carla_lib")  # 项目内相对路径
carla_egg_files = [f for f in os.listdir(CARLA_LIB_DIR) if f.endswith(".egg") and "0.9.10" in f]

if not carla_egg_files:
    print(f"❌ 在 {CARLA_LIB_DIR} 未找到CARLA 0.9.10的egg文件！")
    print("⚠️  请将carla-0.9.10-py3.7-win-amd64.egg复制到项目的carla_lib文件夹")
    sys.exit(1)

# 加载egg文件（自动匹配文件夹内的egg）
carla_egg_path = os.path.join(CARLA_LIB_DIR, carla_egg_files[0])
sys.path.append(carla_egg_path)
print(f"✅ 已加载CARLA egg文件：{carla_egg_path}")

# 导入carla
try:
    import carla

    print("✅ 成功导入carla模块！")
except ImportError:
    print("❌ 导入失败，请确认：1. egg文件版本为0.9.10  2. Python版本为3.7")
    sys.exit(1)

# ====================== 2. 核心配置（无硬编码路径） ======================
CARLA_HOST = "localhost"
CARLA_PORT = 2000
camera_listening = False  # 标记摄像头监听状态


# ====================== 3. 核心运行逻辑（main函数作为入口） ======================
def main():
    global camera_listening
    vehicle = None
    camera = None

    try:
        # 连接CARLA服务器
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(30.0)
        world = client.get_world()
        print(f"\n✅ 成功连接CARLA！当前场景：{world.get_map().name}")

        # 生成红色Model3车辆
        blueprint_lib = world.get_blueprint_library()
        vehicle_bp = blueprint_lib.filter("model3")[0]
        vehicle_bp.set_attribute("color", "255,0,0")  # 红色车辆
        spawn_points = world.get_map().get_spawn_points()

        if not spawn_points:
            print("❌ 未找到车辆生成点，请确认CARLA场景已加载完成")
            sys.exit(1)

        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        print(f"✅ 生成车辆ID：{vehicle.id}（CARLA窗口可见红色车辆）")

        # 挂载摄像头并启动监听（消除警告）
        camera_bp = blueprint_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_transform = carla.Transform(carla.Location(x=2.5, z=1.5))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # 空回调函数（启动监听）
        def empty_callback(data):
            pass

        camera.listen(empty_callback)
        camera_listening = True
        print(f"✅ 挂载摄像头ID：{camera.id}（按V键切换摄像头视角截图）")

        # 控制车辆低速行驶
        print("\n📌 CARLA已实际运行！操作指引：")
        print("   1. 切换到CARLA窗口，可见红色车辆低速行驶")
        print("   2. 按V键切换到摄像头视角，截图保存（论文用）")
        print("   3. 截图完成后，在终端按 Ctrl+C 停止程序")
        vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0))

        # 保持运行（等待用户截图）
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 用户终止程序，开始清理资源...")
    except Exception as e:
        print(f"\n❌ 运行出错：{str(e)}")
        print("⚠️  请先启动CARLA服务器（CarlaUE4.exe）后再运行本脚本")
    finally:
        # 安全清理资源
        if camera:
            if camera_listening:
                camera.stop()
            camera.destroy()
            print("✅ 摄像头资源已清理")

        if vehicle:
            vehicle.destroy()
            print("✅ 车辆资源已清理")

        print("✅ 所有资源清理完成，程序正常退出")


# ====================== 4. 规范入口（仅当作为主脚本运行时执行） ======================
if __name__ == "__main__":
    # 检查carla_lib文件夹是否存在
    if not os.path.exists(CARLA_LIB_DIR):
        os.makedirs(CARLA_LIB_DIR)
        print(f"⚠️  已自动创建carla_lib文件夹：{CARLA_LIB_DIR}")
        print("请将CARLA 0.9.10的egg文件复制到该文件夹后重新运行！")
        sys.exit(1)

    main()