import sys
import os
import time


# ====================== 1. 灵活加载CARLA egg文件（移除绝对路径） ======================
def find_carla_egg():
    """
    从环境变量或默认路径查找CARLA egg文件
    优先读取环境变量CARLA_EGG_PATH，找不到则尝试默认相对路径
    """
    # 方案1：从环境变量读取（推荐）
    carla_egg_path = os.getenv("CARLA_EGG_PATH")
    if carla_egg_path and os.path.exists(carla_egg_path):
        return carla_egg_path

    # 方案2：尝试默认相对路径（如果CARLA和代码在同一目录层级）
    # 请根据你的实际目录结构调整这个相对路径
    default_egg_paths = [
        "./PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg",
        "../WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg",
        "../../WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg"
    ]

    for path in default_egg_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path

    # 方案3：提示用户输入路径
    print("⚠️  未找到CARLA egg文件，请手动输入路径")
    manual_path = input("请输入carla egg文件的完整路径：").strip()
    if os.path.exists(manual_path):
        return manual_path

    return None


# 查找并加载egg文件
carla_egg_path = find_carla_egg()
if not carla_egg_path:
    print("❌ 无法找到CARLA egg文件，请检查路径配置")
    sys.exit(1)

sys.path.append(carla_egg_path)

# 导入carla
try:
    import carla

    print("✅ 成功导入carla模块！")
except ImportError:
    print("❌ 导入失败，请确认Python版本为3.7且egg路径正确")
    sys.exit(1)

# ====================== 2. 核心配置 ======================
CARLA_HOST = "localhost"
CARLA_PORT = 2000
# 标记摄像头是否启动监听（解决警告关键）
camera_listening = False


# ====================== 3. 核心运行逻辑 ======================
def main():
    global camera_listening
    vehicle = None
    camera = None

    try:
        # 连接CARLA
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(30.0)
        world = client.get_world()
        print(f"\n✅ 成功连接CARLA！场景：{world.get_map().name}")

        # 生成红色Model3车辆
        blueprint_lib = world.get_blueprint_library()
        vehicle_bp = blueprint_lib.filter("model3")[0]
        vehicle_bp.set_attribute("color", "255,0,0")
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        print(f"✅ 生成车辆ID：{vehicle.id}（CARLA窗口可见红色车辆）")

        # 挂载摄像头并启动监听（消除警告的关键）
        camera_bp = blueprint_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_transform = carla.Transform(carla.Location(x=2.5, z=1.5))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # 给摄像头绑定空回调（启动监听，避免停止时警告）
        def empty_callback(data):
            pass

        camera.listen(empty_callback)
        camera_listening = True  # 标记已监听
        print(f"✅ 挂载摄像头ID：{camera.id}（按V切换摄像头视角截图）")

        # 控制车辆低速行驶
        print("\n📌 CARLA已实际运行！操作：")
        print("   1. 切换到CARLA窗口，可见红色车辆行驶")
        print("   2. 按V键切换摄像头视角，截图保存（论文用）")
        print("   3. 截图完成后，在PyCharm终端按Ctrl+C停止")
        vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0))

        # 保持运行（等待你截图）
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 你终止了程序，开始清理资源...")
    except Exception as e:
        print(f"\n❌ 运行出错：{str(e)}")
        # 这里也移除了绝对路径，改为提示用户自行启动CARLA
        print("⚠️  请先启动CARLA主程序（CarlaUE4.exe）")
    finally:
        # 清理资源（仅当摄像头已监听时才停止）
        if camera and camera_listening:
            camera.stop()  # 此时停止不会报警告
            camera.destroy()
            print("✅ 摄像头已清理")
        elif camera and not camera_listening:
            camera.destroy()  # 未监听则直接销毁，不执行stop
            print("✅ 摄像头已清理")

        if vehicle:
            vehicle.destroy()
            print("✅ 车辆已清理")
        print("✅ 所有资源清理完成，CARLA可正常关闭")


if __name__ == "__main__":
    main()