"""
CARLA 0.9.14 低画质版专用脚本
- 适配低画质CARLA的API差异
- 解决着色器崩溃/异常类找不到问题
"""
import sys
import os
import carla
import cv2
import numpy as np
import queue

# 全局变量
IMAGE_QUEUE = queue.Queue(maxsize=1)
# 替换为你的低画质CARLA实际路径
CARLA_ROOT = 'D:/123/apps/CARLA_0.9.14/WindowsNoEditor'

# 摄像头回调函数（低画质适配：降低分辨率减少压力）
def image_callback(image):
    try:
        img_bgra = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_bgra = img_bgra.reshape((image.height, image.width, 4))
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

        # 低画质优化：缩小图像尺寸（减少CV窗口渲染压力）
        img_bgr = cv2.resize(img_bgr, (640, 360))

        if IMAGE_QUEUE.full():
            IMAGE_QUEUE.get_nowait()
        IMAGE_QUEUE.put(img_bgr, timeout=0.1)
    except Exception as e:
        print(f"⚠️ 图像回调出错：{e}")

def main():
    camera = None
    vehicle = None

    # 检查CARLA进程是否运行
    def check_carla_running():
        import psutil
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == 'CarlaUE4.exe':
                return True
        return False

    # 前置检查
    print("=" * 60)
    print("--- [低画质CARLA环境检查] ---")
    if not check_carla_running():
        print("❌ 错误：未检测到CarlaUE4.exe进程！")
        print(f"   请先启动：{os.path.join(CARLA_ROOT, 'CarlaUE4.exe')}")
        print("   （建议使用低画质快捷方式启动）")
        return
    print("✅ 检测到CARLA服务器运行")
    print("--- [环境检查完成] ---")
    print("=" * 60)

    try:
        # 1. 连接CARLA服务器（低画质版超时延长）
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(60.0)  # 低画质启动慢，延长超时
        world = client.load_world('Town01')  # 低画质优先用小地图Town01
        world.wait_for_tick()
        print(f"✅ 连接成功！当前地图：{world.get_map().name}")

        # 2. 获取蓝图和生成点
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("❌ 无可用生成点，退出")
            return

        # 3. 生成车辆（低画质选轻量化车型）
        vehicle_bps = blueprint_library.filter('vehicle.seat.leon')  # 轻量化车型
        if not vehicle_bps:
            vehicle_bps = blueprint_library.filter('vehicle.*')[0:1]
        vehicle_bp = vehicle_bps[0]
        vehicle_bp.set_attribute('role_name', 'autopilot')

        # 换生成点避免占用（低画质版生成点易冲突）
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[10])
        vehicle.set_autopilot(True)
        print(f"✅ 生成车辆：{vehicle.type_id}")

        # 4. 挂载摄像头（低画质参数）
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')   # 降低分辨率
        camera_bp.set_attribute('image_size_y', '360')
        camera_bp.set_attribute('fov', '80')
        camera_bp.set_attribute('sensor_tick', '0.1')    # 10fps减少压力
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.8))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        camera.listen(image_callback)
        print("✅ 摄像头挂载成功")

        # 5. 显示画面
        print("\n📌 按 'q' 退出 | 低画质模式已启用")
        cv2.namedWindow('CARLA Low-Quality View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CARLA Low-Quality View', 640, 360)

        while True:
            if not IMAGE_QUEUE.empty():
                img = IMAGE_QUEUE.get(timeout=0.5)
                cv2.imshow('CARLA Low-Quality View', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 修复：低画质版CARLA异常类直接在carla模块下（无exceptions子模块）
    except carla.CarlaConnectionError:
        print("\n❌ 连接失败！")
        print("   解决：1. 确认CarlaUE4.exe已启动 2. 关闭防火墙 3. 检查端口2000")
    except carla.ActorSpawnException:
        print("\n❌ 车辆生成失败！")
        print("   解决：换生成点（如spawn_points[20]）或重启CARLA")
    except AttributeError as e:
        print(f"\n❌ API属性错误：{e}")
        print("   解决：重新安装对应版本的whl包（低画质版CARLA需匹配whl）")
    except Exception as e:
        print(f"\n❌ 未知错误：{e}")
        import traceback
        traceback.print_exc()

    # 清理资源
    finally:
        print("\n--- [清理资源] ---")
        if camera:
            camera.stop()
            camera.destroy()
            print("✅ 销毁摄像头")
        if vehicle:
            vehicle.destroy()
            print("✅ 销毁车辆")
        cv2.destroyAllWindows()
        print("✅ 程序结束")

if __name__ == '__main__':
    # 低画质版需额外导入psutil检查进程（可选）
    try:
        import psutil
    except ImportError:
        print("⚠️ 未安装psutil，跳过CARLA进程检查")
        # 注释掉进程检查相关代码
        def check_carla_running():
            return True
    main()
