"""
CARLA 0.9.14 低画质版专用脚本
- 适配低画质CARLA的API差异（解决各类属性找不到问题）
- 解决着色器崩溃/异常类找不到问题
- 新增：随机地图、车辆基础避让、速度控制输出
"""
import sys
import os
import random
import carla
import cv2
import numpy as np
import queue
import traceback

# ====================== 配置区域（可根据需求修改）======================
# CARLA根目录
CARLA_ROOT = 'D:/123/apps/CARLA_0.9.14/WindowsNoEditor'
# 服务器连接配置
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
CARLA_TIMEOUT = 60.0  # 低画质启动慢，延长超时
# 支持的低画质小地图列表（CARLA 0.9.14）
# 注：低画质版建议只使用小地图Town01/Town02，避免卡顿
SUPPORTED_MAPS = ['Town01', 'Town02']
# 车辆配置
PREFERRED_VEHICLE = 'vehicle.seat.leon'  # 轻量化车型
# 摄像头配置（低画质参数）
CAMERA_RESOLUTION = (640, 360)  # 降低分辨率
CAMERA_FOV = 80
CAMERA_SENSOR_TICK = 0.1  # 10fps减少压力
CAMERA_POSITION = carla.Transform(carla.Location(x=1.5, z=1.8))  # 摄像头挂载位置
# 窗口配置
WINDOW_NAME = 'CARLA Low-Quality View'
WINDOW_SIZE = (640, 360)
# 速度显示配置（低画质版无法通过API限制速度，仅显示实际速度）
SPEED_PRINT_INTERVAL = 10  # 每10帧打印一次速度（减少刷屏）
# =====================================================================

# 全局变量
IMAGE_QUEUE = queue.Queue(maxsize=1)

def check_carla_running():
    """检查CARLA进程（CarlaUE4.exe）是否运行"""
    try:
        import psutil
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == 'CarlaUE4.exe':
                return True
        return False
    except ImportError:
        print("⚠️ 未安装psutil，无法检查CARLA进程，默认认为进程已运行")
        return True

def image_callback(image):
    """摄像头回调函数（低画质适配：降低分辨率减少压力）"""
    try:
        # 将原始数据转换为BGRA格式的numpy数组
        img_bgra = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_bgra = img_bgra.reshape((image.height, image.width, 4))
        # 转换为BGR格式（适配OpenCV）
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
        # 低画质优化：缩小图像尺寸（减少CV窗口渲染压力）
        img_bgr = cv2.resize(img_bgr, WINDOW_SIZE)
        # 保证队列中只有最新的一帧图像，避免堆积
        with IMAGE_QUEUE.mutex:
            IMAGE_QUEUE.queue.clear()
        IMAGE_QUEUE.put(img_bgr, timeout=0.1)
    except Exception as e:
        print(f"⚠️ 图像回调出错：{e}")

def load_random_map(client):
    """随机加载支持的低画质小地图（适配0.9.14）"""
    random_map = random.choice(SUPPORTED_MAPS)
    try:
        world = client.load_world(random_map)
        world.wait_for_tick()
        print(f"✅ 随机加载地图成功：{random_map}（当前地图路径：{world.get_map().name}）")
        return world
    except Exception as e:
        print(f"❌ 加载地图{random_map}失败，使用默认地图Town01：{e}")
        world = client.load_world('Town01')
        world.wait_for_tick()
        return world

def spawn_vehicle(world, blueprint_library, spawn_points):
    """生成车辆并启用基础自动驾驶（低画质版默认支持避让，无额外配置）"""
    # 筛选轻量化车型（减少低画质版性能压力）
    vehicle_bps = blueprint_library.filter(PREFERRED_VEHICLE)
    if not vehicle_bps:
        print(f"⚠️ 未找到{PREFERRED_VEHICLE}，使用默认轻量化车型")
        vehicle_bps = blueprint_library.filter('vehicle.*')[:1]
    vehicle_bp = vehicle_bps[0]
    vehicle_bp.set_attribute('role_name', 'autopilot')

    # 尝试生成车辆（多生成点重试，解决冲突问题）
    max_retry = 5
    retry_count = 0
    vehicle = None
    while retry_count < max_retry and vehicle is None:
        try:
            # 随机选择生成点（增加多样性，降低冲突概率）
            spawn_point = random.choice(spawn_points)
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        except Exception as e:  # 低画质版无专用ActorSpawnException，捕获通用异常
            retry_count += 1
            print(f"⚠️ 生成点被占用，重试第{retry_count}次...：{e}")
    if vehicle is None:
        print("❌ 多次重试后仍无法生成车辆")
        return None

    # 启用基础自动驾驶（低画质版此操作已包含交通避让逻辑）
    vehicle.set_autopilot(True)
    print(f"✅ 生成车辆：{vehicle.type_id}（已启用自动驾驶+基础避让）")
    return vehicle

def spawn_camera(world, blueprint_library, vehicle):
    """挂载摄像头（低画质参数配置，适配0.9.14 API）"""
    try:
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        # 设置摄像头低画质参数（确保与0.9.14 API兼容）
        camera_bp.set_attribute('image_size_x', str(CAMERA_RESOLUTION[0]))
        camera_bp.set_attribute('image_size_y', str(CAMERA_RESOLUTION[1]))
        camera_bp.set_attribute('fov', str(CAMERA_FOV))
        camera_bp.set_attribute('sensor_tick', str(CAMERA_SENSOR_TICK))
        # 生成摄像头并挂载到车辆
        camera = world.spawn_actor(camera_bp, CAMERA_POSITION, attach_to=vehicle)
        # 注册回调函数
        camera.listen(image_callback)
        print("✅ 摄像头挂载成功")
        return camera
    except Exception as e:
        print(f"❌ 摄像头挂载失败：{e}")
        return None

def get_vehicle_speed(vehicle):
    """获取车辆当前速度（km/h，适配CARLA 0.9.14）"""
    velocity = vehicle.get_velocity()
    # 计算速度：√(x² + y² + z²) （m/s），转换为km/h（×3.6）
    speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    speed_kmh = speed_ms * 3.6
    return round(speed_kmh, 1)

def main():
    """主函数：连接CARLA、随机加载地图、生成车辆、显示画面、输出速度"""
    # 初始化资源
    client = None
    world = None
    vehicle = None
    camera = None

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
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(CARLA_TIMEOUT)
        print("✅ 连接CARLA服务器成功")

        # 2. 随机加载低画质地图
        world = load_random_map(client)

        # 3. 获取蓝图和生成点
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("❌ 无可用生成点，退出")
            return

        # 4. 生成车辆（带基础避让功能）
        vehicle = spawn_vehicle(world, blueprint_library, spawn_points)
        if vehicle is None:
            return

        # 5. 挂载摄像头
        camera = spawn_camera(world, blueprint_library, vehicle)
        if camera is None:
            return

        # 6. 显示画面+输出速度
        print("\n📌 操作说明：")
        print(f"   - 按 'q' 退出程序")
        print(f"   - 车辆已启用自动驾驶+基础避让功能")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *WINDOW_SIZE)

        frame_count = 0
        while True:
            # 显示摄像头画面
            if not IMAGE_QUEUE.empty():
                img = IMAGE_QUEUE.get(timeout=0.5)
                cv2.imshow(WINDOW_NAME, img)
            # 实时输出车辆速度（间隔打印，避免刷屏）
            frame_count += 1
            if frame_count % SPEED_PRINT_INTERVAL == 0 and vehicle is not None:
                current_speed = get_vehicle_speed(vehicle)
                print(f"\r当前车辆速度：{current_speed} km/h", end="")
                frame_count = 0
            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n📌 收到退出指令，正在清理资源...")
                break

    # 异常处理（完全适配低画质版CARLA 0.9.14，无专用异常类）
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "timeout" in error_msg:
            print("\n❌ 连接失败！")
            print(f"   解决：1. 确认CarlaUE4.exe已启动 2. 关闭防火墙 3. 检查端口{CARLA_PORT}")
        elif "spawn" in error_msg:
            print("\n❌ 车辆生成失败（无可用生成点）！")
            print("   解决：重启CARLA或更换生成点")
        else:
            print(f"\n❌ 运行时错误：{e}")
    except AttributeError as e:
        print(f"\n❌ API属性错误：{e}")
        print("   解决：1. 确认使用的是CARLA 0.9.14低画质版whl包 2. 重启CARLA和脚本")
    except Exception as e:
        print(f"\n❌ 未知错误：{e}")
        traceback.print_exc()

    # 清理资源（确保低画质版资源正常释放）
    finally:
        print("\n--- [清理资源] ---")
        if camera:
            try:
                camera.stop()
                camera.destroy()
                print("✅ 销毁摄像头")
            except Exception as e:
                print(f"⚠️ 销毁摄像头失败：{e}")
        if vehicle:
            try:
                vehicle.destroy()
                print("✅ 销毁车辆")
            except Exception as e:
                print(f"⚠️ 销毁车辆失败：{e}")
        cv2.destroyAllWindows()
        print("✅ 程序结束")

if __name__ == '__main__':
    main()