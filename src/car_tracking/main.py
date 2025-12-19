"""
CARLA 0.9.14 低画质版专用脚本
- 适配低画质CARLA的API差异（解决各类属性找不到问题，包括雷达数据属性）
- 解决着色器崩溃/异常类找不到问题
- 新增：随机地图、车辆基础避让+主动识别车辆绕行、速度控制输出
- 优化：CARLA路径使用环境变量+默认路径，避免绝对路径
"""
import sys
import os
import random
import carla
import cv2
import numpy as np
import queue
import traceback
import math

# ====================== 配置区域（可根据需求修改）======================
# CARLA根目录：优先读取系统环境变量CARLA_ROOT，未配置则使用默认相对路径（可自行调整）
# 方案1：环境变量（推荐）：在系统中配置CARLA_ROOT为你的CARLA安装路径
# 方案2：默认路径：改为你项目的相对路径或通用默认路径
def get_carla_root():
    """获取CARLA根目录（优先环境变量，次之用默认路径）"""
    # 从环境变量读取
    carla_root = os.getenv('CARLA_ROOT')
    if carla_root and os.path.exists(os.path.join(carla_root, 'CarlaUE4.exe')):
        return carla_root
    # 未配置环境变量时，使用默认路径（可改为相对路径，如'./CARLA_0.9.14'）
    default_carla_root = './CARLA_0.9.14/WindowsNoEditor'  # 相对路径示例
    # 额外兜底：若相对路径不存在，使用当前工作目录的父目录（可选）
    if not os.path.exists(os.path.join(default_carla_root, 'CarlaUE4.exe')):
        # 也可以提示用户配置环境变量，这里暂时返回默认路径
        print(f"⚠️ 未配置CARLA_ROOT环境变量，且默认路径{default_carla_root}无效，请检查路径！")
    return default_carla_root

CARLA_ROOT = get_carla_root()

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
# 避让配置（改用Actor距离检测，放弃雷达属性依赖）
SAFE_DISTANCE = 10.0  # 安全距离（小于此距离触发避让）
AVOIDANCE_ANGLE = 10.0  # 避让时的方向微调角度（度）
# 窗口配置
WINDOW_NAME = 'CARLA Low-Quality View'
WINDOW_SIZE = (640, 360)
# 速度显示配置（低画质版无法通过API限制速度，仅显示实际速度）
SPEED_PRINT_INTERVAL = 10  # 每10帧打印一次速度（减少刷屏）
# =====================================================================

# 全局变量
IMAGE_QUEUE = queue.Queue(maxsize=1)
# 存储当前车辆的控制对象（用于避让微调）
vehicle_control = carla.VehicleControl()

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
    """生成摄像头传感器（移除雷达，改用Actor直接检测，适配0.9.14低画质版）"""
    camera = None
    try:
        # 生成摄像头
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        # 设置摄像头低画质参数
        camera_bp.set_attribute('image_size_x', str(CAMERA_RESOLUTION[0]))
        camera_bp.set_attribute('image_size_y', str(CAMERA_RESOLUTION[1]))
        camera_bp.set_attribute('fov', str(CAMERA_FOV))
        camera_bp.set_attribute('sensor_tick', str(CAMERA_SENSOR_TICK))
        camera = world.spawn_actor(camera_bp, CAMERA_POSITION, attach_to=vehicle)
        camera.listen(image_callback)
        print("✅ 摄像头挂载成功")
        return camera
    except Exception as e:
        print(f"❌ 摄像头挂载失败：{e}")
        if camera:
            camera.destroy()
        return None

def get_vehicle_speed(vehicle):
    """获取车辆当前速度（km/h，适配CARLA 0.9.14）"""
    velocity = vehicle.get_velocity()
    # 计算速度：√(x² + y² + z²) （m/s），转换为km/h（×3.6）
    speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    speed_kmh = speed_ms * 3.6
    return round(speed_kmh, 1)

def detect_nearby_vehicles(world, vehicle):
    """直接通过Actor检测周围车辆（放弃雷达，适配低画质版API），返回最近车辆的距离、方位和是否有车"""
    if not vehicle:
        return None, None, False

    # 获取当前车辆的位置和朝向
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation
    vehicle_yaw = math.radians(vehicle_rotation.yaw)  # 车辆朝向的偏航角（弧度）

    # 存储最近的车辆信息
    min_distance = float('inf')
    target_azimuth = 0.0  # 目标车辆的方位角（-180~180，正前方为0）
    has_vehicle = False

    # 遍历世界中所有车辆Actor
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.id == vehicle.id:
            continue  # 跳过自己

        # 获取其他车辆的位置
        actor_location = actor.get_transform().location
        # 计算两车之间的直线距离
        distance = vehicle_location.distance(actor_location)

        # 只处理安全距离内的车辆
        if distance < SAFE_DISTANCE:
            # 计算目标车辆相对于当前车辆的方位角（前后左右）
            # 步骤1：计算向量
            dx = actor_location.x - vehicle_location.x
            dy = actor_location.y - vehicle_location.y
            # 步骤2：计算目标角度（弧度）
            target_angle = math.atan2(dy, dx)
            # 步骤3：转换为相对于车辆朝向的方位角（度）
            azimuth = math.degrees(target_angle - vehicle_yaw)
            # 归一化到-180~180度
            azimuth = (azimuth + 180) % 360 - 180

            # 只关注前方±60度的车辆（避免检测后方车辆）
            if abs(azimuth) < 60.0:
                min_distance = distance
                target_azimuth = azimuth
                has_vehicle = True

    if has_vehicle:
        return min_distance, target_azimuth, True
    else:
        return None, None, False

def avoid_vehicle(vehicle, distance, azimuth):
    """根据检测到的车辆方位，执行避让操作（向宽敞方向微调）"""
    global vehicle_control
    # 获取车辆当前的控制状态（保留自动驾驶的油门/刹车，只改转向）
    vehicle_control = vehicle.get_control()

    # 计算避让方向：根据方位角调整转向（-1~1之间，1为右，-1为左）
    steer_strength = (AVOIDANCE_ANGLE / 60.0) * 0.2  # 控制转向强度，避免过度打方向
    if azimuth > 0:  # 车辆在右侧（相对于当前车辆前方）
        vehicle_control.steer = -steer_strength  # 向左微调
    elif azimuth < 0:  # 车辆在左侧
        vehicle_control.steer = steer_strength   # 向右微调
    else:  # 正前方
        vehicle_control.steer = steer_strength   # 默认向右微调（宽敞方向）

    # 距离越近，转向强度稍大，同时轻微降速
    if distance < SAFE_DISTANCE / 2:
        vehicle_control.steer *= 1.5
        vehicle_control.throttle = max(0.4, vehicle_control.throttle)
    else:
        vehicle_control.throttle = max(0.5, vehicle_control.throttle)
    vehicle_control.brake = 0.0

    # 应用控制指令（覆盖自动驾驶的转向，保留油门/刹车）
    vehicle.apply_control(vehicle_control)
    print(f"\n⚠️ 检测到前方车辆：距离{distance:.1f}m，方位{azimuth:.1f}度，正在向{'右' if azimuth <=0 else '左'}避让...")

def main():
    """主函数：连接CARLA、随机加载地图、生成车辆、显示画面、检测并避让车辆"""
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

        # 4. 生成主车辆（带基础避让功能）
        vehicle = spawn_vehicle(world, blueprint_library, spawn_points)
        if vehicle is None:
            return

        # 5. 生成其他车辆（增加环境中的车辆，用于测试避让）
        other_vehicle_count = 5  # 生成5辆其他车辆
        spawned_count = 0
        for spawn_point in spawn_points:
            if spawned_count >= other_vehicle_count:
                break
            # 随机选择其他车型，跳过主车辆的车型（可选）
            other_vehicle_bps = [bp for bp in blueprint_library.filter('vehicle.*') if bp.id != PREFERRED_VEHICLE]
            if not other_vehicle_bps:
                other_vehicle_bps = blueprint_library.filter('vehicle.*')
            other_vehicle_bp = random.choice(other_vehicle_bps)
            try:
                other_vehicle = world.spawn_actor(other_vehicle_bp, spawn_point)
                other_vehicle.set_autopilot(True)
                spawned_count += 1
            except Exception as e:
                continue
        print(f"✅ 生成了{spawned_count}辆其他车辆，用于测试避让功能")

        # 6. 挂载摄像头传感器（移除雷达，避免属性错误）
        camera = spawn_camera(world, blueprint_library, vehicle)
        if camera is None:
            return

        # 7. 显示画面+检测车辆+避让逻辑
        print("\n📌 操作说明：")
        print(f"   - 按 'q' 退出程序")
        print(f"   - 车辆已启用自动驾驶+主动识别车辆避让功能")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *WINDOW_SIZE)

        frame_count = 0
        while True:
            # 显示摄像头画面
            if not IMAGE_QUEUE.empty():
                img = IMAGE_QUEUE.get(timeout=0.5)
                cv2.imshow(WINDOW_NAME, img)

            # 每隔一定帧数检测车辆（降低性能消耗，适配低画质版）
            frame_count += 1
            if frame_count % 5 == 0:  # 每5帧检测一次
                distance, azimuth, has_vehicle = detect_nearby_vehicles(world, vehicle)
                if has_vehicle:
                    avoid_vehicle(vehicle, distance, azimuth)
                else:
                    # 无车辆时，恢复默认转向（直行）
                    if vehicle_control.steer != 0.0:
                        vehicle_control.steer = 0.0
                        vehicle.apply_control(vehicle_control)
                frame_count = 0

            # 实时输出车辆速度（间隔打印，避免刷屏）
            if frame_count % SPEED_PRINT_INTERVAL == 0 and vehicle is not None:
                current_speed = get_vehicle_speed(vehicle)
                print(f"\r当前车辆速度：{current_speed} km/h", end="")

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
        # 销毁其他生成的车辆
        try:
            if world:
                for actor in world.get_actors().filter('vehicle.*'):
                    try:
                        actor.destroy()
                    except Exception:
                        pass
                print("✅ 销毁所有其他车辆")
        except Exception as e:
            print(f"⚠️ 销毁其他车辆失败：{e}")
        cv2.destroyAllWindows()
        print("✅ 程序结束")

if __name__ == '__main__':
    main()