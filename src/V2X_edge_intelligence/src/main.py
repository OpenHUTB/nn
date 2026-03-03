#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA 0.9.10 车路协同避障
核心修改：移除直接从模拟器获取数据，改为基于路侧激光雷达感知障碍
"""
import sys
import os
import time
import json
import math
import threading
from typing import Optional
from threading import Lock


# ====================== 1. 智能加载CARLA（无硬编码绝对路径，修复重复导入） ======================
def load_carla() -> Optional[object]:
    """
    智能加载CARLA Python API，优先级：
    1. 系统环境变量 CARLA_ROOT（推荐）
    2. 自动搜索常见目录（当前目录、用户目录、上级目录）
    3. 引导用户手动输入路径
    """
    python_version = f"py{sys.version_info.major}.{sys.version_info.minor}"
    egg_file_patterns = [
        f"carla-0.9.10-{python_version}-win-amd64.egg",
        "carla-0.9.10-py3.7-win-amd64.egg",  # 兼容Python3.7（CARLA 0.9.10主流版本）
        "carla-0.9.10-*.egg"  # 兜底匹配所有0.9.10版本的egg文件
    ]

    # 候选路径列表（无任何硬编码绝对路径）
    candidate_paths = []

    # 优先级1：从环境变量CARLA_ROOT读取
    carla_root = os.getenv("CARLA_ROOT")
    if carla_root and os.path.isdir(carla_root):
        candidate_paths.append(os.path.join(carla_root, "PythonAPI", "carla", "dist"))

# ===================== 1. 自动适配CARLA路径（无绝对路径） =====================
def setup_carla_path():
    """自动配置CARLA路径（优先级：环境变量 > 相对路径 > 提示用户）"""
    # 优先级1：读取环境变量 CARLA_PYTHON_API_PATH
    carla_api_path = os.environ.get("CARLA_PYTHON_API_PATH")
    if carla_api_path and os.path.exists(carla_api_path):
        egg_files = [f for f in os.listdir(carla_api_path) if f.endswith(".egg")]
        if egg_files:
            carla_egg_path = os.path.join(carla_api_path, egg_files[0])
            print(f"🔍 从环境变量加载CARLA egg：{carla_egg_path}")
            sys.path.insert(0, carla_egg_path)
            return True

    # 优先级2：自动查找常见的相对路径
    common_paths = [
        "./PythonAPI/carla/dist",
        "../WindowsNoEditor/PythonAPI/carla/dist",
        "./WindowsNoEditor/PythonAPI/carla/dist"
    ]
    for path in common_paths:
        if os.path.exists(path):
            egg_files = [f for f in os.listdir(path) if f.endswith(".egg")]
            if egg_files:
                carla_egg_path = os.path.join(path, egg_files[0])
                print(f"🔍 自动找到CARLA egg：{carla_egg_path}")
                sys.path.insert(0, carla_egg_path)
                return True

    # 优先级3：提示用户手动输入路径
    print("\n⚠️  未自动找到CARLA PythonAPI路径！")
    print("📌 请先设置环境变量 CARLA_PYTHON_API_PATH，例如：")
    print("   Windows: set CARLA_PYTHON_API_PATH=D:\\WindowsNoEditor\\PythonAPI\\carla\\dist")
    print("   Linux/Mac: export CARLA_PYTHON_API_PATH=/path/to/Carla/PythonAPI/carla/dist")
    manual_path = input("\n请输入CARLA egg文件所在目录（留空退出）：").strip()
    if manual_path and os.path.exists(manual_path):
        egg_files = [f for f in os.listdir(manual_path) if f.endswith(".egg")]
        if egg_files:
            carla_egg_path = os.path.join(manual_path, egg_files[0])
            sys.path.insert(0, carla_egg_path)
            print(f"✅ 手动加载CARLA egg：{carla_egg_path}")
            return True

    return False

# 初始化CARLA路径
print(f"🔍 当前Python解释器路径：{sys.executable}")
print(f"🔍 当前Python版本：{sys.version.split()[0]}")

if not setup_carla_path():
    print("\n❌ 无法找到CARLA egg文件，请检查路径配置！")
    sys.exit(1)

# ====================== 2. 核心参数（远距离停止+渐进减速+激光雷达感知） ======================
DECEL_DISTANCE = 20.0  # 距离<20米开始减速（提前缓冲）
STOP_DISTANCE = 12.0  # 距离<12米完全停止（远离蓝车，不撞）
NORMAL_THROTTLE = 0.7  # 正常直行油门
DECEL_THROTTLE = 0.1  # 减速阶段油门（缓慢靠近）
OBSTACLE_DISTANCE = 25.0  # 蓝车在红车同车道正前方25米（更远初始距离）
BRAKE_FORCE = 1.0  # 满刹车（停止彻底）

# 激光雷达参数（路侧RSU感知）
LIDAR_CHANNELS = 32  # 32线激光雷达
LIDAR_RANGE = 50.0  # 探测范围50米
LIDAR_ROTATION_FREQ = 10.0  # 旋转频率10Hz
LIDAR_POINTS_PER_SEC = 100000  # 点云密度
LIDAR_UPPER_FOV = 10.0  # 上视场角10度
LIDAR_LOWER_FOV = -30.0  # 下视场角-30度
LIDAR_HORIZONTAL_FOV = 60.0  # 水平视场角±30度（仅感知前方道路）

# ====================== 3. 全局变量（新增激光雷达感知相关） ======================
actors = []
lidar_sensor = None
perceived_obstacle_distance = float('inf')  # 激光雷达感知到的最近障碍距离
perception_data_lock = Lock()  # 线程安全锁，防止数据竞争


# ====================== 4. 激光雷达数据回调函数（核心：基于传感器感知障碍） ======================
def lidar_callback(data):
    """
    路侧激光雷达数据回调函数
    处理点云数据，筛选前方道路内的点，计算最近障碍距离
    :param data: 激光雷达原始点云数据（carla.LidarMeasurement）
    """
    global perceived_obstacle_distance
    min_distance = float('inf')

    # 遍历所有点云点，筛选有效障碍点
    for point in data:
        # 1. 筛选水平视场角内的点（仅感知前方±30度）
        horizontal_angle = math.degrees(math.atan2(point.y, point.x))
        if abs(horizontal_angle) > LIDAR_HORIZONTAL_FOV / 2:
            continue

        # 2. 筛选高度范围内的点（过滤地面，仅感知0.5-2.0米高度的障碍）
        if point.z < 0.5 or point.z > 2.0:
            continue

        # 3. 计算点到激光雷达的距离
        distance = math.sqrt(point.x ** 2 + point.y ** 2 + point.z ** 2)
        if distance < LIDAR_RANGE and distance < min_distance:
            min_distance = distance

    # 线程安全更新感知到的障碍距离
    with perception_data_lock:
        perceived_obstacle_distance = min_distance if min_distance != float('inf') else float('inf')


# ====================== 5. 主程序（基于激光雷达感知的避障逻辑） ======================
def main():
    global lidar_sensor, perceived_obstacle_distance
    # 1. 连接CARLA服务器
    try:
        # 1. 连接CARLA+加载地图
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.load_world('Town01')
        world.set_weather(carla.WeatherParameters.ClearNoon)
        print("✅ 连接CARLA成功！加载Town01场景")

        # 2. 清理残留Actor
        for actor in world.get_actors():
            if actor.type_id in ['vehicle.*', 'static.prop.*', 'sensor.*']:
                actor.destroy()

        # 3. 生成红色主车（同车道起点，手动挂前进挡）
        blueprint_lib = world.get_blueprint_library()
        main_car_bp = blueprint_lib.filter('vehicle.tesla.model3')[0]
        main_car_bp.set_attribute('color', '255,0,0')  # 红色
        spawn_points = world.get_map().get_spawn_points()
        main_car_spawn = spawn_points[5]  # 开阔直车道生成点（无围栏）
        main_car = world.spawn_actor(main_car_bp, main_car_spawn)
        actors.append(main_car)

        # 适配0.9.10：手动挂前进挡+解除手刹
        init_control = carla.VehicleControl(
            throttle=NORMAL_THROTTLE,
            steer=0.0,  # 全程直行，不转向
            manual_gear_shift=True,  # 开启手动换挡
            gear=1,  # 前进挡
            hand_brake=False,
            reverse=False
        )
        main_car.apply_control(init_control)
        print("✅ 生成红色主车：同车道起点，手动挂前进挡（直行）")

        # 4. 生成蓝色障碍车（红车同车道正前方25米，y坐标一致=同车道）
        obstacle_car_bp = blueprint_lib.filter('vehicle.tesla.model3')[0]
        obstacle_car_bp.set_attribute('color', '0,0,255')  # 蓝色
        obstacle_transform = carla.Transform(
            carla.Location(
                x=main_car_spawn.location.x + OBSTACLE_DISTANCE,  # 正前方25米
                y=main_car_spawn.location.y,  # 同一车道（y坐标一致）
                z=main_car_spawn.location.z
            ),
            main_car_spawn.rotation
        )
        obstacle_car = world.spawn_actor(obstacle_car_bp, obstacle_transform)
        obstacle_car.apply_control(carla.VehicleControl(hand_brake=True))  # 蓝车静止
        actors.append(obstacle_car)
        print(f"✅ 生成蓝色障碍车：红车同车道正前方{OBSTACLE_DISTANCE}米")

        # 5. 生成路侧边缘节点+挂载激光雷达（V2X感知设备，核心修改）
        # 路侧节点位置：红车起点前方15米，右侧3米，高度3米
        edge_node_transform = carla.Transform(
            carla.Location(
                x=main_car_spawn.location.x + 15,
                y=main_car_spawn.location.y + 3,
                z=3.0
            ),
            main_car_spawn.rotation
        )
        # 加载激光雷达蓝图
        lidar_bp = blueprint_lib.find('sensor.lidar.ray_cast')
        # 配置激光雷达参数
        lidar_bp.set_attribute('channels', str(LIDAR_CHANNELS))
        lidar_bp.set_attribute('range', str(LIDAR_RANGE))
        lidar_bp.set_attribute('rotation_frequency', str(LIDAR_ROTATION_FREQ))
        lidar_bp.set_attribute('points_per_second', str(LIDAR_POINTS_PER_SEC))
        lidar_bp.set_attribute('upper_fov', str(LIDAR_UPPER_FOV))
        lidar_bp.set_attribute('lower_fov', str(LIDAR_LOWER_FOV))
        # 生成激光雷达（挂载在路侧节点位置，无实体节点，直接挂载传感器）
        lidar_sensor = world.spawn_actor(lidar_bp, edge_node_transform)
        # 注册激光雷达回调函数
        lidar_sensor.listen(lidar_callback)
        actors.append(lidar_sensor)
        print("✅ 生成路侧激光雷达（V2X感知设备）：感知前方道路障碍")

        # 6. 初始近视角（紧贴红车，看清同车道蓝车）
        spectator = world.get_spectator()
        spectator_transform = carla.Transform(
            carla.Location(
                x=main_car_spawn.location.x + 4,
                y=main_car_spawn.location.y,
                z=main_car_spawn.location.z + 6  # 稍高，看清25米外蓝车
            ),
            carla.Rotation(pitch=-45, yaw=main_car_spawn.rotation.yaw)  # 直视同车道
        )
        spectator.set_transform(spectator_transform)
        print("✅ 初始视角设置完成：紧贴红车，看清同车道蓝车")

        # 7. 运行提示
        print("\n======= 车路协同避障仿真（激光雷达感知版） =======")
        print(f"✅ 红蓝车：同一车道，蓝车在红车正前方{OBSTACLE_DISTANCE}米")
        print("✅ 感知方式：路侧激光雷达感知障碍（无直接模拟器数据）")
        print("✅ 红车逻辑：直行→20米处减速→12米处完全停止（远离蓝车不撞）")
        print("✅ 镜头：自由操作（左键旋转/滚轮缩放/WASD平移）")
        print("✅ 退出方式：Ctrl+C 停止程序")
        print("==============================================\n")

        main_car_control = init_control
        is_stopped = False  # 红车停止标记

        while True:
            # 从激光雷达感知数据中获取最近障碍距离（线程安全）
            with perception_data_lock:
                current_distance = perceived_obstacle_distance

            # 核心逻辑：渐进减速+远距离停止（基于激光雷达感知距离）
            if not is_stopped:
                if current_distance == float('inf'):
                    # 未感知到障碍，正常直行
                    main_car_control.throttle = NORMAL_THROTTLE
                    main_car_control.brake = 0.0
                    current_speed = math.hypot(main_car.get_velocity().x, main_car.get_velocity().y)
                    print(f"\r【直行中】未感知到障碍 | 当前速度：{current_speed:.2f}m/s", end="")
                elif current_distance > DECEL_DISTANCE:
                    # 阶段1：感知距离>20米，正常直行（无减速）
                    main_car_control.throttle = NORMAL_THROTTLE
                    main_car_control.brake = 0.0
                    current_speed = math.hypot(main_car.get_velocity().x, main_car.get_velocity().y)
                    print(f"\r【直行中】感知障碍距离：{current_distance:.1f}米 | 当前速度：{current_speed:.2f}m/s", end="")
                elif DECEL_DISTANCE >= current_distance > STOP_DISTANCE:
                    # 阶段2：20米≥感知距离>12米，渐进减速（缓慢靠近）
                    main_car_control.throttle = DECEL_THROTTLE
                    main_car_control.brake = 0.0
                    current_speed = math.hypot(main_car.get_velocity().x, main_car.get_velocity().y)
                    print(f"\r【减速中】感知障碍距离：{current_distance:.1f}米 | 当前速度：{current_speed:.2f}m/s", end="")
                else:
                    # 阶段3：感知距离≤12米，满刹车完全停止（远离蓝车，不撞）
                    main_car_control.throttle = 0.0
                    main_car_control.brake = BRAKE_FORCE
                    print(f"\r【已停止】感知障碍距离：{current_distance:.1f}米 → 远离蓝车，完全停止", end="")
                    is_stopped = True
            else:
                # 保持停止状态，避免再次移动
                main_car_control.throttle = 0.0
                main_car_control.brake = BRAKE_FORCE
                with perception_data_lock:
                    current_distance = perceived_obstacle_distance
                print(f"\r【保持停止】感知障碍距离：{current_distance:.1f}米 | 红车静止不动", end="")

            # 持续发送控制指令，确保状态生效
            main_car.apply_control(main_car_control)
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n🛑 程序终止，清理资源...")
    except Exception as e:
        print(f"\n⚠️  运行错误：{e} | 请确认CARLA 0.9.10已启动（localhost:2000）")
    finally:
        # 清理所有资源
        global perceived_obstacle_distance
        perceived_obstacle_distance = float('inf')
        for actor in actors:
            try:
                if actor.is_alive:
                    actor.destroy()
            except:
                pass
        print("✅ 资源清理完成，程序退出！")


# 唯一入口
if __name__ == "__main__":
    main()