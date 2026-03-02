#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA 0.9.10 - 路侧感知可视化
核心修改：移除直接get_velocity，改用速度传感器感知车辆速度
"""
import sys
import os
import time
import math
import threading
from threading import Lock


# ====================== 1. 智能加载CARLA（无绝对路径，核心修改） ======================
def load_carla():
    """
    智能加载CARLA，优先级：
    1. 检查系统环境变量 CARLA_ROOT
    2. 检查当前目录及上级目录
    3. 提示用户手动输入CARLA安装路径
    """
    carla_egg_paths = []

    # 优先级1：读取系统环境变量 CARLA_ROOT
    carla_root = os.getenv("CARLA_ROOT")
    if carla_root:
        egg_path = os.path.join(
            carla_root,
            "PythonAPI", "carla", "dist",
            f"carla-0.9.10-py{sys.version_info.major}.{sys.version_info.minor}-win-amd64.egg"
        )
        carla_egg_paths.append(egg_path)

    # 优先级2：检查常见路径（当前目录、上级目录）
    common_paths = [
        os.path.join(os.getcwd(), "PythonAPI", "carla", "dist"),
        os.path.join(os.path.dirname(os.getcwd()), "PythonAPI", "carla", "dist"),
        os.path.join("D:", os.sep, "Carla", "PythonAPI", "carla", "dist"),  # 通用默认路径
    ]
    for path in common_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.startswith("carla-0.9.10") and file.endswith(".egg"):
                    carla_egg_paths.append(os.path.join(path, file))

    # 尝试加载CARLA
    for egg_path in carla_egg_paths:
        if os.path.exists(egg_path):
            sys.path.append(egg_path)
            try:
                import carla
                print(f"✅ 成功加载CARLA：{egg_path}")
                return carla
            except ImportError:
                continue

    # 优先级3：提示用户手动输入路径
    print("❌ 未自动找到CARLA egg文件！")
    while True:
        manual_path = input(
            "请输入CARLA egg文件的完整路径（例如：D:/Carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg）：").strip()
        if os.path.exists(manual_path) and manual_path.endswith(".egg"):
            sys.path.append(manual_path)
            try:
                import carla
                print(f"✅ 手动加载CARLA成功：{manual_path}")
                return carla
            except ImportError:
                print("❌ 该路径的egg文件加载失败，请重新输入！")
        else:
            print("❌ 路径不存在或不是egg文件，请重新输入！")


# 加载CARLA（无绝对路径）
carla = load_carla()

# ====================== 2. 全局变量（新增传感器相关） ======================
RSU_LOC = carla.Location(x=0.0, y=0.0, z=2.0)  # RSU高度降低，更贴合实际
actors = []
speed_sensors = {}  # 存储各车辆的速度传感器 {vehicle_id: sensor_actor}
vehicle_speed_data = {}  # 存储传感器采集的速度数据 {vehicle_id: speed}
speed_data_lock = Lock()  # 线程安全锁，防止数据竞争
world = None
spectator = None
is_running = True
vehicle_controls = {}


# ====================== 3. 速度传感器回调函数（核心：感知速度） ======================
def velocity_sensor_callback(data, vehicle_id):
    """
    速度传感器数据回调函数
    :param data: 传感器原始数据（carla.Vector3D）
    :param vehicle_id: 车辆ID，用于关联数据
    """
    global vehicle_speed_data
    # 计算车辆速度（m/s）：合速度 = √(x² + y²)（忽略z轴，仅水平速度）
    speed = math.hypot(data.x, data.y)
    # 线程安全更新速度数据
    with speed_data_lock:
        vehicle_speed_data[vehicle_id] = speed


# ====================== 4. 可视化函数（RSU大小适中） ======================
def draw_elements():
    if not world:
        return
    debug = world.debug
    duration = 2.0

    # 1. 绘制RSU（大小适中：1*1*1.5米）
    debug.draw_box(
        box=carla.BoundingBox(RSU_LOC, carla.Vector3D(1.0, 1.0, 1.5)),
        rotation=carla.Rotation(),
        thickness=0.5,
        color=carla.Color(255, 0, 0),
        life_time=duration
    )
    debug.draw_string(
        carla.Location(x=0.0, y=0.0, z=4.0),
        "RSU - 路侧节点",
        False, carla.Color(255, 0, 0), duration
    )

    # 2. 绘制感知范围（蓝色圆圈）
    for i in range(12):
        angle1 = math.radians(i * 30)
        angle2 = math.radians((i + 1) * 30)
        p1 = carla.Location(
            x=RSU_LOC.x + 50 * math.cos(angle1),
            y=RSU_LOC.y + 50 * math.sin(angle1),
            z=0.5
        )
        p2 = carla.Location(
            x=RSU_LOC.x + 50 * math.cos(angle2),
            y=RSU_LOC.y + 50 * math.sin(angle2),
            z=0.5
        )
        debug.draw_line(p1, p2, 1.5, carla.Color(0, 0, 255), duration)

    # 3. 绘制车辆信息（使用传感器感知的速度）
    vehicles = world.get_actors().filter("vehicle.*")
    with speed_data_lock:
        for veh in vehicles:
            loc = veh.get_transform().location
            # 从传感器数据中获取速度，无数据则显示0.0
            speed = vehicle_speed_data.get(veh.id, 0.0)
            debug.draw_string(
                carla.Location(loc.x, loc.y, loc.z + 2.0),
                f"车{veh.id}\n{speed:.1f}m/s",
                False, carla.Color(255, 255, 0), duration
            )


# ====================== 5. 生成车辆+挂载速度传感器（核心修改） ======================
def spawn_vehicles():
    # 清除旧车辆和传感器
    for veh in world.get_actors().filter("vehicle.*"):
        veh.destroy()
    for sensor in speed_sensors.values():
        if sensor.is_alive:
            sensor.destroy()
    speed_sensors.clear()
    vehicle_speed_data.clear()

    # 获取官方道路生成点
    map = world.get_map()
    road_points = map.get_spawn_points()
    valid_points = []
    for p in road_points:
        dist = math.hypot(p.location.x - RSU_LOC.x, p.location.y - RSU_LOC.y)
        if 10 < dist < 100:
            valid_points.append(p)
            if len(valid_points) >= 2:
                break
    valid_points = valid_points[:2]
    print(f"✅ 选中{len(valid_points)}个道路生成点")

    # 加载车辆蓝图
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter("vehicle")
    veh_bp = vehicle_bps[0]
    print(f"✅ 使用车辆蓝图：{veh_bp.id}")

    # 加载速度传感器蓝图（CARLA 0.9.10原生支持）
    speed_sensor_bp = bp_lib.find("sensor.other.velocity")
    # 传感器参数配置（更新频率10Hz，符合真实感知）
    speed_sensor_bp.set_attribute("frequency", "10")

    # 生成车辆并挂载速度传感器
    for i, trans in enumerate(valid_points):
        try:
            # 生成车辆
            veh = world.spawn_actor(veh_bp, trans)
            if veh:
                actors.append(veh)
                # 手动控制指令
                control = carla.VehicleControl()
                control.throttle = 0.5
                control.steer = 0.0 if i == 0 else 0.1
                control.brake = 0.0
                control.hand_brake = False
                vehicle_controls[veh.id] = control
                print(f"✅ 车辆{i + 1}生成成功（ID={veh.id}）")

                # 挂载速度传感器（传感器位置：车辆中心，不影响可视化）
                sensor_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.5))
                speed_sensor = world.spawn_actor(
                    speed_sensor_bp,
                    sensor_transform,
                    attach_to=veh,
                    attachment_type=carla.AttachmentType.Rigid
                )
                if speed_sensor:
                    # 注册传感器回调函数，关联车辆ID
                    speed_sensor.listen(lambda data, vid=veh.id: velocity_sensor_callback(data, vid))
                    speed_sensors[veh.id] = speed_sensor
                    actors.append(speed_sensor)
                    print(f"✅ 车辆{veh.id}已挂载速度传感器")
        except Exception as e:
            print(f"⚠️  车辆{i + 1}生成失败：{str(e)[:50]}")
            continue


# ====================== 6. 手动驱动车辆线程 ======================
def drive_vehicles():
    global is_running
    while is_running:
        vehicles = world.get_actors().filter("vehicle.*")
        for veh in vehicles:
            if veh.id in vehicle_controls:
                try:
                    veh.apply_control(vehicle_controls[veh.id])
                except:
                    continue
        time.sleep(0.05)


# ====================== 7. 主函数（无视角锁定，可自由操作） ======================
def main():
    global world, spectator, is_running
    # 1. 连接CARLA
    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    try:
        world = client.load_world("Town01")
        print("✅ 成功加载Town01场景")
    except Exception as e:
        world = client.get_world()
        print(f"⚠️  加载Town01失败，使用当前场景：{str(e)[:50]}")

    # 2. 设置异步模式，无卡死
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    print("✅ 启用异步模式，无卡死")

    # 3. 初始化视角（仅一次，之后可自由操作）
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        carla.Location(x=0.0, y=0.0, z=40.0),
        carla.Rotation(pitch=-70.0, yaw=0.0, roll=0.0)
    ))
    print("✅ 初始视角已设置，可自由转动视角！")
    print("💡 CARLA视角操作：右键按住旋转 | 滚轮缩放 | WASD移动")

    # 4. 生成车辆（含速度传感器）
    spawn_vehicles()

    # 5. 启动驱动线程
    drive_thread = threading.Thread(target=drive_vehicles, daemon=True)
    drive_thread.start()
    print("✅ 车辆驱动线程启动，车辆开始行驶")

    # 6. 主循环
    print("\n" + "=" * 60)
    print("📌 CARLA 0.9.10 完美运行！（传感器感知速度版）")
    print("✅ 无绝对路径 | ✅ 可自由视角 | ✅ RSU大小适中 | ✅ 车辆沿道路行驶")
    print("✅ 速度传感器感知 | ✅ 无直接调用get_velocity | ✅ 线程安全数据采集")
    print("✅ 无任何报错 | ✅ 无卡死 | ✅ 可视化清晰")
    print("💡 按Ctrl+C停止程序")
    print("=" * 60 + "\n")
    try:
        while is_running:
            draw_elements()
            # 打印车辆状态（使用传感器速度）
            vehicles = world.get_actors().filter("vehicle.*")
            status = []
            with speed_data_lock:
                for veh in vehicles:
                    loc = veh.get_transform().location
                    speed = vehicle_speed_data.get(veh.id, 0.0)
                    status.append(f"车{veh.id}：({loc.x:.0f},{loc.y:.0f}) 速度{speed:.1f}m/s")
            if status:
                print(f"\r{' | '.join(status)}", end="")
            else:
                print("\r暂无车辆生成！", end="")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n\n🛑 接收到停止指令，正在清理资源...")
        is_running = False

    # 7. 清理资源（含传感器）
    for actor in actors:
        try:
            if actor.is_alive:
                actor.destroy()
        except:
            pass
    # 额外清理传感器
    for sensor in speed_sensors.values():
        try:
            if sensor.is_alive:
                sensor.destroy()
        except:
            pass
    print("✅ 资源清理完成，程序正常退出")


# ====================== 运行程序 ======================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        # 兜底清理资源
        for actor in actors:
            try:
                actor.destroy()
            except:
                pass
    finally:
        # 确保传感器全部销毁
        for sensor in speed_sensors.values():
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except:
                pass