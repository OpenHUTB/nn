#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA Vehicle Simulation with Visualization (sd_3/__main__.py)

核心功能：
1. 连接CARLA仿真服务器，清理历史车辆 Actor
2. 生成特斯拉Model3车辆，施加恒定油门控制
3. 基于PygameDisplay显示车辆挂载摄像头的实时画面
4. 基于Plotter绘制车辆X坐标和速度随时间的变化曲线
5. 设置固定的旁观者相机视角，方便观察仿真过程

依赖：
- tools.pygame_display.PygameDisplay：Pygame窗口显示摄像头画面
- tools.plotter_x.Plotter：Matplotlib绘制X坐标/速度曲线
"""

# 加入这几行代码，解决模块导入问题
import sys
import os

# 获取项目根目录（carla-python-examples）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将项目根目录加入Python搜索路径
sys.path.append(PROJECT_ROOT)
import carla
import numpy as np
import time
import pygame
import math
import traceback
from typing import Optional, Tuple

# 导入自定义工具类
from tools.pygame_display import PygameDisplay
from tools.plotter_x import Plotter

# ===================== 配置集中化（便于修改和维护）=====================
class SimConfig:
    """仿真配置类：集中管理所有硬编码参数"""
    # CARLA服务器连接信息
    CARLA_HOST = "localhost"
    CARLA_PORT = 2000
    CARLA_TIMEOUT = 10.0  # 连接超时时间（秒）

    # 车辆配置
    VEHICLE_MODEL = "vehicle.tesla.model3"  # 车辆模型
    VEHICLE_ROLE_NAME = "my_car"  # 车辆角色名（用于清理历史车辆）
    CONSTANT_THROTTLE = 0.8  # 恒定油门值（0-1）
    STEER_ANGLE = 0.0  # 恒定转向角（0=直线行驶）
    BRAKE_VALUE = 0.0  # 刹车值（0=不刹车）

    # 旁观者相机固定视角（可根据地图调整）
    SPECTATOR_LOCATION = carla.Location(x=63.12, y=29.88, z=5.61)
    SPECTATOR_ROTATION = carla.Rotation(pitch=-4.27, yaw=-170.21, roll=0.00)

    # 绘图配置
    DESIRED_SPEED = 0.0  # 期望速度（若无速度控制，设为0）

# ===================== 车辆状态计算工具函数 =====================
def calculate_vehicle_speed_kmh(vehicle: carla.Vehicle) -> float:
    """
    计算车辆当前速度（km/h）
    :param vehicle: CARLA车辆Actor对象
    :return: 车辆速度（km/h）
    """
    velocity = vehicle.get_velocity()
    # 计算速度矢量的模（m/s），转换为km/h（×3.6）
    speed_mps = np.linalg.norm([velocity.x, velocity.y, velocity.z])
    return speed_mps * 3.6

def get_vehicle_location(vehicle: carla.Vehicle) -> carla.Location:
    """
    获取车辆当前位置
    :param vehicle: CARLA车辆Actor对象
    :return: 车辆的Location对象
    """
    return vehicle.get_location()

# ===================== 仿真资源管理工具函数 =====================
def clean_up_prev_vehicles(world: carla.World, role_name: str) -> int:
    """
    清理地图中指定角色名的历史车辆
    :param world: CARLA的World对象
    :param role_name: 车辆角色名
    :return: 成功销毁的车辆数量
    """
    print(f"\n[资源清理] 搜索角色为'{role_name}'的历史车辆...")
    vehicles = world.get_actors().filter("vehicle.*")
    destroyed_count = 0

    for vehicle in vehicles:
        if vehicle.attributes.get("role_name") == role_name:
            print(f"  - 销毁历史车辆：{vehicle.type_id} (ID: {vehicle.id})")
            if vehicle.destroy():
                destroyed_count += 1
            else:
                print(f"  - 销毁车辆{vehicle.id}失败")

    print(f"[资源清理] 共销毁{destroyed_count}辆历史车辆")
    return destroyed_count

def set_spectator_fixed_view(world: carla.World, location: carla.Location, rotation: carla.Rotation) -> bool:
    """
    设置旁观者相机的固定视角
    :param world: CARLA的World对象
    :param location: 旁观者相机位置
    :param rotation: 旁观者相机旋转角度
    :return: 是否设置成功
    """
    try:
        spectator = world.get_spectator()
        spectator_transform = carla.Transform(location, rotation)
        spectator.set_transform(spectator_transform)
        print(f"\n[视角设置] 旁观者相机位置：({location.x:.2f}, {location.y:.2f}, {location.z:.2f})")
        print(f"[视角设置] 旁观者相机旋转：(俯仰：{rotation.pitch:.2f}, 偏航：{rotation.yaw:.2f}, 翻滚：{rotation.roll:.2f})")
        return True
    except Exception as e:
        print(f"[视角设置] 失败：{e}")
        return False

# ===================== 主仿真函数 =====================
def main():
    """主仿真入口函数：完成所有仿真流程的初始化、运行和清理"""
    # 初始化核心变量（所有资源对象初始化为None，便于后续清理）
    client: Optional[carla.Client] = None
    world: Optional[carla.World] = None
    vehicle: Optional[carla.Vehicle] = None
    pygame_display: Optional[PygameDisplay] = None
    plotter: Optional[Plotter] = None
    simulation_start_time: Optional[float] = None

    try:
        # 1. 连接CARLA服务器
        print(f"[CARLA连接] 尝试连接 {SimConfig.CARLA_HOST}:{SimConfig.CARLA_PORT}...")
        client = carla.Client(SimConfig.CARLA_HOST, SimConfig.CARLA_PORT)
        client.set_timeout(SimConfig.CARLA_TIMEOUT)
        world = client.get_world()
        print(f"[CARLA连接] 成功连接到地图：{world.get_map().name}")

        # 2. 清理历史车辆
        clean_up_prev_vehicles(world, SimConfig.VEHICLE_ROLE_NAME)

        # 3. 获取车辆生成点
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("[生成车辆] 地图中未找到可用的生成点！")
        spawn_point = spawn_points[0]
        print(f"[生成车辆] 选择生成点：({spawn_point.location.x:.2f}, {spawn_point.location.y:.2f}, {spawn_point.location.z:.2f})")

        # 4. 加载车辆蓝图并设置属性
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(SimConfig.VEHICLE_MODEL)[0]
        vehicle_bp.set_attribute("role_name", SimConfig.VEHICLE_ROLE_NAME)
        print(f"[生成车辆] 加载车辆蓝图：{vehicle_bp.id}")

        # 5. 生成车辆
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            raise RuntimeError(f"[生成车辆] 在生成点{spawn_point.location}生成车辆失败！")
        print(f"[生成车辆] 成功生成：{vehicle.type_id} (ID: {vehicle.id})")

        # 6. 设置旁观者相机视角
        set_spectator_fixed_view(world, SimConfig.SPECTATOR_LOCATION, SimConfig.SPECTATOR_ROTATION)

        # 7. 初始化可视化组件
        print("\n[可视化] 初始化绘图器（Plotter）...")
        plotter = Plotter()
        plotter.init_plot()
        print("[可视化] 绘图器初始化完成")

        print("[可视化] 初始化Pygame显示窗口...")
        pygame_display = PygameDisplay(world, vehicle)
        print("[可视化] Pygame显示窗口初始化完成")

        # 8. 记录仿真开始时间
        simulation_start_time = time.time()
        print(f"\n[仿真启动] 开始运行仿真（恒定油门：{SimConfig.CONSTANT_THROTTLE}）")
        print("[仿真启动] 按ESC或关闭Pygame窗口停止仿真...")

        # 9. 主仿真循环
        while True:
            # 9.1 处理Pygame事件（关闭窗口、ESC键）
            if pygame_display.parse_events():
                print("[仿真控制] 检测到Pygame退出请求，停止仿真...")
                break

            # 9.2 等待仿真tick（同步仿真时间）
            world.wait_for_tick()

            # 9.3 收集车辆状态数据
            current_time = time.time()
            vehicle_location = get_vehicle_location(vehicle)
            vehicle_speed = calculate_vehicle_speed_kmh(vehicle)
            sim_elapsed_time = current_time - simulation_start_time  # 仿真已运行时间（秒）

            # 9.4 渲染Pygame窗口（摄像头画面）
            pygame_display.render()

            # 9.5 更新绘图器（X坐标、速度曲线）
            if plotter and plotter.is_initialized:
                try:
                    plotter.update_plot(
                        sim_elapsed_time,
                        vehicle_location.x,
                        vehicle_speed,
                        SimConfig.DESIRED_SPEED
                    )
                except Exception as e:
                    print(f"[绘图器] 更新失败（可能已关闭窗口）：{e}")
                    plotter.cleanup_plot()
                    plotter = None

            # 9.6 车辆控制：施加恒定油门
            vehicle_control = carla.VehicleControl(
                throttle=SimConfig.CONSTANT_THROTTLE,
                steer=SimConfig.STEER_ANGLE,
                brake=SimConfig.BRAKE_VALUE
            )
            vehicle.apply_control(vehicle_control)

    # 异常处理
    except KeyboardInterrupt:
        print("\n[仿真中断] 用户按下Ctrl+C，停止仿真...")
    except RuntimeError as e:
        print(f"\n[仿真错误] 运行时错误：{e}")
        traceback.print_exc()
    except Exception as e:
        print(f"\n[仿真错误] 未知异常：{e}")
        traceback.print_exc()

    # 资源清理（无论是否异常，都执行）
    finally:
        print("\n[资源清理] 开始清理仿真资源...")

        # 清理Pygame显示窗口
        if pygame_display:
            print("[资源清理] 销毁Pygame显示窗口...")
            pygame_display.destroy()

        # 清理绘图器
        if plotter and plotter.is_initialized:
            print("[资源清理] 清理绘图器...")
            plotter.cleanup_plot()

        # 销毁车辆
        if vehicle and vehicle.is_alive:
            print(f"[资源清理] 销毁车辆：{vehicle.type_id} (ID: {vehicle.id})")
            if vehicle.destroy():
                print("[资源清理] 车辆销毁成功")
            else:
                print("[资源清理] 车辆销毁失败")

        print("[资源清理] 仿真资源清理完成")

if __name__ == "__main__":
    main()