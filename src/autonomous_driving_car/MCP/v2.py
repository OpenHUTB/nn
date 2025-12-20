#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA Waypoint Following with Simple Speed Control & Pygame Display (sd_4/__main__.py)

核心功能：
1. 连接CARLA仿真服务器，清理历史车辆Actor
2. 生成特斯拉Model3车辆，基于预定义航点实现路径跟随
3. 采用简易的bang-bang速度控制器，根据当前速度与目标速度调整油门/刹车
4. 基于PygameDisplay显示车辆挂载摄像头的实时画面
5. 基于Plotter绘制车辆X坐标、实际速度与期望速度的时间曲线
6. 设置固定的旁观者相机视角，方便观察仿真过程

依赖：
- tools.plotter_x.Plotter：Matplotlib绘制X坐标/速度曲线
- tools.pygame_display.PygameDisplay：Pygame窗口显示摄像头画面
"""

# ===================== 路径处理（解决tools模块导入问题）=====================
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
# ==============================================================================

import carla
import numpy as np
import time
import traceback
from typing import Optional, List, Tuple

# 导入自定义工具类
from tools.plotter_x import Plotter
from tools.pygame_display import PygameDisplay

# ===================== 配置集中化（便于修改和维护）=====================
class SimConfig:
    """仿真配置类：集中管理所有硬编码参数"""
    # CARLA服务器连接信息
    CARLA_HOST = "localhost"
    CARLA_PORT = 2000
    CARLA_TIMEOUT = 10.0  # 连接超时时间（秒）

    # 车辆配置
    VEHICLE_MODEL = "vehicle.tesla.model3"
    VEHICLE_ROLE_NAME = "my_car"

    # 航点配置：[x坐标, y坐标, 期望速度(km/h)]
    WAYPOINTS: List[List[float]] = [
        [-64.0, 24.5, 40.0],  # 起始点（第一段路径的参考点）
        [40.0, 24.5, 40.0],   # 制动点前的行驶点（保持40km/h）
        [70.0, 24.5, 0.0]     # 停止点（目标速度0km/h）
    ]
    WAYPOINT_THRESHOLD = 5.0  # 切换航点的距离阈值（米）
    INITIAL_TARGET_WAYPOINT_ID = 1  # 初始目标航点索引（从1开始，0为起始点）

    # 旁观者相机固定视角
    SPECTATOR_LOCATION = carla.Location(x=63.12, y=29.88, z=5.61)
    SPECTATOR_ROTATION = carla.Rotation(pitch=-4.27, yaw=-170.21, roll=0.00)

# ===================== 车辆状态计算工具函数 =====================
def get_vehicle_location(vehicle: carla.Vehicle) -> carla.Location:
    """
    获取车辆当前位置
    :param vehicle: CARLA车辆Actor对象
    :return: 车辆的Location对象
    """
    return vehicle.get_location()

def calculate_vehicle_speed_kmh(vehicle: carla.Vehicle) -> float:
    """
    计算车辆当前速度（km/h）
    :param vehicle: CARLA车辆Actor对象
    :return: 车辆速度（km/h）
    """
    velocity = vehicle.get_velocity()
    speed_mps = np.linalg.norm([velocity.x, velocity.y, velocity.z])
    return speed_mps * 3.6

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

def set_spectator_fixed_view(world: carla.World) -> bool:
    """
    设置旁观者相机的固定视角
    :param world: CARLA的World对象
    :return: 是否设置成功
    """
    try:
        spectator = world.get_spectator()
        spectator_transform = carla.Transform(
            SimConfig.SPECTATOR_LOCATION,
            SimConfig.SPECTATOR_ROTATION
        )
        spectator.set_transform(spectator_transform)
        print(f"\n[视角设置] 旁观者相机位置：({SimConfig.SPECTATOR_LOCATION.x:.2f}, {SimConfig.SPECTATOR_LOCATION.y:.2f}, {SimConfig.SPECTATOR_LOCATION.z:.2f})")
        print(f"[视角设置] 旁观者相机旋转：(俯仰：{SimConfig.SPECTATOR_ROTATION.pitch:.2f}, 偏航：{SimConfig.SPECTATOR_ROTATION.yaw:.2f}, 翻滚：{SimConfig.SPECTATOR_ROTATION.roll:.2f})")
        return True
    except Exception as e:
        print(f"[视角设置] 失败：{e}")
        return False

# ===================== 航点管理函数 =====================
def update_target_waypoint(
    vehicle_location: carla.Location,
    current_target_id: int,
    waypoints: List[List[float]],
    threshold: float
) -> int:
    """
    检查是否到达当前目标航点，若到达则切换到下一个航点
    :param vehicle_location: 车辆当前位置
    :param current_target_id: 当前目标航点索引
    :param waypoints: 航点列表
    :param threshold: 切换航点的距离阈值
    :return: 更新后的目标航点索引
    """
    # 若已是最后一个航点，不再切换
    if current_target_id >= len(waypoints) - 1:
        return current_target_id

    # 获取当前目标航点的位置
    target_wp_data = waypoints[current_target_id]
    target_loc = carla.Location(x=target_wp_data[0], y=target_wp_data[1])

    # 关键修改：手动计算2D距离（替代CARLA低版本不存在的distance_2d方法）
    dx = vehicle_location.x - target_loc.x
    dy = vehicle_location.y - target_loc.y
    distance = np.sqrt(dx ** 2 + dy ** 2)

    # 若距离小于阈值，切换到下一个航点
    if distance < threshold:
        print(f"[航点更新] 到达航点{current_target_id}（距离：{distance:.1f}m），新目标航点：{current_target_id + 1}")
        return current_target_id + 1

    return current_target_id

# ===================== 速度控制器 =====================
def simple_speed_controller(v_desired: float, v_current: float) -> carla.VehicleControl:
    """
    简易的3状态速度控制器（bang-bang控制）
    根据当前速度与期望速度的差值，全量施加油门或刹车
    :param v_desired: 期望速度（km/h）
    :param v_current: 当前速度（km/h）
    :return: CARLA车辆控制指令
    """
    control = carla.VehicleControl()
    control.steer = 0.0  # 暂不控制转向
    control.throttle = 0.0
    control.brake = 0.0

    # 目标速度为0时，全力刹车
    if v_desired == 0:
        control.brake = 1.0
    # 当前速度小于期望速度，全力加速
    elif v_current < v_desired:
        control.throttle = 1.0
    # 当前速度大于期望速度，全力刹车
    elif v_current > v_desired:
        control.brake = 1.0

    return control

# ===================== 主仿真函数 =====================
def main():
    """主仿真入口函数：完成所有仿真流程的初始化、运行和清理"""
    # 初始化核心变量
    client: Optional[carla.Client] = None
    world: Optional[carla.World] = None
    vehicle: Optional[carla.Vehicle] = None
    plotter: Optional[Plotter] = None
    pygame_display: Optional[PygameDisplay] = None
    simulation_start_time: Optional[float] = None
    target_waypoint_id: int = SimConfig.INITIAL_TARGET_WAYPOINT_ID

    try:
        # 1. 连接CARLA服务器
        print(f"[CARLA连接] 尝试连接 {SimConfig.CARLA_HOST}:{SimConfig.CARLA_PORT}...")
        client = carla.Client(SimConfig.CARLA_HOST, SimConfig.CARLA_PORT)
        client.set_timeout(SimConfig.CARLA_TIMEOUT)
        world = client.get_world()
        print(f"[CARLA连接] 成功连接到地图：{world.get_map().name}")

        # 2. 清理历史车辆
        clean_up_prev_vehicles(world, SimConfig.VEHICLE_ROLE_NAME)

        # 3. 生成车辆（基于第一个航点作为起始位置）
        map_spawn_points = world.get_map().get_spawn_points()
        if not map_spawn_points:
            raise RuntimeError("[生成车辆] 地图中未找到可用的生成点！")

        # 获取起始航点的位置
        start_waypoint = SimConfig.WAYPOINTS[0]
        # 关键修改：将z轴高度从0.5提高到1.5，避免与地面碰撞
        start_location = carla.Location(x=start_waypoint[0], y=start_waypoint[1], z=1.5)

        # 找到距离起始位置最近的地图生成点（用于获取道路朝向）
        spawn_point = min(map_spawn_points, key=lambda sp: sp.location.distance(start_location))
        spawn_point.location = start_location  # 覆盖为自定义起始位置

        print(f"[生成车辆] 目标生成位置：{start_location}，使用地图朝向：{spawn_point.rotation}")

        # 加载车辆蓝图
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(SimConfig.VEHICLE_MODEL)[0]
        vehicle_bp.set_attribute("role_name", SimConfig.VEHICLE_ROLE_NAME)

        # 生成车辆
        print("[生成车辆] 尝试生成车辆...")
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        
        # 关键修改：添加重试逻辑，第一个点失败则尝试第二个地图默认点
        if vehicle is None:
            print("[生成车辆] 第一个点生成失败，尝试第二个地图默认点...")
            spawn_point = map_spawn_points[1]
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is None:
                raise RuntimeError(f"[生成车辆] 生成失败，请检查坐标是否合法！")

        print(f"[生成车辆] 成功生成：{vehicle.type_id} (ID: {vehicle.id})")

        # 4. 设置旁观者相机视角
        set_spectator_fixed_view(world)

        # 5. 初始化可视化组件
        print("\n[可视化] 初始化绘图器（Plotter）...")
        plotter = Plotter()
        plotter.init_plot()
        print("[可视化] 绘图器初始化完成")

        print("[可视化] 初始化Pygame显示窗口...")
        pygame_display = PygameDisplay(world, vehicle)
        print("[可视化] Pygame显示窗口初始化完成")

        # 6. 记录仿真开始时间
        simulation_start_time = time.time()
        target_waypoint_id = SimConfig.INITIAL_TARGET_WAYPOINT_ID

        # 7. 主仿真循环
        print("\n[仿真启动] 开始航点跟随与速度控制。按ESC或关闭Pygame窗口停止仿真...")
        while True:
            current_loop_time = time.time()

            # 处理Pygame事件（关闭窗口、ESC键）
            if pygame_display.parse_events():
                print("[仿真控制] 检测到Pygame退出请求，停止仿真...")
                break

            # 等待仿真tick（同步仿真时间）
            world.wait_for_tick()

            # 8. 收集车辆状态数据
            vehicle_location = get_vehicle_location(vehicle)
            vehicle_speed = calculate_vehicle_speed_kmh(vehicle)
            sim_elapsed_time = current_loop_time - simulation_start_time

            # 9. 更新可视化
            # 更新绘图器
            if plotter and plotter.is_initialized:
                current_desired_speed = SimConfig.WAYPOINTS[target_waypoint_id][2]
                try:
                    plotter.update_plot(sim_elapsed_time, vehicle_location.x, vehicle_speed, current_desired_speed)
                except Exception as e:
                    print(f"[绘图器] 更新失败（可能已关闭窗口）：{e}")
                    plotter.cleanup_plot()
                    plotter = None

            # 渲染Pygame窗口
            pygame_display.render()

            # 10. 航点跟随逻辑
            # 更新目标航点
            target_waypoint_id = update_target_waypoint(
                vehicle_location,
                target_waypoint_id,
                SimConfig.WAYPOINTS,
                SimConfig.WAYPOINT_THRESHOLD
            )

            # 获取当前目标航点的期望速度
            current_desired_speed = SimConfig.WAYPOINTS[target_waypoint_id][2] if 0 <= target_waypoint_id < len(SimConfig.WAYPOINTS) else 0.0

            # 计算车辆控制指令
            control = simple_speed_controller(current_desired_speed, vehicle_speed)

            # 应用控制指令
            vehicle.apply_control(control)

    # 异常处理
    except KeyboardInterrupt:
        print("\n[仿真中断] 用户按下Ctrl+C，停止仿真...")
    except RuntimeError as e:
        print(f"\n[仿真错误] 运行时错误：{e}")
        traceback.print_exc()
    except Exception as e:
        print(f"\n[仿真错误] 未知异常：{e}")
        traceback.print_exc()

    # 资源清理
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