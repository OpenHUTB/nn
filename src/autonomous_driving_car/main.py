#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA Waypoint Following（原生道路航点版）
核心：直接使用CARLA地图的道路航点，车辆100%沿道路行驶
"""

import sys
import os
import carla
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import time

# ===================== 核心配置 =====================
class Config:
    # 速度控制
    TARGET_SPEED = 20.0  # km/h
    PID_KP = 0.3
    PID_KI = 0.02
    PID_KD = 0.01

    # 纯追踪算法参数
    LOOKAHEAD_DISTANCE = 8.0  # 前瞻距离（米）
    MAX_STEER_ANGLE = 30.0    # 最大转向角（度）

    # 摄像头配置
    CAMERA_WIDTH = 800
    CAMERA_HEIGHT = 600
    CAMERA_FOV = 90

    # 车辆配置
    VEHICLE_MODEL = "vehicle.tesla.model3"

    # 可视化配置
    PLOT_SIZE = (12, 10)
    WAYPOINT_COUNT = 50  # 预先生成的道路航点数量

# ===================== 工具类 =====================
class Tools:
    @staticmethod
    def normalize_angle(angle):
        """将角度归一化到[-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @staticmethod
    def get_vehicle_pose(vehicle):
        """获取车辆的位置、朝向和速度"""
        transform = vehicle.get_transform()
        loc = transform.location
        yaw = math.radians(transform.rotation.yaw)
        vel = vehicle.get_velocity()
        speed = 3.6 * np.linalg.norm([vel.x, vel.y, vel.z])
        return loc, yaw, speed

    @staticmethod
    def clear_all_actors(world):
        """清理所有车辆和传感器"""
        for actor in world.get_actors():
            try:
                if actor.type_id.startswith('vehicle') or actor.type_id.startswith('sensor'):
                    actor.destroy()
            except:
                pass
        time.sleep(0.5)
        print("已清理所有残留Actor")

    @staticmethod
    def focus_vehicle(world, vehicle):
        """将CARLA客户端视角聚焦到车辆"""
        spectator = world.get_spectator()
        t = vehicle.get_transform()
        spectator.set_transform(carla.Transform(t.location + carla.Location(x=0, y=-8, z=5), t.rotation))
        print("CARLA客户端已聚焦到车辆")

    @staticmethod
    def generate_road_waypoints(world, start_loc, count=50, step=2.0):
        """
        从起点沿道路生成连续的原生航点
        :param world: CARLA世界对象
        :param start_loc: 起点位置
        :param count: 航点数量
        :param step: 每个航点的步长（米）
        :return: 航点列表[(x, y, z), ...]
        """
        waypoints = []
        map = world.get_map()
        wp = map.get_waypoint(start_loc)
        for i in range(count):
            waypoints.append((wp.transform.location.x, wp.transform.location.y, wp.transform.location.z))
            # 沿道路下一个航点（直走，不考虑分叉）
            wp = wp.next(step)[0]
        print(f"生成了{len(waypoints)}个原生道路航点")
        return waypoints

# ===================== 摄像头回调 =====================
def camera_callback(image, data_dict):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    data_dict['image'] = array

# ===================== 控制器 =====================
class PIDSpeedController:
    def __init__(self, config):
        self.kp = config.PID_KP
        self.ki = config.PID_KI
        self.kd = config.PID_KD
        self.prev_err = 0.0
        self.integral = 0.0
        self.target_speed = config.TARGET_SPEED

    def calculate(self, current_speed):
        err = self.target_speed - current_speed
        self.integral = np.clip(self.integral + err * 0.05, -1.0, 1.0)
        deriv = (err - self.prev_err) / 0.05
        output = self.kp * err + self.ki * self.integral + self.kd * deriv
        self.prev_err = err
        return np.clip(output, 0.1, 1.0)

class PurePursuitController:
    def __init__(self, config):
        self.lookahead_dist = config.LOOKAHEAD_DISTANCE
        self.max_steer_rad = math.radians(config.MAX_STEER_ANGLE)

    def calculate_steer(self, vehicle_loc, vehicle_yaw, waypoints):
        """
        纯追踪算法计算转向角
        :param vehicle_loc: 车辆位置
        :param vehicle_yaw: 车辆朝向（弧度）
        :param waypoints: 道路航点列表
        :return: 转向角（-1~1）
        """
        # 1. 将航点转换为车辆坐标系
        wp_coords = np.array(waypoints)
        vehicle_x = vehicle_loc.x
        vehicle_y = vehicle_loc.y

        # 旋转和平移（车辆坐标系：x向前，y向左）
        cos_yaw = math.cos(vehicle_yaw)
        sin_yaw = math.sin(vehicle_yaw)
        translated_x = wp_coords[:, 0] - vehicle_x
        translated_y = wp_coords[:, 1] - vehicle_y
        rotated_x = translated_x * cos_yaw + translated_y * sin_yaw
        rotated_y = -translated_x * sin_yaw + translated_y * cos_yaw

        # 2. 找到距离车辆>=前瞻距离的第一个航点
        distances = np.hypot(rotated_x, rotated_y)
        valid_wp_indices = np.where(distances >= self.lookahead_dist)[0]
        if len(valid_wp_indices) == 0:
            return 0.0

        target_idx = valid_wp_indices[0]
        target_x = rotated_x[target_idx]
        target_y = rotated_y[target_idx]

        # 3. 计算转向角（纯追踪公式：steer = arctan(2*L*y/(x²+y²))，L为车辆轴距，这里简化为1.0）
        L = 1.0  # 车辆轴距（米）
        steer_rad = math.atan2(2 * L * target_y, self.lookahead_dist ** 2)

        # 4. 限制转向角
        steer_rad = np.clip(steer_rad, -self.max_steer_rad, self.max_steer_rad)
        steer = steer_rad / self.max_steer_rad

        return steer

# ===================== 可视化 =====================
class Visualizer:
    def __init__(self, config, spawn_loc, initial_waypoints):
        self.waypoints = np.array(initial_waypoints)
        self.trajectory = []
        self.spawn_loc = (spawn_loc.x, spawn_loc.y)

        plt.rcParams['backend'] = 'TkAgg'
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=config.PLOT_SIZE)

        # 绘制道路航点（CARLA原生）
        self.ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], c='blue', s=50, label='Road Waypoints (CARLA)', zorder=3)
        # 绘制生成点
        self.ax.scatter(self.spawn_loc[0], self.spawn_loc[1], c='orange', marker='s', s=150, label='Spawn Point', zorder=5)
        # 轨迹和车辆
        self.traj_line, = self.ax.plot([], [], c='red', linewidth=4, label='Vehicle Trajectory', zorder=2)
        self.vehicle_dot, = self.ax.plot([], [], c='green', marker='o', markersize=20, label='Vehicle', zorder=6)

        self.ax.set_xlabel('X (m)', fontsize=14)
        self.ax.set_ylabel('Y (m)', fontsize=14)
        self.ax.set_title('CARLA Road Following (Native Waypoints)', fontsize=16)
        self.ax.legend(fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.axis('equal')

        plt.show(block=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, vehicle_x, vehicle_y, new_waypoints=None):
        """更新轨迹和航点"""
        self.trajectory.append([vehicle_x, vehicle_y])
        if len(self.trajectory) > 1000:
            self.trajectory = self.trajectory[-1000:]

        # 更新轨迹
        traj = np.array(self.trajectory)
        self.traj_line.set_data(traj[:, 0], traj[:, 1])
        self.vehicle_dot.set_data(vehicle_x, vehicle_y)

        # 更新航点（如果有新航点）
        if new_waypoints is not None:
            self.waypoints = np.array(new_waypoints)
            self.ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], c='blue', s=50, zorder=3)

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ===================== 主函数 =====================
def main():
    config = Config()
    tools = Tools()

    # 初始化OpenCV窗口
    cv2.namedWindow('CARLA Vehicle View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CARLA Vehicle View', config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    # 连接CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0)
        world = client.load_world('Town03')
        map = world.get_map()

        # 设置同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # 清理Actor
        tools.clear_all_actors(world)

        # 获取道路生成点
        spawn_points = map.get_spawn_points()
        spawn_transform = spawn_points[0]
        print(f"使用道路生成点：{spawn_transform.location}")

        # 生成车辆
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find(config.VEHICLE_MODEL)
        vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
        if not vehicle:
            print("车辆生成失败！")
            return
        print(f"车辆{config.VEHICLE_MODEL}生成成功")

        # 聚焦车辆
        tools.focus_vehicle(world, vehicle)

        # 挂载摄像头
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(config.CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(config.CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(config.CAMERA_FOV))
        camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=2.0, z=1.8)), attach_to=vehicle)

        # 摄像头数据
        camera_data = {'image': np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)}
        camera.listen(lambda img: camera_callback(img, camera_data))

        # 生成初始道路航点
        initial_waypoints = tools.generate_road_waypoints(world, spawn_transform.location, config.WAYPOINT_COUNT)

        # 初始化控制器
        speed_controller = PIDSpeedController(config)
        path_controller = PurePursuitController(config)

        # 初始化可视化
        visualizer = Visualizer(config, spawn_transform.location, initial_waypoints)

        # 主循环
        while True:
            world.tick()

            # 获取车辆状态
            vehicle_loc, vehicle_yaw, current_speed = tools.get_vehicle_pose(vehicle)

            # 实时更新道路航点（每10帧更新一次，减少计算量）
            if world.get_snapshot().frame % 10 == 0:
                new_waypoints = tools.generate_road_waypoints(world, vehicle_loc, config.WAYPOINT_COUNT)
            else:
                new_waypoints = None

            # 更新可视化
            visualizer.update(vehicle_loc.x, vehicle_loc.y, new_waypoints)

            # 显示摄像头画面
            cv2.imshow('CARLA Vehicle View', camera_data['image'])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 计算控制量
            steer = path_controller.calculate_steer(vehicle_loc, vehicle_yaw, initial_waypoints)
            throttle = speed_controller.calculate(current_speed)
            brake = 1.0 if current_speed > config.TARGET_SPEED * 2 else 0.0

            # 控制车辆
            vehicle.apply_control(carla.VehicleControl(
                throttle=throttle,
                brake=brake,
                steer=steer,
                hand_brake=False,
                reverse=False
            ))

            # 打印状态
            print(f"速度：{current_speed:.1f}km/h | 位置：({vehicle_loc.x:.1f}, {vehicle_loc.y:.1f}) | 转向：{steer:.2f}", end='\r')

    except Exception as e:
        print(f"\n程序异常：{e}")
    finally:
        # 清理资源
        print("\n清理资源中...")
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if 'vehicle' in locals():
            vehicle.destroy()
        if 'camera' in locals():
            camera.destroy()
        cv2.destroyAllWindows()
        plt.close('all')
        time.sleep(1)
        print("仿真结束")

if __name__ == '__main__':
    main()