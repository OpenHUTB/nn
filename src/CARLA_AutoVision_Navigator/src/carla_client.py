# -*- coding: utf-8 -*-
import carla
import sys
import os
import cv2
import math
import random

# 将根目录加入系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.geometry import get_speed, get_steer_angle


class CarlaClient:
    """负责管理与 CARLA 服务器的通信以及车辆实体的生命周期"""

    def __init__(self):
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.ego_vehicle = None

    def connect(self):
        try:
            print(f"正在尝试连接至 CARLA Server: {config.CARLA_HOST}:{config.CARLA_PORT}...")
            self.client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
            self.client.set_timeout(config.TIMEOUT)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            print("连接状态: 成功!")
        except Exception as e:
            print(f"连接状态: 失败! {e}")
            sys.exit(1)

    def spawn_ego_vehicle(self):
        bp = self.blueprint_library.find(config.VEHICLE_FILTER)
        bp.set_attribute('role_name', 'ego')
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[config.SPAWN_POINT_INDEX] if config.SPAWN_POINT_INDEX < len(
            spawn_points) else random.choice(spawn_points)
        self.ego_vehicle = self.world.spawn_actor(bp, spawn_point)
        print(f"车辆生成成功: ID[{self.ego_vehicle.id}]")
        return self.ego_vehicle

    def cleanup(self):
        if self.ego_vehicle and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
            print("已销毁主车，资源清理完毕。")


if __name__ == "__main__":
    from sensor_manager import SensorManager
    from object_detector import YOLOv3Detector
    from pid_controller import PIDController

    connector = CarlaClient()
    sensors = None
    try:
        connector.connect()
        vehicle = connector.spawn_ego_vehicle()
        carla_map = connector.world.get_map()

        # 初始化感知、检测与双控制器
        sensors = SensorManager(connector.world, vehicle)
        sensors.attach_camera()
        detector = YOLOv3Detector()
        speed_pid = PIDController(config.K_P_SPEED, config.K_I_SPEED, config.K_D_SPEED)
        steer_pid = PIDController(config.K_P_STEER, config.K_I_STEER, config.K_D_STEER)

        print("\n全系统启动：感知 + 控速 + 自动转向。按下 'q' 键退出...")
        while True:
            # 1. 获取当前状态与导航点
            current_loc = vehicle.get_location()
            waypoint = carla_map.get_waypoint(current_loc).next(8.0)[0]
            current_speed = get_speed(vehicle)
            angle_error = get_steer_angle(vehicle, waypoint)

            # 2. 控制指令计算 (纵向+横向)
            speed_signal = speed_pid.run_step(config.TARGET_SPEED, current_speed)
            steer_signal = steer_pid.run_step(0, -angle_error)

            control = vehicle.get_control()
            if speed_signal >= 0:
                control.throttle, control.brake = speed_signal, 0.0
            else:
                control.throttle, control.brake = 0.0, abs(speed_signal)

            control.steer = max(-1.0, min(1.0, steer_signal))
            vehicle.apply_control(control)

            # 3. 感知可视化
            frame = sensors.get_current_frame()
            if frame is not None:
                detections = detector.detect(frame)
                frame = detector.draw_labels(frame, detections)
                cv2.putText(frame, f"Speed: {current_speed:.1f} km/h", (10, 30), 1, 1.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Steer: {control.steer:.2f}", (10, 60), 1, 1.5, (255, 255, 0), 2)
                cv2.imshow("CARLA AutoVision Integration", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if sensors: sensors.cleanup()
        connector.cleanup()
        cv2.destroyAllWindows()