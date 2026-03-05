# -*- coding: utf-8 -*-
import carla
import random
import time
import sys
import os
import math
import cv2

# 将根目录加入系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


# --- 辅助工具函数 ---
def get_speed(vehicle):
    """
    计算车辆当前速度 (km/h)
    """
    v = vehicle.get_velocity()
    # 计算三轴合速度并从 m/s 转换为 km/h
    return 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


class CarlaClient:
    """
    负责管理与 CARLA 服务器的通信以及车辆实体的生命周期
    """

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

        # 初始化传感器、检测器与控制器
        sensors = SensorManager(connector.world, vehicle)
        sensors.attach_camera()
        detector = YOLOv3Detector()
        pid = PIDController(config.K_P_SPEED, config.K_I_SPEED, config.K_D_SPEED)

        print(f"系统就绪。目标速度: {config.TARGET_SPEED} km/h. 按下 'q' 键退出...")

        while True:
            # 1. 获取当前状态
            current_speed = get_speed(vehicle)
            frame = sensors.get_current_frame()

            # 2. 纵向速度控制逻辑
            control_signal = pid.run_step(config.TARGET_SPEED, current_speed)
            control = vehicle.get_control()

            if control_signal >= 0:
                control.throttle = control_signal
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = abs(control_signal)

            control.steer = 0.0  # 初始阶段保持直行
            vehicle.apply_control(control)

            # 3. 视觉感知与目标检测
            if frame is not None:
                detections = detector.detect(frame)
                frame = detector.draw_labels(frame, detections)
                # 在画面上额外显示当前车速
                cv2.putText(frame, f"Speed: {current_speed:.2f} km/h", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("CARLA AutoVision - Control & Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if sensors: sensors.cleanup()
        connector.cleanup()
        cv2.destroyAllWindows()