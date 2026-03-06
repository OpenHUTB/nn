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
    from decision_maker import DecisionMaker  # 引入决策器

    connector = CarlaClient()
    sensors = None
    try:
        connector.connect()
        vehicle = connector.spawn_ego_vehicle()
        carla_map = connector.world.get_map()

        sensors = SensorManager(connector.world, vehicle)
        sensors.attach_camera()
        detector = YOLOv3Detector()
        decision = DecisionMaker()  # 初始化决策器

        speed_pid = PIDController(config.K_P_SPEED, config.K_I_SPEED, config.K_D_SPEED)
        steer_pid = PIDController(config.K_P_STEER, config.K_I_STEER, config.K_D_STEER)

        print("全系统启动：感知+决策+控制。按下 'q' 键退出...")
        while True:
            frame = sensors.get_current_frame()
            current_speed = get_speed(vehicle)

            # 1. 感知与检测
            detections = []
            if frame is not None:
                detections = detector.detect(frame)
                frame = detector.draw_labels(frame, detections)

            # 2. 决策层分析
            # 根据检测结果判断是否需要紧急刹车
            is_emergency = decision.process_detections(detections, config.CAMERA_HEIGHT)

            # 3. 控制信号计算
            waypoint = carla_map.get_waypoint(vehicle.get_location()).next(8.0)[0]
            angle_error = get_steer_angle(vehicle, waypoint)

            # 如果是紧急情况，目标速度设为 0
            target_v = 0.0 if is_emergency else config.TARGET_SPEED
            speed_signal = speed_pid.run_step(target_v, current_speed)
            steer_signal = steer_pid.run_step(0, -angle_error)

            # 4. 应用控制
            control = vehicle.get_control()
            if is_emergency:
                control.throttle = 0.0
                control.brake = 1.0  # 全力刹车
            elif speed_signal >= 0:
                control.throttle = speed_signal
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = abs(speed_signal)

            control.steer = steer_signal
            vehicle.apply_control(control)

            # 5. UI 显示
            if frame is not None:
                if is_emergency:
                    cv2.putText(frame, "EMERGENCY BRAKE!", (250, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                cv2.imshow("CARLA Full System Integration", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if sensors: sensors.cleanup()
        connector.cleanup()
        cv2.destroyAllWindows()