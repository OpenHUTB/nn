# -*- coding: utf-8 -*-
"""
Project: CARLA AutoVision Navigator
Module: Main Entry & Environment Management
Version: v1.0.0
Description: 系统主入口。负责连接 CARLA 服务器、生成主车实体、集成感知与控制循环及 UI 渲染。
Author: wangadsa
License: MIT License
"""

import carla
import sys
import os
import cv2
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.geometry import get_speed, get_steer_angle


class CarlaClient:
    """Manages CARLA server connection and vehicle actor lifecycle."""

    def __init__(self):
        """Initializes the client attributes."""
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.ego_vehicle = None

    def connect(self):
        """Establishes connection to the CARLA simulator server."""
        try:
            print(f"Connecting to CARLA Server: {config.CARLA_HOST}:{config.CARLA_PORT}")
            self.client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
            self.client.set_timeout(config.TIMEOUT)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            print("Connected successfully.")
        except Exception as e:
            print(f"Failed to connect: {e}")
            sys.exit(1)

    def spawn_ego_vehicle(self):
        """Spawns the ego vehicle at the configured spawn point."""
        bp = self.blueprint_library.find(config.VEHICLE_FILTER)
        bp.set_attribute('role_name', 'ego')
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[config.SPAWN_POINT_INDEX] if config.SPAWN_POINT_INDEX < len(
            spawn_points) else random.choice(spawn_points)
        self.ego_vehicle = self.world.spawn_actor(bp, spawn_point)
        return self.ego_vehicle

    def cleanup(self):
        """Destroys actors and cleans up the simulation environment."""
        if self.ego_vehicle and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
            print("Environment cleaned up.")


if __name__ == "__main__":
    from sensor_manager import SensorManager
    from object_detector import YOLOv3Detector
    from pid_controller import PIDController
    from decision_maker import DecisionMaker

    connector = CarlaClient()
    sensors = None
    try:
        connector.connect()
        vehicle = connector.spawn_ego_vehicle()
        carla_map = connector.world.get_map()

        sensors = SensorManager(connector.world, vehicle)
        sensors.attach_camera()
        detector = YOLOv3Detector()
        decision = DecisionMaker()

        speed_pid = PIDController(config.K_P_SPEED, config.K_I_SPEED, config.K_D_SPEED)
        steer_pid = PIDController(config.K_P_STEER, config.K_I_STEER, config.K_D_STEER)

        last_time = time.time()
        print("System fully integrated. Monitoring performance...")

        while True:
            # Performance Monitoring (FPS)
            start_time = time.time()
            fps = 1.0 / (start_time - last_time + 1e-6)
            last_time = start_time

            # 1. Perception
            frame = sensors.get_current_frame()
            detections = detector.detect(frame) if frame is not None else []

            # 2. Decision
            is_emergency = decision.process_detections(detections, config.CAMERA_HEIGHT)

            # 3. Control
            current_speed = get_speed(vehicle)
            waypoint = carla_map.get_waypoint(vehicle.get_location()).next(8.0)[0]
            angle_error = get_steer_angle(vehicle, waypoint)

            target_v = 0.0 if is_emergency else config.TARGET_SPEED
            speed_signal = speed_pid.run_step(target_v, current_speed)
            steer_signal = steer_pid.run_step(0, -angle_error)

            control = vehicle.get_control()
            if is_emergency:
                control.throttle, control.brake = 0.0, 1.0
            else:
                control.throttle = max(0, speed_signal)
                control.brake = abs(min(0, speed_signal))

            control.steer = steer_signal
            vehicle.apply_control(control)

            # 4. Visualization & UI
            if frame is not None:
                frame = detector.draw_labels(frame, detections)
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 90), 1, 1.5, (255, 0, 0), 2)
                if is_emergency:
                    cv2.putText(frame, "AEB ACTIVE", (250, 300), 1, 3, (0, 0, 255), 4)
                cv2.imshow("CARLA AutoVision v1.0-RC", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if sensors: sensors.cleanup()
        connector.cleanup()
        cv2.destroyAllWindows()