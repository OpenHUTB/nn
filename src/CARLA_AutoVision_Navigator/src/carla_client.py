# -*- coding: utf-8 -*-
import carla
import random
import time
import sys
import os

# 将根目录加入系统路径，确保能导入 config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import math


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
        """建立与 CARLA 模拟器的连接"""
        try:
            print(f"正在尝试连接至 CARLA Server: {config.CARLA_HOST}:{config.CARLA_PORT}...")
            self.client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
            self.client.set_timeout(config.TIMEOUT)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            print("连接状态: 成功!")
        except Exception as e:
            print(f"连接状态: 失败! 请检查模拟器是否启动。错误信息: {e}")
            sys.exit(1)

    def spawn_ego_vehicle(self):
        """根据配置在地图上生成主车"""
        # 1. 查找指定的车型
        bp = self.blueprint_library.find(config.VEHICLE_FILTER)
        bp.set_attribute('role_name', 'ego')

        # 2. 获取地图上的生成点
        spawn_points = self.world.get_map().get_spawn_points()

        # 3. 逻辑选择：优先使用配置索引，否则随机
        if 0 <= config.SPAWN_POINT_INDEX < len(spawn_points):
            spawn_point = spawn_points[config.SPAWN_POINT_INDEX]
        else:
            spawn_point = random.choice(spawn_points)

        # 4. 生成车辆
        self.ego_vehicle = self.world.spawn_actor(bp, spawn_point)
        print(f"车辆生成成功: ID[{self.ego_vehicle.id}] 型号[{self.ego_vehicle.type_id}]")
        return self.ego_vehicle

    def cleanup(self):
        """释放资源，销毁生成的 Actor"""
        if self.ego_vehicle and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
            print("已销毁主车，资源清理完毕。")


    def get_speed(vehicle):
        """计算车辆当前速度 (km/h)"""
        v = vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


if __name__ == "__main__":
    from sensor_manager import SensorManager
    from object_detector import YOLOv3Detector
    from pid_controller import PIDController  # 导入新模块
    import cv2

    connector = CarlaClient()
    sensors = None
    try:
        connector.connect()
        vehicle = connector.spawn_ego_vehicle()

        sensors = SensorManager(connector.world, vehicle)
        sensors.attach_camera()
        detector = YOLOv3Detector()

        # 初始化 PID 控制器
        pid = PIDController(config.K_P_SPEED, config.K_I_SPEED, config.K_D_SPEED)

        print(f"系统就绪。目标速度: {config.TARGET_SPEED} km/h. 按下 'q' 键退出...")
        while True:
            # 1. 获取当前状态
            current_speed = get_speed(vehicle)
            frame = sensors.get_current_frame()

            # 2. 纵向控制 (速度控制)
            control_signal = pid.run_step(config.TARGET_SPEED, current_speed)
            control = vehicle.get_control()

            if control_signal >= 0:
                control.throttle = control_signal
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = abs(control_signal)

            # 这里暂时让车直走，后面会加横向控制
            control.steer = 0.0
            vehicle.apply_control(control)

            # 3. 感知处理 (目标检测)
            if frame is not None:
                detections = detector.detect(frame)
                frame = detector.draw_labels(frame, detections)
                cv2.imshow("CARLA AutoVision - Control & Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if sensors:
            sensors.cleanup()
        connector.cleanup()
        cv2.destroyAllWindows()