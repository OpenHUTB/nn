# -*- coding: utf-8 -*-
import carla
import random
import time
import sys
import os

# 将根目录加入系统路径，确保能导入 config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


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


if __name__ == "__main__":
    from sensor_manager import SensorManager
    from object_detector import YOLOv3Detector  # 导入新模块
    import cv2

    connector = CarlaClient()
    sensors = None
    try:
        connector.connect()
        vehicle = connector.spawn_ego_vehicle()

        # 初始化传感器和检测器
        sensors = SensorManager(connector.world, vehicle)
        sensors.attach_camera()
        detector = YOLOv3Detector()

        print("系统就绪，开始实时检测。按下 'q' 键退出...")
        while True:
            frame = sensors.get_current_frame()
            if frame is not None:
                # 1. 执行目标检测
                detections = detector.detect(frame)

                # 2. 绘制检测结果
                frame = detector.draw_labels(frame, detections)

                # 3. 显示画面
                cv2.imshow("CARLA AutoVision - YOLOv3 Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if sensors:
            sensors.cleanup()
        connector.cleanup()
        cv2.destroyAllWindows()