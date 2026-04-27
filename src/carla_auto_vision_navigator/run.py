import time
from src.carla_client import CarlaClient
from src.sensor_manager import SensorManager
from src.object_detector import MultimodalAnomalyDetector
from src.decision_maker import AnomalyDecisionMaker
from src.pid_controller import PIDController


def main():
    # 1. 初始化模块
    carla_client = CarlaClient()
    if not carla_client.connect():
        return

    # 2. 加载非结构化场景
    carla_client.load_unstructured_scene("town07")

    # 3. 生成车辆和异常Actor
    vehicle = carla_client.spawn_vehicle()
    carla_client.spawn_anomaly_actors("town07")

    # 4. 初始化传感器
    sensor_manager = SensorManager(carla_client, vehicle)
    sensor_manager.setup_sensors()
    time.sleep(2)  # 等待传感器初始化

    # 5. 初始化异常检测器
    detector = MultimodalAnomalyDetector()

    # 6. 初始化决策器和控制器
    decision_maker = AnomalyDecisionMaker(vehicle)
    pid_controller = PIDController()

    # 7. 主循环
    try:
        while True:
            # 获取传感器数据
            sensor_data = sensor_manager.get_sensor_data()

            # 异常检测
            anomaly_result = detector.detect_anomaly(sensor_data)

            # 生成决策
            decision, action = decision_maker.make_decision(anomaly_result)

            # 控制车辆
            pid_controller.control_vehicle(vehicle, action)

            # 可视化（可选）
            # sensor_manager.visualize_data()

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("程序终止")
    finally:
        # 清理资源
        sensor_manager.clean_up()
        carla_client.clean_up()


if __name__ == "__main__":
    main()