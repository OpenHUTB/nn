"""
CARLA车辆传感器数据收集模块
"""
import carla
import time


class SensorCollector:
    """车辆传感器数据收集器"""

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.sensors = {}
        self.data = {
            'velocity': 0.0,
            'location': (0.0, 0.0, 0.0),
            'acceleration': (0.0, 0.0, 0.0),
            'steer': 0.0,
            'throttle': 0.0,
            'brake': 0.0
        }

    def setup_sensors(self):
        """配置传感器"""
        blueprint_library = self.world.get_blueprint_library()

        # IMU传感器
        imu_bp = blueprint_library.find("sensor.other.imu")
        imu_transform = carla.Transform(carla.Location(x=0.8, z=0.5))
        self.sensors['imu'] = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)
        self.sensors['imu'].listen(self._imu_callback)

        # 速度传感器（通过轮速传感器）
        speed_bp = blueprint_library.find("sensor.other.speedometer")
        self.sensors['speed'] = self.world.spawn_actor(speed_bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors['speed'].listen(self._speed_callback)

        print("[SENSOR] 传感器配置完成")

    def _imu_callback(self, imu_data):
        """IMU数据回调"""
        self.data['acceleration'] = (
            imu_data.accelerometer.x,
            imu_data.accelerometer.y,
            imu_data.accelerometer.z
        )

    def _speed_callback(self, speed_data):
        """速度数据回调"""
        self.data['velocity'] = speed_data.velocity

    def update_vehicle_state(self):
        """更新车辆状态"""
        control = self.vehicle.get_control()
        self.data['steer'] = control.steer
        self.data['throttle'] = control.throttle
        self.data['brake'] = control.brake

        location = self.vehicle.get_location()
        self.data['location'] = (location.x, location.y, location.z)

    def get_data(self):
        """获取所有传感器数据"""
        self.update_vehicle_state()
        return self.data.copy()

    def display_data(self):
        """显示传感器数据"""
        data = self.get_data()
        print("\r" + " " * 100, end="")
        print(f"\r[DATA] 速度: {data['velocity']:5.1f} km/h | "
              f"位置: ({data['location'][0]:6.1f}, {data['location'][1]:6.1f}) | "
              f"转向: {data['steer']:+.2f} | "
              f"油门: {data['throttle']:.2f} | "
              f"刹车: {data['brake']:.2f}", end="")

    def destroy(self):
        """销毁所有传感器"""
        for sensor in self.sensors.values():
            sensor.destroy()
        print("\n[SENSOR] 传感器已销毁")
