import carla
import sys
import time

class VehicleDashboard:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        
        # 车辆状态数据
        self.speed = 0.0
        self.engine_rpm = 0.0
        self.engine_temperature = 85.0
        self.fuel_level = 100.0
        self.oil_pressure = 3.5
        self.battery_level = 100.0
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        
        # 里程计
        self.odometer = 0.0
        self.last_position = vehicle.get_transform().location
        
        # 时间跟踪
        self.start_time = time.time()
        self.current_time = 0.0
        
        # 初始化传感器
        self.setup_sensors()

    def setup_sensors(self):
        """设置车辆传感器"""
        blueprint_library = self.world.get_blueprint_library()
        
        # IMU传感器
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.05')
        self.imu_sensor = self.world.spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(z=1.0)),
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        
        # GPS传感器
        gps_bp = blueprint_library.find('sensor.other.gnss')
        gps_bp.set_attribute('sensor_tick', '0.1')
        self.gps_sensor = self.world.spawn_actor(
            gps_bp,
            carla.Transform(carla.Location(z=1.0)),
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )

    def update(self):
        """更新所有车辆状态"""
        # 更新速度
        velocity = self.vehicle.get_velocity()
        self.speed = ((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5) * 3.6
        
        # 更新里程
        current_position = self.vehicle.get_transform().location
        distance = current_position.distance(self.last_position)
        self.odometer += distance / 1000  # 转换为公里
        self.last_position = current_position
        
        # 更新发动机参数（模拟）
        self.engine_rpm = min(6000, max(800, self.speed * 100 + 800))
        self.engine_temperature = min(105, 85 + self.speed / 10)
        
        # 更新油量（模拟消耗）
        if self.speed > 10:
            self.fuel_level = max(0, self.fuel_level - 0.001 * self.speed)
        
        # 更新电池电量（模拟）
        if self.speed > 0:
            self.battery_level = max(0, self.battery_level - 0.0005 * self.speed)
        
        # 更新控制状态
        control = self.vehicle.get_control()
        self.throttle = control.throttle
        self.brake = control.brake
        self.steer = control.steer
        
        # 更新时间
        self.current_time = time.time() - self.start_time

    def get_status(self):
        """获取所有状态数据"""
        return {
            'speed': self.speed,
            'rpm': self.engine_rpm,
            'temperature': self.engine_temperature,
            'fuel': self.fuel_level,
            'oil_pressure': self.oil_pressure,
            'battery': self.battery_level,
            'throttle': self.throttle,
            'brake': self.brake,
            'steer': self.steer,
            'odometer': self.odometer,
            'time': self.current_time
        }

    def display(self):
        """显示仪表盘"""
        status = self.get_status()
        
        # 格式化时间
        hours = int(status['time'] // 3600)
        minutes = int((status['time'] % 3600) // 60)
        seconds = int(status['time'] % 60)
        
        print("\n" + "="*60)
        print("                    VEHICLE DASHBOARD                    ")
        print("="*60)
        
        # 速度表
        speed_bar = self._get_progress_bar(status['speed'], 120, '|')
        print(f" Speed: {status['speed']:5.1f} km/h  [{speed_bar}]")
        
        # 发动机转速
        rpm_bar = self._get_progress_bar(status['rpm'], 6000, '#')
        print(f" RPM:   {status['rpm']:5.0f}        [{rpm_bar}]")
        
        # 发动机温度
        temp_color = 'GREEN' if status['temperature'] < 95 else 'YELLOW' if status['temperature'] < 100 else 'RED'
        print(f" Engine Temp: {status['temperature']:4.1f} C [{temp_color}]")
        
        # 油量
        fuel_bar = self._get_progress_bar(status['fuel'], 100, '█')
        print(f" Fuel:   {status['fuel']:5.1f}% [{fuel_bar}]")
        
        # 电池电量
        battery_bar = self._get_progress_bar(status['battery'], 100, '▓')
        print(f" Battery:{status['battery']:5.1f}% [{battery_bar}]")
        
        # 油压
        print(f" Oil Pressure: {status['oil_pressure']:4.1f} bar")
        
        # 控制状态
        print(f" Throttle: {status['throttle']:5.1f} | Brake: {status['brake']:5.1f} | Steer: {status['steer']:5.2f}")
        
        # 里程和时间
        print(f" Odometer: {status['odometer']:6.2f} km | Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        print("="*60)

    def _get_progress_bar(self, value, max_value, symbol='█'):
        """生成进度条"""
        percentage = min(100, (value / max_value) * 100)
        filled = int(percentage / 5)
        empty = 20 - filled
        return f"{symbol * filled}{' ' * empty}"

    def destroy(self):
        """销毁传感器"""
        if hasattr(self, 'imu_sensor'):
            self.imu_sensor.destroy()
        if hasattr(self, 'gps_sensor'):
            self.gps_sensor.destroy()

def main():
    print("=" * 60)
    print("CARLA - Vehicle Status Dashboard")
    print("=" * 60)
    
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        print("[INFO] Connected to CARLA server")
        
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        tesla_bp = blueprint_library.find("vehicle.tesla.model3")
        tesla_bp.set_attribute("color", "0, 0, 0")
        
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(tesla_bp, spawn_points[0])
        
        dashboard = VehicleDashboard(vehicle)
        
        # 启用自动驾驶
        vehicle.set_autopilot(True)
        
        print("[INFO] Vehicle spawned")
        print("[INFO] Dashboard activated")
        print("[INFO] Autopilot enabled")
        print("[INFO] Press Ctrl+C to stop")
        
        try:
            while True:
                dashboard.update()
                dashboard.display()
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted")
        finally:
            print("\n[INFO] Cleaning up...")
            dashboard.destroy()
            vehicle.destroy()
            print("[INFO] Done")
            
    except RuntimeError as e:
        print(f"[ERROR] Runtime error: {e}")
        print("[INFO] Make sure CARLA server is running")
        sys.exit(1)

if __name__ == "__main__":
    main()