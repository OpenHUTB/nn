import carla
import sys
import time
import math

class LaneKeeping:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()
        
        # PID控制器参数
        self.kp = 0.5
        self.ki = 0.1
        self.kd = 0.2
        
        # 状态变量
        self.integral = 0.0
        self.previous_error = 0.0
        self.target_speed = 30.0  # km/h
        
        # 车道保持状态
        self.enabled = True
        self.lane_offset = 0.0
        self.steer_angle = 0.0

    def get_lane_center_offset(self):
        """计算车辆与车道中心线的横向偏移"""
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        
        waypoint = self.map.get_waypoint(vehicle_location)
        
        if waypoint is None:
            return 0.0
        
        lane_center = waypoint.transform.location
        
        forward_vector = vehicle_transform.get_forward_vector()
        right_vector = carla.Vector3D(-forward_vector.y, forward_vector.x, 0)
        
        offset_vector = lane_center - vehicle_location
        self.lane_offset = offset_vector.dot(right_vector)
        
        return self.lane_offset

    def calculate_steer(self, dt):
        """使用PID控制器计算转向角度"""
        error = self.get_lane_center_offset()
        
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        
        steer = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.previous_error = error
        self.steer_angle = max(-1.0, min(1.0, steer))
        
        return self.steer_angle

    def update(self, dt=0.05):
        """更新车道保持控制"""
        if not self.enabled:
            return carla.VehicleControl()
        
        steer = self.calculate_steer(dt)
        
        speed = self.get_current_speed()
        throttle = 0.0
        brake = 0.0
        
        if speed < self.target_speed:
            throttle = 0.3
        elif speed > self.target_speed + 5:
            brake = 0.2
        
        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake
        )

    def get_current_speed(self):
        """获取当前车速（km/h）"""
        velocity = self.vehicle.get_velocity()
        return ((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5) * 3.6

    def set_target_speed(self, speed):
        """设置目标速度"""
        self.target_speed = speed

    def enable(self):
        """启用车道保持"""
        self.enabled = True

    def disable(self):
        """禁用车道保持"""
        self.enabled = False

def main():
    print("=" * 60)
    print("CARLA - Lane Keeping Assist System")
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
        
        lane_keeping = LaneKeeping(vehicle)
        
        print("[INFO] Vehicle spawned")
        print("[INFO] Lane Keeping Assist enabled")
        print(f"[INFO] Target speed: {lane_keeping.target_speed} km/h")
        print("[INFO] Press Ctrl+C to stop")
        
        previous_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                dt = current_time - previous_time
                previous_time = current_time
                
                control = lane_keeping.update(dt)
                vehicle.apply_control(control)
                
                speed = lane_keeping.get_current_speed()
                offset = lane_keeping.lane_offset
                steer = lane_keeping.steer_angle
                
                print(f"\r[INFO] Speed: {speed:.1f} km/h | Offset: {offset:.2f}m | Steer: {steer:.2f}", end="")
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted")
        finally:
            print("\n[INFO] Cleaning up...")
            vehicle.destroy()
            print("[INFO] Done")
            
    except RuntimeError as e:
        print(f"[ERROR] Runtime error: {e}")
        print("[INFO] Make sure CARLA server is running")
        sys.exit(1)

if __name__ == "__main__":
    main()