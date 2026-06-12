import carla
import sys
import time

class SpeedLimitDetector:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()
        
        self.current_speed_limit = 50.0
        self.detected_speed_limit = None
        self.target_speed = 50.0
        
        self.smoothing_factor = 0.1
        
    def detect_speed_limit(self):
        """检测当前位置的限速"""
        vehicle_location = self.vehicle.get_transform().location
        
        # 使用车辆的get_speed_limit方法（如果可用）
        try:
            speed_limit = self.vehicle.get_speed_limit()
            if speed_limit > 0:
                self.detected_speed_limit = speed_limit
                return speed_limit
        except AttributeError:
            pass
        
        # 备选方案：使用路点的lane_id来估算限速
        waypoint = self.map.get_waypoint(vehicle_location)
        if waypoint is not None:
            lane_type = waypoint.lane_type
            
            # 根据车道类型估算限速
            if lane_type == carla.LaneType.Driving:
                return 50.0
            elif lane_type == carla.LaneType.Motorway:
                return 80.0
            elif lane_type == carla.LaneType.Bus:
                return 30.0
            else:
                return 40.0
        
        return None
    
    def update(self):
        """更新限速检测和目标速度"""
        detected = self.detect_speed_limit()
        
        if detected is not None:
            self.current_speed_limit = (1 - self.smoothing_factor) * self.current_speed_limit + \
                                      self.smoothing_factor * detected
            self.target_speed = self.current_speed_limit
            
        return self.target_speed
    
    def get_current_limit(self):
        """获取当前检测到的限速"""
        return self.current_speed_limit
    
    def get_target_speed(self):
        """获取目标速度"""
        return self.target_speed

def spawn_vehicle(world, blueprint_library):
    """尝试在多个生成点生成车辆"""
    tesla_bp = blueprint_library.find("vehicle.tesla.model3")
    tesla_bp.set_attribute("color", "0, 0, 0")
    
    spawn_points = world.get_map().get_spawn_points()
    
    for i, spawn_point in enumerate(spawn_points[:10]):
        try:
            vehicle = world.spawn_actor(tesla_bp, spawn_point)
            print(f"[INFO] Vehicle spawned at spawn point {i}")
            return vehicle
        except RuntimeError:
            continue
    
    print("[ERROR] All spawn points are occupied")
    return None

def main():
    print("=" * 60)
    print("CARLA - Speed Limit Recognition System")
    print("=" * 60)
    
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        print("[INFO] Connected to CARLA server")
        
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        vehicle = spawn_vehicle(world, blueprint_library)
        
        if not vehicle:
            print("[ERROR] Failed to spawn vehicle")
            sys.exit(1)
        
        speed_limit_detector = SpeedLimitDetector(vehicle)
        
        # 启用自动驾驶
        vehicle.set_autopilot(True)
        
        print("[INFO] Speed limit recognition system activated")
        print("[INFO] Press Ctrl+C to stop")
        
        try:
            while True:
                target_speed = speed_limit_detector.update()
                current_speed = ((vehicle.get_velocity().x**2 + 
                                vehicle.get_velocity().y**2 + 
                                vehicle.get_velocity().z**2) ** 0.5) * 3.6
                
                print(f"\r[INFO] Current Speed: {current_speed:5.1f} km/h | "
                      f"Speed Limit: {speed_limit_detector.get_current_limit():5.0f} km/h | "
                      f"Target: {target_speed:5.1f} km/h", end="")
                
                time.sleep(0.1)
                
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