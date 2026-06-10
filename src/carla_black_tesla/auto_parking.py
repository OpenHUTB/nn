import carla
import sys
import time
import math

class RoadEdgeParking:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()
        
        # 停车状态
        self.state = 'driving'  # driving, slowing_down, parking, parked
        self.edge_distance = 0.0
        self.target_edge = 3.0  # 靠边距离（米）
        
        # 停车完成标志
        self.parking_completed = False
        self.parking_start_time = 0

    def detect_road_edge(self):
        """检测道路边缘"""
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        
        # 获取道路边界
        waypoint = self.map.get_waypoint(vehicle_location)
        
        if waypoint is None:
            return 0.0
        
        # 计算到道路边界的距离
        lane_width = waypoint.lane_width
        lane_center = waypoint.transform.location
        
        # 计算横向偏移
        forward_vector = vehicle_transform.get_forward_vector()
        right_vector = carla.Vector3D(-forward_vector.y, forward_vector.x, 0)
        offset_vector = lane_center - vehicle_location
        lateral_offset = offset_vector.dot(right_vector)
        
        # 到边缘的距离 = 车道宽度/2 + 横向偏移
        edge_distance = (lane_width / 2) - lateral_offset
        
        return edge_distance

    def update(self):
        """更新停车状态"""
        if self.parking_completed:
            return carla.VehicleControl(throttle=0, brake=1.0, steer=0, hand_brake=True)
        
        self.edge_distance = self.detect_road_edge()
        vehicle_transform = self.vehicle.get_transform()
        speed = self.get_current_speed()
        
        if self.state == 'driving':
            # 检测是否可以靠边停车
            if self.edge_distance < 2.0:
                print(f"[PARKING] Found edge: {self.edge_distance:.2f}m")
                self.state = 'slowing_down'
            
            # 正常行驶
            return carla.VehicleControl(throttle=0.3, brake=0, steer=0)
        
        elif self.state == 'slowing_down':
            # 减速并靠边
            if speed > 15:
                return carla.VehicleControl(throttle=0.1, brake=0.3, steer=0)
            else:
                print("[PARKING] Slowing down and pulling over...")
                self.state = 'parking'
            
        elif self.state == 'parking':
            # 靠边并停车
            if speed > 0.5:
                # 慢慢靠边
                steer_adjustment = -0.1  # 向右微调
                return carla.VehicleControl(throttle=0.05, brake=0.1, steer=steer_adjustment)
            else:
                # 停车
                self.parking_completed = True
                self.parking_start_time = time.time()
                print("[PARKING] Parked successfully!")
                return carla.VehicleControl(throttle=0, brake=1.0, steer=0, hand_brake=True)
        
        return carla.VehicleControl(throttle=0.3, brake=0, steer=0)

    def get_current_speed(self):
        """获取当前速度（km/h）"""
        velocity = self.vehicle.get_velocity()
        return ((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5) * 3.6

    def is_parked(self):
        """检查是否已停车"""
        return self.parking_completed

def main():
    print("=" * 60)
    print("CARLA - Road Edge Parking System")
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
        
        parking_system = RoadEdgeParking(vehicle)
        
        print("[INFO] Vehicle spawned")
        print("[INFO] Road edge parking system activated")
        print("[INFO] Vehicle will automatically pull over when edge detected")
        print("[INFO] Press Ctrl+C to stop")
        
        try:
            while True:
                control = parking_system.update()
                vehicle.apply_control(control)
                
                speed = parking_system.get_current_speed()
                
                print(f"\r[INFO] State: {parking_system.state} | "
                      f"Speed: {speed:.1f} km/h | "
                      f"Edge: {parking_system.edge_distance:.2f}m | "
                      f"Parked: {parking_system.is_parked()}", end="")
                
                if parking_system.is_parked():
                    print("\n[SUCCESS] Vehicle has parked!")
                    time.sleep(2)
                    break
                
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