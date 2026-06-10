import carla
import math
import time

class PathTracker:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.waypoints = []
        self.current_index = 0
        self.target_speed = 30.0  # km/h
        self.lookahead_distance = 5.0

    def set_path(self, waypoints):
        """设置路径点列表"""
        self.waypoints = waypoints
        self.current_index = 0

    def set_speed(self, speed):
        """设置目标速度 (km/h)"""
        self.target_speed = speed

    def get_current_waypoint(self):
        """获取当前目标路径点"""
        if self.current_index >= len(self.waypoints):
            return None
        return self.waypoints[self.current_index]

    def is_finished(self):
        """检查是否完成路径"""
        return self.current_index >= len(self.waypoints)

    def compute_control(self):
        """计算控制指令"""
        if self.is_finished():
            return carla.VehicleControl(throttle=0, brake=1)
        
        current_pos = self.vehicle.get_location()
        target_wp = self.get_current_waypoint()
        
        # 计算到目标点的距离
        distance = math.sqrt(
            (target_wp.x - current_pos.x)**2 +
            (target_wp.y - current_pos.y)**2
        )
        
        # 如果接近当前路径点，切换到下一个
        if distance < self.lookahead_distance:
            self.current_index += 1
            if self.is_finished():
                return carla.VehicleControl(throttle=0, brake=1)
            target_wp = self.get_current_waypoint()
        
        # 计算转向角度
        forward = self.vehicle.get_transform().get_forward_vector()
        target_dir = carla.Vector3D(
            target_wp.x - current_pos.x,
            target_wp.y - current_pos.y,
            0
        )
        
        # 归一化
        length = math.sqrt(target_dir.x**2 + target_dir.y**2)
        if length > 0:
            target_dir.x /= length
            target_dir.y /= length
        
        # 计算横向误差
        cross = forward.x * target_dir.y - forward.y * target_dir.x
        steer = max(-1, min(1, cross * 2.0))
        
        # 计算速度控制
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2) * 3.6
        
        throttle = 0.5 if speed < self.target_speed else 0
        brake = 0.3 if speed > self.target_speed + 5 else 0
        
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

def main():
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        bp_lib = world.get_blueprint_library()
        tesla_bp = bp_lib.find("vehicle.tesla.model3")
        tesla_bp.set_attribute("color", "0, 0, 0")
        
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(tesla_bp, spawn_points[0])
        
        # 创建路径点
        start = vehicle.get_location()
        waypoints = [
            carla.Location(x=start.x + 50, y=start.y, z=start.z),
            carla.Location(x=start.x + 50, y=start.y + 50, z=start.z),
            carla.Location(x=start.x, y=start.y + 50, z=start.z),
            carla.Location(x=start.x, y=start.y, z=start.z),
        ]
        
        tracker = PathTracker(vehicle)
        tracker.set_path(waypoints)
        tracker.set_speed(20.0)
        
        print("Path tracking started!")
        print(f"Total waypoints: {len(waypoints)}")
        
        while not tracker.is_finished():
            control = tracker.compute_control()
            vehicle.apply_control(control)
            
            pos = vehicle.get_location()
            idx = tracker.current_index
            print(f"Waypoint {idx+1}/{len(waypoints)} | Pos: ({pos.x:.1f}, {pos.y:.1f})")
            
            time.sleep(0.1)
        
        print("Path completed!")
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        vehicle.destroy()

if __name__ == "__main__":
    main()