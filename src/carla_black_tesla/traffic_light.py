import carla
import sys
import time

class TrafficLightDetector:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()
        
        # 交通灯状态
        self.current_light = None
        self.light_state = 'green'
        self.light_distance = float('inf')
        self.stopped_at_light = False
        
        # 控制参数
        self.stop_distance = 5.0
        self.max_detect_distance = 50.0

    def get_traffic_light(self):
        """获取车辆前方最近的交通灯"""
        vehicle_location = self.vehicle.get_transform().location
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)
        
        if vehicle_waypoint is None:
            return None, 'green', float('inf')
        
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        
        nearest_light = None
        min_distance = float('inf')
        
        for light in traffic_lights:
            light_transform = light.get_transform()
            light_location = light_transform.location
            
            distance = vehicle_location.distance(light_location)
            
            if distance < self.max_detect_distance:
                forward_vector = self.vehicle.get_transform().get_forward_vector()
                to_light = (light_location - vehicle_location)
                dot_product = forward_vector.dot(to_light.make_unit_vector())
                
                if dot_product > 0.7:
                    if distance < min_distance:
                        min_distance = distance
                        nearest_light = light
        
        if nearest_light is not None:
            state = nearest_light.state
            state_str = self._get_state_string(state)
            return nearest_light, state_str, min_distance
        
        return None, 'green', float('inf')

    def _get_state_string(self, state):
        """将交通灯状态转换为字符串"""
        if state == carla.TrafficLightState.Red:
            return 'red'
        elif state == carla.TrafficLightState.Yellow:
            return 'yellow'
        elif state == carla.TrafficLightState.Green:
            return 'green'
        else:
            return 'off'

    def update(self):
        """更新交通灯检测状态"""
        self.current_light, self.light_state, self.light_distance = self.get_traffic_light()
        
        # 如果检测到红灯且距离足够近，标记需要停车
        if self.light_state == 'red' and self.light_distance < self.stop_distance:
            self.stopped_at_light = True
        elif self.light_state != 'red' and self.stopped_at_light:
            self.stopped_at_light = False

    def should_stop(self):
        """判断是否应该停车"""
        return self.light_state == 'red' and self.light_distance < self.stop_distance

    def is_stopped_at_light(self):
        """是否已在红绿灯前停车"""
        return self.stopped_at_light

def main():
    print("=" * 60)
    print("CARLA - Traffic Light Recognition System")
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
        
        traffic_detector = TrafficLightDetector(vehicle)
        
        print("[INFO] Vehicle spawned")
        print("[INFO] Traffic light detection enabled")
        print("[INFO] Press Ctrl+C to stop")
        
        try:
            while True:
                traffic_detector.update()
                
                light_state = traffic_detector.light_state
                distance = traffic_detector.light_distance
                should_stop = traffic_detector.should_stop()
                
                # 控制车辆
                if should_stop:
                    control = carla.VehicleControl(throttle=0, brake=1.0, steer=0)
                    print(f"\r[RED LIGHT] Stopping - Distance: {distance:.1f}m", end="")
                else:
                    control = carla.VehicleControl(throttle=0.3, brake=0, steer=0)
                    status = f"[GO] Light: {light_state}"
                    if distance < float('inf'):
                        status += f" - Distance: {distance:.1f}m"
                    print(f"\r{status}", end="")
                
                vehicle.apply_control(control)
                
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