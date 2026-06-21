import carla
import sys
import time

class AutoDriveWithFeatures:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()
        
        # 功能状态
        self.collision_detected = False
        self.red_light_detected = False
        self.lane_offset = 0.0
        
        # 碰撞传感器
        self.collision_sensor = None
        self.setup_collision_sensor()
        
        # 启用自动驾驶
        self.vehicle.set_autopilot(True)
        print("[INFO] Autopilot enabled")

    def setup_collision_sensor(self):
        blueprint = self.world.get_blueprint_library()
        collision_bp = blueprint.find('sensor.other.collision')
        
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(carla.Location(z=2.0)),
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        self.collision_detected = True
        print(f"\n[COLLISION] Detected with {event.other_actor.type_id}")
        
        control = carla.VehicleControl(throttle=0, brake=1.0, hand_brake=True)
        self.vehicle.apply_control(control)
        
        time.sleep(2)
        self.collision_detected = False

    def check_traffic_light(self):
        """检查前方交通灯"""
        vehicle_location = self.vehicle.get_transform().location
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        
        for light in traffic_lights:
            distance = vehicle_location.distance(light.get_transform().location)
            if distance < 50:
                state = light.state
                if state == carla.TrafficLightState.Red:
                    self.red_light_detected = True
                    return True, distance
        self.red_light_detected = False
        return False, float('inf')

    def get_lane_offset(self):
        """获取车道偏移"""
        vehicle_transform = self.vehicle.get_transform()
        waypoint = self.map.get_waypoint(vehicle_transform.location)
        
        if waypoint:
            lane_center = waypoint.transform.location
            forward = vehicle_transform.get_forward_vector()
            right = carla.Vector3D(-forward.y, forward.x, 0)
            offset_vector = lane_center - vehicle_transform.location
            self.lane_offset = offset_vector.dot(right)
        
        return self.lane_offset

    def update(self):
        """更新所有功能"""
        self.get_lane_offset()
        light_detected, distance = self.check_traffic_light()
        
        if light_detected:
            self.vehicle.set_autopilot(False)
            control = carla.VehicleControl(throttle=0, brake=1.0)
            self.vehicle.apply_control(control)
        elif not self.collision_detected:
            self.vehicle.set_autopilot(True)
        
        return {
            'collision': self.collision_detected,
            'red_light': self.red_light_detected,
            'lane_offset': self.lane_offset
        }

    def destroy(self):
        if self.collision_sensor:
            self.collision_sensor.destroy()

def main():
    print("=" * 60)
    print("CARLA - Enhanced AutoDrive System")
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
        
        auto_drive = AutoDriveWithFeatures(vehicle)
        
        print("[INFO] Vehicle spawned")
        print("[INFO] Enhanced auto drive system activated")
        print("[INFO] Features: Collision detection, Traffic light recognition, Lane keeping")
        print("[INFO] Press Ctrl+C to stop")
        
        try:
            while True:
                status = auto_drive.update()
                
                speed = ((vehicle.get_velocity().x**2 + 
                         vehicle.get_velocity().y**2 + 
                         vehicle.get_velocity().z**2) ** 0.5) * 3.6
                
                collision_status = "DETECTED" if status['collision'] else "Safe"
                light_status = "RED" if status['red_light'] else "Green"
                
                print(f"\r[INFO] Speed: {speed:.1f} km/h | Collision: {collision_status} | "
                      f"Light: {light_status} | Offset: {status['lane_offset']:.2f}m", end="")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted")
        finally:
            print("\n[INFO] Cleaning up...")
            auto_drive.destroy()
            vehicle.destroy()
            print("[INFO] Done")
            
    except RuntimeError as e:
        print(f"[ERROR] Runtime error: {e}")
        print("[INFO] Make sure CARLA server is running")
        sys.exit(1)

if __name__ == "__main__":
    main()