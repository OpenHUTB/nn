import carla
import sys
import time

class CollisionDetection:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.collision_sensor = None
        self.has_collided = False
        self.collision_time = 0
        self.brake_active = False

    def setup_collision_sensor(self, world):
        """设置碰撞传感器"""
        blueprint = world.get_blueprint_library()
        collision_bp = blueprint.find('sensor.other.collision')
        
        self.collision_sensor = world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        """碰撞事件回调函数"""
        if not self.has_collided:
            self.has_collided = True
            self.collision_time = time.time()
            self.brake_active = True
            print("[COLLISION] Detected! Activating emergency brake...")
            
            # 立即应用紧急制动
            control = carla.VehicleControl(throttle=0, brake=1.0, steer=0)
            self.vehicle.apply_control(control)

    def update(self):
        """更新碰撞检测状态"""
        # 如果碰撞后已经刹车2秒，解除制动状态
        if self.brake_active and (time.time() - self.collision_time) > 2.0:
            self.brake_active = False
            print("[COLLISION] Brake released after 2 seconds")

    def is_colliding(self):
        """检查是否正在碰撞"""
        return self.has_collided and self.brake_active

    def reset(self):
        """重置碰撞状态"""
        self.has_collided = False
        self.brake_active = False

    def destroy(self):
        """销毁传感器"""
        if self.collision_sensor:
            self.collision_sensor.destroy()

def main():
    print("=" * 60)
    print("CARLA - Collision Detection System")
    print("=" * 60)
    
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        print("[INFO] Connected to CARLA server successfully")
        
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        # 生成车辆
        tesla_bp = blueprint_library.find("vehicle.tesla.model3")
        tesla_bp.set_attribute("color", "0, 0, 0")
        
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(tesla_bp, spawn_points[0])
        
        # 设置碰撞检测
        collision_detector = CollisionDetection(vehicle)
        collision_detector.setup_collision_sensor(world)
        
        # 开启自动驾驶
        vehicle.set_autopilot(True)
        print("[INFO] Autopilot enabled")
        print("[INFO] Collision detection system activated")
        print("[INFO] Press Ctrl+C to stop")
        
        try:
            while True:
                collision_detector.update()
                
                if collision_detector.is_colliding():
                    print("\r[EMERGENCY] BRAKING - Collision detected!", end="")
                else:
                    velocity = vehicle.get_velocity()
                    speed = ((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5) * 3.6
                    print(f"\r[INFO] Speed: {speed:.1f} km/h | Collision: {'DETECTED' if collision_detector.has_collided else 'Safe'}", end="")
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted")
        finally:
            print("\n[INFO] Cleaning up...")
            collision_detector.destroy()
            vehicle.destroy()
            print("[INFO] Done")
            
    except RuntimeError as e:
        print(f"[ERROR] Runtime error: {e}")
        print("[INFO] Make sure CARLA server is running")
        sys.exit(1)

if __name__ == "__main__":
    main()