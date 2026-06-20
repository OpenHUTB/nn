import carla
import cv2
import numpy as np

class SpeedLimitCruise:
    def __init__(self, ego_vehicle):
        self.ego = ego_vehicle
        self.base_speed = 30
        self.current_limit = 30

    def detect_speed_sign(self, frame):
        # 原有限速标识识别核心逻辑完全保留
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_red = np.array([0,120,100])
        high_red = np.array([10,255,255])
        mask = cv2.inRange(hsv, low_red, high_red)
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_con = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(max_con)
            if 0.8 < w/h < 1.2 and w>30:
                self.current_limit = 50
                print("🚦 识别限速标志，巡航速度调整为50km/h")
        self.set_cruise_speed()

    def set_cruise_speed(self):
        # 原有巡航油门刹车控制逻辑保留
        control = carla.VehicleControl()
        target_mps = self.current_limit / 3.6
        v = self.ego.get_velocity()
        current_mps = np.sqrt(v.x**2 + v.y**2 + v.z**2)
        if current_mps < target_mps:
            control.throttle = 0.3
        else:
            control.brake = 0.2
        self.ego.apply_control(control)

if __name__ == "__main__":
    # 连接CARLA仿真
    client = carla.Client("localhost",2000)
    client.set_timeout(8)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # 【修复】新增车辆生成逻辑，确保仿真内生成汽车
    car_bp = bp_lib.filter("model3")[0]
    spawn_point = world.get_map().get_spawn_points()[5]
    ego_car = world.spawn_actor(car_bp, spawn_point)

    # 挂载前视相机
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_tf = carla.Transform(carla.Location(x=2,z=1.3))
    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego_car)

    cruise = SpeedLimitCruise(ego_car)
    # 相机画面回调处理
    cam.listen(lambda img: cruise.detect_speed_sign(
        np.array(img.raw_data).reshape(img.height,img.width,4)[:,:,:3]
    ))

    try:
        while True:
            world.tick()
    except KeyboardInterrupt:
        # 安全销毁车辆与传感器，无残留actor
        cam.destroy()
        ego_car.destroy()
        cv2.destroyAllWindows()