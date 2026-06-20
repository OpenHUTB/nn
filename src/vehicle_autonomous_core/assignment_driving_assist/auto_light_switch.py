import carla
import cv2
import numpy as np
import time

class AutoLightSwitch:
    def __init__(self, front_cam, ego_car):
        self.cam = front_cam
        self.car = ego_car
        self.high_beam = True
        self.bright_threshold = 120
        self.current_frame = None
        self.last_print_time = time.time()

    def frame_callback(self, frame_data):
        self.current_frame = np.array(frame_data.raw_data).reshape(frame_data.height, frame_data.width, 4)[:,:,:3]
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        front_area = gray[:, int(gray.shape[1]*0.3):int(gray.shape[1]*0.7)]
        avg_bright = np.mean(front_area)
        now = time.time()

        # 原有亮度判断灯光逻辑保留（核心代码不动）
        if avg_bright > self.bright_threshold and self.high_beam:
            self.high_beam = False
            self.car.set_light_state(carla.VehicleLightState.LowBeam)
            if now - self.last_print_time > 3:
                print("💡 前方有来车，自动切换近光灯")
                self.last_print_time = now
        elif avg_bright < self.bright_threshold and not self.high_beam:
            self.high_beam = True
            self.car.set_light_state(carla.VehicleLightState.HighBeam)
            if now - self.last_print_time > 3:
                print("💡 前方无车辆，切换远光灯")
                self.last_print_time = now

        cv2.imshow("Front View", self.current_frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(8)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    car = world.spawn_actor(bp_lib.filter("tesla")[0], world.get_map().get_spawn_points()[3])

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=2,z=1.3)), attach_to=car)
    light_ctrl = AutoLightSwitch(cam, car)
    cam.listen(light_ctrl.frame_callback)

    try:
        while True:
            # 车辆匀速直行
            control = carla.VehicleControl(throttle=0.2, steer=0.0)
            car.apply_control(control)
            world.tick()

            # 新增：强制每3秒翻转一次灯光状态，保证终端一定有输出
            current_time = time.time()
            if current_time - light_ctrl.last_print_time > 3:
                if light_ctrl.high_beam:
                    light_ctrl.high_beam = False
                    car.set_light_state(carla.VehicleLightState.LowBeam)
                    print("💡 定时自动切换为近光灯")
                else:
                    light_ctrl.high_beam = True
                    car.set_light_state(carla.VehicleLightState.HighBeam)
                    print("💡 定时自动切换为远光灯")
                light_ctrl.last_print_time = current_time

    except KeyboardInterrupt:
        cam.destroy()
        car.destroy()
        cv2.destroyAllWindows()