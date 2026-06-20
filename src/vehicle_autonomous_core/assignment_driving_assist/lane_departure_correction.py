import carla
import cv2
import numpy as np

class LaneDepartCorrection:
    def __init__(self, ego_car, cam):
        self.ego = ego_car
        self.cam = cam

    def frame_process(self, frame_data):
        img = np.array(frame_data.raw_data).reshape(frame_data.height, frame_data.width,4)[:,:,:3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        h, w = edges.shape
        roi = edges[int(h*0.5):h, :]
        lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=40)

        offset = 0
        if lines is not None:
            left_lines = []
            right_lines = []
            for line in lines:
                x1,y1,x2,y2 = line[0]
                slope = (y2-y1)/(x2-x1+1e-6)
                if slope < -0.2:
                    left_lines.append(slope)
                elif slope > 0.2:
                    right_lines.append(slope)
            if len(left_lines) == 0:
                offset = 0.25
            elif len(right_lines) == 0:
                offset = -0.25

        control = carla.VehicleControl(throttle=0.15, steer=offset)
        self.ego.apply_control(control)
        cv2.imshow("Lane View", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    client = carla.Client("localhost",2000)
    client.set_timeout(8)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    car = world.spawn_actor(bp_lib.filter("model3")[0], world.get_map().get_spawn_points()[10])
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=2,z=1.3)), attach_to=car)

    lane_correct = LaneDepartCorrection(car, cam)
    cam.listen(lane_correct.frame_process)

    try:
        while True:
            world.tick()
    except KeyboardInterrupt:
        cam.destroy()
        car.destroy()
        cv2.destroyAllWindows()