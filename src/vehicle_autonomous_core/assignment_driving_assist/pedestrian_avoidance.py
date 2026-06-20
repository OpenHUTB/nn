import carla
import cv2
import numpy as np

class PedestrianAvoidance:
    def __init__(self, ego_car, depth_cam):
        self.ego = ego_car
        self.depth_cam = depth_cam
        self.safe_dist = 12

    def depth_callback(self, depth_data):
        depth_img = np.array(depth_data.raw_data).reshape(depth_data.height, depth_data.width, 4)
        depth_gray = depth_img[:,:,0]
        roi = depth_gray[int(depth_gray.shape[0]*0.6):, :]
        min_dist = np.min(roi)

        if min_dist < self.safe_dist:
            print("🚶 前方行人距离过近，紧急制动避让")
            brake_ctrl = carla.VehicleControl(brake=1.0, throttle=0)
            self.ego.apply_control(brake_ctrl)
        else:
            cruise_ctrl = carla.VehicleControl(throttle=0.2, brake=0)
            self.ego.apply_control(cruise_ctrl)

if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(8)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    car = world.spawn_actor(bp_lib.filter("model3")[0], world.get_map().get_spawn_points()[7])
    depth_bp = bp_lib.find("sensor.camera.depth")
    depth_cam = world.spawn_actor(depth_bp, carla.Transform(carla.Location(x=2,z=1.2)), attach_to=car)

    avoid = PedestrianAvoidance(car, depth_cam)
    depth_cam.listen(avoid.depth_callback)

    try:
        while True:
            world.tick()
    except KeyboardInterrupt:
        depth_cam.destroy()
        car.destroy()