import carla
import random
import cv2
import numpy as np

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 同步模式
    tm = client.get_trafficmanager()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)

    # 生成车辆
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.*model3*')[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = None

    for spawn in random.sample(spawn_points, len(spawn_points)):
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn)
            break
        except:
            continue

    if not vehicle:
        return

    vehicle.set_autopilot(True)
    spectator = world.get_spectator()

    # 90° 俯视跟随
    def update_top_view():
        trans = vehicle.get_transform()
        camera_loc = trans.location + carla.Location(z=20)
        camera_rot = carla.Rotation(pitch=-90, yaw=trans.rotation.yaw)
        spectator.set_transform(carla.Transform(camera_loc, camera_rot))

    # 相机设置
    cam_w, cam_h = 640, 480
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(cam_w))
    camera_bp.set_attribute('image_size_y', str(cam_h))

    cameras = []
    frame_dict = {}

    def callback(data, name):
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = array.reshape((cam_h, cam_w, 4))[:, :, :3]
        frame_dict[name] = array

    # 四个方向相机
    cam_configs = [
        {"name": "front",  "x": 1.8, "y":  0.0, "z": 1.8, "pitch": 0, "yaw":   0},
        {"name": "back",   "x":-2.0, "y":  0.0, "z": 1.8, "pitch": 0, "yaw": 180},
        {"name": "left",   "x": 0.0, "y": -1.0, "z": 1.8, "pitch": 0, "yaw": -90},
        {"name": "right",  "x": 0.0, "y":  1.0, "z": 1.8, "pitch": 0, "yaw":  90},
    ]

    for cfg in cam_configs:
        trans = carla.Transform(
            carla.Location(x=cfg['x'], y=cfg['y'], z=cfg['z']),
            carla.Rotation(pitch=cfg['pitch'], yaw=cfg['yaw'])
        )
        cam = world.spawn_actor(camera_bp, trans, attach_to=vehicle)
        cam.listen(lambda data, name=cfg['name']: callback(data, name))
        cameras.append(cam)

    cv2.namedWindow("Camera 2x2 Monitor", cv2.WINDOW_NORMAL)

    try:
        while True:
            world.tick()
            update_top_view()

           
            if len(frame_dict) < 4:
                continue

            img_front = frame_dict['front']
            img_back  = frame_dict['back']
            img_left  = frame_dict['left']
            img_right = frame_dict['right']

            # 2x2 拼接
            top    = np.hstack((img_front, img_back))
            bottom = np.hstack((img_left, img_right))
            mosaic = np.vstack((top, bottom))

            cv2.imshow("Camera 2x2 Monitor", mosaic)
            if cv2.waitKey(1) == 27:
                break

    finally:
        for cam in cameras:
            cam.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()