import carla
import numpy as np

class Sensors:
    def __init__(self, world, vehicle, cam_w=640, cam_h=480):
        self.world = world
        self.vehicle = vehicle
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.cameras = []
        self.lidar = None
        self.frame_dict = {}
        self.lidar_data = None

    def setup_all(self):
        self.setup_cameras()
        self.setup_lidar()

    def setup_cameras(self):
        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.cam_w))
        cam_bp.set_attribute('image_size_y', str(self.cam_h))

        configs = [
            {"name":"front", "x":1.8,"y":0,"z":1.8,"pitch":0,"yaw":0},
            {"name":"back", "x":-2,"y":0,"z":1.8,"pitch":0,"yaw":180},
            {"name":"left", "x":0,"y":-1,"z":1.8,"pitch":0,"yaw":-90},
            {"name":"right", "x":0,"y":1,"z":1.8,"pitch":0,"yaw":90},
        ]
        for cfg in configs:
            trans = carla.Transform(
                carla.Location(x=cfg['x'],y=cfg['y'],z=cfg['z']),
                carla.Rotation(pitch=cfg['pitch'],yaw=cfg['yaw'])
            )
            cam = self.world.spawn_actor(cam_bp, trans, attach_to=self.vehicle)
            cam.listen(lambda d, n=cfg['name']: self.cam_callback(d, n))
            self.cameras.append(cam)

    def cam_callback(self, data, name):
        arr = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((self.cam_h, self.cam_w, 4))[:,:,:3]
        self.frame_dict[name] = arr

    def setup_lidar(self):
        bp_lib = self.world.get_blueprint_library()
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range','50')
        lidar_bp.set_attribute('rotation_frequency','20')
        lidar_bp.set_attribute('channels','64')
        trans = carla.Transform(carla.Location(z=2.0))
        self.lidar = self.world.spawn_actor(lidar_bp, trans, attach_to=self.vehicle)
        self.lidar.listen(self.lidar_callback)

    def lidar_callback(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4')).reshape(-1,4)
        self.lidar_data = points[:,:3]

    def destroy(self):
        for c in self.cameras:
            try:
                if c.is_alive: c.destroy()
            except:
                pass
        try:
            if self.lidar.is_alive: self.lidar.destroy()
        except:
            pass