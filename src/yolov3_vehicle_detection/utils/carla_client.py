import carla
import random
import time
import sys
import os
import numpy as np
import cv2
import queue

from ..config import config


class CarlaClient:
    """
    CARLA 模拟器客户端封装类
    """

    # [Mod] 修改 __init__ 方法，允许覆盖 host 和 port
    def __init__(self, host=None, port=None):
        # 如果传入了参数就用传入的，否则用 config 里的默认值
        self.host = host if host else config.CARLA_HOST
        self.port = port if port else config.CARLA_PORT
        self.timeout = config.CARLA_TIMEOUT

        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.blueprint_library = None
        self.image_queue = queue.Queue()

    def connect(self):
        print(f"[INFO] 正在连接 CARLA 服务器 ({self.host}:{self.port})...")
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            print("[INFO] CARLA 连接成功！")
            return True
        except Exception as e:
            print(f"[ERROR] 连接失败: {e}")
            return False

    def spawn_vehicle(self):
        if not self.world:
            print("[ERROR] 世界未加载，请先连接！")
            return None

        model_name = config.VEHICLE_MODEL
        bp = self.blueprint_library.find(model_name)

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        try:
            self.vehicle = self.world.spawn_actor(bp, spawn_point)
            print(f"[INFO] 车辆生成成功: {self.vehicle.type_id}")
            self.vehicle.set_autopilot(True)
            return self.vehicle
        except Exception as e:
            print(f"[ERROR] 车辆生成失败: {e}")
            return None

    def setup_camera(self):
        if not self.vehicle:
            return
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(config.CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(config.CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(config.CAMERA_FOV))

        spawn_point = carla.Transform(carla.Location(x=config.CAMERA_POS_X, z=config.CAMERA_POS_Z))

        self.camera = self.world.spawn_actor(camera_bp, spawn_point, attach_to=self.vehicle)
        self.camera.listen(lambda image: self._process_image(image))
        print("[INFO] RGB 摄像头安装成功！")

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = np.ascontiguousarray(array[:, :, :3])
        self.image_queue.put(array)

    def destroy_actors(self):
        if self.camera:
            self.camera.destroy()
            self.camera = None
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
        print("[INFO] 所有 Actor 已清理。")