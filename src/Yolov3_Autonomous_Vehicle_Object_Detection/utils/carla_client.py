import carla
import random
import time
import sys
from config import config


class CarlaClient:
    """
    CARLA 模拟器客户端封装类
    负责连接服务器、加载世界、生成车辆与传感器
    """

    def __init__(self):
        self.host = config.CARLA_HOST
        self.port = config.CARLA_PORT
        self.timeout = config.CARLA_TIMEOUT

        self.client = None
        self.world = None
        self.vehicle = None
        self.blueprint_library = None

    def connect(self):
        """
        连接到 CARLA 服务器
        """
        print(f"[INFO] 正在连接 CARLA 服务器 ({self.host}:{self.port})...")
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)

            # 获取世界对象
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()

            print("[INFO] CARLA 连接成功！")
            return True
        except Exception as e:
            print(f"[ERROR] 连接失败: {e}")
            return False

    def spawn_vehicle(self, model_name='vehicle.tesla.model3'):
        """
        在随机生成点生成一辆车辆
        """
        if not self.world:
            print("[ERROR] 世界未加载，请先连接！")
            return None

        # 1. 选择车辆蓝图 (默认特斯拉 Model 3)
        bp = self.blueprint_library.find(model_name)

        # 2. 获取所有建议生成点
        spawn_points = self.world.get_map().get_spawn_points()

        # 3. 随机选择一个点
        spawn_point = random.choice(spawn_points)

        # 4. 生成车辆
        try:
            self.vehicle = self.world.spawn_actor(bp, spawn_point)
            print(f"[INFO] 车辆生成成功: {self.vehicle.type_id} (ID: {self.vehicle.id})")

            # 开启自动驾驶模式 (用于测试)
            self.vehicle.set_autopilot(True)
            return self.vehicle
        except Exception as e:
            print(f"[ERROR] 车辆生成失败: {e}")
            return None

    def destroy_actors(self):
        """
        清理生成的 Actor，防止残留
        """
        if self.vehicle:
            print(f"[INFO] 正在销毁车辆: {self.vehicle.id}")
            self.vehicle.destroy()
            self.vehicle = None