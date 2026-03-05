import carla
import random
import time
import sys
import os
import numpy as np
import cv2
import queue

# ================= 路径修复 =================
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_path)
sys.path.append(project_root)
# ===========================================

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
        self.camera = None
        self.blueprint_library = None

        # 用于在不同线程间传递图像数据的队列
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

    def spawn_vehicle(self, model_name='vehicle.tesla.model3'):
        if not self.world:
            print("[ERROR] 世界未加载，请先连接！")
            return None

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
        """
        安装 RGB 摄像头传感器
        """
        if not self.vehicle:
            print("[ERROR] 车辆未生成，无法安装传感器！")
            return

        # 1. 查找摄像头蓝图
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')

        # 2. 配置摄像头参数 (从 config 读取)
        camera_bp.set_attribute('image_size_x', str(config.CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(config.CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(config.CAMERA_FOV))

        # 3. 设置摄像头安装位置 (坐标系: x前, y右, z上) -> 安装在引擎盖上方
        spawn_point = carla.Transform(carla.Location(x=1.5, z=2.4))

        # 4. 生成传感器并附着(Attach)到车辆上
        self.camera = self.world.spawn_actor(camera_bp, spawn_point, attach_to=self.vehicle)

        # 5. 注册回调函数 (监听数据)
        # 这里的 lambda 是为了把 raw_data 传给处理函数
        self.camera.listen(lambda image: self._process_image(image))
        print("[INFO] RGB 摄像头安装成功！")

    def _process_image(self, image):
        """
        [回调函数] 将 CARLA 原始数据转换为 OpenCV 图像格式
        """
        # 1. 将原始 buffer 转换为 numpy 数组
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))

        # 2. 重塑形状 (Height, Width, 4通道-BGRA)
        array = np.reshape(array, (image.height, image.width, 4))

        # 3. 取前3个通道 (BGR)，扔掉 Alpha 通道
        array = array[:, :, :3]

        # 4. 存入队列 (供主线程消费)
        self.image_queue.put(array)

    def destroy_actors(self):
        if self.camera:
            self.camera.destroy()
            self.camera = None
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
        print("[INFO] 所有 Actor 已清理。")


# 测试代码
if __name__ == "__main__":
    client = CarlaClient()
    if client.connect():
        vehicle = client.spawn_vehicle()
        client.setup_camera()

        # 简单测试：尝试从队列取一张图片看看 shape 对不对
        try:
            print("等待图像数据...")
            img = client.image_queue.get(timeout=5)
            print(f"成功获取图像！尺寸: {img.shape}")
        except queue.Empty:
            print("未接收到图像数据！")

        time.sleep(2)
        client.destroy_actors()