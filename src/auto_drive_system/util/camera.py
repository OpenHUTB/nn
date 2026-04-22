import carla
import numpy as np
import pygame


# --- 1. 渲染对象 ---
class RenderObject(object):
    def __init__(self, width, height):
        # 初始化一个随机噪点表面，防止程序启动时黑屏
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))


# --- 2. 摄像头管理器 ---
class cameraManage():
    def __init__(self, world, ego_vehicle, pygame_size):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.image_size_x = int(pygame_size.get("image_x") / 2)
        self.image_size_y = int(pygame_size.get("image_y") / 2)
        self.cameras = {}

        # 用于存储最新的图像数据，供外部获取
        self.sensor_data = {'Front': None, 'Rear': None, 'Left': None, 'Right': None}

    def camaraGenarate(self):
        # 定义四个摄像头的相对位置和名称
        cameras_transform = [
            (carla.Transform(carla.Location(x=2.0, y=0.0, z=1.3), carla.Rotation(pitch=0, yaw=0, roll=0)), "Front"),
            (carla.Transform(carla.Location(x=-2.0, y=0.0, z=1.3), carla.Rotation(pitch=0, yaw=180, roll=0)), "Rear"),
            (carla.Transform(carla.Location(x=0.0, y=2.0, z=1.3), carla.Rotation(pitch=0, yaw=90, roll=0)), "Left"),
            (carla.Transform(carla.Location(x=0.0, y=-2.0, z=1.3), carla.Rotation(pitch=0, yaw=-90, roll=0)), "Right")
        ]

        # 获取蓝图并设置属性
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('fov', "90")
        camera_bp.set_attribute('image_size_x', str(self.image_size_x))
        camera_bp.set_attribute('image_size_y', str(self.image_size_y))

        # 生成并绑定摄像头
        for camera_ts, camera_sd in cameras_transform:
            camera = self.world.spawn_actor(camera_bp, camera_ts, attach_to=self.ego_vehicle)
            # 绑定回调函数，传入侧边名称
            camera.listen(lambda image, name=camera_sd: self._process_image(image, name))
            self.cameras[camera_sd] = camera

        return self.cameras

    def _process_image(self, image, side):
        """
        图像处理：兼容旧版 CARLA 的写法
        """
        try:
            # 1. 将原始数据转换为 numpy 数组
            # CARLA 的原始数据是 BGRA 格式 (4通道)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))

            # 2. 重塑数组形状 (高, 宽, 4)
            array = np.reshape(array, (image.height, image.width, 4))

            # 3. 去掉 Alpha 通道，只保留 BGR (3通道)
            # 注意：CARLA 默认就是 BGR 格式，不需要像 RGBA 那样反转颜色
            array = array[:, :, :3]

            self.sensor_data[side] = array
        except Exception as e:
            print(f"图像处理错误: {e}")

    def get_data(self):
        """
        提供给外部调用的方法，获取当前的四路图像数据
        """
        return self.sensor_data