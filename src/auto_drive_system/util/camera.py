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
        # 拿到配置里的分辨率，这里除以2是为了适配拼接后的总大小（推测）
        self.image_size_x = int(pygame_size.get("image_x") / 2)
        self.image_size_y = int(pygame_size.get("image_y") / 2)
        self.cameras = {}

        # 定义一个字典，专门用来存四路摄像头的最新画面
        self.sensor_data = {'Front': None, 'Rear': None, 'Left': None, 'Right': None}

    def camaraGenarate(self):
        # 1. 定义四个摄像头的相对位置和名字
        cameras_transform = [
            # 前：车头往前一点
            (carla.Transform(carla.Location(x=2.0, y=0.0, z=1.3), carla.Rotation(pitch=0, yaw=0, roll=0)), "Front"),
            # 后：车尾往后一点，旋转180度
            (carla.Transform(carla.Location(x=-2.0, y=0.0, z=1.3), carla.Rotation(pitch=0, yaw=180, roll=0)), "Rear"),
            # 左：车身左侧，旋转90度
            (carla.Transform(carla.Location(x=0.0, y=2.0, z=1.3), carla.Rotation(pitch=0, yaw=90, roll=0)), "Left"),
            # 右：车身右侧，旋转-90度
            (carla.Transform(carla.Location(x=0.0, y=-2.0, z=1.3), carla.Rotation(pitch=0, yaw=-90, roll=0)), "Right")
        ]

        # 2. 获取摄像头蓝图并配置参数
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('fov', "90")  # 视野角度
        camera_bp.set_attribute('image_size_x', str(self.image_size_x))
        camera_bp.set_attribute('image_size_y', str(self.image_size_y))

        # 3. 生成并绑定摄像头
        for camera_ts, camera_sd in cameras_transform:
            # 把摄像头“粘”在车上
            camera = self.world.spawn_actor(camera_bp, camera_ts, attach_to=self.ego_vehicle)
            # 绑定回调函数：每次摄像头拍到东西，就自动运行 _process_image
            # 注意这里用了 name=camera_sd 默认参数，是为了防止循环闭包问题
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

            # 把处理好的图存进字典，key就是侧边名称（Front/Rear等）
            self.sensor_data[side] = array
        except Exception as e:
            print(f"图像处理错误: {e}")

    def get_data(self):
        """
        提供给外部调用的方法，获取当前的四路图像数据
        """
        return self.sensor_data