import carla
import random
import pygame
import numpy as np

# 渲染对象类来保持和传递PyGame表面
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))

    def update_surface(self, img_combined):
        # 更新渲染的 surface
        self.surface = pygame.surfarray.make_surface(img_combined.swapaxes(0, 1))

# 相机传感器回调，将相机的原始数据重塑为 2D RGB，并应用于 PyGame 表面
def pygame_callback(image, side, sensor_data):
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img = img[:, :, :3]  # Remove alpha channel
    img = img[:, :, ::-1]  # Convert RGB to BGR if necessary
    sensor_data[side] = img

    # 确保所有传感器都有数据，然后拼接图像
    if all(side in sensor_data for side in ['Front', 'Rear', 'Left', 'Right']):
        img_combined_front = np.concatenate((sensor_data['Front'], sensor_data['Rear']), axis=1)
        img_combined_rear = np.concatenate((sensor_data['Left'], sensor_data['Right']), axis=1)
        img_combined = np.concatenate((img_combined_front, img_combined_rear), axis=0)
        
        return img_combined
    return None

class CameraManager:
    def __init__(self, world, ego_vehicle, pygame_size):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.image_size_x = int(pygame_size.get("image_x") / 2)
        self.image_size_y = int(pygame_size.get("image_y") / 2)
        self.cameras = {}
        self.sensor_data = {'Front': None, 'Rear': None, 'Left': None, 'Right': None}

    def create_cameras(self):
        camera_transforms = [
            (carla.Transform(carla.Location(x=2.0, y=0.0, z=1.3), carla.Rotation(pitch=0, yaw=0, roll=0)), "Front"),
            (carla.Transform(carla.Location(x=-2.0, y=0.0, z=1.3), carla.Rotation(pitch=0, yaw=180, roll=0)), "Rear"),
            (carla.Transform(carla.Location(x=0.0, y=2.0, z=1.3), carla.Rotation(pitch=0, yaw=90, roll=0)), "Left"),
            (carla.Transform(carla.Location(x=0.0, y=-2.0, z=1.3), carla.Rotation(pitch=0, yaw=-90, roll=0)), "Right")
        ]

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('image_size_x', str(self.image_size_x))
        camera_bp.set_attribute('image_size_y', str(self.image_size_y))

        # 创建并绑定摄像头
        for transform, side in camera_transforms:
            camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.ego_vehicle)
            camera.listen(lambda image, side=side: pygame_callback(image, side, self.sensor_data))
            self.cameras[side] = camera
        return self.cameras

    def stop_cameras(self):
        # 停止所有摄像头
        for camera in self.cameras.values():
            camera.stop()

if __name__ == "__main__":
    # 连接到客户端并检索世界对象
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # 获取地图的刷出点
    spawn_point = random.choice(world.get_map().get_spawn_points())

    # 创建并生成车辆
    vehicle_bp = world.get_blueprint_library().filter('*vehicle*').filter('vehicle.tesla.*')[0]
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    ego_vehicle.set_autopilot(True)

    # 设置 pygame 窗口尺寸
    pygame_size = {
        "image_x": 1152,
        "image_y": 600
    }

    # 创建摄像头管理器实例并生成摄像头
    camera_manager = CameraManager(world, ego_vehicle, pygame_size)
    cameras = camera_manager.create_cameras()

    # 为渲染实例化对象
    render_object = RenderObject(pygame_size.get("image_x"), pygame_size.get("image_y"))

    # 初始化 pygame 显示
    pygame.init()
    game_display = pygame.display.set_mode((pygame_size.get("image_x"), pygame_size.get("image_y")), 
                                           pygame.HWSURFACE | pygame.DOUBLEBUF)

    # 循环执行
    crashed = False
    while not crashed:
        # 等待同步
        world.tick()

        # 每一帧更新渲染的 camera 图像
        img_combined = pygame_callback(None, 'Front', camera_manager.sensor_data)
        if img_combined is not None:
            render_object.update_surface(img_combined)
            game_display.blit(render_object.surface, (0, 0))
            pygame.display.flip()

        # 获取 pygame 事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

    # 停止自动驾驶并销毁相机和车辆
    ego_vehicle.set_autopilot(False)
    ego_vehicle.destroy()
    camera_manager.stop_cameras()

    # 退出 pygame
    pygame.quit()