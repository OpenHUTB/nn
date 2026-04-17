import carla
import random
import pygame
import numpy as np

# 渲染对象来保持和传递 PyGame 表面
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))

# 相机传感器回调，将相机的原始数据重塑为 2D RGB，并应用于 PyGame 表面
def pygame_callback(image, side):
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    if side == 'Front':
        global Front
        Front = img
    elif side == 'Rear':
        global Rear
        Rear = img
    elif side == 'Left':
        global Left
        Left = img
    elif side == 'Right':
        global Right
        Right = img

    if ('Front' in globals() and 'Rear' in globals()
        and "Left" in globals()and 'Right' in globals()):
        # 横向拼接（前后）（左右）摄像头的画面
        img_combined_front = np.concatenate((Front, Rear), axis=1)
        img_combined_rear = np.concatenate((Left, Right), axis=1)
        # 纵向拼接（前后）（左右）摄像头的画面
        img_combined = np.concatenate((img_combined_front, img_combined_rear), axis=0)
        renderObject.surface = pygame.surfarray.make_surface(img_combined.swapaxes(0, 1))

class cameraManage():
    def __init__(self, world, ego_vehicle, pygame_size):
        self.world = world
        self.cameras = {}
        self.ego_vehicle = ego_vehicle
        self.image_size_x = int(pygame_size.get("image_x") / 2)  # 横向放置两个摄像头的画面
        self.image_size_y = int(pygame_size.get("image_y") / 2)  # 纵向放置两个摄像头的画面

    def camaraGenarate(self):
        cameras_transform = [
            (carla.Transform(carla.Location(x=2.0, y=0.0, z=1.3),  # 前侧摄像头安装位置
                             carla.Rotation(pitch=0, yaw=0, roll=0)), "Front"),
            (carla.Transform(carla.Location(x=-2.0, y=0.0, z=1.3),  # 后侧摄像头安装位置
                             carla.Rotation(pitch=0, yaw=180, roll=0)), "Rear"),
            (carla.Transform(carla.Location(x=0.0, y=2.0, z=1.3),  # 左侧摄像头安装位置
                             carla.Rotation(pitch=0, yaw=90, roll=0)), "Left"),
            (carla.Transform(carla.Location(x=0.0, y=-2.0, z=1.3),  # 右侧的摄像头安装位置
                             carla.Rotation(pitch=0, yaw=-90, roll=0)), "Right")
        ]
        # 查找RGB相机蓝图
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # 设置摄像头的fov为90°
        camera_bp.set_attribute('fov', "90")
        # 设置摄像头的分辨率
        camera_bp.set_attribute('image_size_x', str(self.image_size_x))
        camera_bp.set_attribute('image_size_y', str(self.image_size_y))
        # 生成摄像头
        for index, (camera_ts, camera_sd) in enumerate(cameras_transform):
            camera = self.world.spawn_actor(camera_bp, camera_ts, attach_to=self.ego_vehicle)
            self.cameras[camera_sd] = camera
        return self.cameras


if __name__ == "__main__":
    # 连接到客户端并检索世界对象
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # 获取地图的刷出点
    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle_bp = world.get_blueprint_library().filter('*vehicle*').filter('vehicle.tesla.*')[0]
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    ego_vehicle.set_autopilot(True)

    #设置pygame窗口size,image_x
    pygame_size = {
        "image_x": 1152,
        "image_y": 600
    }

    #调用cameraManage类，生成摄像头
    cameras = cameraManage(world, ego_vehicle, pygame_size).camaraGenarate()

    #采集carla世界中camera的图像
    cameras.get("Front").listen(lambda image: pygame_callback(image, 'Front'))
    cameras.get("Rear").listen(lambda image: pygame_callback(image, 'Rear'))
    cameras.get("Left").listen(lambda image: pygame_callback(image, 'Left'))
    cameras.get("Right").listen(lambda image: pygame_callback(image, 'Right'))

    # 为渲染实例化对象
    renderObject = RenderObject(pygame_size.get("image_x"), pygame_size.get("image_y"))

    # 初始化pygame显示
    pygame.init()
    gameDisplay = pygame.display.set_mode((pygame_size.get("image_x"), pygame_size.get("image_y")),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)
    # 循环执行
    crashed = False
    while not crashed:
        # 等待同步
        world.tick()

        # 按帧更新渲染的 Camera 画面
        gameDisplay.blit(renderObject.surface, (0, 0))
        pygame.display.flip()

        # 获取 pygame 事件
        for event in pygame.event.get():
            # If the window is closed, break the while loop
            if event.type == pygame.QUIT:
                crashed = True

    # 结束
    ego_vehicle.set_autopilot(False)
    ego_vehicle.destroy()
    camera = cameras.values()
    for cam in camera:
        cam.stop
    pygame.quit()
