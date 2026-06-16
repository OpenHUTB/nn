import config_carla
import carla
import cv2
import numpy as np

# 连接本地CARLA仿真服务
client = carla.Client('localhost', 2000)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# 配置RGB相机传感器参数
cam_bp = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '800')
cam_bp.set_attribute('image_size_y', '600')
spawn_point = carla.Transform(carla.Location(x=-5.5, z=2.8))
sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=world.get_actors().filter('*vehicle*')[0])

# 图像存储回调函数
def save_img(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    cv2.imwrite('data/raw/carla.png', array)

# 持续监听相机数据流并保存画面
sensor.listen(lambda img: save_img(img))