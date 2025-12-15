import carla
import pygame
import time
import random
import queue
import cv2
import numpy as np
from threading import Lock

# 自定义线性插值函数（适配同步帧）
def lerp(a, b, t):
    return a + t * (b - a)

# 语义分割调色板（Cityscapes格式，兼容所有CARLA版本）
CITYSCAPES_PALETTE = [
    (0, 0, 0),          # 0: 未标注
    (70, 70, 70),       # 1: 建筑物
    (100, 40, 40),      # 2: 围栏
    (55, 90, 80),       # 3: 其他
    (220, 20, 60),      # 4: 行人
    (153, 153, 153),    # 5: 杆子
    (157, 234, 50),     # 6: 道路线
    (128, 64, 128),     # 7: 道路
    (244, 35, 232),     # 8: 人行道
    (107, 142, 35),     # 9: 植被
    (0, 0, 142),        # 10: 车辆
    (102, 102, 156),    # 11: 墙壁
    (220, 220, 0),      # 12: 交通灯
    (70, 130, 180),     # 13: 交通标志
    (81, 0, 81),        # 14: 天
    (150, 100, 100),    # 15: 地形
    (230, 150, 140),    # 16: 护栏
    (180, 165, 180),    # 17: 栅栏
    (250, 170, 30),     # 18: 静态
    (110, 190, 160),    # 19: 动态
    (170, 120, 50),     # 20: 其他
    (45, 60, 150),      # 21: 水
    (145, 170, 100)     # 22: 路面标记
]

# 1. 连接CARLA服务器并配置强同步模式
client = carla.Client('localhost', 2000)
client.set_timeout(15.0)
world = client.load_world('Town05')

# 启用严格同步模式
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1/30
settings.no_rendering_mode = False
world.apply_settings(settings)

# 2. 初始化同步锁与帧数据缓存
frame_lock = Lock()
latest_snapshot = None

# 绑定帧同步回调
def on_world_tick(snapshot):
    global latest_snapshot
    with frame_lock:
        latest_snapshot = snapshot
world.on_tick(on_world_tick)

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

# 3. 生成主角车辆（Tesla Model3）
model3_bp = bp_lib.find('vehicle.tesla.model3')
vehicle = None
for _ in range(5):
    try:
        vehicle = world.spawn_actor(model3_bp, random.choice(spawn_points))
        print(f"主角车辆生成成功（ID: {vehicle.id}）")
        break
    except:
        time.sleep(0.5)
if not vehicle:
    raise Exception("主角车辆生成失败，请重启CARLA服务器")

# 4. 初始化RGB摄像头（保留shutter_speed，该传感器支持）
def init_rgb_camera(vehicle):
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1024')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('shutter_speed', '100')  # RGB摄像头支持此属性
    camera_transform = carla.Transform(
        carla.Location(x=2.0, z=1.5),
        carla.Rotation(pitch=-5)
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    print("RGB摄像头初始化完成")
    return camera, image_queue

# 5. 初始化语义分割摄像头（移除shutter_speed，该传感器不支持）
def init_semantic_camera(vehicle):
    """初始化语义分割摄像头，返回传感器和数据队列"""
    sem_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    # 仅保留语义分割摄像头支持的属性
    sem_bp.set_attribute('image_size_x', '1024')
    sem_bp.set_attribute('image_size_y', '720')
    sem_bp.set_attribute('fov', '90')
    # 移除shutter_speed设置（语义分割摄像头不支持）
    sem_transform = carla.Transform(
        carla.Location(x=2.0, z=1.5),
        carla.Rotation(pitch=-5)
    )
    sem_camera = world.spawn_actor(sem_bp, sem_transform, attach_to=vehicle)
    sem_queue = queue.Queue()
    sem_camera.listen(sem_queue.put)  # 语义数据存入队列
    print("语义分割摄像头初始化完成")
    return sem_camera, sem_queue

# 初始化摄像头（同时初始化RGB和语义分割）
rgb_camera, rgb_queue = init_rgb_camera(vehicle)
sem_camera, sem_queue = init_semantic_camera(vehicle)  # 新增语义摄像头

# 6. 生成NPC车辆（保留原有逻辑）
npc_count = 100
print(f"开始生成{npc_count}辆NPC车辆...")
for i in range(npc_count):
    vehicle_bp = random.choice(bp_lib.filter('vehicle'))
    if 'tesla' in vehicle_bp.id:
        continue
    spawn_point = random.choice(spawn_points)
    if spawn_point.location.distance(vehicle.get_location()) < 20:
        continue
    world.try_spawn_actor(vehicle_bp, spawn_point)
    if i % 20 == 0:
        world.tick()
        time.sleep(0.1)

# 统计实际生成数量
all_vehicles = world.get_actors().filter('*vehicle*')
actual_npc_count = len(all_vehicles) - 1
print(f"NPC生成完成 | 实际数量: {actual_npc_count}辆（总车辆: {len(all_vehicles)}）")

# 7. 启动所有车辆自动驾驶
tm = client.get_trafficmanager(8000)
tm.set_synchronous_mode(True)
for v in all_vehicles:
    v.set_autopilot(True, tm.get_port())

# 8. 平滑视角函数（保留原有逻辑）
def set_spectator_smooth(last_transform=None):
    spectator = world.get_spectator()
    with frame_lock:
        if not latest_snapshot:
            return last_transform
        vehicle_snapshot = latest_snapshot.find(vehicle.id)
        if not vehicle_snapshot:
            return last_transform
        vehicle_tf = vehicle_snapshot.get_transform()
    
    target_tf = carla.Transform(
        vehicle_tf.transform(carla.Location(x=-8, z=3, y=0.5)),
        vehicle_tf.rotation
    )
    
    if last_transform is None:
        spectator.set_transform(target_tf)
        return target_tf
    
    smooth_loc = carla.Location(
        x=lerp(last_transform.location.x, target_tf.location.x, 0.15),
        y=lerp(last_transform.location.y, target_tf.location.y, 0.15),
        z=lerp(last_transform.location.z, target_tf.location.z, 0.15)
    )
    smooth_rot = carla.Rotation(
        pitch=lerp(last_transform.rotation.pitch, target_tf.rotation.pitch, 0.15),
        yaw=lerp(last_transform.rotation.yaw, target_tf.rotation.yaw, 0.15),
        roll=lerp(last_transform.rotation.roll, target_tf.rotation.roll, 0.15)
    )
    smooth_tf = carla.Transform(smooth_loc, smooth_rot)
    spectator.set_transform(smooth_tf)
    return smooth_tf

# 9. 主循环（处理图像显示）
print("\n程序运行中，按Ctrl+C或任一窗口按'q'退出...")
print("功能：RGB摄像头 + 语义分割摄像头 + 车辆自动驾驶 + 平滑视角")
last_spectator_tf = None
clock = pygame.time.Clock()

try:
    world.tick()
    last_spectator_tf = set_spectator_smooth()
    
    while True:
        world.tick()
        last_spectator_tf = set_spectator_smooth(last_spectator_tf)
        
        # 处理RGB图像
        if not rgb_queue.empty():
            rgb_image = rgb_queue.get()
            rgb_img = np.reshape(np.copy(rgb_image.raw_data), 
                                (rgb_image.height, rgb_image.width, 4))
            cv2.imshow('RGB Camera', rgb_img)
            if cv2.waitKey(1) == ord('q'):
                break
        
        # 处理语义分割图像
        if not sem_queue.empty():
            sem_image = sem_queue.get()
            # 提取语义分割原始数据（单通道类别ID）
            sem_data = np.reshape(np.copy(sem_image.raw_data), 
                                (sem_image.height, sem_image.width, 4))[:, :, 2].astype(np.int32)
            # 映射到Cityscapes调色板（转换为RGB可视化）
            sem_rgb = np.zeros((sem_image.height, sem_image.width, 3), dtype=np.uint8)
            for i in range(len(CITYSCAPES_PALETTE)):
                sem_rgb[sem_data == i] = CITYSCAPES_PALETTE[i]
            cv2.imshow('Semantic Segmentation', sem_rgb)
            if cv2.waitKey(1) == ord('q'):
                break
        
        clock.tick(30)

except KeyboardInterrupt:
    print("\n用户中断，清理资源...")
finally:
    # 清理所有传感器
    rgb_camera.stop()
    rgb_camera.destroy()
    sem_camera.stop()
    sem_camera.destroy()
    
    # 恢复CARLA设置
    settings.synchronous_mode = False
    tm.set_synchronous_mode(False)
    world.apply_settings(settings)
    
    # 销毁所有车辆
    for v in all_vehicles:
        if v.is_alive:
            v.destroy()
    
    cv2.destroyAllWindows()
    print("资源清理完成，同步模式已关闭")