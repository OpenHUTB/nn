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
    (0, 0, 0),          # 0: Unlabeled
    (70, 70, 70),       # 1: Building
    (100, 40, 40),      # 2: Fence
    (55, 90, 80),       # 3: Other
    (220, 20, 60),      # 4: Pedestrian (red)
    (153, 153, 153),    # 5: Pole
    (157, 234, 50),     # 6: RoadLine
    (128, 64, 128),     # 7: Road
    (244, 35, 232),     # 8: Sidewalk
    (107, 142, 35),     # 9: Vegetation
    (0, 0, 142),        # 10: Vehicle (blue)
    (102, 102, 156),    # 11: Wall
    (220, 220, 0),      # 12: TrafficLight
    (70, 130, 180),     # 13: TrafficSign
    (81, 0, 81),        # 14: Sky
    (150, 100, 100),    # 15: Terrain
    (230, 150, 140),    # 16: GuardRail
    (180, 165, 180),    # 17: Fence
    (250, 170, 30),     # 18: Static
    (110, 190, 160),    # 19: Dynamic
    (170, 120, 50),     # 20: Other
    (45, 60, 150),      # 21: Water
    (145, 170, 100)     # 22: RoadMarking
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

# 4. 初始化RGB摄像头
def init_rgb_camera(vehicle):
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1024')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('shutter_speed', '100')
    camera_transform = carla.Transform(
        carla.Location(x=2.0, z=1.5),
        carla.Rotation(pitch=-5)
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    print("RGB摄像头初始化完成")
    return camera, image_queue

# 5. 初始化语义分割摄像头
def init_semantic_camera(vehicle):
    sem_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute('image_size_x', '1024')
    sem_bp.set_attribute('image_size_y', '720')
    sem_bp.set_attribute('fov', '90')
    sem_transform = carla.Transform(
        carla.Location(x=2.0, z=1.5),
        carla.Rotation(pitch=-5)
    )
    sem_camera = world.spawn_actor(sem_bp, sem_transform, attach_to=vehicle)
    sem_queue = queue.Queue()
    sem_camera.listen(sem_queue.put)
    print("语义分割摄像头初始化完成")
    return sem_camera, sem_queue

# 初始化摄像头
rgb_camera, rgb_queue = init_rgb_camera(vehicle)
sem_camera, sem_queue = init_semantic_camera(vehicle)

# 6. 生成NPC车辆
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

# ==================== 行人生成核心逻辑 ====================
# 6.1 生成行人（walker）
walker_count = 50  # 生成50个行人（可调整）
walkers = []       # 存储行人actor
walker_controllers = []  # 存储行人控制器（用于移动）
print(f"\n开始生成{walker_count}个行人...")

# 获取行人蓝图（随机选择不同行人模型）
walker_bps = bp_lib.filter('walker.pedestrian.*')

# 获取行人生成点（使用地图的行人专用生成点，或随机点）
walker_spawn_points = []
for _ in range(walker_count * 2):  # 生成双倍候选点，避免重叠
    spawn_point = carla.Transform()
    # 随机位置（围绕主角车，半径50-200米，避免太近）
    spawn_point.location = world.get_random_location_from_navigation()
    if spawn_point.location is not None:
        # 确保行人不在车辆正前方（避免生成失败）
        if spawn_point.location.distance(vehicle.get_location()) > 20:
            walker_spawn_points.append(spawn_point)

# 生成行人
for i in range(walker_count):
    if i >= len(walker_spawn_points):
        break  # 候选点用完则停止
    walker_bp = random.choice(walker_bps)
    # 设置行人为不可碰撞（避免卡死）
    walker_bp.set_attribute('is_invincible', 'false')
    try:
        walker = world.spawn_actor(walker_bp, walker_spawn_points[i])
        walkers.append(walker)
        # 每生成10个行人同步一次，避免服务器卡顿
        if i % 10 == 0:
            world.tick()
            time.sleep(0.05)
    except:
        continue

# 6.2 生成行人控制器并启动自主移动
if walkers:
    # 获取行人控制器蓝图
    controller_bp = bp_lib.find('controller.ai.walker')
    # 启动交通管理器（行人也需要同步）
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    
    for walker in walkers:
        # 生成控制器并绑定到行人
        controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
        walker_controllers.append(controller)
        # 启动行人自主行走（随机目标点，速度1-3 m/s）
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(random.uniform(1.0, 3.0))

# 统计实际生成的行人数量
actual_walker_count = len(walkers)
print(f"行人生成完成 | 实际数量: {actual_walker_count}个")
# =================================================================

# 7. 启动所有车辆自动驾驶
tm = client.get_trafficmanager(8000)
tm.set_synchronous_mode(True)
for v in all_vehicles:
    v.set_autopilot(True, tm.get_port())

# 8. 平滑视角函数
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

# 9. 主循环（核心：RGB与语义分割图像拼接显示 + 性能监控）
print("\n程序运行中，按Ctrl+C或窗口按'q'退出...")
print(f"功能：RGB与语义分割图像拼接显示 + {actual_npc_count}辆车辆 + {actual_walker_count}个行人 + 平滑视角 + 性能监控")
last_spectator_tf = None
clock = pygame.time.Clock()

# ==================== 新增：性能监控初始化 ====================
start_time = time.time()
frame_counter = 0
current_fps = 0.0
# =================================================================

try:
    world.tick()
    last_spectator_tf = set_spectator_smooth()
    
    while True:
        world.tick()
        last_spectator_tf = set_spectator_smooth(last_spectator_tf)
        
        # ==================== 新增：实时FPS计算 ====================
        frame_counter += 1
        # 每30帧更新一次FPS（避免频繁计算）
        if frame_counter % 30 == 0:
            elapsed_time = time.time() - start_time
            current_fps = 30.0 / elapsed_time if elapsed_time > 0 else 0.0
            start_time = time.time()
            frame_counter = 0
        # =================================================================
        
        # 同时获取RGB和语义分割图像（确保帧同步）
        if not rgb_queue.empty() and not sem_queue.empty():
            # 处理RGB图像（移除alpha通道，保留RGB）
            rgb_image = rgb_queue.get()
            rgb_img = np.reshape(np.copy(rgb_image.raw_data), 
                                (rgb_image.height, rgb_image.width, 4))[:, :, :3]  # 取前3通道（RGB）
            
            # 处理语义分割图像（转换为彩色可视化）
            sem_image = sem_queue.get()
            sem_data = np.reshape(np.copy(sem_image.raw_data), 
                                (sem_image.height, sem_image.width, 4))[:, :, 2].astype(np.int32)
            sem_rgb = np.zeros((sem_image.height, sem_image.width, 3), dtype=np.uint8)
            for i in range(len(CITYSCAPES_PALETTE)):
                sem_rgb[sem_data == i] = CITYSCAPES_PALETTE[i]
            
            # 横向拼接两张图像（宽度合并，高度不变）
            combined_img = cv2.hconcat([rgb_img, sem_rgb])
            
            # ==================== 修复：调整标题位置，为性能监控腾出空间 ====================
            cv2.putText(combined_img, 
                       f"RGB Image | Semantic Segmentation (Vehicles:{actual_npc_count} Pedestrians:{actual_walker_count})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            # =================================================================
            
            # ==================== 修复：性能监控移到左上角标题下方，避免被裁剪 ====================
            # 性能信息列表
            perf_info = [
                f"FPS: {current_fps:.1f}",
                f"RGB Queue: {rgb_queue.qsize()}",
                f"Sem Queue: {sem_queue.qsize()}",
                f"Sync Frame: {world.get_snapshot().frame}",
                f"Fixed Delta: {settings.fixed_delta_seconds:.3f}s"
            ]
            
            # 绘制位置：左上角，标题下方（y从60开始）
            perf_x = 10
            perf_y = 60  # 标题在30位置，这里从60开始
            perf_line_height = 25
            perf_color = (0, 255, 255)  # 黄色字体
            
            for idx, info in enumerate(perf_info):
                y_pos = perf_y + idx * perf_line_height
                # 绘制半透明黑色背景（提升可读性）
                cv2.rectangle(combined_img, 
                              (perf_x - 5, y_pos - 15), 
                              (perf_x + 220, y_pos + 5), 
                              (0, 0, 0), -1)
                # 绘制性能文本
                cv2.putText(combined_img, info, 
                           (perf_x, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, perf_color, 2)
            # =================================================================
            
            # 显示拼接后的图像
            cv2.imshow('CARLA RGB + Semantic Segmentation (with Pedestrians & Performance)', combined_img)
            if cv2.waitKey(1) == ord('q'):
                break
        
        clock.tick(30)

except KeyboardInterrupt:
    print("\n用户中断，清理资源...")
finally:
    # ==================== 清理行人资源 ====================
    # 停止并销毁行人控制器
    for controller in walker_controllers:
        if controller.is_alive:
            controller.stop()
            controller.destroy()
    # 销毁行人
    for walker in walkers:
        if walker.is_alive:
            walker.destroy()
    print(f"已销毁{len(walker_controllers)}个行人控制器 + {len(walkers)}个行人")
    # =================================================================
    
    # 清理传感器
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
    print(f"资源清理完成，同步模式已关闭（销毁{len(all_vehicles)}辆车辆）")