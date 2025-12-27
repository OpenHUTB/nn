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

# ==================== 新增：语义密度热力图生成函数 ====================
def generate_density_heatmap(sem_data, target_classes=[4, 10], width=1024, height=720):
    """
    生成指定语义类别的密度热力图（行人+车辆为默认目标）
    :param sem_data: 语义分割原始数据（int32数组，shape=(H,W)）
    :param target_classes: 目标语义类别列表（4=行人，10=车辆）
    :param width/height: 图像分辨率
    :return: 彩色密度热力图（RGB格式）
    """
    # 1. 生成目标类别掩码（仅保留行人和车辆）
    mask = np.zeros((height, width), dtype=np.uint8)
    for cls in target_classes:
        mask[sem_data == cls] = 255  # 目标类别像素设为255，背景0
    
    # 2. 高斯模糊平滑（模拟密度分布，核越大越平滑）
    blurred_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    # 3. 转换为彩色热力图（JET色板：蓝→青→黄→红，代表密度从低到高）
    heatmap = cv2.applyColorMap(blurred_mask, cv2.COLORMAP_JET)
    
    # 4. 优化视觉效果：降低背景透明度，突出目标区域
    heatmap = cv2.addWeighted(heatmap, 0.9, np.zeros_like(heatmap), 0.1, 0)
    
    # 5. 添加热力图标注（右下角说明）
    cv2.putText(heatmap, "Density: Pedestrian(Red) + Vehicle(Blue)", 
               (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return heatmap
# =================================================================

# ==================== 第11次提交新增：语义类别实时计数函数 ====================
def semantic_class_count(sem_data, class_mapping):
    """
    统计指定语义类别的像素数量，并估算画面内目标数量（按单目标像素阈值）
    :param sem_data: 语义分割原始数据（H,W），int32类型
    :param class_mapping: 字典 {类别名: 类别ID}
    :return: 计数结果字典 {类别名: 近似目标数}
    """
    count_dict = {}
    # 单目标像素阈值（经验值：行人≈200像素，车辆≈500像素，交通灯≈50像素）
    pixel_thresholds = {
        "Pedestrian": 200,
        "Vehicle": 500,
        "TrafficLight": 50
    }
    
    for cls_name, cls_id in class_mapping.items():
        # 统计该类别像素总数
        pixel_count = np.sum(sem_data == cls_id)
        # 估算近似目标数（避免0除，最少计为0）
        threshold = pixel_thresholds.get(cls_name, 200)
        approx_count = pixel_count // threshold if pixel_count >= threshold else 0
        count_dict[cls_name] = approx_count
    return count_dict
# =================================================================

# ==================== 第12次提交新增：关键语义目标高亮标注函数 ====================
def semantic_target_highlight(rgb_img, sem_data, highlight_classes={4:(0,0,255), 10:(255,0,0)}, contour_thickness=2):
    """
    对指定语义类别进行边缘检测并绘制轮廓，高亮关键目标
    :param rgb_img: 原始RGB图像（H,W,3）
    :param sem_data: 语义分割原始数据（H,W），int32类型
    :param highlight_classes: 字典 {类别ID: 轮廓颜色}，默认4=行人(红)、10=车辆(蓝)
    :param contour_thickness: 轮廓线宽度
    :return: 叠加高亮轮廓的RGB图像
    """
    # 复制原始图像，避免修改原数据
    highlighted_img = rgb_img.copy()
    
    for cls_id, color in highlight_classes.items():
        # 1. 生成该类别的二值掩码
        cls_mask = np.uint8(sem_data == cls_id) * 255
        # 2. Canny边缘检测（调整阈值控制边缘灵敏度）
        edges = cv2.Canny(cls_mask, 50, 150)
        # 3. 查找轮廓（仅保留外部轮廓，减少计算量）
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 4. 绘制轮廓（过滤小轮廓，避免噪声）
        min_contour_area = 50  # 过滤面积小于50像素的噪声轮廓
        for cnt in contours:
            if cv2.contourArea(cnt) > min_contour_area:
                cv2.drawContours(highlighted_img, [cnt], -1, color, contour_thickness)
    
    # 5. 添加高亮说明文字（画面左下角）
    highlight_tips = "Highlight: Pedestrian(Red) | Vehicle(Blue)"
    cv2.putText(highlighted_img, highlight_tips, 
               (10, rgb_img.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return highlighted_img
# =================================================================

# ==================== 第14次提交核心：语义分割与RGB融合叠加函数 ====================
def semantic_rgb_fusion(rgb_img, sem_data, palette=CITYSCAPES_PALETTE, alpha=0.3):
    """
    实现RGB图像与语义分割图像的融合叠加显示
    :param rgb_img: 原始RGB图像（H,W,3）
    :param sem_data: 语义分割原始数据（H,W），int32类型
    :param palette: 语义分割调色板
    :param alpha: 语义层透明度（0~1，0=仅RGB，1=仅语义）
    :return: 融合后的图像（H,W,3）
    """
    # 1. 生成语义分割彩色图
    sem_rgb = np.zeros_like(rgb_img)
    for i in range(len(palette)):
        sem_rgb[sem_data == i] = palette[i]
    
    # 2. 融合RGB与语义分割图（alpha=语义层权重，beta=RGB层权重）
    fused_img = cv2.addWeighted(sem_rgb, alpha, rgb_img, 1 - alpha, 0)
    
    # 3. 添加融合说明文字（左上角）
    fusion_tips = f"Semantic-RGB Fusion (Alpha={alpha:.1f})"
    cv2.putText(fused_img, fusion_tips, 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 4. 添加关键类别颜色说明（右下角）
    class_tips = "Pedestrian(Red) | Vehicle(Blue) | Road(Purple)"
    cv2.putText(fused_img, class_tips, 
               (10, rgb_img.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return fused_img
# =================================================================

# ==================== 第15次提交新增：可视化模式提示生成函数 ====================
def generate_mode_hint(current_mode, fusion_alpha):
    """
    生成当前可视化模式的提示文字
    :param current_mode: 当前模式编号
    :param fusion_alpha: 融合透明度
    :return: 提示文字列表
    """
    mode_names = {
        1: "Basic Mode (RGB+Sem / Top+Heatmap)",
        2: "Fusion Mode (RGB+Fusion / Sem+Heatmap)",
        3: "Simplified Mode (RGB+Heatmap / Top+Count)",
        4: "Full Sem Mode (Sem+Fusion / Heatmap+Top)"
    }
    control_hints = [
        "Controls: 1/2/3/4=Switch Mode | ↑↓=Adjust Fusion Alpha | R=Reset | Q=Quit",
        f"Current Mode: {mode_names.get(current_mode, 'Basic Mode')}",
        f"Fusion Alpha: {fusion_alpha:.1f} (0=RGB Only, 1=Sem Only)"
    ]
    return control_hints
# =================================================================

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

# 4. 初始化摄像头通用函数（复用代码，修复shutter_speed属性问题）
def init_camera(vehicle, camera_type, transform, width=1024, height=720, fov=90):
    """
    初始化摄像头（RGB/语义分割）
    :param vehicle: 挂载的车辆
    :param camera_type: 摄像头类型（'rgb'/'semantic'）
    :param transform: 摄像头位姿
    :param width/height: 图像分辨率
    :param fov: 视场角
    :return: 摄像头actor + 数据队列
    """
    if camera_type == 'rgb':
        camera_bp = bp_lib.find('sensor.camera.rgb')
    elif camera_type == 'semantic':
        camera_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    else:
        raise ValueError("camera_type must be 'rgb' or 'semantic'")
    
    # 通用属性（所有摄像头都支持）
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera_bp.set_attribute('fov', str(fov))
    
    # 仅RGB摄像头设置shutter_speed（语义分割摄像头不支持）
    if camera_type == 'rgb':
        camera_bp.set_attribute('shutter_speed', '100')
    
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    print(f"{camera_type.upper()}摄像头初始化完成（{transform.location}）")
    return camera, image_queue

# 4.1 初始化前视RGB摄像头（原有）
front_rgb_transform = carla.Transform(
    carla.Location(x=2.0, z=1.5),
    carla.Rotation(pitch=-5)
)
front_rgb_camera, front_rgb_queue = init_camera(vehicle, 'rgb', front_rgb_transform)

# 4.2 初始化前视语义分割摄像头（修复shutter_speed问题）
front_sem_transform = carla.Transform(
    carla.Location(x=2.0, z=1.5),
    carla.Rotation(pitch=-5)
)
front_sem_camera, front_sem_queue = init_camera(vehicle, 'semantic', front_sem_transform)

# 4.3 新增：初始化俯视RGB摄像头（鸟瞰视角）
top_rgb_transform = carla.Transform(
    carla.Location(x=0.0, z=8.0),  # 车辆正上方8米
    carla.Rotation(pitch=-90)      # 垂直向下俯视
)
top_rgb_camera, top_rgb_queue = init_camera(vehicle, 'rgb', top_rgb_transform, fov=120)  # 广角120°覆盖更多区域

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

# 9. 主循环（核心：多视角+热力图+计数+高亮+融合+键盘交互可视化模式）
print("\n程序运行中，按以下按键操作：")
print("1/2/3/4 = 切换可视化模式 | ↑/↓ = 调整融合透明度 | R = 重置模式 | Q = 退出")
print(f"功能：多模式切换 + 语义计数 + 目标高亮 + 语义-RGB融合 + {actual_npc_count}辆车辆 + {actual_walker_count}个行人 + 性能监控")
last_spectator_tf = None
clock = pygame.time.Clock()

# ==================== 性能监控初始化 ====================
start_time = time.time()
frame_counter = 0
current_fps = 0.0
# ==================== 第11次提交新增：定义需要计数的语义类别 ====================
count_class_mapping = {
    "Pedestrian": 4,    # 行人
    "Vehicle": 10,      # 车辆
    "TrafficLight": 12  # 交通灯
}
# ==================== 第12次提交新增：定义需要高亮的语义类别 ====================
highlight_class_mapping = {
    4: (0, 0, 255),     # 行人 - 红色轮廓
    10: (255, 0, 0)     # 车辆 - 蓝色轮廓
}
# ==================== 第15次提交：可视化模式配置 ====================
current_mode = 1  # 默认模式1（基础模式）
fusion_alpha = 0.3  # 融合透明度（0~1）
mode_reset_flag = False  # 模式重置标记
# =================================================================

try:
    world.tick()
    last_spectator_tf = set_spectator_smooth()
    
    while True:
        world.tick()
        last_spectator_tf = set_spectator_smooth(last_spectator_tf)
        
        # ==================== 实时FPS计算 ====================
        frame_counter += 1
        if frame_counter % 30 == 0:
            elapsed_time = time.time() - start_time
            current_fps = 30.0 / elapsed_time if elapsed_time > 0 else 0.0
            start_time = time.time()
            frame_counter = 0
        # =================================================================
        
        # 同时获取三个摄像头数据（帧同步：前视RGB + 前视语义 + 俯视RGB）
        if not front_rgb_queue.empty() and not front_sem_queue.empty() and not top_rgb_queue.empty():
            # 1. 处理前视RGB图像
            front_rgb_image = front_rgb_queue.get()
            front_rgb_img = np.reshape(np.copy(front_rgb_image.raw_data), 
                                     (720, 1024, 4))[:, :, :3]
            
            # 2. 处理前视语义分割图像
            front_sem_image = front_sem_queue.get()
            front_sem_data = np.reshape(np.copy(front_sem_image.raw_data), 
                                      (720, 1024, 4))[:, :, 2].astype(np.int32)
            front_sem_rgb = np.zeros((720, 1024, 3), dtype=np.uint8)
            for i in range(len(CITYSCAPES_PALETTE)):
                front_sem_rgb[front_sem_data == i] = CITYSCAPES_PALETTE[i]
            
            # 3. 处理俯视RGB图像
            top_rgb_image = top_rgb_queue.get()
            top_rgb_img = np.reshape(np.copy(top_rgb_image.raw_data), 
                                    (720, 1024, 4))[:, :, :3]
            
            # ==================== 生成语义密度热力图 ====================
            density_heatmap = generate_density_heatmap(front_sem_data, target_classes=[4, 10])
            # =================================================================
            
            # ==================== 计算语义类别计数 ====================
            class_count_result = semantic_class_count(front_sem_data, count_class_mapping)
            # =================================================================
            
            # ==================== 生成高亮RGB图像 ====================
            front_rgb_img_highlight = semantic_target_highlight(front_rgb_img.copy(), front_sem_data, highlight_class_mapping)
            # =================================================================
            
            # ==================== 生成语义-RGB融合图像 ====================
            fused_img = semantic_rgb_fusion(front_rgb_img, front_sem_data, alpha=fusion_alpha)
            # =================================================================
            
            # ==================== 第15次提交核心：根据当前模式动态拼接图像 ====================
            if current_mode == 1:
                # 模式1：基础模式（原有布局）
                # 上=RGB高亮 + 语义分割 | 下=俯视RGB + 热力图
                upper_part = cv2.hconcat([front_rgb_img_highlight, front_sem_rgb])
                lower_part = cv2.hconcat([top_rgb_img, density_heatmap])
            elif current_mode == 2:
                # 模式2：融合模式
                # 上=RGB高亮 + 语义融合 | 下=语义分割 + 热力图
                upper_part = cv2.hconcat([front_rgb_img_highlight, fused_img])
                lower_part = cv2.hconcat([front_sem_rgb, density_heatmap])
            elif current_mode == 3:
                # 模式3：精简模式
                # 上=RGB高亮 + 热力图 | 下=俯视RGB + 计数可视化（生成纯黑背景+计数文字）
                count_vis = np.zeros((720, 1024, 3), dtype=np.uint8)
                count_title = "Semantic Count (Frame)"
                cv2.putText(count_vis, count_title, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                count_items = [
                    f"Pedestrian: {class_count_result['Pedestrian']}",
                    f"Vehicle: {class_count_result['Vehicle']}",
                    f"TrafficLight: {class_count_result['TrafficLight']}"
                ]
                for idx, item in enumerate(count_items):
                    cv2.putText(count_vis, item, (50, 200 + idx * 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                upper_part = cv2.hconcat([front_rgb_img_highlight, density_heatmap])
                lower_part = cv2.hconcat([top_rgb_img, count_vis])
            elif current_mode == 4:
                # 模式4：全语义模式
                # 上=语义分割 + 融合图 | 下=热力图 + 俯视RGB
                upper_part = cv2.hconcat([front_sem_rgb, fused_img])
                lower_part = cv2.hconcat([density_heatmap, top_rgb_img])
            else:
                # 默认回退到模式1
                upper_part = cv2.hconcat([front_rgb_img_highlight, front_sem_rgb])
                lower_part = cv2.hconcat([top_rgb_img, density_heatmap])
            
            combined_img = cv2.vconcat([upper_part, lower_part])
            # =================================================================
            
            # 5. 添加视角标题（根据模式动态调整）
            mode_titles = {
                1: ["Front View (RGB + Target Highlight)", "Front View (Semantic Segmentation)",
                    "Top View (RGB / Bird's Eye)", "Density Heatmap (Pedestrian + Vehicle)"],
                2: ["Front View (RGB + Target Highlight)", "Front View (Semantic-RGB Fusion)",
                    "Front View (Semantic Segmentation)", "Density Heatmap (Pedestrian + Vehicle)"],
                3: ["Front View (RGB + Target Highlight)", "Density Heatmap (Pedestrian + Vehicle)",
                    "Top View (RGB / Bird's Eye)", "Semantic Count Visualization"],
                4: ["Front View (Semantic Segmentation)", "Front View (Semantic-RGB Fusion)",
                    "Density Heatmap (Pedestrian + Vehicle)", "Top View (RGB / Bird's Eye)"]
            }
            titles = mode_titles.get(current_mode, mode_titles[1])
            # 上半部分左标题
            cv2.putText(combined_img, titles[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            # 上半部分右标题
            cv2.putText(combined_img, titles[1], (1024 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            # 下半部分左标题
            cv2.putText(combined_img, titles[2], (10, 720 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            # 下半部分右标题
            cv2.putText(combined_img, titles[3], (1024 + 10, 720 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # 6. 绘制性能监控 + 模式控制提示（左上角）
            perf_info = [
                f"FPS: {current_fps:.1f}",
                f"Sync Frame: {world.get_snapshot().frame}",
                f"Vehicles: {actual_npc_count} | Pedestrians: {actual_walker_count}"
            ]
            # 生成模式控制提示
            mode_hints = generate_mode_hint(current_mode, fusion_alpha)
            all_info = perf_info + mode_hints
            
            perf_x = 10
            perf_y = 60
            perf_line_height = 25
            perf_color = (0, 255, 255)  # 黄色
            for idx, info in enumerate(all_info):
                y_pos = perf_y + idx * perf_line_height
                # 半透明背景
                cv2.rectangle(combined_img, 
                              (perf_x - 5, y_pos - 15), 
                              (perf_x + 600, y_pos + 5), 
                              (0, 0, 0), -1)
                cv2.putText(combined_img, info, (perf_x, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, perf_color, 2)
            
            # 7. 绘制语义计数面板（仅模式1/2/4显示，模式3已集成到布局）
            if current_mode in [1, 2, 4]:
                count_x = combined_img.shape[1] - 320
                count_y = 30
                count_color = (255, 255, 0)
                count_bg_color = (0, 0, 0)
                # 绘制计数面板背景
                cv2.rectangle(combined_img, 
                              (count_x - 10, count_y - 10), 
                              (combined_img.shape[1] - 10, count_y + 100), 
                              count_bg_color, -1)
                # 绘制计数标题和内容
                cv2.putText(combined_img, "Semantic Count (Frame)", 
                           (count_x, count_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, count_color, 2)
                count_items = [
                    f"Pedestrian: {class_count_result['Pedestrian']}",
                    f"Vehicle: {class_count_result['Vehicle']}",
                    f"TrafficLight: {class_count_result['TrafficLight']}"
                ]
                for idx, item in enumerate(count_items):
                    y_pos = count_y + (idx + 1) * 28
                    cv2.putText(combined_img, item, (count_x, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, count_color, 2)
            
            # 8. 显示最终图像
            cv2.namedWindow('CARLA Multi-Mode Visualization (Keyboard Interactive)', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('CARLA Multi-Mode Visualization (Keyboard Interactive)', 1920, 1080)
            cv2.imshow('CARLA Multi-Mode Visualization (Keyboard Interactive)', combined_img)
            
            # ==================== 第15次提交核心：键盘事件处理 ====================
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_mode = 1
                print(f"切换到模式1：基础模式")
            elif key == ord('2'):
                current_mode = 2
                print(f"切换到模式2：融合模式（当前Alpha={fusion_alpha:.1f}）")
            elif key == ord('3'):
                current_mode = 3
                print(f"切换到模式3：精简模式")
            elif key == ord('4'):
                current_mode = 4
                print(f"切换到模式4：全语义模式")
            elif key == ord('r') or key == ord('R'):
                # 重置模式和透明度
                current_mode = 1
                fusion_alpha = 0.3
                mode_reset_flag = True
                print(f"已重置：模式1 + 融合Alpha=0.3")
            elif key == 2490368:  # 上方向键（增加Alpha）
                fusion_alpha = min(fusion_alpha + 0.1, 1.0)
                print(f"融合Alpha调整为：{fusion_alpha:.1f}")
            elif key == 2621440:  # 下方向键（减少Alpha）
                fusion_alpha = max(fusion_alpha - 0.1, 0.0)
                print(f"融合Alpha调整为：{fusion_alpha:.1f}")
            # =================================================================
        
        clock.tick(30)

except KeyboardInterrupt:
    print("\n用户中断，清理资源...")
finally:
    # ==================== 清理所有摄像头资源 ====================
    # 前视RGB
    front_rgb_camera.stop()
    front_rgb_camera.destroy()
    # 前视语义
    front_sem_camera.stop()
    front_sem_camera.destroy()
    # 俯视RGB
    top_rgb_camera.stop()
    top_rgb_camera.destroy()
    # =================================================================
    
    # ==================== 清理行人资源 ====================
    for controller in walker_controllers:
        if controller.is_alive:
            controller.stop()
            controller.destroy()
    for walker in walkers:
        if walker.is_alive:
            walker.destroy()
    print(f"已销毁{len(walker_controllers)}个行人控制器 + {len(walkers)}个行人")
    # =================================================================
    
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