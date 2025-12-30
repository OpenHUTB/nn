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

# ==================== 语义分割核心类别（用于量化评估，过滤无意义类别） ====================
EVAL_CLASSES = {
    "Pedestrian": 4,
    "Vehicle": 10,
    "Road": 7,
    "Sidewalk": 8,
    "Building": 1,
    "Vegetation": 9,
    "TrafficLight": 12,
    "TrafficSign": 13
}
# =================================================================

# ==================== 新增：语义密度热力图生成函数 ====================
def generate_density_heatmap(sem_data, target_classes=[4, 10], width=1024, height=720):
    """
    生成指定语义类别的密度热力图（行人+车辆为默认目标）
    :param sem_data: 语义分割原始数据（int32数组，shape=(H,W)）
    :param target_classes: 目标语义类别列表（4=行人，10=车辆）
    :param width/height: 图像分辨率
    :return: 彩色密度热力图（RGB格式）
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for cls in target_classes:
        mask[sem_data == cls] = 255
    blurred_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    heatmap = cv2.applyColorMap(blurred_mask, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(heatmap, 0.9, np.zeros_like(heatmap), 0.1, 0)
    cv2.putText(heatmap, "Density: Pedestrian(Red) + Vehicle(Blue)", 
               (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return heatmap
# =================================================================

# ==================== 第11次提交：语义类别实时计数函数 ====================
def semantic_class_count(sem_data, class_mapping):
    count_dict = {}
    pixel_thresholds = {"Pedestrian": 200, "Vehicle": 500, "TrafficLight": 50}
    for cls_name, cls_id in class_mapping.items():
        pixel_count = np.sum(sem_data == cls_id)
        threshold = pixel_thresholds.get(cls_name, 200)
        approx_count = pixel_count // threshold if pixel_count >= threshold else 0
        count_dict[cls_name] = approx_count
    return count_dict
# =================================================================

# ==================== 第12次提交：关键语义目标高亮标注函数 ====================
def semantic_target_highlight(rgb_img, sem_data, highlight_classes={4:(0,0,255), 10:(255,0,0)}, contour_thickness=2):
    highlighted_img = rgb_img.copy()
    for cls_id, color in highlight_classes.items():
        cls_mask = np.uint8(sem_data == cls_id) * 255
        edges = cv2.Canny(cls_mask, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 50
        for cnt in contours:
            if cv2.contourArea(cnt) > min_contour_area:
                cv2.drawContours(highlighted_img, [cnt], -1, color, contour_thickness)
    cv2.putText(highlighted_img, "Highlight: Pedestrian(Red) | Vehicle(Blue)", 
               (10, rgb_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return highlighted_img
# =================================================================

# ==================== 第14次提交：语义分割与RGB融合叠加函数 ====================
def semantic_rgb_fusion(rgb_img, sem_data, palette=CITYSCAPES_PALETTE, alpha=0.3):
    sem_rgb = np.zeros_like(rgb_img)
    for i in range(len(palette)):
        sem_rgb[sem_data == i] = palette[i]
    fused_img = cv2.addWeighted(sem_rgb, alpha, rgb_img, 1 - alpha, 0)
    cv2.putText(fused_img, f"Semantic-RGB Fusion (Alpha={alpha:.1f})", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(fused_img, "Pedestrian(Red) | Vehicle(Blue) | Road(Purple)", 
               (10, rgb_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return fused_img
# =================================================================

# ==================== 第16次提交核心：语义分割量化评估函数 ====================
def semantic_quantitative_evaluation(pred_sem_data, gt_sem_data, eval_classes):
    """
    实时计算语义分割量化指标（学术核心指标）
    :param pred_sem_data: 预测语义数据（可替换为实际模型输出，此处用CARLA语义模拟）
    :param gt_sem_data: 真实语义数据（CARLA语义摄像头输出，作为ground truth）
    :param eval_classes: 待评估类别字典 {类别名: 类别ID}
    :return: 评估结果字典（mIoU、PA、各类别IoU）
    """
    height, width = pred_sem_data.shape
    total_pixels = height * width
    class_iou = {}
    correct_pixels = 0  # 全局正确像素数（PA计算用）
    
    for cls_name, cls_id in eval_classes.items():
        # 计算TP（真阳性）、FP（假阳性）、FN（假阴性）
        tp = np.sum((pred_sem_data == cls_id) & (gt_sem_data == cls_id))
        fp = np.sum((pred_sem_data == cls_id) & (gt_sem_data != cls_id))
        fn = np.sum((pred_sem_data != cls_id) & (gt_sem_data == cls_id))
        
        # 计算IoU（避免0除）
        iou = tp / (tp + fp + fn + 1e-8)
        class_iou[cls_name] = round(iou, 3)
        
        # 累计全局正确像素
        correct_pixels += tp
    
    # 计算核心指标
    pa = correct_pixels / total_pixels  # 像素准确率（Pixel Accuracy）
    miou = np.mean(list(class_iou.values()))  # 平均交并比（mean IoU）
    
    return {
        "mIoU": round(miou, 3),
        "PA": round(pa, 3),
        "class_IoU": class_iou
    }

def generate_evaluation_visualization(eval_result, width=1024, height=720):
    """
    生成量化评估可视化面板（用于画面显示）
    :param eval_result: 评估结果字典
    :param width/height: 面板分辨率
    :return: 评估可视化图像（RGB格式）
    """
    eval_vis = np.zeros((height, width, 3), dtype=np.uint8)
    # 标题
    cv2.putText(eval_vis, "Semantic Segmentation Quantitative Evaluation", 
               (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    # 核心指标（mIoU、PA）
    core_metrics = [
        f"mIoU (mean IoU): {eval_result['mIoU']:.3f}",
        f"PA (Pixel Accuracy): {eval_result['PA']:.3f}"
    ]
    for idx, metric in enumerate(core_metrics):
        cv2.putText(eval_vis, metric, (50, 180 + idx * 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    # 各类别IoU（分行显示）
    cv2.putText(eval_vis, "Class-wise IoU:", (50, 350), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    class_iou_items = list(eval_result["class_IoU"].items())
    for idx, (cls_name, iou) in enumerate(class_iou_items):
        y_pos = 420 + (idx // 2) * 50
        x_pos = 50 if idx % 2 == 0 else 550
        text = f"{cls_name}: {iou:.3f}"
        cv2.putText(eval_vis, text, (x_pos, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return eval_vis
# =================================================================

# ==================== 第15次提交：可视化模式提示生成函数 ====================
def generate_mode_hint(current_mode, fusion_alpha, eval_enabled):
    mode_names = {
        1: "Basic Mode (RGB+Sem / Top+Heatmap)",
        2: "Fusion Mode (RGB+Fusion / Sem+Heatmap)",
        3: "Simplified Mode (RGB+Heatmap / Top+Count)",
        4: "Full Sem Mode (Sem+Fusion / Heatmap+Top)",
        5: "Evaluation Mode (RGB+Fusion / Eval+Heatmap)"  # 新增模式5
    }
    control_hints = [
        "Controls: 1-5=Switch Mode | ↑↓=Adjust Fusion Alpha | E=Toggle Evaluation | R=Reset | Q=Quit",
        f"Current Mode: {mode_names.get(current_mode, 'Basic Mode')}",
        f"Fusion Alpha: {fusion_alpha:.1f} | Evaluation: {'Enabled' if eval_enabled else 'Disabled'}"
    ]
    return control_hints
# =================================================================

# 1. CARLA服务器连接与配置
client = carla.Client('localhost', 2000)
client.set_timeout(15.0)
world = client.load_world('Town05')
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1/30
world.apply_settings(settings)

# 2. 同步锁与帧缓存
frame_lock = Lock()
latest_snapshot = None
def on_world_tick(snapshot):
    global latest_snapshot
    with frame_lock:
        latest_snapshot = snapshot
world.on_tick(on_world_tick)

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

# 3. 生成主角车辆
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

# 4. 摄像头初始化函数
def init_camera(vehicle, camera_type, transform, width=1024, height=720, fov=90):
    if camera_type == 'rgb':
        camera_bp = bp_lib.find('sensor.camera.rgb')
    elif camera_type == 'semantic':
        camera_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    else:
        raise ValueError("camera_type must be 'rgb' or 'semantic'")
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera_bp.set_attribute('fov', str(fov))
    if camera_type == 'rgb':
        camera_bp.set_attribute('shutter_speed', '100')
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    print(f"{camera_type.upper()}摄像头初始化完成（{transform.location}）")
    return camera, image_queue

# 4.1 前视RGB摄像头
front_rgb_transform = carla.Transform(carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=-5))
front_rgb_camera, front_rgb_queue = init_camera(vehicle, 'rgb', front_rgb_transform)

# 4.2 前视语义分割摄像头（同时作为GT和预测模拟）
front_sem_transform = carla.Transform(carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=-5))
front_sem_camera, front_sem_queue = init_camera(vehicle, 'semantic', front_sem_transform)

# 4.3 俯视RGB摄像头
top_rgb_transform = carla.Transform(carla.Location(x=0.0, z=8.0), carla.Rotation(pitch=-90))
top_rgb_camera, top_rgb_queue = init_camera(vehicle, 'rgb', top_rgb_transform, fov=120)

# 5. 生成NPC车辆
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
all_vehicles = world.get_actors().filter('*vehicle*')
actual_npc_count = len(all_vehicles) - 1
print(f"NPC生成完成 | 实际数量: {actual_npc_count}辆")

# 6. 生成行人
walker_count = 50
walkers = []
walker_controllers = []
print(f"\n开始生成{walker_count}个行人...")
walker_bps = bp_lib.filter('walker.pedestrian.*')
walker_spawn_points = []
for _ in range(walker_count * 2):
    spawn_point = carla.Transform()
    spawn_point.location = world.get_random_location_from_navigation()
    if spawn_point.location is not None and spawn_point.location.distance(vehicle.get_location()) > 20:
        walker_spawn_points.append(spawn_point)
for i in range(walker_count):
    if i >= len(walker_spawn_points):
        break
    walker_bp = random.choice(walker_bps)
    walker_bp.set_attribute('is_invincible', 'false')
    try:
        walker = world.spawn_actor(walker_bp, walker_spawn_points[i])
        walkers.append(walker)
        if i % 10 == 0:
            world.tick()
            time.sleep(0.05)
    except:
        continue
if walkers:
    controller_bp = bp_lib.find('controller.ai.walker')
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    for walker in walkers:
        controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
        walker_controllers.append(controller)
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(random.uniform(1.0, 3.0))
actual_walker_count = len(walkers)
print(f"行人生成完成 | 实际数量: {actual_walker_count}个")

# 7. 启动车辆自动驾驶
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

# 9. 主循环（集成量化评估功能）
print("\n程序运行中，按以下按键操作：")
print("1-5=切换可视化模式 | ↑/↓=调整融合透明度 | E=开启/关闭量化评估 | R=重置 | Q=退出")
print(f"功能：多模式切换+量化评估+语义融合+高亮计数+{actual_npc_count}辆车辆+{actual_walker_count}个行人")
last_spectator_tf = None
clock = pygame.time.Clock()

# 初始化参数
start_time = time.time()
frame_counter = 0
current_fps = 0.0
count_class_mapping = {"Pedestrian": 4, "Vehicle": 10, "TrafficLight": 12}
highlight_class_mapping = {4: (0, 0, 255), 10: (255, 0, 0)}
current_mode = 1
fusion_alpha = 0.3
eval_enabled = True  # 默认开启量化评估
# =================================================================

try:
    world.tick()
    last_spectator_tf = set_spectator_smooth()
    
    while True:
        world.tick()
        last_spectator_tf = set_spectator_smooth(last_spectator_tf)
        
        # 实时FPS计算
        frame_counter += 1
        if frame_counter % 30 == 0:
            elapsed_time = time.time() - start_time
            current_fps = 30.0 / elapsed_time if elapsed_time > 0 else 0.0
            start_time = time.time()
            frame_counter = 0
        
        # 获取摄像头数据
        if not front_rgb_queue.empty() and not front_sem_queue.empty() and not top_rgb_queue.empty():
            # 处理图像数据
            front_rgb_image = front_rgb_queue.get()
            front_rgb_img = np.reshape(np.copy(front_rgb_image.raw_data), (720, 1024, 4))[:, :, :3]
            
            front_sem_image = front_sem_queue.get()
            front_sem_data = np.reshape(np.copy(front_sem_image.raw_data), (720, 1024, 4))[:, :, 2].astype(np.int32)
            front_sem_rgb = np.zeros((720, 1024, 3), dtype=np.uint8)
            for i in range(len(CITYSCAPES_PALETTE)):
                front_sem_rgb[front_sem_data == i] = CITYSCAPES_PALETTE[i]
            
            top_rgb_image = top_rgb_queue.get()
            top_rgb_img = np.reshape(np.copy(top_rgb_image.raw_data), (720, 1024, 4))[:, :, :3]
            
            # 生成辅助图像
            density_heatmap = generate_density_heatmap(front_sem_data)
            class_count_result = semantic_class_count(front_sem_data, count_class_mapping)
            front_rgb_img_highlight = semantic_target_highlight(front_rgb_img.copy(), front_sem_data, highlight_class_mapping)
            fused_img = semantic_rgb_fusion(front_rgb_img, front_sem_data, alpha=fusion_alpha)
            
            # ==================== 第16次提交：量化评估计算与可视化 ====================
            # 模拟预测数据（实际项目中替换为模型输出）
            pred_sem_data = front_sem_data  # 此处用CARLA语义模拟预测（无噪声）
            eval_result = semantic_quantitative_evaluation(pred_sem_data, front_sem_data, EVAL_CLASSES)
            eval_vis = generate_evaluation_visualization(eval_result)
            # =================================================================
            
            # 多模式图像拼接（新增模式5：评估模式）
            if current_mode == 1:
                upper_part = cv2.hconcat([front_rgb_img_highlight, front_sem_rgb])
                lower_part = cv2.hconcat([top_rgb_img, density_heatmap])
            elif current_mode == 2:
                upper_part = cv2.hconcat([front_rgb_img_highlight, fused_img])
                lower_part = cv2.hconcat([front_sem_rgb, density_heatmap])
            elif current_mode == 3:
                count_vis = np.zeros((720, 1024, 3), dtype=np.uint8)
                cv2.putText(count_vis, "Semantic Count (Frame)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                count_items = [f"Pedestrian: {class_count_result['Pedestrian']}", f"Vehicle: {class_count_result['Vehicle']}", f"TrafficLight: {class_count_result['TrafficLight']}"]
                for idx, item in enumerate(count_items):
                    cv2.putText(count_vis, item, (50, 200 + idx * 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                upper_part = cv2.hconcat([front_rgb_img_highlight, density_heatmap])
                lower_part = cv2.hconcat([top_rgb_img, count_vis])
            elif current_mode == 4:
                upper_part = cv2.hconcat([front_sem_rgb, fused_img])
                lower_part = cv2.hconcat([density_heatmap, top_rgb_img])
            elif current_mode == 5:  # 模式5：评估模式
                upper_part = cv2.hconcat([front_rgb_img_highlight, fused_img])
                lower_part = cv2.hconcat([eval_vis, density_heatmap])  # 新增评估面板
            else:
                upper_part = cv2.hconcat([front_rgb_img_highlight, front_sem_rgb])
                lower_part = cv2.hconcat([top_rgb_img, density_heatmap])
            
            combined_img = cv2.vconcat([upper_part, lower_part])
            
            # 模式标题（新增模式5标题）
            mode_titles = {
                1: ["Front View (RGB + Highlight)", "Semantic Segmentation", "Top View (Bird's Eye)", "Density Heatmap"],
                2: ["Front View (RGB + Highlight)", "Semantic-RGB Fusion", "Semantic Segmentation", "Density Heatmap"],
                3: ["Front View (RGB + Highlight)", "Density Heatmap", "Top View (Bird's Eye)", "Semantic Count"],
                4: ["Semantic Segmentation", "Semantic-RGB Fusion", "Density Heatmap", "Top View (Bird's Eye)"],
                5: ["Front View (RGB + Highlight)", "Semantic-RGB Fusion", "Quantitative Evaluation", "Density Heatmap"]
            }
            titles = mode_titles.get(current_mode, mode_titles[1])
            cv2.putText(combined_img, titles[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(combined_img, titles[1], (1024 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(combined_img, titles[2], (10, 720 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(combined_img, titles[3], (1024 + 10, 720 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # 性能监控+模式提示
            perf_info = [
                f"FPS: {current_fps:.1f}",
                f"Sync Frame: {world.get_snapshot().frame}",
                f"Vehicles: {actual_npc_count} | Pedestrians: {actual_walker_count}",
                f"mIoU: {eval_result['mIoU']:.3f} | PA: {eval_result['PA']:.3f}"  # 新增量化指标显示
            ]
            mode_hints = generate_mode_hint(current_mode, fusion_alpha, eval_enabled)
            all_info = perf_info + mode_hints
            
            perf_x = 10
            perf_y = 60
            perf_line_height = 25
            perf_color = (0, 255, 255)
            for idx, info in enumerate(all_info):
                y_pos = perf_y + idx * perf_line_height
                cv2.rectangle(combined_img, (perf_x - 5, y_pos - 15), (perf_x + 600, y_pos + 5), (0, 0, 0), -1)
                cv2.putText(combined_img, info, (perf_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, perf_color, 2)
            
            # 语义计数面板（模式1/2/4显示）
            if current_mode in [1, 2, 4]:
                count_x = combined_img.shape[1] - 320
                count_y = 30
                cv2.rectangle(combined_img, (count_x - 10, count_y - 10), (combined_img.shape[1] - 10, count_y + 100), (0, 0, 0), -1)
                cv2.putText(combined_img, "Semantic Count (Frame)", (count_x, count_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                count_items = [f"Pedestrian: {class_count_result['Pedestrian']}", f"Vehicle: {class_count_result['Vehicle']}", f"TrafficLight: {class_count_result['TrafficLight']}"]
                for idx, item in enumerate(count_items):
                    y_pos = count_y + (idx + 1) * 28
                    cv2.putText(combined_img, item, (count_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            
            # 显示图像
            cv2.namedWindow('CARLA Multi-Mode + Quantitative Evaluation', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('CARLA Multi-Mode + Quantitative Evaluation', 1920, 1080)
            cv2.imshow('CARLA Multi-Mode + Quantitative Evaluation', combined_img)
            
            # 键盘事件处理（新增E键控制评估开关）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'): current_mode = 1; print(f"切换到模式1：基础模式")
            elif key == ord('2'): current_mode = 2; print(f"切换到模式2：融合模式")
            elif key == ord('3'): current_mode = 3; print(f"切换到模式3：精简模式")
            elif key == ord('4'): current_mode = 4; print(f"切换到模式4：全语义模式")
            elif key == ord('5'): current_mode = 5; print(f"切换到模式5：量化评估模式")
            elif key == ord('e') or key == ord('E'):
                eval_enabled = not eval_enabled
                print(f"量化评估{'开启' if eval_enabled else '关闭'}")
            elif key == ord('r') or key == ord('R'):
                current_mode = 1; fusion_alpha = 0.3; eval_enabled = True
                print(f"已重置：模式1+融合Alpha=0.3+量化评估开启")
            elif key == 2490368: fusion_alpha = min(fusion_alpha + 0.1, 1.0); print(f"融合Alpha调整为：{fusion_alpha:.1f}")
            elif key == 2621440: fusion_alpha = max(fusion_alpha - 0.1, 0.0); print(f"融合Alpha调整为：{fusion_alpha:.1f}")
        
        clock.tick(30)

except KeyboardInterrupt:
    print("\n用户中断，清理资源...")
finally:
    # 清理资源
    front_rgb_camera.stop(); front_rgb_camera.destroy()
    front_sem_camera.stop(); front_sem_camera.destroy()
    top_rgb_camera.stop(); top_rgb_camera.destroy()
    for controller in walker_controllers:
        if controller.is_alive:
            controller.stop(); controller.destroy()
    for walker in walkers:
        if walker.is_alive:
            walker.destroy()
    print(f"已销毁{len(walker_controllers)}个行人控制器 + {len(walkers)}个行人")
    settings.synchronous_mode = False
    tm.set_synchronous_mode(False)
    world.apply_settings(settings)
    for v in all_vehicles:
        if v.is_alive:
            v.destroy()
    cv2.destroyAllWindows()
    print(f"资源清理完成，销毁{len(all_vehicles)}辆车辆")