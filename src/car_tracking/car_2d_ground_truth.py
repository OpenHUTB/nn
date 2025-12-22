import carla
import queue
import random
import cv2
import numpy as np

# ===================== 原始代码依赖的工具函数（完全对齐原始逻辑，仅新增跟踪/数据相关）=====================
# COCO类别名称（原始代码的版本）
from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from utils.box_utils import draw_bounding_boxes
from utils.projection import *
from utils.world import *

# 新增：跟踪相关全局变量（不影响原始逻辑）
tracked_vehicles = {}  # key:车辆ID, value:{'distance':距离, 'frame':帧数, 'color':颜色}
frame_counter = 0  # 全局帧数计数器

# 新增：数据缓存（保存最近50帧的历史数据，用于动态图表）
MAX_HISTORY = 50
history_frames = []  # 帧数缓存
history_vehicles = []  # 实时车辆数量缓存
history_max_dist = []  # 最大距离缓存


def get_vehicle_color(vehicle_id):
    """为车辆生成固定唯一颜色（用于跟踪）"""
    np.random.seed(vehicle_id)
    return tuple(np.random.randint(0, 255, 3).tolist())


# 重写draw_bounding_boxes以兼容跟踪（保留原始函数参数，新增跟踪标注）
def custom_draw_bounding_boxes(image, boxes, labels, class_names, ids=None, track_data=None):
    """保留原始画框逻辑，新增跟踪颜色和距离标注"""
    # 调用原始的draw_bounding_boxes函数（保证原始功能）
    img = draw_bounding_boxes(image, boxes, labels, class_names, ids)
    # 新增：叠加跟踪数据标注（不破坏原始画框）
    if ids is not None and track_data is not None:
        for i, box in enumerate(boxes):
            vid = ids[i]
            if vid in track_data:
                x1, y1, x2, y2 = map(int, box)
                color = track_data[vid]['color']
                dist = track_data[vid]['distance']
                # 绘制距离文本（在原始标注下方）
                cv2.putText(img, f"Dist: {dist:.1f}m", (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                # 用跟踪颜色绘制框的外框（不覆盖原始绿色框）
                cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color, 1)
    return img


# 新增：绘制动态统计图表（仅修复除零错误，其他不变）
def draw_dynamic_chart(width, height, frames, vehicles, max_dist):
    """
    绘制实时折线图：x轴=帧数，y轴=车辆数量/距离
    参数：
        width, height: 图表尺寸
        frames: 帧数列表
        vehicles: 车辆数量列表
        max_dist: 最大距离列表
    返回：
        图表图像
    """
    # 初始化黑色背景图表
    chart = np.zeros((height, width, 3), dtype=np.uint8)
    # 绘制标题
    cv2.putText(chart, "Real-Time Statistics (Last 50 Frames)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    # 数据为空时返回空图表
    if len(frames) == 0:
        return chart

    # ========== 核心修复：处理除零错误 ==========
    max_veh = max(vehicles) if vehicles else 1
    if max_veh == 0:  # 当车辆数量最大值为0时，强制设为1避免除法错误
        max_veh = 1
    max_d = max(max_dist) if max_dist else 1
    if max_d == 0:  # 当距离最大值为0时，强制设为1避免除法错误
        max_d = 1
    # ==========================================

    # 归一化数据到图表坐标（适配y轴范围）
    # x轴：帧数映射到图表宽度（从50px开始，留空坐标轴）
    x_coords = np.array(
        [50 + (x - frames[0]) * (width - 100) / (frames[-1] - frames[0] if frames[-1] != frames[0] else 1) for x in
         frames], dtype=int)
    # y轴：车辆数量（绿色）映射到图表高度（从30px到height-30px）
    y_veh = np.array([height - 30 - (v / max_veh) * (height - 60) for v in vehicles], dtype=int)
    # y轴：最大距离（红色）映射到图表高度
    y_dist = np.array([height - 30 - (d / max_d) * (height - 60) for d in max_dist], dtype=int)

    # 绘制网格
    for y in range(0, height, 50):
        cv2.line(chart, (50, y), (width - 50, y), (50, 50, 50), 1)
    for x in range(50, width, 50):
        cv2.line(chart, (x, 30), (x, height - 30), (50, 50, 50), 1)

    # 绘制折线：车辆数量（绿色）
    if len(x_coords) > 1:
        for i in range(1, len(x_coords)):
            cv2.line(chart, (x_coords[i - 1], y_veh[i - 1]), (x_coords[i], y_veh[i]), (0, 255, 0), 2)
        # 绘制数据点
        for x, y in zip(x_coords, y_veh):
            cv2.circle(chart, (x, y), 2, (0, 255, 0), -1)

    # 绘制折线：最大距离（红色）
    if len(x_coords) > 1:
        for i in range(1, len(x_coords)):
            cv2.line(chart, (x_coords[i - 1], y_dist[i - 1]), (x_coords[i], y_dist[i]), (0, 0, 255), 2)
        # 绘制数据点
        for x, y in zip(x_coords, y_dist):
            cv2.circle(chart, (x, y), 2, (0, 0, 255), -1)

    # 绘制图例
    cv2.putText(chart, "Current Vehicles (green)", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(chart, "Max Distance (red)", (200, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 绘制当前数值标注
    if len(vehicles) > 0 and len(max_dist) > 0:
        cv2.putText(chart, f"Now: {vehicles[-1]} cars | {max_dist[-1]:.1f}m", (width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return chart


# ===================== 相机回调函数（完全保留原始代码）=====================
def camera_callback(image, rgb_image_queue):
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data),
                                   (image.height, image.width, 4)))


# ===================== 主程序逻辑封装到main函数中 ======================
def main():
    # 声明使用全局变量（若需要修改全局变量则必须声明）
    global frame_counter, history_frames, history_vehicles, history_max_dist, tracked_vehicles
    # 重置全局变量（避免多次运行时数据残留）
    frame_counter = 0
    tracked_vehicles = {}
    history_frames = []
    history_vehicles = []
    history_max_dist = []

    # Part 1（完全保留原始代码，窗口尺寸为640*640）
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Get the world spectator
    spectator = world.get_spectator()

    # Get the map spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Spawn the ego vehicle
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

    # Spawn the camera（原始尺寸 640*640）
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '640')

    camera_init_trans = carla.Transform(carla.Location(x=1, z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    # Create a queue to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(lambda image: camera_callback(image, image_queue))

    # Clear existing NPCs
    clear_npc(world)
    clear_static_vehicle(world)

    # Part 2（完全保留原始代码）
    # Remember the edge pairs
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
             [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

    # Get the world to camera matrix
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Get the attributes from the camera
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    # Spawn NPCs（完全保留原始代码）
    for i in range(50):
        vehicle_bp = bp_lib.filter('vehicle')

        # Exclude bicycle
        car_bp = [bp for bp in vehicle_bp if int(
            bp.get_attribute('number_of_wheels')) == 4]
        npc = world.try_spawn_actor(random.choice(
            car_bp), random.choice(spawn_points))

        if npc:
            npc.set_autopilot(True)

    if vehicle:
        vehicle.set_autopilot(True)

    # 新增：数据统计变量（保留原逻辑）
    total_vehicles = 0
    max_distance = 0.0
    current_vehicles = 0

    # Main Loop（保留原始核心逻辑，仅替换数据面板为动态图表）
    while True:
        try:
            world.tick()
            frame_counter += 1

            # Move the spectator to the top of the vehicle（原始代码）
            if vehicle:
                transform = carla.Transform(vehicle.get_transform().transform(
                    carla.Location(x=-4, z=50)), carla.Rotation(yaw=-180, pitch=-90))
                spectator.set_transform(transform)

            # Retrieve and reshape the image（原始代码）
            image = image_queue.get()

            # Get the camera matrix（原始代码）
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            boxes = []
            ids = []
            # 新增：跟踪数据存储（当前帧）
            track_data = {}
            current_vehicles = 0
            max_distance = 0.0  # 重置当前帧最大距离

            # 原始代码的核心循环（完全保留）
            for npc in world.get_actors().filter('*vehicle*'):
                if vehicle and npc.id != vehicle.id:
                    bb = npc.bounding_box
                    dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                    max_distance = max(max_distance, dist)  # 更新当前帧最大距离

                    # Filter for the vehicles within 50m（原始代码）
                    if dist < 50:
                        # Calculate the dot product（原始代码）
                        forward_vec = vehicle.get_transform().get_forward_vector()
                        ray = npc.get_transform().location - vehicle.get_transform().location

                        if forward_vec.dot(ray) > 0:
                            verts = [v for v in bb.get_world_vertices(
                                npc.get_transform())]

                            points_2d = []

                            for vert in verts:
                                ray0 = vert - camera.get_transform().location
                                cam_forward_vec = camera.get_transform().get_forward_vector()

                                if (cam_forward_vec.dot(ray0) > 0):
                                    p = get_image_point(vert, K, world_2_camera)
                                else:
                                    p = get_image_point(vert, K_b, world_2_camera)

                                points_2d.append(p)

                            x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(
                                points_2d, edges, image_h, image_w)

                            # Exclude very small bounding boxes（原始代码）
                            if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                                if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max),
                                                                                                         image_h,
                                                                                                         image_w):
                                    ids.append(npc.id)
                                    boxes.append(
                                        np.array([x_min, y_min, x_max, y_max]))
                                    # 新增：填充跟踪数据（不影响原始逻辑）
                                    if npc.id not in tracked_vehicles:
                                        tracked_vehicles[npc.id] = {'color': get_vehicle_color(npc.id)}
                                    tracked_vehicles[npc.id]['distance'] = dist
                                    tracked_vehicles[npc.id]['frame'] = frame_counter
                                    track_data[npc.id] = tracked_vehicles[npc.id]
                                    current_vehicles += 1
                                    # 统计数据
                                    total_vehicles = max(total_vehicles, len(tracked_vehicles))

            # 新增：更新历史数据缓存（只保留最近50帧）
            history_frames.append(frame_counter)
            history_vehicles.append(current_vehicles)
            history_max_dist.append(max_distance)
            # 截断缓存，保持固定长度
            if len(history_frames) > MAX_HISTORY:
                history_frames.pop(0)
                history_vehicles.pop(0)
                history_max_dist.pop(0)

            # 原始代码的框处理（完全保留）
            boxes = np.array(boxes)
            labels = np.array([2] * len(boxes))
            probs = np.array([1.0] * len(boxes))

            # 原始代码的画框逻辑（替换为自定义函数，保留原始功能+新增跟踪）
            if len(boxes) > 0:
                output = custom_draw_bounding_boxes(
                    image, boxes, labels, COCO_CLASS_NAMES, ids, track_data)
            else:
                output = image

            # 替换：动态统计图表（原始尺寸 400宽度）
            # 转换图像为RGB（处理4通道）
            if output.shape[-1] == 4:
                output_rgb = output[..., :3]
            else:
                output_rgb = output.copy()
            # 生成动态图表（尺寸：宽度400，高度与图像一致）
            chart_image = draw_dynamic_chart(400, image_h, history_frames, history_vehicles, history_max_dist)
            # 拼接图像和图表
            combined_image = np.hstack((output_rgb, chart_image))

            # 保留：跟踪监测窗口（原始尺寸 400*300）
            track_window = np.zeros((400, 300, 3), dtype=np.uint8)
            cv2.putText(track_window, "Vehicle Tracking Monitor", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            y_offset = 60
            # 显示前10辆跟踪的车辆
            for vid, data in list(tracked_vehicles.items())[:10]:
                color = data['color']
                dist = data['distance']
                cv2.putText(track_window, f"ID: {vid} | Dist: {dist:.1f}m", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color, 1)
                y_offset += 30
                if y_offset > 380:
                    break

            # 原始代码的显示逻辑（保留）
            cv2.imshow('2D Ground Truth', combined_image)
            cv2.imshow('Vehicle Tracking Monitor', track_window)

            if cv2.waitKey(1) == ord('q'):
                break

        except KeyboardInterrupt as e:
            break

    # 原始代码的清理逻辑（保留，新增异常处理）
    try:
        clear(world, camera)
    except Exception as e:
        print(f"Cleanup warning: {e}")
    cv2.destroyAllWindows()


# ===================== Python规范的主程序入口 ======================
if __name__ == '__main__':
    main()