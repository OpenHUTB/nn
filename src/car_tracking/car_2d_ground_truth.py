import carla
import queue
import random
import cv2
import numpy as np

# ===================== 依赖导入（保留原始依赖）=====================
from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from utils.box_utils import draw_bounding_boxes
from utils.projection import *
from utils.world import *

# ===================== 配置常量（集中管理，便于修改）=====================
# 相机配置
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640
# 图表配置
CHART_WIDTH = 400
CHART_HEIGHT = CAMERA_HEIGHT  # 与相机高度一致
MAX_HISTORY_FRAMES = 50  # 最近50帧数据
# 跟踪窗口配置
TRACK_WINDOW_WIDTH = 300
TRACK_WINDOW_HEIGHT = 400
# 绘图配置
FONT_SCALE_SMALL = 0.4
FONT_SCALE_MEDIUM = 0.6
LINE_THICKNESS = 2
POINT_RADIUS = 2


# ===================== 工具函数（独立封装，提升复用性）=====================
def get_vehicle_color(vehicle_id):
    """为车辆生成固定唯一的RGB颜色（基于ID种子，保证跟踪时颜色不变）"""
    np.random.seed(vehicle_id)
    return tuple(np.random.randint(0, 255, 3).tolist())


def custom_draw_bounding_boxes(image, boxes, labels, class_names, ids=None, track_data=None):
    """保留原始边界框绘制逻辑，叠加跟踪数据（距离+颜色外框）"""
    # 调用原始画框函数，保证核心功能不变
    img = draw_bounding_boxes(image, boxes, labels, class_names, ids)

    # 叠加跟踪数据标注（不破坏原始框）
    if ids is not None and track_data is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            vid = ids[i]
            if vid in track_data:
                x1, y1, x2, y2 = map(int, box)
                color = track_data[vid]['color']
                dist = track_data[vid]['distance']
                # 绘制距离文本（在原始标注下方）
                cv2.putText(
                    img, f"Dist: {dist:.1f}m", (x1, y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, color, 1
                )
                # 绘制跟踪颜色外框（不覆盖原始绿色框）
                cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color, 1)
    return img


def init_chart_background(width, height):
    """初始化图表背景（绘制固定元素：标题、网格、图例，避免每帧重复绘制）"""
    chart = np.zeros((height, width, 3), dtype=np.uint8)
    # 绘制标题
    cv2.putText(
        chart, "Real-Time Statistics (Last 50 Frames)", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_MEDIUM, (255, 255, 255), LINE_THICKNESS
    )
    # 绘制网格（浅灰色，提升视觉效果）
    grid_color = (50, 50, 50)
    # 水平网格线
    for y in range(50, height - 30, 50):
        cv2.line(chart, (50, y), (width - 50, y), grid_color, 1)
    # 垂直网格线
    for x in range(50, width - 50, 50):
        cv2.line(chart, (x, 30), (x, height - 30), grid_color, 1)
    # 绘制图例（固定位置，提升可读性）
    cv2.putText(
        chart, "Current Vehicles (green)", (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, (0, 255, 0), 1
    )
    cv2.putText(
        chart, "Max Distance (red)", (200, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, (0, 0, 255), 1
    )
    return chart


def draw_dynamic_chart(history_frames, history_vehicles, history_max_dist):
    """
    绘制实时动态折线图（仅绘制变化的折线和数据点，复用固定背景）
    参数：
        history_frames: 最近50帧的帧数列表
        history_vehicles: 最近50帧的车辆数量列表
        history_max_dist: 最近50帧的最大距离列表
    返回：
        绘制完成的图表图像
    """
    # 初始化图表背景（固定元素）
    chart = init_chart_background(CHART_WIDTH, CHART_HEIGHT)

    # 数据为空时直接返回背景
    if len(history_frames) == 0:
        return chart

    # ========== 数据归一化（优化计算逻辑，避免除零错误）==========
    # 车辆数量归一化（映射到图表y轴范围：30 ~ CHART_HEIGHT-30）
    max_veh = max(history_vehicles) if history_vehicles else 1
    max_veh = max_veh if max_veh != 0 else 1  # 处理除零
    norm_veh = [(v / max_veh) * (CHART_HEIGHT - 60) for v in history_vehicles]
    y_veh = np.array([CHART_HEIGHT - 30 - v for v in norm_veh], dtype=int)

    # 最大距离归一化
    max_d = max(history_max_dist) if history_max_dist else 1
    max_d = max_d if max_d != 0 else 1  # 处理除零
    norm_dist = [(d / max_d) * (CHART_HEIGHT - 60) for d in history_max_dist]
    y_dist = np.array([CHART_HEIGHT - 30 - d for d in norm_dist], dtype=int)

    # x轴归一化（动态滚动，仅显示最近50帧的x坐标）
    x_coords = np.array([
        50 + (i * (CHART_WIDTH - 100) / (len(history_frames) - 1 if len(history_frames) > 1 else 1))
        for i in range(len(history_frames))
    ], dtype=int)

    # ========== 绘制折线和数据点（优化绘制逻辑，提升流畅度）==========
    # 绘制车辆数量折线（绿色）
    if len(x_coords) > 1:
        cv2.polylines(chart, [np.column_stack((x_coords, y_veh))], isClosed=False, color=(0, 255, 0),
                      thickness=LINE_THICKNESS)
    # 绘制车辆数量数据点
    for x, y in zip(x_coords, y_veh):
        cv2.circle(chart, (x, y), POINT_RADIUS, (0, 255, 0), -1)

    # 绘制最大距离折线（红色）
    if len(x_coords) > 1:
        cv2.polylines(chart, [np.column_stack((x_coords, y_dist))], isClosed=False, color=(0, 0, 255),
                      thickness=LINE_THICKNESS)
    # 绘制最大距离数据点
    for x, y in zip(x_coords, y_dist):
        cv2.circle(chart, (x, y), POINT_RADIUS, (0, 0, 255), -1)

    # ========== 绘制当前数值标注（实时更新）==========
    if len(history_vehicles) > 0 and len(history_max_dist) > 0:
        current_veh = history_vehicles[-1]
        current_dist = history_max_dist[-1]
        cv2.putText(
            chart, f"Now: {current_veh} cars | {current_dist:.1f}m",
            (CHART_WIDTH - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL,
            (255, 255, 255), 1
        )

    return chart


def camera_callback(image, rgb_image_queue):
    """相机回调函数（保留原始逻辑，简化代码）"""
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)))


def convert_image_format(image):
    """将4通道BGRA图像转换为3通道RGB图像（提取前3通道，简化逻辑）"""
    return image[..., :3] if image.shape[-1] == 4 else image.copy()


# ===================== 主程序逻辑（优化结构，提升可读性）=====================
def main():
    # 初始化Carla客户端和世界
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # 设置同步模式（保留原始逻辑）
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 获取观众和生成点（保留原始逻辑）
    spectator = world.get_spectator()
    spawn_points = world.get_map().get_spawn_points()

    # 生成主车辆（保留原始逻辑，增加异常处理）
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if not vehicle:
        print("警告：主车辆生成失败，程序退出！")
        return

    # 生成相机（使用配置常量，简化代码）
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    camera_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    camera_init_trans = carla.Transform(carla.Location(x=1, z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    # 初始化图像队列（保留原始逻辑）
    image_queue = queue.Queue()
    camera.listen(lambda image: camera_callback(image, image_queue))

    # 清理现有NPC（保留原始逻辑）
    clear_npc(world)
    clear_static_vehicle(world)

    # 2D框计算相关参数（保留原始逻辑）
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
             [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
    fov = camera_bp.get_attribute("fov").as_float()
    K = build_projection_matrix(CAMERA_WIDTH, CAMERA_HEIGHT, fov)
    K_b = build_projection_matrix(CAMERA_WIDTH, CAMERA_HEIGHT, fov, is_behind_camera=True)

    # 生成NPC车辆（保留原始逻辑，使用配置常量）
    for _ in range(50):
        vehicle_bp_list = bp_lib.filter('vehicle')
        car_bp = [bp for bp in vehicle_bp_list if int(bp.get_attribute('number_of_wheels')) == 4]
        if car_bp:
            npc = world.try_spawn_actor(random.choice(car_bp), random.choice(spawn_points))
            if npc:
                npc.set_autopilot(True)
    vehicle.set_autopilot(True)

    # 初始化跟踪和数据缓存变量（改为局部变量，减少全局变量）
    tracked_vehicles = {}  # key:车辆ID, value:{'color':颜色, 'distance':距离, 'frame':帧数}
    frame_counter = 0
    history_frames = []  # 最近50帧的帧数
    history_vehicles = []  # 最近50帧的车辆数量
    history_max_dist = []  # 最近50帧的最大距离

    # 主循环（优化逻辑，提升可读性）
    try:
        while True:
            world.tick()
            frame_counter += 1

            # 移动观众视角到车辆顶部（保留原始逻辑）
            transform = carla.Transform(
                vehicle.get_transform().transform(carla.Location(x=-4, z=50)),
                carla.Rotation(yaw=-180, pitch=-90)
            )
            spectator.set_transform(transform)

            # 获取相机图像（保留原始逻辑）
            image = image_queue.get()

            # 更新相机矩阵（保留原始逻辑）
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # 初始化当前帧变量
            boxes = []
            ids = []
            track_data = {}
            current_vehicles = 0
            max_distance = 0.0

            # 检测车辆并计算2D边界框（保留原始核心逻辑）
            for npc in world.get_actors().filter('*vehicle*'):
                if npc.id != vehicle.id:
                    # 计算车辆距离和最大距离
                    dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                    max_distance = max(max_distance, dist)

                    # 过滤50米内、车辆前方的目标
                    if dist < 50:
                        forward_vec = vehicle.get_transform().get_forward_vector()
                        ray = npc.get_transform().location - vehicle.get_transform().location
                        if forward_vec.dot(ray) > 0:
                            # 计算3D顶点的2D投影
                            verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
                            points_2d = []
                            for vert in verts:
                                ray0 = vert - camera.get_transform().location
                                cam_forward_vec = camera.get_transform().get_forward_vector()
                                if cam_forward_vec.dot(ray0) > 0:
                                    p = get_image_point(vert, K, world_2_camera)
                                else:
                                    p = get_image_point(vert, K_b, world_2_camera)
                                points_2d.append(p)

                            # 计算2D边界框
                            x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(
                                points_2d, edges, CAMERA_HEIGHT, CAMERA_WIDTH
                            )

                            # 过滤小框和超出画布的框
                            if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                                if point_in_canvas((x_min, y_min), CAMERA_HEIGHT, CAMERA_WIDTH) and \
                                        point_in_canvas((x_max, y_max), CAMERA_HEIGHT, CAMERA_WIDTH):
                                    ids.append(npc.id)
                                    boxes.append(np.array([x_min, y_min, x_max, y_max]))
                                    # 更新跟踪数据
                                    if npc.id not in tracked_vehicles:
                                        tracked_vehicles[npc.id] = {'color': get_vehicle_color(npc.id)}
                                    tracked_vehicles[npc.id]['distance'] = dist
                                    tracked_vehicles[npc.id]['frame'] = frame_counter
                                    track_data[npc.id] = tracked_vehicles[npc.id]
                                    current_vehicles += 1

            # 更新历史数据缓存（仅保留最近50帧，优化逻辑）
            history_frames.append(frame_counter)
            history_vehicles.append(current_vehicles)
            history_max_dist.append(max_distance)
            # 截断数据，保持固定长度
            if len(history_frames) > MAX_HISTORY_FRAMES:
                history_frames.pop(0)
                history_vehicles.pop(0)
                history_max_dist.pop(0)

            # 绘制边界框（保留原始逻辑，使用自定义函数）
            boxes = np.array(boxes)
            labels = np.array([2] * len(boxes))
            probs = np.array([1.0] * len(boxes))
            output = custom_draw_bounding_boxes(
                image, boxes, labels, COCO_CLASS_NAMES, ids, track_data
            ) if len(boxes) > 0 else image

            # 转换图像格式并拼接图表（优化逻辑）
            output_rgb = convert_image_format(output)
            chart_image = draw_dynamic_chart(history_frames, history_vehicles, history_max_dist)
            combined_image = np.hstack((output_rgb, chart_image))

            # 绘制跟踪监测窗口（保留原始功能，优化代码）
            track_window = np.zeros((TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                track_window, "Vehicle Tracking Monitor", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_MEDIUM, (255, 255, 255), LINE_THICKNESS
            )
            y_offset = 60
            # 显示前10辆跟踪的车辆
            for vid, data in list(tracked_vehicles.items())[:10]:
                if y_offset > TRACK_WINDOW_HEIGHT - 20:
                    break
                color = data['color']
                dist = data.get('distance', 0.0)
                cv2.putText(
                    track_window, f"ID: {vid} | Dist: {dist:.1f}m", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, color, 1
                )
                y_offset += 30

            # 显示窗口（保留原始逻辑）
            cv2.imshow('2D Ground Truth', combined_image)
            cv2.imshow('Vehicle Tracking Monitor', track_window)

            # 按q退出（保留原始逻辑）
            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        print("程序被用户中断！")
    finally:
        # 清理资源（优化异常处理，确保资源释放）
        try:
            camera.destroy()
            clear(world, camera)
        except Exception as e:
            print(f"清理资源时警告：{e}")
        cv2.destroyAllWindows()
        # 恢复Carla世界设置（新增，避免同步模式残留）
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == '__main__':
    main()