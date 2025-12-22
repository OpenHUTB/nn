import carla
import queue
import random
import cv2
import numpy as np

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from utils.box_utils import draw_bounding_boxes
from utils.projection import *
from utils.world import *

# -------------------------- 新增：车辆类型、距离、速度统计工具函数 --------------------------
def get_vehicle_brand_type(actor):
    """提取车辆的品牌和车型（从type_id中解析，如vehicle.lincoln.mkz → Lincoln MKZ）"""
    try:
        parts = actor.type_id.split('.')
        if len(parts) >= 3:
            brand = parts[1].capitalize()
            model = parts[2].upper()
            return f"{brand} {model}"
        return "Unknown Vehicle"
    except:
        return "Unknown Vehicle"

def get_vehicle_speed(vehicle):
    """获取车辆的速度（km/h）"""
    try:
        velocity = vehicle.get_velocity()
        # 转换为km/h：速度向量的模 × 3.6（m/s → km/h）
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
        return round(speed, 1)
    except:
        return 0.0

def calculate_vehicle_stats(vehicle_data):
    """统计车辆类型、距离区间、速度区间"""
    # 1. 车辆类型统计
    type_count = {}
    # 2. 距离区间统计（0-10m, 10-20m, 20-30m, 30-40m, 40-50m）
    distance_ranges = {"0-10m": 0, "10-20m": 0, "20-30m": 0, "30-40m": 0, "40-50m": 0}
    # 3. 速度区间统计（0-10km/h, 10-20km/h, 20-30km/h, >30km/h）
    speed_ranges = {"0-10km/h": 0, "10-20km/h": 0, "20-30km/h": 0, ">30km/h": 0}

    for _, v_type, dist, speed in vehicle_data:
        # 类型统计
        type_count[v_type] = type_count.get(v_type, 0) + 1
        # 距离区间统计
        if dist < 10:
            distance_ranges["0-10m"] += 1
        elif dist < 20:
            distance_ranges["10-20m"] += 1
        elif dist < 30:
            distance_ranges["20-30m"] += 1
        elif dist < 40:
            distance_ranges["30-40m"] += 1
        else:
            distance_ranges["40-50m"] += 1
        # 速度区间统计
        if speed < 10:
            speed_ranges["0-10km/h"] += 1
        elif speed < 20:
            speed_ranges["10-20km/h"] += 1
        elif speed < 30:
            speed_ranges["20-30km/h"] += 1
        else:
            speed_ranges[">30km/h"] += 1

    return type_count, distance_ranges, speed_ranges

def calculate_perception_stats(vehicle_distances, valid_boxes_count):
    """基础感知统计（总车辆数、平均距离等）"""
    stats = {
        "total_vehicles": len(vehicle_distances),
        "valid_boxes": valid_boxes_count,
        "avg_distance": np.mean(vehicle_distances) if vehicle_distances else 0.0,
        "max_distance": np.max(vehicle_distances) if vehicle_distances else 0.0,
        "avg_speed": np.mean([d[3] for d in vehicle_data]) if vehicle_data else 0.0  # 平均速度
    }
    return stats

# -------------------------- 小框图核心函数（替换为新的统计维度） --------------------------
def create_small_view_layout(main_img, base_stats, vehicle_data, type_count, distance_ranges, speed_ranges, CAMERA_WIDTH=640, CAMERA_HEIGHT=640):
    """主视图+右侧统计面板（车辆类型、距离、速度统计）"""
    canvas_width = CAMERA_WIDTH + 350
    canvas_height = CAMERA_HEIGHT
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 240  # 浅灰色背景

    # 1. 主视图（左侧640x640）
    canvas[:CAMERA_HEIGHT, :CAMERA_WIDTH, :] = main_img

    # 2. 右侧统计面板
    panel_x_start = CAMERA_WIDTH + 20
    # 标题
    cv2.putText(canvas, "Vehicle Perception Stats", (panel_x_start, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # 基础统计数据
    base_stats_text = [
        f"Total Vehicles (50m): {base_stats['total_vehicles']}",
        f"Valid 2D Boxes: {base_stats['valid_boxes']}",
        f"Avg Distance: {base_stats['avg_distance']:.1f}m",
        f"Max Distance: {base_stats['max_distance']:.1f}m",
        f"Avg Speed: {base_stats['avg_speed']:.1f}km/h"
    ]
    y_start = 70
    line_height = 30
    for text in base_stats_text:
        cv2.putText(canvas, text, (panel_x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_start += line_height

    # -------------------------- 统计1：车辆类型（前5种，避免过长） --------------------------
    cv2.putText(canvas, "Vehicle Type (Top5)", (panel_x_start, y_start + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_start += 40
    # 按数量排序，取前5种
    sorted_types = sorted(type_count.items(), key=lambda x: x[1], reverse=True)[:5]
    for v_type, count in sorted_types:
        if y_start > 220:  # 为后续统计留出空间
            break
        # 缩短过长的车型名称（避免超出面板）
        display_type = v_type if len(v_type) <= 15 else v_type[:12] + "..."
        text = f"{display_type}: {count} vehicles"
        cv2.putText(canvas, text, (panel_x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        y_start += 25

    # -------------------------- 统计2：距离区间（彩色进度条显示） --------------------------
    cv2.putText(canvas, "Distance Distribution", (panel_x_start, y_start + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_start += 40
    # 距离区间颜色映射（不同区间不同颜色）
    dist_color_map = {
        "0-10m": (0, 0, 255), "10-20m": (0, 165, 255), "20-30m": (0, 255, 255),
        "30-40m": (0, 255, 0), "40-50m": (255, 0, 0)
    }
    for dist_range, count in distance_ranges.items():
        if y_start > 350:
            break
        # 绘制彩色小方块
        color = dist_color_map.get(dist_range, (128, 128, 128))
        sq_x1 = panel_x_start
        sq_y1 = y_start - 8
        sq_x2 = panel_x_start + 15
        sq_y2 = y_start + 8
        cv2.rectangle(canvas, (sq_x1, sq_y1), (sq_x2, sq_y2), color, -1)
        cv2.rectangle(canvas, (sq_x1, sq_y1), (sq_x2, sq_y2), (0, 0, 0), 1)
        # 绘制距离区间和数量
        text = f"{dist_range}: {count} vehicles"
        cv2.putText(canvas, text, (panel_x_start + 20, y_start + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        y_start += 25

    # -------------------------- 统计3：速度区间（彩色进度条显示） --------------------------
    cv2.putText(canvas, "Speed Distribution", (panel_x_start, y_start + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_start += 40
    # 速度区间颜色映射
    speed_color_map = {
        "0-10km/h": (128, 128, 128), "10-20km/h": (0, 255, 0),
        "20-30km/h": (0, 255, 255), ">30km/h": (0, 0, 255)
    }
    for speed_range, count in speed_ranges.items():
        if y_start > canvas_height - 20:
            break
        # 绘制彩色小方块
        color = speed_color_map.get(speed_range, (128, 128, 128))
        sq_x1 = panel_x_start
        sq_y1 = y_start - 8
        sq_x2 = panel_x_start + 15
        sq_y2 = y_start + 8
        cv2.rectangle(canvas, (sq_x1, sq_y1), (sq_x2, sq_y2), color, -1)
        cv2.rectangle(canvas, (sq_x1, sq_y1), (sq_x2, sq_y2), (0, 0, 0), 1)
        # 绘制速度区间和数量
        text = f"{speed_range}: {count} vehicles"
        cv2.putText(canvas, text, (panel_x_start + 20, y_start + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        y_start += 25

    return canvas

# -------------------------- 原始代码：相机回调函数 --------------------------
def camera_callback(image, rgb_image_queue):
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data),
                        (image.height, image.width, 4)))

# -------------------------- 主程序（替换为新的统计维度） --------------------------
def main():
    # 1. 连接CARLA并设置超时
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)
    world = client.get_world()

    # 2. 配置仿真环境（同步模式+Traffic Manager）
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 初始化Traffic Manager（车辆移动核心）
    tm = client.get_trafficmanager(8000)
    tm.set_global_distance_to_leading_vehicle(2.0)
    tm.set_random_device_seed(42)
    tm.global_percentage_speed_difference(20)  # 车辆速度80%限速

    # 3. 获取出生点并生成主角车辆
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("❌ 没有可用的出生点！")
        return

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2017')  # 更换为更常见的车型
    ego_vehicle = None
    for sp in random.sample(spawn_points, min(10, len(spawn_points))):
        ego_vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if ego_vehicle:
            break
    if not ego_vehicle:
        print("❌ 主角车辆生成失败！")
        return
    ego_vehicle.set_autopilot(True, tm.get_port())

    # 4. 生成相机（保留原始参数）
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '640')
    camera_init_trans = carla.Transform(carla.Location(x=1, z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
    image_queue = queue.Queue()
    camera.listen(lambda image: camera_callback(image, image_queue))

    # 5. 清理旧NPC并生成新NPC（生成不同品牌的车辆，丰富类型统计）
    clear_npc(world)
    clear_static_vehicle(world)

    # 选择不同品牌的车辆蓝图（丰富类型统计）
    vehicle_blueprints = [
        bp for bp in bp_lib.filter('vehicle')
        if int(bp.get_attribute('number_of_wheels')) == 4 and
        not bp.id.endswith('cycle') and not bp.id.endswith('motorcycle')
    ]
    # 生成50辆NPC车辆（不同品牌）
    for i in range(50):
        if not vehicle_blueprints:
            break
        npc_bp = random.choice(vehicle_blueprints)
        npc_vehicle = None
        # 遍历多个出生点，确保NPC生成成功
        for sp in random.sample(spawn_points, min(5, len(spawn_points))):
            npc_vehicle = world.try_spawn_actor(npc_bp, sp)
            if npc_vehicle:
                break
        if npc_vehicle:
            npc_vehicle.set_autopilot(True, tm.get_port())

    # 6. 初始化 spectator 视角
    spectator = world.get_spectator()
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
             [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

    # 7. 主循环（车辆感知核心）
    try:
        while True:
            world.tick()

            # 更新 spectator 视角（跟随主角车辆）
            transform = carla.Transform(ego_vehicle.get_transform().transform(
                carla.Location(x=-4, z=50)), carla.Rotation(yaw=-180, pitch=-90))
            spectator.set_transform(transform)

            # 获取相机图像（跳过空队列）
            if image_queue.empty():
                continue
            image = image_queue.get()

            # 更新投影矩阵（每帧更新，确保投影准确）
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
            image_w = camera_bp.get_attribute("image_size_x").as_int()
            image_h = camera_bp.get_attribute("image_size_y").as_int()
            fov = camera_bp.get_attribute("fov").as_float()
            K = build_projection_matrix(image_w, image_h, fov)
            K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

            boxes = []
            ids = []
            vehicle_data = []  # 格式：(id, type, distance, speed)
            vehicle_distances = []

            # 遍历所有车辆（筛选+投影）
            for npc in world.get_actors().filter('*vehicle*'):
                if npc.id == ego_vehicle.id:
                    continue

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(ego_vehicle.get_transform().location)

                # 过滤1：50米内
                if dist > 50:
                    continue

                # 过滤2：正前方（向量单位化，点积阈值0.1）
                forward_vec = ego_vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - ego_vehicle.get_transform().location
                ray = ray.make_unit_vector()
                dot_product = forward_vec.dot(ray)
                if dot_product <= 0.1:
                    continue

                # 3D转2D投影（过滤无效点）
                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                points_2d = []
                for vert in verts:
                    ray0 = vert - camera.get_transform().location
                    cam_forward_vec = camera.get_transform().get_forward_vector()
                    if cam_forward_vec.dot(ray0) > 0:
                        p = get_image_point(vert, K, world_2_camera)
                    else:
                        p = get_image_point(vert, K_b, world_2_camera)
                    if not (np.isnan(p[0]) or np.isnan(p[1])):
                        points_2d.append(p)

                # 至少4个有效点才计算边界框
                if len(points_2d) < 4:
                    continue

                x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(points_2d, edges, image_h, image_w)
                box_width = x_max - x_min
                box_height = y_max - y_min
                box_area = box_width * box_height

                # 过滤无效小框（降低阈值，避免漏检）
                if box_area > 50 and box_width > 10:
                    if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):
                        ids.append(npc.id)
                        boxes.append(np.array([x_min, y_min, x_max, y_max]))
                        # 收集车辆类型、距离、速度（核心：新的统计数据）
                        v_type = get_vehicle_brand_type(npc)
                        v_speed = get_vehicle_speed(npc)
                        vehicle_data.append((npc.id, v_type, dist, v_speed))
                        vehicle_distances.append(dist)

            # 绘制边界框（保留原始调用）
            boxes = np.array(boxes)
            labels = np.array([2] * len(boxes))
            probs = np.array([1.0] * len(boxes))
            output_image = image
            if len(boxes) > 0:
                output_image = draw_bounding_boxes(image, boxes, labels, COCO_CLASS_NAMES, ids)

            # 计算统计数据（新的维度）
            type_count, distance_ranges, speed_ranges = calculate_vehicle_stats(vehicle_data)
            # 基础统计（补充平均速度）
            base_stats = {
                "total_vehicles": len(vehicle_distances),
                "valid_boxes": len(boxes),
                "avg_distance": np.mean(vehicle_distances) if vehicle_distances else 0.0,
                "max_distance": np.max(vehicle_distances) if vehicle_distances else 0.0,
                "avg_speed": np.mean([d[3] for d in vehicle_data]) if vehicle_data else 0.0
            }

            # 生成小框图并显示
            if output_image.shape[-1] == 4:
                main_img = output_image[:, :, :3].astype(np.uint8)
            else:
                main_img = output_image.astype(np.uint8)
            canvas = create_small_view_layout(main_img, base_stats, vehicle_data, type_count, distance_ranges, speed_ranges, image_w, image_h)
            cv2.imshow('2D Ground Truth (Vehicle Stats)', canvas)

            # 退出条件：按q键
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n🛑 程序被用户中断")
    finally:
        # 清理资源（恢复异步模式）
        settings.synchronous_mode = False
        world.apply_settings(settings)
        clear(world, camera)
        ego_vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()