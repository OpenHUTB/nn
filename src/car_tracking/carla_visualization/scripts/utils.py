# scripts/utils.py
import carla
import cv2
import numpy as np
import random

# ===================== 依赖导入（保留原代码的导入逻辑）=====================
try:
    from what.models.detection.datasets.coco import COCO_CLASS_NAMES
    from utils.box_utils import draw_bounding_boxes
    from utils.projection import get_image_point, build_projection_matrix, point_in_canvas, get_2d_box_from_3d_edges
    from utils.world import clear_npc, clear_static_vehicle
except ImportError:
    # 保留原代码的替代实现
    COCO_CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']
    def draw_bounding_boxes(image, boxes, labels, class_names, ids=None):
        img = image.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = class_names[labels[i]] if labels[i] < len(class_names) else 'unknown'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img
    def build_projection_matrix(w, h, fov, is_behind_camera=False):
        fov_rad = np.deg2rad(fov)
        fx = w / (2 * np.tan(fov_rad / 2))
        fy = h / (2 * np.tan(fov_rad / 2))
        cx = w / 2
        cy = h / 2
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        if is_behind_camera:
            matrix[0, 0] = -fx
        return matrix
    def get_image_point(loc, K, w2c):
        loc_vector = np.array([loc.x, loc.y, loc.z, 1])
        point_in_cam = np.dot(w2c, loc_vector)
        point_in_cam = point_in_cam / point_in_cam[3]
        point_in_2d = np.dot(K, point_in_cam[:3])
        point_in_2d = point_in_2d / point_in_2d[2]
        return (point_in_2d[0], point_in_2d[1])
    def point_in_canvas(point, h, w):
        return 0 <= point[0] <= w and 0 <= point[1] <= h
    def get_2d_box_from_3d_edges(points, edges, h, w):
        xs = [p[0] for p in points if point_in_canvas(p, h, w)]
        ys = [p[1] for p in points if point_in_canvas(p, h, w)]
        return min(xs) if xs else 0, max(xs) if xs else w, min(ys) if ys else 0, max(ys) if ys else h
    def clear_npc(world):
        for actor in world.get_actors().filter('*vehicle*'):
            actor.destroy()
    def clear_static_vehicle(world):
        pass

# ===================== 配置常量（保留原代码的所有配置）=====================
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
# 天气配置
WEATHER_SWITCH_INTERVAL = 10  # 随机天气切换间隔（秒，0表示不自动切换）
SUPPORTED_WEATHERS = {
    1: "ClearNoon",  # 晴天正午
    2: "CloudyNoon",  # 多云正午
    3: "RainyNoon",  # 雨天正午
    4: "Sunset",  # 黄昏
    5: "Foggy",  # 雾天
    6: "Stormy"  # 暴雨
}

# 3D可视化配置
DISTANCE_THRESHOLD = 80  # 车辆3D框显示距离阈值（米）
TRAFFIC_LIGHT_DISTANCE = 60  # 红绿灯显示距离阈值（米）
EDGES = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
         [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]  # 3D边界框边
# 颜色定义（BGR格式）
VEHICLE_3D_COLOR = (0, 255, 0)  # 车辆3D框默认颜色
TRAFFIC_LIGHT_COLORS = {
    0: (0, 255, 0),  # 绿色
    1: (0, 255, 255),  # 黄色
    2: (0, 0, 255),  # 红色
    3: (255, 255, 255)  # 白色（未知状态）
}
TRAFFIC_LIGHT_STATE_NAMES = {
    0: "GREEN",
    1: "YELLOW",
    2: "RED",
    3: "UNKNOWN"
}
# 全局开关（用于节点间通信控制）
SHOW_VEHICLES_3D = True
SHOW_TRAFFIC_LIGHTS = True
SHOW_TRAFFIC_LIGHTS_STATE = True

# ===================== 工具函数（保留原代码的所有工具函数）=====================
def get_vehicle_color(vehicle_id):
    """为车辆生成固定唯一的RGB颜色（基于ID种子）"""
    np.random.seed(vehicle_id)
    return tuple(np.random.randint(0, 255, 3).tolist())

def custom_draw_bounding_boxes(image, boxes, labels, class_names, ids=None, track_data=None):
    """保留原始边界框绘制逻辑，叠加跟踪数据"""
    img = draw_bounding_boxes(image, boxes, labels, class_names, ids)
    if ids is not None and track_data is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            vid = ids[i]
            if vid in track_data:
                x1, y1, x2, y2 = map(int, box)
                color = track_data[vid]['color']
                dist = track_data[vid]['distance']
                # 绘制距离文本
                cv2.putText(
                    img, f"Dist: {dist:.1f}m", (x1, y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, color, 1
                )
                # 绘制跟踪颜色外框
                cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color, 1)
    return img

def init_chart_background(width, height):
    """初始化图表背景"""
    chart = np.zeros((height, width, 3), dtype=np.uint8)
    # 绘制标题
    cv2.putText(
        chart, "Real-Time Statistics (Last 50 Frames)", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_MEDIUM, (255, 255, 255), LINE_THICKNESS
    )
    # 绘制网格
    grid_color = (50, 50, 50)
    for y in range(50, height - 30, 50):
        cv2.line(chart, (50, y), (width - 50, y), grid_color, 1)
    for x in range(50, width - 50, 50):
        cv2.line(chart, (x, 30), (x, height - 30), grid_color, 1)
    # 绘制图例
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
    """绘制实时动态折线图"""
    chart = init_chart_background(CHART_WIDTH, CHART_HEIGHT)
    if len(history_frames) == 0:
        return chart
    # 数据归一化
    max_veh = max(history_vehicles) if history_vehicles else 1
    max_veh = max_veh if max_veh != 0 else 1
    norm_veh = [(v / max_veh) * (CHART_HEIGHT - 60) for v in history_vehicles]
    y_veh = np.array([CHART_HEIGHT - 30 - v for v in norm_veh], dtype=int)

    max_d = max(history_max_dist) if history_max_dist else 1
    max_d = max_d if max_d != 0 else 1
    norm_dist = [(d / max_d) * (CHART_HEIGHT - 60) for d in history_max_dist]
    y_dist = np.array([CHART_HEIGHT - 30 - d for d in norm_dist], dtype=int)

    x_coords = np.array([
        50 + (i * (CHART_WIDTH - 100) / (len(history_frames) - 1 if len(history_frames) > 1 else 1))
        for i in range(len(history_frames))
    ], dtype=int)

    # 绘制折线和数据点
    if len(x_coords) > 1:
        cv2.polylines(chart, [np.column_stack((x_coords, y_veh))], isClosed=False, color=(0, 255, 0),
                      thickness=LINE_THICKNESS)
        cv2.polylines(chart, [np.column_stack((x_coords, y_dist))], isClosed=False, color=(0, 0, 255),
                      thickness=LINE_THICKNESS)
    for x, y in zip(x_coords, y_veh):
        cv2.circle(chart, (x, y), POINT_RADIUS, (0, 255, 0), -1)
    for x, y in zip(x_coords, y_dist):
        cv2.circle(chart, (x, y), POINT_RADIUS, (0, 0, 255), -1)

    # 绘制当前数值标注
    if len(history_vehicles) > 0 and len(history_max_dist) > 0:
        current_veh = history_vehicles[-1]
        current_dist = history_max_dist[-1]
        cv2.putText(
            chart, f"Now: {current_veh} cars | {current_dist:.1f}m",
            (CHART_WIDTH - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL,
            (255, 255, 255), 1
        )
    return chart

def convert_image_format(image):
    """将4通道BGRA图像转换为3通道RGB图像"""
    return image[..., :3] if image.shape[-1] == 4 else image.copy()

def draw_3d_objects(image, world, camera, vehicle, K, K_b):
    """在图像上绘制3D车辆边界框和交通信号灯"""
    try:
        img = image.copy()
        height, width = CAMERA_HEIGHT, CAMERA_WIDTH
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        vehicle_count_3d = 0
        traffic_light_count = 0

        # 绘制车辆3D边界框
        if SHOW_VEHICLES_3D:
            vehicles = list(world.get_actors().filter('*vehicle*'))
            for npc in vehicles:
                if npc.id == vehicle.id:
                    continue
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                if dist >= DISTANCE_THRESHOLD:
                    continue
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location
                if forward_vec.dot(ray) <= 0:
                    continue

                bb = npc.bounding_box
                verts = bb.get_world_vertices(npc.get_transform())
                points_2d = []
                for vert in verts:
                    ray0 = vert - camera.get_transform().location
                    cam_forward_vec = camera.get_transform().get_forward_vector()
                    if cam_forward_vec.dot(ray0) > 0:
                        p = get_image_point(vert, K, world_2_camera)
                    else:
                        p = get_image_point(vert, K_b, world_2_camera)
                    points_2d.append(p)

                for edge in EDGES:
                    p1 = points_2d[edge[0]]
                    p2 = points_2d[edge[1]]
                    if point_in_canvas(p1, height, width) or point_in_canvas(p2, height, width):
                        thickness = max(1, int(2 - dist / 50))
                        color_intensity = max(50, int(255 - dist))
                        color = (0, color_intensity, 0)
                        cv2.line(img, (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])), color, thickness)
                vehicle_count_3d += 1

        # 绘制交通信号灯
        if SHOW_TRAFFIC_LIGHTS:
            traffic_lights = list(world.get_actors().filter('*traffic_light*'))
            for light in traffic_lights:
                dist = light.get_transform().location.distance(vehicle.get_transform().location)
                if dist >= TRAFFIC_LIGHT_DISTANCE:
                    continue
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = light.get_transform().location - vehicle.get_transform().location
                if forward_vec.dot(ray) <= 0:
                    continue

                location = light.get_transform().location
                ray0 = location - camera.get_transform().location
                cam_forward_vec = camera.get_transform().get_forward_vector()
                if cam_forward_vec.dot(ray0) > 0:
                    point_2d = get_image_point(location, K, world_2_camera)
                else:
                    point_2d = get_image_point(location, K_b, world_2_camera)

                if not point_in_canvas(point_2d, height, width):
                    continue

                x, y = int(point_2d[0]), int(point_2d[1])
                light_state = light.get_state()
                state_mapping = {
                    carla.TrafficLightState.Green: 0,
                    carla.TrafficLightState.Yellow: 1,
                    carla.TrafficLightState.Red: 2,
                }
                state_idx = state_mapping.get(light_state, 3)
                light_color = TRAFFIC_LIGHT_COLORS[state_idx]
                state_name = TRAFFIC_LIGHT_STATE_NAMES[state_idx]

                radius = max(6, int(15 - dist / 20))
                cv2.circle(img, (x, y), radius, light_color, -1)
                cv2.circle(img, (x, y), radius, (255, 255, 255), 1)

                if SHOW_TRAFFIC_LIGHTS_STATE and radius > 4:
                    text_size = cv2.getTextSize(state_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x = x - text_size[0] // 2
                    text_y = y - radius - 5
                    cv2.rectangle(img, (text_x - 3, text_y - text_size[1] - 3),
                                  (text_x + text_size[0] + 3, text_y + 3),
                                  (40, 40, 40), -1)
                    cv2.putText(img, state_name, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                traffic_light_count += 1

        return img, vehicle_count_3d, traffic_light_count
    except Exception as e:
        print(f"3D物体绘制错误：{e}")
        return image, 0, 0

def set_weather(world, weather_type):
    """设置Carla世界的天气"""
    weather = carla.WeatherParameters()
    if weather_type == "ClearNoon":
        weather.sun_altitude_angle = 60.0
        weather.cloudiness = 0.0
        weather.precipitation = 0.0
        weather.wetness = 0.0
        weather.fog_density = 0.0
    elif weather_type == "CloudyNoon":
        weather.sun_altitude_angle = 60.0
        weather.cloudiness = 80.0
        weather.wind_intensity = 20.0
        weather.precipitation = 0.0
        weather.wetness = 0.0
        weather.fog_density = 0.0
    elif weather_type == "RainyNoon":
        weather.sun_altitude_angle = 60.0
        weather.cloudiness = 80.0
        weather.precipitation = 50.0
        weather.precipitation_deposits = 20.0
        weather.wetness = 80.0
        weather.wind_intensity = 30.0
        weather.fog_density = 10.0
    elif weather_type == "Sunset":
        weather.sun_altitude_angle = 10.0
        weather.sun_azimuth_angle = 180.0
        weather.cloudiness = 30.0
        weather.wind_intensity = 10.0
        weather.precipitation = 0.0
        weather.fog_density = 5.0
    elif weather_type == "Foggy":
        weather.sun_altitude_angle = 30.0
        weather.cloudiness = 20.0
        weather.fog_density = 90.0
        weather.fog_distance = 10.0
        weather.fog_falloff = 1.0
        weather.precipitation = 0.0
        weather.wetness = 0.0
    elif weather_type == "Stormy":
        weather.sun_altitude_angle = 30.0
        weather.cloudiness = 100.0
        weather.precipitation = 100.0
        weather.precipitation_deposits = 50.0
        weather.wetness = 100.0
        weather.wind_intensity = 70.0
        weather.fog_density = 30.0
    world.set_weather(weather)
    print(f"当前天气已切换为：{weather_type}")

def get_random_weather():
    """获取随机的天气类型"""
    weather_codes = list(SUPPORTED_WEATHERS.keys())
    random_code = random.choice(weather_codes)
    return SUPPORTED_WEATHERS[random_code]
