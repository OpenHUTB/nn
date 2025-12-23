import carla
import queue
import random
import cv2
import numpy as np
import time

# ===================== 依赖导入（保留原始依赖，新增3D可视化相关）=====================
# 注意：如果以下导入报错，请根据实际环境调整，或注释后自行实现相关函数
try:
    from what.models.detection.datasets.coco import COCO_CLASS_NAMES
    from utils.box_utils import draw_bounding_boxes
    from utils.projection import get_image_point, build_projection_matrix, point_in_canvas, get_2d_box_from_3d_edges
    from utils.world import clear_npc, clear_static_vehicle
except ImportError:
    # 提供替代实现，避免代码无法运行
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

# ===================== 配置常量（集中管理，便于修改，新增3D可视化配置）=====================
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
# 天气配置（完全自定义，不依赖Carla预定义属性）
WEATHER_SWITCH_INTERVAL = 10  # 随机天气切换间隔（秒，0表示不自动切换）
SUPPORTED_WEATHERS = {
    1: "ClearNoon",  # 晴天正午
    2: "CloudyNoon",  # 多云正午
    3: "RainyNoon",  # 雨天正午
    4: "Sunset",  # 黄昏
    5: "Foggy",  # 雾天
    6: "Stormy"  # 暴雨
}

# 新增：3D可视化配置
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
SHOW_VEHICLES_3D = True  # 是否显示车辆3D框
SHOW_TRAFFIC_LIGHTS = True  # 是否显示红绿灯
SHOW_TRAFFIC_LIGHTS_STATE = True  # 是否显示红绿灯状态文字

# ===================== 工具函数（独立封装，提升复用性，新增3D可视化函数）=====================
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

# 新增：3D物体（车辆、红绿灯）绘制函数
def draw_3d_objects(image, world, camera, vehicle, K, K_b):
    """在图像上绘制3D车辆边界框和交通信号灯（整合红绿灯真值数据和投影功能）"""
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

                # 计算距离，过滤超出阈值的车辆
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                if dist >= DISTANCE_THRESHOLD:
                    continue

                # 检查是否在车辆前方
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location
                if forward_vec.dot(ray) <= 0:
                    continue

                # 获取车辆3D边界框顶点
                bb = npc.bounding_box
                verts = bb.get_world_vertices(npc.get_transform())

                # 投影到2D图像平面（3D转2D核心逻辑）
                points_2d = []
                for vert in verts:
                    ray0 = vert - camera.get_transform().location
                    cam_forward_vec = camera.get_transform().get_forward_vector()
                    if cam_forward_vec.dot(ray0) > 0:
                        p = get_image_point(vert, K, world_2_camera)
                    else:
                        p = get_image_point(vert, K_b, world_2_camera)
                    points_2d.append(p)

                # 绘制3D边界框的边
                for edge in EDGES:
                    p1 = points_2d[edge[0]]
                    p2 = points_2d[edge[1]]
                    if point_in_canvas(p1, height, width) or point_in_canvas(p2, height, width):
                        # 距离衰减：越远越细、颜色越浅
                        thickness = max(1, int(2 - dist / 50))
                        color_intensity = max(50, int(255 - dist))
                        color = (0, color_intensity, 0)
                        cv2.line(img, (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])), color, thickness)
                vehicle_count_3d += 1

        # 绘制交通信号灯（整合红绿灯真值数据获取和状态绘制）
        if SHOW_TRAFFIC_LIGHTS:
            # 1. 获取所有红绿灯（真值数据）
            traffic_lights = list(world.get_actors().filter('*traffic_light*'))
            for light in traffic_lights:
                # 计算距离，过滤超出阈值的红绿灯
                dist = light.get_transform().location.distance(vehicle.get_transform().location)
                if dist >= TRAFFIC_LIGHT_DISTANCE:
                    continue

                # 检查是否在车辆前方
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = light.get_transform().location - vehicle.get_transform().location
                if forward_vec.dot(ray) <= 0:
                    continue

                # 2. 将3D世界坐标投影到2D图像坐标（核心投影逻辑）
                location = light.get_transform().location
                ray0 = location - camera.get_transform().location
                cam_forward_vec = camera.get_transform().get_forward_vector()
                if cam_forward_vec.dot(ray0) > 0:
                    point_2d = get_image_point(location, K, world_2_camera)
                else:
                    point_2d = get_image_point(location, K_b, world_2_camera)

                # 检查点是否在画布内
                if not point_in_canvas(point_2d, height, width):
                    continue

                x, y = int(point_2d[0]), int(point_2d[1])

                # 3. 获取红绿灯状态（真值数据）并映射颜色
                light_state = light.get_state()
                # 状态映射：Green->0, Yellow->1, Red->2, 其他->3
                state_mapping = {
                    carla.TrafficLightState.Green: 0,
                    carla.TrafficLightState.Yellow: 1,
                    carla.TrafficLightState.Red: 2,
                }
                state_idx = state_mapping.get(light_state, 3)
                light_color = TRAFFIC_LIGHT_COLORS[state_idx]
                state_name = TRAFFIC_LIGHT_STATE_NAMES[state_idx]

                # 绘制红绿灯圆形标记（距离衰减）
                radius = max(6, int(15 - dist / 20))
                cv2.circle(img, (x, y), radius, light_color, -1)
                cv2.circle(img, (x, y), radius, (255, 255, 255), 1)  # 白色描边

                # 绘制红绿灯状态文字（可选）
                if SHOW_TRAFFIC_LIGHTS_STATE and radius > 4:  # 半径足够大时才显示文字
                    text_size = cv2.getTextSize(state_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x = x - text_size[0] // 2
                    text_y = y - radius - 5  # 文字在圆形上方
                    # 绘制文字背景框，提升可读性
                    cv2.rectangle(img, (text_x - 3, text_y - text_size[1] - 3),
                                  (text_x + text_size[0] + 3, text_y + 3),
                                  (40, 40, 40), -1)  # 深色背景
                    # 绘制状态文字
                    cv2.putText(img, state_name, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                traffic_light_count += 1

        return img, vehicle_count_3d, traffic_light_count
    except Exception as e:
        print(f"3D物体绘制错误：{e}")
        return image, 0, 0

# ===================== 天气控制函数（完全自定义，不依赖任何预定义属性）=====================
def set_weather(world, weather_type):
    """
    设置Carla世界的天气（完全通过参数配置，兼容所有Carla版本）
    参数：
        world: Carla的world对象
        weather_type: 天气类型（字符串，支持ClearNoon/CloudyNoon/RainyNoon/Sunset/Foggy/Stormy）
    """
    # 初始化天气参数（所有参数默认值为0）
    weather = carla.WeatherParameters()

    if weather_type == "ClearNoon":
        # 晴天正午：无云、无雨、无雾，太阳高度高（60度）
        weather.sun_altitude_angle = 60.0  # 太阳高度角（0=地平线，90=天顶）
        weather.cloudiness = 0.0  # 云量（0-100）
        weather.precipitation = 0.0  # 降雨量（0-100）
        weather.wetness = 0.0  # 地面湿润度（0-100）
        weather.fog_density = 0.0  # 雾浓度（0-100）

    elif weather_type == "CloudyNoon":
        # 多云正午：高云量（80），太阳高度角60度，少量风
        weather.sun_altitude_angle = 60.0
        weather.cloudiness = 80.0
        weather.wind_intensity = 20.0  # 风速（0-100）
        weather.precipitation = 0.0
        weather.wetness = 0.0
        weather.fog_density = 0.0

    elif weather_type == "RainyNoon":
        # 雨天正午：中雨（50）、高云量（80）、地面湿润（80）、风速30
        weather.sun_altitude_angle = 60.0
        weather.cloudiness = 80.0
        weather.precipitation = 50.0  # 降雨量
        weather.precipitation_deposits = 20.0  # 降水沉积（地面积水）
        weather.wetness = 80.0  # 地面湿润度
        weather.wind_intensity = 30.0
        weather.fog_density = 10.0  # 少量雾

    elif weather_type == "Sunset":
        # 黄昏：太阳高度角低（10度）、橙色色调、少量云、低风速
        weather.sun_altitude_angle = 10.0  # 太阳高度角低（黄昏效果）
        weather.sun_azimuth_angle = 180.0  # 太阳方位角（180=西方）
        weather.cloudiness = 30.0
        weather.wind_intensity = 10.0
        weather.precipitation = 0.0
        weather.fog_density = 5.0  # 轻微雾（增强黄昏氛围）

    elif weather_type == "Foggy":
        # 雾天：高雾浓度（90）、低可见距离（10）、低太阳高度、少量云
        weather.sun_altitude_angle = 30.0
        weather.cloudiness = 20.0
        weather.fog_density = 90.0  # 雾浓度（越高雾越浓）
        weather.fog_distance = 10.0  # 雾的可见距离（越小雾越近）
        weather.fog_falloff = 1.0  # 雾的衰减率
        weather.precipitation = 0.0
        weather.wetness = 0.0

    elif weather_type == "Stormy":
        # 暴雨：最大降雨量（100）、高云量（100）、大风（70）、地面完全湿润（100）、中度雾
        weather.sun_altitude_angle = 30.0
        weather.cloudiness = 100.0
        weather.precipitation = 100.0  # 最大降雨量
        weather.precipitation_deposits = 50.0  # 大量积水
        weather.wetness = 100.0  # 地面完全湿润
        weather.wind_intensity = 70.0  # 大风
        weather.fog_density = 30.0  # 中度雾（暴雨伴随雾）

    # 应用天气设置到世界
    world.set_weather(weather)
    print(f"当前天气已切换为：{weather_type}")

def get_random_weather():
    """获取随机的天气类型（从SUPPORTED_WEATHERS中随机选择）"""
    weather_codes = list(SUPPORTED_WEATHERS.keys())
    random_code = random.choice(weather_codes)
    return SUPPORTED_WEATHERS[random_code]

# ===================== 主程序逻辑（优化结构，修复所有已知问题，整合3D可视化）=====================
def main():
    # 声明全局变量用于按键控制3D显示
    global SHOW_VEHICLES_3D, SHOW_TRAFFIC_LIGHTS, SHOW_TRAFFIC_LIGHTS_STATE
    # 初始化变量，用于资源清理
    camera = None
    vehicle = None
    world = None
    try:
        # 初始化Carla客户端和世界
        client = carla.Client('localhost', 2000)
        # 设置超时时间，避免连接卡顿
        client.set_timeout(10.0)
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
        # 先停止监听（防止之前的残留），再开始监听
        if camera.is_listening:
            camera.stop()
        camera.listen(lambda image: camera_callback(image, image_queue))

        # 清理现有NPC（保留原始逻辑）
        clear_npc(world)
        clear_static_vehicle(world)

        # 2D框计算相关参数（保留原始逻辑，复用K和K_b用于3D投影）
        edges = EDGES  # 复用3D边界框边定义
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
        if vehicle:
            vehicle.set_autopilot(True)

        # 初始化跟踪和数据缓存变量（改为局部变量，减少全局变量）
        tracked_vehicles = {}  # key:车辆ID, value:{'color':颜色, 'distance':距离, 'frame':帧数}
        frame_counter = 0
        history_frames = []  # 最近50帧的帧数
        history_vehicles = []  # 最近50帧的车辆数量
        history_max_dist = []  # 最近50帧的最大距离

        # ========== 初始化天气（新增）==========
        current_weather = "ClearNoon"  # 默认晴天
        set_weather(world, current_weather)
        last_weather_switch_time = time.time()  # 记录最后一次天气切换时间

        # 主循环（优化逻辑，提升可读性，加入3D可视化）
        while True:
            world.tick()
            frame_counter += 1

            # ========== 天气切换逻辑（新增）==========
            # 1. 自动随机切换天气（如果配置了间隔）
            auto_switch = False
            if WEATHER_SWITCH_INTERVAL > 0:
                current_time = time.time()
                if current_time - last_weather_switch_time >= WEATHER_SWITCH_INTERVAL:
                    current_weather = get_random_weather()
                    set_weather(world, current_weather)
                    last_weather_switch_time = current_time
                    auto_switch = True

            # 2. 键盘按键处理（合并逻辑，避免重复调用cv2.waitKey，新增3D显示控制）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # 按数字键1-6切换对应天气（仅在非自动切换时响应，避免冲突）
            elif not auto_switch and key in [ord(str(code)) for code in SUPPORTED_WEATHERS.keys()]:
                weather_code = int(chr(key))
                current_weather = SUPPORTED_WEATHERS[weather_code]
                set_weather(world, current_weather)
                last_weather_switch_time = time.time()  # 重置自动切换时间
            # 新增：按键控制3D显示
            elif key == ord('v'):
                SHOW_VEHICLES_3D = not SHOW_VEHICLES_3D
                print(f"车辆3D框显示：{'开启' if SHOW_VEHICLES_3D else '关闭'}")
            elif key == ord('t'):
                SHOW_TRAFFIC_LIGHTS = not SHOW_TRAFFIC_LIGHTS
                print(f"红绿灯显示：{'开启' if SHOW_TRAFFIC_LIGHTS else '关闭'}")
            elif key == ord('s'):
                SHOW_TRAFFIC_LIGHTS_STATE = not SHOW_TRAFFIC_LIGHTS_STATE
                print(f"红绿灯状态文字显示：{'开启' if SHOW_TRAFFIC_LIGHTS_STATE else '关闭'}")

            # 移动观众视角到车辆顶部（保留原始逻辑）
            if vehicle:
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
                if vehicle and npc.id != vehicle.id:
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
            labels = np.array([2] * len(boxes))  # 2对应COCO的car类别
            probs = np.array([1.0] * len(boxes))
            output = custom_draw_bounding_boxes(
                image, boxes, labels, COCO_CLASS_NAMES, ids, track_data
            ) if len(boxes) > 0 else image

            # ========== 新增：绘制3D物体（车辆3D框、红绿灯）==========
            output_3d, _, _ = draw_3d_objects(output, world, camera, vehicle, K, K_b)

            # 转换图像格式并拼接图表（优化逻辑，使用带3D的图像）
            output_rgb = convert_image_format(output_3d)
            chart_image = draw_dynamic_chart(history_frames, history_vehicles, history_max_dist)
            combined_image = np.hstack((output_rgb, chart_image))

            # ========== 绘制天气信息到图像（新增）==========
            cv2.putText(
                combined_image, f"Weather: {current_weather}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_MEDIUM, (255, 255, 255), 2
            )
            # 绘制天气和3D控制提示
            cv2.putText(
                combined_image, "Press 1-6:Weather | V:3D Vehicles | T:Traffic Lights | S:Light State | Q:Quit",
                (10, CAMERA_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, (255, 255, 255), 1
            )

            # 绘制跟踪监测窗口（保留原始功能，优化代码）
            track_window = np.zeros((TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                track_window, "Vehicle Tracking Monitor", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_MEDIUM, (255, 255, 255), LINE_THICKNESS
            )
            # 新增：在跟踪窗口显示当前天气和3D显示状态
            cv2.putText(
                track_window, f"Current Weather: {current_weather}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, (255, 255, 255), 1
            )
            cv2.putText(
                track_window, f"3D Vehicles: {'ON' if SHOW_VEHICLES_3D else 'OFF'}",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, (0, 255, 0), 1
            )
            cv2.putText(
                track_window, f"Traffic Lights: {'ON' if SHOW_TRAFFIC_LIGHTS else 'OFF'}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, (0, 255, 255), 1
            )
            y_offset = 135
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
            cv2.imshow('2D Ground Truth + 3D Visualization', combined_image)
            cv2.imshow('Vehicle Tracking Monitor', track_window)

    except KeyboardInterrupt:
        print("程序被用户中断！")
    except Exception as e:
        print(f"程序运行出错：{e}")
    finally:
        # 清理资源（修复传感器注销警告，优化异常处理）
        print("开始清理资源...")
        # 1. 停止相机监听并销毁（避免注销警告）
        if camera:
            if camera.is_listening:
                camera.stop()
            camera.destroy()
            print("相机已销毁")
        # 2. 销毁主车辆
        if vehicle:
            vehicle.destroy()
            print("主车辆已销毁")
        # 3. 恢复Carla世界设置（避免同步模式残留）
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            print("Carla同步模式已关闭")
        # 4. 关闭OpenCV窗口
        cv2.destroyAllWindows()
        print("资源清理完成")

if __name__ == '__main__':
    main()