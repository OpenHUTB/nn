import carla
import queue
import random
import cv2
import numpy as np
import time  # 用于计算FPS
from collections import deque  # 用于平滑FPS计算

# 替换为你实际的导入路径，如果没有这些模块，先实现基础版本
# 这里先实现缺失的工具函数，避免代码报错
# ---------------------- 缺失工具函数的基础实现 ----------------------
COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def draw_bounding_boxes(image, boxes, labels, class_names, ids=None):
    """
    绘制边界框（基础实现）
    """
    img = image.copy()
    # 转换为BGR格式（Carla的图像是RGBA，OpenCV是BGR）
    img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = class_names[labels[i]] if labels[i] < len(class_names) else 'unknown'
        id_text = f'ID: {ids[i]}' if ids is not None else ''
        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制标签和ID
        cv2.putText(img, f'{label} {id_text}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """
    构建相机投影矩阵（基础实现）
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    if is_behind_camera:
        K[1, 1] = -K[1, 1]  # 翻转y轴
    return K

def get_image_point(world_point, K, world_to_camera):
    """
    将3D世界坐标转换为2D图像坐标
    """
    # 将Carla Location转换为齐次坐标
    point = np.array([world_point.x, world_point.y, world_point.z, 1])
    # 转换到相机坐标系
    camera_point = np.dot(world_to_camera, point)
    # 转换到图像坐标系（归一化）
    image_point = np.dot(K, camera_point[:3])
    # 透视除法
    image_point = image_point / image_point[2]
    return np.array([image_point[0], image_point[1]])

def get_2d_box_from_3d_edges(points_2d, edges, h, w):
    """
    从3D边缘的2D点计算边界框
    """
    x_coords = [p[0] for p in points_2d]
    y_coords = [p[1] for p in points_2d]
    x_min = max(0, min(x_coords))
    x_max = min(w, max(x_coords))
    y_min = max(0, min(y_coords))
    y_max = min(h, max(y_coords))
    return x_min, x_max, y_min, y_max

def point_in_canvas(point, h, w):
    """
    判断点是否在图像画布内
    """
    x, y = point
    return 0 <= x <= w and 0 <= y <= h

def clear_npc(world):
    """
    清除现有NPC车辆
    """
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()

def clear_static_vehicle(world):
    """
    清除静态车辆（与clear_npc功能重复，这里简化）
    """
    pass

def clear(world, *actors):
    """
    清除指定的actor
    """
    settings = world.get_settings()
    settings.synchronous_mode = False  # 关闭同步模式
    world.apply_settings(settings)
    for actor in actors:
        if actor.is_alive:
            actor.destroy()
    # 清除剩余的NPC
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()
# ---------------------- 工具函数结束 ----------------------

def camera_callback(image, rgb_image_queue):
    """
    相机回调函数，将图像数据存入队列
    """
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data),
                        (image.height, image.width, 4)))

def get_vehicle_speed(vehicle):
    """
    获取车辆的速度（km/h）
    """
    velocity = vehicle.get_velocity()
    speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed

def main():
    # 初始化Carla客户端
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # 设置超时时间，避免连接失败
    world = client.get_world()

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 获取 spectator 和生成点
    spectator = world.get_spectator()
    spawn_points = world.get_map().get_spawn_points()

    # 生成ego车辆
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = None
    # 循环尝试生成，避免生成失败
    while not vehicle:
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

    # 生成相机
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '640')
    camera_init_trans = carla.Transform(carla.Location(x=1, z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    # 图像队列
    image_queue = queue.Queue()
    camera.listen(lambda image: camera_callback(image, image_queue))

    # 清除现有NPC
    clear_npc(world)
    clear_static_vehicle(world)

    # 3D边界框的边缘对
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
             [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

    # 相机参数
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()
    K = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    # 生成50个NPC车辆
    for i in range(50):
        car_bp = [bp for bp in bp_lib.filter('vehicle') if int(bp.get_attribute('number_of_wheels')) == 4]
        if car_bp:
            npc = world.try_spawn_actor(random.choice(car_bp), random.choice(spawn_points))
            if npc:
                npc.set_autopilot(True)
    vehicle.set_autopilot(True)

    # ---------------------- 修复：天气预设（纯英文名称，解决问号问题） ----------------------
    # 1. 先打印当前版本支持的天气预设（可选，用于调试）
    weather_attributes = [attr for attr in dir(carla.WeatherParameters) if not attr.startswith('_')]
    print("="*50)
    print("Current Carla Weather Presets:", weather_attributes)
    print("="*50)

    # 2. 步骤1：定义仅使用你版本支持的**预设常量**，纯英文名称
    preset_weathers = [
        (carla.WeatherParameters.ClearNoon, "ClearNoon"),
        (carla.WeatherParameters.CloudyNoon, "CloudyNoon"),
        (carla.WeatherParameters.WetNoon, "WetNoon"),
        (carla.WeatherParameters.WetCloudyNoon, "WetCloudyNoon"),
        (carla.WeatherParameters.MidRainyNoon, "MidRainyNoon"),
        (carla.WeatherParameters.HardRainNoon, "HardRainNoon"),
        (carla.WeatherParameters.SoftRainNoon, "SoftRainNoon"),
        (carla.WeatherParameters.ClearSunset, "ClearSunset"),
        (carla.WeatherParameters.CloudySunset, "CloudySunset"),
        (carla.WeatherParameters.WetSunset, "WetSunset")
    ]

    # 3. 步骤2：创建自定义雾天天气（调优参数，雾天效果更明显）
    foggy_weather = carla.WeatherParameters()
    # 核心雾天参数（调优后）
    foggy_weather.fog_density = 0.95  # 雾浓度（0-1，调至接近1，雾更浓）
    foggy_weather.fog_distance = 5.0  # 雾的起始距离（越小，雾离相机越近）
    foggy_weather.fog_falloff = 0.1  # 雾的衰减率（越小，雾的范围越大，更均匀）
    # 辅助参数，增强雾天视觉效果
    foggy_weather.sun_altitude_angle = 10.0  # 太阳高度角（越小，光线越暗，雾更明显）
    foggy_weather.cloudiness = 0.9  # 云量（越大，天空越暗，雾的对比越强）
    foggy_weather.scattering_intensity = 0.1  # 散射强度（越小，光线越弱）
    foggy_weather.wetness = 0.1  # 地面湿度（降低，避免和雨天混淆）
    foggy_weather.precipitation = 0.0  # 关闭降水，纯雾天
    # 自定义雾天名称（纯英文）
    foggy_weather_name = "CustomFoggyNoon (Heavy Fog)"

    # 4. 步骤3：合并预设和自定义天气，组成最终的天气列表（元组：天气对象，天气名称）
    weather_presets = [(wp, name) for wp, name in preset_weathers] + [(foggy_weather, foggy_weather_name)]
    random.shuffle(weather_presets)  # 随机打乱天气顺序，切换更自然

    current_weather_idx = 0
    # 初始设置天气
    initial_wp, initial_name = weather_presets[current_weather_idx]
    world.set_weather(initial_wp)
    print(f"Initial Weather: {initial_name}")
    weather_switch_interval = 10.0  # 每10秒切换一次天气
    last_weather_switch = time.time()

    # ---------------------- FPS计算初始化 ----------------------
    frame_times = deque(maxlen=10)
    last_frame_time = time.time()
    # ---------------------- 初始化结束 ----------------------

    # 主循环
    try:
        while True:
            world.tick()
            current_time = time.time()

            # 移动spectator到车辆上方
            transform = carla.Transform(vehicle.get_transform().transform(
                carla.Location(x=-4, z=50)), carla.Rotation(yaw=-180, pitch=-90))
            spectator.set_transform(transform)

            # 获取图像
            image = image_queue.get()

            # 更新相机的世界到相机矩阵
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # 检测车辆并计算2D边界框
            boxes = []
            ids = []
            vehicles = world.get_actors().filter('*vehicle*')
            for npc in vehicles:
                if npc.id != vehicle.id:
                    bb = npc.bounding_box
                    dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                    if dist < 50:
                        forward_vec = vehicle.get_transform().get_forward_vector()
                        ray = npc.get_transform().location - vehicle.get_transform().location
                        if forward_vec.dot(ray) > 0:
                            verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                            points_2d = []
                            for vert in verts:
                                ray0 = vert - camera.get_transform().location
                                cam_forward_vec = camera.get_transform().get_forward_vector()
                                if cam_forward_vec.dot(ray0) > 0:
                                    p = get_image_point(vert, K, world_2_camera)
                                else:
                                    p = get_image_point(vert, K_b, world_2_camera)
                                points_2d.append(p)
                            x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(
                                points_2d, edges, image_h, image_w)
                            if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                                if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):
                                    ids.append(npc.id)
                                    boxes.append(np.array([x_min, y_min, x_max, y_max]))

            # 绘制边界框
            boxes = np.array(boxes)
            labels = np.array([2] * len(boxes))  # 2: car
            probs = np.array([1.0] * len(boxes))
            if len(boxes) > 0:
                output_image = draw_bounding_boxes(image, boxes, labels, COCO_CLASS_NAMES, ids)
            else:
                # 没有边界框时，直接转换图像格式
                output_image = image[:, :, :3]
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGBA2BGR)

            # ---------------------- 实时天气切换（纯英文名称打印） ----------------------
            if current_time - last_weather_switch > weather_switch_interval:
                current_weather_idx = (current_weather_idx + 1) % len(weather_presets)
                current_wp, current_name = weather_presets[current_weather_idx]
                world.set_weather(current_wp)
                last_weather_switch = current_time
                # 醒目打印天气切换信息
                print("-"*50)
                print(f"Weather Switched to: {current_name}")
                print("-"*50)
            # ---------------------- 天气切换结束 ----------------------

            # ---------------------- 实时统计面板绘制（纯英文，无问号） ----------------------
            # 1. 计算FPS
            frame_time = current_time - last_frame_time
            frame_times.append(frame_time)
            fps = 1.0 / np.mean(frame_times) if frame_times else 0.0
            last_frame_time = current_time

            # 2. 统计数据
            total_vehicles = len(vehicles)  # 总车辆数
            detected_vehicles = len(boxes)  # 检测到的车辆数
            ego_speed = get_vehicle_speed(vehicle)  # ego车辆速度（km/h）
            # 获取当前天气名称（纯英文）
            current_wp, current_weather = weather_presets[current_weather_idx]

            # 3. 调整面板大小（纯英文无需太宽）
            panel_x = 10
            panel_y = 10
            panel_width = 380  # 从450调回380，适配纯英文
            panel_height = 160
            # 绘制半透明矩形背景
            overlay = output_image.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            alpha = 0.6  # 透明度
            cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)

            # 绘制统计文本（纯英文，字体清晰）
            text_y_offset = 30
            text_params = (cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # 字体、大小、颜色、粗细
            cv2.putText(output_image, f"FPS: {fps:.1f}", (panel_x + 15, panel_y + text_y_offset), *text_params)
            cv2.putText(output_image, f"Total Vehicles: {total_vehicles}", (panel_x + 15, panel_y + text_y_offset * 2), *text_params)
            cv2.putText(output_image, f"Detected Vehicles: {detected_vehicles}", (panel_x + 15, panel_y + text_y_offset * 3), *text_params)
            cv2.putText(output_image, f"Ego Speed: {ego_speed:.1f} km/h", (panel_x + 15, panel_y + text_y_offset * 4), *text_params)
            # 处理稍长的天气名称（如CustomFoggyNoon (Heavy Fog)）
            if len(current_weather) > 22:
                weather_part1 = current_weather[:22]
                weather_part2 = current_weather[22:]
                cv2.putText(output_image, f"Weather: {weather_part1}", (panel_x + 15, panel_y + text_y_offset * 5), *text_params)
                cv2.putText(output_image, f"         {weather_part2}", (panel_x + 15, panel_y + text_y_offset * 6), *text_params)
            else:
                cv2.putText(output_image, f"Weather: {current_weather}", (panel_x + 15, panel_y + text_y_offset * 5), *text_params)
            # ---------------------- 统计面板绘制结束 ----------------------

            # 显示图像
            cv2.imshow('2D Ground Truth (with Stats & Weather)', output_image)
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nUser Interrupted the Program")
    finally:
        # 清理资源
        clear(world, camera, vehicle)
        cv2.destroyAllWindows()
        print("Program Ended, Resources Cleared")

if __name__ == '__main__':
    main()