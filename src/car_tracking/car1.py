import carla
import queue
import random
import cv2
import numpy as np
import time
from collections import deque

# ---------------------- 替代utils的基础工具函数 ----------------------
def draw_bounding_boxes(image, boxes, labels, class_names, ids=None):
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = class_names[labels[i]] if labels[i] < len(class_names) else 'unknown'
        id_text = f'ID: {ids[i]}' if ids is not None else ''
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {id_text}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    if is_behind_camera:
        K[1, 1] = -K[1, 1]
    return K

def get_image_point(world_point, K, world_to_camera):
    point = np.array([world_point.x, world_point.y, world_point.z, 1])
    camera_point = np.dot(world_to_camera, point)
    image_point = np.dot(K, camera_point[:3])
    image_point = image_point / image_point[2]
    return np.array([image_point[0], image_point[1]])

def point_in_canvas(point, h, w):
    x, y = point
    return 0 <= x <= w and 0 <= y <= h

def clear_npc(world):
    for actor in world.get_actors().filter('*vehicle*'):
        try:
            actor.destroy()
        except Exception:
            pass

def clear_static_vehicle(world):
    pass

def clear(world, *actors):
    """只对传感器调用stop()，车辆直接destroy()"""
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
    except Exception as e:
        print(f"警告：关闭同步模式失败 - {e}")
    for actor in actors:
        try:
            if actor and actor.is_alive:
                if 'sensor' in actor.type_id:
                    actor.stop()
                actor.destroy()
        except Exception as e:
            print(f"警告：销毁Actor失败 - {e}")
    clear_npc(world)

def get_vehicle_speed(vehicle):
    velocity = vehicle.get_velocity()
    speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed

# ---------------------- 配置参数（集中管理） ----------------------
class Config:
    CARLA_HOST = 'localhost'
    CARLA_PORT = 2000
    CARLA_TIMEOUT = 10.0
    SYNC_MODE = True
    FIXED_DELTA = 0.05
    NPC_COUNT = 50
    MAX_DETECTION_DIST = 50.0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 640
    CAMERA_FOV = 90.0
    CAMERA_POS = carla.Transform(carla.Location(x=1, z=2))
    FPS_WINDOW_SIZE = 10
    PANEL_ALPHA = 0.6
    WEATHER_SWITCH_INTERVAL = 5.0
    ENABLE_WEATHER_SWITCH = True
    WEATHER_DISPLAY_MAX_LEN = 20

# ---------------------- 核心功能封装 ----------------------
class CameraSensor:
    def __init__(self, world, vehicle, config):
        self.world = world
        self.vehicle = vehicle
        self.config = config
        self.bp = self._create_bp()
        self.actor = None
        self.queue = queue.Queue(maxsize=1)
        self._spawn()

    def _create_bp(self):
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.config.CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(self.config.CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(self.config.CAMERA_FOV))
        return camera_bp

    def _spawn(self):
        try:
            self.actor = self.world.spawn_actor(self.bp, self.config.CAMERA_POS, attach_to=self.vehicle)
            self.actor.listen(lambda img: self._callback(img))
        except carla.exceptions.CarlaError as e:
            raise RuntimeError(f"相机生成失败：{e}")

    def _callback(self, image):
        if not self.queue.full():
            img_data = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            self.queue.put(img_data)

    def get_image(self):
        try:
            return self.queue.get(timeout=0.1)
        except queue.Empty:
            # 图像为空时返回空的numpy数组，而非None，避免分支跳过
            return np.zeros((self.config.CAMERA_HEIGHT, self.config.CAMERA_WIDTH, 4), dtype=np.uint8)

    def destroy(self):
        if self.actor:
            self.actor.stop()
            self.actor.destroy()

class Vehicle3dBoxRenderer:
    """完整保留3D边界框渲染逻辑"""
    def __init__(self, world, ego_vehicle, camera, config):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.camera = camera
        self.config = config
        self.edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
                      [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
        self.K = build_projection_matrix(
            config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FOV
        )
        self.K_b = build_projection_matrix(
            config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FOV, is_behind_camera=True
        )

    def render(self, image):
        img_h, img_w = self.config.CAMERA_HEIGHT, self.config.CAMERA_WIDTH
        world_2_camera = np.array(self.camera.actor.get_transform().get_inverse_matrix())
        ego_transform = self.ego_vehicle.get_transform()
        ego_forward = ego_transform.get_forward_vector()
        camera_transform = self.camera.actor.get_transform()
        cam_forward = camera_transform.get_forward_vector()

        total_vehicles = 0
        rendered_vehicles = 0

        for npc in self.world.get_actors().filter('*vehicle*'):
            total_vehicles += 1
            if npc.id == self.ego_vehicle.id:
                continue

            dist = npc.get_transform().location.distance(ego_transform.location)
            if dist > self.config.MAX_DETECTION_DIST:
                continue

            ray = npc.get_transform().location - ego_transform.location
            if ego_forward.dot(ray) <= 0:
                continue

            rendered_vehicles += 1
            bb = npc.bounding_box
            verts = [v for v in bb.get_world_vertices(npc.get_transform())]
            points_2d = []

            for vert in verts:
                ray0 = vert - camera_transform.location
                if cam_forward.dot(ray0) > 0:
                    p = get_image_point(vert, self.K, world_2_camera)
                else:
                    p = get_image_point(vert, self.K_b, world_2_camera)
                points_2d.append(p)

            for edge in self.edges:
                p1 = points_2d[edge[0]]
                p2 = points_2d[edge[1]]
                p1_in = point_in_canvas(p1, img_h, img_w)
                p2_in = point_in_canvas(p2, img_h, img_w)
                if p1_in or p2_in:
                    cv2.line(
                        image,
                        (int(p1[0]), int(p1[1])),
                        (int(p2[0]), int(p2[1])),
                        (255, 0, 0, 255),
                        1
                    )

        return image, total_vehicles, rendered_vehicles

# ---------------------- 天气管理类 ----------------------
class WeatherManager:
    def __init__(self, world, config):
        self.world = world
        self.config = config
        self.current_weather_name = ""
        self.weather_list = [
            (carla.WeatherParameters.ClearNoon, "ClearNoon"),
            (carla.WeatherParameters.CloudyNoon, "CloudyNoon"),
            (carla.WeatherParameters.WetNoon, "WetNoon"),
            (carla.WeatherParameters.MidRainyNoon, "MidRainyNoon"),
            (carla.WeatherParameters.HardRainNoon, "HardRainNoon"),
            (carla.WeatherParameters.ClearSunset, "ClearSunset"),
            (carla.WeatherParameters.CloudySunset, "CloudySunset"),
            (carla.WeatherParameters.WetSunset, "WetSunset"),
            (self._create_foggy_weather(), "FoggyNoon")
        ]
        self.current_index = 0
        self.last_switch_time = time.perf_counter()
        self._apply_weather(self.current_index)

    def _create_foggy_weather(self):
        weather = carla.WeatherParameters()
        weather.fog_density = 0.7
        weather.fog_distance = 10.0
        weather.cloudiness = 0.8
        weather.precipitation = 0.0
        return weather

    def _apply_weather(self, index):
        self.current_index = index % len(self.weather_list)
        weather_params, weather_name = self.weather_list[self.current_index]
        self.world.set_weather(weather_params)
        self.world.tick()
        self.current_weather_name = weather_name
        print(f"\n=== 天气已切换为：{weather_name} ===")

    def update(self):
        if not self.config.ENABLE_WEATHER_SWITCH:
            return self.current_weather_name
        current_time = time.perf_counter()
        if current_time - self.last_switch_time >= self.config.WEATHER_SWITCH_INTERVAL:
            self.current_index += 1
            self._apply_weather(self.current_index)
            self.last_switch_time = current_time
        return self.current_weather_name

    def switch_manually(self):
        self.current_index += 1
        self._apply_weather(self.current_index)
        self.last_switch_time = time.perf_counter()
        return self.current_weather_name

# ---------------------- 主函数 ----------------------
def main():
    config = Config()
    # 1. 初始化Carla客户端
    try:
        client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
        client.set_timeout(config.CARLA_TIMEOUT)
        world = client.get_world()
    except carla.exceptions.CarlaError as e:
        print(f"Carla连接失败：{e}")
        return

    # 2. 设置同步模式
    try:
        settings = world.get_settings()
        settings.synchronous_mode = config.SYNC_MODE
        settings.fixed_delta_seconds = config.FIXED_DELTA
        world.apply_settings(settings)
    except carla.exceptions.CarlaError as e:
        print(f"设置同步模式失败：{e}")
        return

    # 3. 获取基础组件
    spectator = world.get_spectator()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("没有找到生成点")
        return

    # 4. 生成Ego车辆
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle_bp.set_attribute('role_name', 'ego')
    vehicle = None
    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            break
    if not vehicle:
        print("Ego车辆生成失败")
        return

    # 5. 清除现有NPC
    clear_npc(world)
    clear_static_vehicle(world)

    # 6. 生成NPC车辆
    car_bp = [bp for bp in bp_lib.filter('vehicle') if int(bp.get_attribute('number_of_wheels')) == 4]
    if not car_bp:
        print("没有找到四轮车辆蓝图")
        return
    spawn_point_idx = 0
    npc_count = 0
    while npc_count < config.NPC_COUNT and spawn_point_idx < len(spawn_points) * 2:
        spawn_point = spawn_points[spawn_point_idx % len(spawn_points)]
        npc_bp = random.choice(car_bp)
        if npc_bp.has_attribute('color'):
            color = random.choice(npc_bp.get_attribute('color').recommended_values)
            npc_bp.set_attribute('color', color)
        npc = world.try_spawn_actor(npc_bp, spawn_point)
        if npc:
            npc.set_autopilot(True)
            npc_count += 1
        spawn_point_idx += 1
    print(f"成功生成 {npc_count} 个NPC车辆")
    time.sleep(0.5)
    vehicle.set_autopilot(True)

    # 7. 初始化组件
    try:
        camera = CameraSensor(world, vehicle, config)
        renderer = Vehicle3dBoxRenderer(world, vehicle, camera, config)
        weather_manager = WeatherManager(world, config)
    except RuntimeError as e:
        print(e)
        clear(world, vehicle)
        return

    # 8. 初始化FPS统计
    frame_times = deque(maxlen=config.FPS_WINDOW_SIZE)
    last_frame_time = time.perf_counter()

    # 创建显示窗口（提前创建，避免窗口闪烁）
    cv2.namedWindow('3D Ground Truth + Weather + Stats', cv2.WINDOW_NORMAL)

    # 主循环
    try:
        while True:
            if config.SYNC_MODE:
                world.tick()
            current_time = time.perf_counter()

            # 移动 spectator 到车辆上方
            transform = carla.Transform(
                vehicle.get_transform().transform(carla.Location(x=-4, z=50)),
                carla.Rotation(yaw=-180, pitch=-90)
            )
            spectator.set_transform(transform)

            # 获取相机图像（此时不会返回None，而是空数组）
            image = camera.get_image()

            # 图像处理
            image_bgr = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGBA2BGR)

            # 绘制3D边界框（完整保留，即使图像为空也会执行，变量必然赋值）
            image_bgr, total_veh, rendered_veh = renderer.render(image_bgr)

            # 更新天气（每次循环都执行，确保最新）
            current_weather = weather_manager.update()

            # 计算FPS和车辆速度
            frame_time = current_time - last_frame_time
            frame_times.append(frame_time)
            fps = 1.0 / np.mean(frame_times) if frame_times else 0.0
            last_frame_time = current_time
            ego_speed = get_vehicle_speed(vehicle)

            # ---------------------- 确保天气在面板显示的核心修改 ----------------------
            # 1. 绘制半透明背景面板
            panel_x, panel_y = 10, 10
            panel_w, panel_h = 420, 220
            overlay = image_bgr.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, config.PANEL_ALPHA, image_bgr, 1 - config.PANEL_ALPHA, 0, image_bgr)
            # 2. 绘制文本（天气信息优先显示，确保不被遮挡）
            text_y_step = 30
            text_pos_y = panel_y + text_y_step
            text_style = (cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # 先显示天气信息（核心：放在前面，避免被其他信息挤掉）
            cv2.putText(image_bgr, f"Current Weather: {current_weather}", (panel_x + 15, text_pos_y), *text_style)
            text_pos_y += text_y_step
            # 再显示其他统计信息
            cv2.putText(image_bgr, f"FPS: {fps:.1f}", (panel_x + 15, text_pos_y), *text_style)
            text_pos_y += text_y_step
            cv2.putText(image_bgr, f"Ego Speed: {ego_speed:.1f} km/h", (panel_x + 15, text_pos_y), *text_style)
            text_pos_y += text_y_step
            cv2.putText(image_bgr, f"Total Vehicles: {total_veh}", (panel_x + 15, text_pos_y), *text_style)
            text_pos_y += text_y_step
            cv2.putText(image_bgr, f"Rendered 3D Boxes: {rendered_veh}", (panel_x + 15, text_pos_y), *text_style)
            text_pos_y += text_y_step
            cv2.putText(image_bgr, f"Press 'q' to quit | 'w' to switch weather", (panel_x + 15, text_pos_y), *text_style)

            # 显示图像（强制刷新窗口）
            cv2.imshow('3D Ground Truth + Weather + Stats', image_bgr)
            # 键盘事件处理（手动切换天气后立即刷新）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                current_weather = weather_manager.switch_manually()
                # 手动切换后强制刷新显示
                cv2.imshow('3D Ground Truth + Weather + Stats', image_bgr)

    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序异常：{e}")
    finally:
        camera.destroy()
        clear(world, vehicle)
        cv2.destroyAllWindows()
        print("资源已清理，程序结束")

if __name__ == '__main__':
    main()