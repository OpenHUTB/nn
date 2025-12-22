# 导入必要的库
import carla  # Carla仿真器的Python API
import queue  # 用于相机图像的队列存储
import random  # 用于随机生成NPC车辆和颜色
import cv2  # OpenCV库，用于图像处理和显示
import numpy as np  # 数值计算库，用于矩阵和数组操作
import time  # 用于计时和FPS计算
from collections import deque  # 用于平滑FPS计算的双端队列

# ---------------------- 替代utils的基础工具函数 ----------------------
def draw_bounding_boxes(image, boxes, labels, class_names, ids=None):
    """
    在图像上绘制2D边界框（包含类别标签和ID）
    参数：
        image: 输入图像（numpy数组）
        boxes: 边界框坐标列表，格式为[[x1, y1, x2, y2], ...]
        labels: 边界框对应的类别标签索引列表
        class_names: 类别名称列表（如COCO类别）
        ids: 可选，边界框对应的对象ID列表（如车辆ID）
    返回：
        绘制了边界框的图像
    """
    img = image.copy()  # 复制图像，避免修改原图像
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)  # 转换为整数坐标
        # 获取类别名称
        label = class_names[labels[i]] if labels[i] < len(class_names) else 'unknown'
        id_text = f'ID: {ids[i]}' if ids is not None else ''  # 拼接ID文本
        # 绘制矩形边界框（绿色，线宽2）
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制类别和ID文本（绿色，字体大小0.5，线宽2）
        cv2.putText(img, f'{label} {id_text}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """
    构建相机的投影矩阵，用于将3D世界坐标转换为2D图像坐标
    参数：
        w: 图像宽度
        h: 图像高度
        fov: 相机视场角（度数）
        is_behind_camera: 是否为相机后方的点（需要翻转y轴）
    返回：
        3x3的投影矩阵K
    """
    # 计算焦距（基于视场角和图像宽度）
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    # 初始化投影矩阵（单位矩阵）
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal  # x和y方向的焦距
    K[0, 2] = w / 2.0  # 图像中心x坐标
    K[1, 2] = h / 2.0  # 图像中心y坐标
    if is_behind_camera:
        K[1, 1] = -K[1, 1]  # 翻转y轴，处理相机后方的点
    return K

def get_image_point(world_point, K, world_to_camera):
    """
    将3D世界坐标转换为2D图像坐标
    参数：
        world_point: Carla的Location对象（3D世界坐标）
        K: 相机投影矩阵
        world_to_camera: 世界坐标系到相机坐标系的转换矩阵
    返回：
        2D图像坐标（x, y）
    """
    # 将3D世界坐标转换为齐次坐标（4维）
    point = np.array([world_point.x, world_point.y, world_point.z, 1])
    # 转换到相机坐标系（4维）
    camera_point = np.dot(world_to_camera, point)
    # 转换到图像坐标系（3维，归一化）
    image_point = np.dot(K, camera_point[:3])
    # 透视除法，得到2D图像坐标
    image_point = image_point / image_point[2]
    return np.array([image_point[0], image_point[1]])

def point_in_canvas(point, h, w):
    """
    判断2D点是否在图像画布范围内
    参数：
        point: 2D点坐标（x, y）
        h: 图像高度
        w: 图像宽度
    返回：
        布尔值，True表示在画布内，False表示超出范围
    """
    x, y = point
    return 0 <= x <= w and 0 <= y <= h

def clear_npc(world):
    """
    清除仿真世界中所有的NPC车辆
    参数：
        world: Carla的World对象
    """
    for actor in world.get_actors().filter('*vehicle*'):
        try:
            actor.destroy()  # 销毁车辆Actor
        except Exception:
            pass  # 忽略销毁失败的异常

def clear_static_vehicle(world):
    """
    清除静态车辆（预留函数，与clear_npc功能重复，暂为空实现）
    参数：
        world: Carla的World对象
    """
    pass

def clear(world, *actors):
    """
    清理指定的Actor并关闭同步模式，最后清除剩余的NPC车辆
    注意：区分传感器和车辆Actor，传感器需要先stop()再destroy()
    参数：
        world: Carla的World对象
        *actors: 要清理的Actor列表（如相机、车辆）
    """
    try:
        # 关闭同步模式，恢复仿真器默认设置
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
    except Exception as e:
        print(f"警告：关闭同步模式失败 - {e}")
    # 遍历并销毁指定的Actor
    for actor in actors:
        try:
            if actor and actor.is_alive:
                # 传感器Actor（如相机）需要先停止回调，再销毁
                if 'sensor' in actor.type_id:
                    actor.stop()
                actor.destroy()
        except Exception as e:
            print(f"警告：销毁Actor失败 - {e}")
    # 清除剩余的NPC车辆
    clear_npc(world)

def get_vehicle_speed(vehicle):
    """
    计算车辆的速度（单位：km/h）
    参数：
        vehicle: Carla的Vehicle对象
    返回：
        车辆的实时速度（km/h）
    """
    velocity = vehicle.get_velocity()  # 获取车辆的速度矢量（m/s）
    # 计算速度的模长（m/s），转换为km/h（乘以3.6）
    speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed

# ---------------------- 配置参数（集中管理） ----------------------
class Config:
    """
    配置参数类，集中管理所有硬编码参数，便于修改和维护
    所有参数均为静态属性，可直接通过Config.XXX访问
    """
    # Carla客户端配置
    CARLA_HOST = 'localhost'  # Carla服务器地址
    CARLA_PORT = 2000  # Carla服务器端口
    CARLA_TIMEOUT = 10.0  # 连接超时时间（秒）
    # 仿真模式配置
    SYNC_MODE = True  # 是否启用同步模式
    FIXED_DELTA = 0.05  # 同步模式的固定时间步长（秒）
    # NPC车辆配置
    NPC_COUNT = 50  # 要生成的NPC车辆数量
    MAX_DETECTION_DIST = 50.0  # 车辆检测的最大距离（米）
    # 相机配置
    CAMERA_WIDTH = 640  # 相机图像宽度
    CAMERA_HEIGHT = 640  # 相机图像高度
    CAMERA_FOV = 90.0  # 相机视场角（度数）
    CAMERA_POS = carla.Transform(carla.Location(x=1, z=2))  # 相机相对于车辆的位置（前1米，高2米）
    # FPS统计配置
    FPS_WINDOW_SIZE = 10  # 计算FPS的窗口大小（最近10帧）
    # 面板配置
    PANEL_ALPHA = 0.6  # 统计面板的透明度（0-1）
    # 天气配置
    WEATHER_SWITCH_INTERVAL = 5.0  # 天气自动切换的时间间隔（秒）
    ENABLE_WEATHER_SWITCH = True  # 是否启用天气自动切换
    WEATHER_DISPLAY_MAX_LEN = 20  # 天气名称的最大显示长度

# ---------------------- 核心功能封装 ----------------------
class CameraSensor:
    """
    相机传感器类，封装相机的创建、图像采集和销毁逻辑
    负责与Carla的相机传感器交互，提供图像获取接口
    """
    def __init__(self, world, vehicle, config):
        """
        初始化相机传感器
        参数：
            world: Carla的World对象
            vehicle: 要挂载相机的车辆（Ego车辆）
            config: 配置参数对象（Config类实例）
        """
        self.world = world  # 仿真世界
        self.vehicle = vehicle  # 挂载相机的车辆
        self.config = config  # 配置参数
        self.bp = self._create_bp()  # 相机蓝图
        self.actor = None  # 相机Actor对象
        self.queue = queue.Queue(maxsize=1)  # 图像队列（最大长度1，避免缓存过多）
        self._spawn()  # 生成相机Actor

    def _create_bp(self):
        """
        私有方法：创建相机的蓝图（Blueprint）
        返回：
            配置好的相机蓝图对象
        """
        bp_lib = self.world.get_blueprint_library()  # 获取蓝图库
        camera_bp = bp_lib.find('sensor.camera.rgb')  # 查找RGB相机蓝图
        # 设置相机参数
        camera_bp.set_attribute('image_size_x', str(self.config.CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(self.config.CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(self.config.CAMERA_FOV))
        return camera_bp

    def _spawn(self):
        """
        私有方法：生成相机Actor并挂载到车辆上，设置图像回调
        抛出：
            RuntimeError: 相机生成失败时抛出异常
        """
        try:
            # 生成相机Actor，挂载到车辆的指定位置
            self.actor = self.world.spawn_actor(self.bp, self.config.CAMERA_POS, attach_to=self.vehicle)
            # 设置图像回调函数，将图像存入队列
            self.actor.listen(lambda img: self._callback(img))
        except carla.exceptions.CarlaError as e:
            raise RuntimeError(f"相机生成失败：{e}")

    def _callback(self, image):
        """
        私有方法：相机的图像回调函数，处理原始图像并存入队列
        参数：
            image: Carla的Image对象（原始图像数据）
        """
        if not self.queue.full():
            # 将原始图像数据转换为numpy数组（高度×宽度×4（RGBA））
            img_data = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            self.queue.put(img_data)  # 存入队列

    def get_image(self):
        """
        获取相机的最新图像
        若队列空，返回空的numpy数组（避免程序中断）
        返回：
            图像numpy数组（高度×宽度×4（RGBA））
        """
        try:
            return self.queue.get(timeout=0.1)  # 超时时间0.1秒
        except queue.Empty:
            # 返回空图像（黑色背景）
            return np.zeros((self.config.CAMERA_HEIGHT, self.config.CAMERA_WIDTH, 4), dtype=np.uint8)

    def destroy(self):
        """
        销毁相机Actor，释放资源
        """
        if self.actor:
            self.actor.stop()  # 停止图像回调
            self.actor.destroy()  # 销毁Actor

class Vehicle3dBoxRenderer:
    """
    车辆3D边界框渲染类，封装3D边界框的计算和绘制逻辑
    负责将仿真世界中的车辆3D边界框转换为2D图像上的线框并绘制
    """
    def __init__(self, world, ego_vehicle, camera, config):
        """
        初始化3D边界框渲染器
        参数：
            world: Carla的World对象
            ego_vehicle: Ego车辆对象
            camera: 相机传感器对象（CameraSensor类实例）
            config: 配置参数对象（Config类实例）
        """
        self.world = world  # 仿真世界
        self.ego_vehicle = ego_vehicle  # Ego车辆
        self.camera = camera  # 相机传感器
        self.config = config  # 配置参数
        # 3D边界框的边缘对（定义立方体的12条边）
        self.edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
                      [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
        # 构建相机投影矩阵（前向和后向）
        self.K = build_projection_matrix(
            config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FOV
        )
        self.K_b = build_projection_matrix(
            config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FOV, is_behind_camera=True
        )

    def render(self, image):
        """
        在图像上绘制所有符合条件的车辆3D边界框线框
        参数：
            image: 输入图像（numpy数组，BGR格式）
        返回：
            绘制了3D边界框的图像、总车辆数、已渲染的车辆数
        """
        img_h, img_w = self.config.CAMERA_HEIGHT, self.config.CAMERA_WIDTH
        # 获取世界坐标系到相机坐标系的转换矩阵（逆矩阵）
        world_2_camera = np.array(self.camera.actor.get_transform().get_inverse_matrix())
        ego_transform = self.ego_vehicle.get_transform()  # Ego车辆的变换
        ego_forward = ego_transform.get_forward_vector()  # Ego车辆的前向矢量
        camera_transform = self.camera.actor.get_transform()  # 相机的变换
        cam_forward = camera_transform.get_forward_vector()  # 相机的前向矢量

        total_vehicles = 0  # 总车辆数
        rendered_vehicles = 0  # 已渲染的车辆数

        # 遍历仿真世界中的所有车辆
        for npc in self.world.get_actors().filter('*vehicle*'):
            total_vehicles += 1
            # 过滤掉Ego车辆自身
            if npc.id == self.ego_vehicle.id:
                continue

            # 过滤超出检测距离的车辆
            dist = npc.get_transform().location.distance(ego_transform.location)
            if dist > self.config.MAX_DETECTION_DIST:
                continue

            # 过滤Ego车辆后方的车辆（只处理前方的车辆）
            ray = npc.get_transform().location - ego_transform.location
            if ego_forward.dot(ray) <= 0:
                continue

            rendered_vehicles += 1
            bb = npc.bounding_box  # 车辆的3D边界框
            # 获取边界框的8个顶点在世界坐标系中的坐标
            verts = [v for v in bb.get_world_vertices(npc.get_transform())]
            points_2d = []  # 存储顶点的2D图像坐标

            # 遍历每个顶点，转换为2D图像坐标
            for vert in verts:
                ray0 = vert - camera_transform.location
                # 判断顶点是否在相机前方
                if cam_forward.dot(ray0) > 0:
                    p = get_image_point(vert, self.K, world_2_camera)
                else:
                    p = get_image_point(vert, self.K_b, world_2_camera)
                points_2d.append(p)

            # 绘制3D边界框的12条边
            for edge in self.edges:
                p1 = points_2d[edge[0]]
                p2 = points_2d[edge[1]]
                # 判断边的两个顶点是否至少有一个在画布内
                p1_in = point_in_canvas(p1, img_h, img_w)
                p2_in = point_in_canvas(p2, img_h, img_w)
                if p1_in or p2_in:
                    # 绘制线框（蓝色，线宽1）
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
    """
    天气管理类，封装天气的初始化、自动切换和手动切换逻辑
    支持Carla预设天气和自定义雾天天气
    """
    def __init__(self, world, config):
        """
        初始化天气管理器
        参数：
            world: Carla的World对象
            config: 配置参数对象（Config类实例）
        """
        self.world = world  # 仿真世界
        self.config = config  # 配置参数
        self.current_weather_name = ""  # 当前天气名称
        # 天气列表：元组（天气参数对象，天气名称）
        self.weather_list = [
            (carla.WeatherParameters.ClearNoon, "ClearNoon"),  # 晴朗中午
            (carla.WeatherParameters.CloudyNoon, "CloudyNoon"),  # 多云中午
            (carla.WeatherParameters.WetNoon, "WetNoon"),  # 潮湿中午
            (carla.WeatherParameters.MidRainyNoon, "MidRainyNoon"),  # 中雨中午
            (carla.WeatherParameters.HardRainNoon, "HardRainNoon"),  # 大雨中午
            (carla.WeatherParameters.ClearSunset, "ClearSunset"),  # 晴朗日落
            (carla.WeatherParameters.CloudySunset, "CloudySunset"),  # 多云日落
            (carla.WeatherParameters.WetSunset, "WetSunset"),  # 潮湿日落
            (self._create_foggy_weather(), "FoggyNoon")  # 自定义雾天
        ]
        self.current_index = 0  # 当前天气的索引
        self.last_switch_time = time.perf_counter()  # 上一次天气切换的时间
        self._apply_weather(self.current_index)  # 应用初始天气

    def _create_foggy_weather(self):
        """
        私有方法：创建自定义雾天天气参数
        返回：
            配置好的雾天天气参数对象
        """
        weather = carla.WeatherParameters()  # 初始化天气参数
        weather.fog_density = 0.7  # 雾的浓度（0-1）
        weather.fog_distance = 10.0  # 雾的起始距离（米）
        weather.cloudiness = 0.8  # 云量（0-1）
        weather.precipitation = 0.0  # 降水量（0-1，关闭降水）
        return weather

    def _apply_weather(self, index):
        """
        私有方法：应用指定索引的天气
        参数：
            index: 天气列表的索引
        """
        self.current_index = index % len(self.weather_list)  # 循环索引
        weather_params, weather_name = self.weather_list[self.current_index]
        self.world.set_weather(weather_params)  # 设置天气
        self.world.tick()  # 触发仿真步，使天气生效
        self.current_weather_name = weather_name  # 更新当前天气名称
        print(f"\n=== 天气已切换为：{weather_name} ===")  # 打印天气切换信息

    def update(self):
        """
        更新天气状态（自动切换）
        返回：
            当前天气名称
        """
        if not self.config.ENABLE_WEATHER_SWITCH:
            return self.current_weather_name
        current_time = time.perf_counter()
        # 判断是否达到切换时间间隔
        if current_time - self.last_switch_time >= self.config.WEATHER_SWITCH_INTERVAL:
            self.current_index += 1  # 切换到下一个天气
            self._apply_weather(self.current_index)
            self.last_switch_time = current_time  # 更新切换时间
        return self.current_weather_name

    def switch_manually(self):
        """
        手动切换天气（用于键盘触发）
        返回：
            切换后的天气名称
        """
        self.current_index += 1  # 切换到下一个天气
        self._apply_weather(self.current_index)
        self.last_switch_time = time.perf_counter()  # 更新切换时间
        return self.current_weather_name

# ---------------------- 主函数 ----------------------
def main():
    """
    程序主函数，负责整体流程的调度：
    1. 初始化Carla客户端和仿真世界
    2. 生成Ego车辆和NPC车辆
    3. 初始化相机、3D渲染器、天气管理器
    4. 主循环：处理图像、渲染3D框、更新天气、显示统计信息
    5. 资源清理
    """
    config = Config()  # 初始化配置参数

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
    spectator = world.get_spectator()  # 旁观者相机（用于视角控制）
    spawn_points = world.get_map().get_spawn_points()  # 车辆生成点列表
    if not spawn_points:
        print("没有找到生成点")
        return

    # 4. 生成Ego车辆（遍历生成点，确保生成成功）
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')  # 选择林肯MKZ车辆蓝图
    vehicle_bp.set_attribute('role_name', 'ego')  # 标记为Ego车辆
    vehicle = None
    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            break
    if not vehicle:
        print("Ego车辆生成失败")
        return

    # 5. 清除现有NPC车辆
    clear_npc(world)
    clear_static_vehicle(world)

    # 6. 生成NPC车辆（按顺序选择生成点，避免重叠；随机设置颜色）
    car_bp = [bp for bp in bp_lib.filter('vehicle') if int(bp.get_attribute('number_of_wheels')) == 4]
    if not car_bp:
        print("没有找到四轮车辆蓝图")
        return
    spawn_point_idx = 0  # 生成点索引
    npc_count = 0  # 已生成的NPC数量
    # 循环生成，直到达到指定数量或遍历完生成点两次
    while npc_count < config.NPC_COUNT and spawn_point_idx < len(spawn_points) * 2:
        spawn_point = spawn_points[spawn_point_idx % len(spawn_points)]  # 循环使用生成点
        npc_bp = random.choice(car_bp)  # 随机选择车辆蓝图
        # 随机设置车辆颜色（如果蓝图支持颜色属性）
        if npc_bp.has_attribute('color'):
            color = random.choice(npc_bp.get_attribute('color').recommended_values)
            npc_bp.set_attribute('color', color)
        # 尝试生成NPC车辆
        npc = world.try_spawn_actor(npc_bp, spawn_point)
        if npc:
            npc.set_autopilot(True)  # 启用自动驾驶
            npc_count += 1
        spawn_point_idx += 1
    print(f"成功生成 {npc_count} 个NPC车辆")
    time.sleep(0.5)  # 短暂延迟，确保车辆稳定
    vehicle.set_autopilot(True)  # Ego车辆启用自动驾驶

    # 7. 初始化核心组件
    try:
        camera = CameraSensor(world, vehicle, config)  # 相机传感器
        renderer = Vehicle3dBoxRenderer(world, vehicle, camera, config)  # 3D边界框渲染器
        weather_manager = WeatherManager(world, config)  # 天气管理器
    except RuntimeError as e:
        print(e)
        clear(world, vehicle)
        return

    # 8. 初始化FPS统计（双端队列存储最近的帧时间）
    frame_times = deque(maxlen=config.FPS_WINDOW_SIZE)
    last_frame_time = time.perf_counter()

    # 创建显示窗口（提前创建，避免窗口闪烁）
    cv2.namedWindow('3D Ground Truth + Weather + Stats', cv2.WINDOW_NORMAL)

    # 主循环
    try:
        while True:
            if config.SYNC_MODE:
                world.tick()  # 触发仿真步（同步模式必须调用）
            current_time = time.perf_counter()

            # 移动旁观者相机到Ego车辆上方（鸟瞰视角）
            transform = carla.Transform(
                vehicle.get_transform().transform(carla.Location(x=-4, z=50)),
                carla.Rotation(yaw=-180, pitch=-90)
            )
            spectator.set_transform(transform)

            # 获取相机图像（RGBA格式）
            image = camera.get_image()

            # 图像处理：将RGBA转换为BGR格式（OpenCV默认）
            image_bgr = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGBA2BGR)

            # 绘制3D边界框，获取统计数据
            image_bgr, total_veh, rendered_veh = renderer.render(image_bgr)

            # 更新天气状态（自动切换）
            current_weather = weather_manager.update()

            # 计算FPS和车辆速度
            frame_time = current_time - last_frame_time
            frame_times.append(frame_time)
            fps = 1.0 / np.mean(frame_times) if frame_times else 0.0  # 平均FPS
            last_frame_time = current_time
            ego_speed = get_vehicle_speed(vehicle)  # Ego车辆速度

            # ---------------------- 绘制实时统计面板 ----------------------
            # 1. 绘制半透明黑色背景面板
            panel_x, panel_y = 10, 10  # 面板左上角坐标
            panel_w, panel_h = 420, 220  # 面板宽度和高度
            overlay = image_bgr.copy()  # 复制图像用于绘制背景
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
            # 混合背景和原图像，实现半透明效果
            cv2.addWeighted(overlay, config.PANEL_ALPHA, image_bgr, 1 - config.PANEL_ALPHA, 0, image_bgr)

            # 2. 绘制统计文本（白色字体，清晰可见）
            text_y_step = 30  # 文本行间距
            text_pos_y = panel_y + text_y_step  # 初始文本y坐标
            text_style = (cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # 字体样式
            # 优先显示天气信息（避免被遮挡）
            cv2.putText(image_bgr, f"Current Weather: {current_weather}", (panel_x + 15, text_pos_y), *text_style)
            text_pos_y += text_y_step
            # 依次显示其他统计信息
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

            # 键盘事件处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break  # 按q退出循环
            elif key == ord('w'):
                # 按w手动切换天气，并刷新显示
                current_weather = weather_manager.switch_manually()
                cv2.imshow('3D Ground Truth + Weather + Stats', image_bgr)

    except KeyboardInterrupt:
        print("\n用户中断程序")  # 捕获键盘中断（Ctrl+C）
    except Exception as e:
        print(f"程序异常：{e}")  # 捕获其他异常
    finally:
        # 资源清理：销毁相机、车辆，关闭窗口
        camera.destroy()
        clear(world, vehicle)
        cv2.destroyAllWindows()
        print("资源已清理，程序结束")

# 程序入口
if __name__ == '__main__':
    main()

