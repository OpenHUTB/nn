"""
V2X 路侧智能感知系统
====================
基于 CARLA 0.9.16 + YOLOv8n 预训练模型

功能：
  1. 路侧摄像头部署 — 模拟真实交通监控视角
  2. YOLOv8n 实时目标检测 — 车辆、行人、交通标志识别
  3. V2X 信息面板 — 检测统计、预警消息推送
  4. 多天气切换 — 晴天/多云/雨天/大雾/黄昏/夜晚
  5. 动态交通流 — NPC 车辆 + 行人自动生成

按键操作：
  W — 切换天气场景
  C — 切换摄像头机位（路口/直道/高空鸟瞰）
  S — 保存当前画面截图
  Q / ESC — 退出系统

运行前请先启动 CARLA 服务器 (CarlaUE4.exe)
"""

import carla
import cv2
import numpy as np
import time
import random
import os
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("=" * 50)
    print("错误: 未安装 ultralytics")
    print("请运行: pip install ultralytics")
    print("=" * 50)
    sys.exit(1)


# ==================== 系统配置 ====================
CARLA_HOST = 'localhost'
CARLA_PORT = 2000

# 摄像头参数
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 100
CAMERA_HEIGHT_M = 8.0         # 摄像头架设高度（米）
CAMERA_PITCH = -25.0          # 俯仰角（负值向下倾斜）
CAMERA_SPAWN_INDEX = 0        # 参考路点索引，可修改选择不同路段

# 交通流参数
NPC_VEHICLE_COUNT = 60
NPC_WALKER_COUNT = 30

# 检测参数
YOLO_CONFIDENCE = 0.30
DETECT_INTERVAL = 2          # 每 N 帧检测一次（提升 FPS）

# 多摄像头机位配置 (相对参考路点的偏移)
CAMERA_POSITIONS = [
    {"name": "Intersection", "fwd": 8.0,  "right": 5.0, "z": 8.0,  "pitch": -25.0, "fov": 100},
    {"name": "Straight",     "fwd": 0.0,  "right": 8.0, "z": 6.0,  "pitch": -15.0, "fov": 90},
    {"name": "Bird-Eye",     "fwd": 0.0,  "right": 0.0, "z": 14.0, "pitch": -50.0, "fov": 110},
]

# V2X 关注的 COCO 类别
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
PERSON_CLASSES = {'person'}

# 检测框颜色 (BGR 格式)
COLOR_VEHICLE = (0, 230, 0)     # 绿色 — 车辆
COLOR_PERSON = (0, 0, 230)      # 红色 — 行人
COLOR_OTHER = (230, 165, 0)     # 橙色 — 其他
COLOR_PANEL_BG = (30, 30, 30)   # 面板背景

# 虚拟围栏配置
FENCE_Y_RATIO = 0.58             # 围栏线纵坐标占画面高度的比例（0~1）
FENCE_COLOR   = (0, 215, 255)    # 围栏正常颜色（金黄色，BGR）

# 截图保存目录
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


class V2XRoadsideSystem:
    """V2X 路侧智能感知系统"""

    def __init__(self):
        self.client = None
        self.world = None
        self.original_settings = None
        self.tm = None
        self.camera = None
        self.latest_frame = None
        self.has_new_frame = False
        self.model = None
        self.spawned_vehicles = []
        self.spawned_walkers = []
        self.walker_controllers = []
        self.weather_index = 0
        self.weather_presets = []
        self.fps = 0.0
        self.frame_count = 0
        self.fps_timer = time.time()
        self.stats = {'vehicles': 0, 'pedestrians': 0, 'total': 0}
        self.screenshot_count = 0
        self.start_time = time.time()
        # 多摄像头
        self.cam_index = 0
        self.cam_ref_point = None        # 参考路点
        # 隔帧检测
        self.tick_count = 0
        self.last_detections = None
        # 行人闯入报警
        self.ped_alert_timer = 0.0       # 报警持续到此时间戳
        # 虚拟围栏入侵检测
        self.fence_y = int(CAMERA_HEIGHT * FENCE_Y_RATIO)
        self.intrusion_event_count = 0   # 累计入侵事件次数
        self.prev_intrusion = False       # 上一帧是否存在入侵
        self.intrusion_alert_timer = 0.0  # 入侵警报持续到此时间戳
        self.intrusion_objects = 0        # 当前帧检测到的入侵目标数
        # 检测历史记录（用于底部统计条）
        self.history_vehicles = []
        self.history_pedestrians = []
        self.history_max_len = 120       # 最近 120 帧

    # ==================== 初始化 ====================

    def setup(self):
        """初始化所有组件"""
        self._connect_carla()
        self._init_weather()
        self._load_model()
        self._deploy_camera()
        self._spawn_traffic()
        print("\n" + "=" * 55)
        print("  V2X 路侧智能感知系统已启动")
        print("  按键: W=天气 | C=机位 | S=截图 | Q/ESC=退出")
        print("=" * 55 + "\n")

    def _connect_carla(self):
        """连接 CARLA 服务器并开启同步模式"""
        print("[1/5] 连接 CARLA 服务器...")
        try:
            self.client = carla.Client(CARLA_HOST, CARLA_PORT)
            self.client.set_timeout(15.0)
            self.world = self.client.get_world()
        except RuntimeError as e:
            print(f"错误: 无法连接 CARLA ({e})")
            print("请确保 CarlaUE4.exe 已启动")
            sys.exit(1)

        # 保存原始设置以便退出时恢复
        self.original_settings = self.world.get_settings()

        # 开启同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 30.0
        self.world.apply_settings(settings)

        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)

        map_name = self.world.get_map().name.split('/')[-1]
        print(f"  已连接 | 地图: {map_name}")

    def _init_weather(self):
        """初始化天气预设列表"""
        fog = carla.WeatherParameters(
            cloudiness=90.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=10.0, sun_altitude_angle=45.0,
            fog_density=70.0, fog_distance=10.0, fog_falloff=2.0
        )
        night = carla.WeatherParameters(
            cloudiness=20.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=5.0, sun_altitude_angle=-30.0,
            fog_density=5.0, fog_distance=50.0
        )
        self.weather_presets = [
            ("Clear",  carla.WeatherParameters.ClearNoon),
            ("Cloudy", carla.WeatherParameters.CloudyNoon),
            ("Rainy",  carla.WeatherParameters.WetNoon),
            ("Storm",  carla.WeatherParameters.HardRainNoon),
            ("Foggy",  fog),
            ("Sunset", carla.WeatherParameters.ClearSunset),
            ("Night",  night),
        ]

    def _load_model(self):
        """加载 YOLOv8n 预训练模型（COCO 80类，自动下载）"""
        print("[2/5] 加载 YOLOv8n 检测模型...")
        self.model = YOLO('yolov8n.pt')
        print("  模型加载完成（COCO 80类）")

    def _deploy_camera(self):
        """在路侧高处部署 RGB 监控摄像头（支持多机位切换）"""
        print("[3/5] 部署路侧摄像头...")
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            print("  错误: 地图无可用路点")
            sys.exit(1)

        idx = min(CAMERA_SPAWN_INDEX, len(spawn_points) - 1)
        self.cam_ref_point = spawn_points[idx]
        self._switch_camera_position(first_time=True)

    def _switch_camera_position(self, first_time=False):
        """切换摄像头机位"""
        # 销毁旧摄像头
        if self.camera and self.camera.is_alive:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None

        ref = self.cam_ref_point
        pos = CAMERA_POSITIONS[self.cam_index]

        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
        cam_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
        cam_bp.set_attribute('fov', str(pos['fov']))

        forward = ref.get_forward_vector()
        right = ref.get_right_vector()
        cam_location = carla.Location(
            x=ref.location.x + forward.x * pos['fwd'] + right.x * pos['right'],
            y=ref.location.y + forward.y * pos['fwd'] + right.y * pos['right'],
            z=pos['z']
        )
        cam_transform = carla.Transform(
            cam_location,
            carla.Rotation(pitch=pos['pitch'], yaw=ref.rotation.yaw, roll=0.0)
        )

        self.camera = self.world.spawn_actor(cam_bp, cam_transform)
        self.camera.listen(self._on_image)

        # 同步观察者视角
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(
            carla.Location(x=cam_location.x, y=cam_location.y, z=cam_location.z + 5),
            carla.Rotation(pitch=pos['pitch'] - 10, yaw=ref.rotation.yaw, roll=0.0)
        ))

        if not first_time:
            print(f"  摄像头切换 -> {pos['name']}")
        else:
            print(f"  摄像头已部署 | 机位: {pos['name']} | 高度: {pos['z']}m")

    def _on_image(self, image):
        """摄像头帧回调（CARLA 返回 BGRA 格式，取前3通道为 BGR 供 OpenCV 使用）"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (CAMERA_HEIGHT, CAMERA_WIDTH, 4))
        self.latest_frame = array[:, :, :3].copy()
        self.has_new_frame = True

    def _spawn_traffic(self):
        """生成 NPC 车辆和行人，优先在摄像头附近生成以确保画面丰富"""
        print(f"[4/5] 生成交通流 (车辆: {NPC_VEHICLE_COUNT}, 行人: {NPC_WALKER_COUNT})...")
        bp_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        # 按与摄像头的距离排序，优先在附近生成
        cam_loc = self.cam_ref_point.location
        spawn_points.sort(key=lambda sp: sp.location.distance(cam_loc))

        # —— 生成 NPC 车辆 ——
        vehicle_bps = bp_lib.filter('vehicle.*')
        for i in range(min(NPC_VEHICLE_COUNT, len(spawn_points))):
            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                bp.set_attribute('color', random.choice(
                    bp.get_attribute('color').recommended_values))
            v = self.world.try_spawn_actor(bp, spawn_points[i])
            if v:
                v.set_autopilot(True, self.tm.get_port())
                self.spawned_vehicles.append(v)

        # —— 生成 NPC 行人 ——
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        controller_bp = bp_lib.find('controller.ai.walker')
        for _ in range(NPC_WALKER_COUNT):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            bp = random.choice(walker_bps)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            walker = self.world.try_spawn_actor(bp, carla.Transform(loc))
            if walker:
                self.spawned_walkers.append(walker)
                ctrl = self.world.spawn_actor(controller_bp, carla.Transform(), walker)
                self.walker_controllers.append(ctrl)
                ctrl.start()
                dest = self.world.get_random_location_from_navigation()
                if dest:
                    ctrl.go_to_location(dest)
                ctrl.set_max_speed(random.uniform(1.0, 2.5))

        print(f"  生成完成 | 车辆: {len(self.spawned_vehicles)} | 行人: {len(self.spawned_walkers)}")

        # 让世界运行几帧使 NPC 开始运动
        for _ in range(10):
            self.world.tick()

    # ==================== 检测与绘制 ====================

    def _detect(self, frame):
        """YOLOv8 前向推理（隔帧检测提升帧率）"""
        self.tick_count += 1
        if self.tick_count % DETECT_INTERVAL == 0 or self.last_detections is None:
            results = self.model(frame, verbose=False, conf=YOLO_CONFIDENCE)
            self.last_detections = results[0]
        return self.last_detections

    def _draw_detections(self, frame, detections):
        """在画面上绘制检测框、类别标签、置信度，并检测虚拟围栏入侵"""
        v_count = 0
        p_count = 0
        intrusion_count = 0

        for box in detections.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 按类别选择基础颜色并计数
            if cls_name in VEHICLE_CLASSES:
                base_color = COLOR_VEHICLE
                v_count += 1
            elif cls_name in PERSON_CLASSES:
                base_color = COLOR_PERSON
                p_count += 1
            else:
                base_color = COLOR_OTHER

            # 判断是否越过虚拟围栏（目标底部进入受限区域）
            intruding = (y2 > self.fence_y and
                         cls_name in (VEHICLE_CLASSES | PERSON_CLASSES))
            if intruding:
                intrusion_count += 1
                color = (0, 60, 255)   # 橙红色标记入侵目标
                thickness = 3
                cv2.putText(frame, "!INTRUSION!", (x1, max(y1 - 24, 36)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 255), 2)
            else:
                color = base_color
                thickness = 2

            # 检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 标签（背景 + 文字）
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self.stats = {
            'vehicles': v_count,
            'pedestrians': p_count,
            'total': len(detections.boxes)
        }

        # 更新检测历史
        self.history_vehicles.append(v_count)
        self.history_pedestrians.append(p_count)
        if len(self.history_vehicles) > self.history_max_len:
            self.history_vehicles.pop(0)
            self.history_pedestrians.pop(0)

        # 触发行人闯入报警
        if p_count > 0:
            self.ped_alert_timer = time.time() + 1.5  # 报警持续 1.5 秒

        # 虚拟围栏：上升沿计数（新一波入侵才累加，避免持续存在的目标重复计数）
        self.intrusion_objects = intrusion_count
        if intrusion_count > 0 and not self.prev_intrusion:
            self.intrusion_event_count += 1
        if intrusion_count > 0:
            self.intrusion_alert_timer = time.time() + 2.0
        self.prev_intrusion = intrusion_count > 0

        return frame

    def _draw_virtual_fence(self, frame):
        """绘制虚拟围栏线：虚线效果 + 区域标注，入侵时变橙红色闪烁"""
        fy = self.fence_y
        # 入侵时围栏线变橙红色并闪烁，否则显示金黄色
        if time.time() < self.intrusion_alert_timer:
            color = (0, 60, 255) if int(time.time() * 4) % 2 == 0 else (0, 140, 255)
        else:
            color = FENCE_COLOR

        # 虚线效果：18px 实线 + 10px 间隔
        dash, gap = 18, 10
        x = 0
        while x < CAMERA_WIDTH:
            cv2.line(frame, (x, fy), (min(x + dash, CAMERA_WIDTH), fy), color, 2)
            x += dash + gap

        # 围栏标签（左侧带深色背景）
        label = "[ VIRTUAL FENCE ]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (8, fy - th - 6), (8 + tw + 6, fy + 4), (20, 20, 20), -1)
        cv2.putText(frame, label, (11, fy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 右侧区域标注
        cv2.putText(frame, "SAFE ZONE",
                    (CAMERA_WIDTH - 120, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 220, 80), 1)
        cv2.putText(frame, "RESTRICTED ZONE",
                    (CAMERA_WIDTH - 178, fy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 100, 255), 1)
        return frame

    def _draw_v2x_panel(self, frame):
        """绘制右上角 V2X 路侧单元信息面板"""
        pw, ph = 300, 300
        x0 = CAMERA_WIDTH - pw - 10
        y0 = 40

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (0, 255, 255), 1)

        # 标题
        cv2.putText(frame, "V2X RSU Panel", (x0 + 12, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.line(frame, (x0 + 10, y0 + 38), (x0 + pw - 10, y0 + 38), (0, 255, 255), 1)

        # 检测统计
        y = y0 + 62
        cv2.putText(frame, f"Vehicles:    {self.stats['vehicles']}",
                    (x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_VEHICLE, 2)
        y += 26
        cv2.putText(frame, f"Pedestrians: {self.stats['pedestrians']}",
                    (x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_PERSON, 2)
        y += 26
        cv2.putText(frame, f"Total:       {self.stats['total']}",
                    (x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 26
        i_color = (0, 100, 255) if self.intrusion_objects > 0 else (160, 160, 160)
        cv2.putText(frame, f"Intrusions:  {self.intrusion_event_count} events",
                    (x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, i_color, 1)

        # 摄像头机位
        y += 30
        cam_name = CAMERA_POSITIONS[self.cam_index]['name']
        cv2.putText(frame, f"Camera: {cam_name} [{self.cam_index+1}/{len(CAMERA_POSITIONS)}]",
                    (x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 天气信息
        y += 26
        weather_name = self.weather_presets[self.weather_index][0]
        cv2.putText(frame, f"Weather: {weather_name}",
                    (x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # V2X 预警广播
        y += 28
        cv2.putText(frame, "V2X Broadcast:", (x0 + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        messages = self._get_v2x_messages()
        for msg in messages[:3]:
            y += 22
            cv2.putText(frame, msg, (x0 + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 220, 255), 1)

        return frame

    def _get_v2x_messages(self):
        """根据检测结果和天气生成 V2X 预警广播"""
        msgs = []
        if self.intrusion_objects > 0:
            msgs.append(f"> FENCE BREACH: {self.intrusion_objects} target(s)")
        if self.stats['pedestrians'] > 0:
            msgs.append(f"> {self.stats['pedestrians']} pedestrian(s) ahead")
        if self.stats['vehicles'] > 5:
            msgs.append(f"> Heavy traffic: {self.stats['vehicles']} vehicles")
        elif self.stats['vehicles'] > 0:
            msgs.append(f"> Traffic flow: {self.stats['vehicles']} vehicles")

        weather_name = self.weather_presets[self.weather_index][0]
        if weather_name in ('Rainy', 'Storm'):
            msgs.append("> Wet road, reduce speed")
        elif weather_name == 'Foggy':
            msgs.append("> Low visibility, use fog lights")
        elif weather_name == 'Night':
            msgs.append("> Night driving, headlights on")

        if not msgs:
            msgs.append("> All clear")
        return msgs

    def _draw_header(self, frame):
        """绘制顶部标题栏（模拟监控画面样式）"""
        cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH, 32), COLOR_PANEL_BG, -1)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"V2X RSU-CAM | {ts} | FPS: {self.fps:.0f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

        # 录制指示灯（红点闪烁效果）
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (CAMERA_WIDTH - 20, 16), 6, (0, 0, 255), -1)

        return frame

    def _draw_footer(self, frame):
        """绘制底部操作提示栏"""
        y0 = CAMERA_HEIGHT - 30
        cv2.rectangle(frame, (0, y0), (CAMERA_WIDTH, CAMERA_HEIGHT), COLOR_PANEL_BG, -1)
        cv2.putText(frame, "W: Weather | C: Camera | S: Screenshot | Q: Quit",
                    (8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # 运行时长
        elapsed = time.time() - self.start_time
        m, s = divmod(int(elapsed), 60)
        cv2.putText(frame, f"Uptime: {m:02d}:{s:02d}",
                    (CAMERA_WIDTH - 150, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return frame

    def _draw_pedestrian_alert(self, frame):
        """行人闯入时绘制红色边框闪烁警报"""
        if time.time() < self.ped_alert_timer:
            # 红色边框闪烁
            thickness = 8 if int(time.time() * 4) % 2 == 0 else 4
            cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH - 1, CAMERA_HEIGHT - 1),
                          (0, 0, 255), thickness)
            # 警告文字
            text = "! PEDESTRIAN ALERT !"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            tx = (CAMERA_WIDTH - tw) // 2
            # 半透明红色背景条
            overlay = frame.copy()
            cv2.rectangle(overlay, (tx - 20, 36), (tx + tw + 20, 36 + th + 16), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, text, (tx, 36 + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return frame

    def _draw_intrusion_alert(self, frame):
        """虚拟围栏入侵时绘制橙红色边框与文字警报（与行人红色警报视觉区分）"""
        if time.time() < self.intrusion_alert_timer:
            thick = 6 if int(time.time() * 4) % 2 == 0 else 3
            cv2.rectangle(frame, (2, 2), (CAMERA_WIDTH - 3, CAMERA_HEIGHT - 3),
                          (0, 80, 255), thick)
            text = f"! FENCE INTRUSION  Total: {self.intrusion_event_count} !"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = (CAMERA_WIDTH - tw) // 2
            ty = min(self.fence_y + 45, CAMERA_HEIGHT - 40)
            overlay = frame.copy()
            cv2.rectangle(overlay, (tx - 12, ty - th - 6),
                          (tx + tw + 12, ty + 8), (0, 40, 160), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.putText(frame, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
        return frame

    # ==================== 控制操作 ====================

    def switch_weather(self):
        """切换到下一个天气预设"""
        self.weather_index = (self.weather_index + 1) % len(self.weather_presets)
        name, params = self.weather_presets[self.weather_index]
        self.world.set_weather(params)
        print(f"  天气切换 -> {name}")

    def save_screenshot(self, frame):
        """保存当前画面截图到 results/ 目录"""
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        self.screenshot_count += 1
        fname = os.path.join(
            SCREENSHOT_DIR,
            f"v2x_{time.strftime('%H%M%S')}_{self.screenshot_count:03d}.png"
        )
        cv2.imwrite(fname, frame)
        print(f"  截图已保存: {fname}")

    def _update_fps(self):
        """每秒更新帧率"""
        self.frame_count += 1
        now = time.time()
        if now - self.fps_timer >= 1.0:
            self.fps = self.frame_count / (now - self.fps_timer)
            self.frame_count = 0
            self.fps_timer = now

    # ==================== 主循环 ====================

    def run(self):
        """系统主循环：采集 → 检测 → 绘制 → 显示"""
        window_name = 'V2X Road-Side Perception'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, CAMERA_WIDTH, CAMERA_HEIGHT)

        print("[5/5] 系统运行中...\n")

        try:
            while True:
                self.world.tick()
                self._update_fps()

                if not self.has_new_frame or self.latest_frame is None:
                    continue
                self.has_new_frame = False
                frame = self.latest_frame.copy()

                # 检测 + 绘制
                detections = self._detect(frame)
                frame = self._draw_detections(frame, detections)
                frame = self._draw_virtual_fence(frame)
                frame = self._draw_v2x_panel(frame)
                frame = self._draw_intrusion_alert(frame)
                frame = self._draw_pedestrian_alert(frame)
                frame = self._draw_header(frame)
                frame = self._draw_footer(frame)

                cv2.imshow(window_name, frame)

                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:     # Q 或 ESC
                    break
                elif key == ord('w') or key == ord('W'):
                    self.switch_weather()
                elif key == ord('c') or key == ord('C'):
                    self.cam_index = (self.cam_index + 1) % len(CAMERA_POSITIONS)
                    self._switch_camera_position()
                    self.last_detections = None  # 切换后强制重新检测
                elif key == ord('s') or key == ord('S'):
                    self.save_screenshot(frame)

        except KeyboardInterrupt:
            print("\n用户中断 (Ctrl+C)")

    # ==================== 资源清理 ====================

    def cleanup(self):
        """销毁所有 CARLA Actor，恢复服务器设置"""
        print("\n正在清理资源...")

        # 停止并销毁摄像头
        if self.camera and self.camera.is_alive:
            self.camera.stop()
            self.camera.destroy()

        # 停止并销毁行人控制器
        for ctrl in self.walker_controllers:
            if ctrl.is_alive:
                ctrl.stop()
                ctrl.destroy()

        # 销毁行人
        for w in self.spawned_walkers:
            if w.is_alive:
                w.destroy()

        # 销毁车辆
        for v in self.spawned_vehicles:
            if v.is_alive:
                v.destroy()

        # 恢复 CARLA 原始设置
        if self.world and self.original_settings:
            self.world.apply_settings(self.original_settings)
        if self.tm:
            self.tm.set_synchronous_mode(False)

        cv2.destroyAllWindows()
        print(f"清理完成 | 车辆: {len(self.spawned_vehicles)} | 行人: {len(self.spawned_walkers)}")


# ==================== 程序入口 ====================

def main():
    print("=" * 55)
    print("  V2X 路侧智能感知系统")
    print("  CARLA 0.9.16 + YOLOv8n")
    print("=" * 55 + "\n")

    system = V2XRoadsideSystem()
    try:
        system.setup()
        system.run()
    finally:
        system.cleanup()


if __name__ == '__main__':
    main()
