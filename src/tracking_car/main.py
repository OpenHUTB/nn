"""
main.py - CARLA多目标跟踪系统主程序
入口文件，协调各个模块运行
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np
import carla
import torch
import queue

# 添加当前目录到路径，确保可以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
try:
    import utils
    import sensors
    import tracker
    from loguru import logger
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保以下文件在同一目录下:")
    print("  - utils.py")
    print("  - sensors.py")
    print("  - tracker.py")
    sys.exit(1)


# ======================== 配置管理 ========================

def load_config(config_path=None):
    """
    加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    # 默认配置
    default_config = {
        # CARLA连接
        'host': 'localhost',
        'port': 2000,
        'timeout': 20.0,
        
        # 传感器
        'img_width': 640,
        'img_height': 480,
        'fov': 90,
        'sensor_tick': 0.05,
        'use_lidar': True,
        'lidar_channels': 32,
        'lidar_range': 100.0,
        'lidar_points_per_second': 500000,
        
        # 检测
        'yolo_model': 'yolov8n.pt',
        'conf_thres': 0.5,
        'iou_thres': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'yolo_imgsz_max': 320,
        
        # 跟踪
        'max_age': 5,
        'min_hits': 3,
        'kf_dt': 0.05,
        'max_speed': 50.0,
        
        # 行为分析
        'stop_speed_thresh': 1.0,
        'stop_frames_thresh': 5,
        'overtake_speed_ratio': 1.5,
        'overtake_dist_thresh': 50.0,
        'lane_change_thresh': 0.5,
        'brake_accel_thresh': 2.0,
        'turn_angle_thresh': 15.0,
        'danger_dist_thresh': 10.0,
        'predict_frames': 10,
        'track_history_len': 20,
        
        # 可视化
        'window_width': 1280,
        'window_height': 720,
        'display_fps': 30,
        
        # 天气
        'weather': 'clear',
        'num_npcs': 20,
        
        # 自车
        'ego_vehicle_filter': 'vehicle.tesla.model3',
        'ego_vehicle_color': '255,0,0',
    }
    
    # 如果提供了配置文件，尝试加载
    if config_path and os.path.exists(config_path):
        loaded_config = utils.load_yaml_config(config_path)
        if loaded_config:
            # 合并配置（加载的配置覆盖默认配置）
            for key, value in loaded_config.items():
                if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                    # 递归合并字典
                    default_config[key].update(value)
                else:
                    default_config[key] = value
            logger.info(f"✅ 已加载配置文件: {config_path}")
    
    return default_config


def setup_carla_client(config):
    """
    设置CARLA客户端
    
    Args:
        config: 配置字典
        
    Returns:
        tuple: (client, world) or (None, None)
    """
    try:
        logger.info(f"正在连接CARLA服务器 {config['host']}:{config['port']}...")
        client = carla.Client(config['host'], config['port'])
        client.set_timeout(config['timeout'])
        
        world = client.get_world()
        
        # 设置同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # 设置交通管理器
        try:
            tm = client.get_trafficmanager(8000)
            tm.set_global_distance_to_leading_vehicle(2.0)
            tm.set_respawn_dormant_vehicles(True)
            tm.set_hybrid_physics_mode(True)
            tm.set_hybrid_physics_radius(50.0)
            tm.global_percentage_speed_difference(0)
        except Exception as e:
            logger.warning(f"交通管理器设置失败: {e}")
        
        logger.info("✅ CARLA客户端连接成功")
        return client, world
        
    except Exception as e:
        logger.error(f"❌ 连接CARLA服务器失败: {e}")
        return None, None


def set_weather(world, weather_name):
    """
    设置天气
    
    Args:
        world: CARLA世界对象
        weather_name: 天气名称
    """
    weather_presets = {
        'clear': carla.WeatherParameters.ClearNoon,
        'cloudy': carla.WeatherParameters.CloudyNoon,
        'rain': carla.WeatherParameters.HardRainNoon,
        'fog': carla.WeatherParameters.SoftRainNoon,
        'night': carla.WeatherParameters.ClearNight,
        'wet': carla.WeatherParameters.WetNoon,
        'wet_cloudy': carla.WeatherParameters.WetCloudyNoon,
    }
    
    if weather_name in weather_presets:
        world.set_weather(weather_presets[weather_name])
        logger.info(f"🌤️  天气已设置为: {weather_name}")
    else:
        logger.warning(f"未知天气: {weather_name}, 使用晴天")


# ======================== 可视化（英文版） ========================

class Visualizer:
    """可视化管理器（英文版，解决乱码问题）"""
    
    def __init__(self, config):
        self.config = config
        self.window_name = "CARLA Object Tracking"
        
        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 
                        config.get('window_width', 1280), 
                        config.get('window_height', 720))
        
        # 车辆类别颜色映射
        self.class_colors = {
            'car': (255, 0, 0),      # 蓝色 - 小汽车
            'bus': (0, 255, 0),      # 绿色 - 公交车
            'truck': (0, 0, 255),    # 红色 - 卡车
            'default': (255, 255, 0) # 青色 - 默认
        }
        
        # 行为状态颜色映射（优先级从高到低）
        self.behavior_colors = {
            'dangerous': (0, 0, 255),      # 红色 - 危险
            'stopped': (0, 255, 255),      # 黄色 - 停车
            'overtaking': (255, 0, 255),   # 紫色 - 超车
            'lane_changing': (0, 255, 255), # 青色 - 变道
            'turning': (0, 255, 255),      # 青色 - 转弯
            'accelerating': (255, 0, 0),   # 蓝色 - 加速
            'braking': (0, 165, 255),      # 橙色 - 刹车
            'normal': (0, 255, 0)          # 绿色 - 正常行驶
        }
        
        # 行为状态文本映射（使用英文）
        self.behavior_texts = {
            'dangerous': 'DANGER',
            'stopped': 'STOP',
            'overtaking': 'OVERTAKE',
            'lane_changing': 'LANE CHANGE',
            'turning': 'TURNING',
            'accelerating': 'ACCEL',
            'braking': 'BRAKE',
            'normal': 'NORMAL'
        }
        
        logger.info("✅ 可视化器初始化完成（英文版）")
    
    def _get_behavior_color(self, track_info):
        """
        根据行为状态返回对应颜色
        
        Args:
            track_info: 跟踪目标信息字典
            
        Returns:
            tuple: BGR颜色值
        """
        if not track_info:
            return self.behavior_colors['normal']
        
        # 优先级：危险 > 停车 > 超车 > 变道/转弯 > 加速/刹车 > 正常
        if track_info.get('is_dangerous', False):
            return self.behavior_colors['dangerous']
        elif track_info.get('is_stopped', False):
            return self.behavior_colors['stopped']
        elif track_info.get('is_overtaking', False):
            return self.behavior_colors['overtaking']
        elif track_info.get('is_lane_changing', False):
            return self.behavior_colors['lane_changing']
        elif track_info.get('is_turning', False):
            return self.behavior_colors['turning']
        elif track_info.get('is_accelerating', False):
            return self.behavior_colors['accelerating']
        elif track_info.get('is_braking', False):
            return self.behavior_colors['braking']
        else:
            return self.behavior_colors['normal']
    
    def _get_behavior_text(self, track_info):
        """
        根据行为状态返回对应文本
        
        Args:
            track_info: 跟踪目标信息字典
            
        Returns:
            str: 行为文本
        """
        if not track_info:
            return self.behavior_texts['normal']
        
        # 优先级：危险 > 停车 > 超车 > 变道/转弯 > 加速/刹车 > 正常
        if track_info.get('is_dangerous', False):
            return self.behavior_texts['dangerous']
        elif track_info.get('is_stopped', False):
            return self.behavior_texts['stopped']
        elif track_info.get('is_overtaking', False):
            return self.behavior_texts['overtaking']
        elif track_info.get('is_lane_changing', False):
            return self.behavior_texts['lane_changing']
        elif track_info.get('is_turning', False):
            return self.behavior_texts['turning']
        elif track_info.get('is_accelerating', False):
            return self.behavior_texts['accelerating']
        elif track_info.get('is_braking', False):
            return self.behavior_texts['braking']
        else:
            return self.behavior_texts['normal']
    
    def _get_class_name(self, class_id):
        """
        根据类别ID获取类别名称
        
        Args:
            class_id: 类别ID
            
        Returns:
            str: 类别名称
        """
        class_map = {
            2: 'car',
            5: 'bus',
            7: 'truck',
        }
        return class_map.get(int(class_id), 'default')
    
    def _adjust_color_brightness(self, color, factor):
        """
        调整颜色亮度
        
        Args:
            color: 原始颜色 (B, G, R)
            factor: 亮度因子 (0.0-1.0)
            
        Returns:
            tuple: 调整后的颜色
        """
        return tuple(int(c * factor) for c in color)
    
    def draw_detections(self, image, boxes, ids, classes, tracks_info=None):
        """
        绘制检测和跟踪结果
        
        Args:
            image: 原始图像
            boxes: 边界框数组
            ids: 跟踪ID数组
            classes: 类别数组
            tracks_info: 跟踪详细信息
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        if not utils.valid_img(image):
            return image
        
        result = image.copy()
        
        # 绘制顶部信息栏
        result = self._draw_info_panel(result, len(boxes))
        
        # 绘制边界框和ID
        for i, (bbox, track_id, class_id) in enumerate(zip(boxes, ids, classes)):
            try:
                x1, y1, x2, y2 = map(int, bbox)
                
                # 确保坐标有效
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # 获取当前目标的详细信息
                track_info = None
                if tracks_info and i < len(tracks_info):
                    track_info = tracks_info[i]
                
                # 根据行为状态选择颜色
                behavior_color = self._get_behavior_color(track_info)
                
                # 根据车辆类别选择基础颜色
                class_name = self._get_class_name(class_id)
                class_color = self.class_colors.get(class_name, self.class_colors['default'])
                
                # 融合颜色：70%行为颜色 + 30%类别颜色
                color = tuple(
                    int(behavior_color[j] * 0.7 + class_color[j] * 0.3)
                    for j in range(3)
                )
                
                # 绘制渐变色边框（外深内浅）
                border_width = 3
                for thickness in range(border_width, 0, -1):
                    # 计算当前层的颜色亮度
                    brightness = 0.3 + 0.7 * (thickness / border_width)
                    layer_color = self._adjust_color_brightness(color, brightness)
                    
                    # 绘制边框层
                    offset = border_width - thickness
                    cv2.rectangle(result, 
                                (x1 - offset, y1 - offset), 
                                (x2 + offset, y2 + offset), 
                                layer_color, 
                                1)
                
                # 绘制ID标签背景（使用行为颜色）
                id_text = f"ID:{track_id}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # 标签背景
                label_bg_top = y1 - text_height - 8
                label_bg_bottom = y1
                label_bg_right = x1 + text_width + 8
                
                cv2.rectangle(result, 
                            (x1, label_bg_top),
                            (label_bg_right, label_bg_bottom), 
                            behavior_color, -1)
                
                # 标签边框
                cv2.rectangle(result, 
                            (x1, label_bg_top),
                            (label_bg_right, label_bg_bottom), 
                            (255, 255, 255), 1)
                
                # 绘制ID文本
                cv2.putText(result, id_text, 
                          (x1 + 4, y1 - 4),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 绘制行为状态（如果可用）
                if track_info:
                    # 获取行为文本
                    behavior_text = self._get_behavior_text(track_info)
                    
                    # 在右上角绘制行为状态
                    (text_width, text_height), _ = cv2.getTextSize(
                        behavior_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # 文本位置（右上角）
                    text_x = x2 - text_width - 5
                    text_y = y1 + text_height + 5
                    
                    # 绘制文本背景
                    cv2.rectangle(result,
                                (text_x - 3, text_y - text_height - 3),
                                (text_x + text_width + 3, text_y + 3),
                                behavior_color, -1)
                    
                    # 绘制文本
                    cv2.putText(result, behavior_text,
                              (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # 绘制速度信息（如果可用）
                    if 'speed' in track_info:
                        speed = track_info['speed']
                        speed_text = f"{speed:.1f}m/s"
                        (speed_width, speed_height), _ = cv2.getTextSize(
                            speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                        )
                        
                        # 速度显示在左下角
                        speed_x = x1 + 5
                        speed_y = y2 - 5
                        
                        # 速度背景
                        cv2.rectangle(result,
                                    (speed_x - 2, speed_y - speed_height - 2),
                                    (speed_x + speed_width + 2, speed_y + 2),
                                    (0, 0, 0), -1)
                        
                        # 速度文本
                        cv2.putText(result, speed_text,
                                  (speed_x, speed_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
            except Exception as e:
                logger.debug(f"绘制边界框时出错: {e}")
                continue
        
        return result
    
    def _draw_info_panel(self, image, track_count):
        """绘制信息面板（英文）"""
        h, w = image.shape[:2]
        
        # 信息面板背景（半透明黑色）
        panel_height = 80
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # 标题（英文）
        title = "CARLA Multi-Object Tracking System"
        cv2.putText(image, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 状态信息（英文）
        status_lines = [
            f"Tracking: {track_count} objects",
            f"ESC: Exit | W: Weather | S: Screenshot",
            f"P: Pause | M: Show/Hide Legend"
        ]
        
        # 绘制状态信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(status_lines):
            y_pos = 55 + i * 20
            cv2.putText(image, line, (10, y_pos), 
                       font, 0.5, (255, 255, 255), 1)
        
        return image
    
    def draw_color_legend(self, image):
        """
        绘制颜色说明图例（英文）
        
        Args:
            image: 原始图像
            
        Returns:
            np.ndarray: 添加了图例的图像
        """
        h, w = image.shape[:2]
        
        # 图例背景（右侧半透明）
        legend_width = 200
        legend_height = 300
        legend_x = w - legend_width - 20
        legend_y = 100
        
        overlay = image.copy()
        cv2.rectangle(overlay, 
                     (legend_x, legend_y),
                     (legend_x + legend_width, legend_y + legend_height),
                     (40, 40, 40), -1)
        image = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)
        
        # 图例标题（英文）
        cv2.putText(image, "Color Legend", (legend_x + 10, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 行为状态颜色说明（英文）
        behaviors = [
            ('dangerous', 'Dangerous'),
            ('stopped', 'Stopped'),
            ('overtaking', 'Overtaking'),
            ('lane_changing', 'Lane Change'),
            ('accelerating', 'Accelerating'),
            ('braking', 'Braking'),
            ('normal', 'Normal')
        ]
        
        y_offset = 60
        for behavior_key, behavior_name in behaviors:
            # 颜色方块
            color = self.behavior_colors.get(behavior_key, (255, 255, 255))
            cv2.rectangle(image,
                         (legend_x + 10, legend_y + y_offset),
                         (legend_x + 30, legend_y + y_offset + 15),
                         color, -1)
            
            # 行为名称
            text = behavior_name
            cv2.putText(image, text,
                       (legend_x + 40, legend_y + y_offset + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 25
        
        # 车辆类别说明（英文）
        cv2.putText(image, "Vehicle Types:", (legend_x + 10, legend_y + y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        classes = [
            ('car', 'Car'),
            ('bus', 'Bus'),
            ('truck', 'Truck')
        ]
        
        y_offset += 40
        for class_key, class_name in classes:
            # 颜色方块
            color = self.class_colors.get(class_key, (255, 255, 255))
            cv2.rectangle(image,
                         (legend_x + 10, legend_y + y_offset),
                         (legend_x + 30, legend_y + y_offset + 15),
                         color, -1)
            
            # 类别名称
            text = class_name
            cv2.putText(image, text,
                       (legend_x + 40, legend_y + y_offset + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 25
        
        return image
    
    def show(self, image, wait_key=1):
        """
        显示图像
        
        Args:
            image: 要显示的图像
            wait_key: 等待时间（毫秒）
        """
        if utils.valid_img(image):
            cv2.imshow(self.window_name, image)
            return cv2.waitKey(wait_key)
        return -1
    
    def destroy(self):
        """销毁窗口"""
        cv2.destroyAllWindows()
        logger.info("✅ 可视化窗口已关闭")


# ======================== 主程序 ========================

class CarlaTrackingSystem:
    """CARLA跟踪系统主类"""
    
    def __init__(self, config):
        self.config = config
        self.running = False
        
        # 核心组件
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.sensor_manager = None
        self.detector = None
        self.tracker = None
        self.visualizer = None
        
        # 性能监控
        self.fps_counter = utils.FPSCounter(window_size=15)
        self.perf_monitor = utils.PerformanceMonitor()
        
        # 状态变量
        self.current_weather = config.get('weather', 'clear')
        self.frame_count = 0
        self.show_legend = True  # 是否显示颜色说明
        
        # 检测线程相关
        self.detection_thread = None
        self.image_queue = None
        self.result_queue = None
        
        logger.info("✅ 跟踪系统初始化完成（英文版）")
    
    def initialize(self):
        """初始化系统"""
        try:
            # 1. 连接CARLA
            self.client, self.world = setup_carla_client(self.config)
            if not self.client or not self.world:
                return False
            
            # 等待CARLA世界稳定
            logger.info("等待CARLA世界稳定...")
            for i in range(10):
                self.world.tick()
                time.sleep(0.1)
            
            # 2. 设置天气
            set_weather(self.world, self.current_weather)
            
            # 3. 清理现有的车辆
            logger.info("清理现有车辆...")
            sensors.clear_all_actors(self.world, [])
            time.sleep(1.0)
            
            # 4. 创建自车
            self.ego_vehicle = sensors.create_ego_vehicle(self.world, self.config)
            if not self.ego_vehicle:
                logger.error("❌ 创建自车失败")
                return False
            
            # 等待自车稳定
            time.sleep(0.5)
            
            # 5. 生成NPC车辆
            npc_count = sensors.spawn_npc_vehicles(self.world, self.config)
            logger.info(f"✅ 生成 {npc_count} 个NPC车辆")
            
            # 等待NPC车辆生成
            time.sleep(0.5)
            
            # 6. 初始化传感器
            self.sensor_manager = sensors.SensorManager(self.world, self.ego_vehicle, self.config)
            if not self.sensor_manager.setup():
                logger.error("❌ 传感器初始化失败")
                return False
            
            # 7. 初始化检测器
            self.detector = tracker.YOLODetector(self.config)
            
            # 8. 初始化跟踪器
            self.tracker = tracker.SORTTracker(self.config)
            
            # 9. 初始化可视化器
            self.visualizer = Visualizer(self.config)
            
            # 10. 设置检测线程
            use_async = self.config.get('use_async_detection', True)
            if use_async:
                self._setup_detection_thread()
            
            logger.info("🎉 系统初始化完成，准备开始跟踪")
            return True
            
        except Exception as e:
            logger.error(f"❌ 系统初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_detection_thread(self):
        """设置检测线程"""
        try:
            import queue
            self.image_queue = queue.Queue(maxsize=2)
            self.result_queue = queue.Queue(maxsize=2)
            
            self.detection_thread = tracker.DetectionThread(
                detector=self.detector,
                input_queue=self.image_queue,
                output_queue=self.result_queue,
                maxsize=2
            )
            self.detection_thread.start()
            logger.info("✅ 检测线程已启动")
        except Exception as e:
            logger.warning(f"检测线程设置失败，使用同步模式: {e}")
            self.detection_thread = None
    
    def run(self):
        """运行主循环"""
        import time
        import queue
        
        if not self.initialize():
            logger.error("❌ 系统初始化失败，无法运行")
            return
        
        self.running = True
        logger.info("🚀 开始跟踪...")
        
        try:
            while self.running:
                # 开始帧计时
                self.perf_monitor.start_frame()
                
                # 1. 更新CARLA世界
                self.world.tick()
                
                # 2. 获取传感器数据
                sensor_data = self.sensor_manager.get_sensor_data()
                image = sensor_data.get('image')
                
                if not utils.valid_img(image):
                    logger.warning("获取到无效图像，跳过本帧")
                    time.sleep(0.1)
                    continue
                
                # 3. 执行检测（同步或异步）
                detections = []
                detection_start = time.time()
                
                if self.detection_thread and self.detection_thread.is_alive():
                    # 异步检测
                    if not self.image_queue.full():
                        self.image_queue.put(image.copy())
                    
                    try:
                        processed_image, detections = self.result_queue.get(timeout=0.05)
                        if processed_image is not None:
                            image = processed_image
                    except queue.Empty:
                        # 队列为空，使用上一次的检测结果
                        pass
                else:
                    # 同步检测
                    detections = self.detector.detect(image)
                
                detection_time = time.time() - detection_start
                self.perf_monitor.record_detection_time(detection_time)
                
                # 4. 更新跟踪器
                ego_center = (self.config['img_width'] // 2, self.config['img_height'] // 2)
                
                # 获取LiDAR检测结果（如果可用）
                lidar_detections = sensor_data.get('lidar_objects', [])
                
                tracking_start = time.time()
                boxes, ids, classes = self.tracker.update(
                    detections=detections,
                    ego_center=ego_center,
                    lidar_detections=lidar_detections if lidar_detections else None
                )
                tracking_time = time.time() - tracking_start
                self.perf_monitor.record_tracking_time(tracking_time)
                
                # 5. 获取跟踪详细信息
                tracks_info = self.tracker.get_tracks_info()
                
                # 6. 更新FPS
                fps = self.fps_counter.update()
                
                # 7. 可视化
                result_image = self.visualizer.draw_detections(
                    image=image,
                    boxes=boxes,
                    ids=ids,
                    classes=classes,
                    tracks_info=tracks_info
                )
                
                # 添加颜色说明图例（如果启用）
                if self.show_legend:
                    result_image = self.visualizer.draw_color_legend(result_image)
                
                # 在图像上显示FPS
                if utils.valid_img(result_image):
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(result_image, fps_text, (self.config['img_width'] - 100, 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 8. 显示结果
                key = self.visualizer.show(result_image, wait_key=1)
                
                # 9. 处理键盘输入
                self._handle_keyboard_input(key)
                
                # 10. 帧率控制
                self._control_frame_rate(fps)
                
                # 11. 更新状态
                self.frame_count += 1
                self.perf_monitor.end_frame()
                
                # 12. 定期打印状态
                if self.frame_count % 100 == 0:
                    self._print_status()
                
        except KeyboardInterrupt:
            logger.info("🛑 用户中断程序")
        except Exception as e:
            logger.error(f"❌ 运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def _handle_keyboard_input(self, key):
        """处理键盘输入"""
        # ESC键退出
        if key == 27:  # ESC
            logger.info("🛑 ESC键按下，退出程序")
            self.running = False
        
        # W键切换天气
        elif key == ord('w') or key == ord('W'):
            weather_list = ['clear', 'cloudy', 'rain', 'fog', 'night']
            current_idx = weather_list.index(self.current_weather) if self.current_weather in weather_list else 0
            next_idx = (current_idx + 1) % len(weather_list)
            self.current_weather = weather_list[next_idx]
            set_weather(self.world, self.current_weather)
            logger.info(f"🌤️  天气切换到: {self.current_weather}")
        
        # S键保存截图
        elif key == ord('s') or key == ord('S'):
            self._save_screenshot()
        
        # P键暂停/继续
        elif key == ord('p') or key == ord('P'):
            logger.info("⏸️  程序暂停，按任意键继续...")
            cv2.waitKey(0)
            logger.info("▶️  程序继续")
        
        # M键切换颜色说明显示
        elif key == ord('m') or key == ord('M'):
            self.show_legend = not self.show_legend
            status = "显示" if self.show_legend else "隐藏"
            logger.info(f"🎨 颜色说明图例: {status}")
    
    def _control_frame_rate(self, current_fps):
        """控制帧率"""
        import time
        target_fps = self.config.get('display_fps', 30)
        if target_fps <= 0:
            return
        
        target_interval = 1.0 / target_fps
        
        # 如果帧率过高，适当休眠
        if current_fps > target_fps * 1.2:  # 允许20%的波动
            sleep_time = max(0, target_interval - (1.0 / current_fps))
            time.sleep(sleep_time)
    
    def _save_screenshot(self):
        """保存截图"""
        try:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}_{self.frame_count:06d}.png"
            
            # 获取当前显示的图像
            screenshot = self.sensor_manager.get_camera_image()
            if utils.valid_img(screenshot):
                utils.save_image(screenshot, filename)
                logger.info(f"📸 截图已保存: {filename}")
        except Exception as e:
            logger.warning(f"保存截图失败: {e}")
    
    def _print_status(self):
        """打印系统状态"""
        stats = self.perf_monitor.get_stats()
        tracks_info = self.tracker.get_tracks_info()
        
        # 统计行为类型
        behaviors = {
            'stopped': 0, 
            'overtaking': 0, 
            'lane_changing': 0,
            'turning': 0,
            'accelerating': 0,
            'braking': 0,
            'dangerous': 0,
            'normal': 0
        }
        
        for track in tracks_info:
            if track.get('is_dangerous', False):
                behaviors['dangerous'] += 1
            elif track.get('is_stopped', False):
                behaviors['stopped'] += 1
            elif track.get('is_overtaking', False):
                behaviors['overtaking'] += 1
            elif track.get('is_lane_changing', False):
                behaviors['lane_changing'] += 1
            elif track.get('is_turning', False):
                behaviors['turning'] += 1
            elif track.get('is_accelerating', False):
                behaviors['accelerating'] += 1
            elif track.get('is_braking', False):
                behaviors['braking'] += 1
            else:
                behaviors['normal'] += 1
        
        logger.info(f"📊 状态: 帧数={self.frame_count}, "
                   f"FPS={stats['avg_fps']:.1f}, "
                   f"目标数={len(tracks_info)}, "
                   f"危险={behaviors['dangerous']}, "
                   f"停车={behaviors['stopped']}, "
                   f"超车={behaviors['overtaking']}")
    
    def cleanup(self):
        """清理资源"""
        logger.info("🧹 正在清理资源...")
        
        # 停止检测线程
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.join(timeout=2.0)
        
        # 销毁可视化器
        if self.visualizer:
            self.visualizer.destroy()
        
        # 销毁传感器
        if self.sensor_manager:
            self.sensor_manager.destroy()
        
        # 清理CARLA演员
        if self.world:
            # 排除自车ID（如果存在）
            exclude_ids = [self.ego_vehicle.id] if self.ego_vehicle and self.ego_vehicle.is_alive else []
            sensors.clear_all_actors(self.world, exclude_ids)
        
        # 恢复CARLA设置
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        # 打印最终性能统计
        if self.perf_monitor:
            self.perf_monitor.print_stats()
        
        logger.info("✅ 资源清理完成")


# ======================== 主函数 ========================

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CARLA多目标跟踪系统')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA服务器地址 (默认: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA服务器端口 (默认: 2000)')
    parser.add_argument('--weather', type=str, default='clear',
                       choices=['clear', 'cloudy', 'rain', 'fog', 'night'],
                       help='初始天气 (默认: clear)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO模型路径 (默认: yolov8n.pt)')
    parser.add_argument('--conf-thres', type=float, default=0.5,
                       help='检测置信度阈值 (默认: 0.5)')
    parser.add_argument('--no-lidar', action='store_true',
                       help='禁用LiDAR')
    
    args = parser.parse_args()
    
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, 
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
               level="INFO")
    
    # 记录开始时间
    start_time = time.time()
    logger.info("=" * 50)
    logger.info("🚗 CARLA多目标跟踪系统启动（英文版）")
    logger.info("=" * 50)
    
    try:
        # 1. 加载配置
        config = load_config(args.config)
        
        # 2. 用命令行参数覆盖配置
        if args.host:
            config['host'] = args.host
        if args.port:
            config['port'] = args.port
        if args.weather:
            config['weather'] = args.weather
        if args.model:
            config['yolo_model'] = args.model
        if args.conf_thres:
            config['conf_thres'] = args.conf_thres
        if args.no_lidar:
            config['use_lidar'] = False
        
        # 3. 创建并运行跟踪系统
        system = CarlaTrackingSystem(config)
        system.run()
        
    except Exception as e:
        logger.error(f"❌ 程序运行异常: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 计算运行时间
        run_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info(f"⏱️  程序运行时间: {run_time:.1f}秒")
        logger.info("👋 程序结束")
        logger.info("=" * 50)


if __name__ == "__main__":
    # 检查必要的导入
    try:
        import torch
    except ImportError:
        print("❌ 未找到PyTorch，请安装: pip install torch")
        sys.exit(1)
    
    try:
        import carla
    except ImportError:
        print("❌ 未找到CARLA Python API")
        print("请从CARLA安装目录复制PythonAPI/carla到项目目录")
        sys.exit(1)
    
    # 运行主程序
    main()