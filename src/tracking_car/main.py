"""
main.py - CARLA Multi-Object Tracking System
Enhanced version: Color ID encoding + Independent statistics window
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
import psutil

# Add current directory to path to ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
try:
    import utils
    import sensors
    import tracker
    from loguru import logger
except ImportError as e:
    print(f"[ERROR] Import module failed: {e}")
    print("Please ensure the following files are in the same directory:")
    print("  - utils.py")
    print("  - sensors.py")
    print("  - tracker.py")
    sys.exit(1)

# ======================== Configuration Management ========================

def load_config(config_path=None):
    """
    Load configuration with hot reload support
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    import os
    import time
    
    # Default configuration
    default_config = {
        # CARLA connection
        'host': 'localhost',
        'port': 2000,
        'timeout': 20.0,
        
        # Sensors
        'img_width': 640,
        'img_height': 480,
        'fov': 90,
        'sensor_tick': 0.05,
        'use_lidar': True,
        'lidar_channels': 32,
        'lidar_range': 100.0,
        'lidar_points_per_second': 500000,
        
        # Detection
        'yolo_model': 'yolov8n.pt',
        'conf_thres': 0.5,
        'iou_thres': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'yolo_imgsz_max': 320,
        
        # Tracking
        'max_age': 5,
        'min_hits': 3,
        'kf_dt': 0.05,
        'max_speed': 50.0,
        
        # Behavior analysis
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
        
        # Visualization
        'window_width': 1280,
        'window_height': 720,
        'display_fps': 30,
        
        # Weather
        'weather': 'clear',
        'num_npcs': 20,
        
        # Ego vehicle
        'ego_vehicle_filter': 'vehicle.tesla.model3',
        'ego_vehicle_color': '255,0,0',
        
        # ============ 新增：热重载默认配置 ============
        'hot_reload': {
            'enabled': True,
            'check_interval': 100,
            'safe_keys_only': True,
        },
        'hot_reload_safe_keys': [
            'conf_thres', 'iou_thres', 'display_fps',
            'max_age', 'min_hits', 'adaptive_fps',
            'min_fps', 'max_fps', 'fov', 'yolo_imgsz_max',
            'stop_speed_thresh', 'danger_dist_thresh', 'weather'
        ],
        
        # 性能优化
        'adaptive_fps': True,
        'min_fps': 10,
        'max_fps': 60,
        
        # 调试
        'debug_detection': False,
        'debug_tracking': False,
        # =============================================
    }
    
    # 记录提供的配置文件路径
    provided_config_path = config_path
    
    # 如果未提供配置文件路径，尝试查找默认文件
    if not config_path:
        possible_paths = [
            'config.yaml',
            'config_optimized.yaml',
            './config/config.yaml',
            '../config.yaml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                logger.info(f"[CONFIG] 发现配置文件: {path}")
                break
    
    # 如果找到配置文件，尝试加载它
    if config_path and os.path.exists(config_path):
        try:
            loaded_config = utils.load_yaml_config(config_path)
            
            if loaded_config:
                logger.info(f"[CONFIG] 成功加载配置文件: {config_path}")
                
                # 深度合并配置字典
                def deep_merge(target, source):
                    """递归合并字典"""
                    for key, value in source.items():
                        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                            # 递归合并字典
                            deep_merge(target[key], value)
                        else:
                            # 直接赋值
                            target[key] = value
                    return target
                
                # 合并配置
                default_config = deep_merge(default_config, loaded_config)
                
                # ============ 关键修复：确保配置路径正确记录 ============
                # 使用绝对路径，避免相对路径问题
                absolute_path = os.path.abspath(config_path)
                default_config['_config_path'] = absolute_path
                
                # 获取文件修改时间
                try:
                    default_config['_config_mtime'] = os.path.getmtime(absolute_path)
                    logger.info(f"[CONFIG] 配置文件路径: {absolute_path}")
                    logger.info(f"[CONFIG] 最后修改时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(default_config['_config_mtime']))}")
                except Exception as e:
                    logger.warning(f"[CONFIG] 无法获取文件修改时间: {e}")
                    default_config['_config_mtime'] = 0
                # ====================================================
                
                # 记录实际加载的配置文件路径（用于日志）
                default_config['_loaded_config_path'] = config_path
                
            else:
                logger.warning(f"[CONFIG] 配置文件内容为空或无效: {config_path}")
                
        except Exception as e:
            logger.error(f"[CONFIG] 加载配置文件失败 {config_path}: {e}")
            logger.info("[CONFIG] 将使用默认配置")
    
    elif provided_config_path:  # 用户指定了路径但文件不存在
        logger.warning(f"[CONFIG] 指定的配置文件不存在: {provided_config_path}")
        logger.info("[CONFIG] 将使用默认配置")
    
    else:
        logger.info("[CONFIG] 未指定配置文件，使用默认配置")
    
    # ============ 修复：确保热重载配置存在 ============
    # 确保 hot_reload 配置存在
    if 'hot_reload' not in default_config:
        default_config['hot_reload'] = {
            'enabled': True,
            'check_interval': 100,
            'safe_keys_only': True,
        }
        logger.info("[CONFIG] 添加默认热重载配置")
    
    # 确保 hot_reload_safe_keys 存在
    if 'hot_reload_safe_keys' not in default_config:
        default_config['hot_reload_safe_keys'] = [
            'conf_thres', 'iou_thres', 'display_fps',
            'max_age', 'min_hits', 'adaptive_fps',
            'min_fps', 'max_fps', 'fov', 'yolo_imgsz_max',
            'stop_speed_thresh', 'danger_dist_thresh', 'weather'
        ]
    
    # 确保 _config_path 存在（即使没有配置文件）
    if '_config_path' not in default_config:
        # 如果使用默认配置，创建一个虚拟路径
        if provided_config_path and os.path.exists(provided_config_path):
            default_config['_config_path'] = os.path.abspath(provided_config_path)
        else:
            # 使用当前目录的默认路径
            default_config['_config_path'] = os.path.abspath('config.yaml')
            logger.info(f"[CONFIG] 设置默认配置文件路径: {default_config['_config_path']}")
    
    # 确保 _config_mtime 存在
    if '_config_mtime' not in default_config:
        default_config['_config_mtime'] = 0
    
    # 记录最终使用的配置路径
    config_path_display = default_config.get('_config_path', 'Unknown')
    logger.info(f"[CONFIG] 最终配置路径: {config_path_display}")
    logger.info(f"[CONFIG] 热重载状态: {'已启用' if default_config['hot_reload'].get('enabled', True) else '已禁用'}")
    # =================================================
    
    return default_config

def setup_carla_client(config):
    """
    Setup CARLA client
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (client, world) or (None, None)
    """
    try:
        logger.info(f"Connecting to CARLA server {config['host']}:{config['port']}...")
        client = carla.Client(config['host'], config['port'])
        client.set_timeout(config['timeout'])
        
        world = client.get_world()
        
        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Setup traffic manager
        try:
            tm = client.get_trafficmanager(8000)
            tm.set_global_distance_to_leading_vehicle(2.0)
            tm.set_respawn_dormant_vehicles(True)
            tm.set_hybrid_physics_mode(True)
            tm.set_hybrid_physics_radius(50.0)
            tm.global_percentage_speed_difference(0)
        except Exception as e:
            logger.warning(f"Traffic manager setup failed: {e}")
        
        logger.info("[OK] CARLA client connected successfully")
        return client, world
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to connect to CARLA server: {e}")
        return None, None

def set_weather(world, weather_name):
    """
    Set weather
    
    Args:
        world: CARLA world object
        weather_name: Weather name
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
        logger.info(f"[WEATHER] Weather set to: {weather_name}")
    else:
        logger.warning(f"Unknown weather: {weather_name}, using clear weather")

# ======================== Visualization (Enhanced: Independent stats window) ========================

class Visualizer:
    """Visualization manager (Enhanced: Color ID encoding + Independent stats window)"""
    
    def __init__(self, config):
        self.config = config
        self.window_name = "CARLA Object Tracking"
        self.stats_window_name = "Statistics Panel"
        
        # Create main window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 
                        config.get('window_width', 1280), 
                        config.get('window_height', 720))
        
        # Create independent statistics window
        cv2.namedWindow(self.stats_window_name, cv2.WINDOW_NORMAL)
        # Set statistics window size
        stats_width = 600
        stats_height = 800
        cv2.resizeWindow(self.stats_window_name, stats_width, stats_height)
        
        # Move statistics window position (avoid overlapping main window)
        cv2.moveWindow(self.stats_window_name, 
                      config.get('window_width', 1280) + 50,  # Place to the right of main window
                      100)                                    # Vertical position
        
        # Statistics panel state
        self.show_stats_window = True  # Whether to show independent stats window
        self.stats_image = None        # Statistics panel image
        self.stats_update_interval = 2  # Statistics update interval (frames)
        self.stats_frame_counter = 0   # Frame counter
        
        # Vehicle class color mapping
        self.class_colors = {
            'car': (255, 0, 0),      # Blue - Car
            'bus': (0, 255, 0),      # Green - Bus
            'truck': (0, 0, 255),    # Red - Truck
            'default': (255, 255, 0) # Cyan - Default
        }
        
        # Behavior state color mapping (priority from high to low)
        self.behavior_colors = {
            'dangerous': (0, 0, 255),      # Red - Dangerous (too close)
            'stopped': (0, 255, 255),      # Yellow - Stopped
            'overtaking': (255, 0, 255),   # Purple - Overtaking
            'lane_changing': (0, 255, 255), # Cyan - Lane changing
            'turning': (0, 255, 255),      # Cyan - Turning
            'accelerating': (255, 0, 0),   # Blue - Accelerating
            'braking': (0, 165, 255),      # Orange - Braking
            'normal': (0, 255, 0)          # Green - Normal driving
        }
        
        # Behavior state icon mapping (ASCII only)
        self.behavior_icons = {
            'dangerous': '[!]',    # Warning
            'stopped': '[S]',      # Stopped
            'overtaking': '[F]',   # Fast
            'lane_changing': '[<>]', # Lane change
            'turning': '[->]',     # Turn
            'accelerating': '[A]', # Accelerating
            'braking': '[B]',      # Braking
            'normal': '[-]'        # Normal
        }
        
        # Performance data history
        self.fps_history = []
        self.detection_time_history = []
        self.tracking_time_history = []
        self.max_history_length = 100  # Increased history length for more detailed charts
        
        # State history (for trend analysis)
        self.object_count_history = []
        self.cpu_usage_history = []
        self.memory_usage_history = []
        
        # 3D点云可视化相关属性（新增）
        self.show_pointcloud = False  # 是否显示点云窗口
        self.pcd_window_name = "LiDAR Point Cloud"
        self.pcd_vis = None  # Open3D可视化器对象
        self.pcd_geometry_added = False
        self.pcd_update_counter = 0  # 点云更新计数器
        
        logger.info("[OK] Visualizer initialized (Color ID encoding + Independent statistics window + PointCloud)")
    
    def init_pointcloud_visualizer(self):
        """初始化点云可视化器"""
        if not self.config.get('use_lidar', True):
            return False
        
        try:
            import open3d as o3d
            self.pcd_vis = o3d.visualization.Visualizer()
            self.pcd_vis.create_window(
                window_name=self.pcd_window_name,
                width=800,
                height=600,
                left=100,
                top=100
            )
            
            # 设置背景颜色
            opt = self.pcd_vis.get_render_option()
            opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
            opt.point_size = 1.5
            
            self.pcd_geometry_added = False
            logger.info("[POINTCLOUD] 点云可视化器初始化完成")
            return True
            
        except Exception as e:
            logger.warning(f"点云可视化器初始化失败: {e}")
            return False
    
    def update_pointcloud(self, pointcloud_data):
        """
        更新点云显示
        
        Args:
            pointcloud_data: 点云数据 (numpy array)
        """
        if not self.show_pointcloud or not self.pcd_vis or pointcloud_data is None:
            return
        
        try:
            # 每2帧更新一次，避免性能问题
            self.pcd_update_counter += 1
            if self.pcd_update_counter % 2 != 0:
                return
                
            import open3d as o3d
            
            # 创建点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud_data)
            
            # 根据高度着色
            if len(pointcloud_data) > 0:
                z_min = pointcloud_data[:, 2].min()
                z_max = pointcloud_data[:, 2].max()
                z_range = max(z_max - z_min, 1e-6)
                
                colors = np.zeros((len(pointcloud_data), 3))
                normalized_z = (pointcloud_data[:, 2] - z_min) / z_range
                colors[:, 0] = normalized_z  # 红色通道（高处）
                colors[:, 2] = 1 - normalized_z  # 蓝色通道（低处）
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 更新或添加几何体
            if not self.pcd_geometry_added:
                self.pcd_vis.add_geometry(pcd)
                self.pcd_geometry_added = True
            else:
                self.pcd_vis.clear_geometries()
                self.pcd_vis.add_geometry(pcd)
            
            # 更新可视化
            self.pcd_vis.poll_events()
            self.pcd_vis.update_renderer()
            
        except Exception as e:
            logger.warning(f"更新点云失败: {e}")
    
    def toggle_pointcloud_display(self):
        """切换点云显示"""
        if not self.config.get('use_lidar', True):
            logger.warning("LiDAR功能已禁用，无法显示点云")
            return
        
        self.show_pointcloud = not self.show_pointcloud
        
        if self.show_pointcloud:
            # 初始化点云可视化器
            if not self.pcd_vis:
                if not self.init_pointcloud_visualizer():
                    self.show_pointcloud = False
                    return
            logger.info("[POINTCLOUD] 点云显示已开启")
        else:
            # 关闭点云窗口
            if self.pcd_vis:
                try:
                    self.pcd_vis.destroy_window()
                except:
                    pass
                self.pcd_vis = None
                self.pcd_geometry_added = False
            logger.info("[POINTCLOUD] 点云显示已关闭")
    
    def _get_behavior_color(self, track_info):
        """
        Get color based on behavior state
        
        Args:
            track_info: Tracking target information dictionary
            
        Returns:
            tuple: BGR color value
        """
        if not track_info:
            return self.behavior_colors['normal']
        
        # Priority: dangerous > stopped > overtaking > lane changing/turning > accelerating/braking > normal
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
    
    def _get_behavior_icon(self, track_info):
        """
        Get icon based on behavior state
        
        Args:
            track_info: Tracking target information dictionary
            
        Returns:
            str: Behavior icon
        """
        if not track_info:
            return self.behavior_icons['normal']
        
        # Priority: dangerous > stopped > overtaking > lane changing/turning > accelerating/braking > normal
        if track_info.get('is_dangerous', False):
            return self.behavior_icons['dangerous']
        elif track_info.get('is_stopped', False):
            return self.behavior_icons['stopped']
        elif track_info.get('is_overtaking', False):
            return self.behavior_icons['overtaking']
        elif track_info.get('is_lane_changing', False):
            return self.behavior_icons['lane_changing']
        elif track_info.get('is_turning', False):
            return self.behavior_icons['turning']
        elif track_info.get('is_accelerating', False):
            return self.behavior_icons['accelerating']
        elif track_info.get('is_braking', False):
            return self.behavior_icons['braking']
        else:
            return self.behavior_icons['normal']
    
    def _get_class_name(self, class_id):
        """
        Get class name based on class ID
        
        Args:
            class_id: Class ID
            
        Returns:
            str: Class name
        """
        class_map = {
            2: 'car',
            5: 'bus',
            7: 'truck',
        }
        return class_map.get(int(class_id), 'default')
    
    def _adjust_color_brightness(self, color, factor):
        """
        Adjust color brightness
        
        Args:
            color: Original color (B, G, R)
            factor: Brightness factor (0.0-1.0)
            
        Returns:
            tuple: Adjusted color
        """
        return tuple(int(c * factor) for c in color)
    
    def update_performance_data(self, fps, detection_time, tracking_time, stats_data=None):
        """
        Update performance data (enhanced, supports more data)
        
        Args:
            fps: Current frame rate
            detection_time: Detection time (seconds)
            tracking_time: Tracking time (seconds)
            stats_data: Statistics data dictionary
        """
        self.fps_history.append(fps)
        self.detection_time_history.append(detection_time * 1000)  # Convert to milliseconds
        self.tracking_time_history.append(tracking_time * 1000)    # Convert to milliseconds
        
        # If there is statistics data, also update state history
        if stats_data:
            self.object_count_history.append(stats_data.get('total_objects', 0))
            self.cpu_usage_history.append(stats_data.get('cpu_usage', 0))
            self.memory_usage_history.append(stats_data.get('memory_usage', 0))
        
        # Keep history data length
        for history_list in [
            self.fps_history,
            self.detection_time_history,
            self.tracking_time_history,
            self.object_count_history,
            self.cpu_usage_history,
            self.memory_usage_history
        ]:
            if len(history_list) > self.max_history_length:
                history_list.pop(0)
    
    def create_stats_window_image(self, stats_data):
        """
        Create image for independent statistics window
        
        Args:
            stats_data: Statistics data dictionary
            
        Returns:
            np.ndarray: Statistics panel image
        """
        # Create statistics panel image (light gray background)
        stats_width = 600
        stats_height = 800
        stats_image = np.ones((stats_height, stats_width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # 1. Title area
        title_height = 80
        cv2.rectangle(stats_image, (0, 0), (stats_width, title_height), (50, 50, 80), -1)
        
        title = "CARLA Statistics Panel"
        cv2.putText(stats_image, title, 
                   (stats_width // 2 - 150, title_height // 2 + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        subtitle = "Press T to toggle display"
        cv2.putText(stats_image, subtitle,
                   (stats_width // 2 - 140, title_height // 2 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_offset = title_height + 20
        
        # 2. System status section
        y_offset = self._draw_stats_section(stats_image, "System Status", y_offset, stats_data, self._draw_system_stats)
        
        # 3. Object statistics section
        y_offset = self._draw_stats_section(stats_image, "Object Statistics", y_offset, stats_data, self._draw_object_stats)
        
        # 4. Performance charts section
        y_offset = self._draw_stats_section(stats_image, "Performance Charts", y_offset, stats_data, self._draw_performance_charts)
        
        # 5. History trends section
        if len(self.fps_history) > 5:
            y_offset = self._draw_stats_section(stats_image, "History Trends", y_offset, stats_data, self._draw_trend_charts)
        
        # 6. Bottom information
        bottom_y = stats_height - 30
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(stats_image, f"Update time: {timestamp}", 
                   (20, bottom_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        frame_info = f"Total frames: {stats_data.get('total_frames', 0)}"
        cv2.putText(stats_image, frame_info,
                   (stats_width - 150, bottom_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        return stats_image
    
    def _draw_stats_section(self, image, title, y_start, stats_data, draw_function):
        """
        General template for drawing statistics section
        
        Returns:
            int: Starting Y coordinate of next section
        """
        section_height = 200  # Default height for each section
        
        # Section background
        cv2.rectangle(image, (10, y_start), (590, y_start + section_height), (255, 255, 255), -1)
        cv2.rectangle(image, (10, y_start), (590, y_start + section_height), (220, 220, 220), 2)
        
        # Section title
        cv2.putText(image, title, (20, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        
        # Draw dividing line
        cv2.line(image, (20, y_start + 35), (580, y_start + 35), (200, 200, 200), 1)
        
        # Call specific drawing function
        content_y = y_start + 50
        content_y = draw_function(image, content_y, stats_data)
        
        # If drawing function returned new Y coordinate, use it; otherwise use default height
        if content_y > y_start + section_height:
            section_height = content_y - y_start
        
        return y_start + section_height + 20
    
    def _draw_system_stats(self, image, y_start, stats_data):
        """
        Draw system status information
        """
        x_left = 30
        x_right = 300
        y = y_start
        
        # Define status items
        status_items = [
            ("FPS", f"{stats_data.get('fps', 0):.1f}", 
             (0, 255, 0) if stats_data.get('fps', 0) > 20 else (0, 165, 255)),
            ("Runtime", f"{stats_data.get('run_time', 0):.0f}s", (100, 100, 100)),
            ("CPU Usage", f"{stats_data.get('cpu_usage', 0):.1f}%",
             (0, 255, 0) if stats_data.get('cpu_usage', 0) < 70 else (0, 165, 255) if stats_data.get('cpu_usage', 0) < 90 else (0, 0, 255)),
            ("Memory Usage", f"{stats_data.get('memory_usage', 0):.1f}%",
             (0, 255, 0) if stats_data.get('memory_usage', 0) < 70 else (0, 165, 255) if stats_data.get('memory_usage', 0) < 90 else (0, 0, 255)),
            ("Detection Thread", stats_data.get('detection_thread', 'Unknown'),
             (0, 255, 0) if stats_data.get('detection_thread') == 'Running' else (0, 0, 255)),
            ("Avg Frame Time", f"{stats_data.get('avg_frame_time', 0):.1f}ms",
             (0, 255, 0) if stats_data.get('avg_frame_time', 0) < 33 else (0, 165, 255) if stats_data.get('avg_frame_time', 0) < 50 else (0, 0, 255)),
        ]
        
        # Draw in two columns
        for i, (label, value, color) in enumerate(status_items):
            x = x_left if i % 2 == 0 else x_right
            current_y = y + (i // 2) * 30
            
            # Label
            cv2.putText(image, f"{label}:", (x, current_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
            
            # Value
            cv2.putText(image, value, (x + 120, current_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return y + (len(status_items) // 2 + 1) * 30
    
    def _draw_object_stats(self, image, y_start, stats_data):
        """
        Draw object statistics information
        """
        y = y_start
        
        # Total objects
        total_objects = stats_data.get('total_objects', 0)
        cv2.putText(image, f"Total Objects: {total_objects}", (30, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        y += 30
        
        # Vehicle type distribution (horizontal bar chart)
        vehicle_counts = stats_data.get('vehicle_counts', {})
        if vehicle_counts:
            cv2.putText(image, "Vehicle Type Distribution:", (30, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            y += 25
            
            max_count = max(vehicle_counts.values()) if vehicle_counts.values() else 1
            bar_width = 200
            
            types = ['car', 'bus', 'truck']
            type_names = {'car': 'Car', 'bus': 'Bus', 'truck': 'Truck'}
            
            for i, v_type in enumerate(types):
                count = vehicle_counts.get(v_type, 0)
                # Bar chart
                bar_length = int((count / max_count) * bar_width) if max_count > 0 else 0
                color = self.class_colors.get(v_type, (100, 100, 100))
                
                cv2.rectangle(image, (150, y - 10), (150 + bar_length, y + 5), color, -1)
                
                # Text
                cv2.putText(image, type_names[v_type], (30, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
                cv2.putText(image, f"{count}", (370, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
                
                y += 25
            y += 10
        
        # Behavior distribution
        behavior_counts = stats_data.get('behavior_counts', {})
        if behavior_counts:
            cv2.putText(image, "Behavior Distribution:", (30, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            y += 25
            
            # Only show non-zero behaviors
            displayed_behaviors = 0
            for behavior, count in behavior_counts.items():
                if count > 0 and behavior in self.behavior_colors:
                    color = self.behavior_colors[behavior]
                    icon = self.behavior_icons.get(behavior, '•')
                    
                    cv2.putText(image, f"{icon} {behavior}: {count}", (50, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y += 20
                    displayed_behaviors += 1
            
            y += 10 if displayed_behaviors > 0 else 0
        
        return y
    
    def _draw_performance_charts(self, image, y_start, stats_data):
        """
        Draw performance charts
        """
        chart_x = 30
        chart_y = y_start
        chart_width = 540
        chart_height = 120
        
        # Chart background
        cv2.rectangle(image, (chart_x, chart_y), 
                     (chart_x + chart_width, chart_y + chart_height), 
                     (250, 250, 250), -1)
        cv2.rectangle(image, (chart_x, chart_y), 
                     (chart_x + chart_width, chart_y + chart_height), 
                     (200, 200, 200), 1)
        
        if len(self.fps_history) > 1:
            # Draw FPS curve (green)
            self._draw_chart_curve(image, chart_x, chart_y, chart_width, chart_height,
                                 self.fps_history, (0, 180, 0), "FPS", 60)
            
            # Draw detection time curve (red)
            if self.detection_time_history:
                self._draw_chart_curve(image, chart_x, chart_y, chart_width, chart_height,
                                     self.detection_time_history, (200, 0, 0), "Detect(ms)", 100)
            
            # Draw tracking time curve (blue)
            if self.tracking_time_history:
                self._draw_chart_curve(image, chart_x, chart_y, chart_width, chart_height,
                                     self.tracking_time_history, (0, 0, 200), "Track(ms)", 50)
        
        # Chart title
        cv2.putText(image, "Real-time Performance Trend (Last 100 frames)", 
                   (chart_x + 10, chart_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        
        return chart_y + chart_height + 20
    
    def _draw_trend_charts(self, image, y_start, stats_data):
        """
        Draw history trend charts
        """
        chart_x = 30
        chart_y = y_start
        chart_width = 540
        chart_height = 100
        
        # Object count trend
        if len(self.object_count_history) > 1:
            cv2.putText(image, "Object Count Trend:", (chart_x, chart_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            # Chart background
            cv2.rectangle(image, (chart_x, chart_y), 
                         (chart_x + chart_width, chart_y + chart_height), 
                         (250, 250, 250), -1)
            cv2.rectangle(image, (chart_x, chart_y), 
                         (chart_x + chart_width, chart_y + chart_height), 
                         (200, 200, 200), 1)
            
            # Draw object count curve
            self._draw_chart_curve(image, chart_x, chart_y, chart_width, chart_height,
                                 self.object_count_history, (100, 0, 200), "Objects", 
                                 max(self.object_count_history) if self.object_count_history else 20)
            
            chart_y += chart_height + 30
        
        # System resource trend
        cv2.putText(image, "System Resource Trend:", (chart_x, chart_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Chart background
        cv2.rectangle(image, (chart_x, chart_y), 
                     (chart_x + chart_width, chart_y + chart_height), 
                     (250, 250, 250), -1)
        cv2.rectangle(image, (chart_x, chart_y), 
                     (chart_x + chart_width, chart_y + chart_height), 
                     (200, 200, 200), 1)
        
        # Draw CPU and memory curves
        if len(self.cpu_usage_history) > 1:
            self._draw_chart_curve(image, chart_x, chart_y, chart_width, chart_height,
                                 self.cpu_usage_history, (200, 100, 0), "CPU%", 100)
        
        if len(self.memory_usage_history) > 1:
            self._draw_chart_curve(image, chart_x, chart_y, chart_width, chart_height,
                                 self.memory_usage_history, (0, 100, 200), "Memory%", 100)
        
        return chart_y + chart_height + 20
    
    def _draw_chart_curve(self, image, x, y, width, height, data, color, label, max_value):
        """
        Draw chart curve (enhanced, with label)
        """
        if len(data) < 2:
            return
        
        points = []
        data_len = len(data)
        
        for i, value in enumerate(data):
            # Normalize to 0-1 range
            normalized = min(1.0, value / max_value) if max_value > 0 else 0
            
            # Calculate coordinates
            point_x = int(x + (i / (data_len - 1)) * width) if data_len > 1 else x
            point_y = int(y + height - normalized * height)
            
            points.append((point_x, point_y))
        
        # Draw curve
        for i in range(1, len(points)):
            cv2.line(image, points[i-1], points[i], color, 2)
        
        # Draw label
        label_x = x + width - 80
        label_y = y + 15
        
        # Color marker
        cv2.circle(image, (label_x - 10, label_y), 4, color, -1)
        cv2.putText(image, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
    
    def draw_detections(self, image, boxes, ids, classes, tracks_info=None):
        """
        Draw detection and tracking results
        
        Args:
            image: Original image
            boxes: Bounding box array
            ids: Tracking ID array
            classes: Class array
            tracks_info: Tracking detailed information
            
        Returns:
            np.ndarray: Drawn image
        """
        if not utils.valid_img(image):
            return image
        
        result = image.copy()
        
        # Draw top information panel
        result = self._draw_info_panel(result, len(boxes))
        
        # Draw bounding boxes and IDs
        for i, (bbox, track_id, class_id) in enumerate(zip(boxes, ids, classes)):
            try:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Ensure coordinates are valid
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Get current target's detailed information
                track_info = None
                if tracks_info and i < len(tracks_info):
                    track_info = tracks_info[i]
                
                # Select color based on behavior state
                behavior_color = self._get_behavior_color(track_info)
                
                # Select base color based on vehicle class
                class_name = self._get_class_name(class_id)
                class_color = self.class_colors.get(class_name, self.class_colors['default'])
                
                # Blend colors: 70% behavior color + 30% class color
                color = tuple(
                    int(behavior_color[j] * 0.7 + class_color[j] * 0.3)
                    for j in range(3)
                )
                
                # Draw gradient border (dark outside, light inside)
                border_width = 3
                for thickness in range(border_width, 0, -1):
                    # Calculate current layer's color brightness
                    brightness = 0.3 + 0.7 * (thickness / border_width)
                    layer_color = self._adjust_color_brightness(color, brightness)
                    
                    # Draw border layer
                    offset = border_width - thickness
                    cv2.rectangle(result, 
                                (x1 - offset, y1 - offset), 
                                (x2 + offset, y2 + offset), 
                                layer_color, 
                                1)
                
                # Draw ID label background (use behavior color)
                id_text = f"ID:{track_id}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Label background
                label_bg_top = y1 - text_height - 8
                label_bg_bottom = y1
                label_bg_right = x1 + text_width + 8
                
                cv2.rectangle(result, 
                            (x1, label_bg_top),
                            (label_bg_right, label_bg_bottom), 
                            behavior_color, -1)
                
                # Label border
                cv2.rectangle(result, 
                            (x1, label_bg_top),
                            (label_bg_right, label_bg_bottom), 
                            (255, 255, 255), 1)
                
                # Draw ID text
                cv2.putText(result, id_text, 
                          (x1 + 4, y1 - 4),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw behavior icon (if available)
                if track_info:
                    # Get behavior icon
                    behavior_icon = self._get_behavior_icon(track_info)
                    
                    # Draw behavior status at top-right corner
                    behavior_text = behavior_icon
                    (icon_width, icon_height), _ = cv2.getTextSize(
                        behavior_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Icon position (top-right corner)
                    icon_x = x2 - icon_width - 5
                    icon_y = y1 + icon_height + 5
                    
                    # Draw icon background
                    cv2.rectangle(result,
                                (icon_x - 3, icon_y - icon_height - 3),
                                (icon_x + icon_width + 3, icon_y + 3),
                                behavior_color, -1)
                    
                    # Draw icon
                    cv2.putText(result, behavior_text,
                              (icon_x, icon_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw speed information (if available)
                    if 'speed' in track_info:
                        speed = track_info['speed']
                        speed_text = f"{speed:.1f}m/s"
                        (speed_width, speed_height), _ = cv2.getTextSize(
                            speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                        )
                        
                        # Speed display at bottom-left corner
                        speed_x = x1 + 5
                        speed_y = y2 - 5
                        
                        # Speed background
                        cv2.rectangle(result,
                                    (speed_x - 2, speed_y - speed_height - 2),
                                    (speed_x + speed_width + 2, speed_y + 2),
                                    (0, 0, 0), -1)
                        
                        # Speed text
                        cv2.putText(result, speed_text,
                                  (speed_x, speed_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
            except Exception as e:
                logger.debug(f"Error drawing bounding box: {e}")
                continue
        
        return result
    
    def _draw_info_panel(self, image, track_count):
        """Draw information panel"""
        h, w = image.shape[:2]
        
        # Information panel background (semi-transparent black)
        panel_height = 100
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # Title
        title = "CARLA Multi-Object Tracking System"
        cv2.putText(image, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Status information
        status_lines = [
            f"Tracked Objects: {track_count}",
            f"ESC: Exit | W: Weather | S: Screenshot",
            f"P: Pause | T: Stats Window | M: Color Legend",
            f"L: PointCloud | V: View Mode"  # 添加点云提示
        ]
        
        status_lines = [
            f"Tracked Objects: {track_count}",
            f"ESC: Exit | W: Weather | S: Screenshot",
            f"P: Pause | T: Stats | M: Color Legend",
            f"L: PointCloud | V: View | O: SignDetect",  # 修改这一行
        ]

        # Draw status information
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(status_lines):
            y_pos = 55 + i * 20
            cv2.putText(image, line, (10, y_pos), 
                       font, 0.5, (255, 255, 255), 1)
        
        return image
    
    def draw_color_legend(self, image):
        """
        Draw color legend
        
        Args:
            image: Original image
            
        Returns:
            np.ndarray: Image with legend added
        """
        h, w = image.shape[:2]
        
        # Legend background (right side semi-transparent)
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
        
        # Legend title
        cv2.putText(image, "Color Legend", (legend_x + 10, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Behavior state color explanation
        behaviors = [
            ('dangerous', 'Danger', '[!]'),
            ('stopped', 'Stopped', '[S]'),
            ('overtaking', 'Overtaking', '[F]'),
            ('lane_changing', 'Lane Change', '[<>]'),
            ('accelerating', 'Accelerating', '[A]'),
            ('braking', 'Braking', '[B]'),
            ('normal', 'Normal', '[-]')
        ]
        
        y_offset = 60
        for behavior_key, behavior_name, icon in behaviors:
            # Color block
            color = self.behavior_colors.get(behavior_key, (255, 255, 255))
            cv2.rectangle(image,
                         (legend_x + 10, legend_y + y_offset),
                         (legend_x + 30, legend_y + y_offset + 15),
                         color, -1)
            
            # Behavior name
            text = f"{icon} {behavior_name}"
            cv2.putText(image, text,
                       (legend_x + 40, legend_y + y_offset + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 25
        
        # Vehicle class explanation
        cv2.putText(image, "Vehicle Types:", (legend_x + 10, legend_y + y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        classes = [
            ('car', 'Car', ''),
            ('bus', 'Bus', ''),
            ('truck', 'Truck', '')
        ]
        
        y_offset += 40
        for class_key, class_name, icon in classes:
            # Color block
            color = self.class_colors.get(class_key, (255, 255, 255))
            cv2.rectangle(image,
                         (legend_x + 10, legend_y + y_offset),
                         (legend_x + 30, legend_y + y_offset + 15),
                         color, -1)
            
            # Class name
            text = class_name
            cv2.putText(image, text,
                       (legend_x + 40, legend_y + y_offset + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 25
        
        return image
    
    def _draw_info_panel(self, image, track_count):
        """绘制信息面板（添加热重载状态）"""
        h, w = image.shape[:2]
        
        # 信息面板背景（半透明黑色）
        panel_height = 120  # 稍微增加高度
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # 标题
        title = "CARLA Tracking System"
        cv2.putText(image, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
        status_lines = [
            f"Objects: {track_count}",
            f"ESC:Exit | W:Weather | S:Screenshot",
            f"P:Pause | T:Stats | M:Legend",
            f"L:PointCloud | V:View | R:Reload",
        ]
        
        # 绘制状态信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(status_lines):
            y_pos = 55 + i * 20
            cv2.putText(image, line, (10, y_pos), 
                       font, 0.5, (255, 255, 255), 1)
        
        return image

    def show(self, image, stats_data=None):
        """
        显示图像和统计窗口（优化防闪烁）
        """
        # 双缓冲：创建临时图像副本
        if utils.valid_img(image):
            display_image = image.copy()
        
            # 添加FPS显示到图像副本（而不是原图）
            fps_text = f"FPS: {stats_data.get('fps', 0):.1f}" if stats_data else "FPS: N/A"
            cv2.putText(display_image, fps_text, 
                       (self.config.get('img_width', 640) - 100, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
            # 显示主窗口
            cv2.imshow(self.window_name, display_image)
    
        # 优化统计窗口更新频率（降低到每5帧更新一次）
        if self.show_stats_window and stats_data is not None:
            self.stats_frame_counter += 1
        
            # 每5帧更新一次统计窗口（原来是每2帧）
            if self.stats_frame_counter >= 10:
                try:
                    self.stats_image = self.create_stats_window_image(stats_data)
                    if self.stats_image is not None:
                        # 使用双缓冲显示统计窗口
                        cv2.imshow(self.stats_window_name, self.stats_image)
                    self.stats_frame_counter = 0
                except Exception as e:
                    logger.debug(f"Stats window update failed: {e}")
    
        # 等待键（稍微增加等待时间，减少CPU占用）
        return cv2.waitKey(1) & 0xFF  # 添加 & 0xFF 确保跨平台兼容性
        
    def destroy(self):
        """Destroy all windows"""
        # 销毁点云可视化器
        if self.pcd_vis:
            try:
                self.pcd_vis.destroy_window()
            except:
                pass
        
        # 销毁其他窗口
        cv2.destroyAllWindows()
        logger.info("[OK] All visualization windows closed")

# ======================== Main Program ========================

class CarlaTrackingSystem:
    """CARLA tracking system main class"""
    
    def __init__(self, config):
        self.config = config
        self.running = False
        
        # Core components
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.sensor_manager = None
        self.detector = None
        self.tracker = None
        self.visualizer = None
        
        # Performance monitoring
        self.fps_counter = utils.FPSCounter(window_size=15)
        self.perf_monitor = utils.PerformanceMonitor()
        
        # State variables
        self.current_weather = config.get('weather', 'clear')
        self.frame_count = 0
        self.show_legend = True  # Whether to show color legend
        self.start_time = time.time()  # Program start time
        
        # 新增：视角控制
        self.current_view_mode = 'satellite'  # 默认卫星视角

        # Detection thread related
        self.detection_thread = None
        self.image_queue = None
        self.result_queue = None
        
         # 添加交通标志检测器
        self.traffic_sign_detector = None
        self.enable_sign_detection = config.get('enable_sign_detection', False)

        logger.info("[OK] Tracking system initialized (Color ID encoding + Independent statistics window)")
    
    def initialize(self):
        """Initialize system"""
        try:
            # 1. Connect to CARLA
            self.client, self.world = setup_carla_client(self.config)
            if not self.client or not self.world:
                return False
            
            # Wait for CARLA world to stabilize
            logger.info("Waiting for CARLA world to stabilize...")
            for i in range(10):
                self.world.tick()
                time.sleep(0.1)
            
            # 2. Set weather
            set_weather(self.world, self.current_weather)
            
            # 3. Clean up existing vehicles
            logger.info("Clearing existing vehicles...")
            sensors.clear_all_actors(self.world, [])
            time.sleep(1.0)
            
            # 4. Create ego vehicle
            self.ego_vehicle = sensors.create_ego_vehicle(self.world, self.config)
            if not self.ego_vehicle:
                logger.error("[ERROR] Failed to create ego vehicle")
                return False
            
            # Wait for ego vehicle to stabilize
            time.sleep(0.5)
            
            # 5. Generate NPC vehicles
            npc_count = sensors.spawn_npc_vehicles(self.world, self.config)
            logger.info(f"[OK] Spawned {npc_count} NPC vehicles")
            
            # Wait for NPC vehicles to spawn
            time.sleep(0.5)
            
            # 6. Initialize sensors
            self.sensor_manager = sensors.SensorManager(self.world, self.ego_vehicle, self.config)
            if not self.sensor_manager.setup():
                logger.error("[ERROR] Sensor initialization failed")
                return False
            
            # 7. Initialize detector
            self.detector = tracker.YOLODetector(self.config)
            
            # 8. Initialize tracker
            self.tracker = tracker.SORTTracker(self.config)
            
            # 9. Initialize visualizer
            self.visualizer = Visualizer(self.config)
            
            # 10. Setup detection thread
            use_async = self.config.get('use_async_detection', True)
            if use_async:
                self._setup_detection_thread()
            
            logger.info("[OK] System initialization complete, ready to start tracking")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] System initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        if self.enable_sign_detection:
            try:
                from sign_detector import TrafficSignDetector
                self.traffic_sign_detector = TrafficSignDetector(
                    config.get('traffic_sign', {})
                )
                logger.info("✅ 交通标志检测器已启用")
            except Exception as e:
                logger.warning(f"交通标志检测器初始化失败: {e}")
                self.traffic_sign_detector = None

    def _setup_detection_thread(self):
        """Setup detection thread"""
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
            logger.info("[OK] Detection thread started")
        except Exception as e:
            logger.warning(f"Detection thread setup failed, using synchronous mode: {e}")
            self.detection_thread = None
    
    def _collect_statistics_data(self, fps, detection_time, tracking_time, tracks_info):
        """
        Collect statistics data
        
        Args:
            fps: Current frame rate
            detection_time: Detection time
            tracking_time: Tracking time
            tracks_info: Tracking information list
            
        Returns:
            dict: Statistics data
        """
        # Get system performance data
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # Get GPU usage (if available)
        try:
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization()
            else:
                gpu_usage = 0
        except:
            gpu_usage = 0
        
        # Count vehicle types
        vehicle_counts = {'car': 0, 'bus': 0, 'truck': 0}
        for track in tracks_info:
            class_name = track.get('class_name', '').lower()
            if class_name in vehicle_counts:
                vehicle_counts[class_name] += 1
        
        # Count behavior types
        behavior_counts = {
            'dangerous': 0, 'stopped': 0, 'overtaking': 0,
            'lane_changing': 0, 'turning': 0, 'accelerating': 0,
            'braking': 0, 'normal': 0
        }
        
        for track in tracks_info:
            if track.get('is_dangerous', False):
                behavior_counts['dangerous'] += 1
            elif track.get('is_stopped', False):
                behavior_counts['stopped'] += 1
            elif track.get('is_overtaking', False):
                behavior_counts['overtaking'] += 1
            elif track.get('is_lane_changing', False):
                behavior_counts['lane_changing'] += 1
            elif track.get('is_turning', False):
                behavior_counts['turning'] += 1
            elif track.get('is_accelerating', False):
                behavior_counts['accelerating'] += 1
            elif track.get('is_braking', False):
                behavior_counts['braking'] += 1
            else:
                behavior_counts['normal'] += 1
        
        # Get performance monitoring data
        perf_stats = self.perf_monitor.get_stats()
        
        # Detection thread status
        detection_thread_status = 'Running' if self.detection_thread and self.detection_thread.is_alive() else 'Not running'
        
        # 点云状态（新增）
        pointcloud_status = 'Enabled' if self.config.get('use_lidar', True) else 'Disabled'
        if self.visualizer:
            pointcloud_status += ' | Showing' if self.visualizer.show_pointcloud else ' | Hidden'
        
        return {
            # System status
            'fps': fps,
            'total_frames': self.frame_count,
            'run_time': time.time() - self.start_time,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'gpu_usage': gpu_usage,
            'detection_thread': detection_thread_status,
            
            # Object statistics
            'total_objects': len(tracks_info),
            'vehicle_counts': vehicle_counts,
            'behavior_counts': {k: v for k, v in behavior_counts.items() if v > 0},
            
            # Performance metrics
            'avg_detection_time': detection_time * 1000,  # Convert to milliseconds
            'avg_tracking_time': tracking_time * 1000,    # Convert to milliseconds
            'avg_frame_time': perf_stats.get('avg_frame_time', 0),
            
            # 点云状态（新增）
            'pointcloud_status': pointcloud_status,
            'pointcloud_enabled': self.config.get('use_lidar', True),
            'pointcloud_showing': self.visualizer.show_pointcloud if self.visualizer else False,
            
            # Raw data (for charts)
            'detection_time': detection_time,
            'tracking_time': tracking_time,
        }
    
    def _check_and_reload_config(self):
        """
        检查并重新加载配置文件
        """
        try:
            # 1. 检查是否启用热重载
            hot_reload_config = self.config.get('hot_reload', {})
            if not hot_reload_config.get('enabled', True):
                return
        
            # 2. 检查间隔
            check_interval = hot_reload_config.get('check_interval', 100)
            if self.frame_count % check_interval != 0:
                return
        
            # 3. 获取配置文件路径
            config_path = self.config.get('_config_path')
            if not config_path:
                logger.debug("[HOT RELOAD] 无配置文件路径")
                return
          
            # 4. 检查文件是否存在
            if not os.path.exists(config_path):
                logger.warning(f"[HOT RELOAD] 配置文件不存在: {config_path}")
                return
        
            # 5. 检查文件是否被修改
            try:
                current_mtime = os.path.getmtime(config_path)
            except Exception as e:
                logger.warning(f"[HOT RELOAD] 无法获取文件修改时间: {e}")
                return
        
            last_mtime = self.config.get('_config_mtime', 0)
         
            if current_mtime <= last_mtime:
                return  # 文件未修改
        
            logger.info("🔄 [HOT RELOAD] 检测到配置文件更新，重新加载...")
        
            # 6. 重新加载配置
            new_config = load_config(config_path)
         
            # 7. 安全更新配置参数
            safe_keys_only = hot_reload_config.get('safe_keys_only', True)
            updated_keys = []
        
            if safe_keys_only:
                # 只更新安全参数列表中的键
                safe_keys = self.config.get('hot_reload_safe_keys', [])
                for key in safe_keys:
                    if key in new_config and key in self.config:
                        old_val = self.config[key]
                        new_val = new_config[key]
                        if old_val != new_val:
                            self.config[key] = new_val
                            updated_keys.append((key, old_val, new_val))
            else:
                # 更新所有参数（危险模式）
                for key, new_val in new_config.items():
                    if not key.startswith('_'):  # 跳过内部键
                        old_val = self.config.get(key)
                        if old_val != new_val:
                            self.config[key] = new_val
                            updated_keys.append((key, old_val, new_val))
        
            # 8. 更新配置文件修改时间
            self.config['_config_mtime'] = current_mtime
        
            # 9. 应用配置更新到各组件
            if updated_keys:
                self._apply_config_updates(updated_keys)
            
                # 打印更新日志
                logger.info("✅ [HOT RELOAD] 配置更新完成:")
                for key, old_val, new_val in updated_keys:
                    logger.info(f"     {key}: {old_val} → {new_val}")
            else:
                logger.info("📝 [HOT RELOAD] 配置文件已重新加载，但无参数变化")
            
        except Exception as e:
            logger.warning(f"[HOT RELOAD] 配置更新失败: {e}")
            import traceback
            traceback.print_exc()
                
    def _apply_config_updates(self, updated_keys):
        """
        将配置更新应用到各组件
        """
        for key, old_val, new_val in updated_keys:
            try:
                # 1. 应用到检测器
                if self.detector:
                    if key == 'conf_thres':
                        self.detector.conf_thres = new_val
                        logger.debug(f"更新检测器置信度阈值: {new_val}")
                    elif key == 'iou_thres':
                        self.detector.iou_thres = new_val
                        logger.debug(f"更新检测器IOU阈值: {new_val}")
                    elif key == 'yolo_imgsz_max':
                        logger.warning("⚠️ YOLO输入尺寸变更需要重启检测器才能生效")
                
                # 2. 应用到跟踪器
                if self.tracker:
                    if key == 'max_age':
                        self.tracker.max_age = new_val
                        logger.debug(f"更新跟踪器max_age: {new_val}")
                    elif key == 'min_hits':
                        self.tracker.min_hits = new_val
                        logger.debug(f"更新跟踪器min_hits: {new_val}")
                    elif key == 'iou_thres':
                        self.tracker.iou_threshold = new_val
                        logger.debug(f"更新跟踪器IOU阈值: {new_val}")
                
                # 3. 应用到传感器
                if self.sensor_manager:
                    if key == 'fov':
                        logger.warning("⚠️ 相机FOV变更需要重启传感器才能生效")
                    elif key == 'use_lidar':
                        logger.warning("⚠️ LiDAR开关变更需要重启程序才能生效")
                
                # 4. 应用到行为分析参数
                if key in ['stop_speed_thresh', 'danger_dist_thresh']:
                    logger.debug(f"更新行为分析参数 {key}: {new_val}")
                
                # 5. 应用到天气
                if key == 'weather' and self.world:
                    from sensors import set_weather  # 需要导入
                    set_weather(self.world, new_val)
                    logger.info(f"🌤️  天气已更新: {new_val}")
                
                # 6. 帧率相关参数已在 _control_frame_rate 中处理
                
            except Exception as e:
                logger.warning(f"[热重载] 应用配置更新失败 {key}: {e}")
    
    def _force_reload_config(self):
        """
        强制重新加载配置文件（手动触发）
        """
        config_path = self.config.get('_config_path')
        if not config_path or not os.path.exists(config_path):
            logger.warning("无法强制重载：配置文件路径无效")
            return
        
        logger.info("🔄 [热重载] 手动触发配置重新加载...")
        # 强制将修改时间设为0，确保会重新加载
        self.config['_config_mtime'] = 0

    def run(self):
        """Run main loop"""
        import time
        import queue
        
        if not self.initialize():
            logger.error("[ERROR] System initialization failed, cannot run")
            return
        
        self.running = True
        logger.info("[START] Starting tracking...")
        
        try:
            while self.running:
                # Start frame timing
                self.perf_monitor.start_frame()
                
                # 1. Update CARLA world
                self.world.tick()
                
                # 2. Get sensor data
                sensor_data = self.sensor_manager.get_sensor_data()
                image = sensor_data.get('image')
                
                if not utils.valid_img(image):
                    logger.warning("Invalid image received, skipping frame")
                    time.sleep(0.1)
                    continue
                
                # 3. Execute detection (synchronous or asynchronous)
                detections = []
                detection_start = time.time()
                
                if self.detection_thread and self.detection_thread.is_alive():
                    # Asynchronous detection
                    if not self.image_queue.full():
                        self.image_queue.put(image.copy())
                    
                    try:
                        processed_image, detections = self.result_queue.get(timeout=0.05)
                        if processed_image is not None:
                            image = processed_image
                    except queue.Empty:
                        # Queue empty, use previous detection result
                        pass
                else:
                    # Synchronous detection
                    detections = self.detector.detect(image)
                
                detection_time = time.time() - detection_start
                self.perf_monitor.record_detection_time(detection_time)
                
                # 4. Update tracker
                ego_center = (self.config['img_width'] // 2, self.config['img_height'] // 2)
                
                # Get LiDAR detection results (if available)
                lidar_detections = sensor_data.get('lidar_objects', [])
                
                tracking_start = time.time()
                boxes, ids, classes = self.tracker.update(
                    detections=detections,
                    ego_center=ego_center,
                    lidar_detections=lidar_detections if lidar_detections else None
                )
                tracking_time = time.time() - tracking_start
                self.perf_monitor.record_tracking_time(tracking_time)
                
                # 5. Get tracking detailed information
                tracks_info = self.tracker.get_tracks_info()
                
                # 6. Update FPS
                fps = self.fps_counter.update()
                
                # 7. Collect statistics data
                stats_data = self._collect_statistics_data(fps, detection_time, tracking_time, tracks_info)
                
                # 8. Update visualizer's performance data
                self.visualizer.update_performance_data(fps, detection_time, tracking_time, stats_data)
                
                # 9. Visualization
                result_image = self.visualizer.draw_detections(
                    image=image,
                    boxes=boxes,
                    ids=ids,
                    classes=classes,
                    tracks_info=tracks_info
                )
                
                # 10. 显示结果（传递统计数据）
                key = self.visualizer.show(result_image, stats_data=stats_data)
                
                # 11. 处理键盘输入
                self._handle_keyboard_input(key)
                
                # 12. 帧率控制
                self._control_frame_rate(fps)
                
                # ============ 新增：热重载检查 ============
                self._check_and_reload_config()
                # ========================================
                
                # 13. 更新状态
                self.frame_count += 1
                self.perf_monitor.end_frame()

                # 添加点云显示（新增）
                if sensor_data.get('pointcloud') is not None:
                    self.visualizer.update_pointcloud(sensor_data['pointcloud'])
                
                # Add color legend (if enabled)
                if self.show_legend:
                    result_image = self.visualizer.draw_color_legend(result_image)
                
                # Display FPS on image (top)
                if utils.valid_img(result_image):
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(result_image, fps_text, (self.config['img_width'] - 100, 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 10. Display results (pass statistics data for updating statistics window)
                key = self.visualizer.show(result_image, stats_data=stats_data)
                
                # 11. Handle keyboard input
                self._handle_keyboard_input(key)
                
                # 12. Frame rate control
                self._control_frame_rate(fps)
                
                # 13. Update state
                self.frame_count += 1
                self.perf_monitor.end_frame()
                
                # 14. Periodically print status
                if self.frame_count % 100 == 0:
                    self._print_status(stats_data)
                
                 # ============ 新增：交通标志检测 ============
                detected_signs = []
                if self.traffic_sign_detector and self.enable_sign_detection:
                    try:
                        # 检测标志
                        detected_signs = self.traffic_sign_detector.detect(
                            image, 
                            ego_speed=self.ego_vehicle.get_velocity().length() if self.ego_vehicle else 0.0
                        )
                    
                        # 在图像上绘制标志
                        if self.traffic_sign_detector.config.get('show_signs', True):
                            image = self.traffic_sign_detector.draw_signs(image, detected_signs)
                    
                    except Exception as e:
                        logger.debug(f"标志检测失败: {e}")

        except KeyboardInterrupt:
            logger.info("[STOP] User interrupted program")
        except Exception as e:
            logger.error(f"[ERROR] Runtime error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def _handle_keyboard_input(self, key):
        """Handle keyboard input"""
        # ESC key to exit
        if key == 27:  # ESC
            logger.info("[STOP] ESC pressed, exiting program")
            self.running = False
        
        # W key to switch weather
        elif key == ord('w') or key == ord('W'):
            weather_list = ['clear', 'cloudy', 'rain', 'fog', 'night']
            current_idx = weather_list.index(self.current_weather) if self.current_weather in weather_list else 0
            next_idx = (current_idx + 1) % len(weather_list)
            self.current_weather = weather_list[next_idx]
            set_weather(self.world, self.current_weather)
            logger.info(f"[WEATHER] Weather switched to: {self.current_weather}")
        
        # S key to save screenshot
        elif key == ord('s') or key == ord('S'):
            self._save_screenshot()
        
        # P key to pause/resume
        elif key == ord('p') or key == ord('P'):
            logger.info("[PAUSE] Program paused, press any key to continue...")
            cv2.waitKey(0)
            logger.info("[RESUME] Program resumed")
        
        # T key to toggle statistics window display
        elif key == ord('t') or key == ord('T'):
            self.visualizer.show_stats_window = not self.visualizer.show_stats_window
            status = "Show" if self.visualizer.show_stats_window else "Hide"
            logger.info(f"[STATS] Statistics window: {status}")
            
            # If hiding window, need to close it
            if not self.visualizer.show_stats_window:
                try:
                    cv2.destroyWindow(self.visualizer.stats_window_name)
                except:
                    pass  # Window may already be closed
        
        # M key to toggle color legend display
        elif key == ord('m') or key == ord('M'):
            self.show_legend = not self.show_legend
            status = "Show" if self.show_legend else "Hide"
            logger.info(f"[LEGEND] Color legend: {status}")
        
        # L键切换点云显示（新增）
        elif key == ord('l') or key == ord('L'):
            if self.visualizer:
                self.visualizer.toggle_pointcloud_display()
        
        # 新增：V键切换视角模式
        elif key == ord('v') or key == ord('V'):
            if self.sensor_manager:
                self.sensor_manager.cycle_view_mode()
                # 获取当前视角模式并显示
                current_mode = self.sensor_manager.spectator_manager.view_mode
                mode_names = {
                    'satellite': '卫星视角',
                    'behind': '后方视角', 
                    'first_person': '第一人称视角'
                }
                mode_name = mode_names.get(current_mode, current_mode)
                logger.info(f"[VIEW] 切换到 {mode_name}")
        
         # 新增：R键手动重载配置 
        elif key == ord('r') or key == ord('R'):
            self._force_reload_config()
        
          # 添加：O键切换标志检测
        elif key == ord('o') or key == ord('O'):
            self.enable_sign_detection = not self.enable_sign_detection
            status = "开启" if self.enable_sign_detection else "关闭"
            logger.info(f"🚦 交通标志检测: {status}")

    def _control_frame_rate(self, current_fps):
        """自适应帧率控制（简单版）"""
        import time
        import psutil
    
        # 获取当前系统负载
        cpu_usage = psutil.cpu_percent()
    
        # 根据CPU使用率动态调整目标FPS
        if cpu_usage > 85:  # 非常高负载
            target_fps = max(10, self.config.get('display_fps', 30) * 0.5)
        elif cpu_usage > 70:  # 高负载
            target_fps = max(15, self.config.get('display_fps', 30) * 0.7)
        elif cpu_usage < 40:  # 低负载，可以尝试更高FPS
            target_fps = min(60, self.config.get('display_fps', 30) * 1.5)
        elif cpu_usage < 60:  # 中等负载
            target_fps = min(45, self.config.get('display_fps', 30) * 1.2)
        else:  # 正常负载
            target_fps = self.config.get('display_fps', 30)
    
        if target_fps <= 0:
            return
    
        target_interval = 1.0 / target_fps
    
        # 如果帧率过高，适当休眠
        if current_fps > target_fps * 1.2:  # 允许20%波动
            sleep_time = max(0, target_interval - (1.0 / current_fps))
            time.sleep(sleep_time)
    
    
    def _print_status(self, stats_data):
        """Print system status"""
        total_objects = stats_data.get('total_objects', 0)
        fps = stats_data.get('fps', 0)
        cpu_usage = stats_data.get('cpu_usage', 0)
        
        logger.info(f"[STATUS] Frames={self.frame_count}, "
                   f"FPS={fps:.1f}, "
                   f"Objects={total_objects}, "
                   f"CPU={cpu_usage:.1f}%")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("[CLEANUP] Cleaning up resources...")
        
        # Stop detection thread
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.join(timeout=2.0)
        
        # Destroy visualizer
        if self.visualizer:
            self.visualizer.destroy()
        
        # Destroy sensors
        if self.sensor_manager:
            self.sensor_manager.destroy()
        
        # Clean up CARLA actors
        if self.world:
            # Exclude ego vehicle ID (if exists)
            exclude_ids = [self.ego_vehicle.id] if self.ego_vehicle and self.ego_vehicle.is_alive else []
            sensors.clear_all_actors(self.world, exclude_ids)
        
        # Restore CARLA settings
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        # Print final performance statistics
        if self.perf_monitor:
            self.perf_monitor.print_stats()
        
        # Print final runtime
        total_time = time.time() - self.start_time
        logger.info(f"[TIME] Total runtime: {total_time:.1f} seconds")
        logger.info(f"[STATS] Average FPS: {self.frame_count/total_time:.1f}" if total_time > 0 else "")
        
        logger.info("[OK] Resource cleanup complete")

# ======================== Main Function ========================

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CARLA Multi-Object Tracking System')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA server address (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port (default: 2000)')
    parser.add_argument('--weather', type=str, default='clear',
                       choices=['clear', 'cloudy', 'rain', 'fog', 'night'],
                       help='Initial weather (default: clear)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--conf-thres', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--no-lidar', action='store_true',
                       help='Disable LiDAR')
    parser.add_argument('--no-stats', action='store_true',
                       help='Do not show statistics window at startup')
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, 
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
               level="INFO")
    
    # Record start time
    start_time = time.time()
    logger.info("=" * 50)
    logger.info("CARLA Multi-Object Tracking System (Enhanced)")
    logger.info("=" * 50)
    
    try:
        # 1. Load configuration
        config = load_config(args.config)
        
        # 2. Override configuration with command line arguments
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
        
        # 3. Create and run tracking system
        system = CarlaTrackingSystem(config)
        
        # Set initial display state
        if args.no_stats:
            system.visualizer.show_stats_window = False
        
        system.run()
        
    except Exception as e:
        logger.error(f"[ERROR] Program runtime error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Calculate runtime
        run_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info(f"[TIME] Program runtime: {run_time:.1f} seconds")
        logger.info("[END] Program ended")
        logger.info("=" * 50)

if __name__ == "__main__":
    # 检查配置
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch not found, please install: pip install torch")
        sys.exit(1)
    
    try:
        import carla
    except ImportError:
        print("[ERROR] CARLA Python API not found")
        print("Please copy PythonAPI/carla from CARLA installation directory to project directory")
        sys.exit(1)
    
    try:
        import psutil
    except ImportError:
        print("[ERROR] psutil not found, please install: pip install psutil")
        sys.exit(1)
    
    # Run main program
    main()