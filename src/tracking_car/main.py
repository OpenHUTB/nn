"""
main.py - CARLA多目标跟踪系统主程序
增强版：彩色ID编码 + 独立统计窗口
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


# ======================== 可视化（增强版：独立统计窗口） ========================

class Visualizer:
    """可视化管理器（增强版：彩色ID编码 + 独立统计窗口）"""
    
    def __init__(self, config):
        self.config = config
        self.window_name = "CARLA Object Tracking"
        self.stats_window_name = "📊 实时统计面板"
        
        # 创建主窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 
                        config.get('window_width', 1280), 
                        config.get('window_height', 720))
        
        # 创建独立统计窗口
        cv2.namedWindow(self.stats_window_name, cv2.WINDOW_NORMAL)
        # 设置统计窗口大小
        stats_width = 600
        stats_height = 800
        cv2.resizeWindow(self.stats_window_name, stats_width, stats_height)
        
        # 移动统计窗口位置（避免遮挡主窗口）
        cv2.moveWindow(self.stats_window_name, 
                      config.get('window_width', 1280) + 50,  # 放在主窗口右侧
                      100)                                    # 垂直位置
        
        # 统计面板状态
        self.show_stats_window = True  # 是否显示独立统计窗口
        self.stats_image = None        # 统计面板图像
        self.stats_update_interval = 2  # 统计更新间隔（帧数）
        self.stats_frame_counter = 0   # 帧计数器
        
        # 车辆类别颜色映射
        self.class_colors = {
            'car': (255, 0, 0),      # 蓝色 - 小汽车
            'bus': (0, 255, 0),      # 绿色 - 公交车
            'truck': (0, 0, 255),    # 红色 - 卡车
            'default': (255, 255, 0) # 青色 - 默认
        }
        
        # 行为状态颜色映射（优先级从高到低）
        self.behavior_colors = {
            'dangerous': (0, 0, 255),      # 红色 - 危险（距离过近）
            'stopped': (0, 255, 255),      # 黄色 - 停车
            'overtaking': (255, 0, 255),   # 紫色 - 超车
            'lane_changing': (0, 255, 255), # 青色 - 变道
            'turning': (0, 255, 255),      # 青色 - 转弯
            'accelerating': (255, 0, 0),   # 蓝色 - 加速
            'braking': (0, 165, 255),      # 橙色 - 刹车
            'normal': (0, 255, 0)          # 绿色 - 正常行驶
        }
        
        # 行为状态图标映射
        self.behavior_icons = {
            'dangerous': '⚠',    # 警告
            'stopped': '🛑',     # 停止
            'overtaking': '💨',  # 超车
            'lane_changing': '↔', # 变道
            'turning': '↪',      # 转弯
            'accelerating': '🚀', # 加速
            'braking': '🛑',     # 刹车
            'normal': '→'        # 正常
        }
        
        # 性能数据历史
        self.fps_history = []
        self.detection_time_history = []
        self.tracking_time_history = []
        self.max_history_length = 100  # 增加历史长度用于更详细的图表
        
        # 状态历史（用于趋势分析）
        self.object_count_history = []
        self.cpu_usage_history = []
        self.memory_usage_history = []
        
        logger.info("✅ 可视化器初始化完成（彩色ID编码 + 独立统计窗口）")
    
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
    
    def _get_behavior_icon(self, track_info):
        """
        根据行为状态返回对应图标
        
        Args:
            track_info: 跟踪目标信息字典
            
        Returns:
            str: 行为图标
        """
        if not track_info:
            return self.behavior_icons['normal']
        
        # 优先级：危险 > 停车 > 超车 > 变道/转弯 > 加速/刹车 > 正常
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
    
    def update_performance_data(self, fps, detection_time, tracking_time, stats_data=None):
        """
        更新性能数据（增强版，支持更多数据）
        
        Args:
            fps: 当前帧率
            detection_time: 检测时间（秒）
            tracking_time: 跟踪时间（秒）
            stats_data: 统计数据字典
        """
        self.fps_history.append(fps)
        self.detection_time_history.append(detection_time * 1000)  # 转换为毫秒
        self.tracking_time_history.append(tracking_time * 1000)    # 转换为毫秒
        
        # 如果有统计数据，也更新状态历史
        if stats_data:
            self.object_count_history.append(stats_data.get('total_objects', 0))
            self.cpu_usage_history.append(stats_data.get('cpu_usage', 0))
            self.memory_usage_history.append(stats_data.get('memory_usage', 0))
        
        # 保持历史数据长度
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
        创建独立统计窗口的图像
        
        Args:
            stats_data: 统计数据字典
            
        Returns:
            np.ndarray: 统计面板图像
        """
        # 创建统计面板图像（浅灰色背景）
        stats_width = 600
        stats_height = 800
        stats_image = np.ones((stats_height, stats_width, 3), dtype=np.uint8) * 240  # 浅灰色背景
        
        # 1. 标题区域
        title_height = 80
        cv2.rectangle(stats_image, (0, 0), (stats_width, title_height), (50, 50, 80), -1)
        
        title = "🚗 CARLA 实时统计面板"
        cv2.putText(stats_image, title, 
                   (stats_width // 2 - 150, title_height // 2 + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        subtitle = "独立窗口 - 按 T 键切换显示"
        cv2.putText(stats_image, subtitle,
                   (stats_width // 2 - 140, title_height // 2 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_offset = title_height + 20
        
        # 2. 系统状态区块
        y_offset = self._draw_stats_section(stats_image, "⚙️ 系统状态", y_offset, stats_data, self._draw_system_stats)
        
        # 3. 目标统计区块
        y_offset = self._draw_stats_section(stats_image, "🎯 目标统计", y_offset, stats_data, self._draw_object_stats)
        
        # 4. 性能图表区块
        y_offset = self._draw_stats_section(stats_image, "📈 性能图表", y_offset, stats_data, self._draw_performance_charts)
        
        # 5. 历史趋势区块
        if len(self.fps_history) > 5:
            y_offset = self._draw_stats_section(stats_image, "📊 历史趋势", y_offset, stats_data, self._draw_trend_charts)
        
        # 6. 底部信息
        bottom_y = stats_height - 30
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(stats_image, f"更新时间: {timestamp}", 
                   (20, bottom_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        frame_info = f"总帧数: {stats_data.get('total_frames', 0)}"
        cv2.putText(stats_image, frame_info,
                   (stats_width - 150, bottom_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        return stats_image
    
    def _draw_stats_section(self, image, title, y_start, stats_data, draw_function):
        """
        绘制统计区块的通用模板
        
        Returns:
            int: 下一个区块的起始Y坐标
        """
        section_height = 200  # 每个区块默认高度
        
        # 区块背景
        cv2.rectangle(image, (10, y_start), (590, y_start + section_height), (255, 255, 255), -1)
        cv2.rectangle(image, (10, y_start), (590, y_start + section_height), (220, 220, 220), 2)
        
        # 区块标题
        cv2.putText(image, title, (20, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        
        # 绘制分割线
        cv2.line(image, (20, y_start + 35), (580, y_start + 35), (200, 200, 200), 1)
        
        # 调用具体的绘制函数
        content_y = y_start + 50
        content_y = draw_function(image, content_y, stats_data)
        
        # 如果绘制函数返回了新的Y坐标，使用它；否则使用默认高度
        if content_y > y_start + section_height:
            section_height = content_y - y_start
        
        return y_start + section_height + 20
    
    def _draw_system_stats(self, image, y_start, stats_data):
        """
        绘制系统状态信息
        """
        x_left = 30
        x_right = 300
        y = y_start
        
        # 定义状态项
        status_items = [
            ("FPS", f"{stats_data.get('fps', 0):.1f}", 
             (0, 255, 0) if stats_data.get('fps', 0) > 20 else (0, 165, 255)),
            ("运行时间", f"{stats_data.get('run_time', 0):.0f}s", (100, 100, 100)),
            ("CPU使用率", f"{stats_data.get('cpu_usage', 0):.1f}%",
             (0, 255, 0) if stats_data.get('cpu_usage', 0) < 70 else (0, 165, 255) if stats_data.get('cpu_usage', 0) < 90 else (0, 0, 255)),
            ("内存使用率", f"{stats_data.get('memory_usage', 0):.1f}%",
             (0, 255, 0) if stats_data.get('memory_usage', 0) < 70 else (0, 165, 255) if stats_data.get('memory_usage', 0) < 90 else (0, 0, 255)),
            ("检测线程", stats_data.get('detection_thread', '未知'),
             (0, 255, 0) if stats_data.get('detection_thread') == '运行中' else (0, 0, 255)),
            ("平均帧时间", f"{stats_data.get('avg_frame_time', 0):.1f}ms",
             (0, 255, 0) if stats_data.get('avg_frame_time', 0) < 33 else (0, 165, 255) if stats_data.get('avg_frame_time', 0) < 50 else (0, 0, 255)),
        ]
        
        # 分两列绘制
        for i, (label, value, color) in enumerate(status_items):
            x = x_left if i % 2 == 0 else x_right
            current_y = y + (i // 2) * 30
            
            # 标签
            cv2.putText(image, f"{label}:", (x, current_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
            
            # 值
            cv2.putText(image, value, (x + 120, current_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return y + (len(status_items) // 2 + 1) * 30
    
    def _draw_object_stats(self, image, y_start, stats_data):
        """
        绘制目标统计信息
        """
        y = y_start
        
        # 总目标数
        total_objects = stats_data.get('total_objects', 0)
        cv2.putText(image, f"总目标数: {total_objects}", (30, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        y += 30
        
        # 车辆类型分布（横向条形图）
        vehicle_counts = stats_data.get('vehicle_counts', {})
        if vehicle_counts:
            cv2.putText(image, "车辆类型分布:", (30, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            y += 25
            
            max_count = max(vehicle_counts.values()) if vehicle_counts.values() else 1
            bar_width = 200
            
            types = ['car', 'bus', 'truck']
            type_names = {'car': '小汽车 🚗', 'bus': '公交车 🚌', 'truck': '卡车 🚚'}
            
            for i, v_type in enumerate(types):
                count = vehicle_counts.get(v_type, 0)
                # 条形图
                bar_length = int((count / max_count) * bar_width) if max_count > 0 else 0
                color = self.class_colors.get(v_type, (100, 100, 100))
                
                cv2.rectangle(image, (150, y - 10), (150 + bar_length, y + 5), color, -1)
                
                # 文本
                cv2.putText(image, type_names[v_type], (30, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
                cv2.putText(image, f"{count}", (370, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
                
                y += 25
            y += 10
        
        # 行为分布
        behavior_counts = stats_data.get('behavior_counts', {})
        if behavior_counts:
            cv2.putText(image, "行为分布:", (30, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            y += 25
            
            # 只显示非零行为
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
        绘制性能图表
        """
        chart_x = 30
        chart_y = y_start
        chart_width = 540
        chart_height = 120
        
        # 图表背景
        cv2.rectangle(image, (chart_x, chart_y), 
                     (chart_x + chart_width, chart_y + chart_height), 
                     (250, 250, 250), -1)
        cv2.rectangle(image, (chart_x, chart_y), 
                     (chart_x + chart_width, chart_y + chart_height), 
                     (200, 200, 200), 1)
        
        if len(self.fps_history) > 1:
            # 绘制FPS曲线（绿色）
            self._draw_chart_curve(image, chart_x, chart_y, chart_width, chart_height,
                                 self.fps_history, (0, 180, 0), "FPS", 60)
            
            # 绘制检测时间曲线（红色）
            if self.detection_time_history:
                self._draw_chart_curve(image, chart_x, chart_y, chart_width, chart_height,
                                     self.detection_time_history, (200, 0, 0), "检测(ms)", 100)
            
            # 绘制跟踪时间曲线（蓝色）
            if self.tracking_time_history:
                self._draw_chart_curve(image, chart_x, chart_y, chart_width, chart_height,
                                     self.tracking_time_history, (0, 0, 200), "跟踪(ms)", 50)
        
        # 图表标题
        cv2.putText(image, "实时性能趋势（最近100帧）", 
                   (chart_x + 10, chart_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        
        return chart_y + chart_height + 20
    
    def _draw_trend_charts(self, image, y_start, stats_data):
        """
        绘制历史趋势图表
        """
        chart_x = 30
        chart_y = y_start
        chart_width = 540
        chart_height = 100
        
        # 目标数量趋势
        if len(self.object_count_history) > 1:
            cv2.putText(image, "目标数量趋势:", (chart_x, chart_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            # 图表背景
            cv2.rectangle(image, (chart_x, chart_y), 
                         (chart_x + chart_width, chart_y + chart_height), 
                         (250, 250, 250), -1)
            cv2.rectangle(image, (chart_x, chart_y), 
                         (chart_x + chart_width, chart_y + chart_height), 
                         (200, 200, 200), 1)
            
            # 绘制目标数量曲线
            self._draw_chart_curve(image, chart_x, chart_y, chart_width, chart_height,
                                 self.object_count_history, (100, 0, 200), "目标数", 
                                 max(self.object_count_history) if self.object_count_history else 20)
            
            chart_y += chart_height + 30
        
        # 系统资源趋势
        cv2.putText(image, "系统资源趋势:", (chart_x, chart_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # 图表背景
        cv2.rectangle(image, (chart_x, chart_y), 
                     (chart_x + chart_width, chart_y + chart_height), 
                     (250, 250, 250), -1)
        cv2.rectangle(image, (chart_x, chart_y), 
                     (chart_x + chart_width, chart_y + chart_height), 
                     (200, 200, 200), 1)
        
        # 绘制CPU和内存曲线
        if len(self.cpu_usage_history) > 1:
            self._draw_chart_curve(image, chart_x, chart_y, chart_width, chart_height,
                                 self.cpu_usage_history, (200, 100, 0), "CPU%", 100)
        
        if len(self.memory_usage_history) > 1:
            self._draw_chart_curve(image, chart_x, chart_y, chart_width, chart_height,
                                 self.memory_usage_history, (0, 100, 200), "内存%", 100)
        
        return chart_y + chart_height + 20
    
    def _draw_chart_curve(self, image, x, y, width, height, data, color, label, max_value):
        """
        绘制图表曲线（增强版，带标签）
        """
        if len(data) < 2:
            return
        
        points = []
        data_len = len(data)
        
        for i, value in enumerate(data):
            # 归一化到0-1范围
            normalized = min(1.0, value / max_value) if max_value > 0 else 0
            
            # 计算坐标
            point_x = int(x + (i / (data_len - 1)) * width) if data_len > 1 else x
            point_y = int(y + height - normalized * height)
            
            points.append((point_x, point_y))
        
        # 绘制曲线
        for i in range(1, len(points)):
            cv2.line(image, points[i-1], points[i], color, 2)
        
        # 绘制标签
        label_x = x + width - 80
        label_y = y + 15
        
        # 颜色标记
        cv2.circle(image, (label_x - 10, label_y), 4, color, -1)
        cv2.putText(image, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
    
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
                
                # 绘制行为图标（如果可用）
                if track_info:
                    # 获取行为图标
                    behavior_icon = self._get_behavior_icon(track_info)
                    
                    # 在右上角绘制行为状态
                    behavior_text = behavior_icon
                    (icon_width, icon_height), _ = cv2.getTextSize(
                        behavior_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # 图标位置（右上角）
                    icon_x = x2 - icon_width - 5
                    icon_y = y1 + icon_height + 5
                    
                    # 绘制图标背景
                    cv2.rectangle(result,
                                (icon_x - 3, icon_y - icon_height - 3),
                                (icon_x + icon_width + 3, icon_y + 3),
                                behavior_color, -1)
                    
                    # 绘制图标
                    cv2.putText(result, behavior_text,
                              (icon_x, icon_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
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
        """绘制信息面板"""
        h, w = image.shape[:2]
        
        # 信息面板背景（半透明黑色）
        panel_height = 80
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # 标题
        title = "🚗 CARLA 多目标跟踪系统"
        cv2.putText(image, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 状态信息
        status_lines = [
            f"跟踪目标: {track_count}",
            f"按 ESC 退出 | 按 W 切换天气 | 按 S 保存截图",
            f"按 P 暂停 | 按 T 显示/隐藏统计窗口 | 按 M 显示/隐藏颜色说明"
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
        绘制颜色说明图例
        
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
        
        # 图例标题
        cv2.putText(image, "颜色说明", (legend_x + 10, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 行为状态颜色说明
        behaviors = [
            ('dangerous', '危险', '⚠'),
            ('stopped', '停车', '🛑'),
            ('overtaking', '超车', '💨'),
            ('lane_changing', '变道', '↔'),
            ('accelerating', '加速', '🚀'),
            ('braking', '刹车', '🛑'),
            ('normal', '正常', '→')
        ]
        
        y_offset = 60
        for behavior_key, behavior_name, icon in behaviors:
            # 颜色方块
            color = self.behavior_colors.get(behavior_key, (255, 255, 255))
            cv2.rectangle(image,
                         (legend_x + 10, legend_y + y_offset),
                         (legend_x + 30, legend_y + y_offset + 15),
                         color, -1)
            
            # 行为名称
            text = f"{icon} {behavior_name}"
            cv2.putText(image, text,
                       (legend_x + 40, legend_y + y_offset + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 25
        
        # 车辆类别说明
        cv2.putText(image, "车辆类别:", (legend_x + 10, legend_y + y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        classes = [
            ('car', '小汽车', '🚗'),
            ('bus', '公交车', '🚌'),
            ('truck', '卡车', '🚚')
        ]
        
        y_offset += 40
        for class_key, class_name, icon in classes:
            # 颜色方块
            color = self.class_colors.get(class_key, (255, 255, 255))
            cv2.rectangle(image,
                         (legend_x + 10, legend_y + y_offset),
                         (legend_x + 30, legend_y + y_offset + 15),
                         color, -1)
            
            # 类别名称
            text = f"{icon} {class_name}"
            cv2.putText(image, text,
                       (legend_x + 40, legend_y + y_offset + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 25
        
        return image
    
    def show(self, image, stats_data=None):
        """
        显示图像和统计窗口
        
        Args:
            image: 主窗口图像
            stats_data: 统计数据（用于更新统计窗口）
            
        Returns:
            int: 按键值
        """
        # 显示主窗口
        if utils.valid_img(image):
            cv2.imshow(self.window_name, image)
        
        # 更新统计窗口（每几帧更新一次，避免过频更新影响性能）
        if self.show_stats_window and stats_data is not None:
            self.stats_frame_counter += 1
            
            if self.stats_frame_counter >= self.stats_update_interval:
                self.stats_image = self.create_stats_window_image(stats_data)
                if self.stats_image is not None:
                    cv2.imshow(self.stats_window_name, self.stats_image)
                self.stats_frame_counter = 0
        
        # 等待按键（短暂等待，保持响应性）
        return cv2.waitKey(1)
    
    def destroy(self):
        """销毁所有窗口"""
        cv2.destroyAllWindows()
        logger.info("✅ 所有可视化窗口已关闭")


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
        self.start_time = time.time()  # 程序开始时间
        
        # 检测线程相关
        self.detection_thread = None
        self.image_queue = None
        self.result_queue = None
        
        logger.info("✅ 跟踪系统初始化完成（彩色ID编码 + 独立统计窗口）")
    
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
    
    def _collect_statistics_data(self, fps, detection_time, tracking_time, tracks_info):
        """
        收集统计数据
        
        Args:
            fps: 当前帧率
            detection_time: 检测时间
            tracking_time: 跟踪时间
            tracks_info: 跟踪信息列表
            
        Returns:
            dict: 统计数据
        """
        # 获取系统性能数据
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # 获取GPU使用率（如果可用）
        try:
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization()
            else:
                gpu_usage = 0
        except:
            gpu_usage = 0
        
        # 统计车辆类型
        vehicle_counts = {'car': 0, 'bus': 0, 'truck': 0}
        for track in tracks_info:
            class_name = track.get('class_name', '').lower()
            if class_name in vehicle_counts:
                vehicle_counts[class_name] += 1
        
        # 统计行为类型
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
        
        # 获取性能监控数据
        perf_stats = self.perf_monitor.get_stats()
        
        # 检测线程状态
        detection_thread_status = '运行中' if self.detection_thread and self.detection_thread.is_alive() else '未运行'
        
        return {
            # 系统状态
            'fps': fps,
            'total_frames': self.frame_count,
            'run_time': time.time() - self.start_time,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'gpu_usage': gpu_usage,
            'detection_thread': detection_thread_status,
            
            # 目标统计
            'total_objects': len(tracks_info),
            'vehicle_counts': vehicle_counts,
            'behavior_counts': {k: v for k, v in behavior_counts.items() if v > 0},
            
            # 性能指标
            'avg_detection_time': detection_time * 1000,  # 转换为毫秒
            'avg_tracking_time': tracking_time * 1000,    # 转换为毫秒
            'avg_frame_time': perf_stats.get('avg_frame_time', 0),
            
            # 原始数据（用于图表）
            'detection_time': detection_time,
            'tracking_time': tracking_time,
        }
    
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
                
                # 7. 收集统计数据
                stats_data = self._collect_statistics_data(fps, detection_time, tracking_time, tracks_info)
                
                # 8. 更新可视化器的性能数据
                self.visualizer.update_performance_data(fps, detection_time, tracking_time, stats_data)
                
                # 9. 可视化
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
                
                # 在图像上显示FPS（顶部）
                if utils.valid_img(result_image):
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(result_image, fps_text, (self.config['img_width'] - 100, 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 10. 显示结果（传入统计数据用于更新统计窗口）
                key = self.visualizer.show(result_image, stats_data=stats_data)
                
                # 11. 处理键盘输入
                self._handle_keyboard_input(key)
                
                # 12. 帧率控制
                self._control_frame_rate(fps)
                
                # 13. 更新状态
                self.frame_count += 1
                self.perf_monitor.end_frame()
                
                # 14. 定期打印状态
                if self.frame_count % 100 == 0:
                    self._print_status(stats_data)
                
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
        
        # T键切换统计窗口显示
        elif key == ord('t') or key == ord('T'):
            self.visualizer.show_stats_window = not self.visualizer.show_stats_window
            status = "显示" if self.visualizer.show_stats_window else "隐藏"
            logger.info(f"📊 独立统计窗口: {status}")
            
            # 如果隐藏窗口，需要关闭它
            if not self.visualizer.show_stats_window:
                try:
                    cv2.destroyWindow(self.visualizer.stats_window_name)
                except:
                    pass  # 窗口可能已经关闭
        
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
    
    def _print_status(self, stats_data):
        """打印系统状态"""
        total_objects = stats_data.get('total_objects', 0)
        fps = stats_data.get('fps', 0)
        cpu_usage = stats_data.get('cpu_usage', 0)
        
        logger.info(f"📊 状态: 帧数={self.frame_count}, "
                   f"FPS={fps:.1f}, "
                   f"目标数={total_objects}, "
                   f"CPU={cpu_usage:.1f}%")
    
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
        
        # 打印最终运行时间
        total_time = time.time() - self.start_time
        logger.info(f"⏱️  总运行时间: {total_time:.1f}秒")
        logger.info(f"📈 平均FPS: {self.frame_count/total_time:.1f}" if total_time > 0 else "")
        
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
    parser.add_argument('--no-stats', action='store_true',
                       help='启动时不显示统计窗口')
    
    args = parser.parse_args()
    
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, 
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
               level="INFO")
    
    # 记录开始时间
    start_time = time.time()
    logger.info("=" * 50)
    logger.info("🚗 CARLA多目标跟踪系统启动（增强版：独立统计窗口）")
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
        
        # 设置初始显示状态
        if args.no_stats:
            system.visualizer.show_stats_window = False
        
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
    
    try:
        import psutil
    except ImportError:
        print("❌ 未找到psutil，请安装: pip install psutil")
        sys.exit(1)
    
    # 运行主程序
    main()