"""
配置管理模块 - 集中管理所有系统参数
添加配置热重载功能
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable
import json
import os
import threading
import time
from collections import defaultdict


@dataclass
class AppConfig:
    """应用配置参数"""
    # 性能参数
    max_image_size: Tuple[int, int] = (1200, 800)
    cache_size: int = 8
    batch_size: int = 30
    
    # 图像处理参数
    adaptive_clip_limit: float = 2.5
    adaptive_grid_size: Tuple[int, int] = (8, 8)
    gaussian_kernel: Tuple[int, int] = (5, 5)
    
    # 检测参数
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    hough_threshold: int = 30
    hough_min_length: int = 25
    hough_max_gap: int = 40
    min_contour_area: float = 0.005
    
    # 方向分析参数
    deviation_threshold: float = 0.15
    width_ratio_threshold: float = 0.7
    confidence_threshold: float = 0.5
    min_confidence_for_direction: float = 0.25
    
    # 路径预测参数
    prediction_steps: int = 10
    prediction_distance: float = 0.75
    min_prediction_points: int = 4
    
    # 视频处理参数
    video_frame_skip: int = 1
    video_fps: int = 10
    camera_id: int = 0
    
    # 置信度参数
    confidence_smoothing_factor: float = 0.7
    quality_weight_lane: float = 0.5
    quality_weight_road: float = 0.3
    quality_weight_consistency: float = 0.2
    
    # 界面参数
    ui_refresh_rate: int = 100
    animation_duration: int = 300
    
    # 车道线检测参数
    lane_detection_methods: List[str] = None
    
    # 性能优化参数（新增）
    enable_frame_buffer: bool = True
    frame_buffer_size: int = 5
    adaptive_skip_frames: bool = True
    max_processing_time: float = 0.1  # 最大处理时间（秒）
    min_fps: int = 5  # 最小FPS
    
    # 错误恢复参数（新增）
    max_error_count: int = 5
    recovery_timeout: int = 10
    auto_recovery: bool = True
    
    def __post_init__(self):
        if self.lane_detection_methods is None:
            self.lane_detection_methods = ['canny', 'sobel', 'gradient']
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'max_image_size': self.max_image_size,
            'cache_size': self.cache_size,
            'batch_size': self.batch_size,
            'adaptive_clip_limit': self.adaptive_clip_limit,
            'adaptive_grid_size': self.adaptive_grid_size,
            'canny_threshold1': self.canny_threshold1,
            'canny_threshold2': self.canny_threshold2,
            'hough_threshold': self.hough_threshold,
            'confidence_threshold': self.confidence_threshold,
            'prediction_steps': self.prediction_steps,
            'prediction_distance': self.prediction_distance,
            'video_frame_skip': self.video_frame_skip,
            'video_fps': self.video_fps,
            'camera_id': self.camera_id,
            'enable_frame_buffer': self.enable_frame_buffer,
            'frame_buffer_size': self.frame_buffer_size,
            'adaptive_skip_frames': self.adaptive_skip_frames,
            'max_processing_time': self.max_processing_time,
            'min_fps': self.min_fps,
            'max_error_count': self.max_error_count,
            'recovery_timeout': self.recovery_timeout,
            'auto_recovery': self.auto_recovery
        }
    
    def save(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"配置已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AppConfig':
        """从文件加载配置"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 从字典创建配置对象
                config = cls()
                for key, value in data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                print(f"配置已从 {filepath} 加载")
                return config
            except Exception as e:
                print(f"加载配置失败: {e}")
                return cls()
        else:
            print(f"配置文件不存在: {filepath}")
            return cls()


class SceneConfig:
    """场景特定配置"""
    
    # 高速公路配置
    HIGHWAY = AppConfig(
        adaptive_clip_limit=2.0,
        canny_threshold1=60,
        canny_threshold2=180,
        hough_threshold=35,
        prediction_distance=0.9,
        confidence_threshold=0.7,
        video_fps=15,
        max_processing_time=0.08
    )
    
    # 城市道路配置
    URBAN = AppConfig(
        adaptive_clip_limit=1.5,
        canny_threshold1=40,
        canny_threshold2=120,
        hough_threshold=25,
        prediction_distance=0.6,
        confidence_threshold=0.5,
        video_fps=10,
        max_processing_time=0.12
    )
    
    # 乡村道路配置
    RURAL = AppConfig(
        adaptive_clip_limit=3.0,
        adaptive_grid_size=(16, 16),
        canny_threshold1=30,
        canny_threshold2=90,
        hough_threshold=20,
        min_contour_area=0.002,
        prediction_distance=0.7,
        confidence_threshold=0.4,
        video_fps=8,
        max_processing_time=0.15
    )
    
    @classmethod
    def get_scene_config(cls, scene_type: str) -> AppConfig:
        """获取场景特定配置"""
        config_map = {
            'highway': cls.HIGHWAY,
            'urban': cls.URBAN,
            'rural': cls.RURAL
        }
        return config_map.get(scene_type, AppConfig())


class ConfigManager:
    """配置管理器 - 支持热重载"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_file: str = "app_config.json"):
        if not hasattr(self, 'initialized'):
            self.config_file = config_file
            self.config = AppConfig()
            self.last_modified = 0
            self.change_callbacks = defaultdict(list)
            self.watch_thread = None
            self.is_watching = False
            self.initialized = True
            
            # 自动加载配置
            self._load_config()
    
    def _load_config(self):
        """加载配置"""
        if os.path.exists(self.config_file):
            try:
                self.config = AppConfig.load(self.config_file)
                self.last_modified = os.path.getmtime(self.config_file)
                print(f"配置管理器初始化完成，使用配置文件: {self.config_file}")
            except Exception as e:
                print(f"配置管理器初始化失败: {e}")
        else:
            print("配置文件不存在，使用默认配置")
    
    def add_change_callback(self, callback: Callable[[AppConfig], None], 
                           component: str = "global"):
        """添加配置变更回调"""
        self.change_callbacks[component].append(callback)
    
    def remove_change_callback(self, callback: Callable[[AppConfig], None], 
                              component: str = "global"):
        """移除配置变更回调"""
        if component in self.change_callbacks:
            self.change_callbacks[component] = [
                cb for cb in self.change_callbacks[component] if cb != callback
            ]
    
    def notify_change(self, component: str = "global"):
        """通知配置变更"""
        if component in self.change_callbacks:
            for callback in self.change_callbacks[component]:
                try:
                    callback(self.config)
                except Exception as e:
                    print(f"配置变更回调执行失败: {e}")
    
    def save_current_config(self) -> bool:
        """保存当前配置"""
        try:
            self.config.save(self.config_file)
            self.last_modified = os.path.getmtime(self.config_file)
            print("当前配置已保存")
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def check_and_reload(self) -> bool:
        """检查并重新加载配置"""
        try:
            if os.path.exists(self.config_file):
                mtime = os.path.getmtime(self.config_file)
                if mtime > self.last_modified:
                    print("检测到配置文件变更，重新加载...")
                    self._load_config()
                    self.notify_change()
                    return True
        except Exception as e:
            print(f"配置重载失败: {e}")
        return False
    
    def start_watching(self, interval: float = 2.0):
        """开始监控配置文件变化"""
        if self.is_watching:
            return
            
        self.is_watching = True
        self.watch_thread = threading.Thread(
            target=self._watch_config_file,
            args=(interval,),
            daemon=True
        )
        self.watch_thread.start()
        print(f"开始监控配置文件变化，检查间隔: {interval}秒")
    
    def stop_watching(self):
        """停止监控配置文件变化"""
        self.is_watching = False
        if self.watch_thread:
            self.watch_thread.join(timeout=1.0)
        print("停止监控配置文件变化")
    
    def _watch_config_file(self, interval: float):
        """监控配置文件变化"""
        while self.is_watching:
            self.check_and_reload()
            time.sleep(interval)
    
    def update_config(self, updates: dict, save_to_file: bool = True):
        """更新配置"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                old_value = getattr(self.config, key)
                setattr(self.config, key, value)
                print(f"配置更新: {key} = {old_value} -> {value}")
        
        if save_to_file:
            self.save_current_config()
        
        self.notify_change()
    
    def reset_to_defaults(self):
        """重置为默认配置"""
        self.config = AppConfig()
        self.save_current_config()
        self.notify_change()
        print("配置已重置为默认值")


# 全局配置管理器实例
config_manager = ConfigManager()