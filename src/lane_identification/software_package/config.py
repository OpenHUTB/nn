"""
配置管理模块 - 集中管理所有系统参数
"""

from dataclasses import dataclass
from typing import Tuple, List
import json
import os

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
    confidence_threshold: float = 0.6
    
    # 路径预测参数
    prediction_steps: int = 10
    prediction_distance: float = 0.75
    min_prediction_points: int = 4
    
    # 置信度参数
    min_confidence_for_direction: float = 0.4
    confidence_smoothing_factor: float = 0.7
    quality_weight_lane: float = 0.5
    quality_weight_road: float = 0.3
    quality_weight_consistency: float = 0.2
    
    # 界面参数
    ui_refresh_rate: int = 100
    animation_duration: int = 300
    
    # 车道线检测参数
    lane_detection_methods: List[str] = None
    
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
            'prediction_distance': self.prediction_distance
        }
    
    def save(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'AppConfig':
        """从文件加载配置"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # 从字典创建配置对象
            config = cls()
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            return config
        else:
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
        confidence_threshold=0.7
    )
    
    # 城市道路配置
    URBAN = AppConfig(
        adaptive_clip_limit=1.5,
        canny_threshold1=40,
        canny_threshold2=120,
        hough_threshold=25,
        prediction_distance=0.6,
        confidence_threshold=0.5
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
        confidence_threshold=0.4
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