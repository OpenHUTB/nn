"""
检测服务 - 封装完整的检测流程，简化 main.py
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import time

from config import AppConfig
from image_processor import SmartImageProcessor, RoadDetector
from lane_detector import LaneDetector
from direction_analyzer import DirectionAnalyzer
from visualizer import Visualizer


class DetectionService:
    """检测服务类 - 统一管理所有检测模块"""

    def __init__(self, config: AppConfig):
        self.config = config
        self._init_modules()

    def _init_modules(self):
        """初始化检测模块"""
        self.image_processor = SmartImageProcessor(self.config)
        self.road_detector = RoadDetector(self.config)
        self.lane_detector = LaneDetector(self.config)
        self.direction_analyzer = DirectionAnalyzer(self.config)
        self.visualizer = Visualizer(self.config)

    def detect_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        检测单张图像

        Returns:
            包含所有检测结果的字典
        """
        try:
            start_time = time.time()

            # 1. 加载和预处理
            result = self.image_processor.load_and_preprocess(image_path)
            if result is None:
                return None

            image, roi_info = result
            light_condition = roi_info.get('light_condition', 'day')
            mask = roi_info.get('mask', np.ones(image.shape[:2], dtype=np.uint8))

            # 2. 道路检测
            road_info = self.road_detector.detect_road(image, mask)

            # 3. 车道线检测
            lane_info = self.lane_detector.detect(image, mask, light_condition)

            # 4. 方向分析
            direction_info = self.direction_analyzer.analyze(road_info, lane_info)

            # 5. 可视化
            visualization = self.visualizer.create_visualization(
                image, road_info, lane_info, direction_info
            )

            processing_time = time.time() - start_time

            return {
                'image': image,
                'visualization': visualization,
                'road_info': road_info,
                'lane_info': lane_info,
                'direction_info': direction_info,
                'processing_time': processing_time,
                'light_condition': light_condition
            }

        except Exception as e:
            print(f"图像检测失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def detect_frame(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        检测视频帧

        Returns:
            (处理后的帧, 检测结果字典)
        """
        try:
            start_time = time.time()

            # 1. 预处理
            processed_frame, roi_info = self.image_processor.preprocess_frame(frame)
            light_condition = roi_info.get('light_condition', 'day')
            mask = roi_info.get('mask', np.ones(processed_frame.shape[:2], dtype=np.uint8))

            # 2-5. 检测流程同图像
            road_info = self.road_detector.detect_road(processed_frame, mask)
            lane_info = self.lane_detector.detect(processed_frame, mask, light_condition)
            direction_info = self.direction_analyzer.analyze(road_info, lane_info)

            visualization = self.visualizer.create_visualization(
                processed_frame, road_info, lane_info, direction_info,
                is_video=True
            )

            processing_time = time.time() - start_time

            result = {
                'road_info': road_info,
                'lane_info': lane_info,
                'direction_info': direction_info,
                'processing_time': processing_time,
                'light_condition': light_condition
            }

            return processed_frame, result

        except Exception as e:
            print(f"帧检测失败: {e}")
            return None

    def update_config(self, new_config: AppConfig):
        """更新配置并重新初始化"""
        self.config = new_config
        self._init_modules()
        print("检测服务配置已更新")
