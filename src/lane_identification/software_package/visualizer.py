"""
可视化模块 - 负责结果可视化显示
支持实时视频显示
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from config import AppConfig

class Visualizer:
    """可视化引擎"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._setup_colors()
    
    def _setup_colors(self):
        """设置颜色方案"""
        self.colors = {
            # 道路相关
            'road_area': (0, 180, 0, 100),
            'road_boundary': (0, 255, 255, 200),
            
            # 车道线
            'left_lane': (255, 100, 100, 200),
            'right_lane': (100, 100, 255, 200),
            'center_line': (255, 255, 0, 180),
            
            # 路径预测
            'future_path': (255, 0, 255, 180),
            'prediction_points': (255, 150, 255, 220),
            
            # 置信度颜色
            'confidence_high': (0, 255, 0),
            'confidence_medium': (255, 165, 0),
            'confidence_low': (255, 0, 0),
            'confidence_very_low': (128, 128, 128),
            
            # 文本颜色
            'text_primary': (255, 255, 255),
            'text_secondary': (200, 200, 200),
            
            # 状态指示器
            'status_active': (0, 255, 0),
            'status_paused': (255, 165, 0),
            'status_stopped': (255, 0, 0)
        }
    
    def create_visualization(self, image: np.ndarray, 
                           road_info: Dict[str, Any],
                           lane_info: Dict[str, Any], 
                           direction_info: Dict[str, Any],
                           is_video: bool = False,
                           frame_info: Dict[str, Any] = None) -> np.ndarray:
        """创建可视化结果"""
        try:
            # 创建副本
            visualization = image.copy()
            
            # 1. 绘制道路区域
            if road_info.get('contour') is not None:
                visualization = self._draw_road_area(visualization, road_info)
            
            # 2. 绘制车道线
            visualization = self._draw_lanes(visualization, lane_info)
            
            # 3. 绘制路径预测
            if lane_info.get('future_path'):
                visualization = self._draw_future_path(visualization, lane_info['future_path'])
            
            # 4. 绘制信息面板
            visualization = self._draw_info_panel(visualization, direction_info, lane_info, is_video, frame_info)
            
            # 5. 绘制方向指示器
            visualization = self._draw_direction_indicator(visualization, direction_info)
            
            # 6. 应用全局效果
            visualization = self._apply_global_effects(visualization)
            
            return visualization
            
        except Exception as e:
            print(f"可视化创建失败: {e}")
            return image
    
    def _draw_road_area(self, image: np.ndarray, road_info: Dict[str, Any]) -> np.ndarray:
        """绘制道路区域"""
        contour = road_info['contour']
        if contour is None or len(contour) == 0:
            return image
        
        # 创建道路图层
        road_layer = image.copy()
        
        # 填充道路区域
        cv2.drawContours(road_layer, [contour], -1, self.colors['road_area'][:3], -1)
        
        # 绘制道路边界
        cv2.drawContours(road_layer, [contour], -1, self.colors['road_boundary'][:3], 2)
        
        # 混合图层
        alpha = self.colors['road_area'][3] / 255.0
        cv2.addWeighted(road_layer, alpha, image, 1 - alpha, 0, image)
        
        return image
    
    def _draw_lanes(self, image: np.ndarray, lane_info: Dict[str, Any]) -> np.ndarray:
        """绘制车道线"""
        lane_layer = image.copy()
        
        # 绘制原始检测线段
        for side, color_key in [('left_lines', 'left_lane'), ('right_lines', 'right_lane')]:
            lines = lane_info.get(side, [])
            color = self.colors[color_key]
            
            for line in lines:
                points = line.get('points', [])
                if len(points) == 2:
                    cv2.line(lane_layer, points[0], points[1], color[:3], 1, cv2.LINE_AA)
        
        # 绘制拟合的车道线
        for side, color_key in [('left_lane', 'left_lane'), ('right_lane', 'right_lane')]:
            lane = lane_info.get(side)
            if lane and 'points' in lane and len(lane['points']) == 2:
                points = lane['points']
                color = self.colors[color_key]
                
                confidence = lane.get('confidence', 0.5)
                thickness = 2 + int(confidence * 4)
                
                cv2.line(lane_layer, points[0], points[1], color[:3], thickness, cv2.LINE_AA)
        
        # 绘制中心线
        center_line = lane_info.get('center_line')
        if center_line and 'points' in center_line and len(center_line['points']) == 2:
            points = center_line['points']
            color = self.colors['center_line']
            cv2.line(lane_layer, points[0], points[1], color[:3], 2, cv2.LINE_AA)
        
        # 混合车道线图层
        cv2.addWeighted(lane_layer, 0.7, image, 0.3, 0, image)
        
        return image
    
    def _draw_future_path(self, image: np.ndarray, future_path: Dict[str, Any]) -> np.ndarray:
        """绘制未来路径"""
        path_points = future_path.get('center_path', [])
        if len(path_points) < 2:
            return image
        
        path_layer = image.copy()
        color = self.colors['future_path']
        
        for i in range(len(path_points) - 1):
            alpha_factor = 0.5 + 0.5 * (i / (len(path_points) - 1))
            line_color = tuple(int(c * alpha_factor) for c in color[:3])
            
            thickness = 5 - int(i / len(path_points) * 3)
            
            cv2.line(path_layer, path_points[i], path_points[i + 1], 
                    line_color, thickness, cv2.LINE_AA)
        
        cv2.addWeighted(path_layer, 0.6, image, 0.4, 0, image)
        
        return image
    
    def _draw_info_panel(self, image: np.ndarray, direction_info: Dict[str, Any],
                        lane_info: Dict[str, Any], is_video: bool = False,
                        frame_info: Dict[str, Any] = None) -> np.ndarray:
        """绘制信息面板"""
        height, width = image.shape[:2]
        
        # 创建半透明背景
        panel_height = 120
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 获取信息
        direction = direction_info.get('direction', '未知')
        confidence = direction_info.get('confidence', 0.0)
        quality = lane_info.get('detection_quality', 0.0)
        
        # 设置颜色
        confidence_color = self._get_confidence_color(confidence)
        
        # 绘制方向信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 1. 方向
        direction_text = f"方向: {direction}"
        cv2.putText(image, direction_text, (20, 35), 
                   font, 1.2, confidence_color, 2)
        
        # 2. 置信度
        confidence_text = f"置信度: {confidence:.1%}"
        cv2.putText(image, confidence_text, (20, 70), 
                   font, 0.9, confidence_color, 2)
        
        # 3. 检测质量
        quality_text = f"检测质量: {quality:.1%}"
        cv2.putText(image, quality_text, (20, 100), 
                   font, 0.7, self.colors['text_secondary'], 1)
        
        # 4. 视频信息
        if is_video and frame_info:
            fps_text = f"FPS: {frame_info.get('fps', 0):.1f}"
            frame_text = f"帧: {frame_info.get('frame_number', 0)}"
            
            cv2.putText(image, fps_text, (width - 200, 35), 
                       font, 0.7, self.colors['text_primary'], 1)
            cv2.putText(image, frame_text, (width - 200, 65), 
                       font, 0.7, self.colors['text_primary'], 1)
        
        # 5. 概率分布
        if 'probabilities' in direction_info:
            probabilities = direction_info['probabilities']
            start_x = width - 200
            start_y = 95 if is_video else 30
            
            for i, (dir_name, prob) in enumerate(probabilities.items()):
                y = start_y + i * 25
                prob_text = f"{dir_name}: {prob:.1%}"
                
                color = self.colors['text_primary'] if dir_name == direction else self.colors['text_secondary']
                
                cv2.putText(image, prob_text, (start_x, y), 
                           font, 0.7, color, 1)
        
        return image
    
    def _draw_direction_indicator(self, image: np.ndarray, 
                                direction_info: Dict[str, Any]) -> np.ndarray:
        """绘制方向指示器"""
        height, width = image.shape[:2]
        direction = direction_info.get('direction', '未知')
        confidence = direction_info.get('confidence', 0.0)
        
        # 指示器位置
        center_x = width // 2
        indicator_y = height - 150
        
        # 创建指示器图层
        indicator_layer = np.zeros_like(image)
        
        if direction == "左转":
            # 左转箭头
            points = np.array([
                [center_x, indicator_y],
                [center_x - 80, indicator_y],
                [center_x - 60, indicator_y - 40],
                [center_x - 100, indicator_y - 40],
                [center_x - 120, indicator_y],
                [center_x - 200, indicator_y],
                [center_x - 100, indicator_y + 80],
                [center_x, indicator_y + 80]
            ], dtype=np.int32)
            base_color = (0, 165, 255)
            
        elif direction == "右转":
            # 右转箭头
            points = np.array([
                [center_x, indicator_y],
                [center_x + 80, indicator_y],
                [center_x + 60, indicator_y - 40],
                [center_x + 100, indicator_y - 40],
                [center_x + 120, indicator_y],
                [center_x + 200, indicator_y],
                [center_x + 100, indicator_y + 80],
                [center_x, indicator_y + 80]
            ], dtype=np.int32)
            base_color = (0, 165, 255)
            
        else:  # 直行或未知
            # 直行箭头
            points = np.array([
                [center_x - 60, indicator_y + 40],
                [center_x, indicator_y - 40],
                [center_x + 60, indicator_y + 40],
                [center_x + 40, indicator_y + 40],
                [center_x + 40, indicator_y + 120],
                [center_x - 40, indicator_y + 120],
                [center_x - 40, indicator_y + 40]
            ], dtype=np.int32)
            base_color = (0, 255, 0)
        
        # 根据置信度调整颜色亮度
        brightness_factor = 0.5 + confidence * 0.5
        color = tuple(int(c * brightness_factor) for c in base_color)
        
        # 绘制指示器
        cv2.fillPoly(indicator_layer, [points], color)
        
        alpha = 0.3 + confidence * 0.5
        cv2.addWeighted(indicator_layer, alpha, image, 1 - alpha, 0, image)
        
        # 绘制边框
        cv2.polylines(image, [points], True, (255, 255, 255), 2, cv2.LINE_AA)
        
        return image
    
    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """根据置信度获取颜色"""
        if confidence >= 0.8:
            return self.colors['confidence_high']
        elif confidence >= 0.6:
            return self.colors['confidence_medium']
        elif confidence >= 0.4:
            return self.colors['confidence_low']
        else:
            return self.colors['confidence_very_low']
    
    def _apply_global_effects(self, image: np.ndarray) -> np.ndarray:
        """应用全局效果"""
        # 轻微锐化
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        cv2.addWeighted(sharpened, 0.3, image, 0.7, 0, image)
        
        return image