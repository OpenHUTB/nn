"""
可视化模块 - 负责结果可视化显示
支持实时视频显示
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from PIL import Image, ImageDraw, ImageFont
from config import AppConfig


class Visualizer:
    """可视化引擎"""

    def __init__(self, config: AppConfig):
        self.config = config
        self._setup_colors()
        self._setup_font()
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}

    def _setup_colors(self):
        """设置颜色方案"""
        self.colors = {
            # 道路相关
            'road_area': (0, 180, 0, 100),
            'road_boundary': (0, 255, 255, 200),

            # 车道线 - 主车道（高亮）
            'left_lane': (255, 100, 100, 200),
            'right_lane': (100, 100, 255, 200),

            # 邻车道（黄色）
            'neighbor_lane': (255, 255, 0, 150),

            # 中心线
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

    def _setup_font(self):
        """设置中文字体"""
        try:
            self.font_large = ImageFont.truetype("msyh.ttc", 28)
            self.font_medium = ImageFont.truetype("msyh.ttc", 20)
            self.font_small = ImageFont.truetype("msyh.ttc", 16)
            self._font_family = "msyh.ttc"
        except IOError:
            try:
                self.font_large = ImageFont.truetype("simhei.ttf", 28)
                self.font_medium = ImageFont.truetype("simhei.ttf", 20)
                self.font_small = ImageFont.truetype("simhei.ttf", 16)
                self._font_family = "simhei.ttf"
            except IOError:
                self.font_large = ImageFont.load_default()
                self.font_medium = ImageFont.load_default()
                self.font_small = ImageFont.load_default()
                self._font_family = None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """获取指定大小的字体（带缓存）"""
        size = max(8, size)
        if size not in self._font_cache:
            if self._font_family:
                try:
                    self._font_cache[size] = ImageFont.truetype(self._font_family, size)
                except IOError:
                    self._font_cache[size] = ImageFont.load_default()
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def _get_base_font(self, font_size: str, font_scale: float = 1.0) -> ImageFont.FreeTypeFont:
        """根据字体大小名称和缩放获取字体"""
        base_sizes = {'large': 28, 'medium': 20, 'small': 16}
        base = base_sizes.get(font_size, 20)
        scaled = max(8, int(base * font_scale))
        return self._get_font(scaled)

    def _batch_put_chinese_text(self, image: np.ndarray,
                                text_items: List[Tuple[Tuple[int, int], str, Tuple[int, int], str, float]]) -> np.ndarray:
        """批量绘制中文文本，只做一次 PIL 转换

        Args:
            text_items: [(position, text, color, font_size, font_scale), ...]
        """
        if not text_items:
            return image

        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        for position, text, color, font_size, font_scale in text_items:
            font = self._get_base_font(font_size, font_scale)
            draw.text(position, text, fill=color[::-1], font=font)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def create_visualization(self, image: np.ndarray,
                             road_info: Dict[str, Any],
                             lane_info: Dict[str, Any],
                             direction_info: Dict[str, Any],
                             is_video: bool = False,
                             frame_info: Dict[str, Any] = None) -> np.ndarray:
        """创建可视化结果"""
        try:
            visualization = image.copy()
            height, width = image.shape[:2]

            scale_x = width / 1200.0
            scale_y = height / 800.0
            scale_factor = min(scale_x, scale_y)

            # 1. 绘制道路区域
            if road_info.get('contour') is not None:
                visualization = self._draw_road_area(visualization, road_info, scale_factor)

            # 2. 绘制车道线
            visualization = self._draw_lanes(visualization, lane_info, scale_factor)

            # 3. 绘制路径预测
            if lane_info.get('future_path'):
                visualization = self._draw_future_path(visualization, lane_info['future_path'], scale_factor)

            # 4. 绘制信息面板（批量文字绘制）
            visualization = self._draw_info_panel(visualization, direction_info, lane_info,
                                                   is_video, frame_info, scale_factor, width, height)

            # 5. 绘制图例
            visualization = self._draw_legend(visualization, lane_info, scale_factor, width, height)

            # 6. 应用全局效果
            visualization = self._apply_global_effects(visualization, scale_factor)

            return visualization

        except Exception as e:
            print(f"可视化创建失败: {e}")
            return image

    def _draw_road_area(self, image: np.ndarray, road_info: Dict[str, Any],
                        scale_factor: float = 1.0) -> np.ndarray:
        """绘制道路区域"""
        contour = road_info['contour']
        if contour is None or len(contour) == 0:
            return image

        road_layer = image.copy()
        cv2.drawContours(road_layer, [contour], -1, self.colors['road_area'][:3], -1)

        boundary_thickness = max(1, int(2 * scale_factor))
        cv2.drawContours(road_layer, [contour], -1, self.colors['road_boundary'][:3], boundary_thickness)

        alpha = self.colors['road_area'][3] / 255.0
        cv2.addWeighted(road_layer, alpha, image, 1 - alpha, 0, image)

        return image

    def _draw_lanes(self, image: np.ndarray, lane_info: Dict[str, Any],
                    scale_factor: float = 1.0) -> np.ndarray:
        """绘制车道线 - 支持多车道显示"""
        lane_layer = image.copy()

        # 1. 绘制所有原始检测线段
        if scale_factor > 0.8:
            for side in ['left_lines', 'right_lines']:
                lines = lane_info.get(side, [])
                for line in lines:
                    points = line.get('points', [])
                    if len(points) == 2:
                        line_thickness = max(1, int(1 * scale_factor))
                        cv2.line(lane_layer, points[0], points[1], (80, 80, 80), line_thickness, cv2.LINE_AA)

        # 2. 绘制邻车道线（黄色虚线）
        for side in ['neighbor_left_lines', 'neighbor_right_lines']:
            lines = lane_info.get(side, [])
            for line in lines:
                points = line.get('points', [])
                if len(points) == 2:
                    line_thickness = max(1, int(2 * scale_factor))
                    cv2.line(lane_layer, points[0], points[1],
                             self.colors['neighbor_lane'][:3], line_thickness, cv2.LINE_AA)

        # 3. 绘制主车道边界线（加粗高亮）
        for side, color_key in [('primary_left_lines', 'left_lane'), ('primary_right_lines', 'right_lane')]:
            for line in lane_info.get(side, []):
                points = line.get('points', [])
                if len(points) == 2:
                    line_thickness = max(2, int(4 * scale_factor))
                    cv2.line(lane_layer, points[0], points[1],
                             self.colors[color_key][:3], line_thickness, cv2.LINE_AA)

        # 4. 绘制拟合的主车道线
        for side, color_key in [('left_lane', 'left_lane'), ('right_lane', 'right_lane')]:
            lane = lane_info.get(side)
            if lane and 'points' in lane and len(lane['points']) == 2:
                points = lane['points']
                color = self.colors[color_key]
                confidence = lane.get('confidence', 0.5)
                thickness = max(3, int((4 + int(confidence * 3)) * scale_factor))
                cv2.line(lane_layer, points[0], points[1], color[:3], thickness, cv2.LINE_AA)

        # 5. 绘制中心线（虚线样式）
        center_line = lane_info.get('center_line')
        if center_line and 'points' in center_line and len(center_line['points']) == 2:
            points = center_line['points']
            color = self.colors['center_line']
            line_thickness = max(1, int(2 * scale_factor))
            self._draw_dashed_line(lane_layer, points[0], points[1], color[:3],
                                   line_thickness, int(10 * scale_factor))

        cv2.addWeighted(lane_layer, 0.8, image, 0.2, 0, image)
        return image

    def _draw_dashed_line(self, image: np.ndarray, pt1: Tuple[int, int],
                          pt2: Tuple[int, int], color: Tuple[int, int, int],
                          thickness: int, dash_length: int = 10):
        """绘制虚线"""
        x1, y1 = pt1
        x2, y2 = pt2

        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx ** 2 + dy ** 2)

        if length == 0:
            return

        num_dashes = int(length / dash_length)
        if num_dashes < 2:
            cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)
            return

        for i in range(0, num_dashes, 2):
            t1 = i / num_dashes
            t2 = (i + 0.5) / num_dashes

            start_x = int(x1 + dx * t1)
            start_y = int(y1 + dy * t1)
            end_x = int(x1 + dx * t2)
            end_y = int(y1 + dy * t2)
            cv2.line(image, (start_x, start_y), (end_x, end_y),
                     color, thickness, cv2.LINE_AA)

    def _draw_future_path(self, image: np.ndarray, future_path: Dict[str, Any],
                          scale_factor: float = 1.0) -> np.ndarray:
        """绘制未来路径"""
        path_points = future_path.get('center_path', [])
        if len(path_points) < 2:
            return image

        path_layer = image.copy()
        color = self.colors['future_path']

        for i in range(len(path_points) - 1):
            alpha_factor = 0.5 + 0.5 * (i / (len(path_points) - 1))
            line_color = tuple(int(c * alpha_factor) for c in color[:3])
            thickness = max(2, int((5 - int(i / len(path_points) * 3)) * scale_factor))
            cv2.line(path_layer, path_points[i], path_points[i + 1],
                     line_color, thickness, cv2.LINE_AA)

        cv2.addWeighted(path_layer, 0.6, image, 0.4, 0, image)
        return image

    def _draw_info_panel(self, image: np.ndarray, direction_info: Dict[str, Any],
                         lane_info: Dict[str, Any], is_video: bool = False,
                         frame_info: Dict[str, Any] = None,
                         scale_factor: float = 1.0,
                         width: int = 0, height: int = 0) -> np.ndarray:
        """绘制信息面板（批量文字绘制，单次 PIL 转换）"""
        if width == 0 or height == 0:
            height, width = image.shape[:2]

        base_panel_height = 140
        panel_height = max(80, int(base_panel_height * scale_factor))

        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        direction = direction_info.get('direction', '未知')
        confidence = direction_info.get('confidence', 0.0)
        quality = lane_info.get('detection_quality', 0.0)

        confidence_color = self._get_confidence_color(confidence)

        base_spacing = 35
        text_spacing = max(20, int(base_spacing * scale_factor))
        left_margin = max(10, int(20 * scale_factor))
        right_margin = max(10, int(20 * scale_factor))
        top_offset = max(5, int(10 * scale_factor))

        # 收集所有文本项，批量绘制
        text_items: List[Tuple[Tuple[int, int], str, Tuple[int, int], str, float]] = []

        # 1. 方向
        text_items.append(((left_margin, top_offset),
                          f"方向: {direction}", confidence_color, 'large', scale_factor))

        # 2. 置信度
        text_items.append(((left_margin, top_offset + text_spacing),
                          f"置信度: {confidence:.1%}", confidence_color, 'medium', scale_factor))

        # 3. 检测质量
        text_items.append(((left_margin, top_offset + text_spacing * 2),
                          f"检测质量: {quality:.1%}", self.colors['text_secondary'], 'small', scale_factor))

        # 4. 车道统计信息
        lane_stats = lane_info.get('lane_statistics', {})
        if lane_stats:
            total_lines = lane_stats.get('total_detected_lines', 0)
            estimated_lanes = lane_stats.get('estimated_lanes', 1)
            is_multi = lane_stats.get('is_multi_lane', False)

            stats_text = f"检测到{total_lines}条线 | 估算{estimated_lanes}车道"
            if is_multi:
                stats_text += " [多车道]"

            text_items.append(((left_margin, top_offset + text_spacing * 3),
                              stats_text, self.colors['text_secondary'], 'small', scale_factor))

        # 5. 视频信息
        if is_video and frame_info:
            fps_text = f"FPS: {frame_info.get('fps', 0):.1f}"
            frame_text = f"帧: {frame_info.get('frame_number', 0)}"

            text_items.append(((width - right_margin - 100, top_offset),
                              fps_text, self.colors['text_primary'], 'small', scale_factor))
            text_items.append(((width - right_margin - 100, top_offset + text_spacing),
                              frame_text, self.colors['text_primary'], 'small', scale_factor))

        # 6. 概率分布
        if 'probabilities' in direction_info:
            probabilities = direction_info['probabilities']
            start_x = width - right_margin - 100
            start_y = top_offset + text_spacing * 2 if is_video else top_offset

            for i, (dir_name, prob) in enumerate(probabilities.items()):
                y = start_y + i * text_spacing
                color = self.colors['text_primary'] if dir_name == direction else self.colors['text_secondary']
                text_items.append(((start_x, y),
                                  f"{dir_name}: {prob:.1%}", color, 'small', scale_factor))

        # 批量绘制所有文字（单次 PIL 转换）
        image = self._batch_put_chinese_text(image, text_items)

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

    def _draw_legend(self, image: np.ndarray, lane_info: Dict[str, Any],
                     scale_factor: float = 1.0,
                     width: int = 0, height: int = 0) -> np.ndarray:
        """绘制图例说明（批量文字绘制）"""
        lane_stats = lane_info.get('lane_statistics', {})
        if not lane_stats or lane_stats.get('total_detected_lines', 0) < 3:
            return image

        if width == 0 or height == 0:
            height, width = image.shape[:2]

        base_legend_width = 180
        base_legend_height = 90
        legend_width = max(100, int(base_legend_width * scale_factor))
        legend_height = max(60, int(base_legend_height * scale_factor))

        margin = max(5, int(10 * scale_factor))
        overlay = image.copy()
        cv2.rectangle(overlay, (width - legend_width - margin, height - legend_height - margin),
                      (width - margin, height - margin), (0, 0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        legend_start_x = width - legend_width - margin + margin
        legend_start_y = height - legend_height - margin

        line_length = max(20, int(30 * scale_factor))
        line_spacing = max(15, int(25 * scale_factor))
        line_thickness = max(1, int(2 * scale_factor))

        # 绘制图例线条
        legend_items = [
            (1, "主车道", (255, 100, 100)),
            (2, "邻车道", (255, 255, 0)),
            (3, "中心线", (255, 255, 0)),
        ]

        for multiplier, label, color in legend_items:
            y = legend_start_y + line_spacing * multiplier
            cv2.line(image, (legend_start_x + margin, y),
                     (legend_start_x + margin + line_length, y),
                     color, line_thickness)

        # 批量绘制图例文字
        text_items = []
        for multiplier, label, _ in legend_items:
            y = legend_start_y + line_spacing * multiplier
            text_items.append((
                (legend_start_x + margin + line_length + 10, y - 10),
                label, self.colors['text_primary'], 'small', scale_factor
            ))

        image = self._batch_put_chinese_text(image, text_items)

        return image

    def _apply_global_effects(self, image: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
        """应用全局效果"""
        if scale_factor < 0.5:
            return image

        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        cv2.addWeighted(sharpened, 0.3, image, 0.7, 0, image)

        return image
