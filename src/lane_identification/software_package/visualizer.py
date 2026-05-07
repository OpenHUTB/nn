"""
可视化模块 - 负责结果可视化显示
支持实时视频显示
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
from config import AppConfig


class Visualizer:
    """可视化引擎"""

    def __init__(self, config: AppConfig):
        self.config = config
        self._setup_colors()
        self._setup_font()

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
            # 尝试加载 Windows 系统自带中文字体
            self.font_large = ImageFont.truetype("msyh.ttc", 28)
            self.font_medium = ImageFont.truetype("msyh.ttc", 20)
            self.font_small = ImageFont.truetype("msyh.ttc", 16)
        except IOError:
            try:
                self.font_large = ImageFont.truetype("simhei.ttf", 28)
                self.font_medium = ImageFont.truetype("simhei.ttf", 20)
                self.font_small = ImageFont.truetype("simhei.ttf", 16)
            except IOError:
                # 备用方案：使用默认字体
                self.font_large = ImageFont.load_default()
                self.font_medium = ImageFont.load_default()
                self.font_small = ImageFont.load_default()

    def _put_chinese_text(self, image: np.ndarray, text: str, position: Tuple[int, int],
                          color: Tuple[int, int, int], font_size: str = 'medium',
                          font_scale: float = 1.0) -> np.ndarray:
        """在图像上绘制中文文本"""
        # 选择字体并根据比例缩放
        base_font_size = {'large': 28, 'medium': 20, 'small': 16}[font_size]
        scaled_font_size = int(base_font_size * font_scale)
        scaled_font_size = max(8, scaled_font_size)  # 最小字体8px

        # 加载对应大小的字体
        try:
            if font_size == 'large':
                font = ImageFont.truetype("msyh.ttc", scaled_font_size)
            elif font_size == 'small':
                font = ImageFont.truetype("msyh.ttc", scaled_font_size)
            else:
                font = ImageFont.truetype("msyh.ttc", scaled_font_size)
        except IOError:
            try:
                font = ImageFont.truetype("simhei.ttf", scaled_font_size)
            except IOError:
                font = ImageFont.load_default()

        # 转换为 PIL Image
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 绘制文本
        draw.text(position, text, fill=color[::-1], font=font)

        # 转回 OpenCV 格式
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

            # 计算缩放比例（以1200x800为基准）
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

            # 4. 绘制信息面板
            visualization = self._draw_info_panel(visualization, direction_info, lane_info,
                                                   is_video, frame_info, scale_factor, width, height)

            # 5. 绘制方向指示器
            visualization = self._draw_direction_indicator(visualization, direction_info, scale_factor, width, height)

            # 6. 绘制图例（新增）
            visualization = self._draw_legend(visualization, lane_info, scale_factor, width, height)

            # 7. 应用全局效果
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

        # 创建道路图层
        road_layer = image.copy()

        # 填充道路区域
        cv2.drawContours(road_layer, [contour], -1, self.colors['road_area'][:3], -1)

        # 绘制道路边界（根据缩放调整线宽）
        boundary_thickness = max(1, int(2 * scale_factor))
        cv2.drawContours(road_layer, [contour], -1, self.colors['road_boundary'][:3], boundary_thickness)

        # 混合图层
        alpha = self.colors['road_area'][3] / 255.0
        cv2.addWeighted(road_layer, alpha, image, 1 - alpha, 0, image)

        return image

    def _draw_lanes(self, image: np.ndarray, lane_info: Dict[str, Any],
                    scale_factor: float = 1.0) -> np.ndarray:
        """绘制车道线 - 支持多车道显示"""
        lane_layer = image.copy()

        # 1. 绘制所有原始检测线段（极淡，避免视觉干扰）
        if scale_factor > 0.6:  # 只在较大图片上显示原始线段
            for side in ['left_lines', 'right_lines']:
                lines = lane_info.get(side, [])

                for line in lines:
                    points = line.get('points', [])
                    if len(points) == 2:
                        line_thickness = max(1, int(1 * scale_factor))
                        cv2.line(lane_layer, points[0], points[1], (100, 100, 100), line_thickness, cv2.LINE_AA)

        # 2. 绘制邻车道线（黄色虚线）
        for side in ['neighbor_left_lines', 'neighbor_right_lines']:
            lines = lane_info.get(side, [])

            for line in lines:
                points = line.get('points', [])
                if len(points) == 2:
                    line_thickness = max(1, int(2 * scale_factor))
                    cv2.line(lane_layer, points[0], points[1],
                             self.colors['neighbor_lane'][:3], line_thickness, cv2.LINE_AA)

        # 3. 绘制主车道边界线（加粗高亮，增强对比度）
        primary_left_lines = lane_info.get('primary_left_lines', [])
        primary_right_lines = lane_info.get('primary_right_lines', [])

        for line in primary_left_lines:
            points = line.get('points', [])
            if len(points) == 2:
                line_thickness = max(2, int(4 * scale_factor))
                cv2.line(lane_layer, points[0], points[1],
                         self.colors['left_lane'][:3], line_thickness, cv2.LINE_AA)

        for line in primary_right_lines:
            points = line.get('points', [])
            if len(points) == 2:
                line_thickness = max(2, int(4 * scale_factor))
                cv2.line(lane_layer, points[0], points[1],
                         self.colors['right_lane'][:3], line_thickness, cv2.LINE_AA)

        # 4. 绘制拟合的主车道线（更高亮）
        for side, color_key in [('left_lane', 'left_lane'), ('right_lane', 'right_lane')]:
            lane = lane_info.get(side)
            if lane and 'points' in lane and len(lane['points']) == 2:
                points = lane['points']
                color = self.colors[color_key]

                confidence = lane.get('confidence', 0.5)
                thickness = max(3, int((4 + int(confidence * 3)) * scale_factor))

                cv2.line(lane_layer, points[0], points[1], color[:3], thickness, cv2.LINE_AA)

        # 5. 绘制中心线（虚线样式，降低干扰）
        center_line = lane_info.get('center_line')
        if center_line and 'points' in center_line and len(center_line['points']) == 2:
            points = center_line['points']
            color = self.colors['center_line']
            line_thickness = max(1, int(2 * scale_factor))
            # 绘制虚线效果
            self._draw_dashed_line(lane_layer, points[0], points[1], color[:3],
                                   line_thickness, int(10 * scale_factor))

        # 混合车道线图层
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

        for i in range(num_dashes):
            t1 = i / num_dashes
            t2 = (i + 0.5) / num_dashes

            if i % 2 == 0:  # 只绘制偶数段
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
        """绘制信息面板"""
        if width == 0 or height == 0:
            height, width = image.shape[:2]

        # 根据缩放比例调整面板高度
        base_panel_height = 140
        panel_height = max(80, int(base_panel_height * scale_factor))

        # 创建半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # 获取信息
        direction = direction_info.get('direction', '未知')
        confidence = direction_info.get('confidence', 0.0)
        quality = lane_info.get('detection_quality', 0.0)

        # 设置颜色
        confidence_color = self._get_confidence_color(confidence)

        # 计算文本位置和间距
        base_spacing = 35
        text_spacing = max(20, int(base_spacing * scale_factor))
        left_margin = max(10, int(20 * scale_factor))
        right_margin = max(10, int(20 * scale_factor))

        # 1. 方向
        direction_text = f"方向: {direction}"
        image = self._put_chinese_text(image, direction_text,
                                       (left_margin, max(5, int(10 * scale_factor))),
                                       confidence_color, 'large', scale_factor)

        # 2. 置信度
        confidence_text = f"置信度: {confidence:.1%}"
        image = self._put_chinese_text(image, confidence_text,
                                       (left_margin, max(5, int(10 * scale_factor)) + text_spacing),
                                       confidence_color, 'medium', scale_factor)

        # 3. 检测质量
        quality_text = f"检测质量: {quality:.1%}"
        image = self._put_chinese_text(image, quality_text,
                                       (left_margin, max(5, int(10 * scale_factor)) + text_spacing * 2),
                                       self.colors['text_secondary'], 'small', scale_factor)

        # 4. 车道统计信息（新增）
        lane_stats = lane_info.get('lane_statistics', {})
        if lane_stats:
            total_lines = lane_stats.get('total_detected_lines', 0)
            estimated_lanes = lane_stats.get('estimated_lanes', 1)
            is_multi = lane_stats.get('is_multi_lane', False)

            stats_text = f"检测到{total_lines}条线 | 估算{estimated_lanes}车道"
            if is_multi:
                stats_text += " [多车道]"

            image = self._put_chinese_text(image, stats_text,
                                           (left_margin, max(5, int(10 * scale_factor)) + text_spacing * 3),
                                           self.colors['text_secondary'], 'small', scale_factor)

        # 5. 视频信息
        if is_video and frame_info:
            fps_text = f"FPS: {frame_info.get('fps', 0):.1f}"
            frame_text = f"帧: {frame_info.get('frame_number', 0)}"

            image = self._put_chinese_text(image, fps_text,
                                           (width - right_margin - 100, max(5, int(10 * scale_factor))),
                                           self.colors['text_primary'], 'small', scale_factor)
            image = self._put_chinese_text(image, frame_text,
                                           (width - right_margin - 100, max(5, int(10 * scale_factor)) + text_spacing),
                                           self.colors['text_primary'], 'small', scale_factor)

        # 6. 概率分布
        if 'probabilities' in direction_info:
            probabilities = direction_info['probabilities']
            start_x = width - right_margin - 100
            start_y = max(5, int(10 * scale_factor)) + text_spacing * 2 if is_video else max(5, int(10 * scale_factor))

            for i, (dir_name, prob) in enumerate(probabilities.items()):
                y = start_y + i * text_spacing
                prob_text = f"{dir_name}: {prob:.1%}"

                color = self.colors['text_primary'] if dir_name == direction else self.colors['text_secondary']

                image = self._put_chinese_text(image, prob_text, (start_x, y), color, 'small', scale_factor)

        return image

    def _draw_direction_indicator(self, image: np.ndarray,
                                  direction_info: Dict[str, Any],
                                  scale_factor: float = 1.0,
                                  width: int = 0, height: int = 0) -> np.ndarray:
        """绘制方向指示器"""
        if width == 0 or height == 0:
            height, width = image.shape[:2]

        direction = direction_info.get('direction', '未知')
        confidence = direction_info.get('confidence', 0.0)

        # 指示器位置 - 根据图片尺寸调整，但限制最小尺寸
        center_x = width // 2
        base_indicator_y = 100  # 距离底部的基准距离
        indicator_y = height - max(80, int(base_indicator_y * max(scale_factor, 0.8)))

        # 根据缩放调整箭头大小，设置更合理的范围
        arrow_scale = max(0.3, min(scale_factor, 1.5))  # 限制在0.3-1.5之间

        # 创建指示器图层
        indicator_layer = np.zeros_like(image)

        if direction == "左转":
            # 左转箭头 - 简化形状
            points = np.array([
                [center_x, indicator_y],
                [center_x - int(60 * arrow_scale), indicator_y],
                [center_x - int(50 * arrow_scale), indicator_y - int(30 * arrow_scale)],
                [center_x - int(80 * arrow_scale), indicator_y - int(30 * arrow_scale)],
                [center_x - int(100 * arrow_scale), indicator_y],
                [center_x - int(160 * arrow_scale), indicator_y],
                [center_x - int(80 * arrow_scale), indicator_y + int(60 * arrow_scale)],
                [center_x, indicator_y + int(60 * arrow_scale)]
            ], dtype=np.int32)
            base_color = (0, 165, 255)

        elif direction == "右转":
            # 右转箭头 - 简化形状
            points = np.array([
                [center_x, indicator_y],
                [center_x + int(60 * arrow_scale), indicator_y],
                [center_x + int(50 * arrow_scale), indicator_y - int(30 * arrow_scale)],
                [center_x + int(80 * arrow_scale), indicator_y - int(30 * arrow_scale)],
                [center_x + int(100 * arrow_scale), indicator_y],
                [center_x + int(160 * arrow_scale), indicator_y],
                [center_x + int(80 * arrow_scale), indicator_y + int(60 * arrow_scale)],
                [center_x, indicator_y + int(60 * arrow_scale)]
            ], dtype=np.int32)
            base_color = (0, 165, 255)

        else:  # 直行或未知
            # 直行箭头 - 简化形状
            points = np.array([
                [center_x - int(40 * arrow_scale), indicator_y + int(30 * arrow_scale)],
                [center_x, indicator_y - int(30 * arrow_scale)],
                [center_x + int(40 * arrow_scale), indicator_y + int(30 * arrow_scale)],
                [center_x + int(30 * arrow_scale), indicator_y + int(30 * arrow_scale)],
                [center_x + int(30 * arrow_scale), indicator_y + int(90 * arrow_scale)],
                [center_x - int(30 * arrow_scale), indicator_y + int(90 * arrow_scale)],
                [center_x - int(30 * arrow_scale), indicator_y + int(30 * arrow_scale)]
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
        border_thickness = max(1, int(2 * scale_factor))
        cv2.polylines(image, [points], True, (255, 255, 255), border_thickness, cv2.LINE_AA)

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
        """绘制图例说明"""
        lane_stats = lane_info.get('lane_statistics', {})
        if not lane_stats or lane_stats.get('total_detected_lines', 0) < 3:
            return image

        if width == 0 or height == 0:
            height, width = image.shape[:2]

        # 创建图例背景 - 根据缩放调整大小
        base_legend_width = 180
        base_legend_height = 90
        legend_width = max(100, int(base_legend_width * scale_factor))
        legend_height = max(60, int(base_legend_height * scale_factor))

        margin = max(5, int(10 * scale_factor))
        overlay = image.copy()
        cv2.rectangle(overlay, (width - legend_width - margin, height - legend_height - margin),
                      (width - margin, height - margin), (0, 0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # 图例文字
        legend_start_x = width - legend_width - margin + margin
        legend_start_y = height - legend_height - margin

        # 计算线条长度和间距
        line_length = max(20, int(30 * scale_factor))
        line_spacing = max(15, int(25 * scale_factor))
        line_thickness = max(1, int(2 * scale_factor))

        # 主车道
        cv2.line(image, (legend_start_x + margin, legend_start_y + line_spacing),
                 (legend_start_x + margin + line_length, legend_start_y + line_spacing),
                 (255, 100, 100), line_thickness)
        image = self._put_chinese_text(image, "主车道",
                                       (legend_start_x + margin + line_length + 10, legend_start_y + line_spacing - 10),
                                       self.colors['text_primary'], 'small', scale_factor)

        # 邻车道
        cv2.line(image, (legend_start_x + margin, legend_start_y + line_spacing * 2),
                 (legend_start_x + margin + line_length, legend_start_y + line_spacing * 2),
                 (255, 255, 0), line_thickness)
        image = self._put_chinese_text(image, "邻车道",
                                       (legend_start_x + margin + line_length + 10, legend_start_y + line_spacing * 2 - 10),
                                       self.colors['text_primary'], 'small', scale_factor)

        # 中心线
        cv2.line(image, (legend_start_x + margin, legend_start_y + line_spacing * 3),
                 (legend_start_x + margin + line_length, legend_start_y + line_spacing * 3),
                 (255, 255, 0), line_thickness)
        image = self._put_chinese_text(image, "中心线",
                                       (legend_start_x + margin + line_length + 10, legend_start_y + line_spacing * 3 - 10),
                                       self.colors['text_primary'], 'small', scale_factor)

        return image

    def _apply_global_effects(self, image: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
        """应用全局效果"""
        # 根据缩放调整锐化强度
        if scale_factor < 0.5:
            # 小图片不应用锐化，避免噪点
            return image

        # 轻微锐化
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)

        cv2.addWeighted(sharpened, 0.3, image, 0.7, 0, image)

        return image
