"""
性能可视化模块
负责绘制性能图表和可视化
作者: xiaoshiyuan888
"""

import cv2
import numpy as np
from collections import deque
import math


class PerformanceVisualizer:
    """性能可视化器"""

    def __init__(self, width=300, height=200):
        self.width = width
        self.height = height
        self.colors = {
            'fps': (0, 255, 0),
            'cpu': (255, 165, 0),
            'memory': (0, 165, 255),
            'recognition': (255, 0, 255),
            'background': (30, 30, 30),
            'grid': (60, 60, 60),
            'text': (200, 200, 200),
            'warning': (255, 165, 0),
            'critical': (255, 0, 0),
            'good': (0, 255, 0)
        }

    def draw_performance_chart(self, frame, data_history, title="", x_offset=0, y_offset=0,
                               max_value=None, color=None, unit=""):
        """绘制单个性能图表"""
        if not data_history or len(data_history) < 2:
            return frame

        # 创建图表画布
        chart_height = 80
        chart_width = 200

        # 绘制图表背景
        cv2.rectangle(frame,
                      (x_offset, y_offset),
                      (x_offset + chart_width, y_offset + chart_height),
                      self.colors['background'], -1)

        cv2.rectangle(frame,
                      (x_offset, y_offset),
                      (x_offset + chart_width, y_offset + chart_height),
                      self.colors['grid'], 1)

        # 绘制标题
        cv2.putText(frame, title, (x_offset + 5, y_offset + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)

        # 获取数据
        values = list(data_history)[-min(20, len(data_history)):]
        if not values:
            return frame

        # 计算最大值
        if max_value is None:
            data_max = max(values)
        else:
            data_max = max_value

        data_min = min(values)

        if data_max == data_min:
            data_max = data_min + 1

        # 绘制网格线
        for i in range(1, 4):
            grid_y = y_offset + chart_height - 20 - i * (chart_height - 40) // 4
            cv2.line(frame,
                     (x_offset + 10, grid_y),
                     (x_offset + chart_width - 10, grid_y),
                     self.colors['grid'], 1)

            # 绘制刻度值
            tick_value = data_min + (data_max - data_min) * i / 4
            tick_text = f"{tick_value:.0f}{unit}"
            cv2.putText(frame, tick_text,
                        (x_offset + chart_width - 35, grid_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)

        # 绘制数据曲线
        points = []
        for i, value in enumerate(values):
            point_x = x_offset + 10 + i * (chart_width - 20) / (len(values) - 1) if len(values) > 1 else x_offset + 10
            point_y = y_offset + chart_height - 20 - (value - data_min) * (chart_height - 40) / (data_max - data_min)
            points.append((int(point_x), int(point_y)))

        # 连接点形成曲线
        chart_color = color if color else self.colors['fps']
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], chart_color, 2)

        # 绘制当前值
        if values:
            current_value = values[-1]
            current_text = f"当前: {current_value:.1f}{unit}"
            cv2.putText(frame, current_text,
                        (x_offset + 10, y_offset + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, chart_color, 1)

        return frame

    def draw_comprehensive_charts(self, frame, performance_analyzer):
        """绘制综合性能图表"""
        x_start = frame.shape[1] - 250
        y_start = 150
        chart_spacing = 90

        # FPS图表
        frame = self.draw_performance_chart(
            frame, performance_analyzer.fps_history,
            "FPS监控", x_start, y_start,
            max_value=60, color=self.colors['fps'], unit="FPS"
        )

        # CPU图表
        frame = self.draw_performance_chart(
            frame, performance_analyzer.cpu_usage_history,
            "CPU使用率", x_start, y_start + chart_spacing,
            max_value=100, color=self.colors['cpu'], unit="%"
        )

        # 内存图表
        frame = self.draw_performance_chart(
            frame, performance_analyzer.memory_usage_history,
            "内存使用率", x_start, y_start + chart_spacing * 2,
            max_value=100, color=self.colors['memory'], unit="%"
        )

        # 识别时间图表
        if len(performance_analyzer.gesture_recognition_times) > 1:
            frame = self.draw_performance_chart(
                frame, performance_analyzer.gesture_recognition_times,
                "识别时间", x_start, y_start + chart_spacing * 3,
                max_value=100, color=self.colors['recognition'], unit="ms"
            )

        return frame

    def draw_performance_gauge(self, frame, value, max_value, label, x_offset, y_offset, color=None):
        """绘制性能仪表盘"""
        gauge_radius = 30
        center_x = x_offset + gauge_radius + 5
        center_y = y_offset + gauge_radius + 5

        # 绘制仪表盘背景
        cv2.circle(frame, (center_x, center_y), gauge_radius, self.colors['background'], -1)
        cv2.circle(frame, (center_x, center_y), gauge_radius, self.colors['grid'], 2)

        # 计算角度
        angle = 180 + (value / max_value) * 180

        # 绘制指针
        end_x = center_x + int(gauge_radius * 0.7 * math.cos(math.radians(angle)))
        end_y = center_y + int(gauge_radius * 0.7 * math.sin(math.radians(angle)))

        gauge_color = color if color else self.colors['good']
        cv2.line(frame, (center_x, center_y), (end_x, end_y), gauge_color, 3)

        # 绘制中心点
        cv2.circle(frame, (center_x, center_y), 5, gauge_color, -1)

        # 绘制标签和值
        cv2.putText(frame, label, (x_offset, y_offset + gauge_radius * 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        cv2.putText(frame, f"{value:.0f}", (center_x - 15, center_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gauge_color, 1)

        return frame

    def draw_performance_gauges(self, frame, performance_analyzer):
        """绘制性能仪表盘组"""
        x_start = frame.shape[1] - 180
        y_start = 150
        gauge_spacing = 80

        # CPU仪表盘
        cpu_usage = performance_analyzer.get_current_cpu_usage()
        cpu_color = self.colors['good']
        if cpu_usage > 80:
            cpu_color = self.colors['critical']
        elif cpu_usage > 60:
            cpu_color = self.colors['warning']

        frame = self.draw_performance_gauge(
            frame, cpu_usage, 100, "CPU", x_start, y_start, cpu_color
        )

        # 内存仪表盘
        memory_usage = performance_analyzer.get_current_memory_usage()
        memory_color = self.colors['good']
        if memory_usage > 85:
            memory_color = self.colors['critical']
        elif memory_usage > 70:
            memory_color = self.colors['warning']

        frame = self.draw_performance_gauge(
            frame, memory_usage, 100, "内存", x_start, y_start + gauge_spacing, memory_color
        )

        # FPS仪表盘
        fps = performance_analyzer.get_current_fps()
        fps_color = self.colors['good']
        if fps < 15:
            fps_color = self.colors['critical']
        elif fps < 25:
            fps_color = self.colors['warning']

        frame = self.draw_performance_gauge(
            frame, min(fps, 60), 60, "FPS", x_start, y_start + gauge_spacing * 2, fps_color
        )

        # 性能评分仪表盘
        performance_score = performance_analyzer.performance_score
        score_color = self.colors['good']
        if performance_score < 40:
            score_color = self.colors['critical']
        elif performance_score < 70:
            score_color = self.colors['warning']

        frame = self.draw_performance_gauge(
            frame, performance_score, 100, "评分", x_start, y_start + gauge_spacing * 3, score_color
        )

        return frame