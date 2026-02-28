"""
用户界面渲染器模块
负责绘制中文界面和可视化元素
作者: xiaoshiyuan888
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ChineseUIRenderer:
    """中文UI渲染器"""

    def __init__(self, speech_manager=None, config=None):
        self.fonts = {}
        self.speech_manager = speech_manager
        self.config = config
        self.load_fonts()

        # 颜色定义
        self.colors = {
            'title': (0, 255, 255),
            'connected': (0, 255, 0),
            'disconnected': (0, 0, 255),
            'flying': (0, 255, 0),
            'landed': (255, 165, 0),
            'warning': (0, 165, 255),
            'info': (255, 255, 255),
            'help': (255, 200, 100),
            'speech_enabled': (0, 255, 0),
            'speech_disabled': (255, 0, 0),
            'performance_good': (0, 255, 0),
            'performance_warning': (255, 165, 0),
            'performance_bad': (255, 0, 0),
            'recording': (255, 50, 50),
            'playback': (50, 50, 255),
            'paused': (255, 255, 0),
            'performance_fast': (0, 255, 0),
            'performance_balanced': (255, 165, 0),
            'performance_accurate': (255, 0, 0),
            'cpu_good': (0, 255, 0),
            'cpu_warning': (255, 165, 0),
            'cpu_critical': (255, 0, 0),
            'memory_good': (0, 255, 0),
            'memory_warning': (255, 165, 0),
            'memory_critical': (255, 0, 0),
            # 新增手势颜色
            'gesture_grab': (255, 0, 255),
            'gesture_rotate_cw': (0, 255, 255),
            'gesture_rotate_ccw': (255, 255, 0),
            'gesture_photo': (255, 165, 0),
            'gesture_return_home': (0, 128, 128),
            'gesture_auto_flight': (128, 0, 128),
            # 飞行模式颜色
            'flight_mode_manual': (0, 255, 0),
            'flight_mode_auto': (255, 165, 0),
            'flight_mode_circle': (0, 255, 255),
            'flight_mode_eight': (255, 0, 255),
            'flight_mode_square': (255, 255, 0),
            'flight_mode_return_home': (0, 128, 128),
            # 性能图表颜色
            'chart_fps': (0, 255, 0),
            'chart_cpu': (255, 165, 0),
            'chart_memory': (0, 165, 255),
            'chart_recognition': (255, 0, 255),
            'chart_background': (30, 30, 30),
            'chart_grid': (60, 60, 60),
            'chart_text': (200, 200, 200),
        }

        print("✓ 中文UI渲染器已初始化")

    def load_fonts(self):
        """加载字体"""
        font_paths = [
            'simhei.ttf',
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            '/System/Library/Fonts/PingFang.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        ]

        for path in font_paths:
            try:
                self.fonts[14] = ImageFont.truetype(path, 14)
                self.fonts[16] = ImageFont.truetype(path, 16)
                self.fonts[18] = ImageFont.truetype(path, 18)
                self.fonts[20] = ImageFont.truetype(path, 20)
                self.fonts[24] = ImageFont.truetype(path, 24)
                print(f"✓ 字体已加载: {path}")
                return
            except:
                continue

        print("⚠ 未找到字体，使用默认")

    def draw_text(self, frame, text, pos, size=16, color=(255, 255, 255)):
        """在图像上绘制文本"""
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)

            font = self.fonts.get(size, self.fonts.get(16))

            # 绘制阴影
            shadow_color = (0, 0, 0)
            shadow_pos = (pos[0] + 1, pos[1] + 1)
            draw.text(shadow_pos, text, font=font, fill=shadow_color)

            # 绘制文字
            rgb_color = color[::-1]  # BGR to RGB
            draw.text(pos, text, font=font, fill=rgb_color)

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except:
            # 备用方案
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        size / 25, color, 1)
            return frame

    def draw_status_bar(self, frame, drone_controller, gesture, confidence, fps, process_time,
                        trajectory_recorder=None, gesture_recognizer=None, performance_analyzer=None):
        """绘制状态栏（增强版）"""
        h, w = frame.shape[:2]

        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # 标题
        title = "手势控制无人机系统 - 增强性能监控版"
        frame = self.draw_text(frame, title, (10, 10), size=20, color=self.colors['title'])

        # 连接状态
        status_color = self.colors['connected'] if drone_controller.connected else self.colors['disconnected']
        status_text = f"无人机: {'已连接' if drone_controller.connected else '未连接'}"
        frame = self.draw_text(frame, status_text, (10, 40), size=16, color=status_color)

        # 飞行状态
        flight_color = self.colors['flying'] if drone_controller.flying else self.colors['landed']
        flight_text = f"飞行状态: {'飞行中' if drone_controller.flying else '已降落'}"
        frame = self.draw_text(frame, flight_text, (10, 65), size=16, color=flight_color)

        # 手势信息
        if confidence > 0.7:
            gesture_color = (0, 255, 0)
        elif confidence > 0.5:
            gesture_color = (255, 165, 0)
        else:
            gesture_color = (200, 200, 200)

        gesture_text = f"当前手势: {gesture}"
        if self.config.get('display', 'show_confidence'):
            # 显示为小数，避免字体问题
            gesture_text += f" ({confidence:.2f})"

        frame = self.draw_text(frame, gesture_text, (w // 2, 40), size=16, color=gesture_color)

        # 性能模式显示
        if gesture_recognizer and self.config.get('display', 'show_performance_mode'):
            mode_info = gesture_recognizer.get_performance_mode_info()

            # 根据模式选择颜色
            if mode_info['mode'] == 'fast':
                mode_color = self.colors['performance_fast']
            elif mode_info['mode'] == 'balanced':
                mode_color = self.colors['performance_balanced']
            else:  # accurate
                mode_color = self.colors['performance_accurate']

            mode_text = f"性能模式: {mode_info['name']}"
            frame = self.draw_text(frame, mode_text, (w // 2, 65), size=14, color=mode_color)

        # 录制/回放状态
        elif trajectory_recorder and self.config.get('display', 'show_recording_status'):
            recorder_status = trajectory_recorder.get_status()

            if recorder_status['is_recording']:
                status_color = self.colors['recording']
                status_text = f"录制中: {recorder_status['trajectory_points']}点"
                frame = self.draw_text(frame, status_text, (w // 2, 65), size=14, color=status_color)
            elif recorder_status['is_playing']:
                if recorder_status['playback_paused']:
                    status_color = self.colors['paused']
                    status_text = f"回放暂停: {recorder_status['playback_index']}/{recorder_status['playback_total']}"
                else:
                    status_color = self.colors['playback']
                    status_text = f"回放中: {recorder_status['playback_index']}/{recorder_status['playback_total']}"
                frame = self.draw_text(frame, status_text, (w // 2, 65), size=14, color=status_color)
            else:
                status_text = f"轨迹点: {recorder_status['trajectory_points']}"
                frame = self.draw_text(frame, status_text, (w // 2, 65), size=12, color=self.colors['info'])

        # 性能统计显示
        if performance_analyzer and self.config.get('display', 'show_performance_stats'):
            perf_stats = performance_analyzer.get_stats_summary()

            # 性能状态颜色
            if perf_stats['performance_status'] == "优秀":
                perf_color = (0, 255, 0)  # 绿色
            elif perf_stats['performance_status'] == "良好":
                perf_color = (100, 255, 100)  # 浅绿色
            elif perf_stats['performance_status'] == "一般":
                perf_color = (255, 255, 0)  # 黄色
            elif perf_stats['performance_status'] == "警告":
                perf_color = (255, 165, 0)  # 橙色
            else:
                perf_color = (255, 0, 0)  # 红色

            perf_text = f"性能: {perf_stats['performance_status']}({perf_stats['performance_score']:.0f}分) {perf_stats['fps']:.0f}FPS"
            frame = self.draw_text(frame, perf_text, (w // 2, 90), size=12, color=perf_color)

        # 系统资源显示
        if performance_analyzer and self.config.get('display', 'show_system_resources'):
            cpu_usage = performance_analyzer.get_current_cpu_usage()
            memory_usage = performance_analyzer.get_current_memory_usage()

            # CPU颜色
            if cpu_usage < 70:
                cpu_color = self.colors['cpu_good']
            elif cpu_usage < 85:
                cpu_color = self.colors['cpu_warning']
            else:
                cpu_color = self.colors['cpu_critical']

            # 内存颜色
            if memory_usage < 70:
                mem_color = self.colors['memory_good']
            elif memory_usage < 85:
                mem_color = self.colors['memory_warning']
            else:
                mem_color = self.colors['memory_critical']

            # 绘制系统资源信息
            sys_text = f"CPU:{cpu_usage:.0f}%({performance_analyzer.get_cpu_trend()}) 内存:{memory_usage:.0f}%({performance_analyzer.get_memory_trend()})"
            frame = self.draw_text(frame, sys_text, (w // 2, 110), size=11, color=self.colors['info'])

        # 语音状态
        if self.config.get('display', 'show_speech_status') and self.speech_manager:
            speech_status = self.speech_manager.get_status()
            speech_color = self.colors['speech_enabled'] if speech_status['enabled'] else self.colors['speech_disabled']
            speech_text = f"语音: {'启用' if speech_status['enabled'] else '禁用'}"
            frame = self.draw_text(frame, speech_text, (w - 200, 90), size=12, color=speech_color)

        # 性能信息
        if self.config.get('display', 'show_fps'):
            perf_text = f"帧率: {fps:.1f}"
            if process_time > 0:
                perf_text += f" | 延迟: {process_time:.1f}ms"

                # 根据处理时间选择颜色
                if process_time < 20:
                    perf_color = self.colors['performance_good']
                elif process_time < 50:
                    perf_color = self.colors['performance_warning']
                else:
                    perf_color = self.colors['performance_bad']
            else:
                perf_color = self.colors['info']

            frame = self.draw_text(frame, perf_text, (w - 200, 65), size=12, color=perf_color)

        # 控制提示
        control_text = "提示: 确保手部完全进入画面，保持稳定手势"
        frame = self.draw_text(frame, control_text, (10, 90), size=12, color=self.colors['info'])

        # 性能快捷键提示
        if performance_analyzer:
            perf_tip = "P:详细报告 L:快照 K:导出"
            frame = self.draw_text(frame, perf_tip, (10, 110), size=11, color=self.colors['help'])

        return frame

    def draw_flight_mode(self, frame, drone_controller):
        """绘制飞行模式信息"""
        if not drone_controller.connected:
            return frame

        h, w = frame.shape[:2]

        # 飞行模式显示
        flight_mode = drone_controller.get_flight_mode()

        # 根据飞行模式选择颜色和文本
        mode_texts = {
            'manual': ('手动模式', self.colors['flight_mode_manual']),
            'auto': ('自动飞行', self.colors['flight_mode_auto']),
            'circle': ('圆形盘旋', self.colors['flight_mode_circle']),
            'eight': ('8字形飞行', self.colors['flight_mode_eight']),
            'square': ('方形轨迹', self.colors['flight_mode_square']),
            'return_home': ('返航中', self.colors['flight_mode_return_home'])
        }

        if flight_mode in mode_texts:
            text, color = mode_texts[flight_mode]
            mode_text = f"飞行模式: {text}"

            # 绘制背景
            text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            bg_x = w - text_size[0] - 20
            bg_y = 150
            bg_w = text_size[0] + 10
            bg_h = text_size[1] + 10

            # 绘制半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            # 绘制文本
            frame = self.draw_text(frame, mode_text, (bg_x + 5, bg_y + 5),
                                  size=14, color=color)

        return frame

    def draw_help_bar(self, frame):
        """绘制帮助栏"""
        if not self.config.get('display', 'show_help'):
            return frame

        h, w = frame.shape[:2]

        # 绘制底部帮助栏
        cv2.rectangle(frame, (0, h - 140), (w, h), (0, 0, 0), -1)

        # 帮助文本
        help_lines = [
            "C:连接  空格:起飞/降落  ESC:退出  W/A/S/D/F/B:键盘控制",
            "H:切换帮助  R:重置识别  T:切换显示模式  D:调试信息",
            "V:切换语音反馈  M:测试语音  P:性能报告  L:性能快照  K:导出日志",
            "O:切换性能模式  Q:自动飞行  E:圆形盘旋  8:8字形  9:方形轨迹",
            "T/Y/U:高度控制  G:返航  X:停止  H:悬停",
            "1:开始录制 2:停止录制 3:保存轨迹 4:回放轨迹 5:清除轨迹 6:暂停/继续"
        ]

        for i, line in enumerate(help_lines):
            y_pos = h - 125 + i * 20
            frame = self.draw_text(frame, line, (10, y_pos), size=14, color=self.colors['help'])

        return frame

    def draw_warning(self, frame, message):
        """绘制警告信息"""
        h, w = frame.shape[:2]

        # 在顶部绘制警告
        warning_bg = np.zeros((40, w, 3), dtype=np.uint8)
        warning_bg[:, :] = (0, 69, 255)

        frame[130:170, 0:w] = cv2.addWeighted(
            frame[130:170, 0:w], 0.3,
            warning_bg, 0.7, 0
        )

        # 绘制警告文本
        frame = self.draw_text(frame, message, (10, 145),
                               size=16, color=self.colors['warning'])

        return frame

    def draw_performance_chart(self, frame, performance_analyzer):
        """绘制性能图表"""
        if not performance_analyzer:
            return frame

        h, w = frame.shape[:2]

        # 绘制图表区域背景
        chart_x = w - 250
        chart_y = 130
        chart_width = 240
        chart_height = 150

        cv2.rectangle(frame, (chart_x, chart_y),
                      (chart_x + chart_width, chart_y + chart_height),
                      self.colors['chart_background'], -1)
        cv2.rectangle(frame, (chart_x, chart_y),
                      (chart_x + chart_width, chart_y + chart_height),
                      self.colors['chart_grid'], 1)

        # 绘制标题
        frame = self.draw_text(frame, "性能监控", (chart_x + 10, chart_y + 20),
                               size=14, color=self.colors['title'])

        # 绘制FPS图表
        if len(performance_analyzer.fps_history) > 1:
            fps_values = list(performance_analyzer.fps_history)[-20:]  # 最近20个值

            # 计算最大值和最小值
            if fps_values:
                max_fps = max(fps_values)
                min_fps = min(fps_values)
                if max_fps == min_fps:
                    max_fps = min_fps + 1

                # 绘制FPS曲线
                points = []
                for i, fps in enumerate(fps_values):
                    x = chart_x + 10 + i * (chart_width - 20) / len(fps_values)
                    y = chart_y + chart_height - 30 - (fps - min_fps) * (chart_height - 50) / (max_fps - min_fps)
                    points.append((int(x), int(y)))

                # 连接点形成曲线
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i + 1], self.colors['chart_fps'], 2)

                # 显示当前FPS值
                current_fps = fps_values[-1] if fps_values else 0
                fps_color = self.colors['performance_good'] if current_fps > 20 else \
                    self.colors['performance_warning'] if current_fps > 10 else \
                        self.colors['performance_bad']

                frame = self.draw_text(frame, f"FPS: {current_fps:.1f}",
                                       (chart_x + 120, chart_y + 40),
                                       size=12, color=fps_color)

        return frame

    def draw_detailed_performance_charts(self, frame, performance_analyzer):
        """绘制详细性能图表"""
        if not performance_analyzer:
            return frame

        h, w = frame.shape[:2]

        # 绘制多个图表区域
        chart_height = 100
        chart_spacing = 10

        # FPS图表
        fps_chart_y = 130
        frame = self.draw_chart(frame, w - 250, fps_chart_y, 240, chart_height,
                               "FPS监控", performance_analyzer.fps_history,
                               self.colors['chart_fps'], "FPS")

        # CPU图表
        cpu_chart_y = fps_chart_y + chart_height + chart_spacing
        frame = self.draw_chart(frame, w - 250, cpu_chart_y, 240, chart_height,
                               "CPU使用率", performance_analyzer.cpu_usage_history,
                               self.colors['chart_cpu'], "%", max_val=100)

        # 内存图表
        memory_chart_y = cpu_chart_y + chart_height + chart_spacing
        frame = self.draw_chart(frame, w - 250, memory_chart_y, 240, chart_height,
                               "内存使用率", performance_analyzer.memory_usage_history,
                               self.colors['chart_memory'], "%", max_val=100)

        # 识别时间图表
        if len(performance_analyzer.gesture_recognition_times) > 1:
            recognition_chart_y = memory_chart_y + chart_height + chart_spacing
            frame = self.draw_chart(frame, w - 250, recognition_chart_y, 240, chart_height,
                                   "识别时间", performance_analyzer.gesture_recognition_times,
                                   self.colors['chart_recognition'], "ms")

        return frame

    def draw_chart(self, frame, x, y, width, height, title, data, color, unit="", max_val=None):
        """绘制单个图表"""
        # 绘制图表背景
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     self.colors['chart_background'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     self.colors['chart_grid'], 1)

        # 绘制标题
        frame = self.draw_text(frame, title, (x + 10, y + 15),
                              size=12, color=self.colors['chart_text'])

        if len(data) < 2:
            return frame

        # 获取数据
        values = list(data)[-min(20, len(data)):]  # 最多显示20个点
        if not values:
            return frame

        # 计算最大值和最小值
        data_max = max(values) if max_val is None else max_val
        data_min = min(values)

        if data_max == data_min:
            data_max = data_min + 1

        # 绘制网格线
        grid_color = self.colors['chart_grid']
        for i in range(1, 4):
            grid_y = y + height - 20 - i * (height - 40) // 4
            cv2.line(frame, (x + 10, grid_y), (x + width - 10, grid_y), grid_color, 1)

            # 绘制刻度值
            tick_value = data_min + (data_max - data_min) * i / 4
            tick_text = f"{tick_value:.0f}{unit}"
            frame = self.draw_text(frame, tick_text, (x + width - 35, grid_y - 5),
                                  size=10, color=self.colors['chart_text'])

        # 绘制数据曲线
        points = []
        for i, value in enumerate(values):
            point_x = x + 10 + i * (width - 20) / (len(values) - 1) if len(values) > 1 else x + 10
            point_y = y + height - 20 - (value - data_min) * (height - 40) / (data_max - data_min)
            points.append((int(point_x), int(point_y)))

        # 连接点形成曲线
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 2)

        # 绘制当前值
        if values:
            current_value = values[-1]
            current_text = f"当前: {current_value:.1f}{unit}"
            frame = self.draw_text(frame, current_text, (x + 10, y + 30),
                                  size=11, color=color)

            # 绘制平均值
            avg_value = sum(values) / len(values)
            avg_text = f"平均: {avg_value:.1f}{unit}"
            frame = self.draw_text(frame, avg_text, (x + width - 80, y + 30),
                                  size=11, color=self.colors['chart_text'])

        return frame