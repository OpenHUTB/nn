# modules/ui_controller.py
import pygame
import sys
import numpy as np
import cv2
import time
import math


class UIController:
    def __init__(self):
        # 初始化Pygame
        pygame.init()

        # 窗口配置
        self.screen_width = 1000  # 稍微加宽以显示更多信息
        self.screen_height = 750

        # 创建窗口
        try:
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height),
                pygame.HWSURFACE | pygame.DOUBLEBUF
            )
        except:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        pygame.display.set_caption("AI无人机人脸跟踪系统 - 多窗口版")
        self.clock = pygame.time.Clock()

        # 字体
        self.font_small = pygame.font.SysFont(None, 18)
        self.font_normal = pygame.font.SysFont(None, 22)
        self.font_large = pygame.font.SysFont(None, 28)
        self.font_title = pygame.font.SysFont(None, 32, bold=True)

        # 颜色
        self.COLORS = {
            'bg': (15, 20, 30),
            'panel_bg': (25, 30, 40),
            'panel_border': (60, 80, 100),
            'text': (220, 230, 240),
            'text_secondary': (160, 180, 200),
            'success': (0, 200, 100),
            'warning': (255, 180, 0),
            'error': (255, 80, 80),
            'drone': (0, 180, 255),
            'detection': (255, 100, 100),
            'tracking': (255, 200, 0),
            'camera': (100, 200, 255),
            'voice': (200, 100, 255),
            'data': (100, 255, 200),
        }

        # 显示区域
        self.display_width = 640
        self.display_height = 480
        self.display_x = (self.screen_width - self.display_width) // 2
        self.display_y = 50

        # 状态
        self.state = {
            "running": True,
            "drone_status": "未连接",
            "drone_position": (0.0, 0.0, 1.0),
            "drone_yaw": 0.0,
            "detected_faces": 0,
            "detected_persons": 0,
            "recognized_person": "Unknown",
            "fps": 0,
            "camera_status": "初始化中...",
            "tracking_mode": "手动",
            "is_flying": False,
            "detection_active": True,
            "recognition_active": False,
            "voice_enabled": False,
            "data_logging": False,
            "flight_time": 0,
            "total_distance": 0.0,
        }

        # 3D窗口和语音状态
        self.drone_3d_window_open = False
        self.drone_3d_status = "未连接"
        self.voice_status = "未启用"
        self.data_status = "未记录"

        # 性能监控
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.fps_history = []

        # 错误信息
        self.error_messages = []

        # 动画
        self.animation_time = time.time()

        # 飞行路径
        self.flight_path = []
        self.max_path_points = 100

        # 语音消息队列
        self.voice_messages = []

        # 数据记录
        self.data_records = []

        print("✅ UI控制器初始化完成")

    def add_voice_message(self, message):
        """添加语音消息"""
        self.voice_messages.append({
            'message': message,
            'time': time.time(),
            'played': False
        })
        print(f"🗣️  语音: {message}")

    def add_data_record(self, record_type, data):
        """添加数据记录"""
        record = {
            'timestamp': time.time(),
            'type': record_type,
            'data': data
        }
        self.data_records.append(record)

        # 保持记录数量
        if len(self.data_records) > 1000:
            self.data_records = self.data_records[-1000:]

    def update_drone_state(self, state):
        """更新无人机状态"""
        old_position = self.state['drone_position']
        self.state.update(state)

        # 计算飞行距离
        if self.state['is_flying']:
            new_position = self.state['drone_position']
            dx = new_position[0] - old_position[0]
            dy = new_position[1] - old_position[1]
            dz = new_position[2] - old_position[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            self.state['total_distance'] += distance

        # 更新飞行路径
        self.flight_path.append(self.state['drone_position'])
        if len(self.flight_path) > self.max_path_points:
            self.flight_path = self.flight_path[-self.max_path_points:]

    def update_3d_window_status(self, is_open, status="已连接"):
        """更新3D窗口状态"""
        self.drone_3d_window_open = is_open
        self.drone_3d_status = status
        self.state['drone_3d_status'] = status

        # 添加语音提示
        if is_open and self.state['voice_enabled']:
            self.add_voice_message(f"3D窗口{status}")

    def update_voice_status(self, enabled, status="已启用"):
        """更新语音状态"""
        self.state['voice_enabled'] = enabled
        self.voice_status = status

    def update_data_status(self, logging, status="记录中"):
        """更新数据记录状态"""
        self.state['data_logging'] = logging
        self.data_status = status

        # 添加语音提示
        if logging and self.state['voice_enabled']:
            self.add_voice_message(f"数据记录{status}")

    # ...（handle_events、update_frame_display等方法保持不变，只需在相应位置添加新功能调用）...

    def _draw_status_panel(self):
        """绘制状态面板"""
        panel_x = 20
        panel_y = self.display_y + self.display_height + 20
        panel_width = self.screen_width - 40
        panel_height = 160

        # 背景
        pygame.draw.rect(
            self.screen,
            self.COLORS['panel_bg'],
            (panel_x, panel_y, panel_width, panel_height),
            0, 10
        )

        # 边框
        border_color = self.COLORS['panel_border']
        if self.state['is_flying']:
            current_time = time.time()
            pulse = (math.sin(current_time * 3) + 1) / 2
            border_color = (
                int(border_color[0] * (0.7 + 0.3 * pulse)),
                int(border_color[1] * (0.7 + 0.3 * pulse)),
                int(border_color[2] * (0.7 + 0.3 * pulse))
            )

        pygame.draw.rect(
            self.screen,
            border_color,
            (panel_x, panel_y, panel_width, panel_height),
            2, 10
        )

        # 三列显示
        col_width = panel_width // 3 - 20

        # 第一列：无人机状态
        left_col_x = panel_x + 15
        drone_status = [
            ("无人机状态:", self.state['drone_status'],
             self.COLORS['success'] if "已连接" in self.state['drone_status'] else self.COLORS['warning']),
            ("飞行状态:", "飞行中" if self.state['is_flying'] else "地面",
             self.COLORS['drone'] if self.state['is_flying'] else self.COLORS['text_secondary']),
            ("控制模式:", self.state['tracking_mode'],
             self.COLORS['tracking'] if self.state['tracking_mode'] == "追踪" else self.COLORS['text']),
            ("3D窗口:", self.drone_3d_status,
             self.COLORS['success'] if "已连接" in self.drone_3d_status else self.COLORS['text_secondary']),
        ]

        line_height = 22
        for i, (label, value, color) in enumerate(drone_status):
            label_surf = self.font_small.render(label, True, self.COLORS['text_secondary'])
            self.screen.blit(label_surf, (left_col_x, panel_y + 15 + i * line_height))
            value_surf = self.font_normal.render(value, True, color)
            self.screen.blit(value_surf, (left_col_x + 80, panel_y + 15 + i * line_height))

        # 第二列：检测状态
        middle_col_x = panel_x + col_width + 25
        detection_status = [
            ("检测状态:", "活跃" if self.state['detection_active'] else "暂停",
             self.COLORS['success'] if self.state['detection_active'] else self.COLORS['warning']),
            ("人脸检测:", f"{self.state['detected_faces']} 个",
             self.COLORS['detection'] if self.state['detected_faces'] > 0 else self.COLORS['text_secondary']),
            ("行人检测:", f"{self.state['detected_persons']} 个",
             self.COLORS['detection'] if self.state['detected_persons'] > 0 else self.COLORS['text_secondary']),
            ("识别结果:", self.state['recognized_person'],
             self.COLORS['success'] if self.state['recognized_person'] != "Unknown" else self.COLORS['text_secondary']),
        ]

        for i, (label, value, color) in enumerate(detection_status):
            label_surf = self.font_small.render(label, True, self.COLORS['text_secondary'])
            self.screen.blit(label_surf, (middle_col_x, panel_y + 15 + i * line_height))
            value_surf = self.font_normal.render(value, True, color)
            self.screen.blit(value_surf, (middle_col_x + 80, panel_y + 15 + i * line_height))

        # 第三列：系统功能
        right_col_x = panel_x + 2 * col_width + 35
        system_status = [
            ("语音播报:", "启用" if self.state['voice_enabled'] else "禁用",
             self.COLORS['voice'] if self.state['voice_enabled'] else self.COLORS['text_secondary']),
            ("数据记录:", "进行中" if self.state['data_logging'] else "停止",
             self.COLORS['data'] if self.state['data_logging'] else self.COLORS['text_secondary']),
            ("飞行时间:", f"{self.state['flight_time']:.0f}秒", self.COLORS['text']),
            ("飞行距离:", f"{self.state['total_distance']:.1f}米", self.COLORS['text']),
        ]

        for i, (label, value, color) in enumerate(system_status):
            label_surf = self.font_small.render(label, True, self.COLORS['text_secondary'])
            self.screen.blit(label_surf, (right_col_x, panel_y + 15 + i * line_height))
            value_surf = self.font_normal.render(value, True, color)
            self.screen.blit(value_surf, (right_col_x + 80, panel_y + 15 + i * line_height))

    def _draw_system_info(self):
        """绘制系统信息"""
        info_x = self.screen_width - 220
        info_y = 20

        # 背景
        pygame.draw.rect(
            self.screen,
            (30, 35, 45, 200),
            (info_x - 10, info_y - 10, 210, 120),
            0, 5
        )

        # FPS
        fps_color = self.COLORS['success'] if self.state['fps'] >= 25 else \
            self.COLORS['warning'] if self.state['fps'] >= 15 else \
                self.COLORS['error']
        fps_surf = self.font_normal.render(f"FPS: {self.state['fps']}", True, fps_color)
        self.screen.blit(fps_surf, (info_x, info_y))

        # 3D窗口状态
        drone_color = self.COLORS['success'] if "已连接" in self.drone_3d_status else self.COLORS['text_secondary']
        drone_surf = self.font_small.render(f"3D窗口: {self.drone_3d_status}", True, drone_color)
        self.screen.blit(drone_surf, (info_x, info_y + 25))

        # 语音状态
        voice_color = self.COLORS['voice'] if self.state['voice_enabled'] else self.COLORS['text_secondary']
        voice_surf = self.font_small.render(f"语音: {'启用' if self.state['voice_enabled'] else '禁用'}", True,
                                            voice_color)
        self.screen.blit(voice_surf, (info_x, info_y + 45))

        # 数据记录状态
        data_color = self.COLORS['data'] if self.state['data_logging'] else self.COLORS['text_secondary']
        data_surf = self.font_small.render(f"数据: {'记录中' if self.state['data_logging'] else '停止'}", True,
                                           data_color)
        self.screen.blit(data_surf, (info_x, info_y + 65))

    def _draw_flight_path(self):
        """绘制飞行路径预览"""
        if len(self.flight_path) < 2:
            return

        # 在画面右下角绘制小地图
        map_size = 120
        map_x = self.screen_width - map_size - 20
        map_y = self.screen_height - map_size - 20

        # 地图背景
        pygame.draw.rect(
            self.screen, (20, 25, 35),
            (map_x, map_y, map_size, map_size), 0, 5
        )

        pygame.draw.rect(
            self.screen, self.COLORS['panel_border'],
            (map_x, map_y, map_size, map_size), 1, 5
        )

        # 地图标题
        map_title = "飞行路径"
        title_surf = self.font_small.render(map_title, True, self.COLORS['text_secondary'])
        self.screen.blit(title_surf, (map_x + 5, map_y + 5))

        # 转换坐标到地图尺寸
        scale = map_size * 0.7 / 10  # 假设飞行范围是10x10单位

        # 绘制路径
        if len(self.flight_path) > 1:
            path_points = []
            for x, y, z in self.flight_path:
                # 转换为地图坐标
                map_px = map_x + map_size // 2 + x * scale
                map_py = map_y + map_size // 2 + y * scale
                path_points.append((map_px, map_py))

            # 绘制路径线
            if len(path_points) > 1:
                pygame.draw.lines(
                    self.screen, self.COLORS['drone'],
                    False, path_points, 2
                )

            # 绘制当前位置
            if path_points:
                last_point = path_points[-1]
                pygame.draw.circle(
                    self.screen, self.COLORS['tracking'],
                    (int(last_point[0]), int(last_point[1])), 3
                )

    def _draw_voice_queue(self):
        """绘制语音消息队列"""
        if not self.voice_messages:
            return

        queue_x = 20
        queue_y = self.screen_height - 60

        # 显示最新的语音消息
        latest_msg = self.voice_messages[-1]
        if time.time() - latest_msg['time'] < 3:  # 显示最近3秒的消息
            msg_text = f"🗣️  {latest_msg['message']}"
            msg_surf = self.font_small.render(msg_text, True, self.COLORS['voice'])
            self.screen.blit(msg_surf, (queue_x, queue_y))

    def update_lightweight(self, frame):
        """更新UI"""
        try:
            # 清除过期错误
            self.clear_old_errors()

            # 填充背景
            self.screen.fill(self.COLORS['bg'])

            # 更新画面显示
            self.update_frame_display(frame)

            # 更新状态面板
            self._draw_status_panel()

            # 绘制控制面板
            self._draw_control_panel()

            # 绘制系统信息
            self._draw_system_info()

            # 绘制飞行路径
            self._draw_flight_path()

            # 绘制语音队列
            self._draw_voice_queue()

            # 更新FPS
            self.update_fps()

            # 更新显示
            pygame.display.flip()

        except Exception as e:
            self.set_error(f"UI更新异常: {e}", "error")

    def save_data_records(self, filename=None):
        """保存数据记录"""
        if not filename:
            import datetime
            filename = f"flight_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            import json
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.data_records, f, ensure_ascii=False, indent=2)
            print(f"✅ 数据已保存到: {filename}")
            return True
        except Exception as e:
            print(f"❌ 数据保存失败: {e}")
            return False

    # ...（其他方法保持不变）...