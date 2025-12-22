# drone_system_complete.py
# !/usr/bin/env python3

print("=" * 60)
print("🚀 AI无人机人脸跟踪系统 - 完整版")
print("=" * 60)

# ============ 第一步：应用兼容性补丁 ============
try:
    import sys
    import pkgutil

    if not hasattr(pkgutil, 'ImpImporter'):
        class ImpImporter:
            def __init__(self, path=None):
                self.path = path

            def find_module(self, fullname, path=None):
                return None

            def load_module(self, fullname):
                raise ImportError(f"无法加载: {fullname}")


        pkgutil.ImpImporter = ImpImporter

    print("✅ 兼容性补丁已应用")
except Exception as e:
    print(f"⚠️  补丁应用失败: {e}")

# ============ 第二步：导入核心模块 ============
import pygame
import cv2
import numpy as np
import time
import random
import json
import os
import threading
import queue


# ============ 第三步：创建简化模块 ============

class SimpleDroneController:
    def __init__(self):
        self.cap = None
        self.init_camera()

    def init_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print("✅ 摄像头初始化成功")
                return True
            else:
                print("⚠️  摄像头打开失败，使用模拟模式")
                return False
        except:
            print("⚠️  摄像头初始化异常，使用模拟模式")
            return False

    def get_frame(self):
        """获取画面"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame

        # 生成模拟画面
        return self._generate_test_frame()

    def _generate_test_frame(self):
        """生成测试画面"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 网格
        for i in range(0, 640, 40):
            cv2.line(frame, (i, 0), (i, 480), (50, 50, 50), 1)
        for i in range(0, 480, 40):
            cv2.line(frame, (0, i), (640, i), (50, 50, 50), 1)

        # 中心十字
        center_x, center_y = 320, 240
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (80, 80, 120), 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (80, 80, 120), 2)

        # 添加一些模拟检测目标
        num_faces = random.randint(0, 3)
        for i in range(num_faces):
            x = random.randint(100, 540)
            y = random.randint(100, 380)
            size = random.randint(40, 80)
            cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        num_persons = random.randint(0, 4)
        for i in range(num_persons):
            x = random.randint(50, 590)
            y = random.randint(50, 430)
            width = random.randint(40, 100)
            height = random.randint(80, 160)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            cv2.putText(frame, f"Person{i + 1}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 文字
        cv2.putText(frame, "AI无人机跟踪系统", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "按T起飞，按Y切换追踪", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 100), 1)
        cv2.putText(frame, "WASD移动，空格/Ctrl升降", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 255), 1)

        return frame

    def disconnect(self):
        """断开连接"""
        if self.cap:
            self.cap.release()
        print("✅ 摄像头已释放")


class UIController:
    def __init__(self):
        # 初始化Pygame
        pygame.init()

        # 窗口配置
        self.screen_width = 1000
        self.screen_height = 750

        # 创建窗口
        try:
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height),
                pygame.HWSURFACE | pygame.DOUBLEBUF
            )
        except:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        pygame.display.set_caption("AI无人机人脸跟踪系统 - 全功能版")
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
            "drone_3d_open": False,
        }

        # 性能监控
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.fps_history = []

        # 错误信息
        self.error_messages = []

        # 飞行路径
        self.flight_path = []
        self.max_path_points = 100

        # 语音消息队列
        self.voice_messages = []

        # 数据记录
        self.data_records = []

        print("✅ UI控制器初始化完成")

    def set_error(self, message, level="error"):
        """设置错误信息"""
        self.error_messages.append({
            'message': message,
            'level': level,
            'time': time.time()
        })
        print(f"UI {level}: {message}")

    def clear_old_errors(self):
        """清除过期错误"""
        current_time = time.time()
        self.error_messages = [
            err for err in self.error_messages
            if current_time - err['time'] < 5.0
        ]

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

        if len(self.data_records) > 1000:
            self.data_records = self.data_records[-1000:]

    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return "quit"
                elif event.key == pygame.K_t:
                    return "takeoff_land"
                elif event.key == pygame.K_y:
                    return "toggle_tracking"
                elif event.key == pygame.K_r:
                    return "reset_position"
                elif event.key == pygame.K_f:
                    return "toggle_fullscreen"
                elif event.key == pygame.K_g:
                    return "toggle_3d_window"
                elif event.key == pygame.K_v:
                    return "toggle_voice"
                elif event.key == pygame.K_d:
                    return "toggle_data_logging"
                elif event.key == pygame.K_s:
                    return "save_data"
                elif event.key == pygame.K_p:
                    return "playback_data"
                elif event.key == pygame.K_h:
                    return "hover_mode"
                elif event.key == pygame.K_l:
                    return "land"
                elif event.key == pygame.K_1:
                    return "mode_manual"
                elif event.key == pygame.K_2:
                    return "mode_tracking"
                elif event.key == pygame.K_3:
                    return "mode_hover"
                elif event.key == pygame.K_4:
                    return "toggle_detection"
                elif event.key == pygame.K_5:
                    return "toggle_recognition"

        return None

    def update_frame_display(self, frame):
        """更新画面显示"""
        if frame is None or frame.size == 0:
            self._draw_no_frame()
            return

        try:
            # 检查帧
            if len(frame.shape) < 2:
                self.set_error("无效帧格式", "warning")
                self._draw_no_frame()
                return

            # 转换颜色
            if len(frame.shape) == 3:
                if frame.shape[2] == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            # 缩放
            try:
                frame_resized = cv2.resize(frame_rgb, (self.display_width, self.display_height))
            except:
                self.set_error("缩放失败", "error")
                self._draw_no_frame()
                return

            # 转换为Surface
            try:
                frame_surface = pygame.surfarray.make_surface(
                    np.transpose(frame_resized, (1, 0, 2))
                )
            except:
                try:
                    frame_bytes = frame_resized.tobytes()
                    frame_surface = pygame.image.frombuffer(
                        frame_bytes,
                        (self.display_width, self.display_height),
                        'RGB'
                    )
                except:
                    self.set_error("Surface创建失败", "error")
                    self._draw_no_frame()
                    return

            # 绘制背景和边框
            pygame.draw.rect(
                self.screen,
                (10, 15, 25),
                (self.display_x - 10, self.display_y - 10,
                 self.display_width + 20, self.display_height + 20),
                0, 8
            )

            # 边框颜色
            border_color = self.COLORS['camera']
            if self.state['tracking_mode'] == "追踪":
                import math
                current_time = time.time()
                pulse = (math.sin(current_time * 3) + 1) / 2
                border_color = (
                    int(255 * pulse),
                    int(200 * pulse),
                    0
                )

            pygame.draw.rect(
                self.screen,
                border_color,
                (self.display_x - 5, self.display_y - 5,
                 self.display_width + 10, self.display_height + 10),
                3, 10
            )

            # 显示画面
            self.screen.blit(frame_surface, (self.display_x, self.display_y))

            # 画面信息
            h, w = frame.shape[:2]
            info_text = f"画面: {w}x{h}"
            info_surf = self.font_small.render(info_text, True, (180, 180, 220))
            self.screen.blit(info_surf, (self.display_x, self.display_y + self.display_height + 5))

        except Exception as e:
            self.set_error(f"画面显示异常: {str(e)[:30]}", "error")
            self._draw_no_frame()

    def _draw_no_frame(self):
        """无画面显示"""
        pygame.draw.rect(
            self.screen,
            (5, 10, 15),
            (self.display_x, self.display_y, self.display_width, self.display_height),
            0, 8
        )

        # 网格
        grid_size = 40
        grid_color = (40, 50, 70)
        for x in range(self.display_x, self.display_x + self.display_width, grid_size):
            pygame.draw.line(
                self.screen, grid_color,
                (x, self.display_y), (x, self.display_y + self.display_height), 1
            )
        for y in range(self.display_y, self.display_y + self.display_height, grid_size):
            pygame.draw.line(
                self.screen, grid_color,
                (self.display_x, y), (self.display_x + self.display_width, y), 1
            )

        # 中心十字
        center_x = self.display_x + self.display_width // 2
        center_y = self.display_y + self.display_height // 2
        pygame.draw.line(
            self.screen, (80, 90, 120),
            (center_x - 20, center_y), (center_x + 20, center_y), 2
        )
        pygame.draw.line(
            self.screen, (80, 90, 120),
            (center_x, center_y - 20), (center_x, center_y + 20), 2
        )

        # 文本
        if self.error_messages:
            error = self.error_messages[-1]
            error_text = error['message']
            error_color = self.COLORS[error['level']]
            error_lines = self._wrap_text(error_text, 50)
            for i, line in enumerate(error_lines):
                error_surf = self.font_normal.render(line, True, error_color)
                error_rect = error_surf.get_rect(center=(center_x, center_y - 30 + i * 25))
                self.screen.blit(error_surf, error_rect)
            tip_text = "按R重试，检查摄像头"
        else:
            wait_surf = self.font_large.render("等待摄像头画面...", True, (255, 255, 0))
            wait_rect = wait_surf.get_rect(center=(center_x, center_y - 30))
            self.screen.blit(wait_surf, wait_rect)
            tip_text = "请检查摄像头连接"

        tip_surf = self.font_small.render(tip_text, True, (150, 150, 150))
        tip_rect = tip_surf.get_rect(center=(center_x, center_y + 30))
        self.screen.blit(tip_surf, tip_rect)

    def _wrap_text(self, text, max_width):
        """文本换行"""
        words = text.split(' ')
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            if len(test_line) > max_width:
                if len(current_line) > 1:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [current_line[-1]]
                else:
                    lines.append(test_line)
                    current_line = []
        if current_line:
            lines.append(' '.join(current_line))
        return lines

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
            import math
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
            ("3D窗口:", "开启" if self.state['drone_3d_open'] else "关闭",
             self.COLORS['success'] if self.state['drone_3d_open'] else self.COLORS['text_secondary']),
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

    def _draw_control_panel(self):
        """绘制控制面板"""
        panel_x = 20
        panel_y = self.screen_height - 220
        panel_width = self.screen_width - 40
        panel_height = 200

        # 背景
        pygame.draw.rect(
            self.screen,
            (30, 35, 45),
            (panel_x, panel_y, panel_width, panel_height),
            0, 10
        )

        # 标题
        title_surf = self.font_large.render("控制面板", True, self.COLORS['text'])
        self.screen.blit(title_surf, (panel_x + 15, panel_y + 10))

        # 分隔线
        pygame.draw.line(
            self.screen, self.COLORS['panel_border'],
            (panel_x + 10, panel_y + 40), (panel_x + panel_width - 10, panel_y + 40), 1
        )

        # 控制说明（三列）
        col_width = panel_width // 3 - 20

        # 第一列：基本控制
        basic_controls = [
            ("T", "起飞/降落"),
            ("Y", "追踪开关"),
            ("R", "重置位置"),
            ("H", "悬停模式"),
            ("L", "安全降落"),
        ]

        for i, (key, desc) in enumerate(basic_controls):
            key_surf = self.font_normal.render(key, True, self.COLORS['drone'])
            desc_surf = self.font_small.render(desc, True, self.COLORS['text_secondary'])
            x, y = panel_x + 20, panel_y + 55 + i * 25
            self.screen.blit(key_surf, (x, y))
            self.screen.blit(desc_surf, (x + 30, y + 2))

        # 第二列：飞行控制
        flight_controls = [
            ("W/S", "前进/后退"),
            ("A/D", "左移/右移"),
            ("空格/Ctrl", "上升/下降"),
            ("Shift+Q/E", "左转/右转"),
        ]

        for i, (key, desc) in enumerate(flight_controls):
            key_surf = self.font_normal.render(key, True, self.COLORS['drone'])
            desc_surf = self.font_small.render(desc, True, self.COLORS['text_secondary'])
            x, y = panel_x + col_width + 40, panel_y + 55 + i * 25
            self.screen.blit(key_surf, (x, y))
            self.screen.blit(desc_surf, (x + 60, y + 2))

        # 第三列：系统控制
        system_controls = [
            ("1/2/3", "手动/追踪/悬停"),
            ("4/5", "检测/识别开关"),
            ("G/V/D", "3D/语音/数据"),
            ("S/P", "保存/回放"),
            ("Q/ESC", "退出程序"),
        ]

        for i, (key, desc) in enumerate(system_controls):
            key_surf = self.font_normal.render(key, True, self.COLORS['text'])
            desc_surf = self.font_small.render(desc, True, self.COLORS['text_secondary'])
            x, y = panel_x + 2 * col_width + 60, panel_y + 55 + i * 25
            self.screen.blit(key_surf, (x, y))
            self.screen.blit(desc_surf, (x + 50, y + 2))

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
        drone_color = self.COLORS['success'] if self.state['drone_3d_open'] else self.COLORS['text_secondary']
        drone_surf = self.font_small.render(f"3D窗口: {'开启' if self.state['drone_3d_open'] else '关闭'}", True,
                                            drone_color)
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

    def update_fps(self):
        """更新FPS"""
        current_time = time.time()
        self.frame_count += 1

        if current_time - self.last_fps_update > 0.5:
            elapsed = current_time - self.last_fps_update
            current_fps = self.frame_count / elapsed

            self.fps_history.append(current_fps)
            if len(self.fps_history) > 10:
                self.fps_history = self.fps_history[-10:]

            self.state["fps"] = int(np.mean(self.fps_history)) if self.fps_history else 0
            self.frame_count = 0
            self.last_fps_update = current_time

    def update_drone_state(self, state):
        """更新无人机状态"""
        old_position = self.state['drone_position']
        self.state.update(state)

        # 计算飞行距离
        if self.state['is_flying']:
            new_position = self.state['drone_position']
            import math
            dx = new_position[0] - old_position[0]
            dy = new_position[1] - old_position[1]
            dz = new_position[2] - old_position[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            self.state['total_distance'] += distance

        # 更新飞行路径
        self.flight_path.append(self.state['drone_position'])
        if len(self.flight_path) > self.max_path_points:
            self.flight_path = self.flight_path[-self.max_path_points:]

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

            # 绘制语音队列
            self._draw_voice_queue()

            # 更新FPS
            self.update_fps()

            # 更新显示
            pygame.display.flip()

        except Exception as e:
            self.set_error(f"UI更新异常: {e}", "error")

    def update_empty(self):
        """无画面更新"""
        self.screen.fill(self.COLORS['bg'])

        center_x = self.screen_width // 2
        center_y = self.screen_height // 2

        # 标题
        title_surf = self.font_title.render("AI无人机跟踪系统", True, self.COLORS['text'])
        title_rect = title_surf.get_rect(center=(center_x, center_y - 100))
        self.screen.blit(title_surf, title_rect)

        # 状态
        status_text = f"系统初始化中... FPS: {self.state['fps']}"
        status_surf = self.font_large.render(status_text, True, self.COLORS['warning'])
        status_rect = status_surf.get_rect(center=(center_x, center_y - 40))
        self.screen.blit(status_surf, status_rect)

        # 提示
        hint_text = "按任意键继续，按Q退出"
        hint_surf = self.font_small.render(hint_text, True, (150, 170, 200))
        hint_rect = hint_surf.get_rect(center=(center_x, center_y + 150))
        self.screen.blit(hint_surf, hint_rect)

        pygame.display.flip()

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

    def quit(self):
        """退出"""
        print("🔄 关闭UI...")
        try:
            pygame.quit()
            print("✅ UI已关闭")
        except:
            pass


class VoiceSynthesizer:
    """语音合成器"""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.message_queue = queue.Queue()
        self.currently_speaking = False
        self.running = False
        self.worker_thread = None

        if self.enabled:
            self.start()

    def speak(self, text):
        """语音播报"""
        if not self.enabled or not text:
            return False

        try:
            self.message_queue.put(text)
            return True
        except:
            return False

    def _voice_worker(self):
        """语音工作线程"""
        while self.running:
            try:
                # 从队列获取消息
                text = self.message_queue.get(timeout=1)

                # 模拟语音播报（实际应用中可以使用pyttsx3）
                print(f"🗣️  语音播报: {text}")

                # 标记任务完成
                self.message_queue.task_done()

                # 短暂暂停
                time.sleep(0.5)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"语音播报错误: {e}")
                time.sleep(1)

    def start(self):
        """启动语音服务"""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self.worker_thread.start()
        print("✅ 语音服务已启动")

    def stop(self):
        """停止语音服务"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
        print("✅ 语音服务已停止")


class DataLogger:
    """数据记录器"""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.records = []
        self.max_records = 1000

        # 创建日志目录
        self.log_dir = "flight_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        print("✅ 数据记录器初始化完成")

    def log_drone_state(self, position, yaw, is_flying, mode):
        """记录无人机状态"""
        if not self.enabled:
            return

        record = {
            'timestamp': time.time(),
            'type': 'drone_state',
            'position': position,
            'yaw': yaw,
            'is_flying': is_flying,
            'mode': mode
        }
        self.records.append(record)

        # 保持记录数量
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]

    def log_detection_result(self, face_count, person_count, recognized_person):
        """记录检测结果"""
        if not self.enabled:
            return

        record = {
            'timestamp': time.time(),
            'type': 'detection',
            'face_count': face_count,
            'person_count': person_count,
            'recognized_person': recognized_person
        }
        self.records.append(record)

    def log_control_action(self, action, params=None):
        """记录控制动作"""
        if not self.enabled:
            return

        record = {
            'timestamp': time.time(),
            'type': 'control',
            'action': action,
            'params': params or {}
        }
        self.records.append(record)

    def save_to_file(self, filename=None):
        """保存数据到文件"""
        if not self.records:
            return False

        try:
            if not filename:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.log_dir, f"flight_{timestamp}.json")

            # 添加文件头信息
            data = {
                'metadata': {
                    'created_at': time.time(),
                    'total_records': len(self.records)
                },
                'records': self.records
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"✅ 已保存 {len(self.records)} 条记录到: {filename}")
            return True

        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            return False

    def get_statistics(self):
        """获取统计信息"""
        if not self.records:
            return {}

        stats = {
            'total_records': len(self.records),
            'flight_time': 0,
            'drone_states': 0,
            'detections': 0,
            'controls': 0
        }

        for record in self.records:
            if record['type'] == 'drone_state':
                stats['drone_states'] += 1
                if record['is_flying']:
                    stats['flight_time'] += 1  # 简化的飞行时间
            elif record['type'] == 'detection':
                stats['detections'] += 1
            elif record['type'] == 'control':
                stats['controls'] += 1

        return stats


class Drone3DViewer:
    """3D无人机查看器"""

    def __init__(self):
        self.window_open = False
        self.running = False
        self.viewer_thread = None

        print("✅ 3D查看器初始化完成")

    def start_viewer(self):
        """启动3D查看器"""
        if self.window_open:
            return True

        try:
            self.running = True
            self.viewer_thread = threading.Thread(target=self._run_viewer, daemon=True)
            self.viewer_thread.start()

            # 等待窗口初始化
            time.sleep(1)
            self.window_open = True

            print("✅ 3D无人机查看器已启动")
            return True

        except Exception as e:
            print(f"❌ 启动3D查看器失败: {e}")
            self.window_open = False
            return False

    def stop_viewer(self):
        """停止3D查看器"""
        self.running = False
        if self.viewer_thread:
            self.viewer_thread.join(timeout=2)
        self.window_open = False
        print("✅ 3D无人机查看器已停止")

    def _run_viewer(self):
        """运行3D查看器（模拟）"""
        try:
            print("🖥️  3D窗口: 正在运行模拟3D视图...")

            while self.running:
                # 模拟3D视图更新
                time.sleep(0.1)

        except Exception as e:
            print(f"3D查看器错误: {e}")


# ============ 第四步：主程序 ============
def main():
    print("\n🎬 初始化系统...")

    # 初始化组件
    try:
        drone = SimpleDroneController()
        print("✅ 无人机控制器")
    except Exception as e:
        print(f"❌ 无人机控制器失败: {e}")
        return

    try:
        ui = UIController()
        print("✅ UI控制器")
    except Exception as e:
        print(f"❌ UI控制器失败: {e}")
        return

    # 初始化3D查看器
    drone_3d = Drone3DViewer()

    # 初始化语音合成器
    voice = VoiceSynthesizer(enabled=True)
    ui.state['voice_enabled'] = True

    # 初始化数据记录器
    data_logger = DataLogger(enabled=True)
    ui.state['data_logging'] = True

    print("✅ 所有组件初始化完成")

    # 运行参数
    frame_count = 0
    is_flying = False
    tracking_mode = "手动"
    detection_active = True
    recognition_active = False

    # 无人机状态
    drone_position = [0.0, 0.0, 1.0]  # x, y, z
    drone_yaw = 0.0

    # 模拟检测结果
    detected_faces = 0
    detected_persons = 0
    recognized_person = "Unknown"

    # 飞行时间统计
    flight_start_time = 0
    total_flight_time = 0

    print("\n🎬 开始主循环...")
    print("控制说明:")
    print("  T - 起飞/降落")
    print("  Y - 切换追踪模式")
    print("  R - 重置位置")
    print("  WASD - 移动控制")
    print("  空格/Ctrl - 上升/下降")
    print("  Shift+Q/E - 左转/右转")
    print("  G - 开关3D窗口")
    print("  V - 开关语音")
    print("  D - 开关数据记录")
    print("  S - 保存数据")
    print("  P - 回放数据")
    print("  Q/ESC - 退出程序")
    print("=" * 50)

    # 语音播报欢迎信息
    voice.speak("无人机系统启动完成")
    ui.add_voice_message("系统启动完成")

    try:
        while ui.state["running"]:
            frame_count += 1

            # 处理事件
            event = ui.handle_events()
            if event == "quit":
                print("用户请求退出")
                break
            elif event == "takeoff_land":
                is_flying = not is_flying
                status = "起飞" if is_flying else "降落"
                print(f"无人机: {status}")

                # 语音播报
                voice.speak(f"无人机{status}")
                ui.add_voice_message(f"无人机{status}")

                # 数据记录
                data_logger.log_control_action("takeoff_land" if is_flying else "land", {"status": status})

                # 更新飞行时间
                if is_flying:
                    flight_start_time = time.time()
                else:
                    if flight_start_time > 0:
                        total_flight_time += time.time() - flight_start_time
                        flight_start_time = 0

                drone_position[2] = 1.5 if is_flying else 0.5

            elif event == "toggle_tracking":
                tracking_mode = "追踪" if tracking_mode == "手动" else "手动"
                print(f"追踪模式: {tracking_mode}")

                # 语音播报
                voice.speak(f"切换到{tracking_mode}模式")
                ui.add_voice_message(f"切换到{tracking_mode}模式")

                # 数据记录
                data_logger.log_control_action("toggle_tracking", {"mode": tracking_mode})

            elif event == "reset_position":
                drone_position = [0.0, 0.0, 1.0]
                drone_yaw = 0.0
                print("位置已重置")

                # 语音播报
                voice.speak("位置已重置")
                ui.add_voice_message("位置已重置")

                # 数据记录
                data_logger.log_control_action("reset_position")

            elif event == "toggle_3d_window":
                if drone_3d.window_open:
                    drone_3d.stop_viewer()
                    ui.state['drone_3d_open'] = False
                    voice.speak("3D窗口已关闭")
                    ui.add_voice_message("3D窗口已关闭")
                else:
                    if drone_3d.start_viewer():
                        ui.state['drone_3d_open'] = True
                        voice.speak("3D窗口已开启")
                        ui.add_voice_message("3D窗口已开启")

            elif event == "toggle_voice":
                voice.enabled = not voice.enabled
                ui.state['voice_enabled'] = voice.enabled
                status = "启用" if voice.enabled else "禁用"
                print(f"语音: {status}")
                ui.add_voice_message(f"语音播报{status}")

            elif event == "toggle_data_logging":
                data_logger.enabled = not data_logger.enabled
                ui.state['data_logging'] = data_logger.enabled
                status = "启动" if data_logger.enabled else "停止"
                print(f"数据记录: {status}")
                voice.speak(f"数据记录{status}")
                ui.add_voice_message(f"数据记录{status}")

            elif event == "save_data":
                if data_logger.records:
                    if data_logger.save_to_file():
                        voice.speak("数据保存成功")
                        ui.add_voice_message("数据保存成功")
                    else:
                        voice.speak("数据保存失败")
                        ui.add_voice_message("数据保存失败")

            elif event == "playback_data":
                if data_logger.records:
                    print("开始回放数据...")
                    voice.speak("开始回放数据")
                    ui.add_voice_message("开始回放数据")
                    # 这里可以添加数据回放逻辑

            elif event == "mode_manual":
                tracking_mode = "手动"
                print("模式: 手动控制")
            elif event == "mode_tracking":
                tracking_mode = "追踪"
                print("模式: 自动追踪")
            elif event == "mode_hover":
                print("模式: 悬停")
            elif event == "toggle_detection":
                detection_active = not detection_active
                status = "开启" if detection_active else "关闭"
                print(f"检测: {status}")
                voice.speak(f"检测功能{status}")
                ui.add_voice_message(f"检测功能{status}")
            elif event == "toggle_recognition":
                recognition_active = not recognition_active
                status = "开启" if recognition_active else "关闭"
                print(f"识别: {status}")
                voice.speak(f"识别功能{status}")
                ui.add_voice_message(f"识别功能{status}")

            # 无人机移动控制
            if is_flying:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_w]:
                    drone_position[0] += 0.1
                    data_logger.log_control_action("move_forward", {"distance": 0.1})
                if keys[pygame.K_s]:
                    drone_position[0] -= 0.1
                    data_logger.log_control_action("move_backward", {"distance": 0.1})
                if keys[pygame.K_a]:
                    drone_position[1] -= 0.1
                    data_logger.log_control_action("move_left", {"distance": 0.1})
                if keys[pygame.K_d]:
                    drone_position[1] += 0.1
                    data_logger.log_control_action("move_right", {"distance": 0.1})
                if keys[pygame.K_SPACE]:
                    drone_position[2] += 0.05
                    data_logger.log_control_action("move_up", {"distance": 0.05})
                if keys[pygame.K_LCTRL]:
                    drone_position[2] = max(0.5, drone_position[2] - 0.05)
                    data_logger.log_control_action("move_down", {"distance": 0.05})
                if keys[pygame.K_q] and keys[pygame.K_LSHIFT]:  # Shift+Q旋转左
                    drone_yaw = (drone_yaw + 2.0) % 360
                    data_logger.log_control_action("rotate_left", {"angle": 2.0})
                if keys[pygame.K_e]:  # E键旋转右
                    drone_yaw = (drone_yaw - 2.0) % 360
                    data_logger.log_control_action("rotate_right", {"angle": 2.0})

            # 更新3D视图
            if drone_3d.window_open:
                # 在实际应用中，这里会更新3D模型的位置和姿态
                pass

            # 获取画面
            frame = drone.get_frame()

            if frame is None:
                ui.update_empty()
                time.sleep(0.1)
                continue

            # 处理画面
            result_frame = frame.copy()
            h, w = frame.shape[:2]

            # 模拟检测结果更新
            if frame_count % 30 == 0:
                if tracking_mode == "追踪" and detection_active:
                    detected_faces = random.randint(1, 3)
                    detected_persons = random.randint(1, 4)
                    if random.random() > 0.5 and recognition_active:
                        names = ["张三", "李四", "王五", "赵六"]
                        recognized_person = random.choice(names)

                        # 语音播报识别结果
                        if voice.enabled:
                            voice.speak(f"识别到{recognized_person}")
                            ui.add_voice_message(f"识别到{recognized_person}")
                    else:
                        recognized_person = "Unknown"
                else:
                    detected_faces = random.randint(0, 2)
                    detected_persons = random.randint(0, 3)
                    recognized_person = "Unknown"

                # 语音播报检测结果
                if voice.enabled and detection_active and (detected_faces > 0 or detected_persons > 0):
                    voice.speak(f"检测到{detected_faces}个人脸，{detected_persons}个行人")
                    ui.add_voice_message(f"检测到{detected_faces}个人脸，{detected_persons}个行人")

                # 数据记录
                if data_logger.enabled and detection_active:
                    data_logger.log_detection_result(detected_faces, detected_persons, recognized_person)

            # 在画面上添加信息
            if tracking_mode == "追踪":
                cv2.putText(result_frame, "追踪中", (w - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            info_y = 30
            if detected_faces > 0:
                cv2.putText(result_frame, f"人脸: {detected_faces}", (10, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                info_y += 30

            if detected_persons > 0:
                cv2.putText(result_frame, f"行人: {detected_persons}", (10, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                info_y += 30

            if recognized_person != "Unknown":
                cv2.putText(result_frame, f"识别: {recognized_person}", (10, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 添加功能状态
            status_y = h - 80
            if drone_3d.window_open:
                cv2.putText(result_frame, "3D窗口: 开启", (w - 150, status_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
                status_y += 20

            if voice.enabled:
                cv2.putText(result_frame, "语音: 启用", (w - 150, status_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 1)
                status_y += 20

            if data_logger.enabled:
                cv2.putText(result_frame, "数据: 记录中", (w - 150, status_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 200), 1)

            # 添加帧编号
            cv2.putText(result_frame, f"帧: {frame_count}", (w - 100, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # 更新无人机位置（模拟漂移）
            if is_flying:
                drone_position[0] += (random.random() - 0.5) * 0.02
                drone_position[1] += (random.random() - 0.5) * 0.02
                drone_yaw = (drone_yaw + 0.2) % 360

                # 更新飞行时间
                if flight_start_time > 0:
                    ui.state['flight_time'] = total_flight_time + (time.time() - flight_start_time)
                else:
                    ui.state['flight_time'] = total_flight_time

            # 数据记录：无人机状态
            if data_logger.enabled:
                data_logger.log_drone_state(tuple(drone_position), drone_yaw, is_flying, tracking_mode)

            # 更新UI状态
            ui.update_drone_state({
                'drone_status': '已连接',
                'is_flying': is_flying,
                'tracking_mode': tracking_mode,
                'drone_position': tuple(drone_position),
                'drone_yaw': drone_yaw,
                'detected_faces': detected_faces,
                'detected_persons': detected_persons,
                'recognized_person': recognized_person,
                'camera_status': f"{w}x{h} @ 30fps",
                'detection_active': detection_active,
                'recognition_active': recognition_active,
                'flight_time': ui.state.get('flight_time', 0),
                'total_distance': ui.state.get('total_distance', 0),
                'drone_3d_open': drone_3d.window_open,
            })

            # 更新UI
            ui.update_lightweight(result_frame)

            # 每100帧打印状态
            if frame_count % 100 == 0:
                print(f"运行中... 帧数: {frame_count}, FPS: {ui.state['fps']}")
                if detection_active:
                    print(f"  检测: {detected_faces}人脸, {detected_persons}行人")
                print(f"  位置: ({drone_position[0]:.1f}, {drone_position[1]:.1f}, {drone_position[2]:.1f})")

                # 显示数据记录统计
                if data_logger.enabled:
                    stats = data_logger.get_statistics()
                    print(f"  数据: {stats['total_records']}条记录")

    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n🔄 清理资源...")

        # 停止3D查看器
        drone_3d.stop_viewer()

        # 停止语音
        voice.stop()

        # 停止数据记录并保存
        if data_logger.records:
            data_logger.save_to_file()

            # 显示统计信息
            stats = data_logger.get_statistics()
            print(f"📊 飞行数据统计:")
            print(f"  总记录数: {stats['total_records']}")
            print(f"  飞行时间: {stats['flight_time']}秒")
            print(f"  无人机状态记录: {stats['drone_states']}")
            print(f"  检测记录: {stats['detections']}")
            print(f"  控制记录: {stats['controls']}")

        # 断开连接
        drone.disconnect()
        ui.quit()
        cv2.destroyAllWindows()

        print("\n✅ 程序已安全退出")
        print("=" * 60)


if __name__ == "__main__":
    main()