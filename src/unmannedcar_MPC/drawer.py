import carla
import config as Config
import numpy as np
import math
import time


class PyGameDrawer():

    def __init__(self, main):
        self.main = main
        self.pygame = main.game.pygame
        self.camera = main.game.camera
        self.font_14 = self.pygame.freetype.SysFont('Times New Roman', 14)

        # 速度显示相关字体
        self.speed_font_large = self.pygame.freetype.SysFont('Arial', 36)
        self.speed_font_small = self.pygame.freetype.SysFont('Arial', 18)

        # 刹车显示相关
        self.brake_font = self.pygame.freetype.SysFont('Arial', 24)

        # 转向显示相关字体
        self.steer_font = self.pygame.freetype.SysFont('Arial', 20)

        # 通用信息字体
        self.info_font = self.pygame.freetype.SysFont('Arial', 16)

        # 初始化时间
        self.start_time = time.time()
        self.frame_count = 0  # 帧计数器

    # draw on the camera perspective

    def __w_locs_2_camera_locs(self, w_locs):
        camera_locs = []
        for w_loc in w_locs:
            bbox = PyGameDrawer.get_location_bbox(w_loc, self.camera)
            if math.isnan(bbox[0, 0]) or math.isnan(bbox[0, 1]):
                camera_locs.append((-1, -1))
            camera_locs.append((int(bbox[0, 0]), int(bbox[0, 1])))
        return camera_locs

    def draw_camera_text(self, location, color, text):
        x, y = self.__w_locs_2_camera_locs([location])[0]
        if x >= 0 and x <= Config.PYGAME_WIDTH and y >= 0 and y <= Config.PYGAME_HEIGHT:
            self.font_14.render_to(self.main.surface, (x, y), text, color)

    def draw_camera_circles(self, w_locs, color, radius):
        cam_locs = self.__w_locs_2_camera_locs(w_locs)
        for cam_loc in cam_locs:
            self.pygame.draw.circle(
                self.main.surface, color, cam_loc, radius, 1)

    def draw_camera_polygon(self, w_locs, color):
        if len(w_locs) < 3:
            return
        points = self.__w_locs_2_camera_locs(w_locs)
        self.pygame.draw.polygon(self.main.surface, color, points, 4)

    def draw_camera_lines(self, color, w_locs, width=1):
        cam_locs = self.__w_locs_2_camera_locs(w_locs)
        for i in range(len(cam_locs) - 1):
            self.__draw_camera_line_safe(color, [cam_locs[i][0], cam_locs[i][1]], [
                cam_locs[i + 1][0], cam_locs[i + 1][1]], width)

    def __draw_camera_line_safe(self, color, pt1, pt2, width=1):
        screen_width = Config.PYGAME_WIDTH
        screen_height = Config.PYGAME_HEIGHT
        if (pt1[0] >= 0 and pt1[0] <= screen_width and pt1[1] >= 0 and pt1[1] <= screen_height and
                pt2[0] >= 0 and pt2[0] <= screen_width and pt2[1] >= 0 and pt2[1] <= screen_height):
            self.pygame.draw.line(self.main.surface, color, pt1, pt2, width)

    # 绘制点的方法
    def draw_point(self, location, color, radius=3):
        """在相机视角下绘制一个点"""
        cam_loc = self.__w_locs_2_camera_locs([location])[0]
        screen_width = Config.PYGAME_WIDTH
        screen_height = Config.PYGAME_HEIGHT
        if cam_loc[0] >= 0 and cam_loc[0] <= screen_width and cam_loc[1] >= 0 and cam_loc[1] <= screen_height:
            self.pygame.draw.circle(self.main.surface, color, cam_loc, radius)

    # 显示速度方法
    def display_speed(self, speed_kmh):
        """在屏幕右上角显示当前速度"""
        # 设置速度显示位置
        screen_width = Config.PYGAME_WIDTH
        pos_x = screen_width - 180  # 屏幕右侧
        pos_y = 30  # 距离顶部30像素

        # 根据速度设置颜色
        if speed_kmh < 30:
            color = (0, 255, 0)  # 绿色 - 低速
        elif speed_kmh < 60:
            color = (255, 255, 0)  # 黄色 - 中速
        elif speed_kmh < 90:
            color = (255, 165, 0)  # 橙色 - 中高速
        else:
            color = (255, 0, 0)  # 红色 - 高速

        # 显示速度值（大字体）
        speed_text = f"{speed_kmh:.1f}"
        self.speed_font_large.render_to(self.main.surface, (pos_x, pos_y), speed_text, color)

        # 显示单位（小字体）
        unit_text = "km/h"
        self.speed_font_small.render_to(self.main.surface, (pos_x + 100, pos_y + 15), unit_text, (200, 200, 200))

        # 绘制速度条背景
        bar_width = 150
        bar_height = 10
        bar_x = pos_x
        bar_y = pos_y + 50

        # 绘制速度条背景
        bar_bg_rect = self.pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        self.pygame.draw.rect(self.main.surface, (50, 50, 50), bar_bg_rect)

        # 绘制速度条填充（根据速度比例）
        speed_ratio = min(speed_kmh / 120.0, 1.0)  # 假设最大速度120 km/h
        bar_filled_width = int(bar_width * speed_ratio)
        bar_filled_rect = self.pygame.Rect(bar_x, bar_y, bar_filled_width, bar_height)
        self.pygame.draw.rect(self.main.surface, color, bar_filled_rect)

        # 绘制速度条边框
        self.pygame.draw.rect(self.main.surface, (255, 255, 255), bar_bg_rect, 1)

    # 显示刹车状态方法
    def display_brake_status(self, is_braking, brake_history, target_speed, frame_count):
        """在屏幕左上角显示刹车状态"""
        self.frame_count = frame_count  # 更新帧计数

        # 设置显示位置（屏幕左上角）
        pos_x = 30
        pos_y = 30

        # 测试模式：在前200帧强制显示刹车状态，让用户能看到效果
        if frame_count < 200:
            # 测试模式：每50帧切换一次状态，演示效果
            test_braking = (frame_count // 50) % 2 == 0
            title_text = f"BRAKE STATUS (TEST MODE) - Target: {target_speed} km/h"
            self.brake_font.render_to(self.main.surface, (pos_x, pos_y), title_text, (200, 200, 200))

            if test_braking:
                # 测试刹车状态：红色闪烁
                brake_text = "BRAKING (TEST)"
                intensity = 200 + 55 * ((frame_count // 5) % 2)  # 闪烁效果

                # 绘制红色背景框
                bg_rect = self.pygame.Rect(pos_x - 10, pos_y + 30, 180, 40)
                self.pygame.draw.rect(self.main.surface, (intensity // 4, 0, 0), bg_rect)
                self.pygame.draw.rect(self.main.surface, (intensity, 0, 0), bg_rect, 3)
                self.brake_font.render_to(self.main.surface, (pos_x + 10, pos_y + 45), brake_text,
                                          (intensity, intensity // 3, intensity // 3))
            else:
                # 测试正常状态：绿色
                brake_text = "NORMAL (TEST)"
                bg_rect = self.pygame.Rect(pos_x - 10, pos_y + 30, 180, 40)
                self.pygame.draw.rect(self.main.surface, (0, 30, 0), bg_rect)
                self.pygame.draw.rect(self.main.surface, (0, 180, 0), bg_rect, 2)
                self.brake_font.render_to(self.main.surface, (pos_x + 10, pos_y + 45), brake_text, (0, 255, 100))

            # 添加测试模式说明
            self.info_font.render_to(self.main.surface, (pos_x, pos_y + 80),
                                     "Test Mode: Forced display of brake states", (255, 255, 0))
            return

        # 正常模式
        # 检查是否在刹车（使用历史记录创建闪烁效果）
        should_flash = False
        if len(brake_history) >= 5:
            # 如果最近5帧中有3帧在刹车，则显示刹车状态
            recent_brakes = brake_history[-5:]
            if sum(recent_brakes) >= 3:
                should_flash = True

        # 显示标题和目标速度
        title_text = f"BRAKE STATUS - Target: {target_speed} km/h"
        self.brake_font.render_to(self.main.surface, (pos_x, pos_y), title_text, (200, 200, 200))

        if is_braking or should_flash:
            # 刹车状态：红色闪烁
            brake_text = "BRAKING"

            # 闪烁效果：根据时间改变亮度
            intensity = 200 + 55 * ((frame_count // 5) % 2)  # 每5帧闪烁一次

            # 绘制红色背景框
            bg_rect = self.pygame.Rect(pos_x - 10, pos_y + 30, 150, 40)
            self.pygame.draw.rect(self.main.surface, (intensity // 4, 0, 0), bg_rect)

            # 绘制红色边框
            self.pygame.draw.rect(self.main.surface, (intensity, 0, 0), bg_rect, 3)

            # 显示"BRAKING"文字
            self.brake_font.render_to(self.main.surface, (pos_x + 10, pos_y + 45), brake_text,
                                      (intensity, intensity // 3, intensity // 3))

            # 添加警告符号
            warning_color = (intensity, intensity, 0)  # 黄色警告
            self.pygame.draw.circle(self.main.surface, warning_color, (pos_x + 120, pos_y + 50), 12)
            warning_font = self.pygame.freetype.SysFont('Arial', 16)
            warning_font.render_to(self.main.surface, (pos_x + 115, pos_y + 44), "!", (0, 0, 0))
        else:
            # 正常状态：绿色
            brake_text = "NORMAL"

            # 绘制绿色背景框
            bg_rect = self.pygame.Rect(pos_x - 10, pos_y + 30, 150, 40)
            self.pygame.draw.rect(self.main.surface, (0, 30, 0), bg_rect)

            # 绘制绿色边框
            self.pygame.draw.rect(self.main.surface, (0, 180, 0), bg_rect, 2)

            # 显示"NORMAL"文字
            self.brake_font.render_to(self.main.surface, (pos_x + 10, pos_y + 45), brake_text, (0, 255, 100))

    # 显示速度历史图表
    def display_speed_history(self, speed_history, target_speed):
        """在屏幕左下角显示速度历史图表"""
        if len(speed_history) < 2:
            return

        # 图表位置和大小
        screen_height = Config.PYGAME_HEIGHT
        chart_x = 30
        chart_y = screen_height - 150
        chart_width = 300
        chart_height = 120

        # 绘制图表背景
        chart_bg_rect = self.pygame.Rect(chart_x, chart_y, chart_width, chart_height)
        self.pygame.draw.rect(self.main.surface, (20, 20, 20), chart_bg_rect)
        self.pygame.draw.rect(self.main.surface, (100, 100, 100), chart_bg_rect, 2)

        # 绘制图表标题
        title_font = self.pygame.freetype.SysFont('Arial', 16)
        title_font.render_to(self.main.surface, (chart_x + 5, chart_y - 20), "SPEED HISTORY", (200, 200, 200))

        # 计算速度和目标速度的最小值、最大值
        all_speeds = speed_history + [target_speed]
        min_speed = min(all_speeds) - 5
        max_speed = max(all_speeds) + 5

        # 绘制目标速度线
        if min_speed <= target_speed <= max_speed:
            target_y = chart_y + chart_height - int((target_speed - min_speed) / (max_speed - min_speed) * chart_height)
            self.pygame.draw.line(
                self.main.surface,
                (0, 255, 0),  # 绿色目标线
                (chart_x, target_y),
                (chart_x + chart_width, target_y),
                2
            )

            # 标注目标速度值
            target_font = self.pygame.freetype.SysFont('Arial', 12)
            target_font.render_to(self.main.surface, (chart_x + chart_width + 5, target_y - 10),
                                  f"Target: {target_speed} km/h", (0, 255, 0))

        # 绘制速度历史曲线
        points = []
        for i, speed in enumerate(speed_history):
            if i >= chart_width:  # 只显示最近chart_width个数据点
                speed_subset = speed_history[-chart_width:]
                break

            x = chart_x + int(i * chart_width / min(len(speed_history), chart_width))
            y = chart_y + chart_height - int((speed - min_speed) / (max_speed - min_speed) * chart_height)
            points.append((x, y))

        # 连接点成线
        if len(points) > 1:
            # 速度线：蓝色
            self.pygame.draw.lines(self.main.surface, (100, 150, 255), False, points, 2)

            # 绘制最后一个点（当前速度）
            if points:
                last_point = points[-1]
                self.pygame.draw.circle(self.main.surface, (255, 255, 255), last_point, 4)

                # 标注当前速度值
                current_speed = speed_history[-1]
                speed_font = self.pygame.freetype.SysFont('Arial', 12)
                speed_font.render_to(
                    self.main.surface,
                    (last_point[0] + 10, last_point[1] - 10),
                    f"{current_speed:.1f} km/h",
                    (255, 255, 255)
                )

    # 显示转向角度功能
    def display_steering(self, steer_angle):
        """在屏幕右下角显示转向角度"""
        # 获取屏幕尺寸
        screen_width = Config.PYGAME_WIDTH
        screen_height = Config.PYGAME_HEIGHT

        # 设置在屏幕右下角
        pos_x = screen_width - 220
        pos_y = 120

        # 将转向角度转换为度数和可视化角度
        angle_degrees = steer_angle * 45  # 假设-1到1对应-45度到45度

        # 根据转向角度设置颜色
        if abs(angle_degrees) < 5:
            color = (0, 255, 0)  # 绿色 - 直行或小角度
        elif abs(angle_degrees) < 15:
            color = (255, 255, 0)  # 黄色 - 中等角度
        else:
            color = (255, 100, 0)  # 橙色 - 大角度

        # 显示标题
        title_text = "STEERING ANGLE"
        self.steer_font.render_to(self.main.surface, (pos_x, pos_y), title_text, (200, 200, 200))

        # 显示角度值
        angle_text = f"{angle_degrees:+.1f}°"
        self.steer_font.render_to(self.main.surface, (pos_x, pos_y + 30), angle_text, color)

        # 显示原始值
        raw_text = f"Raw: {steer_angle:+.3f}"
        self.info_font.render_to(self.main.surface, (pos_x, pos_y + 60), raw_text, (150, 150, 150))

        # 绘制转向可视化指示器
        indicator_width = 180
        indicator_height = 40
        indicator_x = pos_x - 10
        indicator_y = pos_y + 90

        # 绘制背景
        indicator_bg = self.pygame.Rect(indicator_x, indicator_y, indicator_width, indicator_height)
        self.pygame.draw.rect(self.main.surface, (40, 40, 40), indicator_bg)
        self.pygame.draw.rect(self.main.surface, (100, 100, 100), indicator_bg, 2)

        # 绘制中心线
        center_x = indicator_x + indicator_width // 2
        self.pygame.draw.line(
            self.main.surface,
            (200, 200, 200),
            (center_x, indicator_y),
            (center_x, indicator_y + indicator_height),
            2
        )

        # 绘制转向指示器
        indicator_center = center_x + int((indicator_width // 2 - 10) * steer_angle)
        indicator_radius = 12

        # 绘制指示器圆圈
        self.pygame.draw.circle(self.main.surface, color, (indicator_center, indicator_y + indicator_height // 2),
                                indicator_radius)

        # 绘制方向箭头
        arrow_size = 8
        if steer_angle > 0.01:  # 右转
            arrow_points = [
                (indicator_center - arrow_size, indicator_y + indicator_height // 2 - arrow_size),
                (indicator_center - arrow_size, indicator_y + indicator_height // 2 + arrow_size),
                (indicator_center, indicator_y + indicator_height // 2)
            ]
        elif steer_angle < -0.01:  # 左转
            arrow_points = [
                (indicator_center + arrow_size, indicator_y + indicator_height // 2 - arrow_size),
                (indicator_center + arrow_size, indicator_y + indicator_height // 2 + arrow_size),
                (indicator_center, indicator_y + indicator_height // 2)
            ]
        else:  # 直行
            arrow_points = [
                (indicator_center - arrow_size, indicator_y + indicator_height // 2),
                (indicator_center + arrow_size, indicator_y + indicator_height // 2),
                (indicator_center, indicator_y + indicator_height // 2 + arrow_size)
            ]

        self.pygame.draw.polygon(self.main.surface, (255, 255, 255), arrow_points)

        # 添加标签
        label_font = self.pygame.freetype.SysFont('Arial', 12)
        label_font.render_to(self.main.surface, (indicator_x + 5, indicator_y + indicator_height + 5), "LEFT",
                             (150, 150, 150))
        label_font.render_to(self.main.surface,
                             (indicator_x + indicator_width - 40, indicator_y + indicator_height + 5), "RIGHT",
                             (150, 150, 150))
        label_font.render_to(self.main.surface, (center_x - 20, indicator_y + indicator_height + 5), "CENTER",
                             (150, 150, 150))

    # 显示油门和刹车信息
    def display_throttle_info(self, throttle_value, brake_value):
        """在屏幕右侧中部显示油门和刹车信息"""
        # 设置显示位置
        screen_width = Config.PYGAME_WIDTH
        pos_x = screen_width - 220
        pos_y = 250

        # 显示标题
        title_text = "CONTROL INPUTS"
        self.steer_font.render_to(self.main.surface, (pos_x, pos_y), title_text, (200, 200, 200))

        # 油门显示
        throttle_text = f"Throttle: {throttle_value:.2f}"
        throttle_color = (0, int(255 * throttle_value), 0)  # 绿色，亮度随油门变化
        self.info_font.render_to(self.main.surface, (pos_x, pos_y + 30), throttle_text, throttle_color)

        # 油门条
        throttle_bar_width = 150
        throttle_bar_height = 10
        throttle_bar_x = pos_x
        throttle_bar_y = pos_y + 50

        throttle_bar_bg = self.pygame.Rect(throttle_bar_x, throttle_bar_y, throttle_bar_width, throttle_bar_height)
        self.pygame.draw.rect(self.main.surface, (50, 50, 50), throttle_bar_bg)

        throttle_filled_width = int(throttle_bar_width * throttle_value)
        throttle_filled_rect = self.pygame.Rect(throttle_bar_x, throttle_bar_y, throttle_filled_width,
                                                throttle_bar_height)
        self.pygame.draw.rect(self.main.surface, throttle_color, throttle_filled_rect)
        self.pygame.draw.rect(self.main.surface, (255, 255, 255), throttle_bar_bg, 1)

        # 刹车显示
        brake_text = f"Brake: {brake_value:.2f}"
        brake_color = (int(255 * brake_value), 0, 0)  # 红色，亮度随刹车力度变化
        self.info_font.render_to(self.main.surface, (pos_x, pos_y + 70), brake_text, brake_color)

        # 刹车条
        brake_bar_width = 150
        brake_bar_height = 10
        brake_bar_x = pos_x
        brake_bar_y = pos_y + 90

        brake_bar_bg = self.pygame.Rect(brake_bar_x, brake_bar_y, brake_bar_width, brake_bar_height)
        self.pygame.draw.rect(self.main.surface, (50, 50, 50), brake_bar_bg)

        brake_filled_width = int(brake_bar_width * brake_value)
        brake_filled_rect = self.pygame.Rect(brake_bar_x, brake_bar_y, brake_filled_width, brake_bar_height)
        self.pygame.draw.rect(self.main.surface, brake_color, brake_filled_rect)
        self.pygame.draw.rect(self.main.surface, (255, 255, 255), brake_bar_bg, 1)

    # 显示控制模式
    def display_control_mode(self, control_mode):
        """在屏幕顶部中央显示控制模式"""
        screen_width = Config.PYGAME_WIDTH
        pos_x = screen_width // 2 - 100
        pos_y = 10

        if control_mode == "AUTO":
            color = (0, 200, 255)  # 青色
        elif control_mode == "MANUAL":
            color = (255, 200, 0)  # 橙色
        else:
            color = (255, 255, 255)  # 白色

        # 绘制背景框
        bg_rect = self.pygame.Rect(pos_x - 10, pos_y, 220, 40)
        self.pygame.draw.rect(self.main.surface, (20, 20, 20, 128), bg_rect)
        self.pygame.draw.rect(self.main.surface, color, bg_rect, 2)

        # 显示文本
        mode_text = f"CONTROL MODE: {control_mode}"
        mode_font = self.pygame.freetype.SysFont('Arial', 24)
        mode_font.render_to(self.main.surface, (pos_x, pos_y + 10), mode_text, color)

    # 显示帧信息
    def display_frame_info(self, frame_count, dt):
        """在屏幕左下角显示帧信息"""
        screen_height = Config.PYGAME_HEIGHT
        pos_x = 30
        pos_y = screen_height - 200

        # 计算FPS
        fps = 1.0 / dt if dt > 0 else 0

        # 计算运行时间
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # 显示信息
        info_texts = [
            f"Frame: {frame_count}",
            f"FPS: {fps:.1f}",
            f"Time: {elapsed_time:.1f}s",
            f"DT: {dt:.3f}s"
        ]

        # 绘制背景
        bg_rect = self.pygame.Rect(pos_x - 10, pos_y - 10, 150, 100)
        self.pygame.draw.rect(self.main.surface, (20, 20, 20, 128), bg_rect)
        self.pygame.draw.rect(self.main.surface, (100, 100, 100), bg_rect, 1)

        # 显示每行信息
        for i, text in enumerate(info_texts):
            self.info_font.render_to(self.main.surface, (pos_x, pos_y + i * 20), text, (200, 200, 200))

    # 显示碰撞警告
    def display_collision_warning(self, collision_warning, collision_history):
        """在屏幕中央上方显示碰撞警告"""
        screen_width = Config.PYGAME_WIDTH
        screen_height = Config.PYGAME_HEIGHT

        # 设置显示位置（屏幕中央上方）
        pos_x = screen_width // 2 - 150
        pos_y = 150

        # 检查是否需要显示警告（使用历史记录减少闪烁）
        should_warn = False
        if len(collision_history) >= 10:
            # 如果最近10帧中有7帧检测到碰撞风险，则显示警告
            recent_warnings = collision_history[-10:]
            if sum(recent_warnings) >= 7:
                should_warn = True

        if collision_warning or should_warn:
            # 碰撞警告：红色闪烁
            warning_text = "COLLISION WARNING!"

            # 闪烁效果
            intensity = 200 + 55 * ((self.frame_count // 3) % 2)  # 快速闪烁

            # 绘制警告背景
            bg_rect = self.pygame.Rect(pos_x - 20, pos_y - 10, 340, 60)

            # 绘制红色渐变背景
            for i in range(bg_rect.height):
                # 创建渐变红色
                alpha = int(100 * (1.0 - i / bg_rect.height))
                color = (intensity, 0, 0, alpha)
                line_rect = self.pygame.Rect(bg_rect.x, bg_rect.y + i, bg_rect.width, 1)
                self.pygame.draw.rect(self.main.surface, color, line_rect)

            # 绘制边框（闪烁）
            border_color = (intensity, intensity // 2, 0)
            self.pygame.draw.rect(self.main.surface, border_color, bg_rect, 4)

            # 显示警告文字
            warning_font = self.pygame.freetype.SysFont('Arial', 32)
            warning_font.render_to(self.main.surface, (pos_x, pos_y), warning_text, (intensity, intensity, 0))

            # 添加警告图标
            icon_x = pos_x + 280
            icon_y = pos_y + 20
            self.pygame.draw.circle(self.main.surface, (intensity, 0, 0), (icon_x, icon_y), 20)

            # 绘制感叹号
            exclamation_font = self.pygame.freetype.SysFont('Arial', 24)
            exclamation_font.render_to(self.main.surface, (icon_x - 5, icon_y - 15), "!", (255, 255, 255))

            # 添加副标题
            subtitle_text = "Reduce Speed and Steer Carefully"
            subtitle_font = self.pygame.freetype.SysFont('Arial', 16)
            subtitle_font.render_to(self.main.surface, (pos_x, pos_y + 40), subtitle_text, (255, 200, 0))

            # 绘制警告脉冲效果
            pulse_radius = 15 + 5 * math.sin(self.frame_count * 0.2)
            self.pygame.draw.circle(self.main.surface, (intensity, 0, 0, 100), (icon_x, icon_y), int(pulse_radius), 2)

    # 显示驾驶评分
    def display_driving_score(self, score, score_factors, score_history):
        """在屏幕左侧中部显示驾驶评分"""
        screen_width = Config.PYGAME_WIDTH
        screen_height = Config.PYGAME_HEIGHT

        # 设置显示位置
        pos_x = 30
        pos_y = 180

        # 根据评分设置颜色
        if score >= 85:
            color = (0, 255, 0)  # 优秀 - 绿色
            grade = "A"
        elif score >= 70:
            color = (255, 255, 0)  # 良好 - 黄色
            grade = "B"
        elif score >= 60:
            color = (255, 165, 0)  # 及格 - 橙色
            grade = "C"
        else:
            color = (255, 0, 0)  # 差 - 红色
            grade = "D"

        # 绘制评分背景框
        bg_rect = self.pygame.Rect(pos_x - 10, pos_y - 10, 300, 200)
        self.pygame.draw.rect(self.main.surface, (20, 20, 20, 200), bg_rect)
        self.pygame.draw.rect(self.main.surface, (100, 100, 100), bg_rect, 2)

        # 显示标题
        title_text = "DRIVING PERFORMANCE"
        self.steer_font.render_to(self.main.surface, (pos_x, pos_y), title_text, (200, 200, 200))

        # 显示综合评分
        score_text = f"Score: {score:.1f}/100 ({grade})"
        self.steer_font.render_to(self.main.surface, (pos_x, pos_y + 35), score_text, color)

        # 显示各项评分因子
        y_offset = 75
        factor_colors = {
            'speed_stability': (100, 200, 255),  # 蓝色
            'steering_smoothness': (255, 200, 100),  # 橙色
            'brake_usage': (255, 100, 100),  # 红色
            'path_following': (100, 255, 150),  # 绿色
            'safety': (200, 100, 255)  # 紫色
        }

        factor_labels = {
            'speed_stability': "Speed Stability",
            'steering_smoothness': "Steering Smooth",
            'brake_usage': "Brake Usage",
            'path_following': "Path Following",
            'safety': "Safety"
        }

        for factor, label in factor_labels.items():
            factor_score = score_factors.get(factor, 0)
            factor_color = factor_colors.get(factor, (200, 200, 200))

            # 显示因子标签和分数
            factor_text = f"{label}: {factor_score:.1f}"
            self.info_font.render_to(self.main.surface, (pos_x, pos_y + y_offset), factor_text, factor_color)

            # 绘制分数条
            bar_width = 150
            bar_height = 8
            bar_x = pos_x + 120
            bar_y = pos_y + y_offset + 5

            # 绘制背景条
            bar_bg_rect = self.pygame.Rect(bar_x, bar_y, bar_width, bar_height)
            self.pygame.draw.rect(self.main.surface, (50, 50, 50), bar_bg_rect)

            # 绘制填充条
            filled_width = int(bar_width * factor_score / 100)
            filled_rect = self.pygame.Rect(bar_x, bar_y, filled_width, bar_height)
            self.pygame.draw.rect(self.main.surface, factor_color, filled_rect)

            # 绘制边框
            self.pygame.draw.rect(self.main.surface, (150, 150, 150), bar_bg_rect, 1)

            y_offset += 20

        # 绘制评分趋势图
        if len(score_history) >= 2:
            chart_x = pos_x
            chart_y = pos_y + 180
            chart_width = 280
            chart_height = 60

            # 绘制图表背景
            chart_bg = self.pygame.Rect(chart_x, chart_y, chart_width, chart_height)
            self.pygame.draw.rect(self.main.surface, (15, 15, 15), chart_bg)
            self.pygame.draw.rect(self.main.surface, (80, 80, 80), chart_bg, 1)

            # 绘制趋势线
            points = []
            max_history = min(50, len(score_history))  # 最多显示最近50个点

            for i in range(max_history):
                idx = len(score_history) - max_history + i
                if idx >= 0:
                    x = chart_x + int(i * chart_width / max_history)
                    # 分数范围0-100映射到图表高度
                    y = chart_y + chart_height - int(score_history[idx] * chart_height / 100)
                    points.append((x, y))

            if len(points) > 1:
                # 绘制趋势线
                self.pygame.draw.lines(self.main.surface, color, False, points, 2)

                # 绘制当前点
                if points:
                    last_point = points[-1]
                    self.pygame.draw.circle(self.main.surface, color, last_point, 4)
                    self.pygame.draw.circle(self.main.surface, (255, 255, 255), last_point, 2)

            # 添加图表标签
            chart_label = "Score Trend"
            self.info_font.render_to(self.main.surface, (chart_x, chart_y - 15), chart_label, (150, 150, 150))

    # 显示航点导航信息
    def display_waypoint_navigation(self, current_index, waypoints, distance_to_waypoint, reached_count, progress):
        """在屏幕中央下方显示航点导航信息"""
        screen_width = Config.PYGAME_WIDTH
        screen_height = Config.PYGAME_HEIGHT

        # 设置显示位置（屏幕中央下方）
        pos_x = screen_width // 2 - 150
        pos_y = screen_height - 100

        # 绘制背景框
        bg_rect = self.pygame.Rect(pos_x - 15, pos_y - 15, 330, 90)
        self.pygame.draw.rect(self.main.surface, (20, 20, 20, 200), bg_rect)
        self.pygame.draw.rect(self.main.surface, (80, 80, 200), bg_rect, 2)

        # 显示标题
        title_text = "WAYPOINT NAVIGATION"
        title_font = self.pygame.freetype.SysFont('Arial', 20)
        title_font.render_to(self.main.surface, (pos_x, pos_y), title_text, (100, 200, 255))

        if len(waypoints) == 0:
            # 没有航点时显示提示
            no_waypoints_text = "No waypoints available"
            self.info_font.render_to(self.main.surface, (pos_x, pos_y + 30), no_waypoints_text, (150, 150, 150))
            return

        # 显示当前航点信息
        waypoint_info = f"Waypoint: {current_index + 1}/{len(waypoints)}"
        self.info_font.render_to(self.main.surface, (pos_x, pos_y + 30), waypoint_info, (255, 255, 255))

        # 显示到航点的距离
        distance_text = f"Distance: {distance_to_waypoint:.1f}m"
        distance_color = (255, 200, 100) if distance_to_waypoint > 10 else (100, 255, 100)
        self.info_font.render_to(self.main.surface, (pos_x, pos_y + 50), distance_text, distance_color)

        # 显示已到达航点数量
        reached_text = f"Reached: {reached_count}"
        self.info_font.render_to(self.main.surface, (pos_x + 180, pos_y + 30), reached_text, (200, 200, 100))

        # 绘制航点进度条
        progress_bar_width = 200
        progress_bar_height = 8
        progress_bar_x = pos_x
        progress_bar_y = pos_y + 70

        # 绘制进度条背景
        progress_bg_rect = self.pygame.Rect(progress_bar_x, progress_bar_y, progress_bar_width, progress_bar_height)
        self.pygame.draw.rect(self.main.surface, (50, 50, 50), progress_bg_rect)

        # 绘制进度条填充
        filled_width = int(progress_bar_width * progress)
        filled_rect = self.pygame.Rect(progress_bar_x, progress_bar_y, filled_width, progress_bar_height)

        # 根据进度设置颜色
        if progress < 0.33:
            progress_color = (255, 100, 100)  # 红色 - 起始
        elif progress < 0.66:
            progress_color = (255, 200, 100)  # 橙色 - 中间
        else:
            progress_color = (100, 255, 100)  # 绿色 - 接近完成

        self.pygame.draw.rect(self.main.surface, progress_color, filled_rect)
        self.pygame.draw.rect(self.main.surface, (200, 200, 200), progress_bg_rect, 1)

        # 显示进度百分比
        progress_text = f"{progress * 100:.0f}%"
        progress_font = self.pygame.freetype.SysFont('Arial', 12)
        progress_font.render_to(self.main.surface, (progress_bar_x + progress_bar_width + 5, progress_bar_y - 2),
                                progress_text, (200, 200, 200))

        # 在屏幕上绘制航点指示器（小圆点）
        self.draw_waypoint_indicators(waypoints, current_index)

    def draw_waypoint_indicators(self, waypoints, current_index):
        """在屏幕上绘制航点指示器"""
        if len(waypoints) == 0:
            return

        screen_width = Config.PYGAME_WIDTH
        screen_height = Config.PYGAME_HEIGHT

        # 航点指示器区域（屏幕顶部）
        indicator_area_y = 80
        indicator_spacing = screen_width / (len(waypoints) + 1)

        for i, waypoint in enumerate(waypoints):
            # 计算指示器位置
            indicator_x = int((i + 1) * indicator_spacing)
            indicator_y = indicator_area_y

            # 根据航点状态设置颜色
            if i < current_index:
                color = (100, 255, 100, 150)  # 已通过 - 绿色半透明
                radius = 6
            elif i == current_index:
                color = (255, 200, 0)  # 当前目标 - 黄色
                radius = 10
            else:
                color = (200, 200, 200, 100)  # 未到达 - 灰色半透明
                radius = 8

            # 绘制航点指示器
            self.pygame.draw.circle(self.main.surface, color, (indicator_x, indicator_y), radius)

            # 如果是当前航点，添加脉冲效果
            if i == current_index:
                pulse_radius = radius + 3 + 2 * math.sin(self.frame_count * 0.1)
                self.pygame.draw.circle(self.main.surface, (255, 200, 0, 100), (indicator_x, indicator_y),
                                        int(pulse_radius), 2)

                # 添加航点编号
                waypoint_text = f"{i + 1}"
                waypoint_font = self.pygame.freetype.SysFont('Arial', 12)
                waypoint_font.render_to(self.main.surface, (indicator_x - 5, indicator_y - 20), waypoint_text,
                                        (255, 255, 255))

            # 在顶部显示航点导航标题
            if i == 0:
                nav_title = "WAYPOINTS"
                title_font = self.pygame.freetype.SysFont('Arial', 14)
                title_font.render_to(self.main.surface, (screen_width // 2 - 40, indicator_y - 40), nav_title,
                                     (150, 200, 255))

    @staticmethod
    def get_location_bbox(location, camera):
        bb_cords = np.array([[0, 0, 0, 1]])
        cords_x_y_z = PyGameDrawer.location_to_sensor_cords(
            bb_cords, location, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate(
            [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate(
            [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def location_to_sensor_cords(cords, location, sensor):
        world_cord = PyGameDrawer.location_to_world_cords(cords, location)
        sensor_cord = PyGameDrawer._world_to_sensor_cords(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def location_to_world_cords(cords, location):
        bb_transform = carla.Transform(location)
        vehicle_world_matrix = PyGameDrawer.get_matrix(bb_transform)
        world_cords = np.dot(vehicle_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _create_vehicle_bbox_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """
        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor_cords(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """
        world_cord = PyGameDrawer._vehicle_to_world_cords(cords, vehicle)
        sensor_cord = PyGameDrawer._world_to_sensor_cords(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world_cords(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """
        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = PyGameDrawer.get_matrix(bb_transform)
        vehicle_world_matrix = PyGameDrawer.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor_cords(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """
        sensor_world_matrix = PyGameDrawer.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix