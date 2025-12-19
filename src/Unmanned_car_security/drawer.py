import pygame
import numpy as np


class PyGameDrawer:
    def __init__(self, main):
        self.main = main
        self.screen = main.game.screen
        self.font = pygame.font.SysFont('Arial', 32)
        self.small_font = pygame.font.SysFont('Arial', 20)

    def display_speed(self, speed):
        """显示速度在屏幕左上角"""
        # 创建速度文本
        speed_text = self.font.render(f'Speed: {speed:.1f} km/h', True, (255, 255, 255))
        self.screen.blit(speed_text, (20, 20))

    def display_location(self, location):
        """显示位置信息"""
        if location:
            location_text = self.small_font.render(
                f'Location: ({location.x:.1f}, {location.y:.1f})',
                True, (255, 255, 255)
            )
            self.screen.blit(location_text, (20, 70))

    def display_warning(self, warning_message, color, warning_level):
        """显示障碍物警告信息"""
        # 显示警告信息在速度下方
        if warning_message:
            warning_text = self.small_font.render(warning_message, True, color)
            self.screen.blit(warning_text, (20, 110))

            # 显示警告级别
            level_texts = ["安全", "注意", "警告", "危险"]
            level_text = self.small_font.render(f"警告级别: {level_texts[warning_level]}",
                                                True, color)
            self.screen.blit(level_text, (20, 140))

            # 绘制一个简单的状态指示器
            indicator_width = 300
            indicator_height = 20
            indicator_x = 20
            indicator_y = 170

            # 绘制背景条
            pygame.draw.rect(self.screen, (60, 60, 60),
                             (indicator_x, indicator_y, indicator_width, indicator_height),
                             border_radius=5)

            # 根据警告级别绘制不同长度的彩色条
            if warning_level > 0:
                fill_width = int(indicator_width * (warning_level / 3))
                fill_rect = pygame.Rect(indicator_x, indicator_y, fill_width, indicator_height)
                pygame.draw.rect(self.screen, color, fill_rect, border_radius=5)

    def draw_camera(self, image_array):
        """绘制摄像头图像（如果需要）"""
        # 这里可以添加摄像头图像的绘制逻辑
        pass

    def draw_lidar(self, point_cloud):
        """绘制激光雷达点云（如果需要）"""
        # 这里可以添加点云的绘制逻辑
        pass