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
                f'Location: ({location.x:.1f}, {location.y:.1f}, {location.z:.1f})',
                True, (255, 255, 255)
            )
            self.screen.blit(location_text, (20, 70))

    def draw_camera(self, image_array):
        """绘制摄像头图像（如果需要）"""
        # 这里可以添加摄像头图像的绘制逻辑
        pass

    def draw_lidar(self, point_cloud):
        """绘制激光雷达点云（如果需要）"""
        # 这里可以添加点云的绘制逻辑
        pass