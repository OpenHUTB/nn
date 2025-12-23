import pygame
import numpy as np


class PyGameDrawer:
    def __init__(self, main):
        self.main = main
        self.screen = main.game.screen
        self.font = pygame.font.SysFont('Arial', 32)
        self.small_font = pygame.font.SysFont('Arial', 20)
        self.warning_font = pygame.font.SysFont('Arial', 24, bold=True)

        # 摄像头图像存储
        self.camera_image = None

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

    def display_camera(self):
        """显示摄像头图像 - 新增功能"""
        if self.camera_image is not None:
            try:
                # 摄像头图像显示在右上角
                cam_x = self.screen.get_width() - 420  # 右边距20，宽度400
                cam_y = 20  # 上边距

                # 确保图像是有效的numpy数组
                if self.camera_image.shape[0] > 0 and self.camera_image.shape[1] > 0:
                    # 将numpy数组转换为Pygame表面
                    # 注意：numpy数组是(height, width, 3)格式，需要转置为(width, height, 3)
                    image_surface = pygame.surfarray.make_surface(self.camera_image.swapaxes(0, 1))

                    # 调整图像大小以适应显示区域
                    target_width = 400
                    target_height = 300
                    image_surface = pygame.transform.scale(image_surface, (target_width, target_height))

                    # 绘制图像
                    self.screen.blit(image_surface, (cam_x, cam_y))

                    # 绘制边框和标题
                    pygame.draw.rect(self.screen, (100, 100, 100),
                                     (cam_x - 2, cam_y - 2, target_width + 4, target_height + 4),
                                     2, border_radius=5)

                    # 添加摄像头标签
                    camera_label = self.small_font.render("摄像头视图", True, (255, 255, 255))
                    self.screen.blit(camera_label, (cam_x + 150, cam_y + target_height + 10))

            except Exception as e:
                # 如果绘制失败，静默处理（避免影响主程序）
                # 可以在调试时取消注释下面的打印
                # print(f"绘制摄像头图像失败: {e}")
                pass

    def draw_camera(self, image_array):
        """绘制摄像头图像（兼容旧代码）"""
        # 将图像存储起来供display_camera使用
        self.camera_image = image_array

    def draw_lidar(self, point_cloud):
        """绘制激光雷达点云（如果需要）"""
        # 这里可以添加点云的绘制逻辑
        pass