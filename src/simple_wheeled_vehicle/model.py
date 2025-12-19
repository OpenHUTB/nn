import pygame
import math
import sys
import numpy as np
from typing import Tuple, List

# 初始化pygame
pygame.init()

# 颜色定义
BACKGROUND = (30, 30, 40)
CAR_BODY = (70, 130, 180)
CAR_WHEEL = (50, 50, 60)
SENSOR_COLOR = (255, 200, 0, 150)
TEXT_COLOR = (220, 220, 220)
GRID_COLOR = (60, 60, 70)
OBSTACLE_COLOR = (180, 80, 80)
PATH_COLOR = (100, 200, 100, 150)
BUTTON_COLOR = (80, 80, 100)
BUTTON_HOVER_COLOR = (100, 100, 120)

# 小车参数
CAR_WIDTH = 60
CAR_HEIGHT = 40
WHEEL_RADIUS = 12
WHEEL_WIDTH = 6
SENSOR_RANGE = 150
SENSOR_ANGLE = 45  # 度数

# 创建窗口
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("轮式小车模型模拟器")
clock = pygame.time.Clock()


class Button:
    """按钮类"""

    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = BUTTON_COLOR
        self.hover_color = BUTTON_HOVER_COLOR
        self.current_color = self.color
        self.font = pygame.font.SysFont(None, 24)

    def draw(self, surface):
        pygame.draw.rect(surface, self.current_color, self.rect, border_radius=5)
        pygame.draw.rect(surface, (120, 120, 140), self.rect, 2, border_radius=5)

        text_surf = self.font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, pos):
        if self.rect.collidepoint(pos):
            self.current_color = self.hover_color
            return True
        else:
            self.current_color = self.color
            return False

    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False


class Obstacle:
    """障碍物类"""

    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self, surface):
        pygame.draw.circle(surface, OBSTACLE_COLOR, (self.x, self.y), self.radius)
        pygame.draw.circle(surface, (150, 60, 60), (self.x, self.y), self.radius, 2)


class WheeledCar:
    """轮式小车类"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
        self.angle = 0  # 朝向角度（弧度）
        self.speed = 0
        self.max_speed = 4
        self.angular_speed = 0
        self.max_angular_speed = 0.08
        self.left_wheel_speed = 0
        self.right_wheel_speed = 0
        self.sensor_data = [0, 0, 0]  # 左、前、右传感器
        self.path = []  # 记录轨迹
        self.max_path_length = 100

    def update(self, obstacles):
        """更新小车状态"""
        # 更新位置
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)

        # 更新角度
        self.angle += self.angular_speed

        # 归一化角度到0-2π
        self.angle %= 2 * math.pi

        # 更新传感器数据
        self.update_sensors(obstacles)

        # 记录轨迹
        self.path.append((self.x, self.y))
        if len(self.path) > self.max_path_length:
            self.path.pop(0)

    def update_sensors(self, obstacles):
        """更新传感器数据"""
        sensor_angles = [
            self.angle - math.radians(SENSOR_ANGLE),  # 左传感器
            self.angle,  # 前传感器
            self.angle + math.radians(SENSOR_ANGLE)  # 右传感器
        ]

        for i, sensor_angle in enumerate(sensor_angles):
            min_distance = SENSOR_RANGE

            # 传感器射线终点
            end_x = self.x + SENSOR_RANGE * math.cos(sensor_angle)
            end_y = self.y + SENSOR_RANGE * math.sin(sensor_angle)

            # 检查与所有障碍物的交点
            for obstacle in obstacles:
                distance = self.line_obstacle_intersection(
                    self.x, self.y, end_x, end_y, obstacle
                )
                if distance < min_distance:
                    min_distance = distance

            # 存储传感器数据（归一化到0-1）
            self.sensor_data[i] = 1.0 - (min_distance / SENSOR_RANGE)

    def line_obstacle_intersection(self, x1, y1, x2, y2, obstacle):
        """计算线段与障碍物的交点距离"""
        # 从障碍物对象获取属性
        ox = obstacle.x
        oy = obstacle.y
        radius = obstacle.radius

        # 线段向量
        dx = x2 - x1
        dy = y2 - y1

        # 圆心到线段起点的向量
        fx = x1 - ox
        fy = y1 - oy

        # 二次方程系数
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = (fx * fx + fy * fy) - radius * radius

        # 判别式
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            # 没有交点
            return SENSOR_RANGE

        discriminant = math.sqrt(discriminant)

        # 两个解
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        # 检查交点是否在线段上
        hits = []
        if 0 <= t1 <= 1:
            hits.append(t1)
        if 0 <= t2 <= 1:
            hits.append(t2)

        if not hits:
            return SENSOR_RANGE

        # 返回最近交点的距离
        t = min(hits)
        intersection_x = x1 + t * dx
        intersection_y = y1 + t * dy

        return math.sqrt((intersection_x - x1) ** 2 + (intersection_y - y1) ** 2)

    def set_control(self, forward, backward, left, right):
        """设置控制输入"""
        # 计算线速度
        if forward and not backward:
            self.speed = self.max_speed
        elif backward and not forward:
            self.speed = -self.max_speed / 2
        else:
            self.speed = 0

        # 计算角速度
        if left and not right:
            self.angular_speed = -self.max_angular_speed
        elif right and not left:
            self.angular_speed = self.max_angular_speed
        else:
            self.angular_speed = 0

        # 模拟差速驱动
        if self.speed != 0:
            if left:
                self.left_wheel_speed = self.speed * 0.5
                self.right_wheel_speed = self.speed
            elif right:
                self.left_wheel_speed = self.speed
                self.right_wheel_speed = self.speed * 0.5
            else:
                self.left_wheel_speed = self.speed
                self.right_wheel_speed = self.speed
        else:
            self.left_wheel_speed = 0
            self.right_wheel_speed = 0

    def draw(self, surface):
        """绘制小车"""
        # 绘制轨迹
        if len(self.path) > 1:
            pygame.draw.lines(surface, PATH_COLOR, False, self.path, 2)

        # 计算小车四个角的坐标
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)

        half_width = self.width / 2
        half_height = self.height / 2

        # 车身四个顶点
        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]

        # 旋转和平移顶点
        rotated_corners = []
        for x, y in corners:
            # 旋转
            rx = x * cos_angle - y * sin_angle
            ry = x * sin_angle + y * cos_angle
            # 平移
            tx = rx + self.x
            ty = ry + self.y
            rotated_corners.append((tx, ty))

        # 绘制车身
        pygame.draw.polygon(surface, CAR_BODY, rotated_corners)
        pygame.draw.polygon(surface, (40, 80, 130), rotated_corners, 2)

        # 绘制轮子
        wheel_offsets = [
            (-half_width + 5, -half_height - WHEEL_WIDTH / 2),  # 左前轮
            (-half_width + 5, half_height - WHEEL_WIDTH / 2),  # 左后轮
            (half_width - 5, -half_height - WHEEL_WIDTH / 2),  # 右前轮
            (half_width - 5, half_height - WHEEL_WIDTH / 2)  # 右后轮
        ]

        for wx, wy in wheel_offsets:
            # 旋转轮子位置
            rx = wx * cos_angle - wy * sin_angle
            ry = wx * sin_angle + wy * cos_angle
            tx = rx + self.x
            ty = ry + self.y

            # 绘制轮子
            pygame.draw.ellipse(surface, CAR_WHEEL,
                                (tx - WHEEL_RADIUS, ty - WHEEL_WIDTH / 2,
                                 WHEEL_RADIUS * 2, WHEEL_WIDTH))

            # 轮子上的条纹（表示转动）
            stripe_angle = self.angle + pygame.time.get_ticks() / 100
            stripe_x = tx + WHEEL_RADIUS * 0.7 * math.cos(stripe_angle)
            stripe_y = ty + WHEEL_RADIUS * 0.7 * math.sin(stripe_angle)
            pygame.draw.circle(surface, (30, 30, 40), (int(stripe_x), int(stripe_y)), WHEEL_WIDTH // 2)

        # 绘制传感器
        sensor_angles = [
            self.angle - math.radians(SENSOR_ANGLE),
            self.angle,
            self.angle + math.radians(SENSOR_ANGLE)
        ]

        for i, sensor_angle in enumerate(sensor_angles):
            # 传感器探测距离基于传感器数据
            sensor_length = SENSOR_RANGE * (1 - self.sensor_data[i])
            end_x = self.x + sensor_length * math.cos(sensor_angle)
            end_y = self.y + sensor_length * math.sin(sensor_angle)

            # 绘制传感器线
            sensor_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(sensor_surf, SENSOR_COLOR,
                             (self.x, self.y), (end_x, end_y), 2)
            surface.blit(sensor_surf, (0, 0))

            # 绘制传感器终点
            if self.sensor_data[i] > 0:
                pygame.draw.circle(surface, (255, 100, 100),
                                   (int(end_x), int(end_y)), 5)

        # 绘制小车前方的箭头
        front_x = self.x + half_width * cos_angle
        front_y = self.y + half_width * sin_angle
        arrow_length = 20

        arrow_left_x = front_x - arrow_length * math.cos(self.angle - math.pi / 6)
        arrow_left_y = front_y - arrow_length * math.sin(self.angle - math.pi / 6)
        arrow_right_x = front_x - arrow_length * math.cos(self.angle + math.pi / 6)
        arrow_right_y = front_y - arrow_length * math.sin(self.angle + math.pi / 6)

        pygame.draw.polygon(surface, (255, 255, 255), [
            (front_x, front_y),
            (arrow_left_x, arrow_left_y),
            (arrow_right_x, arrow_right_y)
        ])


def draw_grid(surface):
    """绘制网格"""
    grid_size = 50

    # 垂直线
    for x in range(0, WIDTH, grid_size):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, HEIGHT), 1)

    # 水平线
    for y in range(0, HEIGHT, grid_size):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (WIDTH, y), 1)


def draw_info_panel(surface, car, control_state, mode):
    """绘制信息面板"""
    panel_width = 300
    panel_rect = pygame.Rect(WIDTH - panel_width, 0, panel_width, HEIGHT)

    # 绘制面板背景
    pygame.draw.rect(surface, (40, 40, 50), panel_rect)
    pygame.draw.line(surface, (70, 70, 90), (WIDTH - panel_width, 0), (WIDTH - panel_width, HEIGHT), 2)

    font = pygame.font.SysFont(None, 28)
    small_font = pygame.font.SysFont(None, 24)

    # 标题
    title = font.render("小车状态", True, TEXT_COLOR)
    surface.blit(title, (WIDTH - panel_width + 20, 20))

    # 位置信息
    info_y = 60
    infos = [
        f"位置: ({car.x:.1f}, {car.y:.1f})",
        f"角度: {math.degrees(car.angle):.1f}°",
        f"速度: {car.speed:.2f}",
        f"角速度: {math.degrees(car.angular_speed):.2f}°/s",
        f"左轮速度: {car.left_wheel_speed:.2f}",
        f"右轮速度: {car.right_wheel_speed:.2f}"
    ]

    for info in infos:
        text = small_font.render(info, True, TEXT_COLOR)
        surface.blit(text, (WIDTH - panel_width + 20, info_y))
        info_y += 30

    # 传感器数据
    info_y += 20
    sensor_title = font.render("传感器数据", True, TEXT_COLOR)
    surface.blit(sensor_title, (WIDTH - panel_width + 20, info_y))

    info_y += 40
    sensor_labels = ["左传感器", "前传感器", "右传感器"]
    for i, label in enumerate(sensor_labels):
        # 文本
        text = small_font.render(f"{label}: {car.sensor_data[i]:.2f}", True, TEXT_COLOR)
        surface.blit(text, (WIDTH - panel_width + 20, info_y))

        # 进度条
        bar_width = 150
        bar_height = 15
        bar_x = WIDTH - panel_width + 120
        bar_y = info_y + 2

        # 背景条
        pygame.draw.rect(surface, (60, 60, 70), (bar_x, bar_y, bar_width, bar_height), border_radius=3)

        # 填充条
        fill_width = bar_width * car.sensor_data[i]
        color_value = int(255 * car.sensor_data[i])
        bar_color = (255, 255 - color_value, 100)
        pygame.draw.rect(surface, bar_color, (bar_x, bar_y, fill_width, bar_height), border_radius=3)

        info_y += 30

    # 控制状态
    info_y += 20
    control_title = font.render("控制状态", True, TEXT_COLOR)
    surface.blit(control_title, (WIDTH - panel_width + 20, info_y))

    info_y += 40
    control_keys = [
        ("前进", "W / ↑", control_state["forward"]),
        ("后退", "S / ↓", control_state["backward"]),
        ("左转", "A / ←", control_state["left"]),
        ("右转", "D / →", control_state["right"])
    ]

    for label, key, active in control_keys:
        color = (100, 255, 100) if active else (150, 150, 150)
        text = small_font.render(f"{label} ({key}): {'激活' if active else '未激活'}", True, color)
        surface.blit(text, (WIDTH - panel_width + 20, info_y))
        info_y += 30

    # 模式信息
    info_y += 20
    mode_text = font.render(f"模式: {'键盘控制' if mode == 'keyboard' else '自动巡逻'}", True, TEXT_COLOR)
    surface.blit(mode_text, (WIDTH - panel_width + 20, info_y))

    # 操作说明
    info_y = HEIGHT - 150
    instructions_title = font.render("操作说明", True, TEXT_COLOR)
    surface.blit(instructions_title, (WIDTH - panel_width + 20, info_y))

    info_y += 30
    instructions = [
        "W/↑: 前进",
        "S/↓: 后退",
        "A/←: 左转",
        "D/→: 右转",
        "R: 重置小车位置",
        "M: 切换控制模式",
        "C: 清除轨迹",
        "ESC: 退出"
    ]

    for instruction in instructions:
        text = small_font.render(instruction, True, (180, 180, 200))
        surface.blit(text, (WIDTH - panel_width + 20, info_y))
        info_y += 25


def main():
    """主函数"""
    # 创建小车
    car = WheeledCar(WIDTH // 4, HEIGHT // 2)

    # 创建障碍物
    obstacles = [
        Obstacle(300, 200, 30),
        Obstacle(500, 300, 40),
        Obstacle(200, 500, 35),
        Obstacle(600, 150, 25),
        Obstacle(400, 450, 30),
        Obstacle(700, 400, 45)
    ]

    # 控制状态
    control_state = {
        "forward": False,
        "backward": False,
        "left": False,
        "right": False
    }

    # 控制模式：keyboard（键盘控制）或 auto（自动巡逻）
    control_mode = "keyboard"

    # 创建按钮
    reset_button = Button(WIDTH - 280, HEIGHT - 90, 120, 40, "重置位置")
    clear_path_button = Button(WIDTH - 280, HEIGHT - 40, 120, 40, "清除轨迹")
    mode_button = Button(WIDTH - 150, HEIGHT - 90, 120, 40, "切换模式")

    running = True
    auto_timer = 0

    # 主循环
    while running:
        mouse_pos = pygame.mouse.get_pos()

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # 键盘按下事件
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_w, pygame.K_UP):
                    control_state["forward"] = True
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    control_state["backward"] = True
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    control_state["left"] = True
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    control_state["right"] = True
                elif event.key == pygame.K_r:
                    # 重置小车位置
                    car = WheeledCar(WIDTH // 4, HEIGHT // 2)
                elif event.key == pygame.K_c:
                    # 清除轨迹
                    car.path = []
                elif event.key == pygame.K_m:
                    # 切换控制模式
                    control_mode = "auto" if control_mode == "keyboard" else "keyboard"

            # 键盘释放事件
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_w, pygame.K_UP):
                    control_state["forward"] = False
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    control_state["backward"] = False
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    control_state["left"] = False
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    control_state["right"] = False

            # 鼠标事件
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if reset_button.is_clicked(mouse_pos, event):
                    car = WheeledCar(WIDTH // 4, HEIGHT // 2)
                elif clear_path_button.is_clicked(mouse_pos, event):
                    car.path = []
                elif mode_button.is_clicked(mouse_pos, event):
                    control_mode = "auto" if control_mode == "keyboard" else "keyboard"

        # 更新按钮状态
        reset_button.check_hover(mouse_pos)
        clear_path_button.check_hover(mouse_pos)
        mode_button.check_hover(mouse_pos)

        # 自动巡逻模式
        if control_mode == "auto":
            auto_timer += 1

            # 简单的自动巡逻逻辑：前进，遇到障碍物转向
            if car.sensor_data[1] > 0.3:  # 前方有障碍物
                if car.sensor_data[0] < car.sensor_data[2]:  # 左边障碍物更远
                    control_state["left"] = True
                    control_state["right"] = False
                else:  # 右边障碍物更远
                    control_state["left"] = False
                    control_state["right"] = True
                control_state["forward"] = False
                control_state["backward"] = True  # 先后退
            else:
                # 前进
                control_state["forward"] = True
                control_state["backward"] = False

                # 随机转向以避免卡住
                if auto_timer % 120 == 0:  # 每2秒
                    if np.random.random() > 0.5:
                        control_state["left"] = True
                        control_state["right"] = False
                    else:
                        control_state["left"] = False
                        control_state["right"] = True
                elif auto_timer % 60 == 0:  # 每1秒重置转向
                    control_state["left"] = False
                    control_state["right"] = False

        # 更新小车控制
        car.set_control(
            control_state["forward"],
            control_state["backward"],
            control_state["left"],
            control_state["right"]
        )

        # 更新小车状态
        car.update(obstacles)

        # 绘制背景
        screen.fill(BACKGROUND)

        # 绘制网格
        draw_grid(screen)

        # 绘制障碍物
        for obstacle in obstacles:
            obstacle.draw(screen)

        # 绘制小车
        car.draw(screen)

        # 绘制信息面板
        draw_info_panel(screen, car, control_state, control_mode)

        # 绘制按钮
        reset_button.draw(screen)
        clear_path_button.draw(screen)
        mode_button.draw(screen)

        # 更新显示
        pygame.display.flip()

        # 控制帧率
        clock.tick(60)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()