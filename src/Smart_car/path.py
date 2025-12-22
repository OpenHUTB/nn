import pygame
import sys
import math

# 初始化pygame
pygame.init()

# 窗口配置
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("无人车键盘控制.")

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

# 无人车参数
CAR_WIDTH = 40
CAR_HEIGHT = 60
car_x = WINDOW_WIDTH // 2  # 初始x坐标（窗口中心）
car_y = WINDOW_HEIGHT // 2  # 初始y坐标（窗口中心）
car_angle = 0  # 初始角度（0度为向上）
car_speed = 5  # 移动速度
car_color = RED  # 车辆颜色

# 字体初始化（显示控制提示和车辆状态）
font = pygame.font.SysFont("SimHei", 20)  # 支持中文显示
font_large = pygame.font.SysFont("SimHei", 24)


def draw_car(x, y, angle):
    """绘制带方向的无人车（三角形+矩形，直观显示朝向）"""
    # 保存当前坐标系
    pygame.draw.rect(screen, GRAY, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))
    # 绘制道路网格（增强场景感）
    for i in range(0, WINDOW_WIDTH, 50):
        pygame.draw.line(screen, WHITE, (i, 0), (i, WINDOW_HEIGHT), 1)
    for j in range(0, WINDOW_HEIGHT, 50):
        pygame.draw.line(screen, WHITE, (0, j), (WINDOW_WIDTH, j), 1)

    # 平移+旋转坐标系，实现车辆朝向控制
    rotated_car = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
    # 绘制车辆主体（矩形）
    pygame.draw.rect(rotated_car, car_color, (0, 0, CAR_WIDTH, CAR_HEIGHT))
    # 绘制车辆头部（三角形，标识前进方向）
    pygame.draw.polygon(rotated_car, BLACK, [
        (CAR_WIDTH // 2, 0),
        (0, CAR_HEIGHT // 2),
        (CAR_WIDTH, CAR_HEIGHT // 2)
    ])
    # 旋转车辆
    rotated_car = pygame.transform.rotate(rotated_car, -angle)
    # 获取旋转后的矩形区域（用于居中绘制）
    car_rect = rotated_car.get_rect(center=(x, y))
    # 绘制车辆到窗口
    screen.blit(rotated_car, car_rect)


def display_info(direction):
    """显示控制提示和车辆当前状态"""
    # 控制提示文本
    tip_text1 = "键盘控制说明："
    tip_text2 = "W-前进  S-后退  A-左转  D-右转"
    tip_text3 = "空格-停止  Q-退出程序"
    # 车辆状态文本
    status_text = f"当前状态：{direction} | 位置：({int(car_x)}, {int(car_y)}) | 朝向角度：{int(car_angle)}°"

    # 绘制文本（抗锯齿）
    text1 = font.render(tip_text1, True, BLACK)
    text2 = font.render(tip_text2, True, BLACK)
    text3 = font.render(tip_text3, True, BLACK)
    status_text_surf = font_large.render(status_text, True, BLUE)

    # 显示文本位置
    screen.blit(text1, (10, 10))
    screen.blit(text2, (10, 40))
    screen.blit(text3, (10, 70))
    screen.blit(status_text_surf, (10, 100))


def update_car_position(key_pressed):
    """根据键盘输入更新车辆位置和角度"""
    global car_x, car_y, car_angle
    direction = "停止"

    # 角度控制（左转/右转，每次调整5度）
    if key_pressed[pygame.K_a]:  # A键左转
        car_angle += 5
        direction = "左转"
    if key_pressed[pygame.K_d]:  # D键右转
        car_angle -= 5
        direction = "右转"

    # 位置控制（前进/后退，基于当前角度计算位移）
    radian = math.radians(car_angle)  # 角度转弧度
    if key_pressed[pygame.K_w]:  # W键前进
        car_x += car_speed * math.sin(radian)
        car_y -= car_speed * math.cos(radian)
        direction = "前进"
    if key_pressed[pygame.K_s]:  # S键后退
        car_x -= car_speed * math.sin(radian)
        car_y += car_speed * math.cos(radian)
        direction = "后退"

    # 边界检测（防止车辆驶出窗口）
    car_x = max(CAR_WIDTH // 2, min(WINDOW_WIDTH - CAR_WIDTH // 2, car_x))
    car_y = max(CAR_HEIGHT // 2, min(WINDOW_HEIGHT - CAR_HEIGHT // 2, car_y))

    return direction


def main():
    """主函数：运行仿真系统"""
    clock = pygame.time.Clock()  # 控制帧率
    running = True

    while running:
        # 帧率控制（60帧/秒）
        clock.tick(60)

        # 事件监听
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Q键退出
                    running = False
                if event.key == pygame.K_SPACE:  # 空格停止（重置角度无变化，位置不变）
                    pass

        # 获取键盘按键状态
        key_pressed = pygame.key.get_pressed()

        # 更新车辆位置和方向
        current_direction = update_car_position(key_pressed)

        # 绘制场景和车辆
        draw_car(car_x, car_y, car_angle)

        # 显示信息
        display_info(current_direction)

        # 更新窗口显示
        pygame.display.flip()

    # 退出程序
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()