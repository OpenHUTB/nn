import pygame
import sys
import math  # 导入Python内置math库，用于角度弧度转换

# 初始化pygame
pygame.init()

# 窗口配置
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 500
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("机械臂关节升降模拟")

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 128, 255)
RED = (255, 50, 50)

# 机械臂参数
BASE_X = WINDOW_WIDTH // 2  # 机械臂底座x坐标
BASE_Y = WINDOW_HEIGHT - 50  # 机械臂底座y坐标
JOINT_LENGTH = 200  # 关节臂长度（影响升降范围）
joint_angle = 90    # 关节初始角度（90度为垂直向下，0度为水平向左，180度为水平向右）
speed = 1           # 角度变化速度（控制升降快慢）

def draw_arm(screen, base_x, base_y, angle, length):
    """绘制机械臂模型"""
    # 计算关节末端坐标（通过角度转换，使用math库的radians方法）
    # 三角函数计算：y轴向下为正，需调整角度映射
    rad = math.radians(angle)  # 修复：替换为math.radians
    joint_end_x = base_x + length * math.cos(rad)  # 修复：使用math.cos
    joint_end_y = base_y - length * math.sin(rad)  # 修复：使用math.sin，减号实现y轴向上为高度增加

    # 绘制底座
    pygame.draw.circle(screen, BLUE, (base_x, base_y), 20)
    # 绘制关节臂（底座到关节末端）
    pygame.draw.line(screen, RED, (base_x, base_y), (joint_end_x, joint_end_y), 8)
    # 绘制关节末端
    pygame.draw.circle(screen, BLACK, (int(joint_end_x), int(joint_end_y)), 10)
    # 绘制高度提示文字
    height = int(BASE_Y - joint_end_y)  # 计算当前高度（像素值，近似对应实际高度）
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"关节高度：{height}px", True, BLACK)
    screen.blit(text, (10, 10))

# 主循环
clock = pygame.time.Clock()
is_running = True
move_direction = "up"  # 初始运动方向：上升

while is_running:
    # 控制帧率
    clock.tick(60)
    # 填充背景色
    screen.fill(WHITE)

    # 事件处理（关闭窗口、按键控制）
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
        # 按键控制：上箭头上升，下箭头下降
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                move_direction = "up"
            if event.key == pygame.K_DOWN:
                move_direction = "down"

    # 更新关节角度（控制升降）
    if move_direction == "up":
        # 角度增大：关节上升（最大角度170度，避免超出窗口）
        if joint_angle < 170:
            joint_angle += speed
        else:
            move_direction = "down"  # 到达上限后自动下降
    else:
        # 角度减小：关节下降（最小角度10度，避免超出窗口）
        if joint_angle > 10:
            joint_angle -= speed
        else:
            move_direction = "up"  # 到达下限后自动上升

    # 绘制机械臂
    draw_arm(screen, BASE_X, BASE_Y, joint_angle, JOINT_LENGTH)

    # 更新窗口显示
    pygame.display.flip()

# 退出程序
pygame.quit()
sys.exit()