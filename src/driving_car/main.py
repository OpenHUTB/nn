import pygame
import math
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='无人车基础控制')
    parser.add_argument('--width', type=int, default=800, help='窗口宽度（像素）')
    parser.add_argument('--height', type=int, default=600, help='窗口高度（像素）')
    parser.add_argument('--speed', type=float, default=4.0, help='移动速度（像素/帧）')
    parser.add_argument('--turn-speed', type=float, default=2.0, help='转向速度（度/帧）')
    parser.add_argument('--fps', type=int, default=60, help='目标帧率')
    return parser.parse_args()

# 1. 初始化pygame（必须放在最前面）
try:
    pygame.init()
except pygame.error as e:
    print(f"pygame初始化失败: {e}")
    print("请确认系统支持图形显示，无法在无头环境（SSH/Docker）中运行")
    sys.exit(1)

# 2. 屏幕配置
args = parse_args()
SCREEN_W = args.width
SCREEN_H = args.height
try:
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("无人车基础控制")
except pygame.error as e:
    print(f"窗口创建失败: {e}")
    print("请确认系统支持图形显示")
    pygame.quit()
    sys.exit(1)

# 3. 车辆类（封装移动和绘制逻辑）
class Car:
    def __init__(self):
        # 初始位置（屏幕中心）
        self.x = SCREEN_W // 2
        self.y = SCREEN_H // 2
        # 初始方向（向上，角度0为右，90为上）
        self.angle = 90
        # 运动参数
        self.speed = args.speed
        self.turn_speed = args.turn_speed

    def move(self, direction):
        """根据方向移动：forward/backward"""
        rad = math.radians(self.angle)
        if direction == "forward":
            self.x -= self.speed * math.sin(rad)
            self.y -= self.speed * math.cos(rad)
        elif direction == "backward":
            self.x += self.speed * math.sin(rad)
            self.y += self.speed * math.cos(rad)
        # 防止移出屏幕
        self.x = max(20, min(self.x, SCREEN_W - 20))
        self.y = max(20, min(self.y, SCREEN_H - 20))

    def turn(self, direction):
        """根据方向转向：left/right"""
        if direction == "left":
            self.angle = (self.angle + self.turn_speed) % 360
        elif direction == "right":
            self.angle = (self.angle - self.turn_speed) % 360

    def draw(self):
        """绘制车辆（三角形，直观显示方向）"""
        rad = math.radians(self.angle)
        # 三角形三个顶点（车头+车尾两侧）
        front = (self.x - 20 * math.sin(rad), self.y - 20 * math.cos(rad))
        back_l = (self.x + 10 * math.sin(rad + math.pi/2), self.y + 10 * math.cos(rad + math.pi/2))
        back_r = (self.x + 10 * math.sin(rad - math.pi/2), self.y + 10 * math.cos(rad - math.pi/2))
        # 绘制蓝色车身+黑色边框
        pygame.draw.polygon(screen, (0, 0, 255), [front, back_l, back_r])
        pygame.draw.polygon(screen, (0, 0, 0), [front, back_l, back_r], 2)

# 4. 主控制循环（程序核心）
def main():
    car = Car()
    clock = pygame.time.Clock()
    running = True  # 控制循环是否继续

    # 控制说明文字（使用pygame默认字体，兼容所有平台）
    try:
        font = pygame.font.SysFont("Arial", 20)
    except (TypeError, pygame.error):
        font = pygame.font.Font(None, 22)
    tip_text = font.render("↑前进 | ↓后退 | ←左转 | →右转 | ESC退出", True, (0, 0, 0))

    try:
        while running:
            # 1. 处理事件（关闭窗口、按键）
            for event in pygame.event.get():
                # 点击窗口关闭按钮
                if event.type == pygame.QUIT:
                    running = False
                # 按下ESC键退出
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            # 2. 持续检测按键（按住键持续动作）
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                car.move("forward")
            if keys[pygame.K_DOWN]:
                car.move("backward")
            if keys[pygame.K_LEFT]:
                car.turn("left")
            if keys[pygame.K_RIGHT]:
                car.turn("right")

            # 3. 绘制画面（清空→画车辆→画文字）
            screen.fill((255, 255, 255))  # 白色背景
            car.draw()  # 画车辆
            screen.blit(tip_text, (10, 10))  # 画控制说明

            # 4. 更新屏幕+控制帧率
            pygame.display.update()
            clock.tick(args.fps)
    finally:
        # 5. 退出程序（释放资源）
        pygame.quit()

# 5. 启动程序（关键：必须调用main()）
if __name__ == "__main__":
    main()