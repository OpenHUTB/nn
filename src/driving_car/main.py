import pygame
import math

# 1. 初始化pygame（必须放在最前面）
pygame.init()

# 2. 屏幕配置
SCREEN_W = 800
SCREEN_H = 600
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("无人车基础控制")

# 3. 车辆类（封装移动和绘制逻辑）
class Car:
    def __init__(self):
        # 初始位置（屏幕中心）
        self.x = SCREEN_W // 2
        self.y = SCREEN_H // 2
        # 初始方向（向上，角度0为右，90为上）
        self.angle = 90
        # 运动参数
        self.speed = 4
        self.turn_speed = 2

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

    # 控制说明文字
    font = pygame.font.SysFont("Arial", 20)
    tip_text = font.render("↑前进 | ↓后退 | ←左转 | →右转 | ESC退出", True, (0, 0, 0))

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

        # 4. 更新屏幕+控制帧率（60帧/秒，避免画面卡顿）
        pygame.display.update()
        clock.tick(60)

    # 5. 退出程序（释放资源）
    pygame.quit()

# 5. 启动程序（关键：必须调用main()）
if __name__ == "__main__":
    main()