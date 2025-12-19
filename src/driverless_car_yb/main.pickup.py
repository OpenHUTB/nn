import pygame
import sys
import time

# 初始化pygame
pygame.init()

# 配置参数
WIDTH, HEIGHT = 800, 600  # 窗口大小
FPS = 30  # 帧率
CAR_SPEED = 3  # 无人车速度
PASSENGER_SIZE = 15  # 乘客大小
CAR_SIZE = (40, 20)  # 无人车尺寸

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)  # 无人车颜色
RED = (255, 0, 0)  # 乘客颜色
GREEN = (0, 255, 0)  # 目的地颜色
GRAY = (128, 128, 128)  # 道路颜色
YELLOW = (255, 255, 0)  # 文字颜色

# 创建窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("无人车网络控制接送模拟")
clock = pygame.time.Clock()

# 字体设置
font = pygame.font.SysFont("SimHei", 20)  # 支持中文显示


class NetworkController:
    """模拟网络控制器：发送指令给无人车"""

    def __init__(self):
        self.pickup_point = (100, 100)  # 接客点
        self.dropoff_point = (600, 400)  # 目的地
        self.status = "idle"  # 状态：idle/waiting/picking/dropping/complete

    def send_pickup_command(self):
        """发送接客指令"""
        self.status = "picking"
        print("网络指令：前往接客点")
        return self.pickup_point

    def send_dropoff_command(self):
        """发送送客指令"""
        self.status = "dropping"
        print("网络指令：前往目的地")
        return self.dropoff_point


class AutonomousCar:
    """无人车类"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width, self.height = CAR_SIZE
        self.has_passenger = False  # 是否载有乘客
        self.target_x = x
        self.target_y = y

    def move_towards(self, target_x, target_y):
        """向目标点移动"""
        # 计算x方向移动
        if self.x < target_x - self.width // 2:
            self.x += CAR_SPEED
        elif self.x > target_x - self.width // 2:
            self.x -= CAR_SPEED

        # 计算y方向移动
        if self.y < target_y - self.height // 2:
            self.y += CAR_SPEED
        elif self.y > target_y - self.height // 2:
            self.y -= CAR_SPEED

    def is_at_target(self, target_x, target_y):
        """判断是否到达目标点"""
        return (abs(self.x + self.width // 2 - target_x) < 10 and
                abs(self.y + self.height // 2 - target_y) < 10)

    def draw(self, surface):
        """绘制无人车"""
        # 绘制车身
        pygame.draw.rect(surface, BLUE, (self.x, self.y, self.width, self.height))
        # 绘制车轮
        pygame.draw.circle(surface, BLACK, (self.x + 5, self.y + self.height), 3)
        pygame.draw.circle(surface, BLACK, (self.x + self.width - 5, self.y + self.height), 3)
        pygame.draw.circle(surface, BLACK, (self.x + 5, self.y), 3)
        pygame.draw.circle(surface, BLACK, (self.x + self.width - 5, self.y), 3)
        # 如果载有乘客，绘制乘客（车顶上）
        if self.has_passenger:
            pygame.draw.circle(surface, RED,
                               (self.x + self.width // 2, self.y - 10),
                               PASSENGER_SIZE // 2)


def draw_scene(car, controller):
    """绘制整个场景"""
    # 背景
    screen.fill(WHITE)

    # 绘制道路（灰色矩形）
    pygame.draw.rect(screen, GRAY, (50, 50, 700, 500), 0)
    pygame.draw.rect(screen, WHITE, (70, 70, 660, 460), 0)

    # 绘制接客点（红色圆圈）
    pygame.draw.circle(screen, RED, controller.pickup_point, PASSENGER_SIZE)
    screen.blit(font.render("接客点", True, RED),
                (controller.pickup_point[0] + 20, controller.pickup_point[1]))

    # 绘制目的地（绿色圆圈）
    pygame.draw.circle(screen, GREEN, controller.dropoff_point, PASSENGER_SIZE)
    screen.blit(font.render("目的地", True, GREEN),
                (controller.dropoff_point[0] + 20, controller.dropoff_point[1]))

    # 绘制无人车
    car.draw(screen)

    # 绘制状态信息
    status_text = f"状态：{controller.status}"
    if controller.status == "idle":
        status_text += "（点击鼠标发送接客指令）"
    elif controller.status == "picking":
        status_text += "（前往接客点）"
    elif controller.status == "dropping":
        status_text += "（前往目的地）"
    elif controller.status == "complete":
        status_text += "（任务完成）"

    screen.blit(font.render(status_text, True, YELLOW), (20, 20))
    screen.blit(font.render("无人车网络控制接送模拟", True, BLACK), (300, 20))

    pygame.display.flip()


def main():
    """主函数"""
    # 初始化对象
    car = AutonomousCar(WIDTH // 2 - CAR_SIZE[0] // 2, HEIGHT // 2 - CAR_SIZE[1] // 2)  # 初始位置在中心
    controller = NetworkController()

    running = True
    while running:
        clock.tick(FPS)

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 鼠标点击发送接客指令
            if event.type == pygame.MOUSEBUTTONDOWN and controller.status == "idle":
                controller.send_pickup_command()
                car.target_x, car.target_y = controller.pickup_point

        # 无人车行为逻辑
        if controller.status == "picking":
            # 前往接客点
            car.move_towards(car.target_x, car.target_y)
            if car.is_at_target(controller.pickup_point[0], controller.pickup_point[1]):
                car.has_passenger = True  # 接上乘客
                controller.status = "waiting"
                time.sleep(1)  # 停留1秒
                # 发送送客指令
                car.target_x, car.target_y = controller.send_dropoff_command()

        elif controller.status == "dropping":
            # 前往目的地
            car.move_towards(car.target_x, car.target_y)
            if car.is_at_target(controller.dropoff_point[0], controller.dropoff_point[1]):
                car.has_passenger = False  # 放下乘客
                controller.status = "complete"

        # 绘制场景
        draw_scene(car, controller)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()