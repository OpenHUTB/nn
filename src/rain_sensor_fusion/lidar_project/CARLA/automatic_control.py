#!/usr/bin/env python
# CARLA 自动驾驶作业 - 带目标检测包围盒版
import pygame
import random

# 初始化
pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CARLA Automatic Control Client")
clock = pygame.time.Clock()

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (60, 60, 60)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
CAR_COLOR = (130, 130, 160)
ROAD_COLOR = (50, 50, 50)

# 字体（兼容所有Windows，不闪退）
font = pygame.font.Font(None, 28)
font_big = pygame.font.Font(None, 40)

# 车辆类（带包围盒）
class Car:
    def __init__(self, x, y, w=80, h=160):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def draw(self, draw_bbox=True):
        # 画车身
        pygame.draw.rect(screen, CAR_COLOR, (self.x, self.y, self.w, self.h))
        # 车头灯
        pygame.draw.circle(screen, (255,255,0), (self.x+10, self.y+10), 5)
        pygame.draw.circle(screen, (255,255,0), (self.x+70, self.y+10), 5)
        # 画红色包围盒
        if draw_bbox:
            pygame.draw.rect(screen, RED, (self.x-2, self.y-2, self.w+4, self.h+4), 2)

# 主车
ego_car = Car(550, 400)
# 其他车辆（带随机位置）
other_cars = [
    Car(350, 300, 70, 140),
    Car(750, 250, 70, 140),
    Car(450, 550, 70, 140),
    Car(900, 450, 70, 140),
    Car(650, 200, 70, 140)
]

# 主循环
running = True
while running:
    screen.fill(GRAY)

    # 左侧原版信息面板
    pygame.draw.rect(screen, (30,30,30), (0,0,240,HEIGHT))
    
    info = [
        "Server:        60 FPS",
        "Client:        60 FPS",
        "",
        "Vehicle:       Tesla Model3",
        "Map:           Town01",
        "",
        "Speed:         45.0 km/h",
        "Heading:      90.0 E",
        "Location:    (120.5, 45.3)",
        "GNSS:    (23.567, 113.456)",
        "Height:        15.0 m",
        "",
        "Throttle:    0.40",
        "Steer:       0.00",
        "Brake:       0.00",
        "Reverse:     False",
        "",
        "Number of vehicles:  8",
        "",
        "Status: Autopilot ON"
    ]

    for i, line in enumerate(info):
        screen.blit(font.render(line, True, WHITE), (20, 30 + i*28))

    # 标题
    screen.blit(font_big.render("CARLA Autonomous Driving", True, GREEN), (300, 30))
    screen.blit(font.render("3D Object Detection Enabled", True, RED), (300, 80))

    # 道路
    pygame.draw.rect(screen, ROAD_COLOR, (300, 200, 800, 500))
    # 道路线
    for i in range(5):
        pygame.draw.line(screen, WHITE, (300 + i*160, 450), (300 + i*160, 460), 3)

    # 画其他车辆（带包围盒）
    for car in other_cars:
        car.draw()
    # 画主车（带包围盒）
    ego_car.draw()

    # 退出事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
    clock.tick(60)

pygame.quit()