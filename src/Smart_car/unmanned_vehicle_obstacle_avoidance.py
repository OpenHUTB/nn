import pygame
import random
import sys

# 初始化
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("无人车避障模拟")
clock = pygame.time.Clock()

# 颜色
WHITE = (255, 255, 255)
GRAY = (120, 120, 120)
BLUE = (0, 150, 255)
RED = (255, 60, 60)

# 小车
car_w, car_h = 40, 60
car_x = 400 - car_w // 2
car_y = 500
speed = 5

# 障碍物
obs_w, obs_h = 60, 60
obs_x = random.randint(100, 700 - obs_w)
obs_y = 0
obs_speed = 4

# 主循环
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # 键盘控制
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and car_x > 100:
        car_x -= speed
    if keys[pygame.K_RIGHT] and car_x < 700 - car_w:
        car_x += speed

    # 障碍物下落
    obs_y += obs_speed
    if obs_y > 600:
        obs_y = 0
        obs_x = random.randint(100, 700 - obs_w)

    # 避障逻辑：靠近就自动躲开
    car_rect = pygame.Rect(car_x, car_y, car_w, car_h)
    obs_rect = pygame.Rect(obs_x, obs_y, obs_w, obs_h)

    if car_rect.colliderect(obs_rect):
        if car_x < 400:
            car_x += speed * 2
        else:
            car_x -= speed * 2

    # 绘制
    screen.fill(WHITE)
    pygame.draw.rect(screen, GRAY, (100, 0, 600, 600))  # 道路
    pygame.draw.rect(screen, BLUE, car_rect)             # 车
    pygame.draw.rect(screen, RED, obs_rect)             # 障碍

    pygame.display.flip()
    clock.tick(60)
