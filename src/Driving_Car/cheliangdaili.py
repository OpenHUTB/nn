import math
import random
import time
import pygame
from typing import List, Tuple, Optional

# 初始化pygame
pygame.init()

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)


class Vector2:
    """二维向量类，用于处理坐标和移动计算"""

    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    def __add__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector2':
        return Vector2(self.x * scalar, self.y * scalar)

    def magnitude(self) -> float:
        """计算向量的模长"""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self) -> 'Vector2':
        """将向量归一化（单位向量）"""
        mag = self.magnitude()
        if mag == 0:
            return Vector2(0, 0)
        return Vector2(self.x / mag, self.y / mag)

    def distance_to(self, other: 'Vector2') -> float:
        """计算到另一个点的距离"""
        return (self - other).magnitude()

    def __repr__(self) -> str:
        return f"Vector2({self.x:.2f}, {self.y:.2f})"


class Obstacle:
    """障碍物类，表示地图上的障碍物"""

    def __init__(self, position: Vector2, radius: float):
        self.position = position
        self.radius = radius

    def draw(self, screen):
        """在屏幕上绘制障碍物"""
        pygame.draw.circle(screen, GRAY,
                           (int(self.position.x), int(self.position.y)),
                           int(self.radius))


class Map:
    """地图类，包含边界和障碍物信息"""

    def __init__(self, width: float = 800, height: float = 600):
        self.width = width
        self.height = height
        self.obstacles: List[Obstacle] = []

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """添加障碍物到地图"""
        self.obstacles.append(obstacle)

    def is_position_valid(self, position: Vector2) -> bool:
        """检查位置是否有效（在地图范围内且不与障碍物碰撞）"""
        # 检查是否在地图范围内
        if position.x < 0 or position.x > self.width or position.y < 0 or position.y > self.height:
            return False

        # 检查是否与障碍物碰撞
        for obstacle in self.obstacles:
            if position.distance_to(obstacle.position) < obstacle.radius + 5:  # 5是安全距离
                return False

        return True

    def draw(self, screen):
        """绘制地图和障碍物"""
        screen.fill(WHITE)
        for obstacle in self.obstacles:
            obstacle.draw(screen)


class Vehicle:
    """无人车类，表示可以在地图上移动的车辆"""

    def __init__(self, vehicle_id: str, map: Map,
                 start_position: Optional[Vector2] = None,
                 speed: float = 3.0,
                 radius: float = 10.0):
        self.id = vehicle_id
        self.map = map
        self.speed = speed  # 移动速度
        self.radius = radius  # 车辆半径

        # 如果没有指定起始位置，则随机生成一个有效位置
        if start_position and map.is_position_valid(start_position):
            self.position = start_position
        else:
            self.position = self._get_random_valid_position()

        self.destination: Optional[Vector2] = None
        self.roaming_direction = self._get_random_direction()
        self.roaming_change_interval = 3  # 随机漫游时改变方向的时间间隔（秒）
        self.last_direction_change = time.time()

    def _get_random_valid_position(self) -> Vector2:
        """生成一个随机的有效位置"""
        while True:
            pos = Vector2(
                random.uniform(0, self.map.width),
                random.uniform(0, self.map.height)
            )
            if self.map.is_position_valid(pos):
                return pos

    def _get_random_direction(self) -> Vector2:
        """生成一个随机的移动方向"""
        angle = random.uniform(0, 2 * math.pi)
        return Vector2(math.cos(angle), math.sin(angle))

    def set_destination(self, destination: Vector2) -> bool:
        """设置目的地，如果目的地有效则返回True"""
        if self.map.is_position_valid(destination):
            self.destination = destination
            return True
        return False

    def clear_destination(self) -> None:
        """清除目的地，开始随机漫游"""
        self.destination = None
        self.roaming_direction = self._get_random_direction()
        self.last_direction_change = time.time()

    def _avoid_obstacles(self, desired_direction: Vector2) -> Vector2:
        """避开障碍物的路径调整"""
        avoidance_strength = 0.8
        avoidance_direction = Vector2(0, 0)

        # 检查每个障碍物
        for obstacle in self.map.obstacles:
            distance = self.position.distance_to(obstacle.position)
            # 如果距离过近，计算躲避方向
            if distance < obstacle.radius + self.radius + 20:  # 20是安全距离
                # 远离障碍物的方向
                away_from_obstacle = (self.position - obstacle.position).normalize()
                # 距离越近，躲避力度越大
                avoidance_direction += away_from_obstacle * (1 / distance)

        # 结合期望方向和躲避方向
        new_direction = desired_direction + avoidance_direction * avoidance_strength
        return new_direction.normalize()

    def update(self, delta_time: float) -> None:
        """更新车辆位置"""
        # 确定移动方向
        if self.destination:
            # 有目的地，向目的地移动
            desired_direction = (self.destination - self.position).normalize()

            # 检查是否到达目的地
            if self.position.distance_to(self.destination) < self.speed * delta_time:
                self.position = self.destination
                self.clear_destination()  # 到达后开始漫游
                return
        else:
            # 无目的地，随机漫游
            current_time = time.time()
            # 定期改变方向
            if current_time - self.last_direction_change > self.roaming_change_interval:
                self.roaming_direction = self._get_random_direction()
                self.last_direction_change = current_time
            desired_direction = self.roaming_direction

        # 避开障碍物
        move_direction = self._avoid_obstacles(desired_direction)

        # 计算新位置
        new_position = self.position + move_direction * self.speed * delta_time

        # 检查新位置是否有效，如果无效则调整方向
        if not self.map.is_position_valid(new_position):
            # 碰到边界，反弹
            if new_position.x < 0 or new_position.x > self.map.width:
                move_direction = Vector2(-move_direction.x, move_direction.y)
            if new_position.y < 0 or new_position.y > self.map.height:
                move_direction = Vector2(move_direction.x, -move_direction.y)
            new_position = self.position + move_direction * self.speed * delta_time

        self.position = new_position

    def draw(self, screen):
        """在屏幕上绘制车辆"""
        # 绘制车辆主体
        pygame.draw.circle(screen, BLUE,
                           (int(self.position.x), int(self.position.y)),
                           int(self.radius))

        # 绘制方向指示器
        direction = self._get_current_direction()
        nose_pos = self.position + direction * self.radius
        pygame.draw.line(screen, RED,
                         (self.position.x, self.position.y),
                         (nose_pos.x, nose_pos.y), 3)

        # 如果有目的地，绘制目的地和路径
        if self.destination:
            pygame.draw.circle(screen, GREEN,
                               (int(self.destination.x), int(self.destination.y)),
                               5)
            pygame.draw.line(screen, YELLOW,
                             (self.position.x, self.position.y),
                             (self.destination.x, self.destination.y), 1)

    def _get_current_direction(self) -> Vector2:
        """获取当前移动方向"""
        if self.destination:
            return (self.destination - self.position).normalize()
        return self.roaming_direction

    def __repr__(self) -> str:
        dest = self.destination if self.destination else "None"
        return f"Vehicle {self.id}: Position {self.position}, Destination {dest}"


def main():
    # 创建地图
    map_width, map_height = 800, 600
    game_map = Map(map_width, map_height)

    # 添加一些障碍物
    obstacles = [
        Obstacle(Vector2(200, 150), 30),
        Obstacle(Vector2(600, 400), 40),
        Obstacle(Vector2(400, 500), 25),
        Obstacle(Vector2(100, 400), 35),
        Obstacle(Vector2(700, 200), 20),
        Obstacle(Vector2(300, 300), 50)
    ]
    for obs in obstacles:
        game_map.add_obstacle(obs)

    # 创建屏幕
    screen = pygame.display.set_mode((map_width, map_height))
    pygame.display.set_caption("无人车导航模拟")

    # 创建车辆
    vehicle = Vehicle("car_001", game_map, speed=5.0)
    print(f"创建车辆: {vehicle}")

    # 时钟控制
    clock = pygame.time.Clock()
    running = True
    last_update_time = time.time()

    # 字体设置（用于显示信息）
    font = pygame.font.SysFont(None, 24)

    while running:
        # 计算时间差
        current_time = time.time()
        delta_time = current_time - last_update_time
        last_update_time = current_time

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 鼠标点击设置新目的地
                if event.button == 1:  # 左键点击
                    x, y = event.pos
                    destination = Vector2(x, y)
                    if vehicle.set_destination(destination):
                        print(f"设置新目的地: {destination}")

        # 随机设置目的地（偶尔）
        if random.random() < 0.005 and not vehicle.destination:  # 0.5%的概率设置新目的地
            dest_x = random.uniform(0, map_width)
            dest_y = random.uniform(0, map_height)
            destination = Vector2(dest_x, dest_y)
            if vehicle.set_destination(destination):
                print(f"自动设置新目的地: {destination}")

        # 更新车辆位置
        vehicle.update(delta_time)

        # 绘制
        game_map.draw(screen)
        vehicle.draw(screen)

        # 显示车辆信息
        info_text = f"位置: ({vehicle.position.x:.1f}, {vehicle.position.y:.1f})"
        if vehicle.destination:
            info_text += f"  目的地: ({vehicle.destination.x:.1f}, {vehicle.destination.y:.1f})"
        else:
            info_text += "  状态: 漫游中"

        text_surface = font.render(info_text, True, BLACK)
        screen.blit(text_surface, (10, 10))

        # 显示操作提示
        help_text = "左键点击: 设置新目的地 | ESC: 退出"
        help_surface = font.render(help_text, True, BLACK)
        screen.blit(help_surface, (10, map_height - 30))

        # 更新显示
        pygame.display.flip()

        # 控制帧率
        clock.tick(60)

    pygame.quit()
    print("模拟结束")


if __name__ == "__main__":
    main()