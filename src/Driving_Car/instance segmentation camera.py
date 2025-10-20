import cv2
import numpy as np
import pygame
import random
import math
from typing import List, Tuple, Dict, Optional

# 初始化pygame
pygame.init()

# 颜色定义（补充GRAY的定义）
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)
CYAN = (0, 255, 255)
GRAY = (128, 128, 128)  # 补充灰色定义

# 可用颜色列表（用于实例分割标记）
INSTANCE_COLORS = [RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE, PINK, CYAN]


class GameObject:
    """游戏对象基类"""

    def __init__(self, x: float, y: float, width: float, height: float, obj_type: str):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.obj_type = obj_type
        self.velocity_x = 0
        self.velocity_y = 0
        self.instance_id = None  # 实例ID，用于区分同一类别的不同对象

    def update(self, delta_time: float, world_width: float, world_height: float):
        """更新对象位置"""
        self.x += self.velocity_x * delta_time
        self.y += self.velocity_y * delta_time

        # 边界检测
        if self.x < 0:
            self.x = 0
            self.velocity_x = -self.velocity_x
        elif self.x + self.width > world_width:
            self.x = world_width - self.width
            self.velocity_x = -self.velocity_x

        if self.y < 0:
            self.y = 0
            self.velocity_y = -self.velocity_y
        elif self.y + self.height > world_height:
            self.y = world_height - self.height
            self.velocity_y = -self.velocity_y

    def draw(self, surface: pygame.Surface):
        """绘制对象"""
        pass


class Car(GameObject):
    """车辆类"""

    def __init__(self, x: float, y: float, width: float = 40, height: float = 20):
        super().__init__(x, y, width, height, "car")
        # 随机速度
        speed = random.uniform(50, 150)
        angle = random.uniform(0, 2 * math.pi)
        self.velocity_x = math.cos(angle) * speed
        self.velocity_y = math.sin(angle) * speed

    def draw(self, surface: pygame.Surface):
        # 绘制车辆主体
        pygame.draw.rect(surface, BLUE, (self.x, self.y, self.width, self.height))
        # 绘制车窗
        pygame.draw.rect(surface, CYAN, (self.x + 5, self.y + 5, self.width - 10, self.height - 10))
        # 绘制车轮
        wheel_size = 5
        pygame.draw.rect(surface, BLACK, (self.x + 5, self.y, self.width - 10, wheel_size))
        pygame.draw.rect(surface, BLACK, (self.x + 5, self.y + self.height - wheel_size, self.width - 10, wheel_size))


class Pedestrian(GameObject):
    """行人类"""

    def __init__(self, x: float, y: float, width: float = 15, height: float = 30):
        super().__init__(x, y, width, height, "pedestrian")
        # 随机速度
        speed = random.uniform(20, 60)
        angle = random.uniform(0, 2 * math.pi)
        self.velocity_x = math.cos(angle) * speed
        self.velocity_y = math.sin(angle) * speed

    def draw(self, surface: pygame.Surface):
        # 绘制头部
        pygame.draw.circle(surface, PINK, (int(self.x + self.width / 2), int(self.y + self.height / 5)),
                           int(self.width / 2))
        # 绘制身体
        pygame.draw.rect(surface, GREEN, (self.x, self.y + self.height / 5, self.width, self.height * 4 / 5))


class TrafficLight(GameObject):
    """交通灯类（静态）"""

    def __init__(self, x: float, y: float, width: float = 20, height: float = 60):
        super().__init__(x, y, width, height, "traffic_light")
        self.current_state = 0  # 0:红, 1:黄, 2:绿
        self.state_timer = 0
        self.state_durations = [3.0, 1.0, 3.0]  # 红、黄、绿的持续时间（秒）

    def update(self, delta_time: float, world_width: float, world_height: float):
        # 交通灯状态更新
        self.state_timer += delta_time
        if self.state_timer > self.state_durations[self.current_state]:
            self.current_state = (self.current_state + 1) % 3
            self.state_timer = 0

    def draw(self, surface: pygame.Surface):
        # 绘制灯杆
        pygame.draw.rect(surface, GRAY, (self.x, self.y, self.width, self.height))
        # 绘制灯
        light_radius = 8
        light_spacing = 15
        # 红灯
        color = RED if self.current_state == 0 else (100, 0, 0)
        pygame.draw.circle(surface, color, (int(self.x + self.width / 2), int(self.y + light_spacing)), light_radius)
        # 黄灯
        color = YELLOW if self.current_state == 1 else (100, 100, 0)
        pygame.draw.circle(surface, color, (int(self.x + self.width / 2), int(self.y + light_spacing * 2)),
                           light_radius)
        # 绿灯
        color = GREEN if self.current_state == 2 else (0, 100, 0)
        pygame.draw.circle(surface, color, (int(self.x + self.width / 2), int(self.y + light_spacing * 3)),
                           light_radius)


class InstanceSegmentationCamera:
    """实例分割相机类"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.objects: List[GameObject] = []
        self.instance_counter: Dict[str, int] = {}  # 按类型计数实例
        self.font = pygame.font.SysFont(None, 24)

    def add_object(self, obj: GameObject):
        """添加对象到场景中"""
        # 为对象分配实例ID
        if obj.obj_type not in self.instance_counter:
            self.instance_counter[obj.obj_type] = 0
        obj.instance_id = self.instance_counter[obj.obj_type]
        self.instance_counter[obj.obj_type] += 1
        self.objects.append(obj)

    def remove_random_object(self):
        """随机移除一个对象"""
        if self.objects:
            idx = random.randint(0, len(self.objects) - 1)
            removed = self.objects.pop(idx)

    def update(self, delta_time: float):
        """更新所有对象"""
        for obj in self.objects:
            obj.update(delta_time, self.width, self.height)

    def capture_raw_view(self) -> pygame.Surface:
        """捕获原始视图（不进行实例分割标记）"""
        surface = pygame.Surface((self.width, self.height))
        surface.fill(WHITE)

        # 绘制所有对象
        for obj in self.objects:
            obj.draw(surface)

        return surface

    def capture_segmented_view(self) -> pygame.Surface:
        """捕获实例分割视图（同一类别的不同对象用不同颜色标记）"""
        surface = pygame.Surface((self.width, self.height))
        surface.fill(WHITE)

        # 按类型对对象进行分组
        objects_by_type: Dict[str, List[GameObject]] = {}
        for obj in self.objects:
            if obj.obj_type not in objects_by_type:
                objects_by_type[obj.obj_type] = []
            objects_by_type[obj.obj_type].append(obj)

        # 绘制分割后的对象
        for obj_type, objs in objects_by_type.items():
            for i, obj in enumerate(objs):
                # 为每个实例分配不同的颜色
                color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]

                # 绘制实例
                if isinstance(obj, Car):
                    pygame.draw.rect(surface, color, (obj.x, obj.y, obj.width, obj.height))
                    # 绘制实例ID
                    text = self.font.render(f"{obj_type}_{obj.instance_id}", True, BLACK)
                    surface.blit(text, (obj.x, obj.y - 20))
                elif isinstance(obj, Pedestrian):
                    pygame.draw.rect(surface, color, (obj.x, obj.y, obj.width, obj.height))
                    # 绘制实例ID
                    text = self.font.render(f"{obj_type}_{obj.instance_id}", True, BLACK)
                    surface.blit(text, (obj.x, obj.y - 20))
                elif isinstance(obj, TrafficLight):
                    pygame.draw.rect(surface, color, (obj.x, obj.y, obj.width, obj.height))
                    # 绘制实例ID
                    text = self.font.render(f"{obj_type}_{obj.instance_id}", True, BLACK)
                    surface.blit(text, (obj.x, obj.y - 20))

        # 绘制图例
        self._draw_legend(surface, objects_by_type)

        return surface

    def _draw_legend(self, surface: pygame.Surface, objects_by_type: Dict[str, List[GameObject]]):
        """绘制图例，说明每个实例的颜色对应的对象"""
        legend_y = 10
        legend_x = self.width - 180

        pygame.draw.rect(surface, (240, 240, 240), (legend_x - 10, legend_y - 10, 170, 10 + len(INSTANCE_COLORS) * 25))
        pygame.draw.rect(surface, BLACK, (legend_x - 10, legend_y - 10, 170, 10 + len(INSTANCE_COLORS) * 25), 2)

        title = self.font.render("实例分割图例", True, BLACK)
        surface.blit(title, (legend_x, legend_y))
        legend_y += 30

        for i, color in enumerate(INSTANCE_COLORS):
            pygame.draw.rect(surface, color, (legend_x, legend_y, 15, 15))
            text = self.font.render(f"实例 #{i}", True, BLACK)
            surface.blit(text, (legend_x + 25, legend_y))
            legend_y += 25


class SelfDrivingCarSimulator:
    """无人车模拟器"""

    def __init__(self, width: int = 1200, height: int = 600):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("无人车实例分割相机模拟")

        # 创建相机视图区域（分为左右两部分）
        self.camera_width = width // 2
        self.camera_height = height

        # 创建实例分割相机
        self.camera = InstanceSegmentationCamera(self.camera_width, self.camera_height)

        # 添加一些初始对象
        self._initialize_objects()

        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.SysFont(None, 30)

    def _initialize_objects(self):
        """初始化场景对象"""
        # 添加车辆
        for _ in range(3):
            x = random.randint(50, self.camera_width - 100)
            y = random.randint(50, self.camera_height - 100)
            self.camera.add_object(Car(x, y))

        # 添加行人
        for _ in range(4):
            x = random.randint(50, self.camera_width - 50)
            y = random.randint(50, self.camera_height - 50)
            self.camera.add_object(Pedestrian(x, y))

        # 添加交通灯
        self.camera.add_object(TrafficLight(50, self.camera_height // 2 - 30))
        self.camera.add_object(TrafficLight(self.camera_width - 70, self.camera_height // 2 - 30))

    def handle_events(self):
        """处理用户输入事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_c:  # 添加车辆
                    x = random.randint(50, self.camera_width - 100)
                    y = random.randint(50, self.camera_height - 100)
                    self.camera.add_object(Car(x, y))
                elif event.key == pygame.K_p:  # 添加行人
                    x = random.randint(50, self.camera_width - 50)
                    y = random.randint(50, self.camera_height - 50)
                    self.camera.add_object(Pedestrian(x, y))
                elif event.key == pygame.K_r:  # 移除随机对象
                    self.camera.remove_random_object()

    def update(self, delta_time: float):
        """更新模拟状态"""
        self.camera.update(delta_time)

    def render(self):
        """渲染画面"""
        self.screen.fill(WHITE)

        # 获取两种视图
        raw_view = self.camera.capture_raw_view()
        segmented_view = self.camera.capture_segmented_view()

        # 绘制原始视图（左侧）
        self.screen.blit(raw_view, (0, 0))
        # 绘制分割视图（右侧）
        self.screen.blit(segmented_view, (self.camera_width, 0))

        # 绘制视图分隔线
        pygame.draw.line(self.screen, BLACK, (self.camera_width, 0),
                         (self.camera_width, self.camera_height), 3)

        # 绘制标题
        raw_title = self.font.render("原始相机视图", True, BLACK)
        self.screen.blit(raw_title, (self.camera_width // 2 - 80, 10))

        seg_title = self.font.render("实例分割视图", True, BLACK)
        self.screen.blit(seg_title, (self.camera_width + self.camera_width // 2 - 80, 10))

        # 绘制操作提示
        help_texts = [
            "操作提示:",
            "C键: 添加车辆",
            "P键: 添加行人",
            "R键: 移除随机对象",
            "ESC键: 退出"
        ]

        for i, text in enumerate(help_texts):
            text_surface = self.font.render(text, True, BLACK)
            self.screen.blit(text_surface, (10, self.height - 30 * (len(help_texts) - i)))

        pygame.display.flip()

    def run(self):
        """运行模拟器"""
        while self.running:
            delta_time = self.clock.tick(60) / 1000.0  # 转换为秒
            self.handle_events()
            self.update(delta_time)
            self.render()

        pygame.quit()


if __name__ == "__main__":
    simulator = SelfDrivingCarSimulator()
    simulator.run()