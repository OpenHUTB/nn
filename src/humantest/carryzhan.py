import pygame
import heapq
import sys
from enum import Enum

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
ORANGE = (255, 165, 0)

# 定义网格大小和单元格大小
CELL_SIZE = 40
GRID_WIDTH = 20
GRID_HEIGHT = 15

# 定义状态枚举
class State(Enum):
    IDLE = 0
    MOVING_TO_BRICK = 1
    PICKING_BRICK = 2
    MOVING_TO_TARGET = 3
    DROPPING_BRICK = 4

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = State.IDLE
        self.carrying_brick = False
        self.path = []
        self.current_path_index = 0
        self.pickup_timer = 0
        self.drop_timer = 0

    def update(self, grid):
        if self.state == State.IDLE:
            # 寻找最近的砖块
            nearest_brick = self.find_nearest_brick(grid)
            if nearest_brick:
                # 计算到砖块的路径
                self.path = self.a_star(grid, (self.x, self.y), nearest_brick)
                if self.path:
                    self.current_path_index = 0
                    self.state = State.MOVING_TO_BRICK
                else:
                    print("No path to brick found")
            else:
                # 没有砖块，寻找目标位置
                target_pos = self.find_target_position(grid)
                if target_pos and (self.x, self.y) != target_pos:
                    self.path = self.a_star(grid, (self.x, self.y), target_pos)
                    if self.path:
                        self.current_path_index = 0
                        self.state = State.MOVING_TO_TARGET
                    else:
                        print("No path to target found")
        
        elif self.state == State.MOVING_TO_BRICK:
            if self.current_path_index < len(self.path):
                next_x, next_y = self.path[self.current_path_index]
                # 移动到下一个位置
                self.x, self.y = next_x, next_y
                self.current_path_index += 1
            else:
                # 路径结束，检查是否到达砖块位置
                if grid[self.y][self.x] == CellType.BRICK:
                    self.state = State.PICKING_BRICK
                    self.pickup_timer = 30  # 30帧的拾取动画
                else:
                    self.state = State.IDLE
        
        elif self.state == State.PICKING_BRICK:
            if self.pickup_timer > 0:
                self.pickup_timer -= 1
            else:
                # 拾取砖块
                grid[self.y][self.x] = CellType.EMPTY
                self.carrying_brick = True
                
                # 寻找目标位置
                target_pos = self.find_target_position(grid)
                if target_pos:
                    self.path = self.a_star(grid, (self.x, self.y), target_pos)
                    if self.path:
                        self.current_path_index = 0
                        self.state = State.MOVING_TO_TARGET
                    else:
                        print("No path to target found")
                        self.state = State.IDLE
                else:
                    self.state = State.IDLE
        
        elif self.state == State.MOVING_TO_TARGET:
            if self.current_path_index < len(self.path):
                next_x, next_y = self.path[self.current_path_index]
                # 移动到下一个位置
                self.x, self.y = next_x, next_y
                self.current_path_index += 1
            else:
                # 路径结束，检查是否到达目标位置
                if grid[self.y][self.x] == CellType.TARGET:
                    self.state = State.DROPPING_BRICK
                    self.drop_timer = 30  # 30帧的放置动画
                else:
                    self.state = State.IDLE
        
        elif self.state == State.DROPPING_BRICK:
            if self.drop_timer > 0:
                self.drop_timer -= 1
            else:
                # 放置砖块
                self.carrying_brick = False
                
                # 将砖块放置到目标位置
                if grid[self.y][self.x] == CellType.TARGET:
                    grid[self.y][self.x] = CellType.BRICK_ON_TARGET
                    print(f"Brick placed on target at ({self.x}, {self.y})")
                
                # 寻找下一个砖块
                nearest_brick = self.find_nearest_brick(grid)
                if nearest_brick:
                    self.path = self.a_star(grid, (self.x, self.y), nearest_brick)
                    if self.path:
                        self.current_path_index = 0
                        self.state = State.MOVING_TO_BRICK
                    else:
                        print("No path to next brick found")
                        self.state = State.IDLE
                else:
                    self.state = State.IDLE

    def find_nearest_brick(self, grid):
        min_distance = float('inf')
        nearest_brick = None
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if grid[y][x] == CellType.BRICK:
                    distance = abs(self.x - x) + abs(self.y - y)  # Manhattan distance
                    if distance < min_distance:
                        min_distance = distance
                        nearest_brick = (x, y)
        return nearest_brick

    def find_target_position(self, grid):
        # 寻找第一个目标位置
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if grid[y][x] == CellType.TARGET:
                    return (x, y)
        return None

    def a_star(self, grid, start, goal):
        # A*寻路算法
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # 检查起点和终点是否有效
        if (start[0] < 0 or start[0] >= GRID_WIDTH or start[1] < 0 or start[1] >= GRID_HEIGHT or
            goal[0] < 0 or goal[0] >= GRID_WIDTH or goal[1] < 0 or goal[1] >= GRID_HEIGHT):
            return []
        
        # 检查终点是否是障碍物
        if grid[goal[1]][goal[0]] == CellType.OBSTACLE:
            return []
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path[1:]  # 移除起点
            
            # 检查四个方向
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 检查边界
                if (neighbor[0] < 0 or neighbor[0] >= GRID_WIDTH or
                    neighbor[1] < 0 or neighbor[1] >= GRID_HEIGHT):
                    continue
                
                # 检查障碍物
                if grid[neighbor[1]][neighbor[0]] == CellType.OBSTACLE:
                    continue
                
                # 计算移动成本
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # 没有找到路径
        return []

    def draw(self, screen):
        # 绘制机器人
        rect = pygame.Rect(self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, BLUE, rect)
        
        # 如果携带砖块，绘制砖块
        if self.carrying_brick:
            brick_rect = pygame.Rect(
                self.x * CELL_SIZE + 10, 
                self.y * CELL_SIZE + 10, 
                CELL_SIZE - 20, 
                CELL_SIZE - 20
            )
            pygame.draw.rect(screen, ORANGE, brick_rect)
        
        # 绘制状态指示器
        state_colors = {
            State.IDLE: GREEN,
            State.MOVING_TO_BRICK: YELLOW,
            State.PICKING_BRICK: RED,
            State.MOVING_TO_TARGET: YELLOW,
            State.DROPPING_BRICK: RED
        }
        
        indicator_rect = pygame.Rect(
            self.x * CELL_SIZE + CELL_SIZE - 10, 
            self.y * CELL_SIZE + CELL_SIZE - 10, 
            8, 
            8
        )
        pygame.draw.ellipse(screen, state_colors[self.state], indicator_rect)

class CellType(Enum):
    EMPTY = 0
    OBSTACLE = 1
    BRICK = 2
    TARGET = 3
    BRICK_ON_TARGET = 4

class Grid:
    def __init__(self):
        self.cells = [[CellType.EMPTY for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.brick_count = 0
        self.bricks_placed = 0
        self.total_bricks = 0

    def reset(self):
        self.cells = [[CellType.EMPTY for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.brick_count = 0
        self.bricks_placed = 0
        self.total_bricks = 0

    def add_obstacle(self, x, y):
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            self.cells[y][x] = CellType.OBSTACLE

    def add_brick(self, x, y):
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            self.cells[y][x] = CellType.BRICK
            self.brick_count += 1
            self.total_bricks += 1

    def add_target(self, x, y):
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            self.cells[y][x] = CellType.TARGET

    def update_brick_count(self):
        self.brick_count = 0
        self.bricks_placed = 0
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.cells[y][x] == CellType.BRICK:
                    self.brick_count += 1
                elif self.cells[y][x] == CellType.BRICK_ON_TARGET:
                    self.bricks_placed += 1

    def draw(self, screen):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # 绘制单元格背景
                if self.cells[y][x] == CellType.EMPTY:
                    pygame.draw.rect(screen, WHITE, rect)
                elif self.cells[y][x] == CellType.OBSTACLE:
                    pygame.draw.rect(screen, GRAY, rect)
                elif self.cells[y][x] == CellType.BRICK:
                    pygame.draw.rect(screen, WHITE, rect)
                    brick_rect = pygame.Rect(
                        x * CELL_SIZE + 5, 
                        y * CELL_SIZE + 5, 
                        CELL_SIZE - 10, 
                        CELL_SIZE - 10
                    )
                    pygame.draw.rect(screen, ORANGE, brick_rect)
                elif self.cells[y][x] == CellType.TARGET:
                    pygame.draw.rect(screen, WHITE, rect)
                    target_rect = pygame.Rect(
                        x * CELL_SIZE + 5, 
                        y * CELL_SIZE + 5, 
                        CELL_SIZE - 10, 
                        CELL_SIZE - 10
                    )
                    pygame.draw.rect(screen, GREEN, target_rect, 2)
                    pygame.draw.line(screen, GREEN, 
                                   (x * CELL_SIZE + 10, y * CELL_SIZE + 10),
                                   ((x + 1) * CELL_SIZE - 10, (y + 1) * CELL_SIZE - 10), 2)
                    pygame.draw.line(screen, GREEN, 
                                   ((x + 1) * CELL_SIZE - 10, y * CELL_SIZE + 10),
                                   (x * CELL_SIZE + 10, (y + 1) * CELL_SIZE - 10), 2)
                elif self.cells[y][x] == CellType.BRICK_ON_TARGET:
                    pygame.draw.rect(screen, WHITE, rect)
                    target_rect = pygame.Rect(
                        x * CELL_SIZE + 5, 
                        y * CELL_SIZE + 5, 
                        CELL_SIZE - 10, 
                        CELL_SIZE - 10
                    )
                    pygame.draw.rect(screen, GREEN, target_rect, 2)
                    pygame.draw.line(screen, GREEN, 
                                   (x * CELL_SIZE + 10, y * CELL_SIZE + 10),
                                   ((x + 1) * CELL_SIZE - 10, (y + 1) * CELL_SIZE - 10), 2)
                    pygame.draw.line(screen, GREEN, 
                                   ((x + 1) * CELL_SIZE - 10, y * CELL_SIZE + 10),
                                   (x * CELL_SIZE + 10, (y + 1) * CELL_SIZE - 10), 2)
                    
                    brick_rect = pygame.Rect(
                        x * CELL_SIZE + 10, 
                        y * CELL_SIZE + 10, 
                        CELL_SIZE - 20, 
                        CELL_SIZE - 20
                    )
                    pygame.draw.rect(screen, ORANGE, brick_rect)
                
                # 绘制网格线
                pygame.draw.rect(screen, BLACK, rect, 1)

class Game:
    def __init__(self):
        pygame.init()
        self.screen_width = GRID_WIDTH * CELL_SIZE + 200  # 额外空间用于控制面板
        self.screen_height = GRID_HEIGHT * CELL_SIZE
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("机器人搬砖模拟")
        
        self.clock = pygame.time.Clock()
        # 使用系统字体
        try:
            # 尝试加载支持中文的字体
            self.font = pygame.font.SysFont(['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif'], 24)
            # 完全删除字体名称打印语句
        except Exception as e:
            print(f"加载字体失败: {e}")
            print("使用默认字体")
            self.font = pygame.font.Font(None, 24)
        
        
        self.grid = Grid()
        self.robot = Robot(1, 1)
        
        self.mode = "brick"  # 默认模式：放置砖块
        self.running = True
        self.paused = False
        self.show_path = True
        
        # 初始化场景
        self.init_scene()

    def init_scene(self):
        self.grid.reset()
        
        # 添加一些障碍物
        for x in range(5, 10):
            self.grid.add_obstacle(x, 5)
        for y in range(3, 8):
            self.grid.add_obstacle(10, y)
        
        # 添加砖块
        self.grid.add_brick(3, 3)
        self.grid.add_brick(3, 7)
        self.grid.add_brick(7, 3)
        self.grid.add_brick(15, 3)
        self.grid.add_brick(15, 7)
        
        # 添加目标位置
        self.grid.add_target(15, 12)
        self.grid.add_target(16, 12)
        self.grid.add_target(17, 12)
        self.grid.add_target(18, 12)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.init_scene()
                    self.robot = Robot(1, 1)
                elif event.key == pygame.K_p:
                    self.show_path = not self.show_path
                elif event.key == pygame.K_1:
                    self.mode = "brick"
                elif event.key == pygame.K_2:
                    self.mode = "target"
                elif event.key == pygame.K_3:
                    self.mode = "obstacle"
                elif event.key == pygame.K_4:
                    self.mode = "clear"
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    x, y = pygame.mouse.get_pos()
                    grid_x = x // CELL_SIZE
                    grid_y = y // CELL_SIZE
                    
                    if grid_x < GRID_WIDTH and grid_y < GRID_HEIGHT:
                        if self.mode == "brick":
                            self.grid.add_brick(grid_x, grid_y)
                        elif self.mode == "target":
                            self.grid.add_target(grid_x, grid_y)
                        elif self.mode == "obstacle":
                            self.grid.add_obstacle(grid_x, grid_y)
                        elif self.mode == "clear":
                            self.grid.cells[grid_y][grid_x] = CellType.EMPTY
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:  # 右键点击
                    x, y = pygame.mouse.get_pos()
                    grid_x = x // CELL_SIZE
                    grid_y = y // CELL_SIZE
                    
                    if grid_x < GRID_WIDTH and grid_y < GRID_HEIGHT:
                        # 设置机器人位置
                        self.robot.x = grid_x
                        self.robot.y = grid_y
                        self.robot.state = State.IDLE
                        self.robot.path = []

    def draw_ui(self):
        # 绘制控制面板背景
        ui_rect = pygame.Rect(GRID_WIDTH * CELL_SIZE, 0, 200, self.screen_height)
        pygame.draw.rect(self.screen, GRAY, ui_rect)
        
        # 辅助函数：绘制带背景的文字
        def draw_text_with_bg(text, color, x, y):
            text_surface = self.font.render(text, True, color)
            # 创建白色背景
            bg_surface = pygame.Surface((text_surface.get_width() + 8, text_surface.get_height() + 4))
            bg_surface.fill(WHITE)
            bg_surface.blit(text_surface, (4, 2))
            self.screen.blit(bg_surface, (x, y))
            return text_surface.get_width() + 8
        
        # 绘制标题
        draw_text_with_bg("机器人搬砖模拟", BLACK, GRID_WIDTH * CELL_SIZE + 15, 8)
        
        # 绘制模式选择
        draw_text_with_bg("模式:", BLACK, GRID_WIDTH * CELL_SIZE + 20, 50)
        
        # 模式按钮
        brick_color = RED if self.mode == "brick" else BLACK
        target_color = RED if self.mode == "target" else BLACK
        obstacle_color = RED if self.mode == "obstacle" else BLACK
        clear_color = RED if self.mode == "clear" else BLACK
        
        draw_text_with_bg("砖块 (1)", brick_color, GRID_WIDTH * CELL_SIZE + 20, 80)
        draw_text_with_bg("目标 (2)", target_color, GRID_WIDTH * CELL_SIZE + 20, 110)
        draw_text_with_bg("障碍 (3)", obstacle_color, GRID_WIDTH * CELL_SIZE + 20, 140)
        draw_text_with_bg("清除 (4)", clear_color, GRID_WIDTH * CELL_SIZE + 20, 170)
        
        # 绘制控制按钮
        pause_color = RED if self.paused else BLACK
        path_color = RED if self.show_path else BLACK
        
        draw_text_with_bg("重置 (R)", BLACK, GRID_WIDTH * CELL_SIZE + 20, 210)
        draw_text_with_bg("暂停 (空格)", pause_color, GRID_WIDTH * CELL_SIZE + 20, 240)
        draw_text_with_bg("显示路径 (P)", path_color, GRID_WIDTH * CELL_SIZE + 20, 270)
        
        # 绘制机器人状态
        draw_text_with_bg("机器人状态:", BLACK, GRID_WIDTH * CELL_SIZE + 20, 310)
        
        state_names = {
            State.IDLE: "空闲",
            State.MOVING_TO_BRICK: "前往砖块",
            State.PICKING_BRICK: "拾取砖块",
            State.MOVING_TO_TARGET: "前往目标",
            State.DROPPING_BRICK: "放置砖块"
        }
        
        draw_text_with_bg(state_names[self.robot.state], BLACK, GRID_WIDTH * CELL_SIZE + 20, 340)
        
        # 绘制携带状态
        carry_text = "携带砖块: 是" if self.robot.carrying_brick else "携带砖块: 否"
        carry_color = RED if self.robot.carrying_brick else BLACK
        draw_text_with_bg(carry_text, carry_color, GRID_WIDTH * CELL_SIZE + 20, 370)
        
        # 绘制砖块统计
        self.grid.update_brick_count()
        draw_text_with_bg(f"剩余砖块: {self.grid.brick_count}", BLACK, GRID_WIDTH * CELL_SIZE + 20, 400)
        draw_text_with_bg(f"已放置: {self.grid.bricks_placed}/{self.grid.total_bricks}", BLACK, GRID_WIDTH * CELL_SIZE + 20, 430)
        
        # 绘制操作提示
        draw_text_with_bg("左键: 放置元素", BLACK, GRID_WIDTH * CELL_SIZE + 20, 470)
        draw_text_with_bg("右键: 移动机器人", BLACK, GRID_WIDTH * CELL_SIZE + 20, 500)

    def draw_path(self):
        if self.show_path and self.robot.path:
            for i in range(len(self.robot.path) - 1):
                x1, y1 = self.robot.path[i]
                x2, y2 = self.robot.path[i + 1]
                
                start_pos = (
                    x1 * CELL_SIZE + CELL_SIZE // 2,
                    y1 * CELL_SIZE + CELL_SIZE // 2
                )
                end_pos = (
                    x2 * CELL_SIZE + CELL_SIZE // 2,
                    y2 * CELL_SIZE + CELL_SIZE // 2
                )
                
                pygame.draw.line(self.screen, YELLOW, start_pos, end_pos, 2)

    def run(self):
        while self.running:
            self.handle_events()
            
            if not self.paused:
                self.robot.update(self.grid.cells)
            
            self.screen.fill(WHITE)
            self.grid.draw(self.screen)
            
            if not self.paused:
                self.draw_path()
            
            self.robot.draw(self.screen)
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
