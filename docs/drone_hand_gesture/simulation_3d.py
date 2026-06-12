"""
3D仿真显示模块 - 无需GLUT版本
"""
import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *


class Drone3DViewer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height

        # 初始化Pygame和OpenGL
        pygame.init()
        self.screen = pygame.display.set_mode((width, height),
                                              pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("无人机3D仿真系统")

        # OpenGL设置
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)

        # 光源
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

        # 视角
        gluPerspective(45, (width / height), 0.1, 100.0)
        glTranslatef(0.0, -3.0, -15)

        # 无人机模型参数
        self.drone_model = DroneModel()

        # 环境
        self.grid_size = 20
        self.show_grid = True
        self.show_trajectory = True
        self.show_axes = True
        self.show_waypoints = True

        # 相机参数
        self.camera_distance = 15.0
        self.camera_angle_x = 0.0
        self.camera_angle_y = -20.0

    def update_camera(self):
        """更新相机位置"""
        glLoadIdentity()
        gluPerspective(45, (self.width / self.height), 0.1, 100.0)

        # 计算相机位置
        cam_x = self.camera_distance * np.sin(np.radians(self.camera_angle_y)) * np.cos(np.radians(self.camera_angle_x))
        cam_y = self.camera_distance * np.cos(np.radians(self.camera_angle_y))
        cam_z = self.camera_distance * np.sin(np.radians(self.camera_angle_y)) * np.sin(np.radians(self.camera_angle_x))

        gluLookAt(cam_x, cam_y, cam_z,  # 相机位置
                  0, 0, 0,  # 观察点
                  0, 1, 0)  # 上方向

    def render(self, drone_state=None, trajectory=None, waypoints=None):
        """渲染整个场景
        
        Args:
            drone_state: 无人机状态
            trajectory: 轨迹数据
            waypoints: 航点列表
        """
        # 清除缓冲区
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.1, 0.1, 0.15, 1.0)  # 深蓝色背景

        # 更新相机
        self.update_camera()

        # 绘制环境
        if self.show_grid:
            self._draw_grid()

        if self.show_axes:
            self._draw_coordinate_axes()

        # 绘制无人机（如果有状态数据）
        if drone_state:
            self._draw_drone(drone_state)
        else:
            # 如果没有状态数据，绘制默认位置的无人机
            default_state = {
                'position': [0.0, 2.0, 0.0],
                'orientation': [0.0, 0.0, 0.0],
                'armed': True,
                'mode': 'HOVER'
            }
            self._draw_drone(default_state)

        # 绘制轨迹（如果有）
        if self.show_trajectory and trajectory:
            self._draw_trajectory(trajectory)

        # 绘制航点（如果有）
        if self.show_waypoints and waypoints:
            self._draw_waypoints(waypoints)

        # 绘制状态信息
        if drone_state:
            self._draw_status_overlay(drone_state)
        else:
            self._draw_default_overlay()

        # 交换缓冲区
        pygame.display.flip()

    def _draw_grid(self):
        """绘制地面网格"""
        glBegin(GL_LINES)
        glColor3f(0.3, 0.3, 0.3)  # 灰色

        for i in range(-self.grid_size, self.grid_size + 1):
            # X方向线
            glVertex3f(i, 0, -self.grid_size)
            glVertex3f(i, 0, self.grid_size)
            # Z方向线
            glVertex3f(-self.grid_size, 0, i)
            glVertex3f(self.grid_size, 0, i)

        glEnd()

    def _draw_coordinate_axes(self):
        """绘制坐标轴"""
        glLineWidth(2.0)
        glBegin(GL_LINES)

        # X轴 (红色)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(2, 0, 0)

        # Y轴 (绿色)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 2, 0)

        # Z轴 (蓝色)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 2)

        glEnd()
        glLineWidth(1.0)

    def _draw_drone(self, state):
        """绘制无人机"""
        x, y, z = state.get('position', [0.0, 2.0, 0.0])
        roll, pitch, yaw = state.get('orientation', [0.0, 0.0, 0.0])

        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(yaw, 0, 1, 0)
        glRotatef(pitch, 1, 0, 0)
        glRotatef(roll, 0, 0, 1)

        # 根据状态设置颜色
        armed = state.get('armed', True)
        mode = state.get('mode', 'HOVER')

        if armed:
            if mode == 'TAKEOFF':
                color = (0.0, 1.0, 0.0)  # 绿色：起飞中
            elif mode == 'LAND':
                color = (1.0, 0.5, 0.0)  # 橙色：降落中
            elif mode == 'HOVER':
                color = (0.0, 0.8, 1.0)  # 青色：悬停
            else:
                color = (0.0, 0.6, 0.0)  # 深绿色：飞行中
        else:
            color = (0.5, 0.5, 0.5)  # 灰色：未解锁

        self.drone_model.draw(color)
        glPopMatrix()

    def _draw_trajectory(self, trajectory):
        """绘制飞行轨迹"""
        if len(trajectory) < 2:
            return

        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)

        # 渐变颜色：从蓝色到红色
        for i, (x, y, z) in enumerate(trajectory):
            t = i / len(trajectory)
            r = t
            g = 0.5 - 0.5 * t
            b = 1.0 - t
            glColor3f(r, g, b)
            glVertex3f(x, y, z)

        glEnd()
        glLineWidth(1.0)

    def _draw_waypoints(self, waypoints):
        """绘制航点标记"""
        if not waypoints:
            return

        # 航点颜色配置
        waypoint_colors = {
            '起飞': (0.0, 1.0, 0.0),      # 绿色
            '左转': (1.0, 0.5, 0.0),      # 橙色
            '右转': (1.0, 0.5, 0.0),      # 橙色
            '上升': (0.0, 0.5, 1.0),      # 蓝色
            '下降': (0.5, 0.5, 1.0),      # 浅蓝色
            '悬停': (1.0, 1.0, 0.0),      # 黄色
            '降落': (1.0, 0.0, 0.0),      # 红色
        }

        for i, waypoint in enumerate(waypoints):
            # 获取航点位置
            if hasattr(waypoint, 'position'):
                pos = waypoint.position
                label = waypoint.label
                index = waypoint.index
            else:
                pos = waypoint.get('position', [0, 0, 0])
                label = waypoint.get('label', f'WP{i}')
                index = waypoint.get('index', i)

            x, y, z = pos[0], pos[1], pos[2]

            # 获取颜色
            color = waypoint_colors.get(label, (1.0, 1.0, 0.0))  # 默认黄色

            # 绘制航点标记（带圆环的立柱）
            glColor3f(*color)

            # 绘制立柱
            glLineWidth(2.0)
            glBegin(GL_LINES)
            glVertex3f(x, 0, z)
            glVertex3f(x, y, z)
            glEnd()
            glLineWidth(1.0)

            # 绘制地面圆圈
            glColor3f(color[0] * 0.7, color[1] * 0.7, color[2] * 0.7)
            glBegin(GL_LINE_LOOP)
            for j in range(32):
                angle = 2 * np.pi * j / 32
                circle_x = x + 0.3 * np.cos(angle)
                circle_z = z + 0.3 * np.sin(angle)
                glVertex3f(circle_x, 0.01, circle_z)
            glEnd()

            # 绘制顶部标记
            glColor3f(*color)
            self._draw_waypoint_marker(x, y, z)

    def _draw_waypoint_marker(self, x, y, z):
        """绘制航点标记（菱形）"""
        marker_size = 0.2

        glBegin(GL_TRIANGLES)
        # 四个三角形组成的菱形
        # 顶部
        glVertex3f(x, y + marker_size, z)
        glVertex3f(x - marker_size * 0.5, y, z - marker_size * 0.5)
        glVertex3f(x + marker_size * 0.5, y, z + marker_size * 0.5)

        glVertex3f(x, y + marker_size, z)
        glVertex3f(x + marker_size * 0.5, y, z - marker_size * 0.5)
        glVertex3f(x - marker_size * 0.5, y, z + marker_size * 0.5)

        # 底部
        glVertex3f(x, y - marker_size * 0.5, z)
        glVertex3f(x - marker_size * 0.5, y, z - marker_size * 0.5)
        glVertex3f(x + marker_size * 0.5, y, z + marker_size * 0.5)

        glVertex3f(x, y - marker_size * 0.5, z)
        glVertex3f(x + marker_size * 0.5, y, z - marker_size * 0.5)
        glVertex3f(x - marker_size * 0.5, y, z + marker_size * 0.5)
        glEnd()

    def _draw_status_overlay(self, state):
        """绘制状态信息覆盖层"""
        # 暂时禁用文本绘制，只保留3D场景渲染
        pass

    def _draw_default_overlay(self):
        """绘制默认覆盖层（无状态数据时）"""
        # 暂时禁用文本绘制，只保留3D场景渲染
        pass

    def handle_events(self):
        """处理窗口事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                    print(f"网格显示: {'开' if self.show_grid else '关'}")
                elif event.key == pygame.K_t:
                    self.show_trajectory = not self.show_trajectory
                    print(f"轨迹显示: {'开' if self.show_trajectory else '关'}")
                elif event.key == pygame.K_a:
                    self.show_axes = not self.show_axes
                    print(f"坐标轴显示: {'开' if self.show_axes else '关'}")
                elif event.key == pygame.K_SPACE:
                    # 重置视角
                    self.camera_distance = 15.0
                    self.camera_angle_x = 0.0
                    self.camera_angle_y = -20.0
                    print("视角已重置")
                elif event.key == pygame.K_UP:
                    self.camera_angle_y = min(89, self.camera_angle_y + 5)
                elif event.key == pygame.K_DOWN:
                    self.camera_angle_y = max(-89, self.camera_angle_y - 5)
                elif event.key == pygame.K_LEFT:
                    self.camera_angle_x -= 5
                elif event.key == pygame.K_RIGHT:
                    self.camera_angle_x += 5
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.camera_distance = max(5, self.camera_distance - 1)
                elif event.key == pygame.K_MINUS:
                    self.camera_distance = min(50, self.camera_distance + 1)
                elif event.key == pygame.K_w:
                    self.show_waypoints = not self.show_waypoints
                    print(f"航点显示: {'开' if self.show_waypoints else '关'}")
        return True


class DroneModel:
    """无人机3D模型（无需GLUT版本）"""

    def __init__(self):
        self.body_size = [0.8, 0.15, 0.8]  # 机身尺寸
        self.arm_length = 0.7  # 机臂长度
        self.propeller_radius = 0.3  # 螺旋桨半径

    def draw(self, color):
        """绘制无人机模型"""
        glColor3f(*color)

        # 绘制机身
        self._draw_body()

        # 绘制机臂
        self._draw_arms()

        # 绘制螺旋桨
        self._draw_propellers()

    def _draw_body(self):
        """绘制机身"""
        w, h, d = self.body_size
        self._draw_cube(w, h, d)

    def _draw_cube(self, width, height, depth):
        """绘制立方体"""
        w, h, d = width / 2, height / 2, depth / 2

        glBegin(GL_QUADS)

        # 前面 (Z = +d)
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(-w, -h, d)
        glVertex3f(w, -h, d)
        glVertex3f(w, h, d)
        glVertex3f(-w, h, d)

        # 后面 (Z = -d)
        glNormal3f(0.0, 0.0, -1.0)
        glVertex3f(-w, -h, -d)
        glVertex3f(-w, h, -d)
        glVertex3f(w, h, -d)
        glVertex3f(w, -h, -d)

        # 上面 (Y = +h)
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(-w, h, d)
        glVertex3f(w, h, d)
        glVertex3f(w, h, -d)
        glVertex3f(-w, h, -d)

        # 下面 (Y = -h)
        glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(-w, -h, d)
        glVertex3f(-w, -h, -d)
        glVertex3f(w, -h, -d)
        glVertex3f(w, -h, d)

        # 右面 (X = +w)
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(w, -h, d)
        glVertex3f(w, -h, -d)
        glVertex3f(w, h, -d)
        glVertex3f(w, h, d)

        # 左面 (X = -w)
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(-w, -h, d)
        glVertex3f(-w, h, d)
        glVertex3f(-w, h, -d)
        glVertex3f(-w, -h, -d)

        glEnd()

    def _draw_arms(self):
        """绘制机臂"""
        arm_positions = [
            (-self.arm_length, 0, 0),
            (self.arm_length, 0, 0),
            (0, 0, -self.arm_length),
            (0, 0, self.arm_length)
        ]

        glColor3f(0.3, 0.3, 0.3)  # 深灰色机臂

        for x, y, z in arm_positions:
            glPushMatrix()
            glTranslatef(x, y, z)

            # 绘制圆柱体机臂
            self._draw_cylinder(0.03, self.arm_length * 0.8)
            glPopMatrix()

    def _draw_cylinder(self, radius, height):
        """绘制圆柱体"""
        slices = 16
        stacks = 1

        # 绘制侧面
        glBegin(GL_QUAD_STRIP)
        for i in range(slices + 1):
            angle = 2 * np.pi * i / slices
            x = np.cos(angle) * radius
            z = np.sin(angle) * radius

            glNormal3f(np.cos(angle), 0, np.sin(angle))
            glVertex3f(x, -height / 2, z)
            glVertex3f(x, height / 2, z)
        glEnd()

        # 绘制顶部圆盘
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0, 1, 0)
        glVertex3f(0, height / 2, 0)
        for i in range(slices + 1):
            angle = 2 * np.pi * i / slices
            x = np.cos(angle) * radius
            z = np.sin(angle) * radius
            glVertex3f(x, height / 2, z)
        glEnd()

        # 绘制底部圆盘
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0, -1, 0)
        glVertex3f(0, -height / 2, 0)
        for i in range(slices, -1, -1):
            angle = 2 * np.pi * i / slices
            x = np.cos(angle) * radius
            z = np.sin(angle) * radius
            glVertex3f(x, -height / 2, z)
        glEnd()

    def _draw_propellers(self):
        """绘制螺旋桨"""
        prop_positions = [
            (-self.arm_length, 0.1, 0),
            (self.arm_length, 0.1, 0),
            (0, 0.1, -self.arm_length),
            (0, 0.1, self.arm_length)
        ]

        glColor3f(0.8, 0.2, 0.2)  # 红色螺旋桨

        for x, y, z in prop_positions:
            glPushMatrix()
            glTranslatef(x, y, z)

            # 绘制圆盘
            self._draw_disk(self.propeller_radius)
            glPopMatrix()

    def _draw_disk(self, radius):
        """绘制圆盘"""
        slices = 16

        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        for i in range(slices + 1):
            angle = 2 * np.pi * i / slices
            x = np.cos(angle) * radius
            z = np.sin(angle) * radius
            glVertex3f(x, 0, z)
        glEnd()


# ==================== 测试代码 ====================
def test_simulation_3d():
    """测试3D仿真"""
    print("启动3D无人机仿真测试...")
    print("按ESC键退出")
    print("按G键切换网格显示")
    print("按T键切换轨迹显示")
    print("按A键切换坐标轴显示")
    print("按↑↓←→键旋转视角")
    print("按+/-键缩放视角")
    print("按空格键重置视角")

    # 创建3D查看器
    viewer = Drone3DViewer(width=1024, height=768)

    # 模拟一些轨迹数据用于测试
    trajectory = []
    for i in range(100):
        angle = 2 * np.pi * i / 100
        x = np.cos(angle) * 3
        z = np.sin(angle) * 3
        y = 2 + np.sin(angle * 2) * 0.5
        trajectory.append((x, y, z))

    # 模拟无人机状态
    drone_state = {
        'position': [0.0, 2.0, 0.0],
        'orientation': [0.0, 0.0, 0.0],
        'velocity': [0.0, 0.0, 0.0],
        'battery': 87.5,
        'armed': True,
        'mode': 'HOVER'
    }

    clock = pygame.time.Clock()
    running = True
    frame_count = 0

    while running:
        # 处理事件
        running = viewer.handle_events()

        # 模拟无人机移动（用于测试）
        frame_count += 1
        if frame_count % 300 < 150:
            # 向前移动
            drone_state['position'][2] -= 0.02
            drone_state['orientation'][1] = np.radians(-5)  # 轻微俯仰
        else:
            # 向后移动
            drone_state['position'][2] += 0.02
            drone_state['orientation'][1] = np.radians(5)

        # 限制位置
        drone_state['position'][2] = max(-5, min(5, drone_state['position'][2]))

        # 轻微偏航
        drone_state['orientation'][2] = np.radians(np.sin(frame_count * 0.01) * 10)

        # 更新电池（模拟消耗）
        drone_state['battery'] = 87.5 - (frame_count / 6000) * 100
        if drone_state['battery'] < 0:
            drone_state['battery'] = 0

        # 渲染场景
        viewer.render(drone_state, trajectory)

        # 控制帧率
        clock.tick(60)

    pygame.quit()
    print("3D仿真测试结束")


if __name__ == "__main__":
    # 当直接运行这个文件时，执行测试
    test_simulation_3d()