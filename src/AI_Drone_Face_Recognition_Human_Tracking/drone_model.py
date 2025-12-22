# drone_model.py - 完全干净版本
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import sys
import time


# === 无人机类 ===
class Drone:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 1.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.propeller_angle = 0.0
        self.is_flying = False

        self.move_forward = False
        self.move_backward = False
        self.move_left = False
        self.move_right = False
        self.rotate_left = False
        self.rotate_right = False

        self.speed = 2.5
        self.rotate_speed = 150.0
        self.strafe_speed = 1.8

        print("无人机初始化完成")

    def update(self, delta_time):
        if not self.is_flying:
            return

        self.propeller_angle += delta_time * 500.0
        if self.propeller_angle > 360:
            self.propeller_angle -= 360

        move_amount = self.speed * delta_time
        strafe_amount = self.strafe_speed * delta_time
        rotate_amount = self.rotate_speed * delta_time
        yaw_rad = math.radians(self.yaw)

        if self.move_forward:
            self.x += math.sin(yaw_rad) * move_amount
            self.y += -math.cos(yaw_rad) * move_amount
            self.pitch = -8.0

        if self.move_backward:
            self.x += -math.sin(yaw_rad) * move_amount
            self.y += math.cos(yaw_rad) * move_amount
            self.pitch = 8.0

        if self.move_left:
            self.x += -math.cos(yaw_rad) * strafe_amount
            self.y += -math.sin(yaw_rad) * strafe_amount
            self.roll = -8.0

        if self.move_right:
            self.x += math.cos(yaw_rad) * strafe_amount
            self.y += math.sin(yaw_rad) * strafe_amount
            self.roll = 8.0

        if self.rotate_left:
            self.yaw += rotate_amount
        if self.rotate_right:
            self.yaw -= rotate_amount

        self.yaw %= 360.0
        self.pitch *= 0.9
        self.roll *= 0.9

        if self.z < 0.5:
            self.z = 0.5

    def render(self):
        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glRotatef(self.yaw, 0, 0, 1)
        glRotatef(self.pitch, 1, 0, 0)
        glRotatef(self.roll, 0, 1, 0)

        glColor3f(0.3, 0.3, 0.3)
        self._draw_cube(0.2, 0.15, 0.1)

        arm_positions = [
            (0.25, 0.25, 0),
            (-0.25, 0.25, 0),
            (0.25, -0.25, 0),
            (-0.25, -0.25, 0)
        ]

        for i, (ax, ay, az) in enumerate(arm_positions):
            glPushMatrix()
            glTranslatef(ax, ay, az)
            glRotatef(45, 0, 0, 1)
            glColor3f(0.4, 0.4, 0.4)
            self._draw_cylinder(0.015, 0.18)
            glPopMatrix()

            glPushMatrix()
            glTranslatef(ax, ay, 0.08)
            glColor3f(0.6, 0.6, 0.6)
            self._draw_cylinder(0.025, 0.05)
            glPopMatrix()

            glPushMatrix()
            glTranslatef(ax, ay, 0.13)
            glColor3f(0.9, 0.1, 0.1)
            glRotatef(self.propeller_angle + i * 90, 0, 0, 1)
            self._draw_propeller(0.1)
            glPopMatrix()

        glPushMatrix()
        glTranslatef(0, 0, 0.08)
        if self.is_flying:
            blink = int(time.time() * 5) % 2 == 0
            glColor3f(0.0, 1.0 if blink else 0.5, 0.0)
        else:
            glColor3f(1.0, 0.0, 0.0)
        self._draw_sphere(0.02)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0, 0.2, 0.05)
        glColor3f(1.0, 1.0, 0.0)
        self._draw_sphere(0.018)
        glPopMatrix()

        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 0.0)
        glLineWidth(3.0)

        glBegin(GL_LINES)
        glVertex3f(0, 0.1, 0.08)
        glVertex3f(0, 0.3, 0.08)
        glVertex3f(0, 0.3, 0.08)
        glVertex3f(-0.05, 0.22, 0.08)
        glVertex3f(0, 0.3, 0.08)
        glVertex3f(0.05, 0.22, 0.08)
        glVertex3f(-0.05, 0.22, 0.08)
        glVertex3f(0.05, 0.22, 0.08)
        glEnd()

        glEnable(GL_LIGHTING)
        glPopMatrix()

    def _draw_cube(self, width, height, depth):
        w, h, d = width / 2, height / 2, depth / 2
        vertices = [
            [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
            [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]
        ]
        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 3, 7, 4], [1, 2, 6, 5],
            [0, 1, 5, 4], [2, 3, 7, 6]
        ]

        glBegin(GL_QUADS)
        for face in faces:
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()

    def _draw_cylinder(self, radius, height):
        quadric = gluNewQuadric()
        gluCylinder(quadric, radius, radius, height, 16, 1)

        glPushMatrix()
        glTranslatef(0, 0, height)
        gluDisk(quadric, 0, radius, 16, 1)
        glPopMatrix()

        glPushMatrix()
        glRotatef(180, 1, 0, 0)
        gluDisk(quadric, 0, radius, 16, 1)
        glPopMatrix()

    def _draw_sphere(self, radius):
        quadric = gluNewQuadric()
        gluSphere(quadric, radius, 16, 16)

    def _draw_propeller(self, radius):
        glBegin(GL_TRIANGLES)
        for i in range(4):
            angle = i * math.pi / 2
            glVertex3f(0, 0, 0.01)
            glVertex3f(radius * math.cos(angle), radius * math.sin(angle), 0.01)
            glVertex3f(radius * 0.7 * math.cos(angle + 0.3), radius * 0.7 * math.sin(angle + 0.3), 0.01)
        glEnd()

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 1.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        print("无人机位置已重置")


# === 无人机查看器类 ===
class DroneViewer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height

        try:
            self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        except pygame.error as e:
            print(f"无法创建OpenGL窗口: {e}")
            print("尝试使用标准窗口模式...")
            self.screen = pygame.display.set_mode((width, height))

        pygame.display.set_caption("3D无人机查看器")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)

        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])

        self.drone = Drone()
        self.camera_distance = 8.0
        self.camera_angle_x = 30.0
        self.camera_angle_y = -45.0
        self.mouse_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.running = True
        self.path = []
        self.max_path_length = 200

        print("=" * 50)
        print("3D无人机查看器已启动")
        print("控制说明:")
        print("  空格键: 起飞/降落")
        print("  W键: 前进 (朝黄色箭头方向)")  # W显示为前进
        print("  S键: 后退")  # S显示为后退
        print("  A键: 向左平移")
        print("  D键: 向右平移")
        print("  Q键: 左转")
        print("  E键: 右转")
        print("  R键: 重置位置")
        print("  ESC键: 退出")
        print("  鼠标左键拖拽: 旋转视角")
        print("  鼠标滚轮: 缩放")
        print("=" * 50)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.drone.is_flying = not self.drone.is_flying
                    status = "起飞" if self.drone.is_flying else "降落"
                    print(f"无人机: {status}")
                elif event.key == pygame.K_r:
                    self.drone.reset()
                    self.path = []
                    print("位置和路径已重置")
                elif event.key == pygame.K_s:  # S键 - 物理逻辑向前
                    self.drone.move_forward = True
                    print("S键按下: 后退")  # 只改打印信息：显示W键前进
                elif event.key == pygame.K_w:  # W键 - 物理逻辑向后
                    self.drone.move_backward = True
                    print("W键按下: 前进")  # 只改打印信息：显示S键后退
                elif event.key == pygame.K_a:
                    self.drone.move_left = True
                    print("A键按下: 向左平移")
                elif event.key == pygame.K_d:
                    self.drone.move_right = True
                    print("D键按下: 向右平移")
                elif event.key == pygame.K_q:
                    self.drone.rotate_left = True
                    print("Q键按下: 左转")
                elif event.key == pygame.K_e:
                    self.drone.rotate_right = True
                    print("E键按下: 右转")

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_s:  # S键释放
                    self.drone.move_forward = False
                elif event.key == pygame.K_w:  # W键释放
                    self.drone.move_backward = False
                elif event.key == pygame.K_a:
                    self.drone.move_left = False
                elif event.key == pygame.K_d:
                    self.drone.move_right = False
                elif event.key == pygame.K_q:
                    self.drone.rotate_left = False
                elif event.key == pygame.K_e:
                    self.drone.rotate_right = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_dragging = True
                    self.last_mouse_x, self.last_mouse_y = event.pos
                elif event.button == 4:
                    self.camera_distance = max(3.0, self.camera_distance - 0.5)
                elif event.button == 5:
                    self.camera_distance = min(20.0, self.camera_distance + 0.5)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    x, y = event.pos
                    dx = x - self.last_mouse_x
                    dy = y - self.last_mouse_y

                    self.camera_angle_y += dx * 0.5
                    self.camera_angle_x -= dy * 0.5
                    self.camera_angle_x = max(-85.0, min(85.0, self.camera_angle_x))

                    self.last_mouse_x, self.last_mouse_y = x, y

    def update(self, delta_time):
        self.drone.update(delta_time)

        if self.drone.is_flying:
            self.path.append((self.drone.x, self.drone.y, self.drone.z))
            if len(self.path) > self.max_path_length:
                self.path.pop(0)

    def _draw_grid(self):
        glDisable(GL_LIGHTING)
        glColor3f(0.4, 0.4, 0.4)

        glBegin(GL_LINES)
        for i in range(-15, 16):
            glVertex3f(i, -15, 0)
            glVertex3f(i, 15, 0)
            glVertex3f(-15, i, 0)
            glVertex3f(15, i, 0)
        glEnd()

        glEnable(GL_LIGHTING)

    def _draw_path(self):
        if len(self.path) > 1:
            glDisable(GL_LIGHTING)
            glColor3f(1.0, 0.5, 0.0)
            glLineWidth(2.0)

            glBegin(GL_LINE_STRIP)
            for point in self.path:
                glVertex3f(point[0], point[1], point[2] + 0.05)
            glEnd()

            glEnable(GL_LIGHTING)

    def _draw_axes(self):
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)

        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(3, 0, 0)

        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 3, 0)

        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 3)
        glEnd()

        glEnable(GL_LIGHTING)

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.08, 0.12, 0.18, 1.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        cam_x = self.camera_distance * math.sin(math.radians(self.camera_angle_y)) * math.cos(
            math.radians(self.camera_angle_x))
        cam_y = self.camera_distance * math.cos(math.radians(self.camera_angle_y)) * math.cos(
            math.radians(self.camera_angle_x))
        cam_z = self.camera_distance * math.sin(math.radians(self.camera_angle_x))

        gluLookAt(cam_x, cam_y, cam_z, 0, 0, 0, 0, 0, 1)

        self._draw_grid()
        self._draw_axes()
        self._draw_path()
        self.drone.render()

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        last_time = pygame.time.get_ticks()

        print("\n操作流程:")
        print("1. 按空格键起飞")
        print("2. 按W键前进 (朝黄色箭头方向)")  # W显示为前进
        print("3. 按S键后退")  # S显示为后退
        print("4. 按A/D键左右平移")
        print("5. 按Q/E键旋转")
        print("6. 鼠标拖拽旋转视角")
        print("7. 滚轮缩放\n")

        try:
            while self.running:
                current_time = pygame.time.get_ticks()
                delta_time = (current_time - last_time) / 1000.0
                last_time = current_time

                self.handle_events()
                self.update(delta_time)
                self.render()

                clock.tick(60)
        except KeyboardInterrupt:
            print("\n用户中断程序")
        except Exception as e:
            print(f"运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pygame.quit()
            print("程序已退出")


# === 主程序 ===
def main():
    print("启动3D无人机查看器...")

    viewer = DroneViewer()
    viewer.run()


if __name__ == "__main__":
    main()