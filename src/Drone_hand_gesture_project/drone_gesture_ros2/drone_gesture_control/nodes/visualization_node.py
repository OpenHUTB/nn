#!/usr/bin/env python3
"""
可视化节点 - 包含3D无人机显示
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import threading

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        
        # 订阅器
        self.gesture_sub = self.create_subscription(
            String, 'gesture_detected', self.gesture_callback, 10)
        self.command_sub = self.create_subscription(
            String, 'gesture_command', self.command_callback, 10)
        
        # 状态
        self.current_gesture = "none"
        self.current_command = "none"
        self.gesture_history = []
        self.command_history = []
        self.start_time = time.time()
        
        # 无人机模拟状态
        self.drone_position = [0.0, 2.0, 0.0]  # [x, y, z]
        self.drone_orientation = [0.0, 0.0, 0.0]  # [roll, pitch, yaw]
        self.drone_battery = 100.0
        self.drone_armed = True
        self.drone_mode = "HOVER"
        
        # 启动3D可视化线程
        self.viz_thread = threading.Thread(target=self.run_3d_visualization, daemon=True)
        self.viz_thread.start()
        
        # 状态定时器
        self.status_timer = self.create_timer(2.0, self.update_drone_state)
        
        self.get_logger().info('👁️ 3D可视化节点已启动')
        self.get_logger().info('🚀 3D无人机显示已启用')
    
    def gesture_callback(self, msg):
        """手势回调"""
        data = msg.data.split(':')
        if len(data) >= 1:
            self.current_gesture = data[0]
            confidence = data[1] if len(data) > 1 else "0.0"
            
            # 记录历史
            current_time = time.time() - self.start_time
            self.gesture_history.append((current_time, self.current_gesture, confidence))
            
            # 只保留最近10个
            if len(self.gesture_history) > 10:
                self.gesture_history.pop(0)
            
            self.get_logger().info(f'👋 检测到手势: {self.current_gesture} (置信度: {confidence})')
            
            # 根据手势更新无人机位置（模拟）
            self.update_drone_by_gesture(self.current_gesture)
    
    def command_callback(self, msg):
        """命令回调"""
        self.current_command = msg.data
        
        # 记录历史
        current_time = time.time() - self.start_time
        self.command_history.append((current_time, self.current_command))
        
        # 只保留最近10个
        if len(self.command_history) > 10:
            self.command_history.pop(0)
        
        self.get_logger().info(f'🎯 执行命令: {self.current_command}')
        
        # 根据命令更新无人机状态
        self.update_drone_by_command(self.current_command)
    
    def update_drone_by_gesture(self, gesture):
        """根据手势更新无人机位置（模拟）"""
        import numpy as np
        
        # 模拟手势对无人机的影响
        gesture_effects = {
            "pointing_up": [0, 0.1, 0],      # 向上
            "pointing_down": [0, -0.1, 0],   # 向下
            "victory": [0, 0, 0.1],          # 向前
            "thumb_up": [0, 0, -0.1],        # 向后
            "open_palm": [0.1, 0, 0],        # 向右
            "closed_fist": [-0.1, 0, 0],     # 向左
        }
        
        if gesture in gesture_effects:
            effect = gesture_effects[gesture]
            self.drone_position[0] += effect[0]
            self.drone_position[1] += effect[1]
            self.drone_position[2] += effect[2]
            
            # 限制高度
            self.drone_position[1] = max(0.0, min(10.0, self.drone_position[1]))
    
    def update_drone_by_command(self, command):
        """根据命令更新无人机状态"""
        if command == "takeoff":
            self.drone_armed = True
            self.drone_mode = "TAKEOFF"
            self.drone_position[1] = 3.0  # 起飞到3米高度
        elif command == "land":
            self.drone_mode = "LAND"
            self.drone_position[1] = 0.0
        elif command == "hover":
            self.drone_mode = "HOVER"
        elif command == "stop":
            self.drone_mode = "STOP"
    
    def update_drone_state(self):
        """定期更新无人机状态"""
        # 模拟电池消耗
        self.drone_battery = max(0, self.drone_battery - 0.01)
        if self.drone_battery < 20.0:
            self.get_logger().warning(f'🔋 电池电量低: {self.drone_battery:.1f}%')
        
        # 模拟轻微晃动
        import numpy as np
        self.drone_orientation[2] = np.sin(time.time() * 0.5) * 0.1  # 偏航
    
    def run_3d_visualization(self):
        """运行3D可视化"""
        try:
            # 尝试导入OpenGL和Pygame
            import OpenGL.GL as gl
            import OpenGL.GLU as glu
            import pygame
            import numpy as np
            
            # 初始化Pygame
            pygame.init()
            width, height = 1024, 768
            screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL)
            pygame.display.set_caption("无人机3D仿真系统")
            
            # OpenGL设置
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_LIGHT0)
            
            # 光源
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [5.0, 5.0, 5.0, 1.0])
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
            
            # 投影
            glu.gluPerspective(45, (width / height), 0.1, 100.0)
            
            # 相机初始位置
            camera_distance = 10.0
            camera_angle_x = 30.0
            camera_angle_y = -30.0
            
            clock = pygame.time.Clock()
            
            self.get_logger().info('🎮 3D窗口已打开')
            self.get_logger().info('💡 控制提示:')
            self.get_logger().info('   ESC - 退出')
            self.get_logger().info('   ↑↓←→ - 旋转视角')
            self.get_logger().info('   +/- - 缩放视角')
            self.get_logger().info('   空格 - 重置视角')
            
            running = True
            while running and rclpy.ok():
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_UP:
                            camera_angle_y = min(89, camera_angle_y + 5)
                        elif event.key == pygame.K_DOWN:
                            camera_angle_y = max(-89, camera_angle_y - 5)
                        elif event.key == pygame.K_LEFT:
                            camera_angle_x -= 5
                        elif event.key == pygame.K_RIGHT:
                            camera_angle_x += 5
                        elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                            camera_distance = max(5, camera_distance - 1)
                        elif event.key == pygame.K_MINUS:
                            camera_distance = min(50, camera_distance + 1)
                        elif event.key == pygame.K_SPACE:
                            camera_distance = 10.0
                            camera_angle_x = 30.0
                            camera_angle_y = -30.0
                
                # 清除缓冲区
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                gl.glClearColor(0.1, 0.1, 0.15, 1.0)
                
                # 设置模型视图矩阵
                gl.glLoadIdentity()
                
                # 计算相机位置
                cam_x = camera_distance * np.sin(np.radians(camera_angle_y)) * np.cos(np.radians(camera_angle_x))
                cam_y = camera_distance * np.cos(np.radians(camera_angle_y))
                cam_z = camera_distance * np.sin(np.radians(camera_angle_y)) * np.sin(np.radians(camera_angle_x))
                
                glu.gluLookAt(cam_x, cam_y, cam_z,  # 相机位置
                              0, 0, 0,              # 观察点
                              0, 1, 0)              # 上方向
                
                # 绘制地面网格
                self.draw_grid()
                
                # 绘制坐标轴
                self.draw_axes()
                
                # 绘制无人机
                self.draw_drone()
                
                # 绘制状态信息
                self.draw_status_overlay(width, height)
                
                # 交换缓冲区
                pygame.display.flip()
                clock.tick(60)
            
            pygame.quit()
            self.get_logger().info('3D窗口已关闭')
            
        except ImportError as e:
            self.get_logger().warning(f'无法加载3D库: {e}')
            self.get_logger().info('切换到2D终端显示模式')
            self.run_terminal_display()
        except Exception as e:
            self.get_logger().error(f'3D可视化错误: {e}')
            self.run_terminal_display()
    
    def draw_grid(self):
        """绘制地面网格"""
        import OpenGL.GL as gl
        
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(0.3, 0.3, 0.3)
        
        for i in range(-20, 21):
            # X方向线
            gl.glVertex3f(i, 0, -20)
            gl.glVertex3f(i, 0, 20)
            # Z方向线
            gl.glVertex3f(-20, 0, i)
            gl.glVertex3f(20, 0, i)
        
        gl.glEnd()
    
    def draw_axes(self):
        """绘制坐标轴"""
        import OpenGL.GL as gl
        
        gl.glLineWidth(2.0)
        gl.glBegin(gl.GL_LINES)
        
        # X轴 (红色)
        gl.glColor3f(1.0, 0.0, 0.0)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(3, 0, 0)
        
        # Y轴 (绿色)
        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, 3, 0)
        
        # Z轴 (蓝色)
        gl.glColor3f(0.0, 0.0, 1.0)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, 0, 3)
        
        gl.glEnd()
        gl.glLineWidth(1.0)
    
    def draw_drone(self):
        """绘制无人机"""
        import OpenGL.GL as gl
        import numpy as np
        
        # 无人机位置和姿态
        x, y, z = self.drone_position
        roll, pitch, yaw = self.drone_orientation
        
        gl.glPushMatrix()
        gl.glTranslatef(x, y, z)
        gl.glRotatef(np.degrees(yaw), 0, 1, 0)
        gl.glRotatef(np.degrees(pitch), 1, 0, 0)
        gl.glRotatef(np.degrees(roll), 0, 0, 1)
        
        # 根据状态设置颜色
        if self.drone_armed:
            if self.drone_mode == "TAKEOFF":
                color = (0.0, 1.0, 0.0)  # 绿色：起飞
            elif self.drone_mode == "LAND":
                color = (1.0, 0.5, 0.0)  # 橙色：降落
            elif self.drone_mode == "HOVER":
                color = (0.0, 0.8, 1.0)  # 青色：悬停
            else:
                color = (0.0, 0.6, 0.0)  # 深绿：飞行
        else:
            color = (0.5, 0.5, 0.5)  # 灰色：未解锁
        
        gl.glColor3f(*color)
        
        # 绘制无人机机身（立方体）
        size = 0.3
        self.draw_cube(size)
        
        # 绘制机臂
        arm_length = 0.8
        arm_positions = [
            (-arm_length/2, 0, 0),
            (arm_length/2, 0, 0),
            (0, 0, -arm_length/2),
            (0, 0, arm_length/2)
        ]
        
        gl.glColor3f(0.3, 0.3, 0.3)
        for arm_x, arm_y, arm_z in arm_positions:
            gl.glPushMatrix()
            gl.glTranslatef(arm_x, arm_y, arm_z)
            self.draw_cylinder(0.03, 0.05)
            gl.glPopMatrix()
        
        gl.glPopMatrix()
    
    def draw_cube(self, size):
        """绘制立方体"""
        import OpenGL.GL as gl
        
        s = size / 2
        vertices = [
            (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
            (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)
        ]
        
        faces = [
            (0,1,2,3), (1,5,6,2), (5,4,7,6),
            (4,0,3,7), (3,2,6,7), (1,0,4,5)
        ]
        
        gl.glBegin(gl.GL_QUADS)
        for face in faces:
            for vertex in face:
                gl.glVertex3fv(vertices[vertex])
        gl.glEnd()
    
    def draw_cylinder(self, radius, height):
        """绘制圆柱体"""
        import OpenGL.GL as gl
        import numpy as np
        
        slices = 8
        gl.glBegin(gl.GL_QUAD_STRIP)
        for i in range(slices + 1):
            angle = 2 * np.pi * i / slices
            x = np.cos(angle) * radius
            z = np.sin(angle) * radius
            gl.glVertex3f(x, -height/2, z)
            gl.glVertex3f(x, height/2, z)
        gl.glEnd()
    
    def draw_status_overlay(self, width, height):
        """绘制状态信息覆盖层"""
        import OpenGL.GL as gl
        import pygame
        
        # 切换到2D模式
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(0, width, 0, height, -1, 1)
        
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        
        # 禁用3D特性
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_LIGHTING)
        
        # 绘制半透明背景
        gl.glColor4f(0.0, 0.0, 0.0, 0.5)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(10, height - 200)
        gl.glVertex2f(400, height - 200)
        gl.glVertex2f(400, height - 10)
        gl.glVertex2f(10, height - 10)
        gl.glEnd()
        
        # 创建字体
        font = pygame.font.SysFont(None, 24)
        
        # 状态信息
        status_lines = [
            f"无人机状态: {self.drone_mode}",
            f"位置: X={self.drone_position[0]:.2f} Y={self.drone_position[1]:.2f} Z={self.drone_position[2]:.2f}",
            f"电池: {self.drone_battery:.1f}%",
            f"解锁: {'是' if self.drone_armed else '否'}",
            f"当前手势: {self.current_gesture}",
            f"当前命令: {self.current_command}",
            f"运行时间: {time.time() - self.start_time:.0f}秒"
        ]
        
        # 渲染文本
        y_offset = height - 40
        for line in status_lines:
            text_surface = font.render(line, True, (255, 255, 255))
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            
            gl.glRasterPos2d(20, y_offset)
            gl.glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                           gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, text_data)
            y_offset -= 30
        
        # 恢复3D设置
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)
        
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()
    
    def run_terminal_display(self):
        """终端显示模式（备用）"""
        import time
        
        self.get_logger().info('切换到终端显示模式')
        
        while rclpy.ok():
            # 显示状态信息
            status = (
                f"\n{'='*60}\n"
                f"无人机3D仿真系统 (终端模式)\n"
                f"{'='*60}\n"
                f"位置: X={self.drone_position[0]:.2f} Y={self.drone_position[1]:.2f} Z={self.drone_position[2]:.2f}\n"
                f"电池: {self.drone_battery:.1f}%\n"
                f"模式: {self.drone_mode}\n"
                f"解锁: {'是' if self.drone_armed else '否'}\n"
                f"当前手势: {self.current_gesture}\n"
                f"当前命令: {self.current_command}\n"
                f"{'='*60}\n"
            )
            
            print(status)
            time.sleep(2)

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 3D可视化节点正在关闭...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
