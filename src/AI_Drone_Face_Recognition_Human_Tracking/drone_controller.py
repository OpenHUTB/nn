'''无人机控制器模块'''
import numpy as np
from enum import Enum
import pygame
import sys


# ===================== 无人机状态枚举 =====================
class DroneState(Enum):
    LANDED = "Landed"
    FLYING = "Flying"
    EMERGENCY = "Emergency"


# ===================== 虚拟无人机控制器 =====================
class DroneController:
    """无人机控制逻辑"""

    def __init__(self):
        # 物理状态
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0

        # 控制参数
        self.speed = 0.5
        self.rotation_speed = 10.0

        # 限制参数
        self.max_height = 20.0
        self.min_height = 0.0
        self.boundary_xy = 20.0

        # 系统状态
        self.state = DroneState.LANDED
        self.battery = 100.0

        # 飞行记录
        self.flight_path = []
        self.total_distance = 0.0

        # 回调函数
        self.on_position_changed = None
        self.on_state_changed = None

        # 初始记录
        self._record_position()

    def _record_position(self):
        """记录位置"""
        self.flight_path.append({
            'position': self.position.copy(),
            'yaw': self.yaw,
            'battery': self.battery
        })

    def _check_boundaries(self, new_position):
        """检查边界"""
        new_position[0] = np.clip(new_position[0], -self.boundary_xy, self.boundary_xy)
        new_position[1] = np.clip(new_position[1], -self.boundary_xy, self.boundary_xy)
        new_position[2] = np.clip(new_position[2], self.min_height, self.max_height)
        return new_position

    def _calculate_movement(self, direction):
        """计算移动向量"""
        rad_yaw = np.radians(self.yaw)
        cos_yaw = np.cos(rad_yaw)
        sin_yaw = np.sin(rad_yaw)

        move = np.zeros(3)

        if direction == "forward":
            move[0] = self.speed * cos_yaw
            move[1] = self.speed * sin_yaw
        elif direction == "back":
            move[0] = -self.speed * cos_yaw
            move[1] = -self.speed * sin_yaw
        elif direction == "left":
            move[0] = self.speed * sin_yaw
            move[1] = -self.speed * cos_yaw
        elif direction == "right":
            move[0] = -self.speed * sin_yaw
            move[1] = self.speed * cos_yaw
        elif direction == "up":
            move[2] = self.speed
        elif direction == "down":
            move[2] = -self.speed

        return move

    def _consume_battery(self, amount=0.1):
        """消耗电量"""
        self.battery -= amount
        self.battery = max(self.battery, 0.0)

        if self.battery <= 0 and self.state == DroneState.FLYING:
            print("🔴 电量耗尽！强制降落")
            self.emergency_land()

    def takeoff(self, height=1.0):
        """起飞"""
        if self.state == DroneState.LANDED:
            if self.battery > 20:
                self.state = DroneState.FLYING
                self.position[2] = height
                self._consume_battery(0.5)
                self._record_position()
                print(f"✅ 起飞 | 高度: {self.position[2]:.1f}m")

                # 通知状态变化
                if self.on_state_changed:
                    self.on_state_changed(self.get_status())
                if self.on_position_changed:
                    self.on_position_changed(self.position, self.yaw)

                return True
            else:
                print("⚠️ 电量不足20%，禁止起飞")
                return False
        else:
            print("⚠️ 已处于飞行状态")
            return False

    def land(self):
        """降落"""
        if self.state == DroneState.FLYING:
            self.state = DroneState.LANDED
            self.position[2] = 0.0
            self.velocity = np.zeros(3)
            self._consume_battery(0.2)
            self._record_position()
            print(f"✅ 降落 | 位置: {self.position[:2]}")

            # 通知状态变化
            if self.on_state_changed:
                self.on_state_changed(self.get_status())
            if self.on_position_changed:
                self.on_position_changed(self.position, self.yaw)

            return True
        else:
            print("⚠️ 已处于落地状态")
            return False

    def emergency_land(self):
        """紧急降落"""
        print("🆘 执行紧急降落")
        self.state = DroneState.EMERGENCY
        self.velocity = np.zeros(3)
        self.position[2] = 0.0
        self.state = DroneState.LANDED
        self._record_position()

        if self.on_state_changed:
            self.on_state_changed(self.get_status())
        if self.on_position_changed:
            self.on_position_changed(self.position, self.yaw)

    def move(self, direction):
        """移动"""
        if self.state != DroneState.FLYING:
            print("⚠️ 未起飞，无法移动")
            return False

        if self.battery <= 0:
            print("⚠️ 电量耗尽")
            return False

        move_vector = self._calculate_movement(direction)

        if np.all(move_vector == 0):
            return False

        # 计算新位置
        new_position = self.position + move_vector
        new_position = self._check_boundaries(new_position)

        # 计算距离
        distance = np.linalg.norm(new_position - self.position)
        self.total_distance += distance

        # 更新状态
        self.velocity = move_vector
        self.position = new_position
        self._consume_battery()
        self._record_position()

        print(f"🔹 {direction} | 位置: {self.position.round(2)} | 电量: {self.battery:.1f}%")

        # 通知位置变化
        if self.on_position_changed:
            self.on_position_changed(self.position, self.yaw)

        return True

    def rotate(self, direction):
        """旋转"""
        if self.state != DroneState.FLYING:
            print("⚠️ 未起飞，无法旋转")
            return False

        if direction == "left":
            self.yaw += self.rotation_speed
        elif direction == "right":
            self.yaw -= self.rotation_speed
        else:
            return False

        self.yaw %= 360
        self._consume_battery(0.05)
        self._record_position()

        print(f"🔄 旋转 {direction} | 偏航角: {self.yaw:.0f}°")

        # 通知位置变化
        if self.on_position_changed:
            self.on_position_changed(self.position, self.yaw)

        return True

    def set_speed(self, speed):
        """设置速度"""
        if 0.1 <= speed <= 2.0:
            self.speed = speed
            print(f"⚡ 速度设置为: {self.speed:.1f}m/s")
            return True
        else:
            print("⚠️ 速度必须在0.1到2.0m/s之间")
            return False

    def get_status(self):
        """获取状态"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'yaw': self.yaw,
            'state': self.state.value,
            'battery': self.battery,
            'total_distance': self.total_distance
        }


# ===================== 键盘控制器 =====================
class KeyboardController:
    """键盘输入控制器"""

    def __init__(self, drone_controller):
        """
        初始化键盘控制器
        :param drone_controller: DroneController实例
        """
        self.drone = drone_controller

        # 初始化Pygame
        pygame.init()
        pygame.mixer.quit()  # 禁用音频
        self.screen = pygame.display.set_mode((450, 350))
        pygame.display.set_caption("无人机控制面板")
        self.clock = pygame.time.Clock()

        # 运行标志
        self.running = True

    def handle_events(self):
        """处理所有事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

            if event.type == pygame.KEYDOWN:
                # 退出
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False

                # 起飞/降落
                if event.key == pygame.K_t:
                    self.drone.takeoff()
                elif event.key == pygame.K_l:
                    self.drone.land()

                # 移动
                elif event.key == pygame.K_w:
                    self.drone.move("forward")
                elif event.key == pygame.K_s:
                    self.drone.move("back")
                elif event.key == pygame.K_a:
                    self.drone.move("left")
                elif event.key == pygame.K_d:
                    self.drone.move("right")
                elif event.key == pygame.K_UP:
                    self.drone.move("up")
                elif event.key == pygame.K_DOWN:
                    self.drone.move("down")

                # 旋转
                elif event.key == pygame.K_q:
                    self.drone.rotate("left")
                elif event.key == pygame.K_e:
                    self.drone.rotate("right")

                # 调速
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.drone.set_speed(self.drone.speed + 0.1)
                elif event.key == pygame.K_MINUS:
                    self.drone.set_speed(self.drone.speed - 0.1)

                # 重置（新增）
                elif event.key == pygame.K_r:
                    # 重新创建控制器
                    self.drone = DroneController()
                    print("🔄 无人机已重置")

        return True

    def update_display(self):
        """更新控制面板显示"""
        self.screen.fill((245, 245, 245))

        # 绘制标题栏
        title_rect = pygame.Rect(0, 0, 450, 40)
        pygame.draw.rect(self.screen, (70, 130, 180), title_rect)

        # 标题文字
        try:
            font_large = pygame.font.SysFont('microsoftyahei', 24, bold=True)
        except:
            font_large = pygame.font.SysFont(None, 24, bold=True)
        title_text = font_large.render("无人机控制面板", True, (255, 255, 255))
        self.screen.blit(title_text, (10, 8))

        # 状态区域
        try:
            status_font = pygame.font.SysFont('microsoftyahei', 18)
            control_font = pygame.font.SysFont('microsoftyahei', 16)
            small_font = pygame.font.SysFont('microsoftyahei', 14)
        except:
            status_font = pygame.font.SysFont(None, 18)
            control_font = pygame.font.SysFont(None, 16)
            small_font = pygame.font.SysFont(None, 14)

        status = self.drone.get_status()

        # 状态信息
        status_lines = [
            f"状态: {status['state']}",
            f"位置: X:{status['position'][0]:.1f} Y:{status['position'][1]:.1f} Z:{status['position'][2]:.1f}",
            f"偏航角: {status['yaw']:.0f}°",
            f"电量: {status['battery']:.1f}%",
            f"速度: {self.drone.speed:.1f} m/s",
            f"飞行距离: {status['total_distance']:.1f} m"
        ]

        for i, line in enumerate(status_lines):
            color = (0, 100, 0) if "Flying" in line else (50, 50, 50)
            text = status_font.render(line, True, color)
            self.screen.blit(text, (15, 50 + i * 28))

        # 分隔线
        pygame.draw.line(self.screen, (200, 200, 200), (10, 200), (440, 200), 2)

        # 控制指令
        control_lines = [
            "=== 飞行控制 ===",
            "T: 起飞  |  L: 降落  |  R: 重置",
            "W: 前进  |  S: 后退  |  A: 左移  |  D: 右移",
            "↑: 上升  |  ↓: 下降  |  Q: 左转  |  E: 右转",
            "+: 加速  |  -: 减速  |  ESC: 退出",
            "",
            "⚠️ 提示: 点击此窗口获取键盘焦点"
        ]

        for i, line in enumerate(control_lines):
            if "===" in line:
                text = control_font.render(line, True, (70, 130, 180))
            else:
                text = small_font.render(line, True, (80, 80, 80))
            self.screen.blit(text, (15, 210 + i * 22))

        # 绘制边框
        pygame.draw.rect(self.screen, (180, 180, 180), (0, 0, 450, 350), 3)

        pygame.display.flip()

    def run_loop(self, fps=60):
        """运行控制循环"""
        while self.running:
            if not self.handle_events():
                break
            self.update_display()
            self.clock.tick(fps)

    def cleanup(self):
        """清理资源"""
        pygame.quit()
        print("🎮 键盘控制器已关闭")


# ===================== 测试代码 =====================
if __name__ == "__main__":
    print("测试无人机控制器...")

    controller = DroneController()
    keyboard = KeyboardController(controller)

    print("✅ 控制器测试完成")
    print("提示：运行main.py启动完整模拟器")