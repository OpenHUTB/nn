"""无人机控制模块"""
import pyvista as pv
import pygame
import numpy as np
import sys
from enum import Enum
import warnings

# 忽略PyVista版本兼容警告（提升运行体验）
warnings.filterwarnings("ignore")


# ===================== 无人机状态枚举 =====================
class DroneState(Enum):
    LANDED = "Landed"
    FLYING = "Flying"


# ===================== 虚拟无人机类 =====================
class VirtualDrone:
    def __init__(self):
        # 无人机物理参数
        self.position = np.array([0.0, 0.0, 0.0])  # 三维坐标 (x, y, z)
        self.velocity = np.array([0.0, 0.0, 0.0])  # 速度 (x, y, z)
        self.yaw = 0.0  # 偏航角（绕z轴旋转，°）
        self.speed = 0.5  # 飞行速度（m/s）
        self.state = DroneState.LANDED  # 初始状态：落地
        self.battery = 100.0  # 电量（%）
        self.max_height = 10.0  # 最大飞行高度（m）
        self.min_height = 0.0  # 最小高度（m）

    def takeoff(self):
        """起飞（仅落地状态可执行）"""
        if self.state == DroneState.LANDED and self.battery > 20:
            self.state = DroneState.FLYING
            self.position[2] = 1.0  # 起飞到1m高度
            self.battery -= 0.5  # 起飞消耗电量
            print(f"✅ 虚拟无人机起飞 | 当前高度: {self.position[2]:.1f}m")
        elif self.battery <= 20:
            print("⚠️ 电量不足20%，禁止起飞")
        else:
            print("⚠️ 无人机已处于飞行状态")

    def land(self):
        """降落（仅飞行状态可执行）"""
        if self.state == DroneState.FLYING:
            self.state = DroneState.LANDED
            self.position[2] = 0.0  # 落地
            self.velocity = np.zeros(3)  # 速度清零
            self.battery -= 0.2  # 降落消耗电量
            print(f"✅ 虚拟无人机降落 | 最终位置: {self.position[:2]}")
        else:
            print("⚠️ 无人机已处于落地状态")

    def move(self, direction):
        """
        控制无人机移动
        :param direction: 移动方向（forward/back/left/right/up/down）
        """
        if self.state != DroneState.FLYING:
            print("⚠️ 无人机未起飞，无法移动")
            return
        if self.battery <= 0:
            print("⚠️ 电量耗尽，无法移动")
            return

        # 基于偏航角计算实际移动方向（考虑朝向）
        rad_yaw = np.radians(self.yaw)
        cos_yaw = np.cos(rad_yaw)
        sin_yaw = np.sin(rad_yaw)

        # 重置速度
        self.velocity = np.zeros(3)

        # 方向映射（x: 前后, y: 左右, z: 上下）
        if direction == "forward":
            self.velocity[0] = self.speed * cos_yaw
            self.velocity[1] = self.speed * sin_yaw
        elif direction == "back":
            self.velocity[0] = -self.speed * cos_yaw
            self.velocity[1] = -self.speed * sin_yaw
        elif direction == "left":
            self.velocity[0] = self.speed * sin_yaw
            self.velocity[1] = -self.speed * cos_yaw
        elif direction == "right":
            self.velocity[0] = -self.speed * sin_yaw
            self.velocity[1] = self.speed * cos_yaw
        elif direction == "up":
            self.velocity[2] = self.speed
        elif direction == "down":
            self.velocity[2] = -self.speed

        # 更新位置
        new_pos = self.position + self.velocity
        # 高度限制
        new_pos[2] = np.clip(new_pos[2], self.min_height, self.max_height)
        self.position = new_pos

        # 消耗电量
        self.battery -= 0.1
        self.battery = max(self.battery, 0.0)

        print(f"🔹 移动 {direction} | 位置: {self.position.round(1)} | 电量: {self.battery:.1f}%")

    def rotate(self, direction):
        """
        旋转无人机（偏航角）
        :param direction: left/right
        """
        if self.state != DroneState.FLYING:
            print("⚠️ 无人机未起飞，无法旋转")
            return
        if direction == "left":
            self.yaw += 10.0  # 左转10°
        elif direction == "right":
            self.yaw -= 10.0  # 右转10°
        self.yaw %= 360  # 限制在0-360°
        print(f"🔄 旋转 {direction} | 偏航角: {self.yaw:.0f}°")


# ===================== 3D可视化+交互控制 =====================
class DroneSimulator:
    def __init__(self):
        # 初始化虚拟无人机
        self.drone = VirtualDrone()
        # 初始化Pygame（键盘交互）
        pygame.init()
        # PyCharm适配：禁用Pygame音频（避免无音频设备报错）
        pygame.mixer.quit()
        self.screen = pygame.display.set_mode((400, 200))
        pygame.display.set_caption("虚拟无人机控制面板")
        self.clock = pygame.time.Clock()

        # 初始化PyVista（3D可视化）
        self.plotter = pv.Plotter(window_size=(800, 600))
        self.plotter.set_background("lightgray")
        # 创建无人机3D模型
        self._create_drone_model()
        # 添加地面网格
        self._add_ground_plane()

    def _create_drone_model(self):
        """创建简化的无人机3D模型（适配PyCharm+全版本PyVista）"""
        # 机身（立方体）
        body = pv.Cube(center=(0, 0, 0), x_length=0.5, y_length=0.5, z_length=0.2)
        # 螺旋桨（4个圆柱体）
        prop1 = pv.Cylinder(center=(0.3, 0.3, 0.1), direction=(1, 0, 0), radius=0.2, height=0.05)
        prop2 = pv.Cylinder(center=(-0.3, 0.3, 0.1), direction=(-1, 0, 0), radius=0.2, height=0.05)
        prop3 = pv.Cylinder(center=(0.3, -0.3, 0.1), direction=(0, 1, 0), radius=0.2, height=0.05)
        prop4 = pv.Cylinder(center=(-0.3, -0.3, 0.1), direction=(0, -1, 0), radius=0.2, height=0.05)
        props = pv.MultiBlock([prop1, prop2, prop3, prop4])

        # 组合无人机模型：适配所有PyVista版本的颜色设置
        self.drone_actor = self.plotter.add_mesh(body, color="darkblue")  # 机身颜色
        self.props_actor = self.plotter.add_mesh(props, color="gray")  # 螺旋桨颜色

        # 初始化状态标签（保存actor引用，用于后续删除）
        self.label_actor = self.plotter.add_text(
            f"Position: (0.0, 0.0, 0.0) | State: Landed | Battery: 100%",
            position="upper_left",
            font_size=12,
            color="black"
        )

    def _add_ground_plane(self):
        """添加地面网格"""
        ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=50, j_size=50)
        ground.rotate_z(45)  # 旋转45°，网格更明显
        self.plotter.add_mesh(ground, color="lightgreen", opacity=0.5)
        # 添加坐标轴（PyCharm中更清晰）
        self.plotter.add_axes(line_width=2, labels_off=False)

    def _update_3d_view(self):
        """更新3D视图（完全适配PyCharm+旧版PyVista）"""
        try:
            # 更新无人机位置
            self.drone_actor.SetPosition(self.drone.position)
            self.props_actor.SetPosition(self.drone.position)

            # 更新无人机旋转（偏航角）
            self.drone_actor.RotateZ(self.drone.yaw)
            self.props_actor.RotateZ(self.drone.yaw)

            # 更新状态标签：先删旧标签，再加新标签（兼容所有版本）
            self.plotter.remove_actor(self.label_actor)
            new_label_text = (
                f"Position: {self.drone.position.round(1)} | "
                f"State: {self.drone.state.value} | "
                f"Battery: {self.drone.battery:.1f}%"
            )
            self.label_actor = self.plotter.add_text(
                new_label_text,
                position="upper_left",
                font_size=12,
                color="black"
            )

            # 强制刷新视图
            self.plotter.render()
        except Exception as e:
            print(f"视图更新小警告（不影响使用）：{str(e)}")

    def _handle_keyboard(self):
        """处理键盘输入（PyCharm焦点适配）"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.cleanup()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                # 基础控制
                if event.key == pygame.K_t:
                    self.drone.takeoff()
                elif event.key == pygame.K_l:
                    self.drone.land()
                # 移动控制
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
                # 旋转控制
                elif event.key == pygame.K_q:
                    self.drone.rotate("left")
                elif event.key == pygame.K_e:
                    self.drone.rotate("right")
                # 退出（PyCharm中优雅退出）
                elif event.key == pygame.K_ESCAPE:
                    self.cleanup()
                    sys.exit()

    def cleanup(self):
        """PyCharm优雅退出（释放资源）"""
        pygame.quit()
        self.plotter.close()
        print("\n👋 模拟器已优雅退出")

    def run(self):
        """运行模拟器（PyCharm专用优化）"""
        print("=" * 60)
        print("🎮 PyCharm 2025.2.3 虚拟无人机模拟器")
        print("=" * 60)
        print("操作说明：")
        print("  T → 起飞 | L → 降落 | ESC → 退出")
        print("  W/S/A/D → 前/后/左/右 | ↑/↓ → 上升/下降")
        print("  Q/E → 左转/右转（偏航角）")
        print("⚠️  注意：先点击Pygame窗口获取键盘焦点")
        print("=" * 60)

        # PyVista窗口显示（适配PyCharm的交互模式）
        self.plotter.show(interactive_update=True, auto_close=False)

        # 主循环（PyCharm帧率优化）
        while True:
            self._handle_keyboard()
            self._update_3d_view()
            self.clock.tick(30)  # 稳定30FPS，避免PyCharm卡顿


# ===================== PyCharm 一键运行入口 =====================
if __name__ == "__main__":
    # 第一步：安装依赖（复制到PyCharm终端执行）
    # pip install pyvista pygame numpy -i https://pypi.tuna.tsinghua.edu.cn/simple/

    # 第二步：运行程序
    try:
        simulator = DroneSimulator()
        simulator.run()
    except Exception as e:
        print(f"\n❌ 程序运行异常：{str(e)}")
        print("\n💡 解决方案：")
        print("1. 确保已安装依赖：pip install pyvista pygame numpy -i 清华镜像")
        print("2. 关闭PyCharm的\"Power Save Mode\"（省电模式）")
        print("3. 以管理员身份运行PyCharm")
        # 强制清理资源
        pygame.quit()
        sys.exit(1)