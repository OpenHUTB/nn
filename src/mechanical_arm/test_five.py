import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class RoboticArm3D:
    def __init__(self):
        # 机械臂参数：每个关节的长度
        self.link_lengths = [2.0, 1.5, 1.0, 0.5]  # 四个连杆长度
        self.joint_angles = [0.0, 0.0, 0.0, 0.0]  # 四个关节角度（弧度）

        # DH参数：每个连杆的参数 [a, alpha, d, theta]
        # a: 连杆长度, alpha: 连杆扭角, d: 连杆偏移, theta: 关节角度
        self.dh_params = [
            [self.link_lengths[0], 0, 0, self.joint_angles[0]],
            [self.link_lengths[1], np.pi / 2, 0, self.joint_angles[1]],
            [self.link_lengths[2], 0, 0, self.joint_angles[2]],
            [self.link_lengths[3], 0, 0, self.joint_angles[3]]
        ]

        # 初始化图形
        self.fig = plt.figure(figsize=(14, 8))
        self.setup_plot()

    def dh_matrix(self, a, alpha, d, theta):
        """计算DH变换矩阵"""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)

        return np.array([
            [cos_t, -sin_t * cos_a, sin_t * sin_a, a * cos_t],
            [sin_t, cos_t * cos_a, -cos_t * sin_a, a * sin_t],
            [0, sin_a, cos_a, d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self):
        """正向运动学：计算每个关节的位置"""
        T = np.eye(4)  # 单位矩阵
        positions = [np.array([0, 0, 0])]  # 基座位置

        for params in self.dh_params:
            a, alpha, d, theta = params
            T = T @ self.dh_matrix(a, alpha, d, theta)
            position = T[:3, 3]
            positions.append(position)

        return positions

    def setup_plot(self):
        """设置3D图形"""
        # 3D主视图
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.set_title('3D机械臂模型')
        self.ax1.set_xlabel('X轴')
        self.ax1.set_ylabel('Y轴')
        self.ax1.set_zlabel('Z轴')
        self.ax1.set_xlim(-3, 3)
        self.ax1.set_ylim(-3, 3)
        self.ax1.set_zlim(0, 4)

        # XY平面视图
        self.ax2 = self.fig.add_subplot(222)
        self.ax2.set_title('XY平面视图')
        self.ax2.set_xlabel('X轴')
        self.ax2.set_ylabel('Y轴')
        self.ax2.set_xlim(-3, 3)
        self.ax2.set_ylim(-3, 3)
        self.ax2.grid(True)

        # XZ平面视图
        self.ax3 = self.fig.add_subplot(224)
        self.ax3.set_title('XZ平面视图')
        self.ax3.set_xlabel('X轴')
        self.ax3.set_ylabel('Z轴')
        self.ax3.set_xlim(-3, 3)
        self.ax3.set_ylim(0, 4)
        self.ax3.grid(True)

        # 添加控制滑块
        self.setup_sliders()

        # 初始化绘图
        self.update_plot()

    def setup_sliders(self):
        """设置角度控制滑块"""
        # 为滑块留出空间
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.25)

        # 创建滑块轴
        ax_slider1 = plt.axes([0.1, 0.15, 0.8, 0.03])
        ax_slider2 = plt.axes([0.1, 0.10, 0.8, 0.03])
        ax_slider3 = plt.axes([0.1, 0.05, 0.8, 0.03])
        ax_slider4 = plt.axes([0.1, 0.00, 0.8, 0.03])

        # 创建滑块（角度范围：-π 到 π）
        self.slider1 = Slider(ax_slider1, '关节1', -np.pi, np.pi,
                              valinit=self.joint_angles[0], valfmt='%.2f rad')
        self.slider2 = Slider(ax_slider2, '关节2', -np.pi / 2, np.pi / 2,
                              valinit=self.joint_angles[1], valfmt='%.2f rad')
        self.slider3 = Slider(ax_slider3, '关节3', -np.pi / 2, np.pi / 2,
                              valinit=self.joint_angles[2], valfmt='%.2f rad')
        self.slider4 = Slider(ax_slider4, '关节4', -np.pi, np.pi,
                              valinit=self.joint_angles[3], valfmt='%.2f rad')

        # 滑块回调函数
        self.slider1.on_changed(self.update_angles)
        self.slider2.on_changed(self.update_angles)
        self.slider3.on_changed(self.update_angles)
        self.slider4.on_changed(self.update_angles)

        # 添加重置按钮
        ax_reset = plt.axes([0.8, 0.9, 0.1, 0.04])
        self.reset_button = Button(ax_reset, '重置')
        self.reset_button.on_clicked(self.reset_angles)

    def update_angles(self, val):
        """更新关节角度"""
        self.joint_angles = [
            self.slider1.val,
            self.slider2.val,
            self.slider3.val,
            self.slider4.val
        ]

        # 更新DH参数
        for i in range(4):
            self.dh_params[i][3] = self.joint_angles[i]

        self.update_plot()

    def reset_angles(self, event):
        """重置所有角度为0"""
        self.slider1.set_val(0)
        self.slider2.set_val(0)
        self.slider3.set_val(0)
        self.slider4.set_val(0)

    def update_plot(self):
        """更新所有图形"""
        # 清除所有轴
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # 重新设置轴属性
        self.ax1.set_title('3D机械臂模型')
        self.ax1.set_xlabel('X轴')
        self.ax1.set_ylabel('Y轴')
        self.ax1.set_zlabel('Z轴')
        self.ax1.set_xlim(-3, 3)
        self.ax1.set_ylim(-3, 3)
        self.ax1.set_zlim(0, 4)

        self.ax2.set_title('XY平面视图')
        self.ax2.set_xlabel('X轴')
        self.ax2.set_ylabel('Y轴')
        self.ax2.set_xlim(-3, 3)
        self.ax2.set_ylim(-3, 3)
        self.ax2.grid(True)

        self.ax3.set_title('XZ平面视图')
        self.ax3.set_xlabel('X轴')
        self.ax3.set_ylabel('Z轴')
        self.ax3.set_xlim(-3, 3)
        self.ax3.set_ylim(0, 4)
        self.ax3.grid(True)

        # 计算正向运动学
        positions = self.forward_kinematics()

        # 提取坐标
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        z_coords = [p[2] for p in positions]

        # 3D视图
        self.ax1.plot(x_coords, y_coords, z_coords, 'o-', linewidth=3,
                      markersize=8, color='blue', markerfacecolor='red')

        # 添加基座
        self.ax1.plot([0, 0], [0, 0], [0, -0.5], 'k-', linewidth=5)

        # XY平面视图
        self.ax2.plot(x_coords, y_coords, 'o-', linewidth=2,
                      markersize=6, color='blue', markerfacecolor='red')
        self.ax2.plot(0, 0, 'ks', markersize=10)  # 基座

        # XZ平面视图
        self.ax3.plot(x_coords, z_coords, 'o-', linewidth=2,
                      markersize=6, color='green', markerfacecolor='red')
        self.ax3.plot(0, 0, 'ks', markersize=10)  # 基座

        # 显示末端位置
        end_effector = positions[-1]
        self.ax1.text(end_effector[0], end_effector[1], end_effector[2],
                      f'末端: ({end_effector[0]:.2f}, {end_effector[1]:.2f}, {end_effector[2]:.2f})',
                      fontsize=10, color='red')

        # 显示角度信息
        angle_text = f'关节角度:\n' + '\n'.join([f'关节{i + 1}: {angle:.2f} rad ({np.degrees(angle):.1f}°)'
                                                 for i, angle in enumerate(self.joint_angles)])
        self.ax2.text(-2.8, 2.5, angle_text, fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        self.fig.canvas.draw_idle()

    def animate_trajectory(self):
        """演示轨迹动画"""
        fig_anim = plt.figure(figsize=(10, 8))
        ax_anim = fig_anim.add_subplot(111, projection='3d')
        ax_anim.set_title('机械臂轨迹演示')
        ax_anim.set_xlabel('X轴')
        ax_anim.set_ylabel('Y轴')
        ax_anim.set_zlabel('Z轴')
        ax_anim.set_xlim(-3, 3)
        ax_anim.set_ylim(-3, 3)
        ax_anim.set_zlim(0, 4)

        # 生成轨迹点
        t = np.linspace(0, 2 * np.pi, 100)
        x_traj = 2 * np.cos(t)
        y_traj = 2 * np.sin(t)
        z_traj = 2 + 0.5 * np.sin(2 * t)

        # 绘制目标轨迹
        ax_anim.plot(x_traj, y_traj, z_traj, 'r--', alpha=0.5, label='目标轨迹')

        # 动画初始化
        line, = ax_anim.plot([], [], [], 'o-', linewidth=3, markersize=8)

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            return line,

        def animate(i):
            # 简单的逆运动学（简化版）
            theta1 = np.arctan2(y_traj[i], x_traj[i])
            theta2 = 0.5
            theta3 = -0.3
            theta4 = 0.2

            # 更新关节角度
            temp_angles = [theta1, theta2, theta3, theta4]
            temp_dh = self.dh_params.copy()
            for j in range(4):
                temp_dh[j][3] = temp_angles[j]

            # 计算位置
            T = np.eye(4)
            positions = [np.array([0, 0, 0])]
            for params in temp_dh:
                a, alpha, d, theta = params
                T = T @ self.dh_matrix(a, alpha, d, theta)
                positions.append(T[:3, 3])

            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            z_coords = [p[2] for p in positions]

            line.set_data(x_coords, y_coords)
            line.set_3d_properties(z_coords)

            return line,

        anim = animation.FuncAnimation(fig_anim, animate, init_func=init,
                                       frames=len(t), interval=50, blit=True)

        ax_anim.legend()
        plt.show()

        return anim


def main():
    """主函数"""
    print("正在初始化3D机械臂可视化系统...")
    print("=" * 50)
    print("操作说明:")
    print("1. 使用滑块调整各个关节的角度")
    print("2. 点击'重置'按钮将机械臂恢复初始位置")
    print("3. 关闭第一个窗口后，将演示轨迹动画")
    print("=" * 50)

    # 创建机械臂实例
    arm = RoboticArm3D()

    # 显示交互式界面
    plt.show()

    # 询问是否显示动画演示
    response = input("\n是否显示轨迹动画演示？(y/n): ")
    if response.lower() == 'y':
        print("正在生成轨迹动画...")
        arm.animate_trajectory()


if __name__ == "__main__":
    main()