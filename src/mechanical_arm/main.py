import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class RoboticArmWithGripper:
    def __init__(self):
        # 机械臂参数
        self.link_lengths = [2.0, 1.5, 1.0, 0.5]  # 四个连杆长度
        self.joint_angles = [0.0, 0.0, 0.0, 0.0]  # 四个关节角度（弧度）

        # 夹爪参数
        self.gripper_length = 0.3  # 手指长度
        self.gripper_width = 0.15  # 夹爪宽度
        self.gripper_opening = 0.2  # 夹爪开口大小
        self.gripper_angle = 0.0  # 夹爪旋转角度

        # DH参数
        self.update_dh_params()

        # 初始化图形
        self.fig = plt.figure(figsize=(14, 8))
        self.setup_plot()

    def update_dh_params(self):
        """更新DH参数"""
        self.dh_params = [
            [self.link_lengths[0], 0, 0, self.joint_angles[0]],
            [self.link_lengths[1], np.pi / 2, 0, self.joint_angles[1]],
            [self.link_lengths[2], 0, 0, self.joint_angles[2]],
            [self.link_lengths[3], 0, 0, self.joint_angles[3]]
        ]

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
        """正向运动学：计算每个关节和夹爪的位置"""
        T = np.eye(4)
        positions = [np.array([0, 0, 0])]
        transforms = [T.copy()]

        # 计算各关节位置
        for params in self.dh_params:
            a, alpha, d, theta = params
            T = T @ self.dh_matrix(a, alpha, d, theta)
            position = T[:3, 3]
            positions.append(position)
            transforms.append(T.copy())

        # 末端执行器变换矩阵
        end_effector_T = T

        # 计算夹爪位置
        gripper_positions = self.calculate_gripper_positions(end_effector_T)

        return positions, transforms, gripper_positions

    def calculate_gripper_positions(self, end_effector_T):
        """计算夹爪各点位置"""
        # 夹爪基座中心点（末端执行器位置）
        base_pos = end_effector_T[:3, 3]

        # 末端执行器的坐标系方向
        x_axis = end_effector_T[:3, 0]
        y_axis = end_effector_T[:3, 1]
        z_axis = end_effector_T[:3, 2]

        # 计算旋转后的夹爪方向
        rotation_matrix = self.rotation_matrix_z(self.gripper_angle)
        gripper_x = rotation_matrix @ x_axis

        # 三个手指的位置
        finger_positions = []

        # 手指1（中间）
        finger1_tip = base_pos + z_axis * self.gripper_length
        finger1_base = base_pos
        finger_positions.append((finger1_base, finger1_tip))

        # 手指2（左侧）
        offset = gripper_x * (self.gripper_width / 2 + self.gripper_opening / 2)
        finger2_base = base_pos - offset
        finger2_tip = finger2_base + z_axis * self.gripper_length
        finger_positions.append((finger2_base, finger2_tip))

        # 手指3（右侧）
        finger3_base = base_pos + offset
        finger3_tip = finger3_base + z_axis * self.gripper_length
        finger_positions.append((finger3_base, finger3_tip))

        # 夹爪基座连接点
        base_connections = [
            (finger1_base, finger2_base),
            (finger1_base, finger3_base)
        ]

        return {
            'fingers': finger_positions,
            'base_connections': base_connections,
            'end_effector': base_pos
        }

    def rotation_matrix_z(self, angle):
        """绕Z轴旋转矩阵"""
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    def setup_plot(self):
        """设置图形界面"""
        # 3D主视图
        self.ax1 = self.fig.add_subplot(131, projection='3d')
        self.ax1.set_title('Robotic Arm with 3-Finger Gripper')
        self.setup_3d_axis(self.ax1)

        # XY平面视图
        self.ax2 = self.fig.add_subplot(132)
        self.ax2.set_title('Top View (XY Plane)')
        self.ax2.set_xlabel('X Axis')
        self.ax2.set_ylabel('Y Axis')
        self.ax2.set_xlim(-3, 3)
        self.ax2.set_ylim(-3, 3)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_aspect('equal')

        # XZ平面视图
        self.ax3 = self.fig.add_subplot(133)
        self.ax3.set_title('Front View (XZ Plane)')
        self.ax3.set_xlabel('X Axis')
        self.ax3.set_ylabel('Z Axis')
        self.ax3.set_xlim(-3, 3)
        self.ax3.set_ylim(0, 4)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_aspect('equal')

        # 添加控制滑块
        self.setup_controls()

        # 调整子图间距
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)

        # 初始化绘图
        self.update_plot()

    def setup_3d_axis(self, ax):
        """设置3D坐标轴"""
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(0, 4)
        ax.grid(True, alpha=0.3)

    def setup_controls(self):
        """设置控制面板"""
        # 关节角度控制滑块
        slider_y = 0.20
        slider_height = 0.03
        slider_spacing = 0.04

        self.sliders = []
        slider_labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4']
        slider_ranges = [(-np.pi, np.pi), (-np.pi / 2, np.pi / 2),
                         (-np.pi / 2, np.pi / 2), (-np.pi, np.pi)]

        for i in range(4):
            ax_slider = plt.axes([0.15, slider_y - i * slider_spacing, 0.6, slider_height])
            slider = Slider(ax_slider, slider_labels[i],
                            slider_ranges[i][0], slider_ranges[i][1],
                            valinit=self.joint_angles[i],
                            valfmt='%.2f rad')
            slider.on_changed(self.update_from_slider)
            self.sliders.append(slider)

        # 夹爪控制滑块
        gripper_y = slider_y - 5 * slider_spacing

        # 夹爪开口控制
        ax_gripper_open = plt.axes([0.15, gripper_y, 0.6, slider_height])
        self.slider_gripper_open = Slider(ax_gripper_open, 'Gripper Opening',
                                          0.0, 0.4, valinit=self.gripper_opening)
        self.slider_gripper_open.on_changed(self.update_gripper)

        # 夹爪旋转控制
        ax_gripper_rotate = plt.axes([0.15, gripper_y - slider_spacing, 0.6, slider_height])
        self.slider_gripper_rotate = Slider(ax_gripper_rotate, 'Gripper Rotation',
                                            -np.pi, np.pi, valinit=self.gripper_angle)
        self.slider_gripper_rotate.on_changed(self.update_gripper)

        # 按钮控制
        button_y = 0.05
        button_width = 0.15
        button_height = 0.05

        # 重置按钮
        ax_reset = plt.axes([0.15, button_y, button_width, button_height])
        self.reset_button = Button(ax_reset, 'Reset All')
        self.reset_button.on_clicked(self.reset_all)

        # 抓取演示按钮
        ax_grasp = plt.axes([0.35, button_y, button_width, button_height])
        self.grasp_button = Button(ax_grasp, 'Grasp Demo')
        self.grasp_button.on_clicked(self.grasp_demo)

        # 动画按钮
        ax_animate = plt.axes([0.55, button_y, button_width, button_height])
        self.animate_button = Button(ax_animate, 'Animate')
        self.animate_button.on_clicked(self.animate_movement)

    def update_from_slider(self, val):
        """更新关节角度"""
        self.joint_angles = [s.val for s in self.sliders]
        self.update_dh_params()
        self.update_plot()

    def update_gripper(self, val):
        """更新夹爪参数"""
        self.gripper_opening = self.slider_gripper_open.val
        self.gripper_angle = self.slider_gripper_rotate.val
        self.update_plot()

    def reset_all(self, event):
        """重置所有参数"""
        # 重置关节角度
        for i, slider in enumerate(self.sliders):
            slider.set_val(0.0)

        # 重置夹爪
        self.slider_gripper_open.set_val(0.2)
        self.slider_gripper_rotate.set_val(0.0)

    def grasp_demo(self, event):
        """抓取演示动画"""
        # 保存原始开口大小
        original_opening = self.gripper_opening

        # 闭合夹爪
        for i in range(20):
            self.gripper_opening = original_opening * (1 - i / 20)
            self.slider_gripper_open.set_val(self.gripper_opening)
            self.update_plot()
            plt.pause(0.05)

        plt.pause(0.5)

        # 打开夹爪
        for i in range(20):
            self.gripper_opening = original_opening * (i / 20)
            self.slider_gripper_open.set_val(self.gripper_opening)
            self.update_plot()
            plt.pause(0.05)

    def animate_movement(self, event):
        """动画演示"""
        # 保存原始角度
        original_angles = self.joint_angles.copy()

        # 定义动画路径
        path = [
            [0.5, 0.3, -0.2, 0.1],
            [-0.3, 0.6, -0.4, 0.2],
            [0.8, -0.2, 0.5, -0.3],
            original_angles
        ]

        for target_angles in path:
            # 平滑移动到目标角度
            for i in range(20):
                for j in range(4):
                    current = self.joint_angles[j]
                    target = target_angles[j]
                    self.joint_angles[j] = current + (target - current) * (i + 1) / 20
                self.update_dh_params()
                self.update_plot()
                plt.pause(0.03)

            plt.pause(0.5)

        # 恢复原始角度
        for i in range(20):
            for j in range(4):
                current = self.joint_angles[j]
                target = original_angles[j]
                self.joint_angles[j] = current + (target - current) * (i + 1) / 20
            self.update_dh_params()
            self.update_plot()
            plt.pause(0.03)

    def update_plot(self):
        """更新所有图形"""
        # 计算正向运动学
        positions, transforms, gripper_data = self.forward_kinematics()
        end_effector_pos = gripper_data['end_effector']

        # 提取坐标
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        z_coords = [p[2] for p in positions]

        # 1. 3D主视图
        self.ax1.clear()
        self.setup_3d_axis(self.ax1)
        self.ax1.set_title('Robotic Arm with 3-Finger Gripper')

        # 绘制机械臂连杆
        self.ax1.plot(x_coords, y_coords, z_coords, 'o-',
                      linewidth=3, markersize=8, color='blue',
                      markerfacecolor='red', zorder=5)

        # 绘制夹爪
        self.draw_gripper_3d(self.ax1, gripper_data)

        # 2. XY平面视图
        self.ax2.clear()
        self.ax2.set_title('Top View (XY Plane)')
        self.ax2.set_xlabel('X Axis')
        self.ax2.set_ylabel('Y Axis')
        self.ax2.set_xlim(-3, 3)
        self.ax2.set_ylim(-3, 3)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_aspect('equal')

        self.ax2.plot(x_coords, y_coords, 'o-', linewidth=2,
                      markersize=6, color='blue', markerfacecolor='red')
        self.draw_gripper_2d(self.ax2, gripper_data, 'xy')

        # 3. XZ平面视图
        self.ax3.clear()
        self.ax3.set_title('Front View (XZ Plane)')
        self.ax3.set_xlabel('X Axis')
        self.ax3.set_ylabel('Z Axis')
        self.ax3.set_xlim(-3, 3)
        self.ax3.set_ylim(0, 4)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_aspect('equal')

        self.ax3.plot(x_coords, z_coords, 'o-', linewidth=2,
                      markersize=6, color='green', markerfacecolor='red')
        self.draw_gripper_2d(self.ax3, gripper_data, 'xz')

        # 在3D视图中添加信息文本
        angle_text = f'Joint Angles:\n' + '\n'.join([
            f'Joint {i + 1}: {angle:.2f} rad ({np.degrees(angle):.1f}°)'
            for i, angle in enumerate(self.joint_angles)
        ])

        gripper_text = f'Gripper:\nOpening: {self.gripper_opening:.3f} m\nRotation: {np.degrees(self.gripper_angle):.1f}°'

        end_text = f'End Effector:\nX: {end_effector_pos[0]:.3f}\nY: {end_effector_pos[1]:.3f}\nZ: {end_effector_pos[2]:.3f}'

        # 在3D图上添加文本
        self.ax1.text2D(0.05, 0.95, angle_text, transform=self.ax1.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        self.ax1.text2D(0.05, 0.65, gripper_text, transform=self.ax1.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        self.ax1.text2D(0.05, 0.35, end_text, transform=self.ax1.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

        self.fig.canvas.draw_idle()

    def draw_gripper_3d(self, ax, gripper_data):
        """在3D图中绘制夹爪"""
        # 绘制手指
        for base, tip in gripper_data['fingers']:
            ax.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]],
                    'k-', linewidth=4, zorder=10)
            # 手指尖端
            ax.scatter(tip[0], tip[1], tip[2], color='gray', s=50, zorder=11)

        # 绘制基座连接
        for p1, p2 in gripper_data['base_connections']:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    'k-', linewidth=2, zorder=9)

        # 绘制末端执行器
        end_pos = gripper_data['end_effector']
        ax.scatter(end_pos[0], end_pos[1], end_pos[2],
                   color='orange', s=100, zorder=8)

    def draw_gripper_2d(self, ax, gripper_data, plane):
        """在2D图中绘制夹爪投影"""
        for base, tip in gripper_data['fingers']:
            if plane == 'xy':
                ax.plot([base[0], tip[0]], [base[1], tip[1]],
                        'k-', linewidth=3)
            elif plane == 'xz':
                ax.plot([base[0], tip[0]], [base[2], tip[2]],
                        'k-', linewidth=3)


def main():
    """主函数"""
    print("=" * 60)
    print("3D Robotic Arm with 3-Finger Gripper Visualization")
    print("=" * 60)
    print("\nFeatures:")
    print("1. 4-DOF robotic arm with forward kinematics")
    print("2. 3-finger gripper with independent control")
    print("3. Real-time 3D visualization")
    print("4. Multiple views (3D, top, front)")
    print("5. Interactive controls for joints and gripper")
    print("\nControls:")
    print("- Use sliders to adjust joint angles")
    print("- Control gripper opening and rotation")
    print("- Use buttons for reset, grasp demo, and animation")
    print("=" * 60)

    # 创建机械臂实例
    arm = RoboticArmWithGripper()

    # 显示交互式界面
    plt.show()


if __name__ == "__main__":
    main()#结束过程