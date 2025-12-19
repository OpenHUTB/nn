import matplotlib.pyplot as plt
import numpy as np


class ArmVisualizer:
    def __init__(self, link_lengths=None):
        """
        初始化机械臂可视化器
        :param link_lengths: 各连杆长度列表，默认[5, 4, 3]
        """
        # 默认连杆长度
        self.link_lengths = link_lengths if link_lengths else [5, 4, 3]
        self.num_joints = len(self.link_lengths) + 1  # 关节数 = 连杆数 + 1

        # 初始化绘图
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-12, 12)
        self.ax.set_ylim(-12, 12)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_title('3自由度平面机械臂可视化')

    def calculate_joint_positions(self, angles):
        """
        根据关节角度计算各关节坐标
        :param angles: 各关节角度列表（弧度），长度等于连杆数
        :return: 关节坐标列表 [(x0,y0), (x1,y1), ..., (xn,yn)]
        """
        if len(angles) != len(self.link_lengths):
            raise ValueError(f"角度数量应等于连杆数量({len(self.link_lengths)})")

        # 初始关节（基座）在原点
        joints = [(0.0, 0.0)]
        current_x, current_y = 0.0, 0.0
        current_angle = 0.0

        # 计算每个关节的位置
        for i, (angle, length) in enumerate(zip(angles, self.link_lengths)):
            current_angle += angle
            current_x += length * np.cos(current_angle)
            current_y += length * np.sin(current_angle)
            joints.append((current_x, current_y))

        return joints

    def draw_arm(self, angles, save_path=None):
        """
        绘制机械臂并可选保存图片
        :param angles: 关节角度列表（弧度）
        :param save_path: 保存路径，如'robot_arm.png'
        """
        # 清空之前的绘图
        self.ax.clear()
        self.ax.set_xlim(-12, 12)
        self.ax.set_ylim(-12, 12)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_title('3自由度平面机械臂可视化')

        # 计算关节位置
        joints = self.calculate_joint_positions(angles)

        # 提取x和y坐标
        x_coords = [joint[0] for joint in joints]
        y_coords = [joint[1] for joint in joints]

        # 绘制连杆
        self.ax.plot(x_coords, y_coords, 'b-', linewidth=4, label='连杆')

        # 绘制关节（基座用红色，其他关节用蓝色）
        self.ax.plot(x_coords[0], y_coords[0], 'ro', markersize=10, label='基座')
        self.ax.plot(x_coords[1:], y_coords[1:], 'bo', markersize=8, label='关节')

        # 绘制末端执行器
        self.ax.plot(x_coords[-1], y_coords[-1], 'go', markersize=12, label='末端执行器')

        # 添加角度标注
        for i, (x, y) in enumerate(joints[:-1]):
            angle_deg = np.degrees(sum(angles[:i + 1]))
            self.ax.text(x + 0.2, y + 0.2, f'θ{i + 1}={angle_deg:.0f}°',
                         fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

        # 添加连杆长度标注
        for i in range(len(self.link_lengths)):
            mid_x = (x_coords[i] + x_coords[i + 1]) / 2
            mid_y = (y_coords[i] + y_coords[i + 1]) / 2
            self.ax.text(mid_x, mid_y, f'L{i + 1}={self.link_lengths[i]}',
                         fontsize=8, color='darkred')

        # 添加末端坐标标注
        end_x, end_y = joints[-1]
        self.ax.text(end_x + 0.3, end_y - 0.3, f'末端坐标:\n({end_x:.2f}, {end_y:.2f})',
                     fontsize=9, bbox=dict(facecolor='lightgreen', alpha=0.7))

        self.ax.legend(loc='upper right')

        # 保存图片（如果指定路径）
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"机械臂图片已保存至: {save_path}")

        plt.show()


# 示例使用
if __name__ == "__main__":
    # 创建机械臂可视化器（可自定义连杆长度）
    arm_visualizer = ArmVisualizer(link_lengths=[6, 4, 3])

    # 设置关节角度（弧度），可以修改这些值查看不同姿态
    # 示例1：初始姿态（伸展）
    # angles = [0, 0, 0]

    # 示例2：弯曲姿态
    angles = [np.pi / 4, np.pi / 6, -np.pi / 3]

    # 示例3：更复杂的姿态
    # angles = [np.pi/3, -np.pi/4, np.pi/2]

    # 绘制并保存机械臂图片
    arm_visualizer.draw_arm(angles, save_path='robot_arm_visualization.png')