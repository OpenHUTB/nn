import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches

# ==================== PyCharm环境配置 ====================
plt.switch_backend('TkAgg')  # 显示交互窗口（如需静默保存改'Agg'）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200  # 高清渲染
plt.rcParams['lines.antialiased'] = True  # 抗锯齿

# 保存路径（桌面，可自定义）
SAVE_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "精细机械臂可视化.png")


class PrecisionArmVisualizer:
    def __init__(self):
        """初始化精细机械臂参数"""
        # 工业级机械臂参数（模拟三菱RV-2F）
        self.link_params = {
            'link1': {'length': 120, 'width': 15, 'color': '#1f77b4', 'max_angle': 180, 'min_angle': -180},
            'link2': {'length': 100, 'width': 12, 'color': '#ff7f0e', 'max_angle': 150, 'min_angle': -90},
            'link3': {'length': 80, 'width': 10, 'color': '#2ca02c', 'max_angle': 120, 'min_angle': -120},
            'link4': {'length': 60, 'width': 8, 'color': '#d62728', 'max_angle': 180, 'min_angle': -180}
        }
        # 关节参数
        self.joint_params = {
            'base': {'radius': 25, 'color': '#000000', 'bearing_width': 5},
            'joint': {'radius': 20, 'color': '#7f7f7f', 'bearing_width': 4},
            'end_effector': {'radius': 22, 'color': '#9467bd', 'gripper_length': 30, 'gripper_width': 15}
        }
        # 工作空间参数
        self.workspace = {'radius': 350, 'color': '#e377c2', 'alpha': 0.1}

    def calculate_joint_coords(self, angles):
        """高精度计算关节坐标（毫米单位）"""
        # 角度转换（输入为角度，内部转弧度）
        angles_rad = np.radians(angles)
        coords = np.array([[0.0, 0.0]])  # 基座原点

        current_angle = 0.0
        for i, angle in enumerate(angles_rad):
            link_key = f'link{i + 1}'
            link_len = self.link_params[link_key]['length']

            current_angle += angle
            dx = link_len * np.cos(current_angle)
            dy = link_len * np.sin(current_angle)

            next_coord = coords[-1] + np.array([dx, dy])
            coords = np.vstack([coords, next_coord])

        return coords

    def draw_precision_arm(self, angles=[30, -45, 60, -15]):
        """绘制精细机械臂"""
        # 1. 计算坐标
        coords = self.calculate_joint_coords(angles)
        num_joints = len(coords)

        # 2. 创建画布（工业图纸风格）
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xlim(-400, 400)
        ax.set_ylim(-400, 400)
        ax.set_aspect('equal', adjustable='box')

        # 3. 绘制工作空间
        workspace_circle = patches.Circle((0, 0), self.workspace['radius'],
                                          facecolor=self.workspace['color'],
                                          edgecolor=self.workspace['color'],
                                          alpha=self.workspace['alpha'],
                                          label='工作空间范围')
        ax.add_patch(workspace_circle)

        # 4. 绘制连杆（精细分层）
        for i in range(num_joints - 1):
            link_key = f'link{i + 1}'
            link = self.link_params[link_key]

            # 计算连杆中心点和方向
            start = coords[i]
            end = coords[i + 1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = np.hypot(dx, dy)
            angle = np.arctan2(dy, dx)

            # 绘制连杆主体（填充）
            link_rect = patches.Rectangle(
                (start[0], start[1] - link['width'] / 2),
                length, link['width'],
                angle=np.degrees(angle),
                facecolor=link['color'],
                edgecolor='black',
                linewidth=1.5,
                alpha=0.8
            )
            ax.add_patch(link_rect)

            # 绘制连杆轮廓（增强立体感）
            link_outline = patches.Rectangle(
                (start[0], start[1] - link['width'] / 2),
                length, link['width'],
                angle=np.degrees(angle),
                facecolor='none',
                edgecolor='white',
                linewidth=0.8,
                linestyle='-'
            )
            ax.add_patch(link_outline)

            # 连杆参数标注
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y,
                    f'L{i + 1}\n{link["length"]}mm\n{angles[i]}°',
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                    rotation=np.degrees(angle))

        # 5. 绘制关节（带轴承细节）
        # 基座关节
        base = self.joint_params['base']
        base_circle = patches.Circle(coords[0], base['radius'],
                                     facecolor=base['color'], edgecolor='white', linewidth=2)
        ax.add_patch(base_circle)
        # 基座轴承
        base_bearing = patches.Circle(coords[0], base['radius'] - base['bearing_width'],
                                      facecolor='#333333', edgecolor='white', linewidth=1)
        ax.add_patch(base_bearing)

        # 中间关节
        joint = self.joint_params['joint']
        for i in range(1, num_joints - 1):
            # 关节主体
            joint_circle = patches.Circle(coords[i], joint['radius'],
                                          facecolor=joint['color'], edgecolor='white', linewidth=1.5)
            ax.add_patch(joint_circle)
            # 关节轴承
            joint_bearing = patches.Circle(coords[i], joint['radius'] - joint['bearing_width'],
                                           facecolor='#555555', edgecolor='white', linewidth=0.8)
            ax.add_patch(joint_bearing)

            # 关节限位标注
            link_key = f'link{i}'
            link = self.link_params[link_key]
            ax.text(coords[i][0] + 30, coords[i][1] + 30,
                    f'限位：{link["min_angle"]}°~{link["max_angle"]}°',
                    fontsize=7, bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

        # 6. 绘制末端执行器（夹爪细节）
        end_eff = self.joint_params['end_effector']
        end_coord = coords[-1]
        # 末端主体
        end_circle = patches.Circle(end_coord, end_eff['radius'],
                                    facecolor=end_eff['color'], edgecolor='white', linewidth=2)
        ax.add_patch(end_circle)

        # 夹爪
        gripper_len = end_eff['gripper_length']
        gripper_wid = end_eff['gripper_width']
        # 左夹爪
        left_gripper = patches.Rectangle(
            (end_coord[0] + 10, end_coord[1] - gripper_wid / 2),
            gripper_len, gripper_wid / 2,
            facecolor='#888888', edgecolor='black', linewidth=1
        )
        ax.add_patch(left_gripper)
        # 右夹爪
        right_gripper = patches.Rectangle(
            (end_coord[0] + 10, end_coord[1] + gripper_wid / 4),
            gripper_len, gripper_wid / 2,
            facecolor='#888888', edgecolor='black', linewidth=1
        )
        ax.add_patch(right_gripper)

        # 7. 绘制辅助信息
        # 坐标网格（精细刻度）
        ax.grid(True, which='both', alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_xticks(np.arange(-400, 401, 50))
        ax.set_yticks(np.arange(-400, 401, 50))
        ax.tick_params(labelsize=9)

        # 末端坐标高精度标注
        end_x, end_y = end_coord
        ax.text(end_x + 50, end_y - 50,
                f'末端执行器参数\n坐标：({end_x:.2f}, {end_y:.2f}) mm\n姿态角：{sum(angles):.1f}°\n理论负载：2.0 kg',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9))

        # 标题和图例
        ax.set_title('高精度工业机械臂2D可视化（1:10比例）', fontsize=14, pad=20)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

        # 8. 保存高清图片
        try:
            plt.tight_layout()
            fig.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n✅ 精细机械臂图片已保存至：")
            print(f"   {SAVE_PATH}")
            print(f"   图片分辨率：{fig.get_size_inches()[0] * 300}x{fig.get_size_inches()[1] * 300} 像素")
        except Exception as e:
            print(f"\n❌ 保存失败：{str(e)}")

        # 显示交互窗口（PyCharm中可缩放/平移）
        plt.show()


# ==================== 运行入口 ====================
if __name__ == "__main__":
    # 创建可视化器
    visualizer = PrecisionArmVisualizer()

    # 设置关节角度（可自定义，单位：度）
    # 顺序：link1角度, link2角度, link3角度, link4角度
    arm_angles = [45, -60, 75, -30]

    # 绘制精细机械臂
    visualizer.draw_precision_arm(angles=arm_angles)

