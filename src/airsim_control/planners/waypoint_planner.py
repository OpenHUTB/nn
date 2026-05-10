from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


@dataclass
class Waypoint:
    """航点，支持降落参数"""
    x: float
    y: float
    z: float          # 目标高度（向上为正）
    is_landing: bool = False      # 是否触发降落
    descend_speed: float = 1.0   # 下降速度 m/s
    hover_time: float = 3.0      # 悬停时间 s（减速段）
    is_return_home: bool = False  # 降落后是否返回初始点
    fly_speed: float = 1.0        # 飞向该航点时的最大速度 m/s

    def pos(self):
        return np.array([self.x, self.y, self.z])


class WaypointPlanner:
    """航点规划器"""

    def __init__(self, waypoints=None, loop=False):
        if waypoints is None:
            self.waypoints: list[Waypoint] = []
        else:
            # 兼容旧接口：传入普通元组列表
            self.waypoints = [w if isinstance(w, Waypoint) else Waypoint(*w) for w in waypoints]
        self.loop = loop
        self.current_idx = 0
        self.reached_history = []

    def add_waypoint(self, x, y, z, is_landing=False, descend_speed=1.0, hover_time=3.0, is_return_home=False, fly_speed=1.0):
        self.waypoints.append(Waypoint(x, y, z, is_landing, descend_speed, hover_time, is_return_home, fly_speed))

    def get_current_target(self):
        if not self.waypoints or self.current_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.current_idx].pos()

    def advance(self):
        self.reached_history.append(self.current_idx)
        if self.current_idx < len(self.waypoints) - 1:
            self.current_idx += 1
            return False
        elif self.loop:
            self.current_idx = 0
            return False
        return True  # 所有航点完成

    def is_finished(self):
        return not self.loop and self.current_idx >= len(self.waypoints) - 1

    def reset(self):
        self.current_idx = 0
        self.reached_history = []

    def distance_to_target(self, position):
        target = self.get_current_target()
        if target is None:
            return float('inf')
        return np.linalg.norm(position - target)

    def get_progress(self):
        if not self.waypoints:
            return 0, 0
        return self.current_idx + 1, len(self.waypoints)


class WaypointNavigator:
    """3D可视化航点导航器（中文优化版）"""

    def __init__(self, waypoints=None, update_interval=0.5, start_pos=None):
        self.planner = WaypointPlanner(waypoints)
        self.update_interval = update_interval
        self.trajectory = []
        self.fig = None
        self.ax = None
        self.start_pos = start_pos  # 初始位置
        self._init_plot()

    def set_start(self, pos):
        self.start_pos = np.array(pos)

    def _init_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8), facecolor='#1a1a2e')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='#16213e')
        self._draw()

    def _draw(self):
        self.ax.clear()
        waypoints = self.planner.waypoints
        traj = np.array(self.trajectory) if self.trajectory else np.zeros((1, 3))

        # 设置背景
        self.ax.set_facecolor('#16213e')
        self.fig.patch.set_facecolor('#1a1a2e')

        # 地面平面
        wp_arr = np.array([[w.x, w.y, w.z] for w in waypoints]) if waypoints else traj
        x_min, x_max = min(wp_arr[:, 0].min(), traj[:, 0].min()) - 5, max(wp_arr[:, 0].max(), traj[:, 0].max()) + 5
        y_min, y_max = min(wp_arr[:, 1].min(), traj[:, 1].min()) - 5, max(wp_arr[:, 1].max(), traj[:, 1].max()) + 5
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, 10),
            np.linspace(y_min, y_max, 10)
        )
        self.ax.plot_surface(grid_x, grid_y, np.zeros_like(grid_x),
                            alpha=0.15, color='#0f3460', shade=True)
        self.ax.contour(grid_x, grid_y, np.zeros_like(grid_x),
                       zdir='z', offset=0, colors='#0f3460', alpha=0.5, linewidths=0.8)

        # 初始点
        if self.start_pos is not None:
            sp = np.array(self.start_pos)
            self.ax.scatter([sp[0]], [sp[1]], [sp[2]],
                           c='#06d6a0', s=250, marker='D', edgecolors='white',
                           linewidths=2, zorder=8, label='起始点')
            self.ax.text(sp[0] + 0.4, sp[1] + 0.4, sp[2] + 0.4,
                       '起始点', color='#06d6a0', fontsize=10, fontweight='bold')

        # 飞行轨迹（带渐变效果）
        if len(traj) > 1:
            # 轨迹线
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                        color='#00d4ff', linewidth=2.5, alpha=0.9, label='飞行轨迹')
            # 轨迹头部高亮
            self.ax.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                           color='#00d4ff', s=150, marker='o', edgecolors='white',
                           linewidths=1.5, zorder=10, label='当前位置')

        # 普通航点
        if waypoints:
            wp_x = [w.x for w in waypoints]
            wp_y = [w.y for w in waypoints]
            wp_z = [w.z for w in waypoints]

            # 普通航点（蓝色三角）
            normal_wps = [(w.x, w.y, w.z) for w in waypoints if not w.is_landing]
            if normal_wps:
                nx, ny, nz = zip(*normal_wps)
                self.ax.scatter(nx, ny, nz, c='#4cc9f0', s=200, marker='^',
                              edgecolors='white', linewidths=1.2, zorder=5, label='普通航点')
                for i, (wx, wy, wz) in enumerate(normal_wps):
                    self.ax.text(wx + 0.3, wy + 0.3, wz + 0.3,
                               f'WP{i+1}', color='#4cc9f0', fontsize=9, fontweight='bold')

            # 降落航点（红色三角，特殊标记）
            landing_wps = [(w.x, w.y, w.z, i) for i, w in enumerate(waypoints) if w.is_landing]
            for wx, wy, wz, idx in landing_wps:
                self.ax.scatter([wx], [wy], [wz], c='#f72585', s=300, marker='v',
                              edgecolors='yellow', linewidths=2, zorder=6, label='降落航点')
                self.ax.text(wx + 0.3, wy + 0.3, wz + 0.3,
                           f'降落\nWP{idx+1}', color='#f72585', fontsize=9, fontweight='bold')

            # 航线连接线（从起始点到第一个航点，再到后续航点）
            # 降落航点的终点是地面(z=0)，不是悬停高度
            if self.start_pos is not None:
                sp = np.array(self.start_pos)
                wp_x_all = [sp[0]] + wp_x
                wp_y_all = [sp[1]] + wp_y
                # 最后一个是降落航点，终点画到地面
                if waypoints[-1].is_landing:
                    wp_z_all = [sp[2]] + [w.z for w in waypoints[:-1]] + [0]
                else:
                    wp_z_all = [sp[2]] + [w.z for w in waypoints]
                self.ax.plot(wp_x_all, wp_y_all, wp_z_all, 'w--', linewidth=1, alpha=0.4)
            else:
                if waypoints[-1].is_landing:
                    wp_z_all = [w.z for w in waypoints[:-1]] + [0]
                    wp_x_all = [w.x for w in waypoints[:-1]] + [waypoints[-1].x]
                    wp_y_all = [w.y for w in waypoints[:-1]] + [waypoints[-1].y]
                else:
                    wp_x_all, wp_y_all, wp_z_all = wp_x, wp_y, wp_z
                self.ax.plot(wp_x_all, wp_y_all, wp_z_all, 'w--', linewidth=1, alpha=0.4)

            # 当前目标高亮（普通航点时显示，降落航点不重复覆盖）
            if self.planner.current_idx < len(waypoints):
                curr = waypoints[self.planner.current_idx]
                if not curr.is_landing:
                    self.ax.scatter([curr.x], [curr.y], [curr.z],
                                 c='#7209b7', s=400, marker='*', edgecolors='#f8f9fa',
                                 linewidths=2, zorder=7)
                    self.ax.text(curr.x + 0.4, curr.y + 0.4, curr.z + 0.4,
                               '当前目标', color='#f8f9fa', fontsize=10, fontweight='bold')

        # 坐标轴样式
        self.ax.tick_params(colors='#adb5bd', labelsize=9)
        self.ax.xaxis.label.set_color('#adb5bd')
        self.ax.yaxis.label.set_color('#adb5bd')
        self.ax.zaxis.label.set_color('#adb5bd')
        for spine in self.ax.spines.values():
            spine.set_color('#495057')

        self.ax.set_xlabel('X (m)', color='#adb5bd', fontsize=11, labelpad=10)
        self.ax.set_ylabel('Y (m)', color='#adb5bd', fontsize=11, labelpad=10)
        self.ax.set_zlabel('Z 高度 (m)', color='#adb5bd', fontsize=11, labelpad=10)
        self.ax.set_title('无人机航点导航 · 实时3D轨迹', color='white', fontsize=16,
                         fontweight='bold', pad=20)

        # 图例
        handles, labels = self.ax.get_legend_handles_labels()
        # 只保留有标签的元素
        valid_handles = []
        valid_labels = []
        for h, l in zip(handles, labels):
            if l:  # 只保留有标签的元素
                valid_handles.append(h)
                valid_labels.append(l)
        if valid_handles:  # 只有在有有效元素时才创建图例
            self.ax.legend(handles=valid_handles, labels=valid_labels, loc='upper left', fontsize=10, facecolor='#1a1a2e',
                         edgecolor='#495057', labelcolor='white')

        # 网格
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#2a2a4a')
        self.ax.yaxis.pane.set_edgecolor('#2a2a4a')
        self.ax.zaxis.pane.set_edgecolor('#2a2a4a')
        self.ax.grid(True, color='#2a2a4a', linewidth=0.8)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # 坐标轴范围
        all_pts = np.vstack([wp_arr, traj]) if len(traj) > 0 else wp_arr
        margin = 5
        self.ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        self.ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
        z_min = min(all_pts[:, 2].min(), 0) - margin
        z_max = max(all_pts[:, 2].max(), 5) + margin
        self.ax.set_zlim(z_min, z_max)

        # 进度信息
        if waypoints:
            idx = self.planner.current_idx
            total = len(waypoints)
            info_text = f'进度: {idx+1}/{total}'
            self.fig.text(0.88, 0.95, info_text, color='#00d4ff',
                         fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='#16213e',
                                 edgecolor='#00d4ff', alpha=0.8))

        plt.draw()
        plt.pause(0.01)

    def update(self, position):
        self.trajectory.append(position.copy())
        # 减少绘图频率，避免卡顿
        if len(self.trajectory) % 20 == 0:  # 每20次更新才绘图一次
            try:
                self._draw()
            except Exception:
                pass  # 忽略绘图错误

    def show(self):
        plt.ioff()
        self._draw()
        plt.tight_layout()
        plt.show()
