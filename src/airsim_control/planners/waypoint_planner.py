import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class WaypointPlanner:
    """航点规划器"""

    def __init__(self, waypoints=None, loop=False):
        self.waypoints = list(waypoints) if waypoints else []
        self.loop = loop
        self.current_idx = 0
        self.reached_history = []

    def add_waypoint(self, x, y, z):
        self.waypoints.append(np.array([x, y, z]))

    def get_current_target(self):
        if not self.waypoints:
            return None
        return self.waypoints[self.current_idx]

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
    """3D可视化航点导航器"""

    def __init__(self, waypoints=None, update_interval=0.5):
        self.planner = WaypointPlanner(waypoints)
        self.update_interval = update_interval
        self.trajectory = []
        self.fig = None
        self.ax = None
        self._init_plot()

    def _init_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._draw()

    def _draw(self):
        self.ax.clear()
        waypoints = self.planner.waypoints
        traj = np.array(self.trajectory)

        # 航点
        if waypoints:
            wp = np.array(waypoints)
            self.ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2],
                           c='red', s=100, marker='^', label='Waypoints')
            self.ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], 'r--', alpha=0.5, label='Path')

        # 当前目标高亮
        if self.planner.current_idx < len(waypoints):
            curr = waypoints[self.planner.current_idx]
            self.ax.scatter([curr[0]], [curr[1]], [curr[2]],
                           c='green', s=200, marker='*', label='Current')

        # 飞行轨迹
        if len(traj) > 1:
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', alpha=0.7, label='Trajectory')
            self.ax.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                           c='blue', s=80, marker='o', label='Drone')

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Real-time Drone Navigation')
        self.ax.legend()
        self.ax.grid(True)

        # 自适应坐标轴
        all_points = wp if waypoints else np.zeros((1, 3))
        if len(traj) > 0:
            all_points = np.vstack([all_points, traj])
        margin = 5
        self.ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        self.ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        self.ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)

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
        plt.show()
