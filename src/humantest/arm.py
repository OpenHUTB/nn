import mujoco.viewer
import ikpy.chain
import transforms3d as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


def viewer_init(viewer):
    """渲染器的摄像头视角初始化"""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, 0.5, 0.5]
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30


class ForcePlotter:
    """实时可视化接触力"""

    def __init__(self, update_interval=20):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.update_interval = update_interval  # 更新间隔帧数
        self.frame_count = 0  # 帧计数器

    def plot_force_vector(self, force_vector):
        self.frame_count += 1
        if self.frame_count % self.update_interval != 0:
            return  # 跳过本次渲染

        self.ax.clear()

        origin = np.array([0, 0, 0])
        force_magnitude = np.linalg.norm(force_vector)
        force_direction = force_vector / force_magnitude if force_magnitude > 1e-6 else np.zeros(3)

        # 主箭头
        arrow_tip = force_direction * 1.5
        self.ax.quiver(*origin, *arrow_tip, color='r', arrow_length_ratio=0)

        # 蓝色箭头
        self.ax.quiver(*arrow_tip, *(0.5 * force_direction), color='b', arrow_length_ratio=0.5)

        # XY平面投影
        self.ax.plot([0, arrow_tip[0]], [0, arrow_tip[1]], [-2, -2], 'g--')

        # XZ平面投影
        self.ax.plot([0, 0], [2, 2], [0, arrow_tip[2]], 'm--')

        # 力大小指示条
        scaled_force = min(max(force_magnitude / 50, 0), 2)
        self.ax.plot([-2, -2], [2, 2], [0, scaled_force], 'c-')
        self.ax.text(-2, 2, scaled_force, f'Force: {force_magnitude:.1f}', color='c')

        # 坐标系设置
        self.ax.scatter(0, 0, 0, color='k', s=10)
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_title(f'Force Direction')

        plt.draw()
        plt.pause(0.001)
        self.frame_count = 0  # 重置计数器


class ForceSensor:
    def __init__(self, model, data, window_size=100):
        self.model = model
        self.data = data
        self.window_size = window_size
        self.force_history = deque(maxlen=window_size)

    def filter(self):
        """获取并滑动平均滤波力传感器数据(传感器坐标系下)"""
        # 获取MjData中的传感器数据
        force_local_raw = self.data.sensordata[:3].copy() * -1

        # 添加新数据到滑动窗口
        self.force_history.append(force_local_raw)

        # 计算滑动平均
        filtered_force = np.mean(self.force_history, axis=0)

        return filtered_force


class JointSpaceTrajectory:
    """关节空间坐标系下的线性插值轨迹"""

    def __init__(self, start_joints, end_joints, steps):
        self.start_joints = np.array(start_joints)
        self.end_joints = np.array(end_joints)
        self.steps = steps
        self.step = (self.end_joints - self.start_joints) / self.steps
        self.trajectory = self._generate_trajectory()
        self.waypoint = self.start_joints

    def _generate_trajectory(self):
        for i in range(self.steps + 1):
            yield self.start_joints + self.step * i
        # 确保最后精确到达目标关节值
        yield self.end_joints

    def get_next_waypoint(self, qpos):
        # 检查当前的关节值是否已经接近目标路径点。若是，则更新下一个目标路径点；若否，则保持当前目标路径点不变。
        if np.allclose(qpos, self.waypoint, atol=0.02):
            try:
                self.waypoint = next(self.trajectory)
                return self.waypoint
            except StopIteration:
                pass
        return self.waypoint


def main():
    model = mujoco.MjModel.from_xml_path('model/universal_robots_ur5e/scene.xml')
    data = mujoco.MjData(model)
    my_chain = ikpy.chain.Chain.from_urdf_file("model/ur5e.urdf",
                                               active_links_mask=[False, False] + [True] * 6 + [False])

    start_joints = np.array([-1.57, -1.34, 2.65, -1.3, 1.55, 0])  # 对应机械臂初始位姿[-0.13, 0.3, 0.1, 3.14, 0, 1.57]
    data.qpos[:6] = start_joints  # 确保渲染一开始机械臂便处于起始位置，而非MJCF中的默认位置

    # 设置目标点
    ee_pos = [-0.13, 0.6, 0.1]
    ee_euler = [3.14, 0, 1.57]
    ref_pos = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0]
    ee_orientation = tf.euler.euler2mat(*ee_euler)

    joint_angles = my_chain.inverse_kinematics(ee_pos, ee_orientation, "all", initial_position=ref_pos)
    end_joints = joint_angles[2:-1]

    joint_trajectory = JointSpaceTrajectory(start_joints, end_joints, steps=100)

    force_sensor = ForceSensor(model, data)
    force_plotter = ForcePlotter()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer_init(viewer)
        while viewer.is_running():
            waypoint = joint_trajectory.get_next_waypoint(data.qpos[:6])
            data.ctrl[:6] = waypoint

            filtered_force = force_sensor.filter()
            force_plotter.plot_force_vector(filtered_force)

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()