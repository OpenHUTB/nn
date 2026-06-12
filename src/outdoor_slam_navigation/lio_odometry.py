# lio_odometry.py
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from collections import deque
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')


class LIOdometry:
    """LiDAR-IMU紧耦合里程计"""

    def __init__(self, config=None):
        self.config = config or self._default_config()

        # 状态变量
        self.position = np.zeros(3)  # 位置 (x, y, z)
        self.velocity = np.zeros(3)  # 速度
        self.orientation = np.eye(3)  # 旋转矩阵
        self.gravity = np.array([0, 0, -9.81])  # 重力向量

        # 偏差
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

        # 缓冲区
        self.imu_buffer = deque(maxlen=100)
        self.last_lidar_time = None

        # 协方差矩阵
        self.covariance = np.eye(15) * 0.1

        # 预积分
        self.delta_p = np.zeros(3)  # 位置增量
        self.delta_v = np.zeros(3)  # 速度增量
        self.delta_r = np.eye(3)  # 旋转增量

        # === 新增：可视化相关变量 ===
        self.trajectory = []  # 轨迹历史
        self.features_history = []  # 特征点历史
        self.max_history = 1000  # 最大历史记录数
        # ==========================

    def _default_config(self):
        return {
            "imu_noise": {
                "gyro": 0.01,  # rad/s
                "accel": 0.1,  # m/s²
                "gyro_bias": 1e-5,
                "accel_bias": 1e-4
            },
            "lidar_noise": {
                "rotation": 0.001,  # rad
                "translation": 0.01  # m
            },
            "mapping": {
                "voxel_size": 0.2,
                "max_range": 50.0,
                "min_range": 1.0
            }
        }

    def preintegrate_imu(self, imu_data_list):
        """IMU预积分"""
        if not imu_data_list:
            return

        # 重置预积分量
        self.delta_p = np.zeros(3)
        self.delta_v = np.zeros(3)
        self.delta_r = np.eye(3)

        last_time = imu_data_list[0].timestamp

        for i in range(1, len(imu_data_list)):
            current_imu = imu_data_list[i]
            dt = current_imu.timestamp - last_time

            if dt <= 0:
                continue

            # 去除偏差后的测量值
            accel = current_imu.accel - self.accel_bias
            gyro = current_imu.gyro - self.gyro_bias

            # 中值积分
            accel_0 = imu_data_list[i - 1].accel - self.accel_bias
            gyro_0 = imu_data_list[i - 1].gyro - self.gyro_bias

            accel_hat = 0.5 * (accel_0 + accel)
            gyro_hat = 0.5 * (gyro_0 + gyro)

            # 旋转更新
            delta_theta = gyro_hat * dt
            delta_rotation = R.from_rotvec(delta_theta).as_matrix()
            self.delta_r = self.delta_r @ delta_rotation

            # 速度更新
            accel_world = self.delta_r @ accel_hat
            self.delta_v += (accel_world + self.gravity) * dt

            # 位置更新
            self.delta_p += self.delta_v * dt + 0.5 * (accel_world + self.gravity) * dt * dt

            last_time = current_imu.timestamp

    def extract_features(self, point_cloud):
        """从点云中提取几何特征"""
        if point_cloud.shape[0] < 100:
            return [], []

        # 地面分割
        ground_points, non_ground = self._ground_segmentation(point_cloud)

        # 提取平面特征（来自地面点）
        plane_features = self._extract_plane_features(ground_points)

        # 提取边缘特征（来自非地面点）
        edge_features = self._extract_edge_features(non_ground)

        return plane_features, edge_features

    def _ground_segmentation(self, points):
        """简单的地面分割"""
        if points.shape[0] == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))

        # 基于高度的简单分割
        z_values = points[:, 2]
        ground_mask = z_values < 0.2  # 假设地面高度阈值
        ground_points = points[ground_mask]
        non_ground = points[~ground_mask]

        return ground_points, non_ground

    def _extract_plane_features(self, points, num_features=20):
        """提取平面特征点"""
        if points.shape[0] < num_features:
            return points

        # 使用曲率选择最平坦的点
        curvatures = self._compute_curvature(points)
        flat_indices = np.argsort(curvatures)[:num_features]

        return points[flat_indices]

    def _extract_edge_features(self, points, num_features=20):
        """提取边缘特征点"""
        if points.shape[0] < num_features:
            return points

        # 使用曲率选择最尖锐的点
        curvatures = self._compute_curvature(points)
        edge_indices = np.argsort(curvatures)[-num_features:]

        return points[edge_indices]

    def _compute_curvature(self, points, k=5):
        """计算点云曲率"""
        from sklearn.neighbors import NearestNeighbors

        if points.shape[0] < k:
            return np.zeros(points.shape[0])

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)

        curvatures = []
        for i in range(points.shape[0]):
            neighbor_points = points[indices[i][1:]]  # 排除自身
            centered = neighbor_points - neighbor_points.mean(axis=0)
            cov = centered.T @ centered
            eigenvalues = np.linalg.eigvalsh(cov)
            curvature = eigenvalues[0] / (eigenvalues.sum() + 1e-6)
            curvatures.append(curvature)

        return np.array(curvatures)

    def scan_to_map_matching(self, current_features, map_features):
        """当前扫描与地图匹配"""
        if len(current_features) == 0 or len(map_features) == 0:
            return np.eye(4)

        # 使用ICP进行配准
        transformation = self._icp_registration(current_features, map_features)

        return transformation

    def _icp_registration(self, source, target, max_iterations=20):
        """迭代最近点算法"""
        if len(source) < 3 or len(target) < 3:
            return np.eye(4)

        # 构建KD树加速最近邻搜索
        tree = KDTree(target)

        # 初始变换矩阵
        transformation = np.eye(4)

        for iteration in range(max_iterations):
            # 找到最近邻对应点
            distances, indices = tree.query(source)

            # 过滤掉距离太远的点对
            valid_mask = distances < 1.0  # 1米阈值
            if np.sum(valid_mask) < 3:
                break

            source_valid = source[valid_mask]
            target_valid = target[indices[valid_mask]]

            # 计算最优刚体变换
            R_mat, t_vec = self._compute_rigid_transform(source_valid, target_valid)

            # 更新变换
            delta_transform = np.eye(4)
            delta_transform[:3, :3] = R_mat
            delta_transform[:3, 3] = t_vec

            transformation = delta_transform @ transformation

            # 更新源点云
            source = (R_mat @ source.T + t_vec.reshape(3, 1)).T

            # 检查收敛
            if np.linalg.norm(t_vec) < 0.001 and np.linalg.norm(R_mat - np.eye(3)) < 0.001:
                break

        return transformation

    def _compute_rigid_transform(self, A, B):
        """计算点集A到点集B的最优刚体变换"""
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        AA = A - centroid_A
        BB = B - centroid_B

        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)

        R = Vt.T @ U.T

        # 处理反射情况
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_B - R @ centroid_A

        return R, t

    def process_frame(self, point_cloud, imu_data_list, gnss_data=None):
        """处理一帧数据"""
        # IMU预积分
        self.preintegrate_imu(imu_data_list)

        # 提取特征
        plane_features, edge_features = self.extract_features(point_cloud)
        all_features = np.vstack([plane_features, edge_features]) if len(plane_features) > 0 else edge_features

        # 运动估计（这里简化为使用IMU预积分）
        # 在实际系统中，这里会进行扫描匹配
        delta_pose = np.eye(4)
        delta_pose[:3, :3] = self.delta_r
        delta_pose[:3, 3] = self.delta_p

        # 更新状态
        self.orientation = self.orientation @ self.delta_r
        self.position += self.orientation @ self.delta_p
        self.velocity += self.orientation @ self.delta_v

        # 如果有GNSS数据，可以在这里融合
        if gnss_data:
            self._fuse_gnss(gnss_data)

        # 构建当前位姿
        pose = np.eye(4)
        pose[:3, :3] = self.orientation
        pose[:3, 3] = self.position

        # === 新增：记录轨迹和特征用于可视化 ===
        self._record_for_visualization(pose, all_features)
        # =================================

        return pose, all_features

    def _fuse_gnss(self, gnss_data):
        """融合GNSS数据（简化版本）"""
        # 将经纬度转换为局部坐标（需要知道原点）
        # 这里简化为直接信任GNSS（在实际系统中应使用卡尔曼滤波）
        local_pos = self._llh_to_local(gnss_data.lat, gnss_data.lon, gnss_data.alt)

        # 简单的加权融合
        gnss_weight = 0.1  # 信任度
        self.position = (1 - gnss_weight) * self.position + gnss_weight * local_pos

    def _llh_to_local(self, lat, lon, alt):
        """经纬度转局部坐标（简化版）"""
        # 实际实现中需要知道原点的经纬度
        return np.array([lon * 111320.0, lat * 110540.0, alt])  # 近似转换

    # =============================================
    # === 以下是新增的可视化方法，原功能完全不变 ===
    # =============================================

    def _record_for_visualization(self, pose, features):
        """记录数据用于可视化（内部方法）"""
        # 记录轨迹
        self.trajectory.append(pose[:3, 3].copy())

        # 记录特征点
        if isinstance(features, np.ndarray) and features.shape[0] > 0:
            self.features_history.append(features.copy())
        else:
            self.features_history.append(np.zeros((0, 3)))

        # 限制历史数据长度
        if len(self.trajectory) > self.max_history:
            self.trajectory = self.trajectory[-self.max_history:]
        if len(self.features_history) > self.max_history:
            self.features_history = self.features_history[-self.max_history:]

    def show_trajectory(self):
        """显示轨迹（2D和3D）"""
        if not self.trajectory:
            print("No trajectory data to visualize")
            return

        trajectory_array = np.array(self.trajectory)

        fig = plt.figure(figsize=(15, 5))

        # 1. 3D轨迹
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2],
                 'b-', linewidth=2, label='Trajectory')
        if len(trajectory_array) > 1:
            ax1.scatter(trajectory_array[0, 0], trajectory_array[0, 1], trajectory_array[0, 2],
                        c='g', s=100, marker='o', label='Start')
            ax1.scatter(trajectory_array[-1, 0], trajectory_array[-1, 1], trajectory_array[-1, 2],
                        c='r', s=100, marker='s', label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        ax1.grid(True)

        # 2. 2D轨迹（XY平面）
        ax2 = fig.add_subplot(132)
        ax2.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b-', linewidth=2)
        if len(trajectory_array) > 1:
            ax2.scatter(trajectory_array[0, 0], trajectory_array[0, 1], c='g', s=100)
            ax2.scatter(trajectory_array[-1, 0], trajectory_array[-1, 1], c='r', s=100)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('2D Trajectory (XY Plane)')
        ax2.axis('equal')
        ax2.grid(True)

        # 3. 高程变化
        ax3 = fig.add_subplot(133)
        if len(trajectory_array) > 1:
            distances = np.cumsum(np.sqrt(np.sum(np.diff(trajectory_array[:, :2], axis=0) ** 2, axis=1)))
            distances = np.insert(distances, 0, 0)
            ax3.plot(distances, trajectory_array[:, 2], 'b-', linewidth=2)
        ax3.set_xlabel('Distance (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Elevation Profile')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()
        return fig

    def show_features(self, point_cloud=None):
        """显示特征点和点云"""
        if not self.features_history:
            print("No feature data to visualize")
            return

        recent_features = self.features_history[-1]

        fig = plt.figure(figsize=(15, 5))

        # 1. 3D特征点
        ax1 = fig.add_subplot(131, projection='3d')
        if recent_features.shape[0] > 0:
            # 转换特征点到世界坐标系
            if hasattr(self, 'orientation') and hasattr(self, 'position'):
                features_homogeneous = np.hstack([recent_features, np.ones((len(recent_features), 1))])
                current_pose = np.eye(4)
                current_pose[:3, :3] = self.orientation
                current_pose[:3, 3] = self.position
                world_features = (current_pose @ features_homogeneous.T).T[:, :3]

                ax1.scatter(world_features[:, 0], world_features[:, 1], world_features[:, 2],
                            c='red', s=20, label='Features')

            ax1.scatter(self.position[0], self.position[1], self.position[2],
                        c='blue', s=100, marker='s', label='Current Position')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Feature Points in World Frame')
        ax1.legend()
        ax1.grid(True)

        # 2. 特征点统计
        ax2 = fig.add_subplot(132)
        feature_counts = [len(f) for f in self.features_history[-20:] if f.shape[0] > 0]
        if feature_counts:
            ax2.plot(range(len(feature_counts)), feature_counts, 'b-o', linewidth=2, markersize=6)
            ax2.set_xlabel('Recent Frames')
            ax2.set_ylabel('Number of Features')
            ax2.set_title('Feature Count History')
            ax2.grid(True)

        # 3. 点云（如果提供）
        ax3 = fig.add_subplot(133, projection='3d')
        if point_cloud is not None and point_cloud.shape[0] > 0:
            # 随机采样显示
            if point_cloud.shape[0] > 1000:
                indices = np.random.choice(point_cloud.shape[0], 1000, replace=False)
                point_cloud = point_cloud[indices]

            ax3.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                        c='gray', s=1, alpha=0.3, label='Point Cloud')

            if recent_features.shape[0] > 0:
                ax3.scatter(recent_features[:, 0], recent_features[:, 1], recent_features[:, 2],
                            c='red', s=20, label='Features')

            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            ax3.set_title('Latest Point Cloud & Features')
            ax3.legend()

        plt.tight_layout()
        plt.show()
        return fig

    def show_comprehensive_analysis(self):
        """显示综合分析图表"""
        if not self.trajectory:
            print("No data for analysis")
            return

        trajectory_array = np.array(self.trajectory)

        fig = plt.figure(figsize=(16, 12))

        # 1. 3D轨迹
        ax1 = fig.add_subplot(331, projection='3d')
        ax1.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2],
                 'b-', linewidth=2)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.grid(True)

        # 2. 2D轨迹
        ax2 = fig.add_subplot(332)
        ax2.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b-', linewidth=2)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('2D Trajectory')
        ax2.axis('equal')
        ax2.grid(True)

        # 3. 位置分量
        ax3 = fig.add_subplot(333)
        frames = range(len(trajectory_array))
        ax3.plot(frames, trajectory_array[:, 0], 'r-', label='X', linewidth=1)
        ax3.plot(frames, trajectory_array[:, 1], 'g-', label='Y', linewidth=1)
        ax3.plot(frames, trajectory_array[:, 2], 'b-', label='Z', linewidth=1)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Position (m)')
        ax3.set_title('Position Components')
        ax3.legend()
        ax3.grid(True)

        # 4. 速度信息
        ax4 = fig.add_subplot(334)
        try:
            velocity_norm = np.linalg.norm(self.velocity)
            ax4.bar(['Speed'], [velocity_norm], color='skyblue')
            ax4.set_ylabel('m/s')
            ax4.set_title(f'Current Speed: {velocity_norm:.2f} m/s')
            ax4.grid(True, alpha=0.3)
        except:
            pass

        # 5. 特征点统计
        ax5 = fig.add_subplot(335)
        if self.features_history:
            feature_counts = [len(f) for f in self.features_history[-50:] if f.shape[0] > 0]
            if feature_counts:
                ax5.plot(range(len(feature_counts)), feature_counts, 'purple', linewidth=2)
                ax5.set_xlabel('Recent Frames')
                ax5.set_ylabel('Feature Count')
                ax5.set_title('Feature Extraction History')
                ax5.grid(True)

        # 6. 轨迹长度
        ax6 = fig.add_subplot(336)
        if len(trajectory_array) > 1:
            total_distance = np.sum(np.sqrt(np.sum(np.diff(trajectory_array, axis=0) ** 2, axis=1)))
            ax6.text(0.5, 0.5, f'Total Distance:\n{total_distance:.2f} m',
                     ha='center', va='center', fontsize=14, transform=ax6.transAxes)
            ax6.set_title('Trajectory Statistics')
            ax6.axis('off')

        # 7. 最新特征点
        ax7 = fig.add_subplot(337, projection='3d')
        if self.features_history and self.features_history[-1].shape[0] > 0:
            recent_features = self.features_history[-1]
            ax7.scatter(recent_features[:, 0], recent_features[:, 1], recent_features[:, 2],
                        c='red', s=10, alpha=0.6)
            ax7.set_xlabel('X')
            ax7.set_ylabel('Y')
            ax7.set_zlabel('Z')
            ax7.set_title('Latest Features (Local)')

        # 8. 状态信息
        ax8 = fig.add_subplot(338)
        info_text = f"Frames: {len(self.trajectory)}\n"
        info_text += f"Position: ({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})\n"
        info_text += f"Velocity: {np.linalg.norm(self.velocity):.2f} m/s"
        ax8.text(0.1, 0.5, info_text, ha='left', va='center', fontsize=12,
                 transform=ax8.transAxes, family='monospace')
        ax8.set_title('Current State')
        ax8.axis('off')

        # 9. 空图表（预留）
        ax9 = fig.add_subplot(339)
        ax9.text(0.5, 0.5, 'LIO Odometry\nVisualization',
                 ha='center', va='center', fontsize=16, transform=ax9.transAxes)
        ax9.axis('off')

        plt.tight_layout()
        plt.show()
        return fig

    def export_trajectory(self, filename="trajectory.txt", format="kitti"):
        """导出轨迹到文件"""
        if not self.trajectory:
            print("No trajectory to export")
            return

        import os
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

        with open(filename, 'w') as f:
            if format.lower() == "kitti":
                # KITTI格式
                for pos in self.trajectory:
                    # 使用当前方向或单位矩阵
                    rot_matrix = self.orientation if hasattr(self, 'orientation') else np.eye(3)
                    line = ' '.join([f'{val:.6f}' for val in rot_matrix.flatten()])
                    line += f' {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}'
                    f.write(line + '\n')
                print(f"Trajectory exported in KITTI format to {filename}")

            elif format.lower() == "tum":
                # TUM格式
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                for i, pos in enumerate(self.trajectory):
                    timestamp = i * 0.1  # 假设时间间隔
                    rot = self.orientation if hasattr(self, 'orientation') else np.eye(3)
                    rotation = R.from_matrix(rot)
                    quat = rotation.as_quat()  # [x, y, z, w]
                    line = f"{timestamp:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                    line += f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
                    f.write(line + '\n')
                print(f"Trajectory exported in TUM format to {filename}")
            else:
                print(f"Unknown format: {format}")

    def clear_visualization_data(self):
        """清除可视化数据"""
        self.trajectory.clear()
        self.features_history.clear()


# =============================================
# === 以下是为了测试而添加的模拟代码 ===
# =============================================

class MockIMUData:
    """模拟IMU数据类（仅用于测试）"""

    def __init__(self, timestamp, accel, gyro):
        self.timestamp = timestamp
        self.accel = accel
        self.gyro = gyro


class MockGNSSData:
    """模拟GNSS数据类（仅用于测试）"""

    def __init__(self, lat, lon, alt):
        self.lat = lat
        self.lon = lon
        self.alt = alt


def test_demo():
    """测试演示函数（可以直接在PyCharm中运行）"""
    print("Starting LIO Odometry Visualization Demo...")

    # 创建LIOdometry实例
    odom = LIOdometry()

    # 模拟运行10帧
    for i in range(10):
        # 生成模拟点云
        n_points = 500 + np.random.randint(0, 500)
        point_cloud = np.random.randn(n_points, 3) * 3.0
        point_cloud[:, 2] = np.abs(point_cloud[:, 2]) * 0.5

        # 生成模拟IMU数据（轻微运动）
        imu_data = [
            MockIMUData(0.0, np.array([0, 0, -9.81]), np.array([0, 0, 0])),
            MockIMUData(0.01, np.array([0.1 * (i / 10), 0, -9.81]), np.array([0, 0, 0.05 * (i / 10)]))
        ]

        # 处理帧（原有功能）
        pose, features = odom.process_frame(point_cloud, imu_data)

        print(f"Frame {i + 1}: Position = [{pose[0, 3]:.2f}, {pose[1, 3]:.2f}, {pose[2, 3]:.2f}], "
              f"Features = {features.shape[0]}")

    print("\nGenerating visualizations...")

    # 显示轨迹
    odom.show_trajectory()

    # 显示特征点
    # 注意：这里需要最新的点云，我们重新生成一个
    test_point_cloud = np.random.randn(300, 3) * 2.0
    odom.show_features(test_point_cloud)

    # 显示综合分析
    odom.show_comprehensive_analysis()

    # 导出轨迹
    odom.export_trajectory("test_trajectory_kitti.txt", "kitti")
    odom.export_trajectory("test_trajectory_tum.txt", "tum")

    print("\nDemo completed!")
    print("Close the matplotlib windows to exit.")


# 主程序入口
if __name__ == "__main__":
    # 自动运行测试演示
    test_demo()