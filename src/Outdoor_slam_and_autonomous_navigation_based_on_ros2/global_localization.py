# global_localization.py
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button


class GlobalLocalization:
    """全局定位模块"""

    def __init__(self, map_data=None, visualize=True):
        self.map_data = map_data
        self.map_points = None
        self.map_tree = None
        self.map_features = None
        self.initialized = False

        # 定位状态
        self.current_pose = np.eye(4)
        self.confidence = 0.0
        self.localization_mode = "local"  # "local" or "global"

        # 扫描上下文描述子
        self.scan_contexts = {}

        # 可视化相关
        self.visualize = visualize
        self.fig = None
        self.ax = None
        self.history_poses = []
        self.history_confidences = []
        self.scan_history = []

        # 颜色设置
        self.colors = {
            'map': (0.3, 0.3, 0.3, 0.3),  # 灰色半透明
            'current_scan': (1.0, 0.0, 0.0, 0.5),  # 红色
            'matched_scan': (0.0, 1.0, 0.0, 0.5),  # 绿色
            'trajectory': (0.0, 0.0, 1.0, 1.0),  # 蓝色
            'candidates': (1.0, 1.0, 0.0, 0.3),  # 黄色
        }

    def initialize_visualization(self):
        """初始化可视化窗口"""
        if not self.visualize:
            return

        plt.ion()  # 开启交互模式

        self.fig = plt.figure(figsize=(15, 10))

        # 创建3D点云视图
        self.ax1 = self.fig.add_subplot(231, projection='3d')
        self.ax1.set_title('3D Point Cloud View')
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_zlabel('Z (m)')

        # 创建2D俯视图
        self.ax2 = self.fig.add_subplot(232)
        self.ax2.set_title('2D Top View')
        self.ax2.set_xlabel('X (m)')
        self.ax2.set_ylabel('Y (m)')
        self.ax2.set_aspect('equal')

        # 创建轨迹视图
        self.ax3 = self.fig.add_subplot(233)
        self.ax3.set_title('Trajectory and Confidence')
        self.ax3.set_xlabel('Step')
        self.ax3.set_ylabel('Confidence')

        # 创建扫描上下文视图
        self.ax4 = self.fig.add_subplot(234)
        self.ax4.set_title('Scan Context Descriptor')
        self.ax4.set_xlabel('Sector')
        self.ax4.set_ylabel('Ring')

        # 创建位姿误差视图
        self.ax5 = self.fig.add_subplot(235)
        self.ax5.set_title('Pose Error')
        self.ax5.set_xlabel('Step')
        self.ax5.set_ylabel('Error (m/rad)')

        # 创建ICP匹配视图
        self.ax6 = self.fig.add_subplot(236)
        self.ax6.set_title('ICP Matching')
        self.ax6.set_xlabel('X (m)')
        self.ax6.set_ylabel('Y (m)')
        self.ax6.set_aspect('equal')

        plt.tight_layout()
        plt.show(block=False)

    def load_map(self, map_points, map_features=None):
        """加载全局地图"""
        self.map_points = map_points
        self.map_features = map_features

        if map_points is not None and len(map_points) > 0:
            self.map_tree = KDTree(map_points[:, :3])  # 只使用xyz
            self.initialized = True
            print(f"地图已加载，包含 {len(map_points)} 个点")

            # 可视化地图
            if self.visualize:
                if self.fig is None:
                    self.initialize_visualization()
                self._visualize_map()
        else:
            print("警告：地图点云为空")

    def _visualize_map(self):
        """可视化地图点云"""
        if not self.visualize or self.map_points is None:
            return

        # 3D视图
        self.ax1.clear()
        if len(self.map_points) > 10000:  # 太多点的话采样显示
            indices = np.random.choice(len(self.map_points), 10000, replace=False)
            display_points = self.map_points[indices]
        else:
            display_points = self.map_points

        self.ax1.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
                         c=[self.colors['map']], s=1, marker='.')

        # 2D俯视图
        self.ax2.clear()
        self.ax2.scatter(display_points[:, 0], display_points[:, 1],
                         c=[self.colors['map'][:3]], s=1, marker='.', alpha=0.3)

        self.ax1.set_title(f'Map Points: {len(self.map_points)}')
        self.fig.canvas.draw_idle()

    def _visualize_localization(self, scan_points, pose, confidence,
                                matched_points=None, candidates=None):
        """可视化定位结果"""
        if not self.visualize:
            return

        # 变换当前扫描点
        transformed_scan = self._transform_points(scan_points, pose)

        # 更新历史记录
        self.history_poses.append(pose)
        self.history_confidences.append(confidence)
        self.scan_history.append(transformed_scan)

        # 限制历史记录长度
        max_history = 50
        if len(self.history_poses) > max_history:
            self.history_poses = self.history_poses[-max_history:]
            self.history_confidences = self.history_confidences[-max_history:]
            self.scan_history = self.scan_history[-max_history:]

        # 1. 3D点云视图
        self.ax1.clear()
        if len(self.map_points) > 10000:
            indices = np.random.choice(len(self.map_points), 10000, replace=False)
            display_map = self.map_points[indices]
        else:
            display_map = self.map_points

        self.ax1.scatter(display_map[:, 0], display_map[:, 1], display_map[:, 2],
                         c=[self.colors['map']], s=1, marker='.', alpha=0.3, label='Map')

        # 当前扫描
        self.ax1.scatter(transformed_scan[:, 0], transformed_scan[:, 1], transformed_scan[:, 2],
                         c='r', s=5, marker='o', alpha=0.5, label='Current Scan')

        # 轨迹
        if len(self.history_poses) > 1:
            trajectory = np.array([p[:3, 3] for p in self.history_poses])
            self.ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                          c=self.colors['trajectory'], linewidth=2, label='Trajectory')

        self.ax1.set_title(f'3D View - Confidence: {confidence:.3f}')
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_zlabel('Z (m)')
        self.ax1.legend()

        # 2. 2D俯视图
        self.ax2.clear()
        self.ax2.scatter(display_map[:, 0], display_map[:, 1],
                         c=[self.colors['map'][:3]], s=1, marker='.', alpha=0.3)

        self.ax2.scatter(transformed_scan[:, 0], transformed_scan[:, 1],
                         c='r', s=10, marker='o', alpha=0.6, label='Current Scan')

        if matched_points is not None:
            self.ax2.scatter(matched_points[:, 0], matched_points[:, 1],
                             c='g', s=20, marker='x', alpha=0.8, label='Matched Points')

        # 轨迹
        if len(self.history_poses) > 1:
            trajectory = np.array([p[:3, 3] for p in self.history_poses])
            self.ax2.plot(trajectory[:, 0], trajectory[:, 1],
                          c=self.colors['trajectory'], linewidth=2, label='Trajectory')

            # 显示方向箭头
            dx, dy = 2 * np.cos(pose[2, 0]), 2 * np.sin(pose[2, 0])
            self.ax2.arrow(pose[0, 3], pose[1, 3], dx, dy,
                           head_width=0.5, head_length=0.5, fc='blue', ec='blue')

        # 候选位姿
        if candidates is not None:
            for i, candidate in enumerate(candidates[:5]):  # 只显示前5个
                self.ax2.scatter(candidate[0, 3], candidate[1, 3],
                                 c='y', s=50, marker='*', alpha=0.5)

        self.ax2.set_title(f'2D Top View - Pose: [{pose[0, 3]:.1f}, {pose[1, 3]:.1f}]')
        self.ax2.set_xlabel('X (m)')
        self.ax2.set_ylabel('Y (m)')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)

        # 3. 轨迹和置信度
        self.ax3.clear()
        if self.history_confidences:
            steps = range(len(self.history_confidences))
            self.ax3.plot(steps, self.history_confidences, 'b-', linewidth=2, label='Confidence')
            self.ax3.fill_between(steps, 0, self.history_confidences, alpha=0.3)
            self.ax3.set_ylim(0, 1.1)
            self.ax3.set_xlabel('Step')
            self.ax3.set_ylabel('Confidence')
            self.ax3.set_title('Localization Confidence History')
            self.ax3.grid(True, alpha=0.3)
            self.ax3.legend()

        # 4. 扫描上下文
        self.ax4.clear()
        current_context = self.create_scan_context(scan_points)
        im = self.ax4.imshow(current_context, aspect='auto', cmap='viridis')
        self.ax4.set_title('Current Scan Context')
        self.ax4.set_xlabel('Sector')
        self.ax4.set_ylabel('Ring')
        plt.colorbar(im, ax=self.ax4)

        # 5. 位姿误差（如果有真值的话）
        self.ax5.clear()
        if len(self.history_poses) > 1:
            positions = np.array([p[:3, 3] for p in self.history_poses])
            position_changes = np.diff(positions, axis=0)
            position_norms = np.linalg.norm(position_changes, axis=1)

            steps = range(len(position_norms))
            self.ax5.plot(steps, position_norms, 'r-', label='Position Change (m)')

            if len(steps) > 0:
                self.ax5.axhline(y=np.mean(position_norms), color='r', linestyle='--',
                                 alpha=0.5, label=f'Avg: {np.mean(position_norms):.3f}m')

            self.ax5.set_xlabel('Step')
            self.ax5.set_ylabel('Error / Change')
            self.ax5.set_title('Pose Changes')
            self.ax5.grid(True, alpha=0.3)
            self.ax5.legend()

        # 6. ICP匹配视图
        self.ax6.clear()
        if matched_points is not None and len(matched_points) > 0:
            # 显示匹配点对
            for i in range(min(20, len(transformed_scan))):  # 只显示20个匹配对
                scan_pt = transformed_scan[i, :2]
                map_pt = matched_points[i, :2]
                self.ax6.plot([scan_pt[0], map_pt[0]], [scan_pt[1], map_pt[1]],
                              'g-', alpha=0.3, linewidth=0.5)

            self.ax6.scatter(matched_points[:, 0], matched_points[:, 1],
                             c='g', s=20, marker='x', alpha=0.8, label='Map Points')

        self.ax6.scatter(transformed_scan[:, 0], transformed_scan[:, 1],
                         c='r', s=10, marker='o', alpha=0.6, label='Scan Points')

        self.ax6.set_title(f'ICP Matching - Inliers: {len(matched_points) if matched_points is not None else 0}')
        self.ax6.set_xlabel('X (m)')
        self.ax6.set_ylabel('Y (m)')
        self.ax6.legend()
        self.ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        self.fig.canvas.draw_idle()
        plt.pause(0.01)  # 短暂暂停以更新显示

    def localize_with_gnss(self, gnss_position, initial_guess=None):
        """使用GNSS进行初始全局定位"""
        if not self.initialized or self.map_points is None:
            return np.eye(4), 0.0

        # 将GNSS坐标转换为局部坐标
        # 这里需要知道地图的GNSS参考点
        local_position = gnss_position  # 简化：假设已经是局部坐标

        # 在地图中寻找最近的点
        distances, indices = self.map_tree.query(local_position.reshape(1, -1), k=10)

        # 计算平均位置作为初始位姿
        if len(indices) > 0:
            nearest_points = self.map_points[indices[0], :3]
            estimated_position = np.mean(nearest_points, axis=0)

            # 构建位姿（假设水平）
            pose = np.eye(4)
            pose[:3, 3] = estimated_position

            confidence = 1.0 / (1.0 + np.mean(distances))

            # 可视化
            if self.visualize:
                self._visualize_localization(
                    scan_points=np.array([[local_position[0], local_position[1], 0]]),
                    pose=pose,
                    confidence=confidence,
                    matched_points=nearest_points
                )

            return pose, confidence

        return np.eye(4), 0.0

    def localize_with_scan_matching(self, scan_points, initial_guess=None, visualize=True):
        """使用扫描匹配进行定位"""
        if not self.initialized or self.map_points is None:
            return np.eye(4), 0.0

        if initial_guess is None:
            initial_guess = np.eye(4)

        # 使用ICP进行精配准
        refined_pose, matched_points = self._icp_localization(scan_points, initial_guess)

        # 计算匹配分数
        confidence = self._compute_match_confidence(scan_points, refined_pose)

        # 可视化
        if self.visualize and visualize:
            self._visualize_localization(
                scan_points=scan_points,
                pose=refined_pose,
                confidence=confidence,
                matched_points=matched_points
            )

        return refined_pose, confidence

    def _icp_localization(self, scan_points, initial_guess, max_iterations=30):
        """ICP定位"""
        pose = initial_guess.copy()
        all_matched_points = None

        for iteration in range(max_iterations):
            # 变换扫描点
            transformed_points = self._transform_points(scan_points, pose)

            # 寻找最近邻
            distances, indices = self.map_tree.query(transformed_points[:, :3])

            # 过滤掉距离太远的点
            valid_mask = distances < 2.0  # 2米阈值
            if np.sum(valid_mask) < 10:  # 至少需要10个有效点
                break

            valid_scan = scan_points[valid_mask]
            valid_map = self.map_points[indices[valid_mask], :3]

            # 保存匹配点用于可视化
            if iteration == max_iterations - 1:  # 最后一次迭代
                all_matched_points = valid_map

            # 计算最优变换
            R_mat, t_vec = self._compute_rigid_transform(valid_scan[:, :3], valid_map)

            # 更新位姿
            delta_pose = np.eye(4)
            delta_pose[:3, :3] = R_mat
            delta_pose[:3, 3] = t_vec

            pose = delta_pose @ pose

            # 检查收敛
            if np.linalg.norm(t_vec) < 0.001 and np.linalg.norm(R_mat - np.eye(3)) < 0.001:
                break

        return pose, all_matched_points

    def _transform_points(self, points, pose):
        """变换点云"""
        transformed = (pose[:3, :3] @ points[:, :3].T + pose[:3, 3:4]).T

        if points.shape[1] > 3:
            transformed = np.hstack([transformed, points[:, 3:]])

        return transformed

    def _compute_rigid_transform(self, A, B):
        """计算刚体变换"""
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        AA = A - centroid_A
        BB = B - centroid_B

        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)

        R_mat = Vt.T @ U.T

        # 处理反射情况
        if np.linalg.det(R_mat) < 0:
            Vt[-1, :] *= -1
            R_mat = Vt.T @ U.T

        t_vec = centroid_B - R_mat @ centroid_A

        return R_mat, t_vec

    def _compute_match_confidence(self, scan_points, pose):
        """计算匹配置信度"""
        transformed_points = self._transform_points(scan_points, pose)

        distances, _ = self.map_tree.query(transformed_points[:, :3], k=1)

        # 计算内点比例
        inlier_ratio = np.mean(distances < 0.5)  # 0.5米内点阈值

        # 综合置信度
        confidence = min(1.0, inlier_ratio * 2.0)

        return confidence

    def create_scan_context(self, scan_points):
        """创建扫描上下文描述子"""
        num_sectors = 60
        num_rings = 20
        max_range = 50.0

        descriptor = np.zeros((num_rings, num_sectors))

        for point in scan_points:
            x, y, z = point[:3]

            # 转换为极坐标
            radius = np.sqrt(x ** 2 + y ** 2)
            angle = np.arctan2(y, x)

            if radius > max_range:
                continue

            # 计算环和扇区索引
            ring_idx = min(int(radius / max_range * num_rings), num_rings - 1)
            sector_idx = int((angle + np.pi) / (2 * np.pi) * num_sectors) % num_sectors

            # 存储最大高度
            height = z
            descriptor[ring_idx, sector_idx] = max(descriptor[ring_idx, sector_idx], height)

        return descriptor

    def scan_context_match(self, current_descriptor):
        """扫描上下文匹配"""
        if not self.scan_contexts:
            return None, 0.0

        best_match = None
        best_score = 0.0

        for node_id, descriptor in self.scan_contexts.items():
            # 计算描述子相似度
            score = self._descriptor_similarity(current_descriptor, descriptor)

            if score > best_score:
                best_score = score
                best_match = node_id

        return best_match, best_score

    def _descriptor_similarity(self, desc1, desc2):
        """计算描述子相似度"""
        # 列向量的余弦相似度
        similarity = 0.0

        for i in range(desc1.shape[1]):
            col1 = desc1[:, i]
            col2 = desc2[:, i]

            norm1 = np.linalg.norm(col1)
            norm2 = np.linalg.norm(col2)

            if norm1 > 0 and norm2 > 0:
                similarity += np.dot(col1, col2) / (norm1 * norm2)

        return similarity / desc1.shape[1]

    def update_scan_context(self, node_id, descriptor):
        """更新扫描上下文数据库"""
        self.scan_contexts[node_id] = descriptor

    def global_relocalization(self, scan_points, gnss_hint=None):
        """全局重定位"""
        if not self.initialized:
            return np.eye(4), 0.0

        # 方法1：使用扫描上下文
        current_descriptor = self.create_scan_context(scan_points)
        matched_node, context_score = self.scan_context_match(current_descriptor)

        if context_score > 0.7:  # 高置信度匹配
            print(f"扫描上下文匹配成功，分数: {context_score}")
            # 可以返回匹配节点的位姿
            return np.eye(4), context_score

        # 方法2：使用GNSS提示
        if gnss_hint is not None:
            pose, confidence = self.localize_with_gnss(gnss_hint)
            if confidence > 0.5:
                print(f"GNSS提示定位成功，置信度: {confidence}")
                return pose, confidence

        # 方法3：全局ICP（计算量大，作为最后手段）
        print("尝试全局ICP重定位...")

        # 在地图中采样一些候选位姿
        candidates = self._generate_candidates(scan_points)

        best_pose = np.eye(4)
        best_score = 0.0
        best_matched_points = None

        for candidate in candidates:
            pose, matched_points = self._icp_localization(scan_points, candidate, max_iterations=10)
            score = self._compute_match_confidence(scan_points, pose)

            if score > best_score:
                best_score = score
                best_pose = pose
                best_matched_points = matched_points

        # 可视化候选位姿和最佳匹配
        if self.visualize:
            self._visualize_localization(
                scan_points=scan_points,
                pose=best_pose,
                confidence=best_score,
                matched_points=best_matched_points,
                candidates=candidates
            )

        if best_score > 0.3:
            print(f"全局ICP重定位成功，分数: {best_score}")
            return best_pose, best_score

        print("重定位失败")
        return np.eye(4), 0.0

    def _generate_candidates(self, scan_points, num_candidates=20):
        """生成候选位姿"""
        candidates = []

        # 在地图边界内随机采样
        if self.map_points is not None and len(self.map_points) > 0:
            map_min = np.min(self.map_points[:, :3], axis=0)
            map_max = np.max(self.map_points[:, :3], axis=0)

            for _ in range(num_candidates):
                position = np.random.uniform(map_min, map_max)
                yaw = np.random.uniform(-np.pi, np.pi)

                pose = np.eye(4)
                pose[:3, 3] = position
                pose[:3, :3] = R.from_euler('z', yaw).as_matrix()

                candidates.append(pose)

        return candidates

    def save_visualization(self, filename="localization_result.png"):
        """保存可视化结果"""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到 {filename}")

    def close_visualization(self):
        """关闭可视化窗口"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


# 使用示例
def main():
    # 创建模拟地图数据
    map_points = np.random.randn(10000, 3) * 50  # 10000个随机点

    # 创建定位器并启用可视化
    locator = GlobalLocalization(visualize=True)
    locator.load_map(map_points)

    # 模拟扫描数据
    for i in range(50):
        # 创建模拟扫描（部分地图点加上噪声）
        scan_indices = np.random.choice(len(map_points), 100)
        scan_points = map_points[scan_indices] + np.random.randn(100, 3) * 0.1

        # 模拟运动
        motion = np.eye(4)
        motion[:3, 3] = np.array([0.5, 0.2, 0])  # 平移
        if i > 0:
            locator.current_pose = motion @ locator.current_pose

        # 进行定位
        pose, confidence = locator.localize_with_scan_matching(
            scan_points,
            locator.current_pose
        )

        print(f"Step {i}: Confidence = {confidence:.3f}")

        time.sleep(0.1)  # 模拟实时处理

    # 保存结果
    locator.save_visualization()

    # 保持窗口打开
    print("按任意键关闭窗口...")
    plt.show(block=True)


if __name__ == "__main__":
    main()