# global_localization.py
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import socket
import json
import struct
import threading
from queue import Queue
import pandas as pd
import os
import sys

# 尝试导入open3d，如果失败则提供替代方案
try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
    print("open3d导入成功")
except ImportError as e:
    OPEN3D_AVAILABLE = False
    print(f"警告：无法导入open3d，某些功能将受限: {e}")
    print("请安装open3d: pip install open3d")

# 尝试导入ROS相关模块
try:
    import rosbag
    from sensor_msgs import point_cloud2
    import rospy

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("警告：ROS模块不可用，ROS bag功能将受限")

# 尝试导入CARLA模块
try:
    import carla

    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


class SimulatorInterface:
    """模拟器数据接口基类"""

    def __init__(self, simulator_type='custom'):
        self.simulator_type = simulator_type
        self.running = False
        self.data_queue = Queue(maxsize=100)

    def connect(self):
        """连接到模拟器"""
        raise NotImplementedError

    def disconnect(self):
        """断开连接"""
        self.running = False

    def get_scan_data(self):
        """获取扫描数据"""
        raise NotImplementedError

    def get_pose_data(self):
        """获取位姿数据（真值）"""
        raise NotImplementedError


class ROSBagInterface(SimulatorInterface):
    """ROS bag文件接口"""

    def __init__(self, bag_file, pointcloud_topic='/lidar/points', odom_topic='/odom'):
        super().__init__('rosbag')
        self.bag_file = bag_file
        self.pointcloud_topic = pointcloud_topic
        self.odom_topic = odom_topic
        self.bag = None

    def connect(self):
        """打开bag文件"""
        if not ROS_AVAILABLE:
            print("ROS模块不可用，无法使用ROS bag功能")
            return False

        try:
            self.bag = rosbag.Bag(self.bag_file, 'r')
            self.running = True
            print(f"成功打开ROS bag文件: {self.bag_file}")
            return True
        except Exception as e:
            print(f"打开bag文件失败: {e}")
            return False

    def get_scan_data(self):
        """从bag文件读取点云数据"""
        if not self.bag:
            return None

        for topic, msg, t in self.bag.read_messages(topics=[self.pointcloud_topic]):
            if topic == self.pointcloud_topic:
                # 转换点云消息为numpy数组
                points = []
                for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                    points.append([p[0], p[1], p[2]])
                if points:
                    return np.array(points)
        return None

    def get_pose_data(self):
        """从bag文件读取位姿数据"""
        if not self.bag:
            return None

        for topic, msg, t in self.bag.read_messages(topics=[self.odom_topic]):
            if topic == self.odom_topic:
                pose = msg.pose.pose
                # 构建4x4变换矩阵
                transform = np.eye(4)
                transform[0, 3] = pose.position.x
                transform[1, 3] = pose.position.y
                transform[2, 3] = pose.position.z

                # 设置旋转
                q = pose.orientation
                rotation = R.from_quat([q.x, q.y, q.z, q.w])
                transform[:3, :3] = rotation.as_matrix()
                return transform
        return None


class CARLAInterface(SimulatorInterface):
    """CARLA模拟器接口"""

    def __init__(self, host='localhost', port=2000):
        super().__init__('carla')
        self.host = host
        self.port = port
        self.client = None
        self.world = None

    def connect(self):
        """连接到CARLA服务器"""
        if not CARLA_AVAILABLE:
            print("CARLA模块不可用，请先安装: pip install carla")
            return False

        try:
            import carla
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.running = True
            print(f"成功连接到CARLA服务器 {self.host}:{self.port}")
            return True
        except ImportError:
            print("请先安装CARLA Python API: pip install carla")
            return False
        except Exception as e:
            print(f"连接CARLA服务器失败: {e}")
            return False

    def get_lidar_data(self, actor_id=None):
        """获取LiDAR数据"""
        if not self.world:
            return None

        import carla
        # 查找LiDAR传感器
        if actor_id:
            actor = self.world.get_actor(actor_id)
            if actor and 'lidar' in actor.type_id.lower():
                lidar_data = []
                # 这里需要根据CARLA API获取点云数据
                # 实际实现会更复杂，需要处理回调
                return np.array(lidar_data)
        return None


class SocketInterface(SimulatorInterface):
    """Socket数据接口（通用）"""

    def __init__(self, host='localhost', port=9999):
        super().__init__('socket')
        self.host = host
        self.port = port
        self.socket = None
        self.conn = None
        self.receiver_thread = None

    def connect(self):
        """连接到数据服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.conn = self.socket
            self.running = True

            # 启动接收线程
            self.receiver_thread = threading.Thread(target=self._receive_data)
            self.receiver_thread.daemon = True
            self.receiver_thread.start()

            print(f"成功连接到服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接服务器失败: {e}")
            return False

    def _receive_data(self):
        """接收数据线程"""
        while self.running:
            try:
                # 接收数据头（包含数据长度）
                header = self.conn.recv(4)
                if len(header) < 4:
                    continue

                data_len = struct.unpack('I', header)[0]

                # 接收实际数据
                data = b''
                while len(data) < data_len:
                    chunk = self.conn.recv(min(4096, data_len - len(data)))
                    if not chunk:
                        break
                    data += chunk

                if len(data) == data_len:
                    # 解析JSON数据
                    try:
                        message = json.loads(data.decode('utf-8'))
                        if 'type' in message and message['type'] == 'pointcloud':
                            points = np.array(message['points']).reshape(-1, 3)
                            self.data_queue.put(points)
                        elif 'type' in message and message['type'] == 'pose':
                            pose = np.array(message['pose']).reshape(4, 4)
                            self.data_queue.put(('pose', pose))
                    except:
                        pass

            except Exception as e:
                if self.running:
                    print(f"接收数据错误: {e}")
                break

    def get_scan_data(self):
        """获取扫描数据"""
        try:
            # 发送请求
            request = {'type': 'get_pointcloud'}
            data = json.dumps(request).encode('utf-8')
            header = struct.pack('I', len(data))
            self.conn.sendall(header + data)

            # 等待数据
            timeout = 2.0
            start_time = time.time()
            while time.time() - start_time < timeout:
                if not self.data_queue.empty():
                    data = self.data_queue.get()
                    if isinstance(data, np.ndarray):
                        return data
                time.sleep(0.01)
            return None
        except Exception as e:
            print(f"获取扫描数据失败: {e}")
            return None


class KITTIDataInterface(SimulatorInterface):
    """KITTI数据集接口"""

    def __init__(self, data_path, sequence='00'):
        super().__init__('kitti')
        self.data_path = data_path
        self.sequence = sequence
        self.current_frame = 0
        self.total_frames = 0
        self.poses = []

    def connect(self):
        """加载KITTI数据"""
        try:
            # 加载点云文件列表
            velodyne_path = os.path.join(self.data_path, 'sequences', self.sequence, 'velodyne')
            if not os.path.exists(velodyne_path):
                print(f"路径不存在: {velodyne_path}")
                return False

            self.pointcloud_files = sorted([f for f in os.listdir(velodyne_path) if f.endswith('.bin')])
            self.total_frames = len(self.pointcloud_files)

            # 加载位姿真值
            pose_file = os.path.join(self.data_path, 'poses', f'{self.sequence}.txt')
            if os.path.exists(pose_file):
                poses_data = np.loadtxt(pose_file).reshape(-1, 3, 4)
                for pose in poses_data:
                    transform = np.eye(4)
                    transform[:3, :] = pose
                    self.poses.append(transform)

            self.running = True
            print(f"加载KITTI序列 {self.sequence}: {self.total_frames} 帧")
            return True
        except Exception as e:
            print(f"加载KITTI数据失败: {e}")
            return False

    def get_scan_data(self, frame_idx=None):
        """获取指定帧的点云数据"""
        if frame_idx is None:
            frame_idx = self.current_frame
            self.current_frame += 1

        if frame_idx >= self.total_frames:
            return None

        try:
            file_path = os.path.join(self.data_path, 'sequences', self.sequence,
                                     'velodyne', self.pointcloud_files[frame_idx])

            # 读取bin文件
            scan = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            # 只取xyz，忽略反射强度
            points = scan[:, :3]
            return points
        except Exception as e:
            print(f"读取点云帧 {frame_idx} 失败: {e}")
            return None

    def get_pose_data(self, frame_idx=None):
        """获取位姿真值"""
        if frame_idx is None:
            frame_idx = self.current_frame - 1

        if frame_idx < len(self.poses):
            return self.poses[frame_idx]
        return None


class GlobalLocalization:
    """全局定位模块"""

    def __init__(self, map_data=None, visualize=True, simulator=None):
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

        # 数据接口
        self.simulator = simulator

        # 可视化相关
        self.visualize = visualize
        self.fig = None
        self.ax = None
        self.history_poses = []
        self.history_confidences = []
        self.scan_history = []
        self.ground_truth_poses = []

        # 颜色设置
        self.colors = {
            'map': (0.3, 0.3, 0.3, 0.3),  # 灰色半透明
            'current_scan': (1.0, 0.0, 0.0, 0.5),  # 红色
            'matched_scan': (0.0, 1.0, 0.0, 0.5),  # 绿色
            'trajectory': (0.0, 0.0, 1.0, 1.0),  # 蓝色
            'candidates': (1.0, 1.0, 0.0, 0.3),  # 黄色
            'ground_truth': (0.5, 0.0, 0.5, 1.0),  # 紫色
        }

    def set_simulator(self, simulator):
        """设置数据接口"""
        self.simulator = simulator

    def load_map_from_file(self, filepath, file_type='auto'):
        """从文件加载地图"""
        points = None

        if file_type == 'auto':
            # 自动检测文件类型
            if filepath.endswith('.pcd') or filepath.endswith('.ply'):
                file_type = 'pcd'
            elif filepath.endswith('.bin'):
                file_type = 'kitti_bin'
            elif filepath.endswith('.csv'):
                file_type = 'csv'
            elif filepath.endswith('.npy'):
                file_type = 'npy'
            elif filepath.endswith('.txt'):
                file_type = 'txt'

        try:
            if (file_type == 'pcd' or file_type == 'ply') and OPEN3D_AVAILABLE:
                # 使用Open3D读取
                pcd = o3d.io.read_point_cloud(filepath)
                points = np.asarray(pcd.points)
                print(f"使用open3d读取 {file_type} 文件: {filepath}")

            elif file_type == 'kitti_bin':
                # KITTI格式的bin文件
                points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)[:, :3]
                print(f"读取KITTI bin文件: {filepath}")

            elif file_type == 'csv':
                # CSV文件
                df = pd.read_csv(filepath)
                # 尝试不同的列名
                possible_columns = [['x', 'y', 'z'], ['X', 'Y', 'Z'],
                                    ['position_x', 'position_y', 'position_z']]

                for cols in possible_columns:
                    if all(col in df.columns for col in cols):
                        points = df[cols].values
                        break

                if points is None:
                    # 如果前3列是xyz
                    points = df.iloc[:, :3].values

                print(f"读取CSV文件: {filepath}")

            elif file_type == 'npy':
                # numpy文件
                points = np.load(filepath)
                print(f"读取numpy文件: {filepath}")

            elif file_type == 'txt':
                # 文本文件
                points = np.loadtxt(filepath)
                print(f"读取文本文件: {filepath}")

            else:
                print(f"不支持的文件类型: {file_type}")
                return False

            if points is not None:
                self.load_map(points)
                print(f"从 {filepath} 加载了 {len(points)} 个点")
                return True
            else:
                print(f"无法从 {filepath} 读取点云数据")
                return False

        except Exception as e:
            print(f"加载地图文件失败: {e}")
            import traceback
            traceback.print_exc()

        return False

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
                                matched_points=None, candidates=None,
                                ground_truth=None):
        """可视化定位结果"""
        if not self.visualize:
            return

        # 变换当前扫描点
        transformed_scan = self._transform_points(scan_points, pose)

        # 更新历史记录
        self.history_poses.append(pose)
        self.history_confidences.append(confidence)
        self.scan_history.append(transformed_scan)
        if ground_truth is not None:
            self.ground_truth_poses.append(ground_truth)

        # 限制历史记录长度
        max_history = 100
        if len(self.history_poses) > max_history:
            self.history_poses = self.history_poses[-max_history:]
            self.history_confidences = self.history_confidences[-max_history:]
            self.scan_history = self.scan_history[-max_history:]
            if self.ground_truth_poses:
                self.ground_truth_poses = self.ground_truth_poses[-max_history:]

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
                          c=self.colors['trajectory'], linewidth=2, label='Estimated')

            # 真值轨迹
            if self.ground_truth_poses and len(self.ground_truth_poses) > 1:
                gt_trajectory = np.array([p[:3, 3] for p in self.ground_truth_poses])
                self.ax1.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2],
                              c=self.colors['ground_truth'], linewidth=2, linestyle='--',
                              label='Ground Truth')

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
                          c=self.colors['trajectory'], linewidth=2, label='Estimated')

            # 真值轨迹
            if self.ground_truth_poses and len(self.ground_truth_poses) > 1:
                gt_trajectory = np.array([p[:3, 3] for p in self.ground_truth_poses])
                self.ax2.plot(gt_trajectory[:, 0], gt_trajectory[:, 1],
                              c=self.colors['ground_truth'], linewidth=2, linestyle='--',
                              label='Ground Truth')

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
        if self.ground_truth_poses and len(self.ground_truth_poses) > 1:
            # 计算估计位姿与真值的误差
            errors = []
            for i in range(min(len(self.history_poses), len(self.ground_truth_poses))):
                if i < len(self.ground_truth_poses):
                    # 位置误差
                    pos_error = np.linalg.norm(
                        self.history_poses[i][:3, 3] - self.ground_truth_poses[i][:3, 3]
                    )

                    # 角度误差
                    R_est = self.history_poses[i][:3, :3]
                    R_gt = self.ground_truth_poses[i][:3, :3]
                    angle_error = np.arccos((np.trace(R_est.T @ R_gt) - 1) / 2)

                    errors.append(pos_error)

            if errors:
                steps = range(len(errors))
                self.ax5.plot(steps, errors, 'r-', label='Position Error (m)')
                self.ax5.axhline(y=np.mean(errors), color='r', linestyle='--',
                                 alpha=0.5, label=f'Avg: {np.mean(errors):.3f}m')

                self.ax5.set_xlabel('Step')
                self.ax5.set_ylabel('Error (m)')
                self.ax5.set_title('Localization Error vs Ground Truth')
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

        # 获取真值（如果有）
        ground_truth = None
        if self.simulator:
            ground_truth = self.simulator.get_pose_data()

        # 可视化
        if self.visualize and visualize:
            self._visualize_localization(
                scan_points=scan_points,
                pose=refined_pose,
                confidence=confidence,
                matched_points=matched_points,
                ground_truth=ground_truth
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

    def run_with_simulator(self, max_frames=1000, start_frame=0):
        """使用模拟器数据运行定位"""
        if not self.simulator or not self.simulator.running:
            print("模拟器未连接或未运行")
            return

        frame_count = 0

        for i in range(start_frame, start_frame + max_frames):
            # 从模拟器获取数据
            scan_points = self.simulator.get_scan_data()

            if scan_points is None:
                print("无法获取扫描数据")
                break

            # 检查点云质量
            if len(scan_points) < 100:
                print(f"帧 {i}: 点云点数过少 ({len(scan_points)})")
                continue

            print(f"\n帧 {i}: 处理 {len(scan_points)} 个点")

            # 进行定位
            start_time = time.time()

            if frame_count == 0 or self.confidence < 0.3:
                # 低置信度时尝试全局重定位
                pose, confidence = self.global_relocalization(scan_points)
                if confidence > 0.3:
                    self.localization_mode = "local"
                else:
                    self.localization_mode = "global"
            else:
                # 正常局部定位
                pose, confidence = self.localize_with_scan_matching(
                    scan_points,
                    self.current_pose
                )

            processing_time = time.time() - start_time

            # 更新状态
            self.current_pose = pose
            self.confidence = confidence

            print(f"  置信度: {confidence:.3f}")
            print(f"  位姿: [{pose[0, 3]:.2f}, {pose[1, 3]:.2f}, {pose[2, 3]:.2f}]")
            print(f"  处理时间: {processing_time:.3f}秒")
            print(f"  模式: {self.localization_mode}")

            frame_count += 1

            # 控制帧率
            time.sleep(0.05)  # 20Hz

            # 检查是否应该停止
            if not self.simulator.running:
                break

        print(f"\n处理完成，共处理 {frame_count} 帧")

        # 保存结果
        if self.visualize:
            self.save_visualization()
            plt.show(block=True)

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


def main_demo():
    """演示模式：使用合成数据测试"""
    print("=" * 60)
    print("全局定位系统 - 演示模式")
    print("=" * 60)

    # 创建合成地图数据
    map_points = []

    # 创建一些建筑物
    print("生成合成地图...")
    for x in np.arange(-50, 51, 10):
        for y in np.arange(-50, 51, 10):
            # 建筑物点云
            building_points = np.random.randn(100, 3) * 2
            building_points[:, 0] += x
            building_points[:, 1] += y
            building_points[:, 2] = building_points[:, 2] * 5 + 10  # 高度
            map_points.append(building_points)

    # 创建道路
    road_points = []
    for x in np.arange(-50, 51, 0.5):
        for y in np.arange(-5, 6, 0.5):
            road_points.append([x, y, 0])
    map_points.append(np.array(road_points))

    map_points = np.vstack(map_points)

    # 创建定位器
    print("初始化定位器...")
    locator = GlobalLocalization(visualize=True)
    locator.load_map(map_points)

    # 模拟轨迹
    print("开始模拟定位...")
    trajectory = []
    radius = 30.0
    total_frames = 50

    for i in range(total_frames):
        angle = i * 0.1
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        # 模拟当前扫描（地图的一部分加上噪声）
        # 找到附近的点作为扫描
        if i == 0:
            current_pose = np.eye(4)
        else:
            # 模拟运动
            current_pose = np.eye(4)
            current_pose[0, 3] = x
            current_pose[1, 3] = y

            # 朝向切线方向
            yaw = angle + np.pi / 2
            current_pose[:3, :3] = R.from_euler('z', yaw).as_matrix()

        # 获取附近的点作为扫描
        transformed_map = locator._transform_points(map_points, np.linalg.inv(current_pose))

        # 选择前方的点
        mask = (transformed_map[:, 0] > 0) & (transformed_map[:, 0] < 30) & \
               (np.abs(transformed_map[:, 1]) < 10) & \
               (np.random.rand(len(transformed_map)) < 0.1)  # 采样

        scan_points = map_points[mask]

        # 添加噪声
        if len(scan_points) > 0:
            scan_points = scan_points + np.random.randn(*scan_points.shape) * 0.05

            # 进行定位
            if i == 0:
                initial_guess = np.eye(4)
            else:
                # 使用上一帧的位姿作为初始猜测
                initial_guess = locator.current_pose

            pose, confidence = locator.localize_with_scan_matching(scan_points, initial_guess)

            # 更新当前位姿
            locator.current_pose = pose

            print(f"帧 {i + 1}/{total_frames}: 置信度 {confidence:.3f}, 位姿 [{pose[0, 3]:.1f}, {pose[1, 3]:.1f}]")

            time.sleep(0.1)
        else:
            print(f"帧 {i + 1}/{total_frames}: 无有效点云数据")

    # 保存结果
    print("\n保存结果...")
    locator.save_visualization("demo_result.png")
    print("演示完成！按任意键关闭窗口...")
    plt.show(block=True)


def main_from_file():
    """从文件加载地图并测试"""
    print("=" * 60)
    print("全局定位系统 - 文件模式")
    print("=" * 60)

    # 创建定位器
    locator = GlobalLocalization(visualize=True)

    # 从文件加载地图
    map_file = input("请输入地图文件路径（支持.pcd, .ply, .bin, .csv, .npy, .txt）: ").strip()

    if not map_file:
        map_file = "map.pcd"  # 默认文件

    if not os.path.exists(map_file):
        print(f"文件不存在: {map_file}")
        print("使用演示模式生成合成地图...")
        main_demo()
        return

    if locator.load_map_from_file(map_file):
        print("地图加载成功，开始模拟定位...")

        # 在地图内随机生成测试轨迹
        map_min = np.min(locator.map_points[:, :3], axis=0)
        map_max = np.max(locator.map_points[:, :3], axis=0)

        test_frames = 30
        successful_localizations = 0

        for i in range(test_frames):
            # 生成随机位姿
            random_pose = np.eye(4)
            random_pose[:3, 3] = np.random.uniform(map_min, map_max)
            random_pose[:3, :3] = R.from_euler('z', np.random.uniform(-np.pi, np.pi)).as_matrix()

            # 从该位姿生成模拟扫描
            # 获取附近的点
            transformed_map = locator._transform_points(locator.map_points, np.linalg.inv(random_pose))

            # 选择前方的点
            mask = (transformed_map[:, 0] > 0) & (transformed_map[:, 0] < 20) & \
                   (np.abs(transformed_map[:, 1]) < 5) & \
                   (np.random.rand(len(transformed_map)) < 0.05)

            scan_points = locator.map_points[mask]

            if len(scan_points) > 100:
                # 添加噪声
                scan_points = scan_points + np.random.randn(*scan_points.shape) * 0.1

                # 进行定位（使用一个初始猜测）
                initial_guess = np.eye(4)
                initial_guess[:3, 3] = random_pose[:3, 3] + np.random.randn(3) * 5  # 初始猜测有误差

                pose, confidence = locator.localize_with_scan_matching(scan_points, initial_guess)

                # 计算误差
                pos_error = np.linalg.norm(pose[:3, 3] - random_pose[:3, 3])

                if confidence > 0.5 and pos_error < 1.0:
                    successful_localizations += 1

                print(f"测试 {i + 1}/{test_frames}: 置信度 {confidence:.3f}, 位置误差 {pos_error:.2f}m")

                time.sleep(0.2)
            else:
                print(f"测试 {i + 1}/{test_frames}: 点云点数不足 ({len(scan_points)})")

        success_rate = successful_localizations / test_frames * 100
        print(f"\n测试完成！成功率: {success_rate:.1f}%")

        locator.save_visualization("file_test_result.png")
        print("按任意键关闭窗口...")
        plt.show(block=True)
    else:
        print("地图加载失败")


def main_simple_test():
    """简单测试模式"""
    print("=" * 60)
    print("全局定位系统 - 简单测试模式")
    print("=" * 60)

    # 创建简单的测试地图（一个环形）
    theta = np.linspace(0, 2 * np.pi, 1000)
    radius = 20.0
    height = 5.0

    # 环形点云
    ring_points = []
    for t in theta:
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        for h in np.linspace(0, height, 10):
            ring_points.append([x, y, h])

    # 添加一些随机的内部点
    for _ in range(1000):
        r = np.random.uniform(0, radius * 0.8)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = np.random.uniform(0, height)
        ring_points.append([x, y, z])

    map_points = np.array(ring_points)

    # 创建定位器
    locator = GlobalLocalization(visualize=True)
    locator.load_map(map_points)

    # 沿着环形轨迹测试
    test_angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)

    for i, angle in enumerate(test_angles):
        # 真实位姿
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 2.5  # 中间高度

        true_pose = np.eye(4)
        true_pose[0, 3] = x
        true_pose[1, 3] = y
        true_pose[2, 3] = z
        true_pose[:3, :3] = R.from_euler('z', angle + np.pi / 2).as_matrix()

        # 生成扫描数据
        # 选取前面的点
        transformed_map = locator._transform_points(map_points, np.linalg.inv(true_pose))
        mask = (transformed_map[:, 0] > 0) & (transformed_map[:, 0] < 15) & \
               (np.abs(transformed_map[:, 1]) < 8) & \
               (transformed_map[:, 2] > 0) & (transformed_map[:, 2] < height)

        scan_points = map_points[mask]

        if len(scan_points) > 50:
            # 添加噪声
            scan_points = scan_points + np.random.randn(*scan_points.shape) * 0.05

            # 添加初始猜测误差
            initial_guess = true_pose.copy()
            initial_guess[:3, 3] += np.random.randn(3) * 2  # 位置误差
            yaw_error = np.random.uniform(-0.3, 0.3)
            initial_guess[:3, :3] = initial_guess[:3, :3] @ R.from_euler('z', yaw_error).as_matrix()

            # 进行定位
            pose, confidence = locator.localize_with_scan_matching(scan_points, initial_guess)

            # 计算误差
            pos_error = np.linalg.norm(pose[:3, 3] - true_pose[:3, 3])

            print(f"测试 {i + 1}/{len(test_angles)}: "
                  f"置信度 {confidence:.3f}, "
                  f"位置误差 {pos_error:.2f}m")

            time.sleep(0.3)

    locator.save_visualization("simple_test_result.png")
    print("\n测试完成！按任意键关闭窗口...")
    plt.show(block=True)


if __name__ == "__main__":
    print("全局定位系统 v1.0")
    print("=" * 60)

    # 检查依赖
    print("检查依赖...")
    print(f"open3d: {'可用' if OPEN3D_AVAILABLE else '不可用'}")
    print(f"ROS: {'可用' if ROS_AVAILABLE else '不可用'}")
    print(f"CARLA: {'可用' if CARLA_AVAILABLE else '不可用'}")
    print("=" * 60)

    # 简单菜单
    print("\n请选择运行模式:")
    print("1. 演示模式 (使用合成数据)")
    print("2. 文件模式 (从文件加载地图)")
    print("3. 简单测试模式 (环形轨迹)")
    print("4. 退出")

    choice = input("\n请输入选择 (1-4): ").strip()

    if choice == "1":
        main_demo()
    elif choice == "2":
        main_from_file()
    elif choice == "3":
        main_simple_test()
    elif choice == "4":
        print("退出程序")
    else:
        print("无效选择，使用演示模式")
        main_demo()