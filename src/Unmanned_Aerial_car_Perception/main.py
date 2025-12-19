import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 随机种子固定，保证结果可复现
np.random.seed(42)
random.seed(42)


class AutonomousVehiclePerception:
    def __init__(self):
        # 传感器参数初始化
        self.camera_resolution = (640, 480)  # 摄像头分辨率
        self.lidar_fov = 180  # LiDAR水平视场角（度）
        self.lidar_points_num = 1000  # 单次扫描点云数量
        self.lidar_max_range = 50  # LiDAR最大探测距离（米）

        # 感知结果存储
        self.lane_lines = []  # 检测到的车道线
        self.obstacles = []  # 检测到的障碍物（位置+尺寸）
        self.perception_fusion_result = {}  # 融合结果

    def simulate_camera_data(self):
        """模拟摄像头图像数据（生成带车道线的道路图像）"""
        # 创建黑色背景
        img = np.zeros((self.camera_resolution[1], self.camera_resolution[0], 3), dtype=np.uint8)

        # 绘制道路和车道线
        road_color = (50, 50, 50)
        lane_color = (255, 255, 0)
        cv2.rectangle(img, (0, self.camera_resolution[1] // 2),
                      (self.camera_resolution[0], self.camera_resolution[1]),
                      road_color, -1)

        # 绘制两条车道线（带轻微弯曲）
        y = np.linspace(self.camera_resolution[1] // 2, self.camera_resolution[1] - 10, 100)
        x1 = 0.0001 * (y - self.camera_resolution[1] // 2) ** 2 + self.camera_resolution[0] // 3
        x2 = -0.0001 * (y - self.camera_resolution[1] // 2) ** 2 + 2 * self.camera_resolution[0] // 3

        for i in range(len(y) - 1):
            cv2.line(img, (int(x1[i]), int(y[i])), (int(x1[i + 1]), int(y[i + 1])), lane_color, 3)
            cv2.line(img, (int(x2[i]), int(y[i])), (int(x2[i + 1]), int(y[i + 1])), lane_color, 3)

        # 添加随机噪声
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        return img

    def simulate_lidar_data(self):
        """模拟LiDAR点云数据（包含障碍物点云）"""
        # 生成基础点云（道路平面）
        angles = np.linspace(-self.lidar_fov / 2, self.lidar_fov / 2, self.lidar_points_num) * np.pi / 180
        ranges = np.random.uniform(1, self.lidar_max_range, self.lidar_points_num)

        # 转换为笛卡尔坐标（x: 前向, y: 横向, z: 高度）
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        z = np.zeros_like(x)  # 道路平面z=0

        # 添加障碍物（随机生成1-3个障碍物）
        obstacle_num = random.randint(1, 3)
        for _ in range(obstacle_num):
            # 障碍物中心位置（前向10-40米，横向-5到5米）
            obs_x = random.uniform(10, 40)
            obs_y = random.uniform(-5, 5)
            # 障碍物点云范围（2x2米）
            obs_angles = np.linspace(-np.pi / 4, np.pi / 4, 50)
            obs_ranges = np.random.uniform(0.5, 1.5, 50)
            obs_x_points = obs_x + obs_ranges * np.cos(obs_angles)
            obs_y_points = obs_y + obs_ranges * np.sin(obs_angles)
            obs_z_points = np.random.uniform(0, 1.5, 50)  # 障碍物高度0-1.5米

            # 合并障碍物点云
            x = np.concatenate([x, obs_x_points])
            y = np.concatenate([y, obs_y_points])
            z = np.concatenate([z, obs_z_points])

        return np.vstack([x, y, z]).T

    def detect_lane_lines(self, img):
        """基于摄像头图像检测车道线（霍夫变换）"""
        # 预处理：灰度化→高斯模糊→边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 感兴趣区域（ROI）：只保留道路部分
        mask = np.zeros_like(edges)
        roi_vertices = np.array([[(0, self.camera_resolution[1]),
                                  (self.camera_resolution[0] // 3, self.camera_resolution[1] // 2),
                                  (2 * self.camera_resolution[0] // 3, self.camera_resolution[1] // 2),
                                  (self.camera_resolution[0], self.camera_resolution[1])]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # 霍夫直线检测
        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=20,
                                minLineLength=30, maxLineGap=200)

        # 筛选并存储车道线
        self.lane_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                self.lane_lines.append(((x1, y1), (x2, y2)))

        return self.lane_lines

    def detect_obstacles(self, lidar_points):
        """基于LiDAR点云检测障碍物（聚类算法）"""
        # 过滤地面点（z<0.1视为地面）
        non_ground_points = lidar_points[lidar_points[:, 2] > 0.1]
        self.obstacles = []

        if len(non_ground_points) == 0:
            return self.obstacles

        # 简单聚类（距离阈值法）
        clusters = []
        while len(non_ground_points) > 0:
            # 取第一个点作为聚类中心
            cluster_center = non_ground_points[0]
            cluster = [cluster_center]
            # 计算所有点到中心的距离
            distances = np.linalg.norm(non_ground_points - cluster_center, axis=1)
            # 距离<2米的点归为同一聚类
            cluster_points = non_ground_points[distances < 2]
            clusters.append(cluster_points)
            # 移除已聚类的点
            non_ground_points = non_ground_points[distances >= 2]

        # 计算每个障碍物的位置和尺寸
        for cluster in clusters:
            if len(cluster) < 5:  # 过滤噪声点
                continue
            # 障碍物中心
            center_x = np.mean(cluster[:, 0])
            center_y = np.mean(cluster[:, 1])
            # 障碍物尺寸
            size_x = np.max(cluster[:, 0]) - np.min(cluster[:, 0])
            size_y = np.max(cluster[:, 1]) - np.min(cluster[:, 1])
            size_z = np.max(cluster[:, 2]) - np.min(cluster[:, 2])
            # 障碍物距离
            distance = np.sqrt(center_x ** 2 + center_y ** 2)

            self.obstacles.append({
                "center": (center_x, center_y),
                "size": (size_x, size_y, size_z),
                "distance": distance,
                "points": cluster
            })

        return self.obstacles

    def fuse_perception_data(self):
        """融合视觉和LiDAR感知数据"""
        # 融合逻辑：车道线位置 + 障碍物位置/距离 + 障碍物与车道线的相对位置
        self.perception_fusion_result = {
            "lane_lines": self.lane_lines,
            "obstacles": self.obstacles,
            "obstacle_in_lane": [],
            "lane_width": self.camera_resolution[0] // 3  # 估算车道宽度
        }

        # 判断障碍物是否在车道内
        for obs in self.obstacles:
            obs_y = obs["center"][1]
            # 车道横向范围（简化版）
            lane_left = -self.perception_fusion_result["lane_width"] / 2
            lane_right = self.perception_fusion_result["lane_width"] / 2
            if lane_left <= obs_y <= lane_right:
                self.perception_fusion_result["obstacle_in_lane"].append(obs)

        return self.perception_fusion_result

    def visualize_results(self):
        """可视化感知结果"""
        # 1. 可视化摄像头图像和车道线
        img = self.simulate_camera_data()
        for line in self.lane_lines:
            (x1, y1), (x2, y2) = line
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Lane Detection Result", img)

        # 2. 可视化LiDAR点云和障碍物
        lidar_points = self.simulate_lidar_data()
        fig = plt.figure(figsize=(12, 5))

        # 2.1 2D俯视图
        ax1 = fig.add_subplot(121)
        ax1.scatter(lidar_points[:, 0], lidar_points[:, 1], s=1, c='gray', label='LiDAR Points')
        for obs in self.obstacles:
            ax1.scatter(obs["points"][:, 0], obs["points"][:, 1], s=2, c='red',
                        label='Obstacle' if 'Obstacle' not in ax1.get_legend_handles_labels()[1] else "")
        ax1.set_xlabel("X (m) - Forward")
        ax1.set_ylabel("Y (m) - Lateral")
        ax1.set_title("LiDAR Top View")
        ax1.legend()
        ax1.grid(True)

        # 2.2 3D点云图
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2], s=1, c='gray')
        for obs in self.obstacles:
            ax2.scatter(obs["points"][:, 0], obs["points"][:, 1], obs["points"][:, 2], s=2, c='red')
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_zlabel("Z (m)")
        ax2.set_title("3D LiDAR Point Cloud")

        # 3. 打印融合结果
        print("\n=== Perception Fusion Result ===")
        print(f"Detected lane lines: {len(self.lane_lines)}")
        print(f"Detected obstacles: {len(self.obstacles)}")
        print(f"Obstacles in lane: {len(self.perception_fusion_result['obstacle_in_lane'])}")
        for i, obs in enumerate(self.perception_fusion_result['obstacle_in_lane']):
            print(f"  Obstacle {i + 1}: Distance = {obs['distance']:.2f}m, Size = {obs['size']}m")

        plt.tight_layout()
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_perception(self):
        """运行完整的感知流程"""
        print("=== Starting Autonomous Vehicle Perception ===")
        # 1. 读取传感器数据
        camera_img = self.simulate_camera_data()
        lidar_points = self.simulate_lidar_data()

        # 2. 感知检测
        self.detect_lane_lines(camera_img)
        self.detect_obstacles(lidar_points)

        # 3. 数据融合
        self.fuse_perception_data()

        # 4. 结果可视化
        self.visualize_results()


# 运行感知系统
if __name__ == "__main__":
    perception_system = AutonomousVehiclePerception()
    perception_system.run_perception()