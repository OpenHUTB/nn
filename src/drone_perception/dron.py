"""
drone_perception.py
无人机感知模块 - 用于目标检测、深度估计和障碍物感知
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import time


class DronePerception:
    """
    无人机感知模块主类
    提供目标检测、深度估计和障碍物感知功能
    """

    def __init__(self, camera_params: Optional[dict] = None):
        """
        初始化感知模块

        Args:
            camera_params: 相机内参矩阵和畸变系数
        """
        self.camera_params = camera_params or {
            'fx': 920.0, 'fy': 920.0,  # 焦距
            'cx': 640.0, 'cy': 360.0,  # 主点坐标
            'width': 1280, 'height': 720
        }

        # 初始化目标检测器
        self.target_detector = TargetDetector()

        # 初始化深度估计器
        self.depth_estimator = DepthEstimator()

        # 状态变量
        self.obstacles = []
        self.targets = []
        self.last_update_time = time.time()

        print("DronePerception模块初始化完成")

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        处理单帧图像

        Args:
            frame: 输入图像帧 (BGR格式)

        Returns:
            包含感知结果的字典
        """
        if frame is None:
            return {"error": "空帧输入"}

        results = {
            'timestamp': time.time(),
            'frame_shape': frame.shape,
            'targets': [],
            'obstacles': [],
            'depth_map': None
        }

        try:
            # 1. 目标检测
            self.targets = self.target_detector.detect(frame)
            results['targets'] = self.targets

            # 2. 深度估计
            depth_map = self.depth_estimator.estimate(frame)
            results['depth_map'] = depth_map

            # 3. 障碍物检测
            self.obstacles = self.detect_obstacles(frame, depth_map)
            results['obstacles'] = self.obstacles

            # 4. 安全区域分析
            results['safety_score'] = self.analyze_safety(self.obstacles)

            self.last_update_time = time.time()

        except Exception as e:
            results['error'] = str(e)
            print(f"处理帧时出错: {e}")

        return results

    def detect_obstacles(self, frame: np.ndarray, depth_map: Optional[np.ndarray]) -> List[dict]:
        """
        检测障碍物

        Args:
            frame: 输入图像
            depth_map: 深度图

        Returns:
            障碍物列表
        """
        obstacles = []

        # 简单边缘检测作为障碍物提示
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:5]:  # 只处理前5个最大轮廓
            if cv2.contourArea(contour) > 500:  # 面积阈值
                x, y, w, h = cv2.boundingRect(contour)

                obstacle = {
                    'bbox': [x, y, x + w, y + h],
                    'area': cv2.contourArea(contour),
                    'center': (x + w // 2, y + h // 2),
                    'type': 'unknown'
                }

                # 如果有深度信息，估算距离
                if depth_map is not None and y < depth_map.shape[0] and x < depth_map.shape[1]:
                    roi_depth = depth_map[y:y + h, x:x + w]
                    if roi_depth.size > 0:
                        obstacle['estimated_distance'] = float(np.mean(roi_depth))

                obstacles.append(obstacle)

        return obstacles

    def analyze_safety(self, obstacles: List[dict]) -> float:
        """
        分析当前帧的安全性

        Args:
            obstacles: 障碍物列表

        Returns:
            安全分数 (0-1)
        """
        if not obstacles:
            return 1.0

        # 简单的安全评分逻辑
        danger_score = 0

        for obs in obstacles:
            # 障碍物大小越大，危险分数越高
            area_factor = min(obs['area'] / 10000, 1.0)

            # 距离越近，危险分数越高
            if 'estimated_distance' in obs:
                dist_factor = max(0, 1.0 - (obs['estimated_distance'] / 20.0))
            else:
                dist_factor = 0.5

            danger_score += area_factor * dist_factor

        safety_score = max(0, 1.0 - min(danger_score, 1.0))
        return round(safety_score, 2)

    def visualize(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """
        可视化感知结果

        Args:
            frame: 原始图像帧
            results: 感知结果

        Returns:
            可视化后的图像
        """
        vis_frame = frame.copy()

        # 绘制目标边界框
        for target in results.get('targets', []):
            if 'bbox' in target:
                x1, y1, x2, y2 = map(int, target['bbox'])
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if 'class_name' in target:
                    cv2.putText(vis_frame, target['class_name'],
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

        # 绘制障碍物边界框
        for obstacle in results.get('obstacles', []):
            if 'bbox' in obstacle:
                x1, y1, x2, y2 = map(int, obstacle['bbox'])
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 显示估计距离
                if 'estimated_distance' in obstacle:
                    dist_text = f"{obstacle['estimated_distance']:.1f}m"
                    cv2.putText(vis_frame, dist_text,
                                (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)

        # 显示安全分数
        safety_score = results.get('safety_score', 1.0)
        safety_color = (0, 255, 0) if safety_score > 0.7 else (0, 255, 255) if safety_score > 0.4 else (0, 0, 255)

        cv2.putText(vis_frame, f"Safety: {safety_score}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, safety_color, 2)

        # 显示帧信息
        cv2.putText(vis_frame, f"Targets: {len(results.get('targets', []))}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Obstacles: {len(results.get('obstacles', []))}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

        return vis_frame

    def get_status(self) -> dict:
        """
        获取模块状态

        Returns:
            状态字典
        """
        return {
            'active': True,
            'last_update': self.last_update_time,
            'camera_params': self.camera_params,
            'target_count': len(self.targets),
            'obstacle_count': len(self.obstacles)
        }


class TargetDetector:
    """目标检测器"""

    def __init__(self):
        # 这里可以加载YOLO等模型
        # 简化版本使用预定义的HSV颜色范围
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255])
        }

    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        检测目标

        Args:
            frame: 输入图像

        Returns:
            检测到的目标列表
        """
        targets = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color_name, (lower, upper) in self.color_ranges.items():
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)

            mask = cv2.inRange(hsv, lower_np, upper_np)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 300:  # 面积过滤
                    x, y, w, h = cv2.boundingRect(contour)

                    target = {
                        'bbox': [x, y, x + w, y + h],
                        'class_name': color_name,
                        'confidence': 0.8,
                        'area': cv2.contourArea(contour)
                    }
                    targets.append(target)

        return targets


class DepthEstimator:
    """深度估计器"""

    def __init__(self):
        # 这里可以加载MiDaS等深度估计模型
        pass

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        估计深度图

        Args:
            frame: 输入图像

        Returns:
            深度图 (归一化到0-1)
        """
        # 简化版本：使用图像强度作为深度代理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_map = cv2.GaussianBlur(gray, (15, 15), 0)

        # 归一化
        if depth_map.max() > depth_map.min():
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        # 模拟距离：假设图像中心区域更近
        h, w = depth_map.shape
        y_coords, x_coords = np.ogrid[:h, :w]
        center_mask = 1.0 - np.sqrt(((x_coords - w // 2) / (w // 2)) ** 2 +
                                    ((y_coords - h // 2) / (h // 2)) ** 2)
        center_mask = np.clip(center_mask, 0, 1)

        # 结合强度信息和中心权重
        depth_map = 0.7 * depth_map + 0.3 * center_mask

        return depth_map


def test_perception():
    """测试函数"""
    print("=== 测试DronePerception模块 ===")

    # 初始化感知模块
    perception = DronePerception()

    # 创建测试图像
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 添加一些彩色方块模拟目标
    cv2.rectangle(test_frame, (100, 100), (200, 200), (0, 255, 0), -1)  # 绿色
    cv2.rectangle(test_frame, (400, 150), (500, 250), (0, 0, 255), -1)  # 红色
    cv2.rectangle(test_frame, (300, 300), (400, 400), (255, 0, 0), -1)  # 蓝色

    # 处理帧
    results = perception.process_frame(test_frame)

    # 打印结果
    print(f"检测到目标数量: {len(results['targets'])}")
    print(f"检测到障碍物数量: {len(results['obstacles'])}")
    print(f"安全分数: {results['safety_score']}")

    # 可视化
    vis_frame = perception.visualize(test_frame, results)

    # 显示结果
    cv2.imshow('Perception Test', vis_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 显示模块状态
    print("\n模块状态:")
    for key, value in perception.get_status().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # 直接运行测试
    test_perception()